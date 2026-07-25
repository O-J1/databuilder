# databuilder

Config-driven, distributed image dataset builder. Downloads HF datasets,
filters, deupes, then clusters embeds with DINOv3 + usearch k-means and then prunes over represented clusters. Emits a generator and cluster-balanced manifest.

Self-contained library.

```bash
pip install -e ./databuilder            # core pipeline
pip install -e "./databuilder[embed]"   # + torch/transformers for the embed stage
pip install -e "./databuilder[viz]"    # + FastAPI/uvicorn/umap for the viewer
```

## Pipeline

| # | Stage        | Scope     | What it does |
|---|--------------|-----------|--------------|
| 1 | `download`   | rank 0    | one-worker HF/Xet snapshots + format-aware materialization |
| 2 | `headerscan` | per node  | delete: longest side < min, broken headers, aspect beyond 9:23 / 23:9, broken files |
| 3 | `fingerprint`| per node* | one decode: Laplacian filter + xxh3 + phash (12x12) + colorhash |
| 4 | `dedup`      | rank 0    | global exact + near-duplicate delete (keep highest res, then largest file) |
| 5 | `embed`      | per node* | DINOv3 embeddings -> parquet shards (fp16) per GPU |
| 6 | `cluster`    | rank 0    | usearch/sklearn k-means; flag over-represented members (never deletes) |
| 7 | `manifest`   | rank 0    | balanced manifest; parquet always, CSV refused above 1M rows |

\* With `[daft] enabled = true` and `runner = "ray"`, fingerprint and embed run
from rank 0 on a Ray cluster instead (see [Daft execution path](#daft-execution-path)).

## Single machine

```bash
databuilder run --config build.toml --dry-run   # report deletions only
databuilder run --config build.toml             # real run
databuilder stage headerscan --config build.toml --wait   # one stage
```

## Multi-node (shared filesystem)

`work_dir` and `data_dir` must be on the shared FS. World size is pinned per
work_dir; barriers are `_SUCCESS` marker files — no scheduler integration is
required, but SLURM is detected automatically (below).

Rank and world size are resolved with this precedence (first hit wins):

1. `--rank` / `--world-size` flags
2. `DATABUILDER_RANK` / `DATABUILDER_WORLD_SIZE`
3. `RANK` / `WORLD_SIZE`
4. `SLURM_PROCID` / `SLURM_NTASKS` (then `SLURM_NODEID` / `SLURM_NNODES`)

The log states which source was used. Rank 0 runs download and the global
stages (dedup, cluster, manifest); other ranks never contact Hugging Face and
wait at the download barrier before starting their sharded work.
Re-running skips completed stages.

### Manual launch (no scheduler)

```bash
# node i (i = 0..7):
databuilder run --config build.toml --rank $i --world-size 8
```

### SLURM

Under `srun`/`sbatch` no flags or env vars are needed — each task picks up its
rank from `SLURM_PROCID` and the world size from `SLURM_NTASKS`. Launch
**one task per node** (`--ntasks-per-node=1`): the embed stage uses every GPU
on the node, and databuilder logs a warning if the allocation has more than
one task per node. See
[examples/slurm_build.sbatch](examples/slurm_build.sbatch):

```bash
sbatch examples/slurm_build.sbatch build.toml          # 8 nodes (script default)
sbatch --nodes=16 examples/slurm_build.sbatch build.toml  # override node count
```

Do not pin `rank` in the TOML for SLURM runs — the auto-detected values
override the config, but leaving it unset avoids confusion. Re-submitting the
same job resumes from the last completed stage.

## Daft execution path

With `pip install "databuilder[daft]"` and `[daft] enabled = true`, the
fingerprint and embed stages run on [Daft](https://docs.daft.ai): file hashing
(xxh3), perceptual hashing (12x12 phash + colorhash), and image decoding run in
Daft's Rust kernels; HEIC/AVIF/JXL fall back to Pillow plugins automatically.
Artifacts are byte-compatible with the default path, so the two can be mixed
freely across runs (not within one work_dir stage).

- `runner = "native"` — each SLURM rank runs Daft locally on its own shard;
  topology and barriers are unchanged.
- `runner = "ray"` — rank 0 submits the whole stage to an existing Ray cluster
  (`ray_address = "auto"` or `"ray://head:10001"`); the other ranks wait at the
  stage barrier. For embed, set `embedding.concurrency` to the number of model
  replicas (one GPU each) Ray should schedule.

### End-to-end engine benchmark

The standalone benchmark compares the fallback implementation with Daft's
native runner using the pipeline settings and four sources in
`examples/smoke.toml`:

```bash
pip install -e ".[embed]"
python benchmarks/benchmark_engines.py
```

The local dataset path configured in `examples/smoke.toml` must exist, the
benchmark output directory must be writable, and the Hugging Face datasets and
DINOv3 model must be accessible (and authenticated if required) on the first
invocation. Setup is deliberately outside the timed interval: the script runs
the normal download stage against the smoke config's own work/data
directories (already-materialized data is reused, never re-downloaded),
samples exactly 125 images from each source with seed 42, copies that fixed
500-image corpus, and caches the model files. Later invocations reuse and
hash-validate the corpus; pass `--rebuild-corpus` to sample it again.

Each measurement starts a fresh process and work directory, then times all
seven stages through the normal `databuilder run` CLI. Three runs are made per
engine, with their order alternated between rounds. Process and model startup,
inference, and artifact writes are included in the wall-clock duration; source
preparation and model download are not. Timed child processes use the local
model cache in offline mode and are forced to rank 0 with a world size of 1.

Individual timings, mean/min/max values, and
`fallback mean / Daft mean` are printed. A speedup above 1 means Daft was
faster. Raw results and run artifacts are retained under
`.manual/benchmark/`, with the newest result also written to
`.manual/benchmark/latest.json`. Dedup survivor sets and manifest row counts
are checked across all six runs so diverging outputs are not presented as
comparable timings (full manifest contents are not compared: usearch k-means
is unseeded, so cluster assignments legitimately vary between runs). This
hardware-dependent benchmark is not part of the normal pytest suite and has no
pass/fail performance threshold.

## Cluster viewer

```bash
databuilder viz-prepare --config build.toml --sample 200000   # stratified per-cluster sample + UMAP/PCA
databuilder viz --config build.toml --port 8765               # binds 127.0.0.1 ONLY
```

### Accessing the viewer over SSH (secure, no ports opened)

The server binds to `127.0.0.1` and refuses non-loopback hosts. It is invisible
to the network and opens **no** ports on the cluster. From your laptop, reuse
your existing SSH access as an encrypted tunnel:

```bash
ssh -L 8765:127.0.0.1:8765 user@compute-node
# via a bastion/login node:
ssh -J user@bastion -L 8765:127.0.0.1:8765 user@compute-node
```

Then open <http://localhost:8765> locally. Traffic rides inside the SSH
session; nothing is exposed to the internet and no firewall changes are made.

## Config (TOML)

See [examples/build.example.toml](examples/build.example.toml). Key sections:

- `[runtime]` - `work_dir`, `data_dir`, `num_workers`, `world_size`/`rank`
- `[download]` - Hugging Face `snapshot_download` concurrency. It defaults to
  `max_workers = 1`, so snapshots run sequentially on rank 0 and rely on
  HF/Xet for partial-transfer resume. Xet's range and cache-size settings stay
  at Hugging Face defaults. Set `xet_high_performance = true` to let Xet
  maximize rank 0's CPU, disk, and network use. Databuilder does not implement
  per-file HF transfers or its own download resume ledger.
- `[filters]` - `min_longest_side`, `max_tall = "9:23"`, `max_wide = "23:9"`, `laplacian_min/max`
- `[dedup]` - `phash_size = 12`, `phash_max_hamming`, `colorhash_max_hamming`
- `[embedding]` - DINOv3 `model` id (any size), `batch_size`, `devices = "auto"`,
  `concurrency` (Daft runner only: model replicas, one GPU each)
- `[daft]` - `enabled`, `runner = "native"|"ray"`, `ray_address` (see below)
- `[clustering]` - `aggressiveness` (0..1; 0.5 = ~sqrt(N) clusters, 1.0 = ~4x more)
  or explicit `k`; `prune_trigger_sigma` (only clusters sized above
  mean + sigma*std are examined for pruning) and `semdedup_threshold` (inside
  those, members with cosine similarity above the threshold to an already-kept
  member are flagged as `semantic_duplicate`; unique images always survive)
- `[balance]` - `max_per_generator`, `per_generator_cluster_cap`, `max_label_ratio`
  (trim the majority label to <= minority x ratio, round-robin across generators;
  0 disables), `val_fraction`, `emit_csv`
- `[[datasets]]` - one per source. HF: `repo_id`; local: `path` (+ explicit `format`).
  `label` = `real`/`fake` (static), `folder` (infer from path segments), `auto`
  (automatch a column / folder inference), or `column:<c>`. `generator` = static
  name, `folder`, or `column:<c>`. `[datasets.columns]` maps roles explicitly
  (`image`, `label`, `generator`, `split`); automatch covers common names and
  hard-errors listing the schema when a required role cannot be matched.
  `[datasets.label_map]` maps custom folder names / column values to real/fake.
  `row_filter = { column = value }` keeps matching rows; `row_exclude =
  { column = [values] }` drops listed values before image decoding (string
  comparisons are case-insensitive).
  `images = [{ column = "image1", generator_column = "model1" }, ...]` maps
  tables with multiple image fields. Supported materializers are parquet,
  Arrow, JSONL with local image paths, zip, tar/WebDataset, split zip,
  imagefolder, and `raw`. External image URLs are never fetched individually.
  `download_only = true` with `format = "raw"` retains a snapshot but excludes
  it from headerscan and every downstream manifest stage. `source_split` picks
  the HF split to download; `assign_split = "test"` forces
  the whole dataset into one output split (forced val/test bypass balancing
  caps and cluster pruning).

The requested 40-repository corpus is ready in
[`examples/aigc-datasets.toml`](examples/aigc-datasets.toml). It pins every
revision, writes snapshots and caches only below
`/p/data1/datasets/mmlaion/aigc/data`, selects generated outputs (not source
references) from UniPic, and preserves ambiguous or URL-only repositories as
raw snapshots. The card/schema audit and real/fake decisions are recorded in
[`docs/aigc-dataset-labels.md`](docs/aigc-dataset-labels.md).

### URL-backed image datasets

The main pipeline snapshots URL metadata but deliberately does not issue one
HTTP request per image. `kafked/anycrap` and `lehduong/seaart-hq` can instead
be downloaded independently with:

```bash
python scripts/download_url_datasets.py \
  --data-dir /p/data1/datasets/mmlaion/aigc/data \
  --workers 16
```

Use `--dataset anycrap` or `--dataset seaart-hq` to select one source,
`--limit 100 --dry-run` to inspect its metadata count, and
`--skip-metadata-snapshot` to make no Hugging Face call when the pinned
metadata snapshot is already present. Downloads are streamed, size-limited,
decoded before acceptance, atomically renamed, and skipped on rerun when the
URL-derived destination already exists. Success and failure JSONL logs are
written beside the images under `<data-dir>/<dataset>/url-images/`.

This standalone download does not change `download_only = true` in the main
config. To feed the downloaded images into a later build, add their
`url-images` directory as a local `imagefolder` dataset with static
`label = "fake"` and the appropriate generator (`anycrap` or `seaart`).

### Local filesystem datasets

Local `imagefolder` datasets are scanned **in place** and manifest rows use
absolute paths. Their files are **never deleted by default**: filter and dedup
rejects are only recorded in the run artifacts. Set `allow_delete = true` on a
dataset to let the pipeline physically delete originals (run `--dry-run`
first!). With `label = "folder"`, label folders (`real`, `fake`,
`generated`, `ai`, `aigc`, or `label_map` entries) are matched up to two levels
deep, and the run fails immediately at startup when none can be inferred.
Local `parquet`/`zip` sources are materialized into `data_dir` (bytes must be
extracted); the local source files are never deleted.

## Artifacts layout (`work_dir`)

```
state/<stage>/rank_XXXXX.SUCCESS      barriers + resume
artifacts/headerscan/                 kept/removed per rank
artifacts/fingerprint/                hashes + laplacian per rank
artifacts/dedup/survivors.parquet     global post-dedup table
artifacts/embeddings/*.parquet        image_id, path, fp16 embedding (swap-in point for pgvector later)
artifacts/clustering/                 assignments, summary, centroids.npy
artifacts/manifest/manifest.parquet   final balanced manifest (+ .csv when <= 1M rows)
viz/viz.parquet                       sampled 2D projection for the viewer
```

Deletion policy: filter and dedup stages hard-delete files (use `--dry-run`
first); cluster pruning only flags rows and never touches files.
