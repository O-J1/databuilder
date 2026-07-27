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
| 1 | `download`   | rank 0    | one-worker HF/Xet snapshot, streamed into canonical WebDataset shards |
| 2 | `headerscan` | per node  | reject: longest side < min, broken headers, aspect beyond 9:23 / 23:9 |
| 3 | `fingerprint`| per node* | one decode: Laplacian filter + xxh3 + phash (12x12) + colorhash |
| 4 | `dedup`      | rank 0    | global exact + near-duplicate selection (keep highest res, then largest file) |
| 5 | `embed`      | per node* | DINOv3 embeddings -> parquet shards (fp16) per GPU |
| 6 | `cluster`    | rank 0    | usearch/sklearn k-means; flag over-represented members (never deletes) |
| 7 | `manifest`   | rank 0    | balanced manifest; optionally compact shards to final survivors |

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
- `[download]` - Hugging Face `snapshot_download` concurrency and ephemeral
  staging. `max_workers = 1` means one dataset-file transfer at a time on rank
  0; all other ranks wait and never contact Hugging Face. `staging_dir` can be
  placed on a filesystem with enough temporary capacity. Xet owns transfer
  resume and all transfer tuning remains at the user/Hugging Face defaults;
  databuilder only points `HF_XET_CACHE` at staging. Successful conversion
  removes snapshots and Xet cache unless `retain_snapshots` or
  `retain_xet_cache` is explicitly enabled.
- `[storage]` - canonical `webdataset` output. The default limits—3,000,000,000
  bytes or 100,000 samples per shard—match WebDataset's performance-oriented
  `ShardWriter` defaults. `compact_after_manifest = true` rewrites shards to
  contain only final manifest rows and bounds temporary space to one shard.
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
  Arrow, JSONL with local image paths, zip, tar/WebDataset, concatenated
  `multipart_tar` chunks, split zip, imagefolder, and `raw`. External image
  URLs are never fetched individually.
  `download_only = true` with `format = "raw"` stores the selected repository
  files in a tar but excludes it from headerscan and every downstream stage.
  `source_split` picks
  the HF split to download; `assign_split = "test"` forces
  the whole dataset into one output split (forced val/test bypass balancing
  caps and cluster pruning).

The requested 40-repository corpus is ready in
[`examples/aigc-datasets.toml`](examples/aigc-datasets.toml). It pins every
revision, writes only canonical tar/WebDataset output below
`/p/data1/datasets/mmlaion/aigc/data`, selects generated outputs (not source
references) from UniPic, and preserves URL-only repositories as raw tar
archives. Its snapshot/Xet staging is configured separately and is removed
after successful conversion. The card/schema audit and real/fake decisions are recorded in
[`docs/aigc-dataset-labels.md`](docs/aigc-dataset-labels.md).

### Canonical storage and offline migration

Materialized image datasets contain uncompressed tar shards under `shards/`.
Each WebDataset sample is a stable `<image-id>.<extension>` plus
`<image-id>.json` pair. `index.parquet` maps the existing logical image path
to its shard, member, byte offset, size, label, generator, and source split;
`dataset.json` is the shard descriptor. Headerscan, fingerprint, Daft,
embedding, manifests, and the viewer read those byte ranges directly. They do
not reconstruct JPEG/PNG trees.

Existing JSC data can be converted without any Hugging Face request:

```bash
databuilder storage inventory --config examples/aigc-datasets.toml
databuilder storage migrate --existing-only --config examples/aigc-datasets.toml
databuilder storage inventory --config examples/aigc-datasets.toml
```

Migration consumes whatever is already local: completed or partial loose
materializations, legacy `.hf_snapshots`, configured staging snapshots, and
local filesystem sources. A loose source image is removed only after its
containing shard has been closed, validated, fsynced, and committed. Existing
snapshots and Xet cache are removed after the dataset marker is committed.
Rows reported as `needs_download` are left untouched; `storage migrate` never
calls Hugging Face. Regular `databuilder run` also auto-packs a completed
legacy loose dataset before considering a download.

If the old `work_dir` already contains downstream `SUCCESS` markers, point the
next run at a fresh `work_dir`; stage artifacts written before this storage
schema do not contain shard locators. The committed dataset markers still
prevent any re-download.

With `compact_after_manifest = true` (enabled in the AIGC config), filtering,
deduplication, clustering, and balancing first operate as metadata decisions.
The manifest stage then rewrites each shard once with only selected samples and
updates all offsets in `manifest.parquet`. To compact an already-built
manifest manually:

```bash
databuilder storage compact --config examples/aigc-datasets.toml
```

### URL-backed image datasets

The main pipeline snapshots URL metadata but deliberately does not issue one
HTTP request per image. `kafked/anycrap` and `lehduong/seaart-hq` can instead
be downloaded independently with:

```bash
python scripts/download_url_datasets.py \
  --data-dir ./data \
  --staging-dir ./databuilder-staging \
  --workers 16
```

Use `--dataset anycrap` or `--dataset seaart-hq` to select one source,
`--limit 100 --dry-run` to inspect its metadata count, and
`--skip-metadata-snapshot` to make no Hugging Face call when the pinned
metadata snapshot is already present. Downloads are streamed, size-limited,
decoded before acceptance, then committed into the same 3 GB/100k-sample
WebDataset layout. Validated temporary images are deleted after their shard is
committed; an interrupted run reuses them. Metadata snapshots and Xet cache
are removed after success. Only the compact `url-downloads.jsonl` and
`url-failures.jsonl` logs remain beside the shard index—there is no loose
`url-images/` tree.

This standalone download does not edit the TOML. To admit a successfully
downloaded URL dataset to a later build, change that existing entry from
`format = "raw"`/`download_only = true` to `format = "webdataset"`, remove
`download_only`, and retain its static `label = "fake"` and generator. The
committed storage marker makes the download stage reuse the shards.

### Local filesystem datasets

New local `imagefolder` datasets are packed into canonical shards. Their source
files remain untouched unless `allow_delete = true`; legacy in-place datasets
are still readable under the same protection rule. With
`label = "folder"`, label folders (`real`, `fake`,
`generated`, `ai`, `aigc`, or `label_map` entries) are matched up to two levels
deep, and the run fails immediately at startup when none can be inferred.
Local parquet/archive sources are streamed into `data_dir` shards; the local
source files are never deleted.

## Canonical data layout (`data_dir`)

```text
<dataset>/shards/<dataset>-NNNNNN.tar  image + JSON WebDataset samples
<dataset>/index.parquet                logical paths, labels, and byte ranges
<dataset>/dataset.json                 shard list and sizes
<dataset>/.materialized.json           committed storage state
<raw-dataset>/raw/<dataset>.tar        download-only repository payload
```

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

Deletion policy: archive-backed filter/dedup rejects are metadata-only until
manifest compaction; compaction physically removes every row not selected for
the final manifest. Legacy loose and explicitly deletable in-place sources
retain the old immediate-delete behavior. Cluster pruning itself only flags
rows.
