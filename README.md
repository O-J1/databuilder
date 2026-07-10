# databuilder

Config-driven, distributed image dataset builder. Downloads HuggingFace datasets,
filters (size / aspect / broken / Laplacian), deduplicates (md5 + 12x12 phash +
colorhash), embeds with DINOv3, clusters with usearch k-means, prunes
over-represented clusters, and emits a generator- and cluster-balanced manifest.

Self-contained library: install into any venv/uv/pixi environment, no admin needed.

```bash
pip install -e ./databuilder            # core pipeline
pip install -e "./databuilder[embed]"   # + torch/transformers for the embed stage
pip install -e "./databuilder[viz]"    # + FastAPI/uvicorn/umap for the viewer
```

## Pipeline

| # | Stage        | Scope     | What it does |
|---|--------------|-----------|--------------|
| 1 | `download`   | per node  | HF snapshot + materialize (parquet bytes / zip / imagefolder) |
| 2 | `headerscan` | per node  | delete: longest side < min, broken headers, aspect beyond 9:23 / 23:9 |
| 3 | `fingerprint`| per node  | one decode: Laplacian filter + md5 + phash + colorhash |
| 4 | `dedup`      | rank 0    | global exact + near-duplicate delete (keep highest res, then largest file) |
| 5 | `embed`      | per node  | DINOv3 embeddings -> parquet shards (fp16) per GPU |
| 6 | `cluster`    | rank 0    | usearch/sklearn k-means; flag over-represented members (never deletes) |
| 7 | `manifest`   | rank 0    | balanced manifest; parquet always, CSV refused above 1M rows |

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

The log states which source was used. Rank 0 runs the global stages (dedup,
cluster, manifest); other ranks wait at barriers and resume automatically.
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
- `[filters]` - `min_longest_side`, `max_tall = "9:23"`, `max_wide = "23:9"`, `laplacian_min/max`
- `[dedup]` - `phash_size = 12`, `phash_max_hamming`, `colorhash_max_hamming`
- `[embedding]` - DINOv3 `model` id (any size), `batch_size`, `devices = "auto"`
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
  `source_split` picks the HF split to download; `assign_split = "test"` forces
  the whole dataset into one output split (forced val/test bypass balancing
  caps and cluster pruning).

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
