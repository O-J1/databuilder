from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..store import ParquetEmbeddingStore
from ..utils import ParquetShardWriter, iter_parquet_batches

log = logging.getLogger("databuilder.viz")

VIZ_SCHEMA = pa.schema(
    [
        ("image_id", pa.uint64()),
        ("path", pa.string()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("cluster_id", pa.int32()),
        ("pruned", pa.bool_()),
        ("generator", pa.string()),
        ("dataset", pa.string()),
        ("label", pa.string()),
    ]
)

PAIRS_SCHEMA = pa.schema(
    [
        ("kind", pa.string()),  # "dedup" | "cluster"
        ("pruned_image_id", pa.uint64()),
        ("pruned_path", pa.string()),
        ("kept_image_id", pa.uint64()),
        ("kept_path", pa.string()),
        ("cluster_id", pa.int32()),  # -1 for dedup pairs
        ("reason", pa.string()),
        ("dist", pa.float32()),  # embedding L2; 0 for dedup pairs
    ]
)


def stratified_quotas(
    sizes: dict[int, int], sample_size: int, min_per_cluster: int
) -> dict[int, int]:
    """Proportional allocation with a per-cluster floor so every cluster is visible.

    If floors alone exceed sample_size the result overshoots (documented behaviour).
    """
    total = sum(sizes.values())
    if total <= sample_size:
        return dict(sizes)
    quotas = {c: min(s, min_per_cluster) for c, s in sizes.items()}
    remaining = sample_size - sum(quotas.values())
    if remaining <= 0:
        return quotas
    leftover = {c: sizes[c] - quotas[c] for c in sizes}
    denom = sum(leftover.values())
    for c in sorted(sizes):
        if denom <= 0:
            break
        add = int(round(leftover[c] / denom * remaining))
        quotas[c] = min(sizes[c], quotas[c] + add)
    return quotas


def _project(matrix: np.ndarray, seed: int) -> np.ndarray:
    if len(matrix) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    try:
        import umap

        log.info("projecting %d points with UMAP", len(matrix))
        coords = umap.UMAP(n_components=2, random_state=seed).fit_transform(matrix)
    except ImportError:
        from sklearn.decomposition import PCA

        log.info("umap-learn not installed; projecting %d points with PCA", len(matrix))
        components = min(2, matrix.shape[1], max(1, len(matrix) - 1))
        coords = PCA(n_components=components, random_state=seed).fit_transform(matrix)
        if coords.shape[1] < 2:
            coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    return np.asarray(coords, dtype=np.float32)


def prepare(
    work_dir: Path | str,
    sample_size: int = 200_000,
    min_per_cluster: int = 20,
    seed: int = 42,
    data_dir: Path | str | None = None,
    roots: dict[str, str] | None = None,
) -> Path:
    """Stratified-sample embeddings per cluster, project to 2D, write viz.parquet."""
    work_dir = Path(work_dir)
    store = ParquetEmbeddingStore(work_dir / "artifacts" / "embeddings")
    assignments_path = work_dir / "artifacts" / "clustering" / "cluster_assignments.parquet"

    a_ids = np.array([], dtype=np.uint64)
    a_clusters = np.array([], dtype=np.int64)
    a_pruned = np.array([], dtype=bool)
    if assignments_path.exists():
        table = pq.read_table(assignments_path)
        a_ids = table.column("image_id").to_numpy(zero_copy_only=False).astype(np.uint64)
        a_clusters = table.column("cluster_id").to_numpy(zero_copy_only=False).astype(np.int64)
        a_pruned = table.column("pruned").to_numpy(zero_copy_only=False).astype(bool)
        order = np.argsort(a_ids)
        a_ids, a_clusters, a_pruned = a_ids[order], a_clusters[order], a_pruned[order]
    else:
        log.warning("no cluster assignments found; falling back to uniform sampling")

    def cluster_of(ids: np.ndarray) -> np.ndarray:
        if len(a_ids) == 0:
            return np.full(len(ids), -1, dtype=np.int64)
        pos = np.clip(np.searchsorted(a_ids, ids), 0, len(a_ids) - 1)
        found = a_ids[pos] == ids
        return np.where(found, a_clusters[pos], -1)

    def pruned_of(ids: np.ndarray) -> np.ndarray:
        if len(a_ids) == 0:
            return np.zeros(len(ids), dtype=bool)
        pos = np.clip(np.searchsorted(a_ids, ids), 0, len(a_ids) - 1)
        found = a_ids[pos] == ids
        return np.where(found, a_pruned[pos], False)

    if len(a_ids):
        unique, counts = np.unique(a_clusters, return_counts=True)
        sizes = {int(c): int(n) for c, n in zip(unique, counts)}
    else:
        sizes = {-1: store.count()}
    quotas = stratified_quotas(sizes, sample_size, min_per_cluster)

    # Per-cluster reservoir sampling over streamed embedding shards.
    rng = random.Random(seed)
    reservoirs: dict[int, list[tuple[int, str, np.ndarray]]] = {c: [] for c in quotas}
    seen: dict[int, int] = {c: 0 for c in quotas}
    for ids, matrix, paths in store.iter_batches(batch_rows=8192, with_paths=True):
        batch_clusters = cluster_of(ids)
        for i in range(len(ids)):
            cluster = int(batch_clusters[i])
            quota = quotas.get(cluster, 0)
            if quota <= 0:
                continue
            seen[cluster] += 1
            entry = (int(ids[i]), paths[i], matrix[i].astype(np.float16))
            pool = reservoirs[cluster]
            if len(pool) < quota:
                pool.append(entry)
            else:
                j = rng.randrange(seen[cluster])
                if j < quota:
                    pool[j] = entry

    sampled = [entry for pool in reservoirs.values() for entry in pool]
    sampled.sort(key=lambda e: e[0])
    log.info("sampled %d points across %d clusters", len(sampled), len(quotas))

    sampled_ids = np.array([e[0] for e in sampled], dtype=np.uint64)
    sampled_paths = [e[1] for e in sampled]
    matrix = (
        np.stack([e[2] for e in sampled]).astype(np.float32)
        if sampled
        else np.zeros((0, 2), np.float32)
    )
    coords = _project(matrix, seed)
    del matrix

    # Join metadata for the sampled ids from the survivors table.
    survivors = work_dir / "artifacts" / "dedup" / "survivors.parquet"
    meta: dict[int, tuple[str, str, str]] = {}
    if survivors.exists():
        for batch in iter_parquet_batches(
            survivors, columns=["image_id", "generator", "dataset", "label"]
        ):
            for row in batch.to_pylist():
                if row["image_id"] in meta or not _contains(sampled_ids, row["image_id"]):
                    continue
                meta[row["image_id"]] = (row["generator"], row["dataset"], row["label"])

    out_path = work_dir / "viz" / "viz.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clusters = cluster_of(sampled_ids).astype(np.int32)
    pruned = pruned_of(sampled_ids)
    table = pa.Table.from_arrays(
        [
            pa.array(sampled_ids, type=pa.uint64()),
            pa.array(sampled_paths, type=pa.string()),
            pa.array(coords[:, 0], type=pa.float32()),
            pa.array(coords[:, 1], type=pa.float32()),
            pa.array(clusters, type=pa.int32()),
            pa.array(pruned, type=pa.bool_()),
            pa.array([meta.get(i, ("", "", ""))[0] for i in sampled_ids.tolist()]),
            pa.array([meta.get(i, ("", "", ""))[1] for i in sampled_ids.tolist()]),
            pa.array([meta.get(i, ("", "", ""))[2] for i in sampled_ids.tolist()]),
        ],
        schema=VIZ_SCHEMA,
    )
    metadata = {
        b"databuilder.data_dir": str(data_dir or "").encode(),
        b"databuilder.roots": json.dumps(roots or {}).encode(),
    }
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, out_path)
    log.info("wrote %s", out_path)
    return out_path


def _contains(sorted_ids: np.ndarray, value: int) -> bool:
    if len(sorted_ids) == 0:
        return False
    pos = int(np.searchsorted(sorted_ids, np.uint64(value)))
    return pos < len(sorted_ids) and int(sorted_ids[pos]) == value


def prepare_pairs(
    work_dir: Path | str,
    kept_sample: int = 2048,
    seed: int = 42,
) -> Path:
    """Write viz/pairs.parquet: each pruned/removed image next to its kept counterpart.

    Two pair kinds:
    - "dedup": exact/near duplicates removed by the dedup stage, linked to the
      surviving winner via removed.parquet's kept_image_id column.
    - "cluster": rows flagged by cluster pruning, paired with their nearest kept
      embedding in the same cluster (against at most `kept_sample` kept members
      per cluster to bound compute at scale).

    Streams parquet shards throughout; peak memory is bounded by the sorted
    assignment id arrays plus `clusters_with_pruned * kept_sample` embeddings.
    """
    work_dir = Path(work_dir)
    out_path = work_dir / "viz" / "pairs.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ParquetShardWriter(out_path, PAIRS_SCHEMA) as writer:
        n_dedup = _dedup_pairs(work_dir, writer)
        n_cluster = _cluster_pairs(work_dir, writer, kept_sample, seed)
    if n_dedup + n_cluster == 0:
        # ensure the file exists so the server can distinguish "prepared, empty"
        pq.write_table(PAIRS_SCHEMA.empty_table(), out_path)
    log.info("wrote %s (%d dedup pairs, %d cluster pairs)", out_path, n_dedup, n_cluster)
    return out_path


def _dedup_pairs(work_dir: Path, writer: ParquetShardWriter) -> int:
    removed = work_dir / "artifacts" / "dedup" / "removed.parquet"
    survivors = work_dir / "artifacts" / "dedup" / "survivors.parquet"
    if not removed.exists() or not survivors.exists():
        log.info("no dedup artifacts found; skipping dedup pairs")
        return 0
    if "kept_image_id" not in pq.ParquetFile(removed).schema_arrow.names:
        log.warning(
            "%s predates counterpart tracking (no kept_image_id column); "
            "re-run the dedup stage to enable dedup pairs",
            removed,
        )
        return 0

    kept_chunks: list[np.ndarray] = []
    for batch in iter_parquet_batches(removed, columns=["kept_image_id"]):
        kept_chunks.append(
            batch.column("kept_image_id").to_numpy(zero_copy_only=False).astype(np.uint64)
        )
    if not kept_chunks:
        return 0
    winners = np.unique(np.concatenate(kept_chunks))
    winner_paths: list[str] = [""] * len(winners)
    for batch in iter_parquet_batches(survivors, columns=["image_id", "path"]):
        ids = batch.column("image_id").to_numpy(zero_copy_only=False).astype(np.uint64)
        pos = np.searchsorted(winners, ids)
        in_range = pos < len(winners)
        hit = np.zeros(len(ids), dtype=bool)
        hit[in_range] = winners[pos[in_range]] == ids[in_range]
        if not hit.any():
            continue
        paths = batch.column("path").to_pylist()
        for i in np.nonzero(hit)[0]:
            winner_paths[int(pos[i])] = paths[int(i)]

    count = 0
    columns = ["image_id", "path", "kept_image_id", "reason"]
    for batch in iter_parquet_batches(removed, columns=columns):
        for row in batch.to_pylist():
            k = int(np.searchsorted(winners, np.uint64(row["kept_image_id"])))
            writer.append(
                {
                    "kind": "dedup",
                    "pruned_image_id": row["image_id"],
                    "pruned_path": row["path"],
                    "kept_image_id": row["kept_image_id"],
                    "kept_path": winner_paths[k],
                    "cluster_id": -1,
                    "reason": row["reason"],
                    "dist": 0.0,
                }
            )
            count += 1
    return count


def _cluster_pairs(
    work_dir: Path, writer: ParquetShardWriter, kept_sample: int, seed: int
) -> int:
    assignments = work_dir / "artifacts" / "clustering" / "cluster_assignments.parquet"
    if not assignments.exists():
        log.info("no cluster assignments found; skipping cluster pairs")
        return 0
    table = pq.read_table(assignments, columns=["image_id", "cluster_id", "pruned"])
    a_ids = table.column("image_id").to_numpy(zero_copy_only=False).astype(np.uint64)
    a_clusters = table.column("cluster_id").to_numpy(zero_copy_only=False).astype(np.int64)
    a_pruned = table.column("pruned").to_numpy(zero_copy_only=False).astype(bool)
    order = np.argsort(a_ids)
    a_ids, a_clusters, a_pruned = a_ids[order], a_clusters[order], a_pruned[order]
    pruned_clusters = np.unique(a_clusters[a_pruned])
    if len(pruned_clusters) == 0:
        return 0

    def lookup(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = np.clip(np.searchsorted(a_ids, ids), 0, len(a_ids) - 1)
        found = a_ids[pos] == ids
        clusters = np.where(found, a_clusters[pos], -1)
        pruned = np.where(found, a_pruned[pos], False)
        return clusters, pruned

    store = ParquetEmbeddingStore(work_dir / "artifacts" / "embeddings")

    # Pass 1: reservoir-sample kept members per pruned cluster (bounds compute).
    rng = random.Random(seed)
    kept: dict[int, list[tuple[int, str, np.ndarray]]] = {int(c): [] for c in pruned_clusters}
    seen: dict[int, int] = {int(c): 0 for c in pruned_clusters}
    for ids, matrix, paths in store.iter_batches(batch_rows=8192, with_paths=True):
        clusters, pruned = lookup(ids)
        candidates = np.nonzero(~pruned & np.isin(clusters, pruned_clusters))[0]
        for i in candidates:
            cluster = int(clusters[i])
            seen[cluster] += 1
            entry = (int(ids[i]), paths[i], matrix[i].astype(np.float16))
            pool = kept[cluster]
            if len(pool) < kept_sample:
                pool.append(entry)
            else:
                j = rng.randrange(seen[cluster])
                if j < kept_sample:
                    pool[j] = entry

    kept_mats = {
        c: np.stack([e[2] for e in pool]).astype(np.float32)
        for c, pool in kept.items()
        if pool
    }

    # Pass 2: stream pruned rows, emit nearest kept member per row.
    count = 0
    for ids, matrix, paths in store.iter_batches(batch_rows=8192, with_paths=True):
        clusters, pruned = lookup(ids)
        rows = np.nonzero(pruned)[0]
        for cluster in np.unique(clusters[rows]).tolist():
            cluster = int(cluster)
            mat = kept_mats.get(cluster)
            if mat is None:
                continue
            members = rows[clusters[rows] == cluster]
            vecs = matrix[members]
            dist2 = (
                (vecs**2).sum(axis=1)[:, None]
                - 2.0 * vecs @ mat.T
                + (mat**2).sum(axis=1)[None, :]
            )
            best = dist2.argmin(axis=1)
            dists = np.sqrt(np.clip(dist2[np.arange(len(best)), best], 0, None))
            pool = kept[cluster]
            for row_idx, kept_idx, dist in zip(members.tolist(), best.tolist(), dists.tolist()):
                kept_id, kept_path, _ = pool[kept_idx]
                writer.append(
                    {
                        "kind": "cluster",
                        "pruned_image_id": int(ids[row_idx]),
                        "pruned_path": paths[row_idx],
                        "kept_image_id": kept_id,
                        "kept_path": kept_path,
                        "cluster_id": cluster,
                        "reason": "semantic_duplicate",
                        "dist": float(dist),
                    }
                )
                count += 1
    return count
