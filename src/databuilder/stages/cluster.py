from __future__ import annotations

import logging
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..state import RunContext
from ..store import ParquetEmbeddingStore
from ..utils import iter_parquet_batches

log = logging.getLogger("databuilder.cluster")

ASSIGNMENTS_SCHEMA = pa.schema(
    [
        ("image_id", pa.uint64()),
        ("cluster_id", pa.int32()),
        ("dist", pa.float32()),
        ("pruned", pa.bool_()),
        ("prune_reason", pa.string()),
    ]
)
SUMMARY_SCHEMA = pa.schema(
    [("cluster_id", pa.int32()), ("size", pa.int64()), ("pruned", pa.int64())]
)


def choose_k(n: int, k: int, aggressiveness: float) -> int:
    """Explicit k wins; otherwise scale sqrt(n) by aggressiveness.

    aggressiveness 0.5 -> ~sqrt(n) clusters, 1.0 -> ~4*sqrt(n), ->0 -> ~sqrt(n)/4.
    """
    if k > 0:
        return min(k, n)
    derived = int(round(math.sqrt(max(n, 1)) * 4 ** (2 * aggressiveness - 1)))
    return max(1, min(max(16, derived), n))


def run(ctx: RunContext) -> None:
    """Cluster all embeddings, then flag (never delete) semantic near-duplicates.

    Pruning is deliberately conservative: only clusters whose size is an
    extreme outlier (> mean + prune_trigger_sigma * std) are examined, and
    within them only members that are semantic near-duplicates of an already
    kept member (cosine similarity > semdedup_threshold) are flagged. Merely
    large but diverse clusters are left untouched.
    """
    if ctx.dry_run:
        log.info("dry-run: skipping clustering")
        return
    cfg = ctx.cfg.clustering
    store = ParquetEmbeddingStore(ctx.artifact_dir("embeddings"))
    n, dim = store.count(), store.dim()
    k = choose_k(n, cfg.k, cfg.aggressiveness)
    log.info("clustering %d embeddings (dim=%d) into k=%d", n, dim, k)

    centroids = _fit(store, n, dim, k, cfg)
    ids, clusters, dists = _assign(store, centroids)
    exempt = _exempt_mask(ctx, ids)
    pruned = _prune_semantic_duplicates(store, ids, clusters, dists, k, cfg, exempt)

    out_dir = ctx.artifact_dir("clustering")
    np.save(out_dir / "centroids.npy", centroids)
    reasons = np.where(pruned, "semantic_duplicate", "")
    table = pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.uint64()),
            pa.array(clusters.astype(np.int32), type=pa.int32()),
            pa.array(dists.astype(np.float32), type=pa.float32()),
            pa.array(pruned, type=pa.bool_()),
            pa.array(reasons.tolist(), type=pa.string()),
        ],
        schema=ASSIGNMENTS_SCHEMA,
    )
    pq.write_table(table, out_dir / "cluster_assignments.parquet")

    sizes = np.bincount(clusters, minlength=k)
    pruned_counts = np.bincount(clusters[pruned], minlength=k)
    summary = pa.Table.from_arrays(
        [
            pa.array(np.arange(k, dtype=np.int32), type=pa.int32()),
            pa.array(sizes.astype(np.int64), type=pa.int64()),
            pa.array(pruned_counts.astype(np.int64), type=pa.int64()),
        ],
        schema=SUMMARY_SCHEMA,
    )
    pq.write_table(summary, out_dir / "cluster_summary.parquet")
    log.info(
        "clustering done: %d clusters, %d/%d rows pruned", k, int(pruned.sum()), len(pruned)
    )


def _fit(
    store: ParquetEmbeddingStore, n: int, dim: int, k: int, cfg
) -> np.ndarray:
    estimated_bytes = n * dim * 4
    fits_ram = estimated_bytes <= cfg.max_ram_gb * 1e9
    if cfg.backend in {"auto", "usearch"} and fits_ram:
        try:
            return _fit_usearch(store, k, cfg.seed)
        except Exception as exc:  # noqa: BLE001 - fall back to sklearn
            if cfg.backend == "usearch":
                raise
            log.warning("usearch kmeans unavailable (%s); using sklearn", exc)
    return _fit_sklearn(store, k, cfg.seed)


def _fit_usearch(store: ParquetEmbeddingStore, k: int, seed: int) -> np.ndarray:
    from usearch.index import kmeans

    matrix = np.concatenate([mat for _, mat in store.iter_batches(batch_rows=65_536)])
    result = kmeans(matrix, k)
    centroids = getattr(result, "centroids", None)
    if centroids is None:
        assignments = np.asarray(result, dtype=np.int64).reshape(-1)
        centroids = np.zeros((k, matrix.shape[1]), dtype=np.float64)
        counts = np.bincount(assignments, minlength=k).clip(min=1)
        np.add.at(centroids, assignments, matrix)
        centroids /= counts[:, None]
    return np.asarray(centroids, dtype=np.float32)


def _fit_sklearn(store: ParquetEmbeddingStore, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=8192, n_init=3)
    buffer: list[np.ndarray] = []
    buffered = 0
    started = False
    for _epoch in range(2):
        for _, matrix in store.iter_batches(batch_rows=8192):
            if not started:
                buffer.append(matrix)
                buffered += len(matrix)
                if buffered < max(k, 8192):
                    continue
                km.partial_fit(np.concatenate(buffer))
                buffer, buffered, started = [], 0, True
                continue
            km.partial_fit(matrix)
    if not started:
        # tiny dataset: everything is still in the buffer
        matrix = np.concatenate(buffer) if buffer else np.zeros((0, store.dim()), np.float32)
        km = MiniBatchKMeans(
            n_clusters=min(k, max(1, len(matrix))), random_state=seed, n_init=3
        ).fit(matrix)
    return km.cluster_centers_.astype(np.float32)


def _assign(
    store: ParquetEmbeddingStore, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid_norms = (centroids**2).sum(axis=1)
    all_ids: list[np.ndarray] = []
    all_clusters: list[np.ndarray] = []
    all_dists: list[np.ndarray] = []
    for ids, matrix in store.iter_batches(batch_rows=16_384):
        dist2 = (
            (matrix**2).sum(axis=1)[:, None]
            - 2.0 * matrix @ centroids.T
            + centroid_norms[None, :]
        )
        best = dist2.argmin(axis=1)
        all_ids.append(ids)
        all_clusters.append(best.astype(np.int64))
        all_dists.append(np.sqrt(np.clip(dist2[np.arange(len(best)), best], 0, None)))
    if not all_ids:
        empty = np.zeros(0)
        return empty.astype(np.uint64), empty.astype(np.int64), empty.astype(np.float32)
    return (
        np.concatenate(all_ids),
        np.concatenate(all_clusters),
        np.concatenate(all_dists),
    )


def _exempt_mask(ctx: RunContext, ids: np.ndarray) -> np.ndarray | None:
    """Rows from datasets forced to val/test are never pruned."""
    forced = {ds.name for ds in ctx.cfg.datasets if ds.assign_split in {"val", "test"}}
    if not forced:
        return None
    survivors = ctx.artifact_dir("dedup") / "survivors.parquet"
    exempt_ids: list[np.ndarray] = []
    for batch in iter_parquet_batches(survivors, columns=["image_id", "dataset"]):
        batch_ids = batch.column("image_id").to_numpy(zero_copy_only=False).astype(np.uint64)
        in_forced = np.array(
            [name in forced for name in batch.column("dataset").to_pylist()], dtype=bool
        )
        exempt_ids.append(batch_ids[in_forced])
    if not exempt_ids:
        return None
    key = np.sort(np.concatenate(exempt_ids))
    if len(key) == 0:
        return None
    pos = np.clip(np.searchsorted(key, ids), 0, len(key) - 1)
    return key[pos] == ids


def trigger_clusters(sizes: np.ndarray, sigma: float) -> np.ndarray:
    """Cluster ids whose size is an extreme outlier: > mean + sigma * std.

    Statistics use non-empty clusters only. Homogeneous cluster sizes never
    trigger (a cluster must be strictly above the mean even at sigma=0).
    """
    sizes = np.asarray(sizes)
    nonzero = sizes[sizes > 0]
    if len(nonzero) == 0:
        return np.zeros(0, dtype=np.int64)
    cutoff = float(nonzero.mean()) + sigma * float(nonzero.std())
    return np.nonzero(sizes > cutoff)[0].astype(np.int64)


def semdedup_prune(
    embeddings: np.ndarray,
    dists: np.ndarray,
    ids: np.ndarray,
    threshold: float,
    exempt: np.ndarray | None = None,
) -> np.ndarray:
    """Flag semantic near-duplicates among one cluster's members.

    Members are visited exempt-first (forced val/test rows are never pruned
    and serve as anchors), then farthest-from-centroid first (SemDeDup keeps
    diverse representatives), tie-broken by image_id. A member is pruned when
    its cosine similarity to any already-kept member exceeds `threshold`.
    Deterministic; only redundant members are flagged, unique ones survive
    regardless of cluster size.
    """
    n = len(ids)
    pruned = np.zeros(n, dtype=bool)
    if n < 2:
        return pruned
    unit = np.asarray(embeddings, dtype=np.float32)
    unit = unit / np.clip(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12, None)
    is_exempt = exempt if exempt is not None else np.zeros(n, dtype=bool)
    order = np.lexsort((ids, -np.asarray(dists, dtype=np.float64), ~is_exempt))
    kept = np.empty_like(unit)
    count = 0
    for idx in order:
        if not is_exempt[idx] and count:
            sims = kept[:count] @ unit[idx]
            if float(sims.max()) > threshold:
                pruned[idx] = True
                continue
        kept[count] = unit[idx]
        count += 1
    return pruned


def _gather_embeddings(store: ParquetEmbeddingStore, wanted: np.ndarray) -> np.ndarray:
    """Embeddings for `wanted` image_ids, rows aligned with `wanted`."""
    order = np.argsort(wanted)
    sorted_ids = wanted[order]
    out = np.zeros((len(wanted), store.dim()), dtype=np.float32)
    for batch_ids, matrix in store.iter_batches(batch_rows=16_384):
        pos = np.clip(np.searchsorted(sorted_ids, batch_ids), 0, len(sorted_ids) - 1)
        hit = sorted_ids[pos] == batch_ids
        if hit.any():
            out[order[pos[hit]]] = matrix[hit]
    return out


def _prune_semantic_duplicates(
    store: ParquetEmbeddingStore,
    ids: np.ndarray,
    clusters: np.ndarray,
    dists: np.ndarray,
    k: int,
    cfg,
    exempt: np.ndarray | None,
) -> np.ndarray:
    """Hybrid pruning: SemDeDup inside extreme size-outlier clusters only."""
    pruned = np.zeros(len(ids), dtype=bool)
    if len(ids) == 0:
        return pruned
    active = ~exempt if exempt is not None else np.ones(len(ids), dtype=bool)
    sizes = np.bincount(clusters[active], minlength=k)
    triggered = trigger_clusters(sizes, cfg.prune_trigger_sigma)
    if len(triggered) == 0:
        return pruned
    member_rows = np.nonzero(np.isin(clusters, triggered))[0]
    embeddings = _gather_embeddings(store, ids[member_rows])
    log.info(
        "semdedup: examining %d outlier cluster(s) (%d members)",
        len(triggered),
        len(member_rows),
    )
    for cluster_id in triggered:
        local = np.nonzero(clusters[member_rows] == cluster_id)[0]
        rows = member_rows[local]
        pruned[rows] = semdedup_prune(
            embeddings[local],
            dists[rows],
            ids[rows],
            cfg.semdedup_threshold,
            exempt=exempt[rows] if exempt is not None else None,
        )
    return pruned
