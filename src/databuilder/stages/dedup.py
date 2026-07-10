from __future__ import annotations

import logging

import numpy as np
import pyarrow as pa

from ..state import RunContext
from ..utils import ParquetShardWriter, iter_parquet_batches
from .common import dataset_roots, protected_datasets, resolve_abs_from_roots
from .fingerprint import FINGERPRINT_SCHEMA

log = logging.getLogger("databuilder.dedup")

REMOVED_SCHEMA = pa.schema(
    [
        ("image_id", pa.uint64()),
        ("path", pa.string()),
        ("dataset", pa.string()),
        ("reason", pa.string()),
        ("kept_image_id", pa.uint64()),
    ]
)
_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint16)
BRUTE_FORCE_MAX = 30_000
NEIGHBORS = 16


def run(ctx: RunContext) -> None:
    """Global exact + near duplicate removal (rank 0 only).

    Keep-best order inside a duplicate group: highest resolution, then largest
    file size, then smallest image_id (deterministic).
    Memory note: holds one md5->best mapping plus packed phash/colorhash arrays
    for all surviving images; ~60 bytes per image.
    """
    fp_dir = ctx.artifact_dir("fingerprint")
    fp_files = sorted(fp_dir.glob("rank_*.parquet"))
    fp_files = [f for f in fp_files if not f.name.endswith(".removed.parquet")]
    out_dir = ctx.artifact_dir("dedup")

    # Pass 1: best row per md5.
    best: dict[bytes, tuple[int, int, int]] = {}
    for batch in iter_parquet_batches(
        fp_files, columns=["image_id", "md5", "width", "height", "filesize"]
    ):
        for row in batch.to_pylist():
            key = row["md5"]
            rank_key = (row["width"] * row["height"], row["filesize"], -row["image_id"])
            if key not in best or rank_key > best[key]:
                best[key] = rank_key

    # Pass 2: collect md5 winners' hashes for near-duplicate search.
    ids: list[np.ndarray] = []
    res: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    phashes: list[np.ndarray] = []
    colorhashes: list[np.ndarray] = []
    for batch in iter_parquet_batches(
        fp_files,
        columns=["image_id", "md5", "width", "height", "filesize", "phash", "colorhash"],
    ):
        rows = batch.to_pylist()
        keep = [r for r in rows if best[r["md5"]][2] == -r["image_id"]]
        if not keep:
            continue
        ids.append(np.array([r["image_id"] for r in keep], dtype=np.uint64))
        res.append(np.array([r["width"] * r["height"] for r in keep], dtype=np.int64))
        sizes.append(np.array([r["filesize"] for r in keep], dtype=np.int64))
        phashes.append(np.stack([np.frombuffer(r["phash"], dtype=np.uint8) for r in keep]))
        colorhashes.append(
            np.stack([np.frombuffer(r["colorhash"], dtype=np.uint8) for r in keep])
        )

    near_losers: dict[int, int] = {}
    if ids:
        all_ids = np.concatenate(ids)
        all_res = np.concatenate(res)
        all_sizes = np.concatenate(sizes)
        phash_mat = np.vstack(phashes)
        color_mat = np.vstack(colorhashes)
        near_losers = _near_duplicate_losers(
            all_ids,
            all_res,
            all_sizes,
            phash_mat,
            color_mat,
            ctx.cfg.dedup.phash_max_hamming,
            ctx.cfg.dedup.colorhash_max_hamming,
        )
        del all_res, all_sizes, phash_mat, color_mat, ids, res, sizes, phashes, colorhashes

    # Pass 3: write survivors + removals, delete loser files (transient copies only).
    roots = dataset_roots(ctx.cfg)
    keep_files = ctx.cfg.dedup.keep_removed_files
    protected = protected_datasets(ctx.cfg)
    survivors = ParquetShardWriter(out_dir / "survivors.parquet", FINGERPRINT_SCHEMA)
    removed = ParquetShardWriter(out_dir / "removed.parquet", REMOVED_SCHEMA)
    stats = {"kept": 0, "duplicate_md5": 0, "duplicate_phash": 0}
    with survivors, removed:
        for batch in iter_parquet_batches(fp_files):
            for row in batch.to_pylist():
                if best[row["md5"]][2] != -row["image_id"]:
                    reason = "duplicate_md5"
                    # the md5 winner may itself have lost the phash round
                    kept_id = -best[row["md5"]][2]
                    kept_id = near_losers.get(kept_id, kept_id)
                elif row["image_id"] in near_losers:
                    reason = "duplicate_phash"
                    kept_id = near_losers[row["image_id"]]
                else:
                    survivors.append(row)
                    stats["kept"] += 1
                    continue
                stats[reason] += 1
                removed.append(
                    {
                        "image_id": row["image_id"],
                        "path": row["path"],
                        "dataset": row["dataset"],
                        "reason": reason,
                        "kept_image_id": kept_id,
                    }
                )
                if not keep_files and row["dataset"] not in protected:
                    ctx.remove_file(resolve_abs_from_roots(roots, row["path"]))

    log.info(
        "dedup done: %s%s%s",
        stats,
        " (dry-run)" if ctx.dry_run else "",
        " (removed files kept on disk)" if keep_files else "",
    )


def _hamming_rows(matrix: np.ndarray, row: np.ndarray) -> np.ndarray:
    return _POPCOUNT[matrix ^ row].sum(axis=1)


def _candidate_pairs(phash_mat: np.ndarray, max_hamming: int) -> list[tuple[int, int]]:
    try:
        return _candidate_pairs_usearch(phash_mat, max_hamming)
    except ImportError:
        if len(phash_mat) > BRUTE_FORCE_MAX:
            raise RuntimeError(
                "usearch is required for near-duplicate search at this scale; "
                "pip install usearch"
            ) from None
        return _candidate_pairs_brute(phash_mat, max_hamming)


def _candidate_pairs_usearch(phash_mat: np.ndarray, max_hamming: int) -> list[tuple[int, int]]:
    from usearch.index import Index, MetricKind, ScalarKind

    bits = phash_mat.shape[1] * 8
    index = Index(ndim=bits, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    keys = np.arange(len(phash_mat), dtype=np.uint64)
    index.add(keys, phash_mat)
    matches = index.search(phash_mat, NEIGHBORS)
    # usearch returns Matches for a single query row, BatchMatches otherwise
    entries = [matches] if len(phash_mat) == 1 else [matches[i] for i in range(len(phash_mat))]
    pairs: list[tuple[int, int]] = []
    for i, entry in enumerate(entries):
        for key, dist in zip(entry.keys, entry.distances):
            j = int(key)
            if j > i and dist <= max_hamming:
                pairs.append((i, j))
    return pairs


def _candidate_pairs_brute(phash_mat: np.ndarray, max_hamming: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(len(phash_mat) - 1):
        dists = _hamming_rows(phash_mat[i + 1 :], phash_mat[i])
        for offset in np.nonzero(dists <= max_hamming)[0]:
            pairs.append((i, i + 1 + int(offset)))
    return pairs


def _near_duplicate_losers(
    ids: np.ndarray,
    res: np.ndarray,
    sizes: np.ndarray,
    phash_mat: np.ndarray,
    color_mat: np.ndarray,
    phash_max_hamming: int,
    colorhash_max_hamming: int,
) -> dict[int, int]:
    """Map loser image_id -> winning (kept) image_id per near-duplicate group."""
    n = len(ids)
    parent = np.arange(n, dtype=np.int64)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for i, j in _candidate_pairs(phash_mat, phash_max_hamming):
        color_dist = int(_POPCOUNT[color_mat[i] ^ color_mat[j]].sum())
        if color_dist <= colorhash_max_hamming:
            parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    losers: dict[int, int] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        winner = max(members, key=lambda m: (res[m], sizes[m], -int(ids[m])))
        for m in members:
            if m != winner:
                losers[int(ids[m])] = int(ids[winner])
    return losers
