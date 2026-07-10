from __future__ import annotations

import hashlib
import io
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa

from ..state import RunContext
from ..utils import ParquetShardWriter, iter_parquet_batches
from .common import dataset_roots, protected_datasets, resolve_abs_from_roots
from .headerscan import KEPT_SCHEMA, REMOVED_SCHEMA, _init_worker

log = logging.getLogger("databuilder.fingerprint")

FINGERPRINT_SCHEMA = pa.schema(
    list(KEPT_SCHEMA)
    + [
        ("md5", pa.binary(16)),
        ("phash", pa.binary()),
        ("colorhash", pa.binary()),
        ("laplacian", pa.float64()),
    ]
)
BATCH = 2048
LAPLACIAN_MAX_SIDE = 1024

_PHASH_SIZE = 12


def _init_fp_worker(phash_size: int) -> None:
    global _PHASH_SIZE
    _PHASH_SIZE = phash_size
    _init_worker()


def _pack_hash(hash_obj) -> bytes:
    bits = np.asarray(hash_obj.hash, dtype=bool).reshape(-1)
    return np.packbits(bits).tobytes()


def _fingerprint(abs_path: str) -> dict:
    """Full-decode pass: md5 + 12x12 phash + colorhash + Laplacian variance."""
    import cv2
    import imagehash
    from PIL import Image

    try:
        with open(abs_path, "rb") as handle:
            data = handle.read()
        md5 = hashlib.md5(data).digest()
        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            if max(img.size) > LAPLACIAN_MAX_SIDE:
                img.thumbnail((LAPLACIAN_MAX_SIDE, LAPLACIAN_MAX_SIDE))
            gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
            laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            phash = _pack_hash(imagehash.phash(img, hash_size=_PHASH_SIZE))
            colorhash = _pack_hash(imagehash.colorhash(img))
        return {
            "md5": md5,
            "phash": phash,
            "colorhash": colorhash,
            "laplacian": laplacian,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - any failure means undecodable
        return {"error": type(exc).__name__ or "broken"}


def run(ctx: RunContext) -> None:
    filters = ctx.cfg.filters
    roots = dataset_roots(ctx.cfg)
    protected = protected_datasets(ctx.cfg)
    in_path = ctx.artifact_dir("headerscan") / f"rank_{ctx.rank:05d}.kept.parquet"
    out_dir = ctx.artifact_dir("fingerprint")
    kept = ParquetShardWriter(out_dir / f"rank_{ctx.rank:05d}.parquet", FINGERPRINT_SCHEMA)
    removed = ParquetShardWriter(out_dir / f"rank_{ctx.rank:05d}.removed.parquet", REMOVED_SCHEMA)
    stats = {"kept": 0, "broken": 0, "laplacian_low": 0, "laplacian_high": 0}

    with (
        kept,
        removed,
        ProcessPoolExecutor(
            ctx.workers,
            initializer=_init_fp_worker,
            initargs=(ctx.cfg.dedup.phash_size,),
        ) as pool,
    ):
        pending: list[dict] = []

        def flush() -> None:
            paths = [str(resolve_abs_from_roots(roots, row["path"])) for row in pending]
            for row, result in zip(pending, pool.map(_fingerprint, paths, chunksize=16)):
                reason = ""
                if result["error"]:
                    reason = "broken_decode"
                elif result["laplacian"] < filters.laplacian_min:
                    reason = "laplacian_low"
                elif result["laplacian"] > filters.laplacian_max:
                    reason = "laplacian_high"
                if reason:
                    key = "broken" if reason == "broken_decode" else reason
                    stats[key] += 1
                    removed.append(
                        {"path": row["path"], "dataset": row["dataset"], "reason": reason}
                    )
                    if row["dataset"] not in protected:
                        ctx.remove_file(resolve_abs_from_roots(roots, row["path"]))
                    continue
                kept.append(
                    {
                        **row,
                        "md5": result["md5"],
                        "phash": result["phash"],
                        "colorhash": result["colorhash"],
                        "laplacian": result["laplacian"],
                    }
                )
                stats["kept"] += 1
            pending.clear()

        for batch in iter_parquet_batches(in_path, batch_rows=BATCH):
            pending.extend(batch.to_pylist())
            flush()

    log.info("[rank %d] fingerprint done: %s%s", ctx.rank, stats, " (dry-run)" if ctx.dry_run else "")
