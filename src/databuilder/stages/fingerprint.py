from __future__ import annotations

import io
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import xxhash

from ..state import RunContext
from ..utils import ParquetShardWriter, iter_parquet_batches
from ..wds import read_image_bytes
from .common import archived_row, dataset_roots, protected_datasets, resolve_abs_from_roots
from .headerscan import KEPT_SCHEMA, REMOVED_SCHEMA, _init_worker

log = logging.getLogger("databuilder.fingerprint")

FINGERPRINT_SCHEMA = pa.schema(
    list(KEPT_SCHEMA)
    + [
        ("file_hash", pa.uint64()),
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


def _fingerprint(item: tuple[dict, dict[str, str]]) -> dict:
    """Full-decode pass: xxh3 file hash + 12x12 phash + colorhash + Laplacian variance."""
    import cv2
    import imagehash
    from PIL import Image

    try:
        row, roots = item
        data = read_image_bytes(row, roots)
        file_hash = xxhash.xxh3_64_intdigest(data)
        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            if max(img.size) > LAPLACIAN_MAX_SIDE:
                img.thumbnail((LAPLACIAN_MAX_SIDE, LAPLACIAN_MAX_SIDE))
            gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
            laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            phash = _pack_hash(imagehash.phash(img, hash_size=_PHASH_SIZE))
            colorhash = _pack_hash(imagehash.colorhash(img))
        return {
            "file_hash": file_hash,
            "phash": phash,
            "colorhash": colorhash,
            "laplacian": laplacian,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - any failure means undecodable
        return {"error": type(exc).__name__ or "broken"}


def run(ctx: RunContext) -> None:
    if ctx.cfg.daft.enabled:
        _run_daft(ctx)
    else:
        _run_legacy(ctx)


def _run_legacy(ctx: RunContext) -> None:
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
            items = [(row, roots) for row in pending]
            for row, result in zip(pending, pool.map(_fingerprint, items, chunksize=16)):
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
                    if not archived_row(row) and row["dataset"] not in protected:
                        ctx.remove_file(resolve_abs_from_roots(roots, row["path"]))
                    continue
                kept.append(
                    {
                        **row,
                        "file_hash": result["file_hash"],
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


def _run_daft(ctx: RunContext) -> None:
    """Daft execution path: xxh3 + phash/colorhash in Rust, Laplacian as a UDF.

    With the native runner each rank processes its own headerscan shard; with
    the ray runner rank 0 submits every shard to the Ray cluster. Heavy image
    columns are dropped before results stream back to this process, which
    writes the same artifacts as the legacy path.
    """
    from . import daft_exec

    daft = daft_exec.init_runner(ctx.cfg)
    from daft import col, lit
    from daft.functions import image_hash, when
    from daft.functions import hash as daft_hash

    filters = ctx.cfg.filters
    roots = dataset_roots(ctx.cfg)
    protected = protected_datasets(ctx.cfg)
    hs_dir = ctx.artifact_dir("headerscan")
    if ctx.cfg.daft.runner == "ray":
        in_paths = sorted(str(p) for p in hs_dir.glob("rank_*.kept.parquet"))
    else:
        in_paths = [str(hs_dir / f"rank_{ctx.rank:05d}.kept.parquet")]
    out_dir = ctx.artifact_dir("fingerprint")

    laplacian_var = daft_exec.make_laplacian_udf(daft, LAPLACIAN_MAX_SIDE)

    df = daft.read_parquet(in_paths)
    df = daft_exec.with_downloaded_image(daft, df, roots)
    df = df.with_column("file_hash", daft_hash(col("data"), hash_function="xxhash3_64"))
    df = df.with_column(
        "phash", image_hash(col("image"), method="phash", hash_size=ctx.cfg.dedup.phash_size)
    )
    df = df.with_column("colorhash", image_hash(col("image"), method="colorhash", binbits=3))
    df = df.with_column("laplacian", laplacian_var(col("image")))
    df = df.with_column(
        "reason",
        when(col("image").is_null(), lit("broken_decode"))
        .when(col("laplacian") < filters.laplacian_min, lit("laplacian_low"))
        .when(col("laplacian") > filters.laplacian_max, lit("laplacian_high"))
        .otherwise(lit("")),
    )

    kept_cols = [field.name for field in FINGERPRINT_SCHEMA]
    result = df.select(*kept_cols, "reason")  # drops data/image before collection

    kept = ParquetShardWriter(out_dir / f"rank_{ctx.rank:05d}.parquet", FINGERPRINT_SCHEMA)
    removed = ParquetShardWriter(out_dir / f"rank_{ctx.rank:05d}.removed.parquet", REMOVED_SCHEMA)
    stats = {"kept": 0, "broken": 0, "laplacian_low": 0, "laplacian_high": 0}
    with kept, removed:
        for row in result.iter_rows():
            reason = row.pop("reason")
            if not reason:
                kept.append(row)
                stats["kept"] += 1
                continue
            stats["broken" if reason == "broken_decode" else reason] += 1
            removed.append({"path": row["path"], "dataset": row["dataset"], "reason": reason})
            if not archived_row(row) and row["dataset"] not in protected:
                ctx.remove_file(resolve_abs_from_roots(roots, row["path"]))

    log.info(
        "[rank %d] fingerprint (daft/%s) done: %s%s",
        ctx.rank,
        ctx.cfg.daft.runner,
        stats,
        " (dry-run)" if ctx.dry_run else "",
    )
