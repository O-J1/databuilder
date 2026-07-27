from __future__ import annotations

import io
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa

from ..state import RunContext
from ..utils import ParquetShardWriter, image_id, owns
from ..wds import ImageRef, LOCATOR_FIELDS
from .common import (
    archived_row,
    dataset_roots,
    iter_dataset_records,
    protected_datasets,
    pipeline_datasets,
    uses_folder_labels,
)

log = logging.getLogger("databuilder.headerscan")

KEPT_SCHEMA = pa.schema(
    [
        ("image_id", pa.uint64()),
        ("path", pa.string()),
        ("dataset", pa.string()),
        ("label", pa.string()),
        ("generator", pa.string()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("filesize", pa.int64()),
        *LOCATOR_FIELDS,
    ]
)
REMOVED_SCHEMA = pa.schema(
    [("path", pa.string()), ("dataset", pa.string()), ("reason", pa.string())]
)
BATCH = 4096
# Generous ceiling for legitimate ultra-high-res photos (~1.5 GB decoded RGB).
# PIL's default warns at ~89M pixels and hard-fails at ~179M, which real
# datasets exceed; keep a finite bound as decompression-bomb protection.
MAX_IMAGE_PIXELS = 512_000_000


def _init_worker() -> None:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        pillow_heif.register_avif_opener()
    except Exception:  # noqa: BLE001 - plugins are optional
        pass
    try:
        import pillow_jxl  # noqa: F401
    except Exception:  # noqa: BLE001
        pass


def _inspect(item: tuple[dict, dict[str, str]]) -> tuple[int, int, int, str]:
    """Header-only probe. Returns (width, height, filesize, error)."""
    from PIL import Image

    row, roots = item
    try:
        ref = ImageRef.from_row(row)
        if ref.shard:
            source = io.BytesIO(ref.read_bytes(roots))
        else:
            source = Path(roots[ref.dataset]) / ref.path.partition("/")[2]
        with Image.open(source) as img:
            width, height = img.size
        return width, height, ref.size, ""
    except Exception as exc:  # noqa: BLE001 - any failure means unopenable
        return 0, 0, 0, type(exc).__name__ or "broken"


def run(ctx: RunContext) -> None:
    filters = ctx.cfg.filters
    tall, wide = filters.tall_ratio, filters.wide_ratio
    datasets = pipeline_datasets(ctx.cfg)
    roots = dataset_roots(ctx.cfg)
    protected = protected_datasets(ctx.cfg)
    if protected:
        log.info("in-place source datasets protected from deletion: %s", sorted(protected))
    out_dir = ctx.artifact_dir("headerscan")
    kept = ParquetShardWriter(out_dir / f"rank_{ctx.rank:05d}.kept.parquet", KEPT_SCHEMA)
    removed = ParquetShardWriter(out_dir / f"rank_{ctx.rank:05d}.removed.parquet", REMOVED_SCHEMA)
    stats = {"kept": 0, "broken": 0, "too_small": 0, "too_tall": 0, "too_wide": 0}

    with kept, removed, ProcessPoolExecutor(ctx.workers, initializer=_init_worker) as pool:
        batch: list[dict] = []

        def flush() -> None:
            results = pool.map(_inspect, [(row, roots) for row in batch], chunksize=32)
            for row, (width, height, filesize, error) in zip(batch, results):
                rel = row["path"]
                ds_name = row["dataset"]
                reason = ""
                if error:
                    reason = "broken_header"
                elif max(width, height) < filters.min_longest_side:
                    reason = "too_small"
                else:
                    ratio = width / height
                    if ratio < tall:
                        reason = "too_tall"
                    elif ratio > wide:
                        reason = "too_wide"
                if reason:
                    stats[reason if reason != "broken_header" else "broken"] += 1
                    removed.append({"path": rel, "dataset": ds_name, "reason": reason})
                    if not archived_row(row) and ds_name not in protected:
                        ctx.remove_file(Path(roots[ds_name]) / rel.partition("/")[2])
                    continue
                ds = next(d for d in datasets if d.name == ds_name)
                label, generator = row["label"], row["generator"]
                if label == "unknown" and uses_folder_labels(ds):
                    raise RuntimeError(
                        f"dataset {ds.name!r}: cannot infer a label from path {rel!r}. "
                        "Move the file under a recognized label folder or extend "
                        "[datasets.label_map]."
                    )
                kept.append(
                    {
                        "image_id": image_id(rel),
                        "path": rel,
                        "dataset": ds_name,
                        "label": label,
                        "generator": generator,
                        "width": width,
                        "height": height,
                        "filesize": filesize,
                        "shard": row.get("shard") or "",
                        "member": row.get("member") or "",
                        "offset": int(row.get("offset") or 0),
                        "size": int(row.get("size") or filesize),
                    }
                )
                stats["kept"] += 1
            batch.clear()

        for ds in datasets:
            for row in iter_dataset_records(ctx.cfg, ds):
                if not owns(row["path"], ctx.rank, ctx.world_size):
                    continue
                batch.append(row)
                if len(batch) >= BATCH:
                    flush()
        flush()

    log.info("[rank %d] headerscan done: %s%s", ctx.rank, stats, " (dry-run)" if ctx.dry_run else "")
