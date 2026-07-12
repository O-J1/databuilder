from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa

from ..state import RunContext
from ..utils import ParquetShardWriter, image_id, normalize_relpath, owns
from .common import (
    dataset_root,
    iter_dataset_images,
    load_layout,
    protected_datasets,
    resolve_meta,
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


def _inspect(abs_path: str) -> tuple[int, int, int, str]:
    """Header-only probe. Returns (width, height, filesize, error)."""
    from PIL import Image

    try:
        with Image.open(abs_path) as img:
            width, height = img.size
        return width, height, os.path.getsize(abs_path), ""
    except Exception as exc:  # noqa: BLE001 - any failure means unopenable
        return 0, 0, 0, type(exc).__name__ or "broken"


def run(ctx: RunContext) -> None:
    filters = ctx.cfg.filters
    tall, wide = filters.tall_ratio, filters.wide_ratio
    layouts = {ds.name: load_layout(ctx.cfg, ds) for ds in ctx.cfg.datasets}
    protected = protected_datasets(ctx.cfg)
    if protected:
        log.info("in-place source datasets protected from deletion: %s", sorted(protected))
    out_dir = ctx.artifact_dir("headerscan")
    kept = ParquetShardWriter(out_dir / f"rank_{ctx.rank:05d}.kept.parquet", KEPT_SCHEMA)
    removed = ParquetShardWriter(out_dir / f"rank_{ctx.rank:05d}.removed.parquet", REMOVED_SCHEMA)
    stats = {"kept": 0, "broken": 0, "too_small": 0, "too_tall": 0, "too_wide": 0}

    with kept, removed, ProcessPoolExecutor(ctx.workers, initializer=_init_worker) as pool:
        batch: list[tuple[str, Path, str]] = []

        def flush() -> None:
            results = pool.map(_inspect, [str(p) for _, p, _ in batch], chunksize=32)
            for (rel, abs_path, ds_name), (width, height, filesize, error) in zip(
                batch, results
            ):
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
                    if ds_name not in protected:
                        ctx.remove_file(abs_path)
                    continue
                ds = next(d for d in ctx.cfg.datasets if d.name == ds_name)
                rel_parts = Path(rel).parts[1:]  # strip dataset dir
                label, generator = resolve_meta(ds, rel_parts, layouts[ds_name])
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
                    }
                )
                stats["kept"] += 1
            batch.clear()

        for ds in ctx.cfg.datasets:
            root = dataset_root(ctx.cfg, ds)
            for abs_path in iter_dataset_images(root, ds):
                rel = f"{ds.name}/{normalize_relpath(abs_path.relative_to(root))}"
                if not owns(rel, ctx.rank, ctx.world_size):
                    continue
                batch.append((rel, abs_path, ds.name))
                if len(batch) >= BATCH:
                    flush()
        flush()

    log.info("[rank %d] headerscan done: %s%s", ctx.rank, stats, " (dry-run)" if ctx.dry_run else "")
