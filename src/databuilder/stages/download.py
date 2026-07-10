from __future__ import annotations

import json
import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import ConfigError, DatasetConfig
from ..state import RunContext
from ..utils import IMAGE_SUFFIXES, safe_name, sniff_extension
from .common import MATERIALIZED_MARKER, iter_dataset_images

log = logging.getLogger("databuilder.download")

LABEL_COLUMN_CANDIDATES = ("label", "class", "category", "target", "cls")
GENERATOR_COLUMN_CANDIDATES = ("generator", "model", "model_name", "source_model", "source")
SPLIT_COLUMN_CANDIDATES = ("split", "subset", "partition")


@dataclass(frozen=True)
class ColumnMap:
    """Resolved (explicit or automatched) column roles for a parquet dataset."""

    image: str
    image_kind: str  # "struct" (HF Image feature) or "binary"
    label: str | None
    generator: str | None
    split: str | None


def run(ctx: RunContext) -> None:
    """Download/materialize each dataset. Datasets are round-robin sharded."""
    datasets = sorted(ctx.cfg.datasets, key=lambda d: d.name)
    for index, ds in enumerate(datasets):
        if index % ctx.world_size != ctx.rank:
            continue
        if ds.in_place:
            count = sum(1 for _ in iter_dataset_images(Path(ds.path), ds))
            if count == 0:
                raise ConfigError(f"dataset {ds.name!r}: no images found under {ds.path}")
            log.info("dataset %r: %d images in place at %s", ds.name, count, ds.path)
            continue
        target = ctx.data_dir / ds.name
        done_marker = target / MATERIALIZED_MARKER
        if done_marker.exists():
            log.info("dataset %r already materialized, skipping", ds.name)
            continue
        if ctx.dry_run:
            log.info("dry-run: would materialize %r (%s)", ds.name, ds.repo_id or ds.path)
            continue
        stats = _materialize(ctx, ds, target)
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        log.info("dataset %r materialized: %s", ds.name, stats)


def _materialize(ctx: RunContext, ds: DatasetConfig, target: Path) -> dict:
    if ds.is_local:
        source_dir = Path(ds.path)
    else:
        from huggingface_hub import snapshot_download

        source_dir = Path(
            snapshot_download(
                repo_id=ds.repo_id,
                repo_type="dataset",
                revision=ds.revision or None,
                local_dir=ctx.work_dir / "hf" / ds.name,
                allow_patterns=list(ds.allow_patterns) or None,
            )
        )
    fmt = ds.format if ds.format != "auto" else _detect_format(source_dir)
    log.info("dataset %r: format=%s source=%s", ds.name, fmt, source_dir)
    target.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        return _materialize_parquet(ds, source_dir, target)
    if fmt == "zip":
        return _materialize_zip(ds, source_dir, target)
    return _materialize_imagefolder(ds, source_dir, target)


def _detect_format(snapshot_dir: Path) -> str:
    if any(snapshot_dir.rglob("*.parquet")):
        return "parquet"
    if any(snapshot_dir.rglob("*.zip")):
        return "zip"
    if any(p.suffix.lower() in IMAGE_SUFFIXES for p in snapshot_dir.rglob("*") if p.is_file()):
        return "imagefolder"
    raise RuntimeError(
        f"Could not auto-detect layout under {snapshot_dir}. "
        "Set format and/or image_dir for this dataset in the config."
    )


def _find_image_column(schema: pa.Schema) -> tuple[str | None, str]:
    preferred = ("image", "img", "picture", "photo")
    struct_cols: list[str] = []
    binary_cols: list[str] = []
    for field in schema:
        if pa.types.is_struct(field.type):
            child_names = {child.name for child in field.type}
            if "bytes" in child_names:
                struct_cols.append(field.name)
        elif pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type):
            binary_cols.append(field.name)
    for name in preferred:
        if name in struct_cols:
            return name, "struct"
        if name in binary_cols:
            return name, "binary"
    if struct_cols:
        return struct_cols[0], "struct"
    if binary_cols:
        return binary_cols[0], "binary"
    return None, ""


def _image_kind(schema: pa.Schema, name: str) -> str:
    field_type = schema.field(name).type
    if pa.types.is_struct(field_type):
        return "struct"
    return "binary"


def _automatch(schema_names: set[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in schema_names:
            return name
    return None


def resolve_columns(ds: DatasetConfig, schema: pa.Schema) -> ColumnMap:
    """Resolve column roles: explicit config wins, else automatch, else hard error."""
    names = set(schema.names)

    def require(role: str, column: str | None) -> str:
        if column is None:
            raise ConfigError(
                f"dataset {ds.name!r}: could not automatch the {role} column. "
                f"Available columns: {sorted(names)}. "
                f'Set columns.{role} = "<name>" explicitly in the config.'
            )
        if column not in names:
            raise ConfigError(
                f"dataset {ds.name!r}: columns.{role} = {column!r} not in schema "
                f"{sorted(names)}"
            )
        return column

    image = ds.image_column or _find_image_column(schema)[0]
    image = require("image", image)

    if ds.label == "folder":
        raise ConfigError(
            f"dataset {ds.name!r}: label = 'folder' is not valid for parquet "
            "datasets; use a label column or a static 'real'/'fake' label."
        )
    label: str | None = None
    if ds.label_column:
        label = require("label", ds.label_column)
    elif ds.label == "auto":
        label = require("label", _automatch(names, LABEL_COLUMN_CANDIDATES))

    generator: str | None = None
    if ds.generator_column:
        generator = require("generator", ds.generator_column)
    elif not ds.generator:
        generator = _automatch(names, GENERATOR_COLUMN_CANDIDATES)  # optional

    split: str | None = None
    if ds.source_split:
        split = ds.split_column or _automatch(names, SPLIT_COLUMN_CANDIDATES)
        if ds.split_column:
            split = require("split", split)

    return ColumnMap(
        image=image,
        image_kind=_image_kind(schema, image),
        label=label,
        generator=generator,
        split=split,
    )


def _materialize_parquet(ds: DatasetConfig, source_dir: Path, target: Path) -> dict:
    files = sorted(source_dir.rglob("*.parquet"))
    if not files:
        raise ConfigError(f"dataset {ds.name!r}: no parquet files under {source_dir}")
    if ds.source_split:
        split_files = [f for f in files if ds.source_split in f.name]
        files = split_files or files
    cmap = resolve_columns(ds, pq.ParquetFile(files[0]).schema_arrow)
    log.info("dataset %r: resolved columns %s", ds.name, cmap)
    written = skipped = 0
    for file_idx, parquet_path in enumerate(files):
        reader = pq.ParquetFile(parquet_path)
        schema_names = set(reader.schema_arrow.names)
        columns = [c for c in (cmap.image, cmap.label, cmap.generator, cmap.split) if c]
        missing = [c for c in columns if c not in schema_names]
        if missing:
            raise ConfigError(
                f"dataset {ds.name!r}: {parquet_path.name} is missing columns {missing}"
            )
        for batch in reader.iter_batches(batch_size=256, columns=columns):
            rows = batch.to_pylist()
            for row in rows:
                if (
                    cmap.split
                    and isinstance(row.get(cmap.split), str)
                    and row[cmap.split] != ds.source_split
                ):
                    continue
                data = _encoded_bytes(row[cmap.image], cmap.image_kind, parquet_path)
                if data is None:
                    skipped += 1
                    continue
                out_dir = target
                if cmap.label:
                    out_dir = out_dir / safe_name(row.get(cmap.label)).lower()
                if cmap.generator:
                    out_dir = out_dir / safe_name(row.get(cmap.generator))
                out_dir.mkdir(parents=True, exist_ok=True)
                ext = sniff_extension(data)
                name = f"{ds.name}_{file_idx:05d}_{written:09d}{ext}"
                (out_dir / name).write_bytes(data)
                written += 1
        if not ds.keep_archives and not ds.is_local:
            parquet_path.unlink(missing_ok=True)
    return {
        "format": "parquet",
        "written": written,
        "skipped": skipped,
        "layout": {
            "label_dir": cmap.label is not None,
            "generator_dir": cmap.generator is not None,
        },
    }


def _encoded_bytes(value: object, kind: str, parquet_path: Path) -> bytes | None:
    if kind == "binary":
        return value if isinstance(value, bytes) else None
    if not isinstance(value, dict):
        return None
    data = value.get("bytes")
    if isinstance(data, bytes):
        return data
    source = value.get("path")
    if isinstance(source, str):
        candidate = parquet_path.parent / source
        if candidate.exists():
            return candidate.read_bytes()
    return None


def _materialize_zip(ds: DatasetConfig, source_dir: Path, target: Path) -> dict:
    extracted = 0
    for zip_path in sorted(source_dir.rglob("*.zip")):
        out_dir = target / zip_path.stem
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if Path(member.filename).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                # Guard against zip-slip: refuse entries escaping out_dir.
                dest = (out_dir / member.filename).resolve()
                if not dest.is_relative_to(out_dir.resolve()):
                    log.warning("skipping unsafe zip entry %r", member.filename)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1
        if not ds.keep_archives and not ds.is_local:
            zip_path.unlink(missing_ok=True)
    return {"format": "zip", "written": extracted}


def _materialize_imagefolder(ds: DatasetConfig, source_dir: Path, target: Path) -> dict:
    source = source_dir / ds.image_dir if ds.image_dir else source_dir
    if not source.exists():
        raise RuntimeError(f"image_dir {source} does not exist for dataset {ds.name!r}")
    moved = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        dest = target / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))
        moved += 1
    if moved == 0:
        raise RuntimeError(
            f"No images found for dataset {ds.name!r} under {source}. "
            "Specify image_dir in the config."
        )
    return {"format": "imagefolder", "written": moved}
