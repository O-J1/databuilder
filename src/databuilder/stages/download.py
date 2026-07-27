from __future__ import annotations

import bisect
import io
import json
import logging
import os
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import ConfigError, DatasetConfig, ImageColumnConfig
from ..state import RunContext
from ..utils import IMAGE_SUFFIXES, normalize_relpath, safe_name, sniff_extension
from ..wds import (
    DatasetShardWriter,
    archive_raw_snapshot,
    atomic_json,
    copy_stream,
    is_webdataset,
    load_marker,
    remove_empty_directories,
    remove_tree_contents,
)
from .common import (
    MATERIALIZED_MARKER,
    config_layout,
    iter_dataset_images,
    label_from_value,
    load_layout,
    resolve_meta,
)

log = logging.getLogger("databuilder.download")

LABEL_COLUMN_CANDIDATES = ("label", "label_name", "class", "category", "target", "cls")
GENERATOR_COLUMN_CANDIDATES = (
    "generator",
    "model",
    "model_name",
    "source_model",
    "source",
    "source_dataset",
)
SPLIT_COLUMN_CANDIDATES = ("split", "subset", "partition")


@dataclass(frozen=True)
class ColumnMap:
    """Resolved (explicit or automatched) roles for a single-image table."""

    image: str
    image_kind: str
    label: str | None
    generator: str | None
    split: str | None


@dataclass(frozen=True)
class TableMap:
    images: tuple[ImageColumnConfig, ...]
    label: str | None
    generator: str | None
    split: str | None


def run(ctx: RunContext) -> None:
    """Snapshot and materialize every configured dataset on rank 0 only."""
    if ctx.rank != 0:
        raise RuntimeError("download.run must only be invoked on rank 0")
    for ds in sorted(ctx.cfg.datasets, key=lambda item: item.name):
        target = ctx.data_dir / ds.name
        marker = load_marker(target)
        if is_webdataset(target) or marker.get("storage") == "raw_tar":
            log.info("dataset %r already prepared, skipping", ds.name)
            continue
        if marker.get("storage") == "webdataset" and marker.get("state") == "compacting":
            raise RuntimeError(
                f"dataset {ds.name!r} has interrupted shard compaction; run "
                "`databuilder storage compact --config <config>` before the pipeline"
            )
        loose_root = target if target.is_dir() else None
        has_loose = loose_root is not None and next(iter_dataset_images(loose_root, ds), None)
        # A v1 marker means the old loose materialization completed. Pack it
        # without consulting Hugging Face, then remove any redundant snapshot.
        if has_loose and marker and marker.get("storage_version") != 2:
            if ctx.dry_run:
                log.info("dry-run: would migrate completed loose dataset %r", ds.name)
                continue
            stats = migrate_loose_dataset(ctx, ds, target)
            _cleanup_remote_state(ctx, ds)
            log.info("dataset %r migrated to WebDataset: %s", ds.name, stats)
            continue
        if ctx.dry_run:
            log.info("dry-run: would snapshot/materialize %r (%s)", ds.name, ds.repo_id or ds.path)
            continue
        source_dir, cleanup_source = _source_directory(ctx, ds, allow_download=True)
        if ds.download_only:
            stats = archive_raw_snapshot(source_dir, target, ds.name)
            atomic_json(target / MATERIALIZED_MARKER, stats)
        else:
            stats = _materialize(ctx, ds, source_dir, target)
        if cleanup_source and not ctx.cfg.download.retain_snapshots:
            remove_tree_contents(source_dir)
        if not ctx.cfg.download.retain_xet_cache:
            remove_tree_contents(_staging_root(ctx) / ".hf_xet")
        log.info("dataset %r prepared: %s", ds.name, stats)


def _staging_root(ctx: RunContext) -> Path:
    return ctx.cfg.download.staging_dir or (ctx.data_dir / ".databuilder-staging")


def _snapshot_candidates(ctx: RunContext, ds: DatasetConfig) -> tuple[Path, ...]:
    return (
        ctx.data_dir / ".hf_snapshots" / ds.name,
        _staging_root(ctx) / ".hf_snapshots" / ds.name,
    )


def _directory_has_files(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def _source_directory(
    ctx: RunContext, ds: DatasetConfig, *, allow_download: bool
) -> tuple[Path, bool]:
    if ds.is_local:
        return Path(ds.path), False
    for candidate in _snapshot_candidates(ctx, ds):
        if _directory_has_files(candidate):
            log.info("dataset %r: using existing snapshot %s", ds.name, candidate)
            return candidate, True
    if not allow_download:
        raise FileNotFoundError(f"dataset {ds.name!r}: no local snapshot exists")

    staging = _staging_root(ctx)
    local_dir = staging / ".hf_snapshots" / ds.name
    xet_cache = staging / ".hf_xet"
    local_dir.mkdir(parents=True, exist_ok=True)
    xet_cache.mkdir(parents=True, exist_ok=True)
    # Only relocate Xet's cache. All transfer tuning, including high-performance
    # mode, remains at huggingface_hub/user environment defaults.
    os.environ["HF_XET_CACHE"] = str(xet_cache)
    from huggingface_hub import snapshot_download

    source_dir = Path(
        snapshot_download(
            repo_id=ds.repo_id,
            repo_type="dataset",
            revision=ds.revision or None,
            local_dir=local_dir,
            allow_patterns=list(ds.allow_patterns) or None,
            max_workers=ctx.cfg.download.max_workers,
        )
    )
    return source_dir, True


def _new_writer(ctx: RunContext, ds: DatasetConfig, target: Path) -> DatasetShardWriter:
    return DatasetShardWriter(
        target,
        ds.name,
        target_shard_bytes=ctx.cfg.storage.target_shard_bytes,
        max_samples_per_shard=ctx.cfg.storage.max_samples_per_shard,
    )


def _materialize(
    ctx: RunContext, ds: DatasetConfig, source_dir: Path, target: Path
) -> dict:
    writer = _new_writer(ctx, ds, target)
    # Recover any loose files written by an interrupted pre-WDS materializer.
    _pack_loose_into(ctx, ds, target, writer)

    fmt = ds.format if ds.format != "auto" else _detect_format(source_dir)
    log.info("dataset %r: format=%s source=%s", ds.name, fmt, source_dir)
    target.mkdir(parents=True, exist_ok=True)
    try:
        if fmt == "parquet":
            stats = _materialize_parquet(ctx, ds, source_dir, writer)
        elif fmt == "arrow":
            stats = _materialize_arrow(ctx, ds, source_dir, writer)
        elif fmt == "jsonl":
            stats = _materialize_jsonl(ctx, ds, source_dir, writer)
        elif fmt == "zip":
            stats = _materialize_zip(ds, source_dir, writer)
        elif fmt in {"tar", "webdataset", "multipart_tar"}:
            stats = _materialize_tar(ds, source_dir, writer, fmt)
        elif fmt == "multipart_zip":
            stats = _materialize_multipart_zip(ds, source_dir, writer)
        elif fmt == "imagefolder":
            stats = _materialize_imagefolder(ds, source_dir, writer)
        else:
            raise ConfigError(
                f"dataset {ds.name!r}: unsupported materialization format {fmt!r}"
            )
        result = writer.finalize(stats)
        remove_empty_directories(target)
        return result
    except Exception:
        writer.close()
        raise


def _pack_loose_into(
    ctx: RunContext,
    ds: DatasetConfig,
    root: Path,
    writer: DatasetShardWriter,
) -> int:
    """Pack legacy materialized images, deleting each only after shard commit."""
    if not root.is_dir():
        return 0
    layout = load_layout(ctx.cfg, ds)
    packed = 0
    for path in iter_dataset_images(root, ds):
        relative = normalize_relpath(path.relative_to(root))
        logical = f"{ds.name}/{relative}"
        label, generator = resolve_meta(ds, tuple(Path(relative).parts), layout)
        if writer.add(
            path.read_bytes(),
            logical,
            label,
            generator,
            metadata={"migrated_from": relative},
            delete_source=path,
        ):
            packed += 1
    return packed


def migrate_loose_dataset(ctx: RunContext, ds: DatasetConfig, target: Path) -> dict:
    writer = _new_writer(ctx, ds, target)
    try:
        packed = _pack_loose_into(ctx, ds, target, writer)
        result = writer.finalize({"format": "legacy_loose", "migrated": packed})
        remove_empty_directories(target)
        return result
    except Exception:
        writer.close()
        raise


def _cleanup_remote_state(ctx: RunContext, ds: DatasetConfig) -> None:
    if ds.is_local:
        return
    if not ctx.cfg.download.retain_snapshots:
        for path in _snapshot_candidates(ctx, ds):
            remove_tree_contents(path)
    if not ctx.cfg.download.retain_xet_cache:
        remove_tree_contents(_staging_root(ctx) / ".hf_xet")
        remove_tree_contents(ctx.data_dir / ".hf_xet")


def inventory(ctx: RunContext) -> list[dict]:
    """Classify local dataset state without making any network calls."""
    rows: list[dict] = []
    for ds in sorted(ctx.cfg.datasets, key=lambda item: item.name):
        target = ctx.data_dir / ds.name
        marker = load_marker(target)
        loose = target.is_dir() and next(iter_dataset_images(target, ds), None) is not None
        snapshots = [str(path) for path in _snapshot_candidates(ctx, ds) if _directory_has_files(path)]
        if is_webdataset(target):
            status = "webdataset_complete"
        elif marker.get("storage") == "webdataset" and marker.get("state") == "compacting":
            status = "webdataset_compacting"
        elif marker.get("storage") == "raw_tar":
            status = "raw_tar_complete"
        elif loose and marker:
            status = "loose_complete"
        elif loose and snapshots:
            status = "loose_partial_with_snapshot"
        elif loose:
            status = "loose_only"
        elif snapshots:
            status = "snapshot_only"
        elif ds.is_local and _directory_has_files(Path(ds.path)):
            status = "local_source"
        elif ds.download_only and target.is_dir() and marker:
            status = "raw_snapshot_legacy"
        else:
            status = "missing"
        rows.append(
            {
                "dataset": ds.name,
                "status": status,
                "target": str(target),
                "snapshots": snapshots,
            }
        )
    return rows


def migrate_existing(ctx: RunContext) -> list[dict]:
    """Convert only local state. This function never calls Hugging Face."""
    results: list[dict] = []
    for ds in sorted(ctx.cfg.datasets, key=lambda item: item.name):
        target = ctx.data_dir / ds.name
        marker = load_marker(target)
        if is_webdataset(target) or marker.get("storage") == "raw_tar":
            results.append({"dataset": ds.name, "status": "already_complete"})
            continue
        if marker.get("storage") == "webdataset" and marker.get("state") == "compacting":
            results.append({"dataset": ds.name, "status": "compaction_in_progress"})
            continue
        loose = target.is_dir() and next(iter_dataset_images(target, ds), None) is not None
        if ds.download_only and target.is_dir() and marker and not loose:
            stats = archive_raw_snapshot(target, target, ds.name)
            _remove_legacy_raw_files(target)
            atomic_json(target / MATERIALIZED_MARKER, stats)
            _cleanup_remote_state(ctx, ds)
            results.append({"dataset": ds.name, "status": "migrated", **stats})
            continue
        try:
            source, cleanup_source = _source_directory(ctx, ds, allow_download=False)
        except FileNotFoundError:
            source = None
            cleanup_source = False
        if loose and marker and source is None:
            stats = migrate_loose_dataset(ctx, ds, target)
        elif source is not None:
            if ds.download_only:
                stats = archive_raw_snapshot(source, target, ds.name)
                atomic_json(target / MATERIALIZED_MARKER, stats)
            else:
                stats = _materialize(ctx, ds, source, target)
            if cleanup_source and not ctx.cfg.download.retain_snapshots:
                remove_tree_contents(source)
        elif loose:
            # Preserve every locally available image. The marker records that
            # this was a loose-only recovery so inventory remains explicit.
            stats = migrate_loose_dataset(ctx, ds, target)
            stats["recovered_from"] = "loose_only"
            atomic_json(target / MATERIALIZED_MARKER, stats)
        else:
            results.append({"dataset": ds.name, "status": "needs_download"})
            continue
        _cleanup_remote_state(ctx, ds)
        results.append({"dataset": ds.name, "status": "migrated", **stats})
    return results


def _remove_legacy_raw_files(target: Path) -> None:
    """Remove old raw snapshot files but retain the committed raw tar."""
    target = target.resolve()
    raw = (target / "raw").resolve()
    for path in sorted(target.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        resolved = path.resolve()
        if resolved == raw or resolved.is_relative_to(raw) or path.name == MATERIALIZED_MARKER:
            continue
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def _detect_format(snapshot_dir: Path) -> str:
    if any(snapshot_dir.rglob("*.parquet")):
        return "parquet"
    if any(snapshot_dir.rglob("*.arrow")):
        return "arrow"
    if any(snapshot_dir.rglob("*.jsonl")):
        return "jsonl"
    if any(snapshot_dir.rglob("*.zip")):
        return "zip"
    if _tar_files(snapshot_dir):
        return "tar"
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
            if "bytes" in child_names or "path" in child_names:
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
    if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        return "auto"
    if pa.types.is_struct(field_type):
        return "struct"
    return "binary"


def _automatch(schema_names: set[str], candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in schema_names), None)


def _require(ds: DatasetConfig, names: set[str], role: str, column: str | None) -> str:
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


def resolve_columns(ds: DatasetConfig, schema: pa.Schema) -> ColumnMap:
    """Resolve single-image column roles; explicit config wins over automatch."""
    names = set(schema.names)
    image = _require(ds, names, "image", ds.image_column or _find_image_column(schema)[0])
    if ds.label == "folder":
        raise ConfigError(
            f"dataset {ds.name!r}: label = 'folder' is not valid for tabular datasets; "
            "use a label column or a static 'real'/'fake' label."
        )
    label: str | None = None
    if ds.label_column:
        label = _require(ds, names, "label", ds.label_column)
    elif ds.label == "auto":
        label = _require(ds, names, "label", _automatch(names, LABEL_COLUMN_CANDIDATES))

    generator: str | None = None
    if ds.generator_column:
        generator = _require(ds, names, "generator", ds.generator_column)
    elif not ds.generator:
        generator = _automatch(names, GENERATOR_COLUMN_CANDIDATES)

    split: str | None = None
    if ds.source_split:
        split = ds.split_column or _automatch(names, SPLIT_COLUMN_CANDIDATES)
        if ds.split_column:
            split = _require(ds, names, "split", split)
    return ColumnMap(image, _image_kind(schema, image), label, generator, split)


def _resolve_table(ds: DatasetConfig, schema: pa.Schema) -> TableMap:
    if not ds.images:
        single = resolve_columns(ds, schema)
        return TableMap(
            (
                ImageColumnConfig(
                    column=single.image,
                    kind="auto" if single.image_kind == "auto" else "embedded",
                ),
            ),
            single.label,
            single.generator,
            single.split,
        )
    names = set(schema.names)
    for spec in ds.images:
        _require(ds, names, "image", spec.column)
        if spec.generator_column:
            _require(ds, names, "generator", spec.generator_column)
    if ds.label == "folder":
        raise ConfigError(f"dataset {ds.name!r}: label = 'folder' is invalid for tables")
    label = ds.label_column
    if label:
        _require(ds, names, "label", label)
    elif ds.label == "auto":
        label = _require(ds, names, "label", _automatch(names, LABEL_COLUMN_CANDIDATES))
    generator = ds.generator_column
    if generator:
        _require(ds, names, "generator", generator)
    elif not ds.generator:
        generator = _automatch(names, GENERATOR_COLUMN_CANDIDATES)
    split = None
    if ds.source_split:
        split = ds.split_column or _automatch(names, SPLIT_COLUMN_CANDIDATES)
        if ds.split_column:
            _require(ds, names, "split", split)
    for column in (*ds.row_filter, *ds.row_exclude):
        _require(ds, names, "row filter", column)
    return TableMap(ds.images, label, generator, split)


def _is_excluded(value: object, excluded: tuple[object, ...]) -> bool:
    """Match excluded strings case-insensitively; preserve exact matching otherwise."""
    if isinstance(value, str):
        folded = value.casefold()
        return any(isinstance(item, str) and item.casefold() == folded for item in excluded)
    return value in excluded


class _RowMaterializer:
    def __init__(
        self,
        ctx: RunContext,
        ds: DatasetConfig,
        writer: DatasetShardWriter,
        source_dir: Path,
        mapping: TableMap,
    ) -> None:
        self.ctx = ctx
        self.ds = ds
        self.writer = writer
        self.source_dir = source_dir
        self.mapping = mapping
        self.written = 0
        self.added = 0
        self.skipped = 0
        self.filtered = 0

    @property
    def layout(self) -> dict[str, bool]:
        generator_dir = bool(
            self.mapping.generator
            or any(spec.generator or spec.generator_column for spec in self.mapping.images)
        )
        return {"label_dir": self.mapping.label is not None, "generator_dir": generator_dir}

    def add(self, row: dict, file_idx: int, source_path: Path) -> None:
        if any(row.get(column) != expected for column, expected in self.ds.row_filter.items()):
            self.filtered += 1
            return
        if any(
            _is_excluded(row.get(column), excluded)
            for column, excluded in self.ds.row_exclude.items()
        ):
            self.filtered += 1
            return
        if (
            self.mapping.split
            and self.ds.source_split
            and str(row.get(self.mapping.split)) != self.ds.source_split
        ):
            self.filtered += 1
            return
        for image_idx, spec in enumerate(self.mapping.images):
            data = _encoded_bytes(
                self.ctx, row.get(spec.column), spec.kind, source_path, self.source_dir
            )
            if data is None:
                self.skipped += 1
                continue
            parts: list[str] = []
            raw_label = None
            if self.mapping.label:
                raw_label = row.get(self.mapping.label)
                parts.append(safe_name(raw_label).lower())
            generator = None
            if spec.generator:
                generator = spec.generator
            elif spec.generator_column:
                generator = row.get(spec.generator_column)
            elif self.mapping.generator:
                generator = row.get(self.mapping.generator)
            if self.layout["generator_dir"]:
                # Dynamic generator columns are legitimately null for real rows.
                # Keep that provenance explicit instead of turning the legacy
                # `column:<name>` config syntax into a literal directory name.
                parts.append(safe_name(generator))
            suffix = f"_i{image_idx}" if len(self.mapping.images) > 1 else ""
            name = (
                f"{self.ds.name}_{file_idx:05d}_{self.written:09d}{suffix}"
                f"{sniff_extension(data)}"
            )
            relative = normalize_relpath(Path(*parts, name))
            logical = f"{self.ds.name}/{relative}"
            if raw_label is not None:
                label = label_from_value(self.ds, str(raw_label)) or str(raw_label).lower()
            else:
                label = self.ds.label if self.ds.label in {"real", "fake"} else "unknown"
            final_generator = (
                safe_name(generator)
                if self.layout["generator_dir"]
                else (self.ds.generator or self.ds.name)
            )
            source_split = str(row.get(self.mapping.split) or "") if self.mapping.split else ""
            source_meta = {
                "source_file": normalize_relpath(source_path.relative_to(self.source_dir)),
                "image_column": spec.column,
            }
            if self.writer.add(
                data,
                logical,
                label,
                final_generator,
                source_split=source_split,
                metadata=source_meta,
            ):
                self.added += 1
            self.written += 1

    def stats(self, fmt: str) -> dict:
        return {
            "format": fmt,
            "written_from_source": self.added,
            "skipped": self.skipped,
            "filtered": self.filtered,
            "layout": self.layout,
        }


def _selected_columns(ds: DatasetConfig, mapping: TableMap) -> list[str]:
    columns = [spec.column for spec in mapping.images]
    columns += [spec.generator_column for spec in mapping.images if spec.generator_column]
    columns += [mapping.label, mapping.generator, mapping.split]
    columns += list(ds.row_filter)
    columns += list(ds.row_exclude)
    return list(dict.fromkeys(column for column in columns if column))


def _materialize_parquet(
    ctx: RunContext, ds: DatasetConfig, source_dir: Path, shard_writer: DatasetShardWriter
) -> dict:
    files = sorted(source_dir.rglob("*.parquet"))
    if not files:
        raise ConfigError(f"dataset {ds.name!r}: no parquet files under {source_dir}")
    if ds.source_split:
        matched = [path for path in files if ds.source_split in path.name]
        files = matched or files
    mapping = _resolve_table(ds, pq.ParquetFile(files[0]).schema_arrow)
    writer = _RowMaterializer(ctx, ds, shard_writer, source_dir, mapping)
    columns = _selected_columns(ds, mapping)
    log.info("dataset %r: resolved table columns %s", ds.name, mapping)
    for file_idx, parquet_path in enumerate(files):
        reader = pq.ParquetFile(parquet_path)
        missing = sorted(set(columns) - set(reader.schema_arrow.names))
        if missing:
            raise ConfigError(f"dataset {ds.name!r}: {parquet_path.name} misses {missing}")
        for batch in reader.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                writer.add(row, file_idx, parquet_path)
    return writer.stats("parquet")


def _open_arrow(path: Path):
    source = pa.memory_map(str(path), "r")
    try:
        return source, pa.ipc.open_stream(source)
    except pa.ArrowInvalid:
        source.seek(0)
        return source, pa.ipc.open_file(source)


def _materialize_arrow(
    ctx: RunContext, ds: DatasetConfig, source_dir: Path, shard_writer: DatasetShardWriter
) -> dict:
    files = sorted(source_dir.rglob("*.arrow"))
    if not files:
        raise ConfigError(f"dataset {ds.name!r}: no Arrow files under {source_dir}")
    first_source, first_reader = _open_arrow(files[0])
    try:
        mapping = _resolve_table(ds, first_reader.schema)
    finally:
        first_source.close()
    columns = _selected_columns(ds, mapping)
    writer = _RowMaterializer(ctx, ds, shard_writer, source_dir, mapping)
    for file_idx, arrow_path in enumerate(files):
        source, reader = _open_arrow(arrow_path)
        try:
            missing = sorted(set(columns) - set(reader.schema.names))
            if missing:
                raise ConfigError(f"dataset {ds.name!r}: {arrow_path.name} misses {missing}")
            batches = reader if isinstance(reader, pa.ipc.RecordBatchStreamReader) else (
                reader.get_batch(i) for i in range(reader.num_record_batches)
            )
            for batch in batches:
                for row in batch.select(columns).to_pylist():
                    writer.add(row, file_idx, arrow_path)
        finally:
            source.close()
    return writer.stats("arrow")


def _materialize_jsonl(
    ctx: RunContext, ds: DatasetConfig, source_dir: Path, shard_writer: DatasetShardWriter
) -> dict:
    files = sorted(source_dir.rglob("*.jsonl"))
    if not files:
        raise ConfigError(f"dataset {ds.name!r}: no JSONL files under {source_dir}")
    if not ds.images:
        raise ConfigError(f"dataset {ds.name!r}: jsonl requires an images mapping")
    # JSON is schemaless, so validate against a synthetic schema made from the config.
    fields = set(_selected_columns(ds, TableMap(ds.images, ds.label_column, ds.generator_column, None)))
    schema = pa.schema((name, pa.string()) for name in sorted(fields))
    mapping = _resolve_table(ds, schema)
    writer = _RowMaterializer(ctx, ds, shard_writer, source_dir, mapping)
    for file_idx, jsonl_path in enumerate(files):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConfigError(
                        f"dataset {ds.name!r}: invalid JSON at {jsonl_path}:{line_no}"
                    ) from exc
                writer.add(row, file_idx, jsonl_path)
    return writer.stats("jsonl")


def _encoded_bytes(
    ctx: RunContext,
    value: object,
    kind: str,
    source_path: Path,
    source_dir: Path,
) -> bytes | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, dict):
        data = value.get("bytes")
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data)
        value = value.get("path")
    if not isinstance(value, str) or not value:
        return None
    if value.startswith(("http://", "https://")):
        return None
    for candidate in (source_path.parent / value, source_dir / value):
        if candidate.is_file():
            return candidate.read_bytes()
    return None


def _safe_member_name(member_name: str) -> str | None:
    normalized = member_name.replace("\\", "/").lstrip("/")
    parts = Path(normalized).parts
    if not normalized or any(part in {"", ".", ".."} for part in parts):
        return None
    return normalize_relpath(Path(*parts))


def _archive_meta(ds: DatasetConfig, relative: str) -> tuple[str, str]:
    return resolve_meta(ds, tuple(Path(relative).parts), config_layout(ds))


def _materialize_zip(
    ds: DatasetConfig, source_dir: Path, writer: DatasetShardWriter
) -> dict:
    written = 0
    archives = sorted(source_dir.rglob("*.zip"))
    if not archives:
        raise ConfigError(f"dataset {ds.name!r}: no zip files under {source_dir}")
    for zip_path in archives:
        prefix = safe_name(zip_path.stem)
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir() or Path(member.filename).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                safe_member = _safe_member_name(member.filename)
                if safe_member is None:
                    log.warning("skipping unsafe zip entry %r", member.filename)
                    continue
                relative = normalize_relpath(Path(prefix) / safe_member)
                label, generator = _archive_meta(ds, relative)
                with archive.open(member) as src:
                    data = copy_stream(src)
                if writer.add(
                    data,
                    f"{ds.name}/{relative}",
                    label,
                    generator,
                    metadata={"source_archive": zip_path.name, "source_member": member.filename},
                ):
                    written += 1
    return {"format": "zip", "written_from_source": written}


def _tar_files(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file()
        and (path.suffix.lower() == ".tar" or path.name.lower().endswith((".tar.gz", ".tgz")))
    )


def _materialize_tar(
    ds: DatasetConfig, source_dir: Path, writer: DatasetShardWriter, fmt: str
) -> dict:
    written = 0
    archives = _tar_files(source_dir)
    if not archives:
        raise ConfigError(f"dataset {ds.name!r}: no tar files under {source_dir}")

    if fmt == "multipart_tar":
        # Some repositories use `split` to cut one tar byte stream into numbered
        # .tar chunks. Present those chunks as one seekable file without writing
        # a second, enormous combined archive to disk.
        prefix = safe_name(
            archives[0].name.removesuffix(".tar.gz").removesuffix(".tgz").removesuffix(".tar")
        )
        joined = _MultipartFile(archives)
        try:
            with tarfile.open(fileobj=joined, mode="r:*") as archive:
                written += _copy_tar_images(ds, archive, writer, prefix, "multipart_tar")
        except tarfile.ReadError as exc:
            raise ConfigError(
                f"dataset {ds.name!r}: concatenated tar stream is unreadable across "
                f"{len(archives)} chunks under {source_dir}"
            ) from exc
        finally:
            joined.close()
        return {"format": fmt, "written_from_source": written, "parts": len(archives)}

    for tar_path in archives:
        stem = tar_path.name.removesuffix(".tar.gz").removesuffix(".tgz").removesuffix(".tar")
        try:
            with tarfile.open(tar_path, "r:*") as archive:
                written += _copy_tar_images(ds, archive, writer, safe_name(stem), tar_path.name)
        except tarfile.ReadError as exc:
            raise ConfigError(
                f"dataset {ds.name!r}: tar archive {tar_path} is unreadable. "
                "If these are numbered chunks of one tar stream, set "
                "format = 'multipart_tar'."
            ) from exc
    return {"format": fmt, "written_from_source": written}


def _copy_tar_images(
    ds: DatasetConfig,
    archive: tarfile.TarFile,
    writer: DatasetShardWriter,
    prefix: str,
    source_archive: str,
) -> int:
    written = 0
    for member in archive:
        if not member.isfile() or Path(member.name).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        safe_member = _safe_member_name(member.name)
        src = archive.extractfile(member)
        if safe_member is None or src is None:
            log.warning("skipping unsafe/unreadable tar entry %r", member.name)
            continue
        relative = normalize_relpath(Path(prefix) / safe_member)
        label, generator = _archive_meta(ds, relative)
        with src:
            data = copy_stream(src)
        if writer.add(
            data,
            f"{ds.name}/{relative}",
            label,
            generator,
            metadata={"source_archive": source_archive, "source_member": member.name},
        ):
            written += 1
    return written


class _MultipartFile(io.RawIOBase):
    """Seekable view over split zip parts without creating a combined archive."""

    def __init__(self, paths: list[Path]) -> None:
        super().__init__()
        self.paths = paths
        self.offsets = [0]
        for path in paths:
            self.offsets.append(self.offsets[-1] + path.stat().st_size)
        self.position = 0
        self._part_index = -1
        self._part = None

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_CUR:
            offset += self.position
        elif whence == io.SEEK_END:
            offset += self.offsets[-1]
        if offset < 0:
            raise ValueError("negative seek position")
        self.position = min(offset, self.offsets[-1])
        return self.position

    def read(self, size: int = -1) -> bytes:
        remaining = self.offsets[-1] - self.position
        if size < 0 or size > remaining:
            size = remaining
        chunks: list[bytes] = []
        while size:
            index = min(bisect.bisect_right(self.offsets, self.position) - 1, len(self.paths) - 1)
            if index != self._part_index:
                if self._part is not None:
                    self._part.close()
                self._part = self.paths[index].open("rb")
                self._part_index = index
            local = self.position - self.offsets[index]
            self._part.seek(local)
            take = min(size, self.offsets[index + 1] - self.position)
            chunk = self._part.read(take)
            if not chunk:
                break
            chunks.append(chunk)
            self.position += len(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self._part is not None:
            self._part.close()
        super().close()


def _metadata_members(path: Path, output_column: str) -> set[str]:
    selected: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line).get(output_column)
            except (json.JSONDecodeError, AttributeError) as exc:
                raise ConfigError(f"invalid multipart metadata at {path}:{line_no}") from exc
            values = value if isinstance(value, list) else [value]
            selected.update(str(item).replace("\\", "/").lstrip("./") for item in values if item)
    return selected


def _materialize_multipart_zip(
    ds: DatasetConfig, source_dir: Path, writer: DatasetShardWriter
) -> dict:
    parts = sorted(source_dir.glob(ds.multipart_glob))
    metadata = source_dir / ds.metadata_file
    if not parts or not metadata.is_file():
        raise ConfigError(
            f"dataset {ds.name!r}: missing multipart files {ds.multipart_glob!r} "
            f"or metadata {ds.metadata_file!r}"
        )
    selected = _metadata_members(metadata, ds.output_column)
    written = missing = 0
    joined = _MultipartFile(parts)
    try:
        with zipfile.ZipFile(joined) as archive:
            members = {
                info.filename.replace("\\", "/").lstrip("./"): info
                for info in archive.infolist()
                if not info.is_dir()
            }
            for name in sorted(selected):
                info = members.get(name)
                if info is None:
                    missing += 1
                    continue
                if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                safe_member = _safe_member_name(name)
                if safe_member is None:
                    log.warning("skipping unsafe multipart entry %r", name)
                    continue
                label, generator = _archive_meta(ds, safe_member)
                with archive.open(info) as src:
                    data = copy_stream(src)
                if writer.add(
                    data,
                    f"{ds.name}/{safe_member}",
                    label,
                    generator,
                    metadata={"source_member": name},
                ):
                    written += 1
    finally:
        joined.close()
    return {"format": "multipart_zip", "written_from_source": written, "missing": missing}


def _materialize_imagefolder(
    ds: DatasetConfig, source_dir: Path, writer: DatasetShardWriter
) -> dict:
    source = source_dir / ds.image_dir if ds.image_dir else source_dir
    if not source.exists():
        raise RuntimeError(f"image_dir {source} does not exist for dataset {ds.name!r}")
    moved = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = normalize_relpath(path.relative_to(source))
        label, generator = _archive_meta(ds, relative)
        delete_source = path if (not ds.is_local or ds.allow_delete) else None
        if writer.add(
            path.read_bytes(),
            f"{ds.name}/{relative}",
            label,
            generator,
            metadata={"source_path": relative},
            delete_source=delete_source,
        ):
            moved += 1
    if moved == 0:
        raise RuntimeError(
            f"No images found for dataset {ds.name!r} under {source}. "
            "Specify image_dir in the config."
        )
    return {"format": "imagefolder", "written_from_source": moved}
