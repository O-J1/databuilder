from __future__ import annotations

import bisect
import io
import json
import logging
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import ConfigError, DatasetConfig, ImageColumnConfig
from ..state import RunContext
from ..utils import IMAGE_SUFFIXES, safe_name, sniff_extension
from .common import MATERIALIZED_MARKER, iter_dataset_images

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
        if ds.in_place:
            count = sum(1 for _ in iter_dataset_images(Path(ds.path), ds))
            if count == 0:
                raise ConfigError(f"dataset {ds.name!r}: no images found under {ds.path}")
            log.info("dataset %r: %d images in place at %s", ds.name, count, ds.path)
            continue
        target = ctx.data_dir / ds.name
        done_marker = target / MATERIALIZED_MARKER
        if done_marker.exists():
            log.info("dataset %r already prepared, skipping", ds.name)
            continue
        if ctx.dry_run:
            log.info("dry-run: would snapshot/materialize %r (%s)", ds.name, ds.repo_id or ds.path)
            continue
        stats = _materialize(ctx, ds, target)
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        log.info("dataset %r prepared: %s", ds.name, stats)


def _materialize(ctx: RunContext, ds: DatasetConfig, target: Path) -> dict:
    if ds.is_local:
        source_dir = Path(ds.path)
    else:
        # Set the Xet cache before importing huggingface_hub so its constants
        # cannot resolve to a user-home cache outside the configured data tree.
        os.environ["HF_XET_CACHE"] = str(ctx.data_dir / ".hf_xet")
        if ctx.cfg.download.xet_high_performance:
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        from huggingface_hub import snapshot_download

        # Keep every network write beneath data_dir. Xet/HF owns partial state
        # and resume behavior; databuilder deliberately does not download files
        # itself or maintain a second resume manifest.
        local_dir = target if ds.download_only else ctx.data_dir / ".hf_snapshots" / ds.name
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
    if ds.download_only:
        return {"format": "raw", "snapshot": str(source_dir), "download_only": True}

    fmt = ds.format if ds.format != "auto" else _detect_format(source_dir)
    log.info("dataset %r: format=%s source=%s", ds.name, fmt, source_dir)
    target.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        return _materialize_parquet(ctx, ds, source_dir, target)
    if fmt == "arrow":
        return _materialize_arrow(ctx, ds, source_dir, target)
    if fmt == "jsonl":
        return _materialize_jsonl(ctx, ds, source_dir, target)
    if fmt == "zip":
        return _materialize_zip(ds, source_dir, target)
    if fmt in {"tar", "webdataset"}:
        return _materialize_tar(ds, source_dir, target, fmt)
    if fmt == "multipart_zip":
        return _materialize_multipart_zip(ds, source_dir, target)
    if fmt == "imagefolder":
        return _materialize_imagefolder(ds, source_dir, target)
    raise ConfigError(f"dataset {ds.name!r}: unsupported materialization format {fmt!r}")


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
    for column in ds.row_filter:
        _require(ds, names, "row_filter", column)
    return TableMap(ds.images, label, generator, split)


class _RowMaterializer:
    def __init__(
        self,
        ctx: RunContext,
        ds: DatasetConfig,
        target: Path,
        source_dir: Path,
        mapping: TableMap,
    ) -> None:
        self.ctx = ctx
        self.ds = ds
        self.target = target
        self.source_dir = source_dir
        self.mapping = mapping
        self.written = 0
        self.skipped = 0

    @property
    def layout(self) -> dict[str, bool]:
        generator_dir = bool(
            self.mapping.generator
            or any(spec.generator or spec.generator_column for spec in self.mapping.images)
        )
        return {"label_dir": self.mapping.label is not None, "generator_dir": generator_dir}

    def add(self, row: dict, file_idx: int, source_path: Path) -> None:
        if any(row.get(column) != expected for column, expected in self.ds.row_filter.items()):
            return
        if (
            self.mapping.split
            and self.ds.source_split
            and str(row.get(self.mapping.split)) != self.ds.source_split
        ):
            return
        for image_idx, spec in enumerate(self.mapping.images):
            data = _encoded_bytes(
                self.ctx, row.get(spec.column), spec.kind, source_path, self.source_dir
            )
            if data is None:
                self.skipped += 1
                continue
            out_dir = self.target
            if self.mapping.label:
                out_dir /= safe_name(row.get(self.mapping.label)).lower()
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
                out_dir /= safe_name(generator)
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_i{image_idx}" if len(self.mapping.images) > 1 else ""
            name = (
                f"{self.ds.name}_{file_idx:05d}_{self.written:09d}{suffix}"
                f"{sniff_extension(data)}"
            )
            (out_dir / name).write_bytes(data)
            self.written += 1

    def stats(self, fmt: str) -> dict:
        return {
            "format": fmt,
            "written": self.written,
            "skipped": self.skipped,
            "layout": self.layout,
        }


def _selected_columns(ds: DatasetConfig, mapping: TableMap) -> list[str]:
    columns = [spec.column for spec in mapping.images]
    columns += [spec.generator_column for spec in mapping.images if spec.generator_column]
    columns += [mapping.label, mapping.generator, mapping.split]
    columns += list(ds.row_filter)
    return list(dict.fromkeys(column for column in columns if column))


def _materialize_parquet(
    ctx: RunContext, ds: DatasetConfig, source_dir: Path, target: Path
) -> dict:
    files = sorted(source_dir.rglob("*.parquet"))
    if not files:
        raise ConfigError(f"dataset {ds.name!r}: no parquet files under {source_dir}")
    if ds.source_split:
        matched = [path for path in files if ds.source_split in path.name]
        files = matched or files
    mapping = _resolve_table(ds, pq.ParquetFile(files[0]).schema_arrow)
    writer = _RowMaterializer(ctx, ds, target, source_dir, mapping)
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
        _discard_source(ds, parquet_path)
    return writer.stats("parquet")


def _open_arrow(path: Path):
    source = pa.memory_map(str(path), "r")
    try:
        return source, pa.ipc.open_stream(source)
    except pa.ArrowInvalid:
        source.seek(0)
        return source, pa.ipc.open_file(source)


def _materialize_arrow(
    ctx: RunContext, ds: DatasetConfig, source_dir: Path, target: Path
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
    writer = _RowMaterializer(ctx, ds, target, source_dir, mapping)
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
        _discard_source(ds, arrow_path)
    return writer.stats("arrow")


def _materialize_jsonl(
    ctx: RunContext, ds: DatasetConfig, source_dir: Path, target: Path
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
    writer = _RowMaterializer(ctx, ds, target, source_dir, mapping)
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
        _discard_source(ds, jsonl_path)
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


def _safe_destination(base: Path, member_name: str) -> Path | None:
    destination = (base / member_name.replace("\\", "/")).resolve()
    return destination if destination.is_relative_to(base.resolve()) else None


def _materialize_zip(ds: DatasetConfig, source_dir: Path, target: Path) -> dict:
    extracted = 0
    archives = sorted(source_dir.rglob("*.zip"))
    if not archives:
        raise ConfigError(f"dataset {ds.name!r}: no zip files under {source_dir}")
    for zip_path in archives:
        out_dir = target / safe_name(zip_path.stem)
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir() or Path(member.filename).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                dest = _safe_destination(out_dir, member.filename)
                if dest is None:
                    log.warning("skipping unsafe zip entry %r", member.filename)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1
        _discard_source(ds, zip_path)
    return {"format": "zip", "written": extracted}


def _tar_files(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file()
        and (path.suffix.lower() == ".tar" or path.name.lower().endswith((".tar.gz", ".tgz")))
    )


def _materialize_tar(
    ds: DatasetConfig, source_dir: Path, target: Path, fmt: str
) -> dict:
    extracted = 0
    archives = _tar_files(source_dir)
    if not archives:
        raise ConfigError(f"dataset {ds.name!r}: no tar files under {source_dir}")
    for tar_path in archives:
        stem = tar_path.name.removesuffix(".tar.gz").removesuffix(".tgz").removesuffix(".tar")
        out_dir = target / safe_name(stem)
        with tarfile.open(tar_path, "r:*") as archive:
            for member in archive:
                if not member.isfile() or Path(member.name).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                dest = _safe_destination(out_dir, member.name)
                src = archive.extractfile(member)
                if dest is None or src is None:
                    log.warning("skipping unsafe/unreadable tar entry %r", member.name)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1
        _discard_source(ds, tar_path)
    return {"format": fmt, "written": extracted}


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
    ds: DatasetConfig, source_dir: Path, target: Path
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
                dest = _safe_destination(target, name)
                if dest is None:
                    log.warning("skipping unsafe multipart entry %r", name)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                written += 1
    finally:
        joined.close()
    if not ds.keep_archives and not ds.is_local:
        for part in parts:
            part.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)
    return {"format": "multipart_zip", "written": written, "missing": missing}


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
        if ds.is_local:
            shutil.copy2(path, dest)
        else:
            shutil.move(str(path), str(dest))
        moved += 1
    if moved == 0:
        raise RuntimeError(
            f"No images found for dataset {ds.name!r} under {source}. "
            "Specify image_dir in the config."
        )
    return {"format": "imagefolder", "written": moved}


def _discard_source(ds: DatasetConfig, path: Path) -> None:
    if not ds.keep_archives and not ds.is_local:
        path.unlink(missing_ok=True)
