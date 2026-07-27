from __future__ import annotations

import io
import json
import logging
import mimetypes
import os
import re
import sqlite3
import tarfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .utils import IMAGE_SUFFIXES, image_id, normalize_relpath, safe_name, sniff_extension

log = logging.getLogger("databuilder.wds")

STORAGE_VERSION = 2
INDEX_NAME = "index.parquet"
DATASET_DESCRIPTOR = "dataset.json"
CONVERSION_DB = ".conversion.sqlite"
MATERIALIZED_MARKER = ".materialized.json"

LOCATOR_FIELDS = [
    ("shard", pa.string()),
    ("member", pa.string()),
    ("offset", pa.int64()),
    ("size", pa.int64()),
]

INDEX_SCHEMA = pa.schema(
    [
        ("image_id", pa.uint64()),
        ("path", pa.string()),
        ("dataset", pa.string()),
        ("label", pa.string()),
        ("generator", pa.string()),
        ("source_split", pa.string()),
        *LOCATOR_FIELDS,
        ("extension", pa.string()),
    ]
)


def marker_path(root: Path) -> Path:
    return root / MATERIALIZED_MARKER


def load_marker(root: Path) -> dict:
    path = marker_path(root)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def is_webdataset(root: Path) -> bool:
    marker = load_marker(root)
    return (
        marker.get("storage_version") == STORAGE_VERSION
        and marker.get("storage") == "webdataset"
        and marker.get("state", "complete") == "complete"
        and (root / INDEX_NAME).is_file()
    )


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def _padded(size: int) -> int:
    return (size + 511) // 512 * 512


def _safe_extension(logical_path: str, data: bytes) -> str:
    suffix = Path(logical_path).suffix.lower()
    fallback = suffix if suffix in IMAGE_SUFFIXES else ".img"
    return sniff_extension(data, fallback=fallback)


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass(frozen=True)
class ImageRef:
    path: str
    dataset: str
    shard: str = ""
    member: str = ""
    offset: int = 0
    size: int = 0

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "ImageRef":
        return cls(
            path=str(row["path"]),
            dataset=str(row.get("dataset") or str(row["path"]).partition("/")[0]),
            shard=str(row.get("shard") or ""),
            member=str(row.get("member") or ""),
            offset=int(row.get("offset") or 0),
            size=int(row.get("size") or row.get("filesize") or 0),
        )

    def display(self, roots: Mapping[str, str]) -> str:
        root = Path(roots[self.dataset])
        if self.shard:
            return f"wds://{root / self.shard}#{self.member}"
        _, _, rest = self.path.partition("/")
        return str(root / rest)

    def read_bytes(self, roots: Mapping[str, str]) -> bytes:
        root = Path(roots[self.dataset])
        if not self.shard:
            _, _, rest = self.path.partition("/")
            return (root / rest).read_bytes()
        shard_path = root / self.shard
        with shard_path.open("rb", buffering=0) as handle:
            if hasattr(os, "pread"):
                return os.pread(handle.fileno(), self.size, self.offset)
            handle.seek(self.offset)
            return handle.read(self.size)


def read_image_bytes(row: Mapping[str, object], roots: Mapping[str, str]) -> bytes:
    return ImageRef.from_row(row).read_bytes(roots)


def image_media_type(row: Mapping[str, object]) -> str:
    member = str(row.get("member") or row.get("path") or "")
    return mimetypes.guess_type(member)[0] or "application/octet-stream"


def iter_index(root: Path, columns: list[str] | None = None) -> Iterator[dict]:
    index = root / INDEX_NAME
    if not index.is_file():
        return
    parquet = pq.ParquetFile(index)
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        yield from batch.to_pylist()


def lookup_path(root: Path, logical_path: str) -> dict | None:
    index = root / INDEX_NAME
    if not index.is_file():
        return None
    table = pq.read_table(index, filters=[("path", "=", normalize_relpath(logical_path))])
    rows = table.to_pylist()
    return rows[0] if rows else None


def resolve_logical_path(roots: Mapping[str, str], logical_path: str) -> dict:
    dataset, _, rest = normalize_relpath(logical_path).partition("/")
    if dataset not in roots:
        raise KeyError(f"unknown dataset {dataset!r} in {logical_path!r}")
    root = Path(roots[dataset])
    if (root / INDEX_NAME).is_file():
        row = lookup_path(root, logical_path)
        if row is None:
            raise FileNotFoundError(f"{logical_path!r} is absent from {root / INDEX_NAME}")
        return row
    path = root / rest
    return {
        "path": logical_path,
        "dataset": dataset,
        "shard": "",
        "member": "",
        "offset": 0,
        "size": path.stat().st_size,
        "extension": path.suffix.lower(),
    }


class DatasetShardWriter:
    """Transactional writer for one canonical WebDataset.

    SQLite is a source-to-shard conversion journal, not a second download
    resume mechanism. A source is deleted only after its shard is validated.
    """

    def __init__(
        self,
        root: Path,
        dataset: str,
        target_shard_bytes: int = 3_000_000_000,
        max_samples_per_shard: int = 100_000,
    ) -> None:
        self.root = Path(root)
        self.dataset = dataset
        self.target_shard_bytes = target_shard_bytes
        self.max_samples_per_shard = max_samples_per_shard
        self.shards_dir = self.root / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / CONVERSION_DB
        self.db: sqlite3.Connection | None = sqlite3.connect(self.db_path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                path TEXT PRIMARY KEY,
                image_id_hex TEXT NOT NULL,
                dataset TEXT NOT NULL,
                label TEXT NOT NULL,
                generator TEXT NOT NULL,
                source_split TEXT NOT NULL,
                shard TEXT NOT NULL,
                member TEXT NOT NULL,
                offset INTEGER NOT NULL,
                size INTEGER NOT NULL,
                extension TEXT NOT NULL
            )
            """
        )
        self.db.commit()
        self._import_existing_index()
        self._known_existing = bool(self.db.execute("SELECT 1 FROM samples LIMIT 1").fetchone())
        self._remove_orphan_shards()
        self._tar: tarfile.TarFile | None = None
        self._handle = None
        self._partial: Path | None = None
        self._final: Path | None = None
        self._rows: list[dict] = []
        self._row_paths: set[str] = set()
        self._delete_after_commit: set[Path] = set()
        self._shard_payload_bytes = 0
        self._shard_index = self._next_shard_index()
        self.written = 0
        self.skipped_existing = 0
        self._remove_uncommitted_partials()

    def _import_existing_index(self) -> None:
        assert self.db is not None
        if self.db.execute("SELECT 1 FROM samples LIMIT 1").fetchone():
            return
        if not (self.root / INDEX_NAME).is_file():
            return
        rows: list[tuple] = []
        for row in iter_index(self.root):
            rows.append(
                (
                    row["path"], f"{int(row['image_id']):016x}", row["dataset"],
                    row["label"], row["generator"], row.get("source_split") or "",
                    row["shard"], row["member"], int(row["offset"]), int(row["size"]),
                    row["extension"],
                )
            )
            if len(rows) >= 20_000:
                self._insert_rows(rows)
                rows.clear()
        if rows:
            self._insert_rows(rows)

    def _insert_rows(self, rows: list[tuple]) -> None:
        assert self.db is not None
        self.db.executemany(
            "INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        self.db.commit()

    def _remove_uncommitted_partials(self) -> None:
        for path in self.shards_dir.glob("*.tar.partial"):
            path.unlink(missing_ok=True)

    def _remove_orphan_shards(self) -> None:
        assert self.db is not None
        referenced = {
            row[0] for row in self.db.execute("SELECT DISTINCT shard FROM samples").fetchall()
        }
        for path in self.shards_dir.glob("*.tar"):
            relative = normalize_relpath(path.relative_to(self.root))
            if relative not in referenced:
                log.warning("removing uncommitted orphan shard %s", path)
                path.unlink(missing_ok=True)

    def _next_shard_index(self) -> int:
        prefix = re.escape(safe_name(self.dataset))
        found = []
        for path in self.shards_dir.glob("*.tar"):
            match = re.fullmatch(rf"{prefix}-(\d{{6}})\.tar", path.name)
            if match:
                found.append(int(match.group(1)))
        return max(found, default=-1) + 1

    def contains(self, logical_path: str) -> bool:
        logical_path = normalize_relpath(logical_path)
        if logical_path in self._row_paths:
            return True
        if not self._known_existing:
            return False
        assert self.db is not None
        return self.db.execute(
            "SELECT 1 FROM samples WHERE path = ?", (logical_path,)
        ).fetchone() is not None

    def _start_shard(self) -> None:
        name = f"{safe_name(self.dataset)}-{self._shard_index:06d}.tar"
        self._final = self.shards_dir / name
        self._partial = self.shards_dir / f"{name}.partial"
        self._handle = self._partial.open("w+b")
        self._tar = tarfile.open(fileobj=self._handle, mode="w", format=tarfile.USTAR_FORMAT)
        self._shard_payload_bytes = 0

    @staticmethod
    def _estimated_sample_size(image_size: int, metadata_size: int) -> int:
        return 512 + _padded(image_size) + 512 + _padded(metadata_size)

    def add(
        self,
        data: bytes,
        logical_path: str,
        label: str,
        generator: str,
        *,
        source_split: str = "",
        metadata: Mapping[str, object] | None = None,
        delete_source: Path | None = None,
    ) -> bool:
        logical_path = normalize_relpath(logical_path)
        if self.contains(logical_path):
            self.skipped_existing += 1
            if delete_source is not None:
                delete_source.unlink(missing_ok=True)
            return False

        ident = image_id(logical_path)
        key = f"{ident:016x}"
        extension = _safe_extension(logical_path, data)
        member = f"{key}{extension}"
        sample_meta = {
            "__key__": key, "image_id": str(ident), "path": logical_path,
            "dataset": self.dataset, "label": label, "generator": generator,
            "source_split": source_split,
        }
        if metadata:
            sample_meta["source_metadata"] = _jsonable(metadata)
        metadata_bytes = json.dumps(
            sample_meta, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        estimate = self._estimated_sample_size(len(data), len(metadata_bytes))
        if self._tar is not None and (
            len(self._rows) >= self.max_samples_per_shard
            or self._shard_payload_bytes + estimate > self.target_shard_bytes
        ):
            self._commit_shard()
        if self._tar is None:
            self._start_shard()

        assert self._tar is not None and self._handle is not None and self._final is not None
        image_info = tarfile.TarInfo(member)
        image_info.size, image_info.mode, image_info.mtime = len(data), 0o644, 0
        data_offset = self._handle.tell() + 512
        self._tar.addfile(image_info, io.BytesIO(data))

        json_info = tarfile.TarInfo(f"{key}.json")
        json_info.size, json_info.mode, json_info.mtime = len(metadata_bytes), 0o644, 0
        self._tar.addfile(json_info, io.BytesIO(metadata_bytes))

        self._rows.append(
            {
                "path": logical_path, "image_id_hex": f"{ident:016x}",
                "dataset": self.dataset, "label": str(label), "generator": str(generator),
                "source_split": str(source_split or ""),
                "shard": normalize_relpath(self._final.relative_to(self.root)),
                "member": member, "offset": data_offset, "size": len(data),
                "extension": extension,
            }
        )
        self._row_paths.add(logical_path)
        if delete_source is not None:
            self._delete_after_commit.add(Path(delete_source))
        self._shard_payload_bytes += estimate
        self.written += 1
        return True

    @staticmethod
    def _validate_partial(path: Path, expected: int) -> None:
        images = metadata = 0
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                if member.isfile() and member.name.endswith(".json"):
                    metadata += 1
                elif member.isfile():
                    images += 1
        if images != expected or metadata != expected:
            raise RuntimeError(
                f"shard validation failed for {path}: "
                f"{images} images/{metadata} metadata, expected {expected}"
            )

    def _commit_shard(self) -> None:
        if self._tar is None:
            return
        assert self._handle is not None and self._partial is not None and self._final is not None
        assert self.db is not None
        self._tar.close()
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._validate_partial(self._partial, len(self._rows))
        self._partial.replace(self._final)
        rows = [
            (
                row["path"], row["image_id_hex"], row["dataset"], row["label"],
                row["generator"], row["source_split"], row["shard"], row["member"],
                row["offset"], row["size"], row["extension"],
            )
            for row in self._rows
        ]
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
        self._known_existing = True
        for source in self._delete_after_commit:
            try:
                source.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("committed %s but could not remove %s: %s", self._final, source, exc)
        log.info(
            "committed WebDataset shard %s (%d samples, %.2f GiB)",
            self._final, len(self._rows), self._final.stat().st_size / 2**30,
        )
        self._tar = self._handle = self._partial = self._final = None
        self._rows.clear()
        self._row_paths.clear()
        self._delete_after_commit.clear()
        self._shard_payload_bytes = 0
        self._shard_index += 1

    def checkpoint(self) -> None:
        self._commit_shard()

    def _export_index(self) -> int:
        assert self.db is not None
        partial = self.root / f"{INDEX_NAME}.partial"
        partial.unlink(missing_ok=True)
        writer = pq.ParquetWriter(partial, INDEX_SCHEMA)
        total = 0
        try:
            cursor = self.db.execute(
                """
                SELECT image_id_hex, path, dataset, label, generator, source_split,
                       shard, member, offset, size, extension
                FROM samples ORDER BY shard, path
                """
            )
            while rows := cursor.fetchmany(20_000):
                records = [
                    {
                        "image_id": int(row[0], 16), "path": row[1], "dataset": row[2],
                        "label": row[3], "generator": row[4], "source_split": row[5],
                        "shard": row[6], "member": row[7], "offset": row[8],
                        "size": row[9], "extension": row[10],
                    }
                    for row in rows
                ]
                writer.write_table(pa.Table.from_pylist(records, schema=INDEX_SCHEMA))
                total += len(records)
        finally:
            writer.close()
        partial.replace(self.root / INDEX_NAME)
        return total

    def finalize(self, stats: Mapping[str, object] | None = None) -> dict:
        assert self.db is not None
        self._commit_shard()
        total = self._export_index()
        shard_rows = self.db.execute(
            "SELECT shard, COUNT(*) FROM samples GROUP BY shard ORDER BY shard"
        ).fetchall()
        descriptor = {
            "__kind__": "WebDataset", "wids_version": 1, "name": self.dataset,
            "shardlist": [
                {
                    "url": shard, "nsamples": count,
                    "filesize": (self.root / shard).stat().st_size,
                }
                for shard, count in shard_rows if (self.root / shard).is_file()
            ],
        }
        atomic_json(self.root / DATASET_DESCRIPTOR, descriptor)
        result = {
            "storage_version": STORAGE_VERSION, "storage": "webdataset",
            "state": "complete",
            "written": total, "shards": len(descriptor["shardlist"]),
            "target_shard_bytes": self.target_shard_bytes,
            "max_samples_per_shard": self.max_samples_per_shard,
            **dict(stats or {}),
        }
        atomic_json(marker_path(self.root), result)
        self.db.close()
        self.db = None
        self._remove_journal()
        return result

    def _remove_journal(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(str(self.db_path) + suffix).unlink(missing_ok=True)

    def close(self) -> None:
        if self.db is None:
            return
        self._commit_shard()
        self.db.close()
        self.db = None

    def __enter__(self) -> "DatasetShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def remove_empty_directories(root: Path) -> None:
    root = root.resolve()
    if not root.is_dir():
        return
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts), reverse=True,
    )
    for path in directories:
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or resolved == root:
            continue
        try:
            path.rmdir()
        except OSError:
            pass


def archive_raw_snapshot(source: Path, target: Path, dataset: str) -> dict:
    """Store a download-only snapshot as one tar instead of a persistent tree."""
    raw_dir = target / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    final = raw_dir / f"{safe_name(dataset)}.tar"
    partial = final.with_name(final.name + ".partial")
    files = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_relative_to(raw_dir):
            continue
        relative = path.relative_to(source)
        if path.name == MATERIALIZED_MARKER or relative.parts[:2] == (".cache", "huggingface"):
            continue
        files.append(path)
    with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            archive.add(path, arcname=normalize_relpath(path.relative_to(source)), recursive=False)
    with tarfile.open(partial, "r:") as archive:
        count = sum(1 for member in archive if member.isfile())
    if count != len(files):
        raise RuntimeError(f"raw repository archive validation failed for {dataset}")
    partial.replace(final)
    return {
        "storage_version": STORAGE_VERSION, "storage": "raw_tar", "format": "raw",
        "download_only": True, "files": count,
        "archive": normalize_relpath(final.relative_to(target)),
    }


def remove_tree_contents(source: Path, *, keep: set[Path] | None = None) -> None:
    """Delete an exact, resolved staging tree after its replacement is committed."""
    source = source.resolve()
    keep_resolved = {path.resolve() for path in (keep or set())}
    if not source.is_dir():
        return
    for path in sorted(source.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            path.unlink(missing_ok=True)
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(source) or resolved in keep_resolved:
            continue
        if path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    try:
        source.rmdir()
    except OSError:
        pass


def copy_stream(src, chunk_size: int = 1024 * 1024) -> bytes:
    """Read one archive member without materializing it as a loose file."""
    chunks: list[bytes] = []
    while chunk := src.read(chunk_size):
        chunks.append(chunk)
    return b"".join(chunks)


def _create_sample_table(db: sqlite3.Connection) -> None:
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
            path TEXT PRIMARY KEY, image_id_hex TEXT NOT NULL, dataset TEXT NOT NULL,
            label TEXT NOT NULL, generator TEXT NOT NULL, source_split TEXT NOT NULL,
            shard TEXT NOT NULL, member TEXT NOT NULL, offset INTEGER NOT NULL,
            size INTEGER NOT NULL, extension TEXT NOT NULL
        )
        """
    )


def _db_row(row: Mapping[str, object]) -> tuple:
    return (
        row["path"], f"{int(row['image_id']):016x}", row["dataset"], row["label"],
        row["generator"], row.get("source_split") or "", row["shard"], row["member"],
        int(row["offset"]), int(row["size"]), row["extension"],
    )


def _export_index_db(db: sqlite3.Connection, destination: Path) -> int:
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    writer = pq.ParquetWriter(partial, INDEX_SCHEMA)
    total = 0
    try:
        cursor = db.execute(
            """
            SELECT image_id_hex, path, dataset, label, generator, source_split,
                   shard, member, offset, size, extension
            FROM samples ORDER BY shard, path
            """
        )
        while rows := cursor.fetchmany(20_000):
            records = [
                {
                    "image_id": int(row[0], 16), "path": row[1], "dataset": row[2],
                    "label": row[3], "generator": row[4], "source_split": row[5],
                    "shard": row[6], "member": row[7], "offset": row[8],
                    "size": row[9], "extension": row[10],
                }
                for row in rows
            ]
            writer.write_table(pa.Table.from_pylist(records, schema=INDEX_SCHEMA))
            total += len(records)
    finally:
        writer.close()
    partial.replace(destination)
    return total


def _write_compact_tar(path: Path, rows: list[dict], root: Path) -> list[dict]:
    output: list[dict] = []
    roots = {str(rows[0]["dataset"]): str(root)} if rows else {}
    with path.open("w+b") as handle:
        with tarfile.open(fileobj=handle, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for row in rows:
                data = ImageRef.from_row(row).read_bytes(roots)
                info = tarfile.TarInfo(str(row["member"]))
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                offset = handle.tell() + 512
                archive.addfile(info, io.BytesIO(data))
                key = Path(str(row["member"])).stem
                metadata = json.dumps(
                    {
                        "__key__": key, "image_id": str(row["image_id"]),
                        "path": row["path"], "dataset": row["dataset"],
                        "label": row["label"], "generator": row["generator"],
                        "source_split": row.get("source_split") or "",
                    },
                    ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8")
                meta = tarfile.TarInfo(f"{key}.json")
                meta.size, meta.mode, meta.mtime = len(metadata), 0o644, 0
                archive.addfile(meta, io.BytesIO(metadata))
                output.append({**row, "offset": offset, "size": len(data)})
        handle.flush()
        os.fsync(handle.fileno())
    DatasetShardWriter._validate_partial(path, len(rows))
    return output


def compact_dataset(root: Path, selected_paths: set[str]) -> dict:
    """Rewrite each WDS shard in place with selected samples only.

    The extra disk requirement is bounded by one shard. A state marker prevents
    readers from using stale offsets if the operation is interrupted.
    """
    root = Path(root)
    marker = load_marker(root)
    if marker.get("storage") != "webdataset" or not (root / INDEX_NAME).is_file():
        raise RuntimeError(f"{root} is not a materialized WebDataset")
    compact_db_path = root / ".compact.sqlite"
    db = sqlite3.connect(compact_db_path)
    db.execute("PRAGMA synchronous=FULL")
    _create_sample_table(db)
    db.execute(
        "CREATE TABLE IF NOT EXISTS processed (shard TEXT PRIMARY KEY, state TEXT NOT NULL)"
    )
    db.commit()
    marker["state"] = "compacting"
    atomic_json(marker_path(root), marker)

    descriptor = json.loads((root / DATASET_DESCRIPTOR).read_text(encoding="utf-8"))
    shard_names = [entry["url"] for entry in descriptor.get("shardlist", [])]
    kept_total = removed_total = rewritten = 0
    for shard in shard_names:
        state_row = db.execute("SELECT state FROM processed WHERE shard = ?", (shard,)).fetchone()
        final = root / shard
        partial = final.with_name(final.name + ".compact.partial")
        if state_row and state_row[0] == "prepared":
            if partial.is_file():
                partial.replace(final)
            elif not final.is_file():
                pass  # prepared empty shard
            with db:
                db.execute("UPDATE processed SET state='committed' WHERE shard = ?", (shard,))
            continue
        if state_row and state_row[0] == "committed":
            continue

        table = pq.read_table(root / INDEX_NAME, filters=[("shard", "=", shard)])
        original = table.to_pylist()
        kept = [row for row in original if row["path"] in selected_paths]
        kept_total += len(kept)
        removed_total += len(original) - len(kept)
        if len(kept) == len(original):
            with db:
                db.executemany(
                    "INSERT OR REPLACE INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [_db_row(row) for row in kept],
                )
                db.execute("INSERT OR REPLACE INTO processed VALUES (?, 'committed')", (shard,))
            continue

        new_rows = _write_compact_tar(partial, kept, root) if kept else []
        with db:
            db.executemany(
                "INSERT OR REPLACE INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [_db_row(row) for row in new_rows],
            )
            db.execute("INSERT OR REPLACE INTO processed VALUES (?, 'prepared')", (shard,))
        if kept:
            partial.replace(final)
        else:
            final.unlink(missing_ok=True)
        with db:
            db.execute("UPDATE processed SET state='committed' WHERE shard = ?", (shard,))
        rewritten += 1

    total = _export_index_db(db, root / INDEX_NAME)
    active = db.execute(
        "SELECT shard, COUNT(*) FROM samples GROUP BY shard ORDER BY shard"
    ).fetchall()
    descriptor["shardlist"] = [
        {"url": shard, "nsamples": count, "filesize": (root / shard).stat().st_size}
        for shard, count in active if (root / shard).is_file()
    ]
    atomic_json(root / DATASET_DESCRIPTOR, descriptor)
    marker.update({"state": "complete", "written": total, "shards": len(active)})
    atomic_json(marker_path(root), marker)
    db.close()
    for suffix in ("", "-journal", "-wal", "-shm"):
        Path(str(compact_db_path) + suffix).unlink(missing_ok=True)
    return {
        "kept": kept_total, "removed": removed_total, "rewritten_shards": rewritten,
        "shards": len(active),
    }
