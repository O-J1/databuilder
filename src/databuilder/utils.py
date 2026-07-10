from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
    ".tif", ".tiff", ".heic", ".heif", ".jxl", ".avif",
}


def normalize_relpath(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def stable_hash64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def image_id(relpath: str | Path) -> int:
    """Deterministic uint64 id shared by every stage for one image path."""
    return stable_hash64(normalize_relpath(relpath))


def owns(relpath: str | Path, rank: int, world_size: int) -> bool:
    """Static shard assignment: does `rank` own this image path?"""
    if world_size <= 1:
        return True
    return image_id(relpath) % world_size == rank


def safe_name(value: object) -> str:
    name = str(value if value is not None else "unknown").strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).rstrip(". ")
    return name or "unknown"


def sniff_extension(data: bytes, fallback: str = ".img") -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return ".heic"
    if data.startswith(b"\xff\x0a") or data.startswith(b"\x00\x00\x00\x0cJXL "):
        return ".jxl"
    if data.startswith(b"BM"):
        return ".bmp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return ".tif"
    return fallback


class ParquetShardWriter:
    """Buffered single-file parquet writer with bounded memory."""

    def __init__(self, path: Path | str, schema: pa.Schema, flush_rows: int = 20_000):
        self.path = Path(path)
        self.schema = schema
        self.flush_rows = flush_rows
        self.rows_written = 0
        self._rows: list[dict] = []
        self._writer: pq.ParquetWriter | None = None

    def append(self, row: dict) -> None:
        self._rows.append(row)
        if len(self._rows) >= self.flush_rows:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, self.schema)
        self._writer.write_table(pa.Table.from_pylist(self._rows, schema=self.schema))
        self.rows_written += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "ParquetShardWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def collect_parquet_files(paths: Sequence[Path | str] | Path | str) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files: list[Path] = []
    for entry in paths:
        entry = Path(entry)
        if entry.is_dir():
            files.extend(sorted(entry.rglob("*.parquet")))
        elif entry.exists():
            files.append(entry)
    return files


def iter_parquet_batches(
    paths: Sequence[Path | str] | Path | str,
    columns: list[str] | None = None,
    batch_rows: int = 8192,
) -> Iterator[pa.RecordBatch]:
    """Stream record batches from one or more parquet files/directories."""
    files = collect_parquet_files(paths)
    if not files:
        return
    dataset = pa_ds.dataset([str(f) for f in files], format="parquet")
    yield from dataset.to_batches(columns=columns, batch_size=batch_rows)


def count_parquet_rows(paths: Sequence[Path | str] | Path | str) -> int:
    total = 0
    for f in collect_parquet_files(paths):
        total += pq.ParquetFile(f).metadata.num_rows
    return total
