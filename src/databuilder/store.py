from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .utils import collect_parquet_files, count_parquet_rows, iter_parquet_batches

# Embedding shards are plain parquet files. This module is the seam to swap in
# another backend (e.g. postgresql+pgvector) later without touching the stages.


def embedding_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            ("image_id", pa.uint64()),
            ("path", pa.string()),
            ("embedding", pa.list_(pa.float16(), dim)),
        ]
    )


class EmbeddingShardWriter:
    """Streaming writer for one embedding parquet shard."""

    def __init__(self, path: Path | str, dim: int, flush_rows: int = 20_000):
        self.path = Path(path)
        self.dim = dim
        self.flush_rows = flush_rows
        self.rows_written = 0
        self._ids: list[np.ndarray] = []
        self._paths: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._buffered = 0
        self._writer: pq.ParquetWriter | None = None

    def append_batch(self, ids: np.ndarray, paths: list[str], vectors: np.ndarray) -> None:
        if vectors.shape != (len(paths), self.dim):
            raise ValueError(f"expected {(len(paths), self.dim)}, got {vectors.shape}")
        self._ids.append(np.asarray(ids, dtype=np.uint64))
        self._paths.extend(paths)
        self._vectors.append(np.asarray(vectors, dtype=np.float16))
        self._buffered += len(paths)
        if self._buffered >= self.flush_rows:
            self._flush()

    def _flush(self) -> None:
        if not self._buffered:
            return
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, embedding_schema(self.dim))
        flat = np.concatenate(self._vectors).reshape(-1)
        table = pa.Table.from_arrays(
            [
                pa.array(np.concatenate(self._ids), type=pa.uint64()),
                pa.array(self._paths, type=pa.string()),
                pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float16()), self.dim),
            ],
            schema=embedding_schema(self.dim),
        )
        self._writer.write_table(table)
        self.rows_written += self._buffered
        self._ids, self._paths, self._vectors, self._buffered = [], [], [], 0

    def close(self) -> None:
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "EmbeddingShardWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ParquetEmbeddingStore:
    """Read-side access to a directory of embedding parquet shards."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    def count(self) -> int:
        return count_parquet_rows(self.directory)

    def dim(self) -> int:
        files = collect_parquet_files(self.directory)
        if not files:
            raise FileNotFoundError(f"No embedding shards under {self.directory}")
        field = pq.ParquetFile(files[0]).schema_arrow.field("embedding")
        return field.type.list_size

    def iter_batches(
        self, batch_rows: int = 8192, with_paths: bool = False
    ) -> Iterator[tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[str]]]:
        """Yield (ids, float32 matrix[, paths]) batches without loading everything."""
        columns = ["image_id", "path", "embedding"] if with_paths else ["image_id", "embedding"]
        for batch in iter_parquet_batches(self.directory, columns=columns, batch_rows=batch_rows):
            ids = batch.column("image_id").to_numpy(zero_copy_only=False).astype(np.uint64)
            emb_col = batch.column("embedding")
            dim = emb_col.type.list_size
            flat = emb_col.flatten().to_numpy(zero_copy_only=False)
            matrix = flat.astype(np.float32).reshape(-1, dim)
            if with_paths:
                yield ids, matrix, batch.column("path").to_pylist()
            else:
                yield ids, matrix
