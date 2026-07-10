from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from databuilder.utils import (
    ParquetShardWriter,
    count_parquet_rows,
    image_id,
    iter_parquet_batches,
    owns,
    sniff_extension,
    stable_hash64,
)


def test_stable_hash_deterministic():
    assert stable_hash64("ds1/a.png") == stable_hash64("ds1/a.png")
    assert stable_hash64("ds1/a.png") != stable_hash64("ds1/b.png")
    # windows/posix separators normalize to the same id
    assert image_id("ds1\\a.png") == image_id("ds1/a.png")


def test_owns_partitions_completely():
    world = 8
    paths = [f"ds/{i}.png" for i in range(500)]
    owners = [[p for p in paths if owns(p, r, world)] for r in range(world)]
    flattened = sorted(p for chunk in owners for p in chunk)
    assert flattened == sorted(paths)  # every path owned exactly once
    assert all(owners)  # sanity: work is actually spread


def test_sniff_extension():
    assert sniff_extension(b"\xff\xd8\xff\xe0rest") == ".jpg"
    assert sniff_extension(b"\x89PNG\r\n\x1a\nrest") == ".png"
    assert sniff_extension(b"RIFF1234WEBPrest") == ".webp"
    assert sniff_extension(b"garbage") == ".img"


def test_parquet_shard_writer_roundtrip(tmp_path):
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    path = tmp_path / "out.parquet"
    with ParquetShardWriter(path, schema, flush_rows=3) as writer:
        for i in range(10):
            writer.append({"a": i, "b": f"row{i}"})
    assert count_parquet_rows(path) == 10
    rows = [r for batch in iter_parquet_batches(path) for r in batch.to_pylist()]
    assert rows[0] == {"a": 0, "b": "row0"}
    assert pq.read_table(path).num_rows == 10
