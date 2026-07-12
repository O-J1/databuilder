from __future__ import annotations

import pytest
import pyarrow.parquet as pq

from databuilder.config import DaftConfig, DatasetConfig, FiltersConfig
from databuilder.stages import headerscan

from conftest import save_image

DS = DatasetConfig(name="ds1", repo_id="org/x", label="fake", generator="folder")
NO_LAPLACIAN = FiltersConfig(laplacian_min=0.0, laplacian_max=1e12)
DAFT_NATIVE = DaftConfig(enabled=True, runner="native")


def test_rank0_stage_topology_with_ray(make_ctx):
    ray_ctx = make_ctx(daft=DaftConfig(enabled=True, runner="ray"))
    assert ray_ctx.is_rank0_stage("fingerprint")
    assert ray_ctx.is_rank0_stage("embed")
    assert ray_ctx.is_rank0_stage("dedup")
    assert not ray_ctx.is_rank0_stage("headerscan")
    assert ray_ctx.expected_ranks("fingerprint") == 1

    native_ctx = make_ctx(daft=DAFT_NATIVE)
    assert not native_ctx.is_rank0_stage("fingerprint")
    assert not native_ctx.is_rank0_stage("embed")
    assert native_ctx.is_rank0_stage("dedup")

    off_ctx = make_ctx()
    assert not off_ctx.is_rank0_stage("fingerprint")


def test_daft_fingerprint_writes_same_artifacts(make_ctx):
    pytest.importorskip("daft")
    from databuilder.stages import fingerprint

    ctx = make_ctx(datasets=(DS,), filters=NO_LAPLACIAN, daft=DAFT_NATIVE)
    save_image(ctx.data_dir / "ds1" / "g" / "a.png", kind="noise")
    save_image(ctx.data_dir / "ds1" / "g" / "b.png", kind="circle")
    headerscan.run(ctx)
    fingerprint.run(ctx)

    kept = pq.read_table(ctx.artifact_dir("fingerprint") / "rank_00000.parquet").to_pylist()
    assert sorted(row["path"] for row in kept) == ["ds1/g/a.png", "ds1/g/b.png"]
    for row in kept:
        assert isinstance(row["file_hash"], int)
        assert len(row["phash"]) == 18  # 12x12 bits packed
        assert len(row["colorhash"]) > 0
        assert row["laplacian"] > 0.0


def test_daft_fingerprint_same_bytes_same_hash(make_ctx):
    pytest.importorskip("daft")
    import shutil

    from databuilder.stages import fingerprint

    ctx = make_ctx(datasets=(DS,), filters=NO_LAPLACIAN, daft=DAFT_NATIVE)
    base = save_image(ctx.data_dir / "ds1" / "g" / "one.png", kind="circle")
    shutil.copyfile(base, ctx.data_dir / "ds1" / "g" / "two.png")
    headerscan.run(ctx)
    fingerprint.run(ctx)

    kept = pq.read_table(ctx.artifact_dir("fingerprint") / "rank_00000.parquet").to_pylist()
    hashes = {row["path"]: row["file_hash"] for row in kept}
    assert hashes["ds1/g/one.png"] == hashes["ds1/g/two.png"]


def test_daft_fingerprint_filters_and_deletes(make_ctx):
    pytest.importorskip("daft")
    from databuilder.stages import fingerprint

    filters = FiltersConfig(laplacian_min=0.5, laplacian_max=1e12)
    ctx = make_ctx(datasets=(DS,), filters=filters, daft=DAFT_NATIVE)
    save_image(ctx.data_dir / "ds1" / "g" / "flat.png", kind="flat")
    save_image(ctx.data_dir / "ds1" / "g" / "textured.png", kind="noise")
    headerscan.run(ctx)
    fingerprint.run(ctx)

    kept = pq.read_table(ctx.artifact_dir("fingerprint") / "rank_00000.parquet").to_pylist()
    assert [row["path"] for row in kept] == ["ds1/g/textured.png"]
    removed = pq.read_table(
        ctx.artifact_dir("fingerprint") / "rank_00000.removed.parquet"
    ).to_pylist()
    assert removed[0]["reason"] == "laplacian_low"
    assert not (ctx.data_dir / "ds1" / "g" / "flat.png").exists()


def test_daft_fingerprint_feeds_dedup(make_ctx):
    pytest.importorskip("daft")
    import shutil

    from databuilder.stages import dedup, fingerprint

    ctx = make_ctx(datasets=(DS,), filters=NO_LAPLACIAN, daft=DAFT_NATIVE)
    base = save_image(ctx.data_dir / "ds1" / "g" / "circle.png", kind="circle", size=(400, 400))
    shutil.copyfile(base, ctx.data_dir / "ds1" / "g" / "circle_copy.png")
    save_image(ctx.data_dir / "ds1" / "g" / "other.png", kind="noise", size=(400, 400))
    headerscan.run(ctx)
    fingerprint.run(ctx)
    dedup.run(ctx)

    survivors = pq.read_table(ctx.artifact_dir("dedup") / "survivors.parquet").to_pylist()
    paths = sorted(row["path"] for row in survivors)
    assert "ds1/g/other.png" in paths
    assert len(paths) == 2  # one circle survives, the exact copy is removed
    removed = pq.read_table(ctx.artifact_dir("dedup") / "removed.parquet").to_pylist()
    assert len(removed) == 1
    assert removed[0]["reason"] == "duplicate_exact"