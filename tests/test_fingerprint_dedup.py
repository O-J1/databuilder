from __future__ import annotations

import shutil

import pytest
import pyarrow.parquet as pq

from databuilder.config import DatasetConfig, DedupConfig, FiltersConfig
from databuilder.stages import dedup, fingerprint, headerscan

from conftest import save_image

pytest.importorskip("imagehash")
pytest.importorskip("cv2")

DS = DatasetConfig(name="ds1", repo_id="org/x", label="fake", generator="folder")
NO_LAPLACIAN = FiltersConfig(laplacian_min=0.0, laplacian_max=1e12)


def test_fingerprint_flags_flat_images(make_ctx):
    filters = FiltersConfig(laplacian_min=0.5, laplacian_max=1e12)
    ctx = make_ctx(datasets=(DS,), filters=filters)
    save_image(ctx.data_dir / "ds1" / "g" / "flat.png", kind="flat")
    save_image(ctx.data_dir / "ds1" / "g" / "textured.png", kind="noise")
    headerscan.run(ctx)
    fingerprint.run(ctx)

    kept = pq.read_table(ctx.artifact_dir("fingerprint") / "rank_00000.parquet").to_pylist()
    assert [row["path"] for row in kept] == ["ds1/g/textured.png"]
    assert len(kept[0]["md5"]) == 16
    assert len(kept[0]["phash"]) == 18  # 12x12 bits packed
    removed = pq.read_table(
        ctx.artifact_dir("fingerprint") / "rank_00000.removed.parquet"
    ).to_pylist()
    assert removed[0]["reason"] == "laplacian_low"
    assert not (ctx.data_dir / "ds1" / "g" / "flat.png").exists()


def test_dedup_keeps_highest_resolution(make_ctx):
    ctx = make_ctx(datasets=(DS,), filters=NO_LAPLACIAN)
    base = save_image(ctx.data_dir / "ds1" / "g" / "circle.png", kind="circle", size=(400, 400))
    # exact duplicate (same bytes, same md5)
    shutil.copyfile(base, ctx.data_dir / "ds1" / "g" / "circle_copy.png")
    # near duplicate at higher resolution (same structure -> close phash)
    save_image(ctx.data_dir / "ds1" / "g" / "circle_big.png", kind="circle", size=(600, 600))
    # unrelated image that must survive
    save_image(ctx.data_dir / "ds1" / "g" / "other.png", kind="noise", size=(400, 400))

    headerscan.run(ctx)
    fingerprint.run(ctx)
    dedup.run(ctx)

    survivors = pq.read_table(ctx.artifact_dir("dedup") / "survivors.parquet").to_pylist()
    paths = sorted(row["path"] for row in survivors)
    assert paths == ["ds1/g/circle_big.png", "ds1/g/other.png"]

    removed = pq.read_table(ctx.artifact_dir("dedup") / "removed.parquet").to_pylist()
    reasons = {row["path"].rsplit("/", 1)[-1] for row in removed}
    assert reasons == {"circle.png", "circle_copy.png"}
    # every removed row links to the surviving big circle (transitively for md5 losers)
    big_id = next(
        row["image_id"] for row in survivors if row["path"] == "ds1/g/circle_big.png"
    )
    assert all(row["kept_image_id"] == big_id for row in removed)
    assert all(row["image_id"] != row["kept_image_id"] for row in removed)
    assert not (ctx.data_dir / "ds1" / "g" / "circle.png").exists()
    assert (ctx.data_dir / "ds1" / "g" / "circle_big.png").exists()


def test_dedup_keep_removed_files(make_ctx):
    ctx = make_ctx(
        datasets=(DS,),
        filters=NO_LAPLACIAN,
        dedup=DedupConfig(keep_removed_files=True),
    )
    base = save_image(ctx.data_dir / "ds1" / "g" / "circle.png", kind="circle")
    shutil.copyfile(base, ctx.data_dir / "ds1" / "g" / "circle_copy.png")
    headerscan.run(ctx)
    fingerprint.run(ctx)
    dedup.run(ctx)

    removed = pq.read_table(ctx.artifact_dir("dedup") / "removed.parquet").to_pylist()
    assert len(removed) == 1
    assert (ctx.data_dir / "ds1" / "g" / "circle.png").exists()
    assert (ctx.data_dir / "ds1" / "g" / "circle_copy.png").exists()


def test_dedup_dry_run_keeps_files(make_ctx):
    ctx = make_ctx(datasets=(DS,), filters=NO_LAPLACIAN)
    base = save_image(ctx.data_dir / "ds1" / "g" / "circle.png", kind="circle")
    shutil.copyfile(base, ctx.data_dir / "ds1" / "g" / "circle_copy.png")
    headerscan.run(ctx)
    fingerprint.run(ctx)
    ctx.dry_run = True
    dedup.run(ctx)
    assert (ctx.data_dir / "ds1" / "g" / "circle.png").exists()
    assert (ctx.data_dir / "ds1" / "g" / "circle_copy.png").exists()


def test_dedup_protects_in_place_sources(tmp_path, make_ctx):
    src = tmp_path / "originals"
    local = DatasetConfig(
        name="src1", path=str(src), format="imagefolder", label="folder", generator="folder"
    )
    base = save_image(src / "fake" / "g" / "circle.png", kind="circle")
    shutil.copyfile(base, src / "fake" / "g" / "circle_copy.png")
    ctx = make_ctx(datasets=(local,), filters=NO_LAPLACIAN)
    headerscan.run(ctx)
    fingerprint.run(ctx)
    dedup.run(ctx)

    removed = pq.read_table(ctx.artifact_dir("dedup") / "removed.parquet").to_pylist()
    assert len(removed) == 1
    # the duplicate is recorded but the original source file stays on disk
    assert (src / "fake" / "g" / "circle.png").exists()
    assert (src / "fake" / "g" / "circle_copy.png").exists()
