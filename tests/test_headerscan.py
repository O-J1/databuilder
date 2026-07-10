from __future__ import annotations

import pyarrow.parquet as pq

from databuilder.config import DatasetConfig
from databuilder.stages import headerscan

from conftest import save_image

DS = DatasetConfig(name="ds1", repo_id="org/x", label="fake", generator="folder")


def _setup_images(data_dir):
    save_image(data_dir / "ds1" / "gen_a" / "good.png", size=(400, 400))
    save_image(data_dir / "ds1" / "gen_a" / "small.png", size=(100, 100))
    save_image(data_dir / "ds1" / "gen_b" / "tall.png", size=(90, 300))
    save_image(data_dir / "ds1" / "gen_b" / "wide.png", size=(300, 90))
    broken = data_dir / "ds1" / "gen_b" / "broken.jpg"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"not an image at all")


def test_headerscan_filters_and_deletes(make_ctx):
    ctx = make_ctx(datasets=(DS,))
    _setup_images(ctx.data_dir)
    headerscan.run(ctx)

    kept = pq.read_table(ctx.artifact_dir("headerscan") / "rank_00000.kept.parquet").to_pylist()
    assert [row["path"] for row in kept] == ["ds1/gen_a/good.png"]
    assert kept[0]["generator"] == "gen_a"
    assert kept[0]["label"] == "fake"
    assert kept[0]["width"] == 400

    removed = pq.read_table(
        ctx.artifact_dir("headerscan") / "rank_00000.removed.parquet"
    ).to_pylist()
    reasons = {row["path"].rsplit("/", 1)[-1]: row["reason"] for row in removed}
    assert reasons == {
        "small.png": "too_small",
        "tall.png": "too_tall",
        "wide.png": "too_wide",
        "broken.jpg": "broken_header",
    }
    # losers are hard-deleted, the good file remains
    assert (ctx.data_dir / "ds1" / "gen_a" / "good.png").exists()
    assert not (ctx.data_dir / "ds1" / "gen_a" / "small.png").exists()
    assert not (ctx.data_dir / "ds1" / "gen_b" / "broken.jpg").exists()


def test_headerscan_dry_run_keeps_files(make_ctx):
    ctx = make_ctx(datasets=(DS,), dry_run=True)
    _setup_images(ctx.data_dir)
    headerscan.run(ctx)
    assert (ctx.data_dir / "ds1" / "gen_a" / "small.png").exists()
    assert (ctx.data_dir / "ds1" / "gen_b" / "broken.jpg").exists()
