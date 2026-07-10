from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from databuilder.config import ClusteringConfig, ConfigError, DatasetConfig
from databuilder.stages import cluster, headerscan, manifest
from databuilder.stages.cluster import semdedup_prune
from databuilder.stages.common import (
    label_from_value,
    resolve_meta,
    validate_local_datasets,
)

from conftest import build_synthetic_artifacts, save_image


def _local_ds(path: Path, **kwargs) -> DatasetConfig:
    defaults = dict(
        name="local1", path=str(path), format="imagefolder", label="folder",
        generator="folder",
    )
    defaults.update(kwargs)
    return DatasetConfig(**defaults)


def test_resolve_meta_folder_labels(tmp_path):
    ds = _local_ds(tmp_path)
    assert resolve_meta(ds, ("real", "img.png")) == ("real", "local1")
    assert resolve_meta(ds, ("fake", "sd15", "img.png")) == ("fake", "sd15")
    assert resolve_meta(ds, ("train", "generated", "img.png"))[0] == "fake"
    assert resolve_meta(ds, ("misc", "img.png"))[0] == "unknown"


def test_resolve_meta_label_map(tmp_path):
    ds = _local_ds(tmp_path, label_map={"photos": "real", "sd15": "fake"})
    assert resolve_meta(ds, ("photos", "img.png")) == ("real", "local1")
    # sd15 maps to a label, so it must not be used as generator either
    label, generator = resolve_meta(ds, ("sd15", "img.png"))
    assert label == "fake"
    assert generator == "local1"
    assert label_from_value(ds, "PHOTOS") == "real"


def test_validate_local_datasets_fail_fast(tmp_path, make_ctx):
    (tmp_path / "src" / "stuff").mkdir(parents=True)
    ctx = make_ctx(datasets=(_local_ds(tmp_path / "src"),))
    with pytest.raises(ConfigError, match="cannot infer labels"):
        validate_local_datasets(ctx.cfg)

    (tmp_path / "src" / "fake").mkdir()
    validate_local_datasets(ctx.cfg)  # now passes

    missing = make_ctx(datasets=(_local_ds(tmp_path / "nope"),))
    with pytest.raises(ConfigError, match="not a directory"):
        validate_local_datasets(missing.cfg)


def test_headerscan_local_in_place(tmp_path, make_ctx):
    src = tmp_path / "originals"
    save_image(src / "real" / "photo.png", size=(400, 400))
    save_image(src / "fake" / "sd15" / "gen.png", size=(400, 400))
    save_image(src / "fake" / "sd15" / "small.png", size=(100, 100))
    ctx = make_ctx(datasets=(_local_ds(src),))
    headerscan.run(ctx)

    kept = pq.read_table(ctx.artifact_dir("headerscan") / "rank_00000.kept.parquet").to_pylist()
    by_path = {row["path"]: row for row in kept}
    assert by_path["local1/real/photo.png"]["label"] == "real"
    assert by_path["local1/fake/sd15/gen.png"]["label"] == "fake"
    assert by_path["local1/fake/sd15/gen.png"]["generator"] == "sd15"
    # in-place sources are protected: the reject is recorded but stays on disk
    removed = pq.read_table(
        ctx.artifact_dir("headerscan") / "rank_00000.removed.parquet"
    ).to_pylist()
    assert [row["reason"] for row in removed] == ["too_small"]
    assert (src / "fake" / "sd15" / "small.png").exists()
    assert (src / "real" / "photo.png").exists()


def test_headerscan_local_allow_delete_opts_in(tmp_path, make_ctx):
    src = tmp_path / "originals"
    save_image(src / "real" / "photo.png", size=(400, 400))
    save_image(src / "real" / "small.png", size=(100, 100))
    ctx = make_ctx(datasets=(_local_ds(src, allow_delete=True),))
    headerscan.run(ctx)
    # deletion hit the ORIGINAL file, in place
    assert not (src / "real" / "small.png").exists()
    assert (src / "real" / "photo.png").exists()


def test_headerscan_unknown_label_raises(tmp_path, make_ctx):
    src = tmp_path / "originals"
    save_image(src / "real" / "ok.png", size=(400, 400))
    save_image(src / "misc" / "stray.png", size=(400, 400))
    ctx = make_ctx(datasets=(_local_ds(src),))
    with pytest.raises(RuntimeError, match="cannot infer a label"):
        headerscan.run(ctx)


def test_semdedup_exempt_rows_never_pruned():
    emb = np.ones((6, 4), dtype=np.float32)  # every member identical
    dists = np.zeros(6, dtype=np.float32)
    ids = np.arange(6, dtype=np.uint64)
    exempt = np.array([True, True, False, False, False, False])
    pruned = semdedup_prune(emb, dists, ids, threshold=0.9, exempt=exempt)
    assert not pruned[exempt].any()
    # exempt anchors dedupe every remaining identical member
    assert pruned[~exempt].all()


SK2 = ClusteringConfig(backend="sklearn", k=2)


def test_manifest_forced_split_and_absolute_paths(tmp_path, make_ctx):
    pytest.importorskip("sklearn")
    src = tmp_path / "originals"
    src.mkdir()
    ds = DatasetConfig(
        name="ds1",
        path=str(src),
        format="imagefolder",
        label="fake",
        assign_split="test",
    )
    ctx = make_ctx(datasets=(ds,), clustering=SK2)
    build_synthetic_artifacts(ctx)  # writes survivors/embeddings for dataset 'ds1'
    cluster.run(ctx)
    manifest.run(ctx)

    rows = pq.read_table(ctx.artifact_dir("manifest") / "manifest.parquet").to_pylist()
    assert len(rows) == 40  # forced datasets bypass balancing caps entirely
    assert all(row["split"] == "test" for row in rows)
    assert all(Path(row["path"]).is_absolute() for row in rows)
    assert rows[0]["path"].startswith(str(src))
