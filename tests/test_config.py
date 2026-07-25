from __future__ import annotations

from pathlib import Path

import pytest

from databuilder.config import (
    CSV_MAX_ROWS,
    ClusteringConfig,
    ConfigError,
    DatasetConfig,
    load_config,
    parse_ratio,
)

MINIMAL = """
[runtime]
work_dir = "{work}"
data_dir = "{data}"

[[datasets]]
name = "ds1"
repo_id = "org/name"
label = "fake"
"""


def _write(tmp_path, text: str):
    path = tmp_path / "build.toml"
    path.write_text(
        text.format(work=(tmp_path / "w").as_posix(), data=(tmp_path / "d").as_posix()),
        encoding="utf-8",
    )
    return path


def test_minimal_config_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.filters.min_longest_side == 256
    assert cfg.dedup.phash_size == 12
    assert cfg.clustering.backend == "auto"
    assert cfg.balance.emit_csv is True
    assert cfg.daft.enabled is False
    assert cfg.daft.runner == "native"
    assert cfg.embedding.concurrency == 0
    assert cfg.download.max_workers == 1
    assert cfg.download.xet_high_performance is False
    assert cfg.datasets[0].name == "ds1"
    assert CSV_MAX_ROWS == 1_000_000


def test_daft_section(tmp_path):
    text = MINIMAL + '\n[daft]\nenabled = true\nrunner = "ray"\nray_address = "ray://head:10001"\n'
    cfg = load_config(_write(tmp_path, text))
    assert cfg.daft.enabled is True
    assert cfg.daft.runner == "ray"
    assert cfg.daft.ray_address == "ray://head:10001"


def test_daft_bad_runner_rejected(tmp_path):
    text = MINIMAL + '\n[daft]\nrunner = "spark"\n'
    with pytest.raises(ConfigError, match="daft.runner"):
        load_config(_write(tmp_path, text))


def test_negative_embedding_concurrency_rejected(tmp_path):
    text = MINIMAL + "\n[embedding]\nconcurrency = -1\n"
    with pytest.raises(ConfigError, match="concurrency"):
        load_config(_write(tmp_path, text))


def test_runtime_overrides(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL), overrides={"rank": 3, "world_size": 8})
    assert cfg.runtime.rank == 3
    assert cfg.runtime.world_size == 8


def test_bad_rank_rejected(tmp_path):
    with pytest.raises(ConfigError, match="rank"):
        load_config(_write(tmp_path, MINIMAL), overrides={"rank": 2, "world_size": 2})


def test_unknown_key_rejected(tmp_path):
    text = MINIMAL + "\n[filters]\nmin_longest = 5\n"
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(_write(tmp_path, text))


def test_bad_ratio_rejected():
    with pytest.raises(ConfigError):
        parse_ratio("nonsense", "filters.max_tall")
    assert parse_ratio("9:23") == pytest.approx(9 / 23)


def test_dataset_label_validation():
    with pytest.raises(ConfigError, match="label"):
        DatasetConfig(name="x", repo_id="a/b", label="maybe")
    ds = DatasetConfig(name="x", repo_id="a/b", label="column:cls", generator="column:model")
    assert ds.label_column == "cls"
    assert ds.generator_column == "model"
    assert ds.columns == {"label": "cls", "generator": "model"}


def test_dataset_source_exclusivity():
    with pytest.raises(ConfigError, match="exactly one"):
        DatasetConfig(name="x", label="fake")
    with pytest.raises(ConfigError, match="exactly one"):
        DatasetConfig(name="x", repo_id="a/b", path="/data", label="fake")


def test_local_dataset_requires_explicit_format():
    with pytest.raises(ConfigError, match="explicit format"):
        DatasetConfig(name="x", path="/data", label="folder")
    ds = DatasetConfig(name="x", path="/data", format="imagefolder", label="folder")
    assert ds.is_local and ds.in_place
    assert not ds.allow_delete  # in-place sources are protected by default
    ds2 = DatasetConfig(name="y", path="/data", format="parquet", label="auto")
    assert ds2.is_local and not ds2.in_place
    opted = DatasetConfig(
        name="z", path="/data", format="imagefolder", label="folder", allow_delete=True
    )
    assert opted.allow_delete


def test_clustering_prune_validation():
    with pytest.raises(ConfigError, match="prune_trigger_sigma"):
        ClusteringConfig(prune_trigger_sigma=-1.0)
    with pytest.raises(ConfigError, match="semdedup_threshold"):
        ClusteringConfig(semdedup_threshold=0.0)
    with pytest.raises(ConfigError, match="semdedup_threshold"):
        ClusteringConfig(semdedup_threshold=1.5)
    cfg = ClusteringConfig()
    assert cfg.prune_trigger_sigma == 3.0
    assert cfg.semdedup_threshold == 0.96


def test_assign_split_validation():
    with pytest.raises(ConfigError, match="assign_split"):
        DatasetConfig(name="x", repo_id="a/b", label="fake", assign_split="holdout")
    ds = DatasetConfig(name="x", repo_id="a/b", label="fake", assign_split="test")
    assert ds.assign_split == "test"


def test_columns_table_validation():
    with pytest.raises(ConfigError, match="columns"):
        DatasetConfig(name="x", repo_id="a/b", label="fake", columns={"thumbnail": "t"})
    ds = DatasetConfig(
        name="x", repo_id="a/b", label="fake", columns={"image": "jpeg", "split": "part"}
    )
    assert ds.image_column == "jpeg"
    assert ds.split_column == "part"


def test_label_map_validation():
    with pytest.raises(ConfigError, match="label_map"):
        DatasetConfig(name="x", repo_id="a/b", label="folder", label_map={"pics": "genuine"})
    ds = DatasetConfig(
        name="x", repo_id="a/b", label="folder", label_map={"Pics": "real", "SD15": "fake"}
    )
    assert ds.label_map == {"pics": "real", "sd15": "fake"}


def test_old_split_key_gives_guidance(tmp_path):
    text = MINIMAL.replace('label = "fake"', 'label = "fake"\nsplit = "train"')
    with pytest.raises(ConfigError, match="source_split"):
        load_config(_write(tmp_path, text))


def test_max_label_ratio_validation(tmp_path):
    text = MINIMAL + "\n[balance]\nmax_label_ratio = -0.5\n"
    with pytest.raises(ConfigError, match="max_label_ratio"):
        load_config(_write(tmp_path, text))
    ok = MINIMAL + "\n[balance]\nmax_label_ratio = 1.5\n"
    assert load_config(_write(tmp_path, ok)).balance.max_label_ratio == 1.5


def test_duplicate_dataset_names_rejected(tmp_path):
    text = MINIMAL + '\n[[datasets]]\nname = "ds1"\nrepo_id = "org/other"\nlabel = "real"\n'
    with pytest.raises(ConfigError, match="unique"):
        load_config(_write(tmp_path, text))


def test_aspect_ratio_filter_semantics(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    # taller than 9:23 -> ratio below tall_ratio; wider than 23:9 -> above wide_ratio
    assert 90 / 300 < cfg.filters.tall_ratio
    assert 300 / 90 > cfg.filters.wide_ratio
    assert cfg.filters.tall_ratio < 1 < cfg.filters.wide_ratio


def test_download_only_requires_raw():
    with pytest.raises(ConfigError, match="download_only"):
        DatasetConfig(
            name="metadata", repo_id="org/metadata", format="parquet",
            download_only=True, label="unknown"
        )
    ds = DatasetConfig(
        name="metadata", repo_id="org/metadata", format="raw",
        download_only=True, label="unknown"
    )
    assert ds.download_only
    with pytest.raises(ConfigError, match="requires download_only"):
        DatasetConfig(name="raw", repo_id="org/raw", format="raw", label="unknown")


def test_aigc_production_config_is_pinned_and_snapshot_only():
    path = Path(__file__).resolve().parents[1] / "examples" / "aigc-datasets.toml"
    cfg = load_config(path)
    assert cfg.runtime.data_dir.as_posix() == "/p/data1/datasets/mmlaion/aigc/data"
    assert cfg.download.max_workers == 1
    assert cfg.download.xet_high_performance is True
    assert len(cfg.datasets) == 52
    assert len({ds.repo_id for ds in cfg.datasets}) == 41
    assert all(ds.revision for ds in cfg.datasets)
    raw = {ds.name for ds in cfg.datasets if ds.download_only}
    assert raw == {"dim-t2i", "anycrap", "seaart-hq"}
    assert all(ds.format == "raw" for ds in cfg.datasets if ds.download_only)
    assert not any(image.kind == "url" for ds in cfg.datasets for image in ds.images)

    sdaie = [ds for ds in cfg.datasets if ds.repo_id == "ThaneJoss/SDAIE"]
    assert len(sdaie) == 4
    patterns = [pattern for ds in sdaie for pattern in ds.allow_patterns]
    assert len(patterns) == len(set(patterns))

    genimage = [ds for ds in cfg.datasets if ds.repo_id == "shimei123/Genimage"]
    assert len(genimage) == 8
    assert {pattern for ds in genimage for pattern in ds.allow_patterns} == {
        "ADM.zip", "BigGAN.zip", "Midjourney.zip", "SD_v14.zip",
        "SD_v15.zip", "VQDM.zip", "glide.zip", "wukong.zip",
    }
    assert all(ds.label_map == {"ai": "fake", "nature": "real"} for ds in genimage)

    so_fake = next(ds for ds in cfg.datasets if ds.name == "so-fake-set")
    assert so_fake.label_map == {
        "real": "real", "full_synthetic": "fake", "tampered": "fake"
    }

    rr_test = next(ds for ds in cfg.datasets if ds.name == "rrdataset-redigital-test")
    assert rr_test.assign_split == "test"
    assert rr_test.label_map == {"ai": "fake", "real": "real"}

    so_fake_ood = next(ds for ds in cfg.datasets if ds.name == "so-fake-ood")
    assert so_fake_ood.assign_split == "test"
    assert so_fake_ood.label_map["0"] == "real"
    assert so_fake_ood.label_map["1"] == "fake"
    assert so_fake_ood.label_map["2"] == "fake"
