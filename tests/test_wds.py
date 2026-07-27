from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import save_image
from databuilder.config import DatasetConfig, FiltersConfig, StorageConfig
from databuilder.stages import download as download_stage
from databuilder.stages import fingerprint, headerscan
from databuilder.stages.download import inventory, migrate_existing
from databuilder.stages.manifest import MANIFEST_SCHEMA, _compact_and_rewrite
from databuilder.utils import image_id
from databuilder.wds import (
    DatasetShardWriter,
    ImageRef,
    atomic_json,
    compact_dataset,
    is_webdataset,
    iter_index,
)


def test_shards_rotate_and_index_exact_byte_ranges(tmp_path):
    source = save_image(tmp_path / "source.png", size=(32, 32))
    payload = source.read_bytes()
    root = tmp_path / "dataset"
    writer = DatasetShardWriter(root, "dataset", max_samples_per_shard=2)
    expected = {}
    for index in range(3):
        logical = f"dataset/fake/generator/image-{index}.png"
        expected[logical] = payload
        writer.add(payload, logical, "fake", "generator")
    result = writer.finalize()

    rows = list(iter_index(root))
    assert result["shards"] == 2
    assert len(rows) == 3
    assert is_webdataset(root)
    assert not list(root.rglob("*.png"))
    assert all(
        ImageRef.from_row(row).read_bytes({"dataset": str(root)}) == expected[row["path"]]
        for row in rows
    )


def test_writer_recovers_committed_journal_without_duplicate(tmp_path):
    payload = save_image(tmp_path / "source.png", size=(32, 32)).read_bytes()
    root = tmp_path / "dataset"
    logical = "dataset/fake/generator/first.png"
    first = DatasetShardWriter(root, "dataset")
    first.add(payload, logical, "fake", "generator")
    first.close()

    resumed = DatasetShardWriter(root, "dataset")
    assert resumed.contains(logical)
    assert not resumed.add(payload, logical, "fake", "generator")
    resumed.add(payload, "dataset/fake/generator/second.png", "fake", "generator")
    resumed.finalize()
    assert len(list(iter_index(root))) == 2


def test_existing_loose_dataset_migrates_offline_and_deletes_after_commit(
    tmp_path, make_ctx
):
    ds = DatasetConfig(name="legacy", repo_id="org/legacy", label="fake", generator="gen")
    ctx = make_ctx(
        datasets=(ds,),
        storage=StorageConfig(target_shard_bytes=1_048_576, max_samples_per_shard=1),
    )
    loose = save_image(ctx.data_dir / "legacy" / "image.png", size=(32, 32))
    payload = loose.read_bytes()
    (ctx.data_dir / "legacy" / ".materialized.json").write_text(
        json.dumps({"format": "imagefolder", "written": 1}), encoding="utf-8"
    )

    result = migrate_existing(ctx)

    assert result[0]["status"] == "migrated"
    assert not loose.exists()
    rows = list(iter_index(ctx.data_dir / "legacy"))
    assert rows[0]["image_id"] == image_id("legacy/image.png")
    assert ImageRef.from_row(rows[0]).read_bytes(
        {"legacy": str(ctx.data_dir / "legacy")}
    ) == payload
    assert inventory(ctx)[0]["status"] == "webdataset_complete"


def test_compaction_rewrites_offsets_and_removes_rejected_samples(tmp_path):
    payload = save_image(tmp_path / "source.png", size=(32, 32)).read_bytes()
    root = tmp_path / "dataset"
    writer = DatasetShardWriter(root, "dataset", max_samples_per_shard=2)
    paths = [f"dataset/fake/generator/image-{index}.png" for index in range(3)]
    for logical in paths:
        writer.add(payload, logical, "fake", "generator")
    writer.finalize()

    stats = compact_dataset(root, {paths[0], paths[2]})

    rows = list(iter_index(root))
    assert stats["removed"] == 1
    assert {row["path"] for row in rows} == {paths[0], paths[2]}
    assert all(
        ImageRef.from_row(row).read_bytes({"dataset": str(root)}) == payload for row in rows
    )
    assert is_webdataset(root)


def test_headerscan_and_fingerprint_read_archived_images(tmp_path, make_ctx):
    ds = DatasetConfig(name="archived", repo_id="org/archived", label="fake", generator="gen")
    ctx = make_ctx(
        datasets=(ds,),
        filters=FiltersConfig(laplacian_min=0.0, laplacian_max=1_000_000.0),
    )
    payload = save_image(tmp_path / "source.png", size=(300, 300), kind="noise").read_bytes()
    writer = DatasetShardWriter(ctx.data_dir / "archived", "archived")
    writer.add(payload, "archived/image.png", "fake", "gen")
    writer.finalize()

    headerscan.run(ctx)
    fingerprint.run(ctx)

    header_row = pq.read_table(
        ctx.artifact_dir("headerscan") / "rank_00000.kept.parquet"
    ).to_pylist()[0]
    fingerprint_row = pq.read_table(
        ctx.artifact_dir("fingerprint") / "rank_00000.parquet"
    ).to_pylist()[0]
    assert header_row["shard"] and header_row["member"]
    assert fingerprint_row["shard"] == header_row["shard"]
    assert fingerprint_row["offset"] == header_row["offset"]


def test_compacting_state_never_triggers_a_redownload(tmp_path, make_ctx):
    ds = DatasetConfig(name="dataset", repo_id="org/dataset", label="fake")
    ctx = make_ctx(datasets=(ds,))
    root = ctx.data_dir / "dataset"
    writer = DatasetShardWriter(root, "dataset")
    logical = "dataset/image.png"
    writer.add(b"not-decoded-here", logical, "fake", "dataset")
    writer.finalize()
    marker_path = root / ".materialized.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["state"] = "compacting"
    atomic_json(marker_path, marker)

    assert inventory(ctx)[0]["status"] == "webdataset_compacting"
    assert migrate_existing(ctx)[0]["status"] == "compaction_in_progress"
    with pytest.raises(RuntimeError, match="interrupted shard compaction"):
        download_stage.run(ctx)

    compact_dataset(root, {logical})
    assert is_webdataset(root)


def test_manifest_compaction_resumes_and_rewrites_locators(tmp_path, make_ctx):
    ds = DatasetConfig(name="dataset", repo_id="org/dataset", label="fake")
    ctx = make_ctx(datasets=(ds,))
    root = ctx.data_dir / "dataset"
    writer = DatasetShardWriter(root, "dataset", max_samples_per_shard=3)
    paths = [f"dataset/image-{index}.png" for index in range(3)]
    for index, logical in enumerate(paths):
        writer.add(bytes([index]) * 32, logical, "fake", "dataset")
    writer.finalize()
    indexed = {row["path"]: row for row in iter_index(root)}
    selected = paths[:2]
    records = []
    for logical in selected:
        locator = indexed[logical]
        records.append(
            {
                "path": logical,
                "label": 1,
                "split": "train",
                "generator": "dataset",
                "source_dataset": "dataset",
                "width": 1,
                "height": 1,
                "cluster_id": 0,
                "image_id": image_id(logical),
                "file_hash": "0" * 16,
                "laplacian": 1.0,
                "shard": locator["shard"],
                "member": locator["member"],
                "offset": locator["offset"],
                "size": locator["size"],
            }
        )
    manifest = ctx.artifact_dir("manifest") / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(records, schema=MANIFEST_SCHEMA), manifest)
    marker_path = root / ".materialized.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["state"] = "compacting"
    atomic_json(marker_path, marker)

    _compact_and_rewrite(ctx, manifest, keep_maps=False)

    new_index = {row["path"]: row for row in iter_index(root)}
    manifest_rows = pq.read_table(manifest).to_pylist()
    assert set(new_index) == set(selected)
    assert all(row["offset"] == new_index[row["path"]]["offset"] for row in manifest_rows)
    assert is_webdataset(root)
