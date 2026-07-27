from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from databuilder.config import ConfigError, DatasetConfig, DownloadConfig, ImageColumnConfig
from databuilder.stages import download
from databuilder.stages.common import MATERIALIZED_MARKER, load_layout, pipeline_datasets
from databuilder.stages.download import resolve_columns
from databuilder.wds import ImageRef, is_webdataset, iter_index


def _schema(**fields) -> pa.Schema:
    return pa.schema(list(fields.items()))


HF_IMAGE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def test_resolve_columns_automatch():
    ds = DatasetConfig(name="x", repo_id="a/b", label="auto")
    cmap = resolve_columns(
        ds, _schema(image=HF_IMAGE, label=pa.string(), model=pa.string())
    )
    assert (cmap.image, cmap.image_kind) == ("image", "struct")
    assert cmap.label == "label"
    assert cmap.generator == "model"  # optional, automatched


def test_resolve_columns_requires_explicit_when_odd_names():
    ds = DatasetConfig(name="x", repo_id="a/b", label="auto")
    schema = _schema(jpeg_data=pa.binary(), category_v2=pa.string())
    with pytest.raises(ConfigError, match=r'columns\.label = "<name>"'):
        resolve_columns(ds, schema)

    explicit = DatasetConfig(
        name="x",
        repo_id="a/b",
        label="auto",
        columns={"image": "jpeg_data", "label": "category_v2"},
    )
    cmap = resolve_columns(explicit, schema)
    assert (cmap.image, cmap.image_kind) == ("jpeg_data", "binary")
    assert cmap.label == "category_v2"
    assert cmap.generator is None


def test_resolve_columns_missing_image_errors():
    ds = DatasetConfig(name="x", repo_id="a/b", label="fake")
    with pytest.raises(ConfigError, match=r'columns\.image'):
        resolve_columns(ds, _schema(text=pa.string()))


def test_resolve_columns_rejects_folder_label_for_parquet():
    ds = DatasetConfig(name="x", repo_id="a/b", label="folder")
    with pytest.raises(ConfigError, match="folder"):
        resolve_columns(ds, _schema(image=HF_IMAGE))


def _png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (300, 300), (200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _wds_rows(target: Path) -> list[dict]:
    assert is_webdataset(target)
    assert not any(
        path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in target.rglob("*")
    )
    return list(iter_index(target))


def test_materialize_local_parquet_with_layout_marker(tmp_path, make_ctx):
    source = tmp_path / "parquet-src"
    source.mkdir()
    png = _png_bytes()
    table = pa.table(
        {
            "picture": pa.array([png, png], type=pa.binary()),
            "class": ["real", "fake"],
            "model_name": ["camera", "sd15"],
        }
    )
    pq.write_table(table, source / "part0.parquet")

    ds = DatasetConfig(name="localpq", path=str(source), format="parquet", label="auto")
    ctx = make_ctx(datasets=(ds,))
    download.run(ctx)

    target = ctx.data_dir / "localpq"
    rows = _wds_rows(target)
    written = sorted(row["path"].partition("/")[2] for row in rows)
    assert written == [
        "fake/sd15/localpq_00000_000000001.png",
        "real/camera/localpq_00000_000000000.png",
    ]
    assert all(ImageRef.from_row(row).read_bytes({"localpq": str(target)}) == png for row in rows)
    # local source parquet is never deleted
    assert (source / "part0.parquet").exists()
    # automatched layout is recorded and read back for resolve_meta
    marker = json.loads((target / MATERIALIZED_MARKER).read_text(encoding="utf-8"))
    assert marker["layout"] == {"label_dir": True, "generator_dir": True}
    layout = load_layout(ctx.cfg, ds)
    assert layout.label_dir and layout.generator_dir


def test_null_dynamic_generator_materializes_as_unknown(tmp_path, make_ctx):
    source = tmp_path / "nullable-generator"
    source.mkdir()
    pq.write_table(
        pa.table(
            {
                "image": pa.array([_png_bytes()], type=pa.binary()),
                "label": ["real"],
                "generator": pa.array([None], type=pa.string()),
            }
        ),
        source / "rows.parquet",
    )
    ds = DatasetConfig(
        name="nullable-generator",
        path=str(source),
        format="parquet",
        label="column:label",
        generator="column:generator",
        columns={"image": "image"},
    )
    ctx = make_ctx(datasets=(ds,))
    download.run(ctx)

    target = ctx.data_dir / "nullable-generator"
    written = [row["path"].partition("/")[2] for row in _wds_rows(target)]
    assert written == ["real/unknown/nullable-generator_00000_000000000.png"]


def test_materialize_parquet_excludes_configured_rows_case_insensitively(
    tmp_path, make_ctx
):
    source = tmp_path / "excluded-models"
    source.mkdir()
    png = _png_bytes()
    pq.write_table(
        pa.table(
            {
                "image": pa.array([png, png, png], type=pa.binary()),
                "label": ["fake", "fake", "fake"],
                "model": ["good-model", "flux.1-dev", "sana"],
            }
        ),
        source / "rows.parquet",
    )
    ds = DatasetConfig(
        name="excluded-models",
        path=str(source),
        format="parquet",
        label="column:label",
        generator="column:model",
        columns={"image": "image"},
        row_exclude={"model": ["flux.1-dev", "SANA"]},
    )
    ctx = make_ctx(datasets=(ds,))
    download.run(ctx)

    target = ctx.data_dir / "excluded-models"
    written = [row["path"].partition("/")[2] for row in _wds_rows(target)]
    assert written == ["fake/good-model/excluded-models_00000_000000000.png"]
    marker = json.loads((target / MATERIALIZED_MARKER).read_text(encoding="utf-8"))
    assert marker["filtered"] == 2


def test_snapshot_download_is_rank0_single_worker_and_uses_ephemeral_staging(
    tmp_path, make_ctx, monkeypatch
):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(kwargs["local_dir"]) / "metadata.json").write_text("{}", encoding="utf-8")
        cache = Path(kwargs["local_dir"]) / ".cache" / "huggingface" / "download"
        cache.mkdir(parents=True)
        (cache / "metadata.json.lock").write_text("cache", encoding="utf-8")
        return str(kwargs["local_dir"])

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download)
    )
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    monkeypatch.delenv("HF_XET_NUM_CONCURRENT_RANGE_GETS", raising=False)
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    ds = DatasetConfig(
        name="raw-metadata",
        repo_id="org/raw-metadata",
        revision="abc123",
        format="raw",
        download_only=True,
        label="unknown",
    )
    ctx = make_ctx(
        datasets=(ds,),
        download=DownloadConfig(staging_dir=tmp_path / "staging"),
        world_size=8,
        rank=0,
    )
    download.run(ctx)

    assert len(calls) == 1
    call = calls[0]
    assert call["max_workers"] == 1
    assert Path(call["local_dir"]) == tmp_path / "staging" / ".hf_snapshots" / ds.name
    assert "cache_dir" not in call  # local_dir owns its HF metadata cache
    assert os.environ["HF_XET_CACHE"] == str(tmp_path / "staging" / ".hf_xet")
    assert "HF_XET_NUM_CONCURRENT_RANGE_GETS" not in os.environ
    assert "HF_XET_HIGH_PERFORMANCE" not in os.environ
    assert (ctx.data_dir / ds.name / MATERIALIZED_MARKER).exists()
    marker = json.loads((ctx.data_dir / ds.name / MATERIALIZED_MARKER).read_text())
    assert marker["storage"] == "raw_tar"
    archive_path = ctx.data_dir / ds.name / marker["archive"]
    assert archive_path.is_file()
    with tarfile.open(archive_path) as archive:
        assert [member.name for member in archive if member.isfile()] == ["metadata.json"]
    assert not Path(call["local_dir"]).exists()
    assert not (tmp_path / "staging" / ".hf_xet").exists()
    assert pipeline_datasets(ctx.cfg) == ()

    rank1 = make_ctx(datasets=(ds,), world_size=8, rank=1)
    with pytest.raises(RuntimeError, match="rank 0"):
        download.run(rank1)
    assert len(calls) == 1


def test_materialize_multiple_image_columns_with_generators(tmp_path, make_ctx):
    source = tmp_path / "pairs"
    source.mkdir()
    png = _png_bytes()
    pq.write_table(
        pa.table(
            {
                "image1": pa.array([png], type=pa.binary()),
                "image2": pa.array([png], type=pa.binary()),
                "model1": ["generator-a"],
                "model2": ["generator-b"],
            }
        ),
        source / "pairs.parquet",
    )
    ds = DatasetConfig(
        name="pairs",
        path=str(source),
        format="parquet",
        label="fake",
        images=(
            ImageColumnConfig("image1", "embedded", generator_column="model1"),
            ImageColumnConfig("image2", "embedded", generator_column="model2"),
        ),
    )
    ctx = make_ctx(datasets=(ds,))
    download.run(ctx)
    target = ctx.data_dir / "pairs"
    written = sorted(row["path"].partition("/")[2] for row in _wds_rows(target))
    assert written == [
        "generator-a/pairs_00000_000000000_i0.png",
        "generator-b/pairs_00000_000000001_i1.png",
    ]
    marker = json.loads((target / MATERIALIZED_MARKER).read_text(encoding="utf-8"))
    assert marker["layout"] == {"label_dir": False, "generator_dir": True}


def test_materialize_arrow_stream(tmp_path, make_ctx):
    source = tmp_path / "arrow"
    source.mkdir()
    table = pa.table({"image": pa.array([_png_bytes()], type=pa.binary())})
    with pa.OSFile(str(source / "data.arrow"), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    ds = DatasetConfig(
        name="arrow-images",
        path=str(source),
        format="arrow",
        label="fake",
        generator="nano-banana-pro",
        columns={"image": "image"},
    )
    ctx = make_ctx(datasets=(ds,))
    download.run(ctx)
    assert len(_wds_rows(ctx.data_dir / "arrow-images")) == 1


def test_multipart_zip_extracts_only_selected_output(tmp_path, make_ctx):
    source = tmp_path / "multipart"
    source.mkdir()
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("inputs/real.png", _png_bytes())
        archive.writestr("outputs/fake.png", _png_bytes())
    payload = archive_bytes.getvalue()
    midpoint = len(payload) // 2
    (source / "images.part_00").write_bytes(payload[:midpoint])
    (source / "images.part_01").write_bytes(payload[midpoint:])
    (source / "rows.jsonl").write_text(
        json.dumps({"input_images": ["inputs/real.png"], "output_image": "outputs/fake.png"})
        + "\n",
        encoding="utf-8",
    )
    ds = DatasetConfig(
        name="multipart",
        path=str(source),
        format="multipart_zip",
        multipart_glob="images.part_*",
        metadata_file="rows.jsonl",
        output_column="output_image",
        label="fake",
    )
    ctx = make_ctx(datasets=(ds,))
    download.run(ctx)
    target = ctx.data_dir / "multipart"
    assert [row["path"] for row in _wds_rows(target)] == ["multipart/outputs/fake.png"]


def test_multipart_tar_reads_members_across_chunk_boundaries(tmp_path, make_ctx):
    source = tmp_path / "multipart-tar"
    source.mkdir()
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        for name in ("first.png", "second.png"):
            payload = _png_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    payload = archive_bytes.getvalue()
    # Split inside the first member payload so neither part is independently usable.
    midpoint = 512 + len(_png_bytes()) // 2
    (source / "images_0000.tar").write_bytes(payload[:midpoint])
    (source / "images_0001.tar").write_bytes(payload[midpoint:])
    ds = DatasetConfig(
        name="multipart-tar",
        path=str(source),
        format="multipart_tar",
        label="fake",
        generator="test-generator",
    )
    ctx = make_ctx(datasets=(ds,))
    download.run(ctx)

    target = ctx.data_dir / "multipart-tar"
    assert sorted(Path(row["path"]).name for row in _wds_rows(target)) == [
        "first.png", "second.png"
    ]
    marker = json.loads((target / MATERIALIZED_MARKER).read_text(encoding="utf-8"))
    assert marker["format"] == "multipart_tar"
    assert marker["written"] == 2
    assert marker["parts"] == 2
    assert marker["storage"] == "webdataset"
