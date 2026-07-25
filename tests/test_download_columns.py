from __future__ import annotations

import io
import json
import os
import sys
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
    written = sorted(p.relative_to(target).as_posix() for p in target.rglob("*.png"))
    assert written == [
        "fake/sd15/localpq_00000_000000001.png",
        "real/camera/localpq_00000_000000000.png",
    ]
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
    written = [path.relative_to(target).as_posix() for path in target.rglob("*.png")]
    assert written == ["real/unknown/nullable-generator_00000_000000000.png"]


def test_snapshot_download_is_rank0_single_worker_and_stays_in_data_dir(
    tmp_path, make_ctx, monkeypatch
):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        Path(kwargs["local_dir"]).mkdir(parents=True)
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
        download=DownloadConfig(xet_high_performance=True),
        world_size=8,
        rank=0,
    )
    download.run(ctx)

    assert len(calls) == 1
    call = calls[0]
    assert call["max_workers"] == 1
    assert Path(call["local_dir"]) == ctx.data_dir / ds.name
    assert Path(call["local_dir"]).is_relative_to(ctx.data_dir)
    assert "cache_dir" not in call  # local_dir owns its HF metadata cache
    assert os.environ["HF_XET_CACHE"] == str(ctx.data_dir / ".hf_xet")
    assert "HF_XET_NUM_CONCURRENT_RANGE_GETS" not in os.environ
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert (ctx.data_dir / ds.name / MATERIALIZED_MARKER).exists()
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
    written = sorted(path.relative_to(target).as_posix() for path in target.rglob("*.png"))
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
    assert len(list((ctx.data_dir / "arrow-images").rglob("*.png"))) == 1


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
    assert (target / "outputs" / "fake.png").is_file()
    assert not (target / "inputs" / "real.png").exists()
