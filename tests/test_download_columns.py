from __future__ import annotations

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from databuilder.config import ConfigError, DatasetConfig
from databuilder.stages import download
from databuilder.stages.common import MATERIALIZED_MARKER, load_layout
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
