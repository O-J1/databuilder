from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_url_datasets.py"
SPEC = importlib.util.spec_from_file_location("download_url_datasets", SCRIPT)
assert SPEC and SPEC.loader
downloader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_anycrap_candidates_require_real_image_flag(tmp_path):
    path = tmp_path / "data" / "full" / "train.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"id": "one", "image_url": "https://example.test/one.jpg", "has_real_image": True},
        {"id": "two", "image_url": "https://example.test/two.jpg", "has_real_image": False},
        {"id": "three", "image_url": "not-a-url", "has_real_image": True},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    candidates = list(downloader.iter_candidates(downloader.DATASETS["anycrap"], tmp_path))
    assert [(item.row_id, item.url) for item in candidates] == [
        ("one", "https://example.test/one.jpg")
    ]


def test_seaart_candidates_stream_parquet(tmp_path):
    pq.write_table(
        pa.table({"url": ["https://example.test/a.webp", "ftp://invalid/image.jpg"]}),
        tmp_path / "split_1.parquet",
    )
    candidates = list(downloader.iter_candidates(downloader.DATASETS["seaart-hq"], tmp_path))
    assert len(candidates) == 1
    assert candidates[0].url == "https://example.test/a.webp"


def test_download_validates_and_resumes_existing_image(tmp_path, monkeypatch):
    payload = _png_bytes()
    calls = 0

    class Response:
        headers = {"Content-Length": str(len(payload)), "Content-Type": "image/png"}

        def __init__(self):
            self.buffer = io.BytesIO(payload)

        def read(self, size):
            return self.buffer.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    def fake_urlopen(*args, **kwargs):
        nonlocal calls
        calls += 1
        return Response()

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fake_urlopen)
    candidate = downloader.Candidate(
        dataset="seaart-hq",
        url="https://example.test/image-without-extension",
        row_id="1",
        source="split_1.parquet",
        row_number=1,
    )
    result = downloader.download_one(
        candidate,
        tmp_path,
        {"User-Agent": "test"},
        timeout=1,
        retries=0,
        max_bytes=1024 * 1024,
        overwrite=False,
        keep_partials=False,
    )
    assert result.status == "downloaded"
    assert Path(result.path).suffix == ".png"
    assert not (tmp_path / f"{candidate.stem}.part").exists()

    resumed = downloader.download_one(
        candidate,
        tmp_path,
        {"User-Agent": "test"},
        timeout=1,
        retries=0,
        max_bytes=1024 * 1024,
        overwrite=False,
        keep_partials=False,
    )
    assert resumed.status == "skipped"
    assert calls == 1
