#!/usr/bin/env python3
"""Download images referenced by URL-only datasets in the AIGC config.

This is deliberately separate from databuilder's rank-0 snapshot stage. It
supports the two pinned URL-backed repositories in examples/aigc-datasets.toml:

* kafked/anycrap: data/full/train.jsonl -> image_url
* lehduong/seaart-hq: parquet shards -> url

Images are streamed to ``<data-dir>/<dataset>/url-images``. Completed files
are resumable by their URL-derived filename; partial ``.part`` files are never
treated as complete. Every downloaded image is decoded with Pillow before its
atomic rename into place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

LOG = logging.getLogger("download-url-datasets")
DEFAULT_DATA_DIR = Path("/p/data1/datasets/mmlaion/aigc/data")
CHUNK_SIZE = 1024 * 1024
FORMAT_EXTENSIONS = {
    "AVIF": ".avif",
    "BMP": ".bmp",
    "GIF": ".gif",
    "HEIF": ".heif",
    "JPEG": ".jpg",
    "JPEGXL": ".jxl",
    "JXL": ".jxl",
    "PNG": ".png",
    "TIFF": ".tif",
    "WEBP": ".webp",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str
    revision: str
    allow_patterns: tuple[str, ...]
    metadata_format: str
    url_column: str
    id_column: str = ""
    required_bool_column: str = ""


DATASETS = {
    "anycrap": DatasetSpec(
        name="anycrap",
        repo_id="kafked/anycrap",
        revision="bd1297fc0211e4531c86651d2cd55a4d069324af",
        allow_patterns=("data/full/train.jsonl",),
        metadata_format="jsonl",
        url_column="image_url",
        id_column="id",
        required_bool_column="has_real_image",
    ),
    "seaart-hq": DatasetSpec(
        name="seaart-hq",
        repo_id="lehduong/seaart-hq",
        revision="4321a586faf084ddfa6ab415caae4073af13aa67",
        allow_patterns=("*.parquet", "**/*.parquet"),
        metadata_format="parquet",
        url_column="url",
    ),
}


@dataclass(frozen=True)
class Candidate:
    dataset: str
    url: str
    row_id: str
    source: str
    row_number: int

    @property
    def stem(self) -> str:
        return hashlib.sha256(self.url.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DownloadResult:
    candidate: Candidate
    status: str
    path: str = ""
    bytes_written: int = 0
    error: str = ""


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"dataset root (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="dataset to process; repeat as needed (default: both)",
    )
    parser.add_argument("--workers", type=_positive_int, default=16)
    parser.add_argument("--retries", type=_nonnegative_int, default=4)
    parser.add_argument("--timeout", type=_positive_int, default=60)
    parser.add_argument(
        "--max-image-mib",
        type=_positive_int,
        default=100,
        help="reject a response larger than this many MiB",
    )
    parser.add_argument(
        "--limit",
        type=_nonnegative_int,
        default=0,
        help="maximum unique URLs per dataset; 0 means unlimited",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="extra HTTP request header; repeat as needed",
    )
    parser.add_argument(
        "--skip-metadata-snapshot",
        action="store_true",
        help="use metadata already present under data-dir and make no HF call",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-partials",
        action="store_true",
        help="retain failed .part files for inspection (they are not resumed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan metadata and report URL counts without fetching images",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    return parser.parse_args(argv)


def parse_headers(values: list[str]) -> dict[str, str]:
    headers = {"User-Agent": "databuilder-url-downloader/0.1"}
    for value in values:
        name, separator, content = value.partition("=")
        if not separator or not name.strip() or not content.strip():
            raise ValueError(f"invalid --header {value!r}; expected NAME=VALUE")
        headers[name.strip()] = content.strip()
    return headers


def ensure_metadata_snapshot(data_dir: Path, spec: DatasetSpec, skip: bool) -> Path:
    root = data_dir / spec.name
    if skip:
        if not root.is_dir():
            raise FileNotFoundError(
                f"metadata directory does not exist for {spec.name}: {root}"
            )
        return root

    # huggingface_hub reads its environment at import time. Keep Xet chunks
    # below the requested data root, but otherwise retain HF/Xet defaults.
    os.environ["HF_XET_CACHE"] = str(data_dir / ".hf_xet")
    from huggingface_hub import snapshot_download

    LOG.info("snapshotting metadata for %s at %s", spec.repo_id, spec.revision)
    return Path(
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=spec.revision,
            local_dir=root,
            allow_patterns=list(spec.allow_patterns),
            max_workers=1,
        )
    )


def _valid_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return value


def _candidate_from_row(
    spec: DatasetSpec,
    row: dict,
    source: Path,
    row_number: int,
) -> Iterator[Candidate]:
    if spec.required_bool_column and row.get(spec.required_bool_column) is not True:
        return
    raw_urls = row.get(spec.url_column)
    values = raw_urls if isinstance(raw_urls, list) else [raw_urls]
    for image_index, raw_url in enumerate(values):
        url = _valid_url(raw_url)
        if url is None:
            continue
        raw_id = row.get(spec.id_column) if spec.id_column else ""
        row_id = str(raw_id or f"{source.name}:{row_number}")
        if len(values) > 1:
            row_id = f"{row_id}:{image_index}"
        yield Candidate(
            dataset=spec.name,
            url=url,
            row_id=row_id,
            source=source.as_posix(),
            row_number=row_number,
        )


def iter_candidates(spec: DatasetSpec, metadata_root: Path) -> Iterator[Candidate]:
    files = sorted(
        {
            path
            for pattern in spec.allow_patterns
            for path in metadata_root.glob(pattern)
            if path.is_file()
        }
    )
    if spec.metadata_format == "jsonl":
        files = [path for path in files if path.suffix.lower() == ".jsonl"]
        if not files:
            raise FileNotFoundError(f"no JSONL metadata found under {metadata_root}")
        for path in files:
            with path.open("r", encoding="utf-8") as handle:
                for row_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSON at {path}:{row_number}") from exc
                    if not isinstance(row, dict):
                        continue
                    yield from _candidate_from_row(spec, row, path, row_number)
        return

    files = [path for path in files if path.suffix.lower() == ".parquet"]
    if not files:
        raise FileNotFoundError(f"no Parquet metadata found under {metadata_root}")
    for path in files:
        parquet = pq.ParquetFile(path)
        required = {spec.url_column}
        if spec.id_column:
            required.add(spec.id_column)
        if spec.required_bool_column:
            required.add(spec.required_bool_column)
        missing = sorted(required - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"{path} is missing required columns {missing}")
        row_number = 0
        for batch in parquet.iter_batches(batch_size=4096, columns=sorted(required)):
            for row in batch.to_pylist():
                row_number += 1
                yield from _candidate_from_row(spec, row, path, row_number)


def _register_pillow_plugins() -> None:
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        pillow_heif.register_avif_opener()
    except Exception:  # noqa: BLE001 - optional format plugins
        pass
    try:
        import pillow_jxl  # noqa: F401
    except Exception:  # noqa: BLE001 - optional format plugin
        pass


def _validated_extension(path: Path) -> str:
    _register_pillow_plugins()
    try:
        with Image.open(path) as image:
            image_format = (image.format or "").upper()
            image.verify()
    except Exception as exc:  # noqa: BLE001 - malformed responses must be rejected
        raise ValueError(f"response is not a decodable image: {type(exc).__name__}") from exc
    extension = FORMAT_EXTENSIONS.get(image_format)
    if extension is None:
        raise ValueError(f"unsupported decoded image format {image_format!r}")
    return extension


def _existing_image(output_dir: Path, stem: str) -> Path | None:
    for extension in sorted(set(FORMAT_EXTENSIONS.values())):
        candidate = output_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _retry_delay(exc: Exception, attempt: int) -> float:
    if isinstance(exc, urllib.error.HTTPError):
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if retry_after and retry_after.isdigit():
            return min(float(retry_after), 120.0)
    return min(float(2**attempt), 30.0)


def download_one(
    candidate: Candidate,
    output_dir: Path,
    headers: dict[str, str],
    timeout: int,
    retries: int,
    max_bytes: int,
    overwrite: bool,
    keep_partials: bool,
) -> DownloadResult:
    existing = _existing_image(output_dir, candidate.stem)
    if existing is not None and not overwrite:
        return DownloadResult(candidate, "skipped", str(existing), existing.stat().st_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    partial = output_dir / f"{candidate.stem}.part"
    last_error = "unknown error"
    attempts = retries + 1
    for attempt in range(attempts):
        caught: Exception | None = None
        try:
            request = urllib.request.Request(candidate.url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_bytes:
                    raise ValueError(
                        f"Content-Length {content_length} exceeds limit {max_bytes}"
                    )
                written = 0
                with partial.open("wb") as handle:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > max_bytes:
                            raise ValueError(f"response exceeds byte limit {max_bytes}")
                        handle.write(chunk)
            extension = _validated_extension(partial)
            destination = output_dir / f"{candidate.stem}{extension}"
            partial.replace(destination)
            return DownloadResult(candidate, "downloaded", str(destination), written)
        except urllib.error.HTTPError as exc:
            caught = exc
            last_error = f"HTTP {exc.code}: {exc.reason}"
            if exc.code in {400, 401, 403, 404, 410}:
                break
        except Exception as exc:  # noqa: BLE001 - retry network/decode/write failures
            caught = exc
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < attempts and caught is not None:
            time.sleep(_retry_delay(caught, attempt))

    if partial.exists() and not keep_partials:
        partial.unlink()
    return DownloadResult(candidate, "failed", error=last_error)


def _json_record(result: DownloadResult, root: Path) -> dict:
    path = ""
    if result.path:
        resolved = Path(result.path)
        try:
            path = resolved.relative_to(root).as_posix()
        except ValueError:
            path = str(resolved)
    return {
        "time": int(time.time()),
        "dataset": result.candidate.dataset,
        "row_id": result.candidate.row_id,
        "source": result.candidate.source,
        "row_number": result.candidate.row_number,
        "url": result.candidate.url,
        "status": result.status,
        "path": path,
        "bytes": result.bytes_written,
        "error": result.error,
    }


def download_dataset(
    spec: DatasetSpec,
    metadata_root: Path,
    args: argparse.Namespace,
    headers: dict[str, str],
) -> dict[str, int]:
    output_dir = metadata_root / "url-images"
    seen: set[str] = set()
    candidates = iter_candidates(spec, metadata_root)

    if args.dry_run:
        count = 0
        for candidate in candidates:
            if candidate.url in seen:
                continue
            seen.add(candidate.url)
            count += 1
            if args.limit and count >= args.limit:
                break
        LOG.info("dry-run: %s has %d unique downloadable URLs", spec.name, count)
        return {"discovered": count, "downloaded": 0, "skipped": 0, "failed": 0}

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "downloads.jsonl"
    failures_path = output_dir / "failures.jsonl"
    counters = {"discovered": 0, "downloaded": 0, "skipped": 0, "failed": 0}
    max_pending = max(args.workers * 4, args.workers)

    def handle_result(result: DownloadResult, progress: tqdm, manifest, failures) -> None:
        counters[result.status] += 1
        if result.status in {"downloaded", "failed"}:
            record = _json_record(result, metadata_root)
            handle = failures if result.status == "failed" else manifest
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        progress.update(1)

    with (
        manifest_path.open("a", encoding="utf-8") as manifest,
        failures_path.open("a", encoding="utf-8") as failures,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
        tqdm(desc=spec.name, unit="image", dynamic_ncols=True) as progress,
    ):
        pending: set[Future[DownloadResult]] = set()
        for candidate in candidates:
            if candidate.url in seen:
                continue
            seen.add(candidate.url)
            counters["discovered"] += 1
            pending.add(
                executor.submit(
                    download_one,
                    candidate,
                    output_dir,
                    headers,
                    args.timeout,
                    args.retries,
                    args.max_image_mib * 1024 * 1024,
                    args.overwrite,
                    args.keep_partials,
                )
            )
            if len(pending) >= max_pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    handle_result(future.result(), progress, manifest, failures)
            if args.limit and counters["discovered"] >= args.limit:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                handle_result(future.result(), progress, manifest, failures)
    return counters


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        headers = parse_headers(args.header)
    except ValueError as exc:
        LOG.error("%s", exc)
        return 2

    selected = args.dataset or list(DATASETS)
    overall_failed = 0
    for name in selected:
        spec = DATASETS[name]
        try:
            metadata_root = ensure_metadata_snapshot(
                args.data_dir, spec, args.skip_metadata_snapshot
            )
            counters = download_dataset(spec, metadata_root, args, headers)
        except (FileNotFoundError, ValueError) as exc:
            LOG.error("%s: %s", name, exc)
            overall_failed += 1
            continue
        LOG.info("%s complete: %s", name, counters)
        overall_failed += counters["failed"]
    return 1 if overall_failed else 0


if __name__ == "__main__":
    sys.exit(main())
