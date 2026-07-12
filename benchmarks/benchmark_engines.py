from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pyarrow.parquet as pq  # noqa: E402

from databuilder.config import Config, ConfigError, DatasetConfig, load_config  # noqa: E402
from databuilder.stages.common import dataset_root, iter_dataset_images  # noqa: E402

DEFAULT_CONFIG = ROOT / "examples" / "smoke.toml"
DEFAULT_OUTPUT = ROOT / ".manual" / "benchmark"
SEED = 42
RUNS = 3
TOTAL_IMAGES = 500


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark three end-to-end fallback and native-Daft pipeline runs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Source pipeline config (default: examples/smoke.toml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Corpus, run artifacts, and result JSON directory.",
    )
    parser.add_argument(
        "--rebuild-corpus",
        action="store_true",
        help="Discard and deterministically rebuild the sampled corpus.",
    )
    return parser.parse_args(argv)


def _require_dependencies() -> None:
    missing = [
        name
        for name in ("daft", "torch", "transformers")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "Benchmark dependencies are missing: "
            f"{', '.join(missing)}. Install with pip install -e '.[embed]'."
        )


TOPOLOGY_ENV = (
    "DATABUILDER_RANK",
    "DATABUILDER_WORLD_SIZE",
    "RANK",
    "WORLD_SIZE",
    "SLURM_PROCID",
    "SLURM_NODEID",
    "SLURM_NTASKS",
    "SLURM_NNODES",
)


def _subprocess_env(offline: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    for name in TOPOLOGY_ENV:
        env.pop(name, None)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing if existing else "")
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def _run_cli(*args: str, offline: bool = False) -> None:
    subprocess.run(
        [sys.executable, "-m", "databuilder.cli", *args],
        cwd=ROOT,
        env=_subprocess_env(offline),
        check=True,
    )


def _prepare_sources(config_path: Path, cfg: Config) -> None:
    """Materialize sources into the smoke config's own work/data dirs.

    Runs the normal download stage against the unmodified config so a prior
    smoke run's SUCCESS marker and per-dataset .materialized.json markers are
    honoured: already-downloaded data is reused, never fetched again.
    """
    if cfg.runtime.world_size != 1 or cfg.runtime.rank != 0:
        raise RuntimeError(
            "The benchmark source config must use runtime.world_size = 1 and rank = 0."
        )
    print("Preparing smoke data outside the timed runs (reuses materialized data)...")
    _run_cli(
        "stage",
        "download",
        "--config",
        str(config_path),
        "--rank",
        "0",
        "--world-size",
        "1",
    )


def _images_per_dataset(cfg: Config) -> int:
    count = len(cfg.datasets)
    if count == 0 or TOTAL_IMAGES % count:
        raise RuntimeError(
            f"Expected a non-empty dataset list dividing {TOTAL_IMAGES} evenly; got {count}."
        )
    return TOTAL_IMAGES // count


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "corpus" / "selection.json"


def _load_existing_corpus(output_dir: Path, cfg: Config) -> dict[str, Any] | None:
    manifest_path = _manifest_path(output_dir)
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_names = [ds.name for ds in cfg.datasets]
    if (
        payload.get("seed") != SEED
        or payload.get("total_images") != TOTAL_IMAGES
        or payload.get("datasets") != expected_names
    ):
        raise RuntimeError(
            f"Existing corpus at {manifest_path.parent} does not match this benchmark. "
            "Pass --rebuild-corpus to replace it."
        )
    rows = payload.get("images", [])
    paths = [row.get("corpus_path", "") for row in rows]
    counts = Counter(row.get("dataset") for row in rows)
    expected_count = _images_per_dataset(cfg)
    expected_paths = {str(path) for path in paths}
    actual_paths = {
        path.relative_to(manifest_path.parent).as_posix()
        for ds in cfg.datasets
        for path in iter_dataset_images(manifest_path.parent / ds.name, ds)
    }
    invalid = (
        len(rows) != TOTAL_IMAGES
        or len(expected_paths) != TOTAL_IMAGES
        or expected_paths != actual_paths
        or any(counts[ds.name] != expected_count for ds in cfg.datasets)
    )
    bad_hashes = []
    for row, path in zip(rows, paths):
        candidate = manifest_path.parent / path
        if not path or not candidate.is_file() or _sha256(candidate) != row.get("sha256"):
            bad_hashes.append(path)
    if invalid or bad_hashes:
        raise RuntimeError(
            f"Existing benchmark corpus is incomplete or changed "
            f"({len(bad_hashes)} hash mismatches). "
            "Pass --rebuild-corpus to replace it."
        )
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_corpus(output_dir: Path, cfg: Config) -> dict[str, Any]:
    corpus_dir = output_dir / "corpus"
    temporary = output_dir / "corpus.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    per_dataset = _images_per_dataset(cfg)
    images: list[dict[str, str]] = []

    for ds in cfg.datasets:
        root = dataset_root(cfg, ds)
        candidates = list(iter_dataset_images(root, ds))
        if len(candidates) < per_dataset:
            raise RuntimeError(
                f"Dataset {ds.name!r} has {len(candidates)} images under {root}; "
                f"the benchmark needs {per_dataset}."
            )
        rng = random.Random(f"{SEED}:{ds.name}")
        selected = sorted(rng.sample(candidates, per_dataset))
        for source in selected:
            relative = source.relative_to(root)
            destination = temporary / ds.name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            images.append(
                {
                    "dataset": ds.name,
                    "source_path": relative.as_posix(),
                    "corpus_path": destination.relative_to(temporary).as_posix(),
                    "sha256": _sha256(destination),
                }
            )

    payload: dict[str, Any] = {
        "seed": SEED,
        "total_images": len(images),
        "images_per_dataset": per_dataset,
        "datasets": [ds.name for ds in cfg.datasets],
        "images": images,
    }
    (temporary / "selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    temporary.rename(corpus_dir)
    return payload


def _prepare_corpus(
    output_dir: Path, cfg: Config, payload: dict[str, Any] | None
) -> tuple[dict[str, Any], tuple[DatasetConfig, ...]]:
    corpus_dir = output_dir / "corpus"
    if payload is None:
        print(f"Sampling a fixed {TOTAL_IMAGES}-image corpus (seed={SEED})...")
        payload = _build_corpus(output_dir, cfg)
    else:
        print(f"Reusing fixed corpus at {corpus_dir}")

    datasets = tuple(
        DatasetConfig(
            name=ds.name,
            path=str((corpus_dir / ds.name).resolve()),
            format="imagefolder",
            label=ds.label,
            generator=ds.generator,
            assign_split=ds.assign_split,
            label_map=ds.label_map,
        )
        for ds in cfg.datasets
    )
    return payload, datasets


def _preload_model(cfg: Config) -> None:
    print(f"Caching model files outside timing: {cfg.embedding.model}")
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(cfg.embedding.model)
    model = AutoModel.from_pretrained(cfg.embedding.model)
    del processor, model
    gc.collect()


def _warm_corpus(output_dir: Path, corpus: dict[str, Any]) -> None:
    print("Warming the fixed corpus into the filesystem cache outside timing...")
    corpus_dir = output_dir / "corpus"
    for row in corpus["images"]:
        with (corpus_dir / row["corpus_path"]).open("rb") as handle:
            while handle.read(1024 * 1024):
                pass


def _toml_value(value: Any) -> str:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, (int, float)):
        return repr(value)
    raise TypeError(f"Cannot encode TOML value {value!r}")


def _toml_table(
    name: str, value: Any, overrides: dict[str, Any] | None = None
) -> list[str]:
    data = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    data.update(overrides or {})
    lines = [f"[{name}]"]
    lines.extend(f"{key} = {_toml_value(item)}" for key, item in data.items())
    return lines


def _write_run_config(
    path: Path,
    source: Config,
    datasets: tuple[DatasetConfig, ...],
    work_dir: Path,
    daft_enabled: bool,
) -> None:
    sections = [
        _toml_table(
            "runtime",
            source.runtime,
            {
                "work_dir": work_dir.resolve(),
                "data_dir": (work_dir / "data").resolve(),
                "world_size": 1,
                "rank": 0,
            },
        ),
        _toml_table("filters", source.filters),
        _toml_table("dedup", source.dedup),
        _toml_table("embedding", source.embedding),
        _toml_table("clustering", source.clustering),
        _toml_table("balance", source.balance),
        _toml_table(
            "daft",
            source.daft,
            {"enabled": daft_enabled, "runner": "native", "ray_address": "auto"},
        ),
    ]
    lines = [line for section in sections for line in (*section, "")]
    for ds in datasets:
        lines.extend(
            [
                "[[datasets]]",
                f"name = {_toml_value(ds.name)}",
                f"path = {_toml_value(ds.path)}",
                'format = "imagefolder"',
                f"label = {_toml_value(ds.label)}",
                f"generator = {_toml_value(ds.generator)}",
                f"assign_split = {_toml_value(ds.assign_split)}",
                "allow_delete = false",
                "",
            ]
        )
        if ds.label_map:
            lines.append("[datasets.label_map]")
            lines.extend(
                f"{json.dumps(key)} = {_toml_value(value)}"
                for key, value in ds.label_map.items()
            )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_once(
    engine: str,
    round_number: int,
    session_dir: Path,
    source: Config,
    datasets: tuple[DatasetConfig, ...],
) -> dict[str, Any]:
    run_dir = session_dir / "runs" / f"round-{round_number}-{engine}"
    config_path = run_dir / "benchmark.toml"
    work_dir = run_dir / "work"
    _write_run_config(config_path, source, datasets, work_dir, engine == "daft")

    print(f"\nRound {round_number}/{RUNS}: {engine}")
    started = time.perf_counter()
    _run_cli(
        "run",
        "--config",
        str(config_path),
        "--rank",
        "0",
        "--world-size",
        "1",
        offline=True,
    )
    duration = time.perf_counter() - started

    manifest = work_dir / "artifacts" / "manifest" / "manifest.parquet"
    survivors = work_dir / "artifacts" / "dedup" / "survivors.parquet"
    for artifact in (manifest, survivors):
        if not artifact.exists():
            raise RuntimeError(f"Pipeline completed without artifact: {artifact}")
    rows = pq.ParquetFile(manifest).metadata.num_rows
    # Manifest contents are not comparable across runs (usearch kmeans is
    # unseeded, so cluster ids and the label-ratio trim vary); the dedup
    # survivor set is the deterministic artifact the engines must agree on.
    survivor_ids = sorted(
        pq.read_table(survivors, columns=["image_id"]).column("image_id").to_pylist()
    )
    survivors_sha256 = hashlib.sha256(json.dumps(survivor_ids).encode("utf-8")).hexdigest()
    print(f"{engine}: {duration:.3f}s ({rows} manifest rows)")
    return {
        "engine": engine,
        "round": round_number,
        "seconds": duration,
        "manifest_rows": rows,
        "survivor_count": len(survivor_ids),
        "survivors_sha256": survivors_sha256,
        "config": str(config_path),
        "work_dir": str(work_dir),
    }


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for engine in ("fallback", "daft"):
        durations = [run["seconds"] for run in runs if run["engine"] == engine]
        summary[engine] = {
            "mean_seconds": mean(durations),
            "min_seconds": min(durations),
            "max_seconds": max(durations),
        }
    summary["speedup_fallback_over_daft"] = (
        summary["fallback"]["mean_seconds"] / summary["daft"]["mean_seconds"]
    )
    return summary


def _print_results(runs: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print("\nEnd-to-end wall-clock results")
    print("engine    round    seconds    manifest rows")
    print("--------  -----  ---------  -------------")
    for run in runs:
        print(
            f"{run['engine']:<8}  {run['round']:>5}  {run['seconds']:>9.3f}  "
            f"{run['manifest_rows']:>13}"
        )
    print("\nengine       mean        min        max")
    print("--------  ---------  ---------  ---------")
    for engine in ("fallback", "daft"):
        values = summary[engine]
        print(
            f"{engine:<8}  {values['mean_seconds']:>9.3f}  "
            f"{values['min_seconds']:>9.3f}  {values['max_seconds']:>9.3f}"
        )
    speedup = summary["speedup_fallback_over_daft"]
    print(f"\nDaft speedup (fallback mean / Daft mean): {speedup:.3f}x")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    cfg = load_config(config_path)
    if len(cfg.datasets) != 4:
        raise RuntimeError(
            f"This fixed smoke benchmark expects four datasets, got {len(cfg.datasets)}."
        )
    _require_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = None if args.rebuild_corpus else _load_existing_corpus(output_dir, cfg)
    if existing is None:
        _prepare_sources(config_path, cfg)
    corpus, datasets = _prepare_corpus(output_dir, cfg, existing)
    _preload_model(cfg)
    _warm_corpus(output_dir, corpus)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir = output_dir / timestamp
    suffix = 1
    while session_dir.exists():
        session_dir = output_dir / f"{timestamp}-{suffix}"
        suffix += 1

    runs: list[dict[str, Any]] = []
    orders = (("fallback", "daft"), ("daft", "fallback"), ("fallback", "daft"))
    for round_number, engines in enumerate(orders, start=1):
        for engine in engines:
            runs.append(_run_once(engine, round_number, session_dir, cfg, datasets))

    row_counts = Counter(run["manifest_rows"] for run in runs)
    survivor_hashes = Counter(run["survivors_sha256"] for run in runs)
    if len(row_counts) != 1 or len(survivor_hashes) != 1:
        raise RuntimeError(
            "Engine runs produced inconsistent outputs: "
            f"manifest rows={row_counts}, dedup survivor sets={survivor_hashes}"
        )

    summary = _summarize(runs)
    _print_results(runs, summary)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_config": str(config_path),
        "model": cfg.embedding.model,
        "seed": SEED,
        "runs_per_engine": RUNS,
        "total_images": TOTAL_IMAGES,
        "corpus": corpus,
        "runs": runs,
        "summary": summary,
    }
    result_path = session_dir / "results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    result_path.write_text(encoded, encoding="utf-8")
    (output_dir / "latest.json").write_text(encoded, encoding="utf-8")
    print(f"Results written to {result_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ConfigError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"benchmark failed: {exc}") from exc