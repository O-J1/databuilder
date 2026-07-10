from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..config import Config, ConfigError, DatasetConfig
from ..utils import IMAGE_SUFFIXES

# Materialized layout under runtime.data_dir (in-place local datasets keep their
# own structure under ds.path):
#   <dataset name>/                                  static label + static generator
#   <dataset name>/<generator>/...                   generator from column or 'folder'
#   <dataset name>/<label>/<generator>/...           label AND generator from columns
# resolve_meta() decodes label/generator back out of the relative path, so no
# per-image metadata joins are needed downstream. When the download stage
# automatches columns, the resulting layout is recorded in .materialized.json
# and read back through load_layout().

REAL_NAMES = {"real", "human", "natural", "authentic", "photo", "0"}
FAKE_NAMES = {"fake", "ai", "aigc", "generated", "synthetic", "1"}

MATERIALIZED_MARKER = ".materialized.json"


@dataclass(frozen=True)
class DatasetLayout:
    """Which positional directories exist in a dataset's materialized paths."""

    label_dir: bool
    generator_dir: bool


def dataset_by_name(cfg: Config, name: str) -> DatasetConfig | None:
    for ds in cfg.datasets:
        if ds.name == name:
            return ds
    return None


def dataset_root(cfg: Config, ds: DatasetConfig) -> Path:
    """Directory that holds this dataset's images (in place for local imagefolders)."""
    if ds.in_place:
        return Path(ds.path)
    return cfg.runtime.data_dir / ds.name


def dataset_roots(cfg: Config) -> dict[str, str]:
    return {ds.name: str(dataset_root(cfg, ds)) for ds in cfg.datasets}


def protected_datasets(cfg: Config) -> frozenset[str]:
    """Datasets whose source files must never be physically deleted.

    In-place local datasets are protected by default; `allow_delete = true`
    opts a dataset back in. Materialized copies under runtime.data_dir are
    transient and always deletable.
    """
    return frozenset(ds.name for ds in cfg.datasets if ds.in_place and not ds.allow_delete)


def resolve_abs_from_roots(roots: dict[str, str], rel: str) -> Path:
    """Map a canonical '<dataset>/<subpath>' key onto its absolute location."""
    name, _, rest = rel.partition("/")
    base = roots.get(name)
    if base is None:
        raise KeyError(f"unknown dataset {name!r} in path {rel!r}")
    return Path(base) / rest


def config_layout(ds: DatasetConfig) -> DatasetLayout:
    return DatasetLayout(
        label_dir=bool(ds.label_column), generator_dir=bool(ds.generator_column)
    )


def load_layout(cfg: Config, ds: DatasetConfig) -> DatasetLayout:
    """Effective layout: automatched columns recorded at materialization win."""
    marker = cfg.runtime.data_dir / ds.name / MATERIALIZED_MARKER
    if marker.exists():
        meta = json.loads(marker.read_text(encoding="utf-8"))
        layout = meta.get("layout")
        if isinstance(layout, dict):
            return DatasetLayout(
                label_dir=bool(layout.get("label_dir")),
                generator_dir=bool(layout.get("generator_dir")),
            )
    return config_layout(ds)


def label_from_value(ds: DatasetConfig, value: str) -> str | None:
    """Canonical 'real'/'fake' for a folder name or column value, else None."""
    v = str(value).strip().lower()
    mapped = ds.label_map.get(v)
    if mapped:
        return mapped
    if v in REAL_NAMES:
        return "real"
    if v in FAKE_NAMES:
        return "fake"
    return None


def uses_folder_labels(ds: DatasetConfig) -> bool:
    return ds.label == "folder" or (ds.label == "auto" and ds.in_place)


def resolve_meta(
    ds: DatasetConfig, rel_parts: tuple[str, ...], layout: DatasetLayout | None = None
) -> tuple[str, str]:
    """Return (label_str, generator) for an image at <dataset>/<rel_parts...>."""
    dirs = rel_parts[:-1]
    layout = layout or config_layout(ds)
    idx = 0
    if uses_folder_labels(ds):
        label = "unknown"
        for segment in dirs:
            mapped = label_from_value(ds, segment)
            if mapped:
                label = mapped
                break
    elif layout.label_dir:
        raw = dirs[idx] if len(dirs) > idx else "unknown"
        label = label_from_value(ds, raw) or raw.lower()
        idx += 1
    else:
        label = ds.label if ds.label in {"real", "fake"} else "unknown"

    if layout.generator_dir:
        generator = dirs[idx] if len(dirs) > idx else ds.name
    elif ds.generator == "folder":
        candidate = dirs[-1] if dirs else ""
        # never use a label folder ('real', 'fake', ...) as the generator name
        if candidate and label_from_value(ds, candidate) is None:
            generator = candidate
        else:
            generator = ds.name
    elif ds.generator:
        generator = ds.generator
    else:
        generator = ds.name
    return label, generator


def normalize_label(label_str: str) -> int:
    value = label_str.strip().lower()
    if value in REAL_NAMES:
        return 0
    if value in FAKE_NAMES:
        return 1
    return -1


def validate_local_datasets(cfg: Config) -> None:
    """Fail fast, before any stage runs, when local datasets are unusable."""
    for ds in cfg.datasets:
        if not ds.is_local:
            continue
        root = Path(ds.path)
        if not root.is_dir():
            raise ConfigError(f"dataset {ds.name!r}: path {root} is not a directory")
        if ds.in_place and uses_folder_labels(ds):
            candidates: set[str] = set()
            for level1 in root.iterdir():
                if not level1.is_dir():
                    continue
                candidates.add(level1.name)
                for level2 in level1.iterdir():
                    if level2.is_dir():
                        candidates.add(level2.name)
            matched = {c for c in candidates if label_from_value(ds, c)}
            if not matched:
                raise ConfigError(
                    f"dataset {ds.name!r}: cannot infer labels from folder names "
                    f"{sorted(candidates)[:20]} under {root}. Expected folders like "
                    f"'real', 'fake', 'generated', 'ai', 'aigc' (up to two levels "
                    "deep). Fix: add [datasets.label_map] entries, set "
                    "label = 'real'/'fake', or point path at the right root."
                )


def iter_dataset_images(root: Path, ds: DatasetConfig) -> Iterator[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
