from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CSV_MAX_ROWS = 1_000_000

DATASET_FORMATS = {
    "auto",
    "parquet",
    "arrow",
    "jsonl",
    "zip",
    "tar",
    "webdataset",
    "multipart_zip",
    "imagefolder",
    "raw",
}
IMAGE_KINDS = {"auto", "embedded", "path"}
CLUSTER_BACKENDS = {"auto", "usearch", "sklearn"}
DAFT_RUNNERS = {"native", "ray"}
COLUMN_ROLES = {"image", "label", "generator", "split"}
SPLIT_NAMES = {"train", "val", "test"}
EMBEDDING_DTYPES = {"float16", "float32"}


class ConfigError(ValueError):
    """Raised for missing or invalid configuration."""


def parse_ratio(text: str, key: str = "ratio") -> float:
    parts = str(text).split(":")
    if len(parts) != 2:
        raise ConfigError(f"{key} must look like '9:23', got {text!r}")
    try:
        num, den = float(parts[0]), float(parts[1])
        value = num / den
    except (ValueError, ZeroDivisionError) as exc:
        raise ConfigError(f"{key} must look like '9:23', got {text!r}") from exc
    if value <= 0:
        raise ConfigError(f"{key} must be positive, got {text!r}")
    return value


@dataclass(frozen=True)
class RuntimeConfig:
    work_dir: Path
    data_dir: Path
    num_workers: int = 8
    world_size: int = 1
    rank: int = 0
    barrier_timeout_s: float = 86_400.0
    barrier_poll_s: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_dir", Path(self.work_dir))
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        if self.world_size < 1:
            raise ConfigError("runtime.world_size must be >= 1")
        if not 0 <= self.rank < self.world_size:
            raise ConfigError(
                f"runtime.rank must be in [0, {self.world_size}), got {self.rank}"
            )
        if self.num_workers < 1:
            raise ConfigError("runtime.num_workers must be >= 1")


@dataclass(frozen=True)
class DownloadConfig:
    """Hugging Face snapshot concurrency controls."""

    max_workers: int = 1
    xet_high_performance: bool = False

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ConfigError("download.max_workers must be >= 1")
        if not isinstance(self.xet_high_performance, bool):
            raise ConfigError("download.xet_high_performance must be true or false")


@dataclass(frozen=True)
class FiltersConfig:
    min_longest_side: int = 256
    max_tall: str = "9:23"
    max_wide: str = "23:9"
    laplacian_min: float = 2.0
    laplacian_max: float = 15_000.0

    def __post_init__(self) -> None:
        tall = parse_ratio(self.max_tall, "filters.max_tall")
        wide = parse_ratio(self.max_wide, "filters.max_wide")
        if tall >= wide:
            raise ConfigError("filters.max_tall ratio must be smaller than max_wide ratio")
        if self.laplacian_min >= self.laplacian_max:
            raise ConfigError("filters.laplacian_min must be below laplacian_max")

    @property
    def tall_ratio(self) -> float:
        """Delete when width/height is below this."""
        return parse_ratio(self.max_tall)

    @property
    def wide_ratio(self) -> float:
        """Delete when width/height is above this."""
        return parse_ratio(self.max_wide)


@dataclass(frozen=True)
class DedupConfig:
    phash_size: int = 12
    phash_max_hamming: int = 8
    colorhash_max_hamming: int = 6
    keep_removed_files: bool = False


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
    batch_size: int = 64
    image_size: int = 0
    dtype: str = "float16"
    devices: str = "auto"
    flush_rows: int = 20_000
    concurrency: int = 0

    def __post_init__(self) -> None:
        if self.dtype not in EMBEDDING_DTYPES:
            raise ConfigError(
                "embedding.dtype must be one of "
                f"{sorted(EMBEDDING_DTYPES)}, got {self.dtype!r}"
            )
        if self.concurrency < 0:
            raise ConfigError(
                "embedding.concurrency must be >= 0 (0 means auto)"
            )


@dataclass(frozen=True)
class DaftConfig:
    """Optional Daft execution path for the fingerprint and embed stages.

    runner "native" keeps the per-rank SLURM sharding and runs Daft locally on
    each node; runner "ray" submits the whole stage to a Ray cluster from
    rank 0 while the other ranks wait at the stage barrier.
    """

    enabled: bool = False
    runner: str = "native"
    ray_address: str = "auto"

    def __post_init__(self) -> None:
        if self.runner not in DAFT_RUNNERS:
            raise ConfigError(f"daft.runner must be one of {sorted(DAFT_RUNNERS)}")


@dataclass(frozen=True)
class ClusteringConfig:
    backend: str = "auto"
    k: int = 0
    aggressiveness: float = 0.5
    prune_trigger_sigma: float = 3.0
    semdedup_threshold: float = 0.96
    max_ram_gb: float = 64.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.backend not in CLUSTER_BACKENDS:
            raise ConfigError(f"clustering.backend must be one of {sorted(CLUSTER_BACKENDS)}")
        if self.k == 0 and not 0 < self.aggressiveness <= 1:
            raise ConfigError("clustering.aggressiveness must be in (0, 1]")
        if self.prune_trigger_sigma < 0:
            raise ConfigError("clustering.prune_trigger_sigma must be >= 0")
        if not 0 < self.semdedup_threshold <= 1:
            raise ConfigError("clustering.semdedup_threshold must be in (0, 1]")


@dataclass(frozen=True)
class BalanceConfig:
    max_per_generator: int = 0
    per_generator_cluster_cap: int = 0
    max_label_ratio: float = 0.0
    val_fraction: float = 0.02
    test_fraction: float = 0.0
    seed: int = 42
    emit_csv: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.val_fraction + self.test_fraction < 1:
            raise ConfigError("balance.val_fraction + test_fraction must be in [0, 1)")
        if self.max_label_ratio < 0:
            raise ConfigError("balance.max_label_ratio must be >= 0 (0 disables it)")


@dataclass(frozen=True)
class ImageColumnConfig:
    """One image-bearing field in a tabular dataset."""

    column: str
    kind: str = "auto"
    generator: str = ""
    generator_column: str = ""

    def __post_init__(self) -> None:
        if not self.column:
            raise ConfigError("datasets.images entries need a non-empty column")
        if self.kind not in IMAGE_KINDS:
            raise ConfigError(
                f"datasets.images kind must be one of {sorted(IMAGE_KINDS)}, "
                f"got {self.kind!r}"
            )
        if self.generator and self.generator_column:
            raise ConfigError(
                "datasets.images entries cannot set both generator and generator_column"
            )


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    label: str
    repo_id: str = ""
    path: str = ""
    generator: str = ""
    format: str = "auto"
    image_dir: str = ""
    source_split: str = ""
    revision: str = ""
    allow_patterns: tuple[str, ...] = ()
    keep_archives: bool = False
    allow_delete: bool = False
    assign_split: str = ""
    download_only: bool = False
    images: tuple[ImageColumnConfig, ...] = ()
    row_filter: dict = field(default_factory=dict)
    metadata_file: str = ""
    output_column: str = ""
    multipart_glob: str = ""
    columns: dict = field(default_factory=dict)
    label_map: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "allow_patterns", tuple(self.allow_patterns))
        images = tuple(
            image if isinstance(image, ImageColumnConfig) else ImageColumnConfig(**image)
            for image in self.images
        )
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "row_filter", dict(self.row_filter))
        if not self.name:
            raise ConfigError("datasets entries need a non-empty name")
        if bool(self.repo_id) == bool(self.path):
            raise ConfigError(
                f"dataset {self.name!r}: set exactly one of repo_id (HuggingFace) "
                "or path (local directory)"
            )
        if self.format not in DATASET_FORMATS:
            raise ConfigError(
                f"dataset {self.name!r}: format must be one of {sorted(DATASET_FORMATS)}"
            )
        if self.path and self.format == "auto":
            raise ConfigError(
                f"dataset {self.name!r}: local datasets need an explicit format "
                f"({', '.join(sorted(DATASET_FORMATS - {'auto'}))})"
            )
        if self.download_only and self.format != "raw":
            raise ConfigError(
                f"dataset {self.name!r}: download_only datasets must use format = 'raw'"
            )
        if self.format == "raw" and not self.download_only:
            raise ConfigError(
                f"dataset {self.name!r}: format = 'raw' requires download_only = true"
            )
        if self.format == "multipart_zip":
            if not self.multipart_glob or not self.metadata_file or not self.output_column:
                raise ConfigError(
                    f"dataset {self.name!r}: multipart_zip requires multipart_glob, "
                    "metadata_file, and output_column"
                )
        elif self.multipart_glob or self.metadata_file or self.output_column:
            raise ConfigError(
                f"dataset {self.name!r}: multipart_glob/metadata_file/output_column "
                "are only valid with format = 'multipart_zip'"
            )
        if self.assign_split and self.assign_split not in SPLIT_NAMES:
            raise ConfigError(
                f"dataset {self.name!r}: assign_split must be one of {sorted(SPLIT_NAMES)}"
            )

        columns = dict(self.columns)
        # legacy 'column:<name>' syntax feeds the columns table
        if self.label.startswith("column:"):
            columns.setdefault("label", self.label.removeprefix("column:"))
        elif self.label not in {"real", "fake", "folder", "auto", "unknown"}:
            raise ConfigError(
                f"dataset {self.name!r}: label must be 'real', 'fake', 'folder', "
                "'auto', 'unknown', or 'column:<name>'"
            )
        if self.generator.startswith("column:"):
            columns.setdefault("generator", self.generator.removeprefix("column:"))

        unknown_roles = sorted(set(columns) - COLUMN_ROLES)
        if unknown_roles:
            raise ConfigError(
                f"dataset {self.name!r}: [datasets.columns] keys must be within "
                f"{sorted(COLUMN_ROLES)}, got {unknown_roles}"
            )
        if any(not isinstance(v, str) or not v for v in columns.values()):
            raise ConfigError(
                f"dataset {self.name!r}: [datasets.columns] values must be column names"
            )
        object.__setattr__(self, "columns", columns)
        if images and "image" in columns:
            raise ConfigError(
                f"dataset {self.name!r}: set either columns.image or images, not both"
            )

        label_map = {}
        for key, value in dict(self.label_map).items():
            if value not in {"real", "fake"}:
                raise ConfigError(
                    f"dataset {self.name!r}: [datasets.label_map] values must be "
                    f"'real' or 'fake', got {key!r} = {value!r}"
                )
            label_map[str(key).strip().lower()] = value
        object.__setattr__(self, "label_map", label_map)

    @property
    def is_local(self) -> bool:
        return bool(self.path)

    @property
    def in_place(self) -> bool:
        """Local imagefolder datasets are scanned in place; their files are only
        physically deleted when allow_delete is set."""
        return self.is_local and self.format == "imagefolder"

    @property
    def label_column(self) -> str | None:
        return self.columns.get("label")

    @property
    def generator_column(self) -> str | None:
        return self.columns.get("generator")

    @property
    def image_column(self) -> str | None:
        return self.columns.get("image")

    @property
    def split_column(self) -> str | None:
        return self.columns.get("split")


@dataclass(frozen=True)
class Config:
    runtime: RuntimeConfig
    download: DownloadConfig = field(default_factory=DownloadConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    daft: DaftConfig = field(default_factory=DaftConfig)
    datasets: tuple[DatasetConfig, ...] = ()


def _build(cls: type, data: dict[str, Any], section: str) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"Unknown key(s) in [{section}]: {unknown}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigError(f"[{section}]: {exc}") from exc


def load_config(path: Path | str, overrides: dict[str, Any] | None = None) -> Config:
    """Load and validate a TOML config. `overrides` patches [runtime] keys."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    runtime_raw = raw.pop("runtime", None)
    if not isinstance(runtime_raw, dict):
        raise ConfigError("Config must contain a [runtime] table with work_dir and data_dir")
    if overrides:
        runtime_raw.update({k: v for k, v in overrides.items() if v is not None})

    datasets_raw = raw.pop("datasets", [])
    if not isinstance(datasets_raw, list):
        raise ConfigError("[[datasets]] must be an array of tables")

    sections = {
        "download": DownloadConfig,
        "filters": FiltersConfig,
        "dedup": DedupConfig,
        "embedding": EmbeddingConfig,
        "clustering": ClusteringConfig,
        "balance": BalanceConfig,
        "daft": DaftConfig,
    }
    parsed: dict[str, Any] = {}
    for key, cls in sections.items():
        section_raw = raw.pop(key, {})
        if not isinstance(section_raw, dict):
            raise ConfigError(f"[{key}] must be a table")
        parsed[key] = _build(cls, section_raw, key)

    if raw:
        raise ConfigError(f"Unknown top-level section(s): {sorted(raw)}")

    for i, entry in enumerate(datasets_raw):
        if isinstance(entry, dict) and "split" in entry:
            raise ConfigError(
                f"datasets[{i}]: 'split' was renamed. Use source_split to pick which "
                "HF split to download, or assign_split to force the whole dataset "
                "into an output split (train/val/test)."
            )
    datasets = tuple(
        _build(DatasetConfig, entry, f"datasets[{i}]") for i, entry in enumerate(datasets_raw)
    )
    names = [ds.name for ds in datasets]
    if len(names) != len(set(names)):
        raise ConfigError("Dataset names must be unique")

    return Config(
        runtime=_build(RuntimeConfig, runtime_raw, "runtime"),
        datasets=datasets,
        **parsed,
    )
