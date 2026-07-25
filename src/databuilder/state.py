from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config, ConfigError

log = logging.getLogger("databuilder")

STAGES = ("download", "headerscan", "fingerprint", "dedup", "embed", "cluster", "manifest")
RANK0_STAGES = frozenset({"download", "dedup", "cluster", "manifest"})
# With the daft ray runner, rank 0 submits these stages to the Ray cluster and
# the remaining ranks only wait at the stage barrier.
DAFT_RAY_STAGES = frozenset({"fingerprint", "embed"})
REQUIRES: dict[str, tuple[str, ...]] = {
    "download": (),
    "headerscan": ("download",),
    "fingerprint": ("headerscan",),
    "dedup": ("fingerprint",),
    "embed": ("dedup",),
    "cluster": ("embed",),
    "manifest": ("cluster",),
}


@dataclass
class RunContext:
    """Per-process handle on config, sharding identity, and shared-FS state."""

    cfg: Config
    dry_run: bool = False

    @property
    def rank(self) -> int:
        return self.cfg.runtime.rank

    @property
    def world_size(self) -> int:
        return self.cfg.runtime.world_size

    @property
    def workers(self) -> int:
        return self.cfg.runtime.num_workers

    @property
    def data_dir(self) -> Path:
        return self.cfg.runtime.data_dir

    @property
    def work_dir(self) -> Path:
        return self.cfg.runtime.work_dir

    def artifact_dir(self, stage: str) -> Path:
        path = self.work_dir / "artifacts" / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def is_rank0_stage(self, stage: str) -> bool:
        """Stages that run on rank 0 only (globally, not per-rank sharded)."""
        if stage in RANK0_STAGES:
            return True
        daft = self.cfg.daft
        return daft.enabled and daft.runner == "ray" and stage in DAFT_RAY_STAGES

    def marker(self, stage: str, rank: int) -> Path:
        return self.work_dir / "state" / stage / f"rank_{rank:05d}.SUCCESS"

    def expected_ranks(self, stage: str) -> int:
        return 1 if self.is_rank0_stage(stage) else self.world_size

    def rank_done(self, stage: str) -> bool:
        rank = 0 if self.is_rank0_stage(stage) else self.rank
        return self.marker(stage, rank).exists()

    def stage_complete(self, stage: str) -> bool:
        return all(self.marker(stage, r).exists() for r in range(self.expected_ranks(stage)))

    def mark_success(self, stage: str) -> None:
        rank = 0 if self.is_rank0_stage(stage) else self.rank
        path = self.marker(stage, rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.0f}\n", encoding="utf-8")

    def wait_for(self, stage: str) -> None:
        """Block until every expected rank of `stage` has a SUCCESS marker."""
        timeout = self.cfg.runtime.barrier_timeout_s
        poll = self.cfg.runtime.barrier_poll_s
        deadline = time.monotonic() + timeout
        announced = False
        while not self.stage_complete(stage):
            if time.monotonic() > deadline:
                missing = [
                    r
                    for r in range(self.expected_ranks(stage))
                    if not self.marker(stage, r).exists()
                ]
                raise TimeoutError(
                    f"Barrier on stage {stage!r} timed out; missing ranks {missing}"
                )
            if not announced:
                log.info("[rank %d] waiting for stage %r barrier", self.rank, stage)
                announced = True
            time.sleep(poll)

    def ensure_run_meta(self) -> None:
        """Pin world_size for this work_dir so sharding stays consistent across stages."""
        meta_path = self.work_dir / "state" / "run.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("world_size") != self.world_size:
                raise ConfigError(
                    f"work_dir was initialised with world_size={meta.get('world_size')}, "
                    f"got {self.world_size}. Use a fresh work_dir to change world size."
                )
        elif self.rank == 0:
            meta_path.write_text(
                json.dumps({"world_size": self.world_size}), encoding="utf-8"
            )

    def remove_file(self, path: Path) -> bool:
        """Delete a file unless dry-run. Returns True when actually removed."""
        if self.dry_run:
            return False
        Path(path).unlink(missing_ok=True)
        return True
