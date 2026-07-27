from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .stages.common import validate_local_datasets
from .state import REQUIRES, STAGES, RunContext

log = logging.getLogger("databuilder")


def _env_int(*names: str) -> tuple[int, str] | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value), name
    return None


RANK_ENV_VARS = ("DATABUILDER_RANK", "RANK", "SLURM_PROCID", "SLURM_NODEID")
WORLD_ENV_VARS = ("DATABUILDER_WORLD_SIZE", "WORLD_SIZE", "SLURM_NTASKS", "SLURM_NNODES")


def _warn_slurm_topology() -> None:
    """Databuilder expects one process per node (embed uses every GPU on the node)."""
    ntasks = os.environ.get("SLURM_NTASKS")
    nnodes = os.environ.get("SLURM_NNODES")
    per_node = os.environ.get("SLURM_NTASKS_PER_NODE", "")
    multi_per_node = per_node.split("(")[0] not in ("", "1") or (
        ntasks and nnodes and ntasks != nnodes
    )
    if multi_per_node:
        log.warning(
            "SLURM allocation has multiple tasks per node (ntasks=%s, nnodes=%s, "
            "ntasks-per-node=%s). databuilder expects ONE process per node; the "
            "embed stage uses all GPUs on the node. Use --ntasks-per-node=1.",
            ntasks,
            nnodes,
            per_node,
        )


def _resolve_rank_world(args: argparse.Namespace) -> tuple[int | None, int | None]:
    rank = args.rank
    world = args.world_size
    if rank is None:
        found = _env_int(*RANK_ENV_VARS)
        if found is not None:
            rank, source = found
            log.info("rank %d resolved from env %s", rank, source)
    if world is None:
        found = _env_int(*WORLD_ENV_VARS)
        if found is not None:
            world, source = found
            log.info("world_size %d resolved from env %s", world, source)
    if os.environ.get("SLURM_JOB_ID"):
        _warn_slurm_topology()
    return rank, world


def _make_context(args: argparse.Namespace) -> RunContext:
    rank, world_size = _resolve_rank_world(args)
    overrides = {
        "rank": rank,
        "world_size": world_size,
        "num_workers": args.workers,
    }
    cfg = load_config(args.config, overrides)
    validate_local_datasets(cfg)  # fail fast before any stage runs
    ctx = RunContext(cfg=cfg, dry_run=args.dry_run)
    ctx.ensure_run_meta()
    return ctx


def _stage_runner(name: str):
    module = importlib.import_module(f".stages.{name}", package=__package__)
    return module.run


def _execute(ctx: RunContext, name: str) -> None:
    if ctx.rank_done(name):
        log.info("[rank %d] stage %r already complete, skipping", ctx.rank, name)
        return
    log.info("[rank %d] running stage %r%s", ctx.rank, name, " (dry-run)" if ctx.dry_run else "")
    _stage_runner(name)(ctx)
    if ctx.dry_run:
        log.info("[rank %d] dry-run: not writing SUCCESS marker for %r", ctx.rank, name)
        return
    ctx.mark_success(name)


def cmd_run(args: argparse.Namespace) -> int:
    ctx = _make_context(args)
    for name in STAGES:
        if ctx.is_rank0_stage(name) and ctx.rank != 0:
            continue
        for req in REQUIRES[name]:
            ctx.wait_for(req)
        _execute(ctx, name)
    log.info("[rank %d] pipeline finished", ctx.rank)
    return 0


def cmd_stage(args: argparse.Namespace) -> int:
    ctx = _make_context(args)
    name = args.name
    if ctx.is_rank0_stage(name) and ctx.rank != 0:
        raise SystemExit(f"Stage {name!r} only runs on rank 0 (this is rank {ctx.rank}).")
    for req in REQUIRES[name]:
        if args.wait:
            ctx.wait_for(req)
        elif not ctx.stage_complete(req):
            raise SystemExit(
                f"Stage {name!r} requires completed stage {req!r}. "
                "Run it first or pass --wait."
            )
    _execute(ctx, name)
    return 0


def cmd_storage(args: argparse.Namespace) -> int:
    """Inspect or migrate local storage without contacting Hugging Face."""
    from .stages.download import inventory, migrate_existing

    cfg = load_config(args.config, {"rank": 0, "world_size": 1})
    ctx = RunContext(cfg=cfg, dry_run=args.dry_run)
    if args.action == "compact":
        if args.dry_run:
            rows = inventory(ctx)
        else:
            from .stages.manifest import _compact_and_rewrite

            manifest = ctx.work_dir / "artifacts" / "manifest" / "manifest.parquet"
            if not manifest.is_file():
                raise ConfigError(f"manifest missing: {manifest}")
            _compact_and_rewrite(ctx, manifest, keep_maps=False)
            rows = [{"status": "compacted", "manifest": str(manifest)}]
    elif args.action == "inventory" or args.dry_run:
        rows = inventory(ctx)
    else:
        rows = migrate_existing(ctx)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    needs_download = sum(row.get("status") in {"missing", "needs_download"} for row in rows)
    if needs_download:
        log.warning("%d dataset(s) have no locally recoverable source", needs_download)
    return 0


def _resolve_work_dir(args: argparse.Namespace) -> Path:
    if args.work_dir:
        return Path(args.work_dir)
    if args.config:
        return load_config(args.config).runtime.work_dir
    raise SystemExit("Provide --work-dir or --config.")


def _resolve_data_dir(args: argparse.Namespace) -> Path | None:
    if args.data_dir:
        return Path(args.data_dir)
    if args.config:
        return load_config(args.config).runtime.data_dir
    return None


def _resolve_roots(args: argparse.Namespace) -> dict[str, str] | None:
    if args.config:
        from .stages.common import dataset_roots

        return dataset_roots(load_config(args.config))
    return None


def cmd_viz_prepare(args: argparse.Namespace) -> int:
    from .viz.prepare import prepare, prepare_pairs

    prepare(
        work_dir=_resolve_work_dir(args),
        sample_size=args.sample,
        min_per_cluster=args.min_per_cluster,
        seed=args.seed,
        data_dir=_resolve_data_dir(args),
        roots=_resolve_roots(args),
    )
    if args.pairs:
        prepare_pairs(
            work_dir=_resolve_work_dir(args),
            kept_sample=args.pairs_kept_sample,
            seed=args.seed,
        )
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    from .viz.server import serve

    serve(
        work_dir=_resolve_work_dir(args),
        host=args.host,
        port=args.port,
        allow_unsafe_remote=args.allow_unsafe_remote,
        data_dir=_resolve_data_dir(args),
        roots=_resolve_roots(args),
    )
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to TOML config.")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Node rank (env: DATABUILDER_RANK/RANK/SLURM_PROCID/SLURM_NODEID).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Total nodes (env: DATABUILDER_WORLD_SIZE/WORLD_SIZE/SLURM_NTASKS/SLURM_NNODES).",
    )
    parser.add_argument("--workers", type=int, default=None, help="Override runtime.num_workers.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report deletions without touching files; no SUCCESS markers written.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "databuilder", description="Config-driven distributed image dataset builder."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run all pipeline stages in order (with barriers).")
    _add_common(run_p)
    run_p.set_defaults(func=cmd_run)

    stage_p = sub.add_parser("stage", help="Run a single pipeline stage.")
    stage_p.add_argument("name", choices=STAGES)
    _add_common(stage_p)
    stage_p.add_argument("--wait", action="store_true", help="Wait for required stages.")
    stage_p.set_defaults(func=cmd_stage)

    storage_p = sub.add_parser(
        "storage", help="Inventory or migrate existing data without network access."
    )
    storage_p.add_argument("action", choices=("inventory", "migrate", "compact"))
    storage_p.add_argument("--config", required=True)
    storage_p.add_argument(
        "--existing-only",
        action="store_true",
        help="Explicitly document that migration must not download (always enforced).",
    )
    storage_p.add_argument(
        "--dry-run", action="store_true", help="Print inventory; do not migrate."
    )
    storage_p.set_defaults(func=cmd_storage)

    vp = sub.add_parser("viz-prepare", help="Sample embeddings and project to 2D for the viewer.")
    vp.add_argument("--config", default=None)
    vp.add_argument("--work-dir", default=None)
    vp.add_argument("--data-dir", default=None, help="Image root (for thumbnail serving).")
    vp.add_argument("--sample", type=int, default=200_000, help="Max points to visualise.")
    vp.add_argument("--min-per-cluster", type=int, default=20)
    vp.add_argument("--seed", type=int, default=42)
    vp.add_argument(
        "--no-pairs",
        dest="pairs",
        action="store_false",
        help="Skip building the pruned-vs-kept pairs table.",
    )
    vp.add_argument(
        "--pairs-kept-sample",
        type=int,
        default=2048,
        help="Max kept members per cluster to search for nearest counterparts.",
    )
    vp.set_defaults(func=cmd_viz_prepare)

    vz = sub.add_parser("viz", help="Serve the cluster viewer on localhost.")
    vz.add_argument("--config", default=None)
    vz.add_argument("--work-dir", default=None)
    vz.add_argument("--data-dir", default=None, help="Image root (for thumbnail serving).")
    vz.add_argument("--host", default="127.0.0.1")
    vz.add_argument("--port", type=int, default=8765)
    vz.add_argument(
        "--allow-unsafe-remote",
        action="store_true",
        help="Permit binding to a non-loopback host. NOT recommended; use an SSH tunnel instead.",
    )
    vz.set_defaults(func=cmd_viz)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, TimeoutError) as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
