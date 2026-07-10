from __future__ import annotations

import argparse
import logging

from databuilder.cli import RANK_ENV_VARS, WORLD_ENV_VARS, _resolve_rank_world

ALL_VARS = (*RANK_ENV_VARS, *WORLD_ENV_VARS, "SLURM_JOB_ID", "SLURM_NTASKS_PER_NODE")


def _args(rank=None, world_size=None):
    return argparse.Namespace(rank=rank, world_size=world_size)


def _clear(monkeypatch):
    for name in ALL_VARS:
        monkeypatch.delenv(name, raising=False)


def test_no_env_no_flags(monkeypatch):
    _clear(monkeypatch)
    assert _resolve_rank_world(_args()) == (None, None)


def test_slurm_env_detected(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    assert _resolve_rank_world(_args()) == (3, 8)


def test_slurm_node_fallback(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("SLURM_NODEID", "2")
    monkeypatch.setenv("SLURM_NNODES", "4")
    assert _resolve_rank_world(_args()) == (2, 4)


def test_precedence_flags_beat_env(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    assert _resolve_rank_world(_args(rank=1, world_size=2)) == (1, 2)


def test_precedence_generic_env_beats_slurm(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "6")
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    assert _resolve_rank_world(_args()) == (5, 6)


def test_precedence_databuilder_env_wins(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DATABUILDER_RANK", "7")
    monkeypatch.setenv("RANK", "5")
    assert _resolve_rank_world(_args())[0] == 7


def test_topology_warning_multiple_tasks_per_node(monkeypatch, caplog):
    _clear(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "16")
    monkeypatch.setenv("SLURM_NNODES", "8")
    with caplog.at_level(logging.WARNING, logger="databuilder"):
        _resolve_rank_world(_args())
    assert any("ONE process per node" in r.message for r in caplog.records)


def test_no_topology_warning_one_task_per_node(monkeypatch, caplog):
    _clear(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_NNODES", "8")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "1")
    with caplog.at_level(logging.WARNING, logger="databuilder"):
        _resolve_rank_world(_args())
    assert not [r for r in caplog.records if "ONE process per node" in r.message]
