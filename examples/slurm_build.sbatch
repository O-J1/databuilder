#!/bin/bash
#SBATCH --job-name=databuilder
#SBATCH --nodes=8                 # world size = number of nodes
#SBATCH --ntasks-per-node=1       # REQUIRED: one databuilder process per node
#SBATCH --gpus-per-node=8         # embed stage uses every GPU on the node
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=databuilder_%j_%t.log

# Usage: sbatch examples/slurm_build.sbatch build.toml
#
# Requirements:
#   - work_dir and data_dir in the config MUST be on a shared filesystem.
#   - Rank/world size are auto-detected from SLURM_PROCID / SLURM_NTASKS;
#     no --rank/--world-size flags needed.
#   - Rank 0 runs the global stages (dedup, cluster, manifest); other ranks
#     wait at marker-file barriers and resume automatically. Re-submitting
#     the job skips completed stages.

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm_build.sbatch <build.toml>}"

# activate your environment here, e.g.:
# source /shared/venvs/databuilder/bin/activate

srun databuilder run --config "$CONFIG"

# Afterwards, inspect clusters from your laptop (viewer binds 127.0.0.1 only):
#   databuilder viz-prepare --config build.toml --sample 200000   # on one node
#   databuilder viz --config build.toml --port 8765
#   ssh -J user@login-node -L 8765:127.0.0.1:8765 user@compute-node
