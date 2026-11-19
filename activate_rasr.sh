#!/usr/bin/env bash
# Usage (from Project/): source ./activate_rasr.sh
RASR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$RASR_ROOT/H-JEPA/.venv/bin/activate"
export HJEPA_HOME="$RASR_ROOT/hjepa_home"
export PYTHONPATH="$RASR_ROOT:$RASR_ROOT/H-JEPA/h_jepa:$RASR_ROOT/H-JEPA${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$RASR_ROOT/.cache/matplotlib"; mkdir -p "$MPLCONFIGDIR"
export PYTHONWARNINGS=ignore
unset RASR_ROOT
# limit CPU threads per process (several workers share the 32-core box)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
