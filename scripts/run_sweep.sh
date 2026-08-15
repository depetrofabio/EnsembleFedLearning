#!/usr/bin/env bash
# Full experimental matrix for the paper. Run from the repository root on a GPU.
#
#   bash scripts/run_sweep.sh tier1     # minimum defensible set  (~12 GPU-h on a T4)
#   bash scripts/run_sweep.sh tier2     # + ablations             (~10 GPU-h more)
#   bash scripts/run_sweep.sh all
#
# Every run writes results.json / table.md / config_used.json under
# hefl/results/<name>/, so the whole matrix can be aggregated afterwards
# without re-running anything.
#
# Each run is independent — safe to split across machines, or to Ctrl-C and
# resume by commenting out the lines that already finished.

set -uo pipefail
PY="${PY:-python}"
TIER="${1:-tier1}"
LOGDIR="hefl/results/_logs"
mkdir -p "$LOGDIR"

run () {   # run <name> <config> [extra args...]
  local name="$1"; shift
  local cfg="$1"; shift
  if [ -f "hefl/results/${name}/table.md" ]; then
    echo "SKIP  ${name} (already has a table.md)"
    return
  fi
  echo "RUN   ${name}"
  $PY -u -m hefl.run --config "$cfg" --output_dir "./hefl/results/${name}" "$@" \
      > "${LOGDIR}/${name}.log" 2>&1
  if [ $? -eq 0 ]; then echo "OK    ${name}"; else echo "FAIL  ${name} -> ${LOGDIR}/${name}.log"; fi
}

MAIN=hefl/configs/rotation_dirichlet.json

# ---------------------------------------------------------------- tier 1 ----
# The minimum set a reviewer needs: the main setting on 3 seeds, the two
# controls that isolate each heterogeneity axis, and the assignment ablation
# WITHOUT which the whole result is unfalsifiable (K models vs 1 is a capacity
# confound until random assignment is shown to be worse).
tier1 () {
  for s in 42 43 44; do
    run "main_seed${s}"            "$MAIN"                                  --seed "$s"
  done
  for s in 42 43 44; do
    run "rotonly_seed${s}"         hefl/configs/rotation_only.json   --seed "$s"
    run "dironly_seed${s}"         hefl/configs/dirichlet_only.json  --seed "$s"
  done
  # the falsifiability controls, on the main setting
  for s in 42 43 44; do
    run "main_random_seed${s}"     "$MAIN" --assignment random --seed "$s"
    run "main_oracle_seed${s}"     "$MAIN" --assignment oracle --seed "$s"
  done
}

# ---------------------------------------------------------------- tier 2 ----
# Ablations. One seed each is defensible for a sweep as long as tier 1 carries
# the error bars.
tier2 () {
  # how many clusters (K = number of true groups is 4 here)
  for k in 2 3 6 8; do
    run "ablate_K${k}"             "$MAIN" --num_clusters "$k"
  done
  # label-skew strength
  for a in 0.1 1.0 10.0; do
    run "ablate_alpha${a}"         "$MAIN" --alpha "$a"
  done
  # which signal drives the assignment
  for sig in delta_l4fc delta_stem; do
    run "ablate_signal_${sig}"     "$MAIN" --cluster_signal "$sig"
  done
  # server-side combiner fitting instead of federated dense rounds
  run   "ablate_dense_server"      "$MAIN" --dense_mode server
  # logit adjustment for the label axis (cheap, often the largest single win)
  run   "ablate_logit_adjust"      "$MAIN" --logit_adjust
}

case "$TIER" in
  tier1) tier1 ;;
  tier2) tier2 ;;
  all)   tier1; tier2 ;;
  *)     echo "usage: $0 [tier1|tier2|all]"; exit 1 ;;
esac

echo
echo "Done. Aggregate with:  $PY -m hefl.aggregate"
