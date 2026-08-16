# Recovered artefacts

This run completed every computation but crashed on its final write: the output
directory was renamed (`ensemble_v2/` -> `hefl/`) while the process was running,
so `results.json`, `config_used.json` and the model checkpoints were never
written.

`table.md` here is transcribed **verbatim** from the run's stdout, which `run.py`
prints before saving. `results.json` was reconstructed from the same log and
contains only values that appear in it; fields the log does not record
(per-round history beyond the evaluated rounds, subset analysis) are absent.

Model checkpoints are lost. Re-run to regenerate:

    python -m hefl.run --config hefl/configs/rotation_dirichlet_mps.json
