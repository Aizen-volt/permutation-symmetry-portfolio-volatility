"""Sixth series (V1-V3), hypotheses and decision rules fixed before running (docs/JOURNAL.md, section 4g).

V1: MLP sorted by descending weight with ties broken by asset index (MLP-tsort, settings of MLP-sorted).
V2: Linear without one group of features (Linear-noasset, -noeuler, -nopairs, -noport) and with the portfolio-level
    backcasts only (Linear-portonly).
V3: capacity-matched pairs: MLP-small (MLP with hidden (16, 16), about the size of Deep Sets) and DeepSets-L (Deep Sets
    with d = 400, about the size of the MLP); all other settings as tuned.
Main design, h = 1, 5, 22, seeds 0-2, every test year.

Usage: [PVOL_UNIVERSE=stocks] python scripts/42_sixth_series.py V1 V2 V3
Resumable: finished (tag, seed) pairs are skipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment, load_tuned_hp  # noqa: E402

V2_MODELS = ["Linear-noasset", "Linear-noeuler", "Linear-nopairs", "Linear-noport", "Linear-portonly"]

if __name__ == "__main__":
    todo = sys.argv[1:] or ["V1", "V2", "V3"]
    ex = Experiment("main", load_panel())
    hp = load_tuned_hp()
    for e in todo:
        for h in (5, 1, 22):
            for seed in (0, 1, 2):
                if e == "V1":
                    ex.run_nn("MLP-tsort", h, [seed], hp=hp["MLP-sorted"], tag="MLP-tsort")
                elif e == "V2":
                    for m in V2_MODELS:
                        ex.run_nn(m, h, [seed], hp={})
                elif e == "V3":
                    ex.run_nn("MLP", h, [seed], hp={**hp["MLP"], "hidden": [16, 16]}, tag="MLP-small")
                    ex.run_nn("DeepSets", h, [seed], hp={**hp["DeepSets"], "d": 400, "phi": [400], "rho": [400]},
                              tag="DeepSets-L")
                else:
                    raise SystemExit(f"unknown experiment {e}")
        print(f"{e} finished", flush=True)
