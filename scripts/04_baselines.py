"""Targets and classical benchmark forecasts for all experiments/horizons/test years."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.config import HORIZONS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment  # noqa: E402

if __name__ == "__main__":
    exps = sys.argv[1:] or ["main", "heldout"]
    p = load_panel()
    for name in exps:
        ex = Experiment(name, p)
        for h in HORIZONS:
            ex.write_targets(h)
            ex.run_baselines(h)
