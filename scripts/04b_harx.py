"""HAR-X benchmark (added after the first baseline pass)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.config import HORIZONS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment  # noqa: E402

if __name__ == "__main__":
    p = load_panel()
    for name in sys.argv[1:] or ["main", "heldout"]:
        ex = Experiment(name, p)
        for h in HORIZONS:
            ex.run_baselines(h, which=("HARX",))
