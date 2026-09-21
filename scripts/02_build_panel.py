"""Build data/processed/panel.npz from raw prices."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from pvol.data import build_panel  # noqa: E402

if __name__ == "__main__":
    p = build_panel()
    fv = int(p["first_valid"])
    print("T, M, F:", p["X"].shape, "pairs:", p["P"].shape)
    print("first valid index", fv, p["dates"][fv], "last", p["dates"][-1])
    X = p["X"][fv:]
    print("any NaN after first_valid:", np.isnan(X).any(), np.isnan(p["P"][fv:]).any())
    for k, n in enumerate(p["feat_names"]):
        v = X[..., k]
        print(f"{n:20s} mean {v.mean():8.3f} sd {v.std():7.3f} min {v.min():8.3f} max {v.max():8.3f}")
    eig = np.linalg.eigvalsh(p["S_ewma"][fv:].astype(np.float64))
    print("min eigenvalue of EWMA cov over sample:", eig.min(), " median cond:", np.median(eig[:, -1] / np.maximum(eig[:, 0], 1e-20)))
