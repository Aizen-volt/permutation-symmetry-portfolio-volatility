"""Multiple-testing corrections and decision rule of the scripts, and their consistency with the result tables."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from conftest import ROOT, load_script

hyp = load_script("15_hypotheses.py")
third = load_script("24_third_hypotheses.py")
mult = load_script("32_precision_multiplicity.py")
supp = load_script("46_supplement_tables.py")
REP = ROOT / "replication"
UNIVERSES = [REP / "results", REP / "results_stocks"]
needs_tables = pytest.mark.skipif(not (REP / "results" / "main" / "tables").exists(), reason="no result tables")


@pytest.mark.parametrize("seed", range(5))
def test_holm_matches_statsmodels(seed):
    p = np.random.default_rng(seed).uniform(0, 0.2, 9)
    assert np.allclose(hyp.holm(p), multipletests(p, method="holm")[1])


@pytest.mark.parametrize("seed", range(5))
def test_bh_matches_statsmodels(seed):
    p = np.random.default_rng(seed).uniform(0, 0.2, 9)
    assert np.allclose(mult.bh(p), multipletests(p, method="fdr_bh")[1])


def test_holm_edge_cases():
    assert np.allclose(hyp.holm([0.01]), [0.01])
    assert np.allclose(hyp.holm([0.5, 0.6]), [1.0, 1.0])
    assert np.allclose(hyp.holm([0.01, 0.01, 0.04]), [0.03, 0.03, 0.04])


def test_verdict_rule():
    assert hyp.verdict(-0.1, 0.01) == "more accurate"
    assert hyp.verdict(0.1, 0.01) == "less accurate"
    assert hyp.verdict(-0.1, 0.05) == "indistinguishable"
    assert hyp.verdict(-0.1, 0.2) == "indistinguishable"


def test_family_adds_adjusted_p():
    rows = [dict(mean_diff=-1, p=0.01), dict(mean_diff=1, p=0.02), dict(mean_diff=-1, p=0.5)]
    out = hyp.family(rows)
    assert [r["verdict"] for r in out] == ["more accurate", "less accurate", "indistinguishable"]
    assert np.allclose([r["p_holm"] for r in out], [0.03, 0.04, 0.5])
    assert hyp.family([]) == []


def test_compare_and_comparisons(rng):
    T = 400
    La = rng.standard_normal((T, 6)) ** 2
    Lb = La + 0.5
    cols = np.array([0, 2, 4])
    res, d = hyp.compare(La, Lb, cols, 1)
    assert np.isclose(res["mean_diff"], -0.5) and np.allclose(d, -0.5)
    L = {"A": La + rng.normal(0, 0.1, (T, 6)), "B": Lb}
    rows = third.comparisons(L, cols, 1, "X", "in-range", [("A", "B"), ("A", "missing")])
    assert len(rows) == 1 and rows[0]["verdict"] == "more accurate" and rows[0]["hypothesis"] == "X"


def test_cols_of_blocks():
    gidx = {"N5": np.array([0, 1]), "N10": np.array([2]), "N30": np.array([3])}
    assert hyp.cols_of(gidx, "in-range").tolist() == [0, 1, 2]
    assert hyp.cols_of(gidx, "larger-N").tolist() == [3]


def test_family_rows(tmp_path):
    t = tmp_path / "main" / "tables"
    t.mkdir(parents=True)
    pd.DataFrame(dict(hypothesis=["H1", "H1", "H2"], h=[1, 5, 1])).to_csv(t / "x.csv", index=False)
    assert len(mult.family_rows(tmp_path, "x.csv", dict(hypothesis="H1"))) == 2
    assert mult.family_rows(tmp_path, "x.csv", dict(hypothesis="H9")) is None
    assert mult.family_rows(tmp_path, "x.csv", dict(other=1)) is None
    assert mult.family_rows(tmp_path, "missing.csv", {}) is None


def test_supplement_formatting():
    assert supp.f(-1.234) == "$-$1.23" and supp.f(np.nan) == "--" and supp.f(0.5, 3) == "0.500"
    assert supp.pval(0.0004) == "<0.001" and supp.pval(0.0123) == "0.012"
    assert supp.mname("DeepSets-RAW") == "Deep Sets" and supp.mname("PENet-T") == "PE-Net"
    assert supp.nm("Unknown") == "Unknown"


def test_with_effects(tmp_path):
    t = tmp_path / "main" / "tables"
    t.mkdir(parents=True)
    for h in supp.HS:
        pd.DataFrame(dict(block=["in-range"], model=["EWMA"], qlike=[2.0])).to_csv(t / f"losses_h{h}.csv", index=False)
        (tmp_path / "main" / f"h{h}").mkdir()
        np.save(tmp_path / "main" / f"h{h}" / "origins.npy", np.arange(1000))
    d = pd.DataFrame(dict(h=[5], mean_diff=[-0.02], dm=[-2.0]))
    out = supp.with_effects(d, tmp_path)
    assert np.isclose(out.eff[0], -1.0)  # 100 * -0.02 / 2
    half = (out.hi[0] - out.lo[0]) / 2
    assert np.isclose(half, 0.5 * 1.9623, rtol=1e-3)  # t quantile with 999 df times SE of 0.5 points


def test_write_table(tmp_path, monkeypatch):
    monkeypatch.setattr(supp, "OUT", tmp_path)
    supp.write("t", "Cap", "tab:t", "lr", ["a & b\\\\"], ["1 & 2\\\\"])
    tex = (tmp_path / "t.tex").read_text()
    assert "\\caption{Cap}\\label{tab:t}" in tex and "\\begin{tabular}{lr}" in tex and tex.count("rule") == 3


# consistency of the stored result tables (replication/) with the decision rule
def hypothesis_tables():
    return [f for u in UNIVERSES for f in sorted((u / "main" / "tables").glob("hypotheses*.csv"))]


@needs_tables
@pytest.mark.parametrize("path", hypothesis_tables(), ids=lambda p: f"{p.parts[-4]}/{p.name}")
def test_stored_verdicts_follow_rule(path):
    d = pd.read_csv(path)
    if "verdict" not in d.columns or "p_holm" not in d.columns:
        pytest.skip("no verdicts")
    d = d.dropna(subset=["p_holm", "mean_diff"])
    assert (d.p_holm >= d.p - 1e-12).all() and (d.p_holm <= 1 + 1e-12).all()
    expect = [hyp.verdict(m, p) for m, p in zip(d.mean_diff, d.p_holm)]
    assert list(d.verdict) == expect


@needs_tables
def test_stored_global_correction():
    g = pd.read_csv(REP / "results" / "main" / "tables" / "global_multiplicity.csv")
    assert len(g) == 625
    assert np.allclose(g.p_global_holm, hyp.holm(g.p.to_numpy()))
    assert np.allclose(g.p_global_bh, mult.bh(g.p.to_numpy()))
    assert (g.p_global_holm >= g.p_holm - 1e-12).all()


@needs_tables
@pytest.mark.parametrize("u", UNIVERSES, ids=["etf", "stocks"])
def test_stored_holm_within_families(u):
    d = pd.read_csv(u / "main" / "tables" / "hypotheses.csv")
    for _, g in d.groupby(["hypothesis", "h", "block"]):
        assert np.allclose(g.p_holm, hyp.holm(g.p.to_numpy()))


@needs_tables
@pytest.mark.parametrize("u", UNIVERSES, ids=["etf", "stocks"])
def test_stored_losses_relative_to_ewma(u):
    for h in (1, 5, 22):
        d = pd.read_csv(u / "main" / "tables" / f"losses_h{h}.csv")
        assert (d.qlike > 0).all()
        assert d[(d.block == "in-range")].model.is_unique


@needs_tables
def test_stored_tuning_choices_exist():
    hp = json.loads(Path(REP / "results" / "tuning" / "best_hp.json").read_text())
    assert {"MLP", "MLP-sorted", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov", "GRU"} <= set(hp)
    assert hp["MLP-sorted"]["dropout"] == 0.1 and hp["MLP"]["dropout"] == 0.3
