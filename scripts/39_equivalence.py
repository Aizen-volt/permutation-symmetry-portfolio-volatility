"""Confidence intervals and equivalence tests for S1b (symmetric network vs Linear) and H1 (symmetric vs counterpart).

The standard error of every mean daily loss differential is recovered from the HLN-corrected Diebold-Mariano statistic
already stored with it (se = mean / DM), so the intervals use exactly the variance of the tests. Differentials and
interval bounds are expressed in percentage points of the in-range EWMA loss. TOST: the two models are equivalent within
a margin m if the 90% interval lies inside (-m, m), i.e. both one-sided tests reject at 5%; m = 1 and 2 points. The
claim that all five networks of a family are equivalent is an intersection-union test and needs no correction; for
the five individual claims jointly, Bonferroni: both one-sided tests at 5%/5, i.e. the 98% interval inside (-m, m).
Writes results*/main/tables/equivalence.csv for the current universe.

Usage: [PVOL_UNIVERSE=stocks] python scripts/39_equivalence.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from pvol.config import RESULTS  # noqa: E402

T = RESULTS / "main" / "tables"
MARGINS = (1.0, 2.0)
FAMILY = 5  # symmetric networks per family

f4 = pd.read_csv(T / "hypotheses_fourth.csv")
s1b = f4[(f4.hypothesis == "S1b") & (f4.benchmark == "Linear") & (f4.block == "in-range")]
h1 = pd.read_csv(T / "hypotheses.csv")
h1 = h1[(h1.hypothesis == "H1") & (h1.block == "in-range")]
rows = []
for fam, d in (("S1b", s1b), ("H1", h1)):
    for _, r in d.iterrows():
        ewma = pd.read_csv(T / f"losses_h{r.h}.csv").query("block == 'in-range' and model == 'EWMA'").qlike.iloc[0]
        md, se = 100 * r.mean_diff / ewma, 100 * abs(r.mean_diff / r.dm) / ewma
        rec = dict(family=fam, h=r.h, model=r.model, mean_diff_pp=md, se_pp=se,
                   lo95=md - norm.ppf(0.975) * se, hi95=md + norm.ppf(0.975) * se,
                   lo90=md - norm.ppf(0.95) * se, hi90=md + norm.ppf(0.95) * se,
                   lo98=md - norm.ppf(1 - 0.05 / FAMILY) * se, hi98=md + norm.ppf(1 - 0.05 / FAMILY) * se)
        for m in MARGINS:
            rec[f"equivalent_{m:g}pp"] = bool(-m < rec["lo90"] and rec["hi90"] < m)
            rec[f"equivalent_{m:g}pp_bonf"] = bool(-m < rec["lo98"] and rec["hi98"] < m)
        rows.append(rec)
out = pd.DataFrame(rows)
out.to_csv(T / "equivalence.csv", index=False)
print(out.groupby(["family", "h"]).agg(max_abs95=("hi95", lambda s: max(s.abs().max(), out.loc[s.index, "lo95"].abs().max())),
                                       eq1=("equivalent_1pp", "sum"), eq2=("equivalent_2pp", "sum"),
                                       eq2_bonf=("equivalent_2pp_bonf", "sum"), n=("model", "size")).round(1))
