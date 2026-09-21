"""Fixed evaluation portfolios (held constant over the whole test period, rebalanced daily to constant weights)."""
import json

import numpy as np

STRUCTURED = {
    "60/40": {"SPY": 0.6, "AGG": 0.4},
    "AllWeather": {"SPY": 0.30, "TLT": 0.40, "IEF": 0.15, "GLD": 0.075, "DBC": 0.075},
    "GlobalMulti": {"SPY": 0.25, "EFA": 0.15, "EEM": 0.10, "AGG": 0.25, "TIP": 0.05, "VNQ": 0.05, "GLD": 0.05, "DBC": 0.05, "HYG": 0.05},
}

# group name -> (N, weight scheme, dirichlet alpha); 200 random portfolios per group
RANDOM_GROUPS = {
    "N5": (5, "long_only", 1.0),
    "N10": (10, "long_only", 1.0),
    "N20": (20, "long_only", 1.0),
    "N30": (30, "long_only", 1.0),  # larger than any training portfolio
    "N40": (40, "long_only", 1.0),
    "N10-conc": (10, "long_only", 0.3),
    "N10-EW": (10, "long_only", np.inf),
    "N10-LS": (10, "long_short", None),  # long-short weights never seen in training
}


def random_weights(rng, N, scheme, alpha):
    if scheme == "long_only":
        if np.isinf(alpha):
            return np.full(N, 1.0 / N)
        return rng.dirichlet(np.full(N, alpha))
    z = rng.standard_normal(N)
    return z / np.abs(z).sum()


def make_eval_portfolios(tickers, pool_ids=None, n_per_group=200, seed=12345, groups=None, structured=True):
    """List of dicts {group, ids, weights}. pool_ids restricts the random draws (e.g. to held-out assets)."""
    rng = np.random.default_rng(seed)
    pool = np.arange(len(tickers)) if pool_ids is None else np.asarray(pool_ids)
    out = []
    for g, (N, scheme, alpha) in (groups or RANDOM_GROUPS).items():
        if N > len(pool):
            continue
        for _ in range(n_per_group):
            ids = rng.choice(pool, size=N, replace=False)
            out.append(dict(group=g, ids=ids.tolist(), weights=random_weights(rng, N, scheme, alpha).tolist()))
    if structured:
        tix = {t: i for i, t in enumerate(tickers)}
        for g, spec in STRUCTURED.items():
            if all(t in tix for t in spec):  # structured allocations exist only for the ETF universe
                out.append(dict(group=g, ids=[tix[t] for t in spec], weights=list(spec.values())))
        out.append(dict(group="EW-all", ids=list(range(len(tickers))), weights=[1 / len(tickers)] * len(tickers)))
    return out


def save(portfolios, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portfolios))


def load(path):
    return json.loads(path.read_text())
