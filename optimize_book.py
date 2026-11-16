"""
optimize_book.py -- turn fair-value gaps into a near beta-neutral long-short book
that trades off expected return against volatility.

Problem (V = signed dollar positions, in $ millions inside the solver):
    maximize   mu'V - (lam/2) V' Sigma V          (expected P&L minus risk penalty)
    subject to long book = 50, short book = 50    (so gross = 100, net = 0)
               |V_i| >= 1.5 on the chosen side     (minimum size, every name held)
               |V_i| <= max_pos                    (concentration cap, your choice)
               |b_k'V| <= tol for each beta factor (near beta-neutral)
               optional: each group nets to zero

The long/short SIGN of each name is a discrete choice, which makes the problem
non-convex. With 8 names there are only 2^8 sign patterns, so we solve a small
convex QP for every feasible pattern and keep the best objective.

Everything here is a template: inputs (mu, cov, betas) come from your own data.
Run the file to see a demo on SYNTHETIC numbers (not real stock data).
"""
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from group_portfolio import GROUPS, TOTAL_GMV, MIN_POSITION, check_constraints


# ---------------------------- inputs -----------------------------------------
def alpha_from_fair_value(price, fair_value, sigma_month, ic=0.03):
    """Expected monthly return from fair-value gaps (Grinold: alpha = IC * sigma * z).
    price and fair_value must be in the SAME currency and per the SAME share.
    ic (monthly information coefficient) is an ASSUMPTION: label it as such."""
    gap = np.log(pd.Series(fair_value) / pd.Series(price))
    z = (gap - gap.mean()) / gap.std(ddof=0)       # cross-sectional z-score
    return ic * pd.Series(sigma_month) * z


def shrunk_cov(returns, periods_per_month, delta=0.2):
    """Monthly covariance from periodic returns, shrunk toward a diagonal matrix.
    delta is an assumption (sklearn's LedoitWolf picks it from the data instead)."""
    S = returns.cov().values * periods_per_month
    out = (1 - delta) * S + delta * np.diag(np.diag(S))
    return pd.DataFrame(out, index=returns.columns, columns=returns.columns)


def estimate_betas(returns, factors, lagged=()):
    """Beta of each column of `returns` on each column of `factors`, then Blume
    shrinkage (0.67*b + 0.33). Names in `lagged` (markets closing before New York,
    e.g. KRX-based proxies) use a Dimson beta: coefficient on the factor today PLUS
    the coefficient on the factor's previous period, to absorb non-synchronous closes."""
    out = pd.DataFrame(index=returns.columns, columns=factors.columns, dtype=float)
    for f in factors.columns:
        for a in returns.columns:
            y = returns[a].values[1:]
            cols = [np.ones(len(y)), factors[f].values[1:]]
            if a in lagged:
                cols.append(factors[f].values[:-1])
            coef = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0]
            out.loc[a, f] = coef[1:].sum()
    return 0.67 * out + 0.33


# ---------------------------- optimizer --------------------------------------
def _solve_pattern(s, mu, cov, B, lam, tol, gmv, lo, hi, group_idx):
    bounds = [(lo, hi) if si > 0 else (-hi, -lo) for si in s]
    cons = [{"type": "eq", "fun": lambda V: V[s > 0].sum() - gmv / 2},
            {"type": "eq", "fun": lambda V: V[s < 0].sum() + gmv / 2}]
    for b in B:                                    # |b'V| <= tol, one pair per factor
        cons += [{"type": "ineq", "fun": lambda V, b=b: tol - b @ V},
                 {"type": "ineq", "fun": lambda V, b=b: tol + b @ V}]
    for idx in group_idx:                          # optional group neutrality
        cons.append({"type": "eq", "fun": lambda V, idx=idx: V[idx].sum()})
    V0 = np.where(s > 0, gmv / 2 / (s > 0).sum(), -gmv / 2 / (s < 0).sum())
    res = minimize(lambda V: -mu @ V + 0.5 * lam * V @ cov @ V, V0,
                   jac=lambda V: -mu + lam * cov @ V, bounds=bounds,
                   constraints=cons, method="SLSQP",
                   options={"maxiter": 300, "ftol": 1e-12})
    ok = res.success and all(
        (c["fun"](res.x) >= -1e-6) if c["type"] == "ineq" else abs(c["fun"](res.x)) <= 1e-6
        for c in cons)
    return res, ok


def optimize_book(mu, cov, betas, lam=0.1, beta_tol_schedule=(0.5, 1.0, 2.0, 4.0),
                  group_neutral=False, max_pos=20.0):
    """Return (positions_df, info). Tolerances are in $M of beta-weighted exposure.
    The schedule is the PRE-SPECIFIED infeasibility procedure: try the tightest
    tolerance first, loosen only if no sign pattern is feasible; log which was used."""
    names = list(mu.index)
    n, gmv, lo = len(names), TOTAL_GMV / 1e6, MIN_POSITION / 1e6
    mu_v, cov_m = mu.loc[names].values, cov.loc[names, names].values
    B = betas.loc[names].values.T                  # shape (n_factors, n_names)
    group_idx = ([[names.index(m) for m in mem] for mem in GROUPS.values()]
                 if group_neutral else [])
    patterns = []
    for s in itertools.product([1, -1], repeat=n):
        s = np.array(s)
        nl, ns = (s > 0).sum(), (s < 0).sum()
        if nl == 0 or ns == 0:
            continue
        if not (nl * lo <= gmv / 2 <= nl * max_pos and ns * lo <= gmv / 2 <= ns * max_pos):
            continue
        if group_neutral and any(len({s[i] for i in g}) < 2 for g in group_idx):
            continue
        patterns.append(s)
    for tol in beta_tol_schedule:
        best = None
        for s in patterns:
            res, ok = _solve_pattern(s, mu_v, cov_m, B, lam, tol, gmv, lo, max_pos, group_idx)
            if ok and (best is None or res.fun < best[0].fun):
                best = (res, s)
        if best is not None:
            V = best[0].x
            pos = pd.DataFrame({"issuer": names,
                                "side": np.where(V > 0, "long", "short"),
                                "target_usd": V * 1e6, "mu_month": mu_v})
            for f, col in zip(betas.columns, B):
                pos["beta_" + str(f)] = betas.loc[names, f].values
            info = {"beta_tol_used_$M": tol, "patterns_tried": len(patterns),
                    "exp_pnl_month_$": float(mu_v @ V * 1e6),
                    "vol_month_$": float(np.sqrt(V @ cov_m @ V) * 1e6),
                    "beta_exposure_$M": {f: float(b @ V) for f, b in zip(betas.columns, B)}}
            return pos, info
    raise RuntimeError("No feasible book: apply your documented infeasibility rule.")


# ---------------------------- demo (SYNTHETIC data) --------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)                # fixed seed for replay
    names = ["NVIDIA", "AMD", "Intel", "Micron", "ASML", "TSMC", "SK hynix", "Samsung"]
    T = 260                                        # ~5 years of weekly returns
    mkt = pd.Series(rng.normal(0.002, 0.025, T), name="market")
    sec = pd.Series(rng.normal(0.0, 0.03, T), name="sector")
    loads = rng.uniform(0.8, 1.6, len(names))
    rets = pd.DataFrame({n: loads[i] * mkt + 0.8 * sec + rng.normal(0, 0.035, T)
                         for i, n in enumerate(names)})
    factors = pd.concat([mkt, sec], axis=1)
    cov = shrunk_cov(rets, periods_per_month=52 / 12)
    betas = estimate_betas(rets, factors, lagged=("SK hynix", "Samsung"))
    price = pd.Series(rng.uniform(50, 300, len(names)), index=names)
    fv = price * pd.Series(rng.uniform(0.7, 1.4, len(names)), index=names)
    sigma_m = pd.Series(np.sqrt(np.diag(cov)), index=names)
    mu = alpha_from_fair_value(price, fv, sigma_m, ic=0.03)

    pos, info = optimize_book(mu, cov, betas, lam=0.1, group_neutral=False)
    with pd.option_context("display.width", 150, "display.float_format", "{:,.3f}".format):
        print(pos.to_string(index=False))
    print(info)
    print(check_constraints(pos).to_string(index=False))
    print("\nFrontier (lam: expected P&L vs volatility, $/month):")
    for lam in (0.02, 0.1, 0.5, 2.0):
        _, i = optimize_book(mu, cov, betas, lam=lam)
        print(f"  lam={lam:<5} exp P&L {i['exp_pnl_month_$']:>11,.0f}   vol {i['vol_month_$']:>11,.0f}")