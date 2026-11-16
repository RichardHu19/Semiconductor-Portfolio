"""
max_return_book.py -- long-short weights that MAXIMIZE expected return subject to
    * dollar neutrality: $50M long book, $50M short book (gross $100M, net $0)
    * minimum size: every one of the 8 names held, |V_i| >= $1.5M, <= max_pos
    * near-zero beta: |sum_i beta_i * V_i| <= tolerance
    * a volatility cap: sqrt(V' Sigma V) <= vol_cap  (dollar volatility over the horizon)
    * optionally: each group (group_portfolio.GROUPS) nets to zero

INPUTS (all required)
  --inputs  CSV with columns: issuer, price, fair_value, implied_vol [, price_usd]
            price and fair_value: SAME currency and SAME share (a ratio is used).
            implied_vol: ANNUALIZED, decimals (0.45 = 45%).
            price_usd (optional): USD price per share, only used for indicative shares.
  --cov     CSV square covariance matrix, issuers as row and column labels.
            Optional extra row/column named MARKET (e.g. SPY) -> betas from it.
  --vol-cap dollar volatility cap over the horizon, e.g. 3500000

HOW THE INPUTS ARE USED
  expected return  mu_i = IC * sigma_i * z_i, with z = cross-sectional z-score of
                   ln(fair_value/price) and sigma_i the horizon implied vol (Grinold).
                   IC is an ASSUMPTION (--ic, default 0.03).
  covariance       Sigma = D R D: correlations R come from your covariance matrix, the
                   volatilities D from implied vol (forward-looking). --cov-as-is skips
                   this and uses your matrix unchanged (it must be in horizon units).
  beta             from --betas-csv, else the MARKET row of the covariance matrix,
                   else the beta to an equal-weight basket of the 8 names (a SECTOR beta).
  signs            long/short is a discrete choice, so every sign pattern (2^8) is
                   tried; each is a small continuous problem (SLSQP). Best return wins.

INFEASIBILITY RULE (predefined, as the assignment requires): if no book meets the vol
cap, the beta tolerance is doubled up to three times; if still infeasible the program
stops and reports the minimum achievable volatility (or, with --fallback-minvol,
returns the minimum-variance book and says so).

STANDALONE: needs only numpy, pandas and scipy (no other project files).

ROUNDING: final positions are rounded to --round-step (default $100,000) by trying every
down/up rounding and keeping the best one that still meets the vol cap and beta tolerance.

Run: python return_book --inputs inputs.csv --cov cov.csv --vol-cap $6000000
Run on SYNTHETIC data:  python return_book.py --demo
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---- Mandate settings and peer groups (edit here; no other files are needed) ----
TOTAL_GMV = 100_000_000          # $100M gross: $50M long book, $50M short book
MIN_POSITION = 1_500_000         # $1.5M minimum per holding
TOL = 1.0                        # $1 rounding tolerance
GROUPS = {                       # used only with --group-neutral; names must match issuers
    "Memory":         ["Micron", "SK hynix", "Samsung"],
    "AI compute":     ["NVIDIA", "AMD"],
    "Foundry/equip.": ["TSMC", "ASML", "Intel"],
}


class InfeasibleError(Exception):
    pass


def alpha_from_fair_value(price, fair_value, sigma_month, ic=0.5):
    """Expected horizon return from fair-value gaps (Grinold: alpha = IC * sigma * z).
    price and fair_value must be in the SAME currency and per the SAME share.
    ic is an ASSUMPTION: label it as such."""
    gap = np.log(pd.Series(fair_value) / pd.Series(price))
    z = (gap - gap.mean()) / gap.std(ddof=0)                  # cross-sectional z-score
    return ic * pd.Series(sigma_month) * z


def check_constraints(pos):
    """PASS/FAIL table for the mandated constraints. pos needs a target_usd column."""
    v = pos["target_usd"]
    checks = {
        "Gross = $100M": abs(v.abs().sum() - TOTAL_GMV) <= TOL,
        "Net = $0": abs(v.sum()) <= TOL,
        "Long book = $50M": abs(v[v > 0].sum() - TOTAL_GMV / 2) <= TOL,
        "Short book = $50M": abs(-v[v < 0].sum() - TOTAL_GMV / 2) <= TOL,
        "All 8 held": (v.abs() > 0).sum() == 8,
        "Min size $1.5M each": bool((v.abs() >= MIN_POSITION - TOL).all()),
    }
    return pd.DataFrame({"check": list(checks), "result": ["PASS" if ok else "FAIL" for ok in checks.values()]})


# ------------------------------ inputs ---------------------------------------
def make_psd_corr(R):
    """Clip negative eigenvalues (sample matrices can be slightly non-PSD) and
    renormalize to a unit diagonal."""
    w, v = np.linalg.eigh((R + R.T) / 2)
    R2 = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    d = np.sqrt(np.diag(R2))
    return R2 / np.outer(d, d)


def get_betas(cov_raw, names, betas_csv=None):
    if betas_csv:
        b = pd.read_csv(betas_csv).set_index("issuer")["beta"].reindex(names)
        return b, "betas file"
    if "MARKET" in cov_raw.index:
        return cov_raw.loc[names, "MARKET"] / cov_raw.loc["MARKET", "MARKET"], "MARKET row of covariance"
    c = cov_raw.loc[names, names].values.astype(float)
    e = np.ones(len(names)) / len(names)
    return pd.Series(c @ e / (e @ c @ e), index=names), "equal-weight basket (SECTOR beta, not market beta)"


def build_sigma(cov_raw, names, iv_horizon, rescale=True):
    c = cov_raw.loc[names, names].values.astype(float)
    if not rescale:
        return pd.DataFrame(c, index=names, columns=names)
    d = np.sqrt(np.diag(c))
    R = make_psd_corr(c / np.outer(d, d))
    s = iv_horizon.loc[names].values
    return pd.DataFrame(R * np.outer(s, s), index=names, columns=names)


# ------------------------------ optimizer ------------------------------------
def _ok(res, cons, tol=1e-5):
    for c in cons:
        v = c["fun"](res.x)
        if (c["type"] == "eq" and abs(v) > tol) or (c["type"] == "ineq" and v < -tol):
            return False
    return bool(res.success)


def _solve_pattern(s, mu, S, b, tol, cap, gmv, lo, hi, group_idx):
    bounds = [(lo, hi) if si > 0 else (-hi, -lo) for si in s]
    cons = [{"type": "eq", "fun": lambda V: V[s > 0].sum() - gmv / 2},
            {"type": "eq", "fun": lambda V: V[s < 0].sum() + gmv / 2},
            {"type": "ineq", "fun": lambda V: tol - b @ V},
            {"type": "ineq", "fun": lambda V: tol + b @ V}]
    for idx in group_idx:
        cons.append({"type": "eq", "fun": lambda V, idx=idx: V[idx].sum()})
    V0 = np.where(s > 0, gmv / 2 / (s > 0).sum(), -gmv / 2 / (s < 0).sum())
    opt = {"maxiter": 300, "ftol": 1e-12}
    # Stage 1: minimum-variance book for this sign pattern (feasibility + warm start)
    r1 = minimize(lambda V: V @ S @ V, V0, jac=lambda V: 2 * S @ V, bounds=bounds,
                  constraints=cons, method="SLSQP", options=opt)
    if not _ok(r1, cons):
        return None
    V1 = np.clip(r1.x, [b_[0] for b_ in bounds], [b_[1] for b_ in bounds])
    vol1 = float(np.sqrt(V1 @ S @ V1))
    if vol1 > cap + 1e-6:
        return {"minvol": (V1, vol1), "best": None}
    # Stage 2: maximize expected return subject to the volatility cap
    cons2 = cons + [{"type": "ineq", "fun": lambda V: cap ** 2 - V @ S @ V}]
    r2 = minimize(lambda V: -mu @ V, V1, jac=lambda V: -mu, bounds=bounds,
                  constraints=cons2, method="SLSQP", options=opt)
    V2 = np.clip(r2.x, [b_[0] for b_ in bounds], [b_[1] for b_ in bounds]) if _ok(r2, cons2) else V1
    return {"minvol": (V1, vol1), "best": V2}


def solve_book(mu, Sigma, beta, vol_cap_usd, beta_tol_musd=1.0, max_pos_musd=20.0,
               group_neutral=False, fallback_minvol=False):
    names = list(mu.index)
    n, gmv, lo = len(names), TOTAL_GMV / 1e6, MIN_POSITION / 1e6
    cap = vol_cap_usd / 1e6
    mu_v, S, b = mu.values, Sigma.loc[names, names].values, beta.loc[names].values
    group_idx = ([[names.index(m) for m in mem] for mem in GROUPS.values()] if group_neutral else [])
    patterns = []
    for s in itertools.product([1, -1], repeat=n):
        s = np.array(s)
        nl, ns = (s > 0).sum(), (s < 0).sum()
        if nl == 0 or ns == 0:
            continue
        if not (nl * lo <= gmv / 2 <= nl * max_pos_musd and ns * lo <= gmv / 2 <= ns * max_pos_musd):
            continue
        if group_neutral and any(len({s[i] for i in g}) < 2 for g in group_idx):
            continue
        patterns.append(s)

    minvol_by_tol = {}                                           # tolerance -> (V, vol) of min-variance book
    for mult in (1, 2, 4, 8):                                     # predefined relaxation schedule
        tol = beta_tol_musd * mult
        best = None
        for s in patterns:
            r = _solve_pattern(s, mu_v, S, b, tol, cap, gmv, lo, max_pos_musd, group_idx)
            if r is None:
                continue
            if tol not in minvol_by_tol or r["minvol"][1] < minvol_by_tol[tol][1]:
                minvol_by_tol[tol] = r["minvol"]
            if r["best"] is not None and (best is None or mu_v @ r["best"] > mu_v @ best):
                best = r["best"]
        if best is not None:
            return _package(best, names, mu, Sigma, beta, tol, "max return under vol cap")
    if not minvol_by_tol:
        raise InfeasibleError("No sign pattern satisfies size, neutrality and beta constraints.")
    tol0 = min(minvol_by_tol)                                     # tightest tolerance that is feasible at all
    if fallback_minvol:
        V, _ = minvol_by_tol[tol0]
        return _package(V, names, mu, Sigma, beta, tol0,
                        "FALLBACK: minimum-variance book (vol cap not achievable)")
    detail = "; ".join(f"beta tol ${t * 1e6:,.0f}: min vol ${v[1] * 1e6:,.0f}"
                       for t, v in sorted(minvol_by_tol.items()))
    raise InfeasibleError(f"Vol cap ${vol_cap_usd:,.0f} is not achievable. Minimum achievable "
                          f"volatility -> {detail}. Raise the cap, or use --fallback-minvol.")


def _package(V, names, mu, Sigma, beta, tol, status):
    S, b = Sigma.loc[names, names].values, beta.loc[names].values
    pos = pd.DataFrame({"issuer": names, "side": np.where(V > 0, "long", "short"),
                        "target_usd": V * 1e6, "weight_of_gmv": V / (TOTAL_GMV / 1e6),
                        "mu_horizon": mu.values, "beta": b})
    info = {"status": status, "beta_tol_used_usd": tol * 1e6,
            "expected_pnl_usd": float(mu.values @ V * 1e6),
            "vol_usd": float(np.sqrt(V @ S @ V) * 1e6),
            "beta_exposure_usd": float(b @ V * 1e6)}
    info["return_per_vol"] = info["expected_pnl_usd"] / info["vol_usd"]
    return pos, info


# ------------------------------ rounding -------------------------------------
def _stats(V, mu, Sigma, beta):
    """Expected P&L, volatility and beta exposure (USD) of a signed-dollar position Series."""
    n, v = list(V.index), V.values
    S = Sigma.loc[n, n].values
    return {"expected_pnl_usd": float(mu.loc[n].values @ v), "vol_usd": float(np.sqrt(v @ S @ v)),
            "beta_exposure_usd": float(beta.loc[n].values @ v)}


def round_book(V, mu, Sigma, beta, vol_cap, beta_tol, step, max_pos, group_neutral=False):
    """Round signed USD positions V to multiples of `step` so that BOTH books stay at exactly
    $50M. Each position is rounded down or up (never more than one step away), all such
    combinations are tried (at most 2^8), and the one with the highest expected P&L that still
    meets the volatility cap, the beta tolerance (and group neutrality, if requested) wins.
    If no combination meets every limit, the least-violating one is returned and flagged.
    Rounding down never breaches the $1.5M minimum because $1.5M is a multiple of $100k."""
    names = list(V.index)
    sign, a = np.sign(V.values), np.abs(V.values) / step
    fl = np.floor(a + 1e-6)
    cands = [[f] + ([f + 1] if abs(x - f) > 1e-6 and (f + 1) * step <= max_pos + 1e-6 else [])
             for x, f in zip(a, fl)]
    combos = np.array(list(itertools.product(*cands)))               # magnitudes, in steps
    R = combos * step * sign                                         # signed USD
    side = TOTAL_GMV / 2 / step
    ok = (np.isclose(combos[:, sign > 0].sum(1), side) & np.isclose(combos[:, sign < 0].sum(1), side)
          & (np.abs(R) >= MIN_POSITION - 1e-6).all(1))
    if not ok.any():
        raise ValueError("No rounding keeps each side at exactly $50M with every position >= $1.5M "
                         "(check --round-step: it should divide $1.5M and $50M evenly).")
    S, b, m = Sigma.loc[names, names].values, beta.loc[names].values, mu.loc[names].values
    vol, bexp, er = np.sqrt(np.einsum("ij,jk,ik->i", R, S, R)), R @ b, R @ m
    gnet = np.zeros(len(R))
    if group_neutral:
        gnet = sum(np.abs(R[:, [names.index(x) for x in mem]].sum(1)) for mem in GROUPS.values())
    feas = ok & (vol <= vol_cap + 1e-6) & (np.abs(bexp) <= beta_tol + 1e-6) & (gnet <= 1.0)
    if feas.any():
        k, status = np.flatnonzero(feas)[np.argmax(er[feas])], "all limits met after rounding"
    else:
        viol = (np.maximum(vol / vol_cap - 1, 0) + np.maximum(np.abs(bexp) / beta_tol - 1, 0)
                + gnet / TOTAL_GMV)
        viol[~ok] = np.inf
        k, status = int(np.argmin(viol)), "WARNING: no rounding meets every limit; least-violating chosen"
    return pd.Series(R[k], index=names), status


# ------------------------------ driver ---------------------------------------
def run(inp, cov_raw, vol_cap, horizon_days=30, ic=1, beta_tol_musd=1.0, max_pos_musd=20.0,
        betas_csv=None, cov_as_is=False, group_neutral=False, fallback_minvol=False,
        round_step=100_000):
    names = list(inp.index)
    missing = [m for m in names if m not in cov_raw.index]
    if missing:
        raise ValueError(f"Not in covariance matrix: {missing}")
    iv_h = inp["implied_vol"] * np.sqrt(horizon_days / 365)       # horizon vol from annual IV
    mu = alpha_from_fair_value(inp["price"], inp["fair_value"], iv_h, ic=ic)
    beta, beta_src = get_betas(cov_raw, names, betas_csv)
    Sigma = build_sigma(cov_raw, names, iv_h, rescale=not cov_as_is)
    pos, info = solve_book(mu, Sigma, beta, vol_cap, beta_tol_musd, max_pos_musd,
                           group_neutral, fallback_minvol)
    info["rounding_step_usd"] = round_step
    if round_step:
        V0 = pos.set_index("issuer")["target_usd"]
        Vr, info["rounding_status"] = round_book(V0, mu, Sigma, beta, vol_cap, info["beta_tol_used_usd"],
                                                 round_step, max_pos_musd * 1e6, group_neutral)
        info["unrounded"] = _stats(V0, mu, Sigma, beta)
        pos.insert(3, "unrounded_usd", V0.values)
        pos["target_usd"], pos["weight_of_gmv"] = Vr.values, Vr.values / TOTAL_GMV
        info.update(_stats(Vr, mu, Sigma, beta))
        info["return_per_vol"] = info["expected_pnl_usd"] / info["vol_usd"]
        if group_neutral:
            info["group_net_usd"] = {g: float(Vr[mem].sum()) for g, mem in GROUPS.items()}
    if "price_usd" in inp.columns:
        pos["indicative_shares"] = pos["target_usd"].values / inp.loc[names, "price_usd"].values
    info["beta_source"] = beta_src
    return pos, info


def demo_inputs():
    """SYNTHETIC inputs for testing. Not market data."""
    rng = np.random.default_rng(5)
    names = ["NVIDIA", "AMD", "Intel", "Micron", "ASML", "TSMC", "SK hynix", "Samsung"]
    price = pd.Series(rng.uniform(50, 300, 8), index=names)
    inp = pd.DataFrame({"price": price,
                        "fair_value": price * rng.uniform(0.75, 1.30, 8),
                        "implied_vol": rng.uniform(0.35, 0.65, 8)}, index=names)
    loads = rng.uniform(0.9, 1.6, 8)
    mkt_var, idio = 0.04 ** 2 * 52, rng.uniform(0.25, 0.4, 8) ** 2
    full = np.outer(loads, loads) * mkt_var + np.diag(idio)       # annual cov, 1-factor model
    full = np.pad(full, ((0, 1), (0, 1)))
    full[:8, 8] = full[8, :8] = loads * mkt_var
    full[8, 8] = mkt_var
    cov = pd.DataFrame(full, index=names + ["MARKET"], columns=names + ["MARKET"])
    return inp, cov


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs"), ap.add_argument("--cov")
    ap.add_argument("--vol-cap", type=float, help="dollar volatility cap over the horizon")
    ap.add_argument("--horizon-days", type=int, default=30)
    ap.add_argument("--ic", type=float, default=0.03)
    ap.add_argument("--beta-tol", type=float, default=1.0, help="in $M of beta-weighted exposure")
    ap.add_argument("--max-pos", type=float, default=50.0, help="per-name cap in $M")
    ap.add_argument("--betas-csv"), ap.add_argument("--out", default="book_positions.csv")
    ap.add_argument("--cov-as-is", action="store_true")
    ap.add_argument("--group-neutral", action="store_true")
    ap.add_argument("--fallback-minvol", action="store_true")
    ap.add_argument("--round-step", type=float, default=100_000,
                    help="round final positions to this step in USD (default 100000; 0 = no rounding)")
    ap.add_argument("--demo", action="store_true", help="run on SYNTHETIC data")
    a = ap.parse_args()

    if a.demo:
        inp, cov_raw = demo_inputs()
        vol_cap = a.vol_cap or 3_500_000
    else:
        if not (a.inputs and a.cov and a.vol_cap):
            ap.error("--inputs, --cov and --vol-cap are required (or use --demo)")
        inp = pd.read_csv(a.inputs).set_index("issuer")
        cov_raw = pd.read_csv(a.cov, index_col=0)
        vol_cap = a.vol_cap
    try:
        pos, info = run(inp, cov_raw, vol_cap, a.horizon_days, a.ic, a.beta_tol, a.max_pos,
                        a.betas_csv, a.cov_as_is, a.group_neutral, a.fallback_minvol, a.round_step)
    except InfeasibleError as exc:
        raise SystemExit(f"INFEASIBLE: {exc}")

    with pd.option_context("display.width", 160, "display.float_format", "{:,.4f}".format):
        print(pos.to_string(index=False))
    print(f"\nStatus: {info['status']}   (beta source: {info['beta_source']})")
    if info["rounding_step_usd"]:
        u = info["unrounded"]
        print(f"Before rounding: expected P&L ${u['expected_pnl_usd']:,.0f}   Volatility ${u['vol_usd']:,.0f}   "
              f"Beta exposure ${u['beta_exposure_usd']:,.0f}")
        print(f"Rounded to ${info['rounding_step_usd']:,.0f}: {info['rounding_status']}")
        if "group_net_usd" in info:
            print("Group nets after rounding:", {k: round(v) for k, v in info["group_net_usd"].items()})
    print(f"{'After rounding: e' if info['rounding_step_usd'] else 'E'}xpected P&L over {a.horizon_days} days: ${info['expected_pnl_usd']:,.0f}   "
          f"Volatility: ${info['vol_usd']:,.0f} (cap ${vol_cap:,.0f})   "
          f"Beta exposure: ${info['beta_exposure_usd']:,.0f} (tolerance used ${info['beta_tol_used_usd']:,.0f})")
    print("\nConstraint checks:")
    print(check_constraints(pos).to_string(index=False))
    pos.to_csv(a.out, index=False)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()