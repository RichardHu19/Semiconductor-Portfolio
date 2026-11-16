"""
var_es.py -- dollar VaR and Expected Shortfall of the fixed target book (assignment B7).

CONVENTIONS (state them in your memo)
  * Loss L = -(forward dollar P&L). VaR_q is the q-quantile of L; ES_q is the average
    loss in the worst (1-q) fraction of outcomes.
  * Exposures are held FIXED at the target dollars over the horizon (no rebalancing,
    no algorithm simulation).
  * Horizon: --horizon-days (default 30 calendar days), the same horizon as the
    implied-volatility covariance used by max_return_book.py.
  * Mean P&L: --mean zero (default, conservative: mean set to 0) or --mean expected
    (mean = mu'V, the optimizer's expected P&L). Every method uses the same treatment.
  * Equity-book and total-strategy risk are identical when no hedge overlays are used;
    this program covers the equity book only (overlays are not implemented).

METHODS
  1. Parametric normal:  VaR = z_q*sigma_P - mu_P,  ES = sigma_P*phi(z_q)/(1-q) - mu_P.
  2. Parametric Student-t (fat tails), scaled to the same variance (--t-df, default 5).
  3. Monte Carlo, multivariate normal with the same covariance (checks method 1;
     seeded for replay). ES uses the Rockafellar-Uryasev form
     ES = VaR + E[max(L - VaR, 0)]/(1-q), which handles ties and fractional tails.
  4. Historical simulation (optional, needs --hist-returns): the fixed dollar book applied
     to overlapping rolling sums of weekly returns over round(horizon/7) weeks, with the
     sample mean replaced by the chosen mean. Few observations in the tail: treat the
     99% figures with great caution.

INPUTS (same files as the rest of the pipeline)
  --positions  CSV: issuer, target_usd      (book_positions.csv from max_return_book.py)
  --inputs     CSV from inputs.py           (price, fair_value, implied_vol)
  --cov        CSV from inputs.py           (weekly covariance incl. MARKET column)
Covariance = D R D with implied-vol horizon volatilities D and the correlations R of
--cov (as in max_return_book.py), so the VaR uses the same risk model as the optimizer.
Needs max_return_book.py in the same folder.

Run:   python risk.py --positions book_positions.csv --inputs inputs.csv --cov cov.csv
Demo:  python risk.py --demo        (SYNTHETIC data, for testing only)
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from return_book import alpha_from_fair_value, build_sigma, get_betas


# ------------------------------- risk measures --------------------------------
def parametric_normal(sigma_p, mu_p, q):
    z = stats.norm.ppf(q)
    return z * sigma_p - mu_p, sigma_p * stats.norm.pdf(z) / (1 - q) - mu_p


def parametric_t(sigma_p, mu_p, q, df):
    """Student-t scaled so the P&L variance equals sigma_p**2 (needs df > 2)."""
    s = sigma_p * np.sqrt((df - 2) / df)
    t = stats.t.ppf(q, df)
    es = s * stats.t.pdf(t, df) / (1 - q) * (df + t * t) / (df - 1)
    return s * t - mu_p, es - mu_p


def empirical_var_es(losses, q):
    """VaR = empirical q-quantile (an order statistic); ES via Rockafellar-Uryasev."""
    L = np.asarray(losses, dtype=float)
    var = float(np.quantile(L, q, method="inverted_cdf"))
    return var, var + float(np.maximum(L - var, 0).mean()) / (1 - q)


def simulate_losses(V, Sigma, mu_p, n, seed):
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(Sigma)
    r = rng.standard_normal((n, len(V))) @ chol.T
    return -(r @ V + mu_p)


def historical_losses(V, weekly_returns, horizon_days, mu_p):
    """Fixed dollar book on overlapping k-week sums of weekly returns (k = horizon/7)."""
    pnl_w = weekly_returns[list(V.index)].values @ V.values
    k = max(1, int(round(horizon_days / 7)))
    pnl = pd.Series(pnl_w).rolling(k).sum().dropna().values
    pnl = pnl - pnl.mean() + mu_p                           # same mean treatment as the other methods
    return -pnl, len(pnl), k


def risk_contributions(V, Sigma):
    """Euler decomposition: contribution_i = V_i (Sigma V)_i / sigma_P (sums to sigma_P)."""
    m = Sigma.loc[V.index, V.index].values @ V.values
    sigma_p = float(np.sqrt(V.values @ m))
    rc = V.values * m / sigma_p
    standalone = np.abs(V.values) * np.sqrt(np.diag(Sigma.loc[V.index, V.index].values))
    return pd.DataFrame({"position_usd": V.values, "standalone_vol_usd": standalone,
                         "contribution_usd": rc, "contribution_pct": 100 * rc / sigma_p},
                        index=V.index), sigma_p


# ------------------------------- driver ---------------------------------------
def run(V, inp, cov_raw, hist=None, horizon_days=30, ic=0.03, mean="zero",
        confidences=(0.95, 0.99), t_df=5, sims=200_000, seed=12345):
    names = list(V.index)
    iv_h = inp["implied_vol"].loc[names] * np.sqrt(horizon_days / 365)
    Sigma = build_sigma(cov_raw, names, iv_h)
    mu = alpha_from_fair_value(inp["price"].loc[names], inp["fair_value"].loc[names], iv_h, ic=ic)
    beta, beta_src = get_betas(cov_raw, names)

    rc, sigma_p = risk_contributions(V, Sigma)
    exp_pnl = float(mu.loc[names].values @ V.values)
    mu_p = exp_pnl if mean == "expected" else 0.0
    gmv = float(V.abs().sum())

    L_mc = simulate_losses(V.values, Sigma.loc[names, names].values, mu_p, sims, seed)
    L_h, n_h, k_h = historical_losses(V, hist, horizon_days, mu_p) if hist is not None else (None, 0, 0)
    rows = []
    for q in confidences:
        res = {"parametric normal": parametric_normal(sigma_p, mu_p, q),
               f"parametric Student-t (df={t_df})": parametric_t(sigma_p, mu_p, q, t_df),
               f"Monte Carlo normal ({sims:,} draws)": empirical_var_es(L_mc, q)}
        if L_h is not None:
            res[f"historical simulation (n={n_h}, {k_h}-week sums)"] = empirical_var_es(L_h, q)
        for method, (var, es) in res.items():
            rows.append({"method": method, "confidence": q, "VaR_usd": var, "ES_usd": es,
                         "VaR_pct_gmv": 100 * var / gmv, "ES_pct_gmv": 100 * es / gmv})
    info = {"sigma_p_usd": sigma_p, "expected_pnl_usd": exp_pnl, "mean_used_usd": mu_p,
            "beta_exposure_usd": float(beta.loc[names].values @ V.values), "beta_source": beta_src,
            "gmv_usd": gmv, "net_usd": float(V.sum())}
    return pd.DataFrame(rows), rc, info


def demo_inputs():
    """SYNTHETIC inputs for testing. Not market data."""
    import max_return_book as m
    inp, cov_raw = m.demo_inputs()
    pos, _ = m.run(inp, cov_raw, 3_500_000)
    V = pos.set_index("issuer")["target_usd"]
    names = list(V.index)
    rng = np.random.default_rng(1)
    S_week = m.build_sigma(cov_raw, names, inp["implied_vol"] * np.sqrt(7 / 365)).values
    shocks = rng.standard_t(4, (260, len(names))) * np.sqrt(2 / 4)       # fat-tailed weekly returns
    hist = pd.DataFrame(shocks @ np.linalg.cholesky(S_week).T, columns=names,
                        index=pd.date_range("2021-11-12", periods=260, freq="W-FRI"))
    return V, inp, cov_raw, hist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions"), ap.add_argument("--inputs"), ap.add_argument("--cov")
    ap.add_argument("--hist-returns", help="weekly returns CSV (dates in first column)")
    ap.add_argument("--horizon-days", type=int, default=30)
    ap.add_argument("--mean", choices=["zero", "expected"], default="zero")
    ap.add_argument("--ic", type=float, default=0.03)
    ap.add_argument("--confidence", type=float, nargs="+", default=[0.95, 0.99])
    ap.add_argument("--t-df", type=float, default=5)
    ap.add_argument("--sims", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--demo", action="store_true", help="run on SYNTHETIC data")
    a = ap.parse_args()

    if a.demo:
        V, inp, cov_raw, hist = demo_inputs()
    else:
        if not (a.positions and a.inputs and a.cov):
            ap.error("--positions, --inputs and --cov are required (or use --demo)")
        V = pd.read_csv(a.positions).set_index("issuer")["target_usd"]
        inp = pd.read_csv(a.inputs).set_index("issuer")
        cov_raw = pd.read_csv(a.cov, index_col=0)
        hist = pd.read_csv(a.hist_returns, index_col=0, parse_dates=True) if a.hist_returns else None
    os.makedirs(a.out_dir, exist_ok=True)

    table, rc, info = run(V, inp, cov_raw, hist, a.horizon_days, a.ic, a.mean,
                          tuple(a.confidence), a.t_df, a.sims, a.seed)
    pd.set_option("display.width", 170, "display.float_format", "{:,.2f}".format)
    print(f"\nBook: GMV ${info['gmv_usd']:,.0f}, net ${info['net_usd']:,.0f}; horizon {a.horizon_days} days; "
          f"exposures fixed; loss = -P&L")
    print(f"Forecast dollar volatility sigma_P: ${info['sigma_p_usd']:,.0f}   "
          f"Beta exposure: ${info['beta_exposure_usd']:,.0f} ({info['beta_source']})")
    print(f"Expected P&L mu'V: ${info['expected_pnl_usd']:,.0f}   Mean used in VaR/ES: ${info['mean_used_usd']:,.0f} "
          f"(--mean {a.mean})\n")
    print(table.to_string(index=False))
    print("\nRisk contributions (Euler decomposition; contributions sum to sigma_P)")
    print(rc.round(1).to_string())
    table.to_csv(os.path.join(a.out_dir, "var_es_results.csv"), index=False)
    rc.to_csv(os.path.join(a.out_dir, "risk_contributions.csv"))
    print("\nSaved var_es_results.csv and risk_contributions.csv")
    print("Equity-book risk shown; total-strategy risk is identical when no overlays are used.")


if __name__ == "__main__":
    main()