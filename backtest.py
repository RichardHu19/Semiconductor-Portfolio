"""
walkforward_backtest.py -- monthly re-optimized book using CURRENT fair values.

WHAT IT DOES
  At the last weekly close of every month, re-runs the optimizer of max_return_book.py
  with information available at that date, holds the resulting dollar positions (shares
  fixed, so dollars drift with prices) until the next month-end, and reports Sharpe,
  Sortino, maximum drawdown and dollar volatility of the weekly P&L.
  Weights decided at week t earn the returns of weeks t+1 onward (no same-week look-ahead).

INPUTS USED AT EACH REBALANCE DATE t
  fair values      CURRENT Morningstar fair values (--inputs inputs.csv, column fair_value),
                   the same for every date.  *** HINDSIGHT / LOOK-AHEAD BIAS ***
  prices           price at t, in the fair-value basis (USD; Samsung in KRW; SK hynix USD per
                   ADS, proxy-spliced before its listing, anchored to the latest ADS close)
  volatility       REALIZED volatility (last --vol-weeks weekly returns) as a stand-in for
                   implied volatility, which has no free history
  correlation/beta trailing --lookback-weeks of weekly returns (beta to SPY if available,
                   else to an equal-weight basket: a SECTOR beta)
  everything else  the same optimizer settings as max_return_book.py (IC, vol cap, beta
                   tolerance, position cap, $100,000 rounding, fallback to the minimum-
                   variance book if the vol cap is unattainable, flagged in the log)

HOW TO READ THE RESULT (put this in your memo)
  Today's fair values contain information that was not known at past dates, and they
  favour stocks that have since moved toward them, so the results are IN-SAMPLE and
  likely flattering. They test the MECHANICS (turnover, beta neutrality, risk control,
  stability of the weights), NOT the skill of the signal. They are not out-of-sample
  evidence. Prices are dividend-adjusted, so historical fair-value gaps are slightly
  overstated for dividend payers. Carry defaults to zero; execution costs are ignored,
  as in the assignment, but turnover is reported.

Optional --naive adds a benchmark that goes long the 4 largest fair-value gaps and short the
4 smallest, $12.5m each, with no optimization, to show what the optimizer adds.

Needs max_return_book.py, fixed_backtest.py and correlation.py
in the same folder. Each month's optimization takes several seconds.

Run:   python walkforward_backtest.py --prices-csv inputs_prices*.csv --inputs inputs.csv --asof 2026-11-13 --vol-cap 6000000 --naive
Demo:  python walkforward_backtest.py --demo --max-rebalances 4        (SYNTHETIC data)
"""
import argparse
import os
from datetime import date

import numpy as np
import pandas as pd

import return_book as mrb
from fixed_backtest import perf_stats, PERIODS


# ------------------------------- data -----------------------------------------
def load_weekly(prices_csv, start, asof):
    """Weekly USD returns, weekly fair-value-basis prices and SPY returns from a saved snapshot."""
    try:
        import weekly_correlation as wc
    except ImportError:
        import correlation as wc
    px = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    wc.START, wc.END_INCLUSIVE = start, asof                          # weekly_correlation reads these globals
    rets, proxy_weeks = wc.weekly_usd_returns(px)
    rets = rets[rets.index <= pd.Timestamp(asof)]
    wk = px.loc[start:asof].resample("W-FRI").last().ffill(limit=1).reindex(rets.index)
    price = pd.DataFrame({n: wk[t] for n, t in wc.US_NAMES.items()})
    price["Samsung"] = wk["005930.KS"]                                # KRW, same basis as the fair value
    adr = px.loc[start:asof, "SKHY"].dropna()
    idx = (1 + rets["SK hynix"]).cumprod()                            # spliced ADS/proxy return index
    anchor = float(adr.iloc[-1]) / float(idx.loc[:adr.index[-1]].iloc[-1])
    price["SK hynix"] = idx * anchor                                  # USD per ADS-equivalent
    price = price[list(rets.columns)]
    mkt = None
    if "SPY" in px.columns and px["SPY"].dropna().shape[0] > 50:
        spy = px.loc[start:asof, "SPY"].resample("W-FRI").last()
        mkt = (spy / spy.shift(1) - 1).reindex(rets.index).rename("MARKET")
    return rets, price, mkt, proxy_weeks


def demo_data():
    """SYNTHETIC weekly data. Not market data."""
    rng = np.random.default_rng(4)
    names = ["NVIDIA", "AMD", "Intel", "Micron", "ASML", "TSMC", "SK hynix", "Samsung"]
    idx = pd.date_range("2023-01-06", periods=180, freq="W-FRI")
    mkt = pd.Series(rng.normal(0.002, 0.022, len(idx)), index=idx, name="MARKET")
    load = rng.uniform(0.9, 1.6, len(names))
    rets = pd.DataFrame({n: load[i] * mkt + rng.standard_t(5, len(idx)) * 0.03
                         for i, n in enumerate(names)}, index=idx)
    price = 100 * (1 + rets).cumprod()
    fv = price.iloc[-1] * pd.Series(rng.uniform(0.8, 1.3, len(names)), index=names)
    return rets, price, mkt, fv


def month_end_weeks(index):
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


# ------------------------------- strategy -------------------------------------
def decide(t, rets, price, mkt, fv, a):
    """Optimized book at date t using only data up to t. Returns (V, log row)."""
    names = list(rets.columns)
    i = rets.index.get_loc(t)
    w = rets.iloc[i - a.lookback_weeks + 1: i + 1]
    block = w.copy()
    if mkt is not None:
        block["MARKET"] = mkt.reindex(w.index)
    cov_raw = block.dropna().cov()                                     # weekly units; build_sigma uses correlations
    vol_ann = w.iloc[-a.vol_weeks:].std(ddof=1) * np.sqrt(PERIODS)     # realized stand-in for implied vol
    iv_h = vol_ann * np.sqrt(a.horizon_days / 365)
    mu = mrb.alpha_from_fair_value(price.loc[t], fv, iv_h, ic=a.ic)
    Sigma = mrb.build_sigma(cov_raw, names, iv_h)
    beta, _ = mrb.get_betas(cov_raw, names)
    pos, info = mrb.solve_book(mu, Sigma, beta, a.vol_cap, a.beta_tol, a.max_pos,
                               group_neutral=False, fallback_minvol=True)
    V0 = pos.set_index("issuer")["target_usd"]
    V, rstatus = mrb.round_book(V0, mu, Sigma, beta, a.vol_cap, info["beta_tol_used_usd"],
                                a.round_step, a.max_pos * 1e6)
    st = mrb._stats(V, mu, Sigma, beta)
    return V, {"date": t, "status": info["status"], "rounding": rstatus, **st,
               "n_long": int((V > 0).sum()), "n_short": int((V < 0).sum())}


def naive_book(t, price, fv):
    gap = np.log(fv / price.loc[t]).sort_values(ascending=False)
    V = pd.Series(0.0, index=gap.index)
    V[gap.index[:4]], V[gap.index[-4:]] = 12.5e6, -12.5e6
    return V


def simulate(targets, rets, fee, fund, rebate):
    """Hold dollar targets from each rebalance week; dollars drift with returns in between."""
    dollars, pnl, turn = None, {}, {}
    for t in rets.index:
        if dollars is not None:
            r = rets.loc[t]
            lm, sm = dollars[dollars > 0].sum(), -dollars[dollars < 0].sum()
            carry = (-fee * sm + rebate * sm - fund * lm) * 7 / 360
            pnl[t] = float(dollars.values @ r.values) + carry
            dollars = dollars * (1 + r)
        if t in targets:
            new = targets[t].reindex(rets.columns)
            turn[t] = float((new - (dollars if dollars is not None else 0.0)).abs().sum())
            dollars = new.copy()
    return pd.Series(pnl), pd.Series(turn)


# ------------------------------- driver ---------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices-csv"), ap.add_argument("--inputs")
    ap.add_argument("--asof", default=date.today().isoformat()), ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--lookback-weeks", type=int, default=104), ap.add_argument("--vol-weeks", type=int, default=26)
    ap.add_argument("--vol-cap", type=float, default=6_000_000), ap.add_argument("--ic", type=float, default=0.03)
    ap.add_argument("--beta-tol", type=float, default=1.0, help="$M of beta-weighted exposure")
    ap.add_argument("--max-pos", type=float, default=20.0, help="per-name cap, $M")
    ap.add_argument("--round-step", type=float, default=100_000), ap.add_argument("--horizon-days", type=int, default=30)
    ap.add_argument("--borrow-fee", type=float, default=0.0), ap.add_argument("--funding-rate", type=float, default=0.0)
    ap.add_argument("--rebate-rate", type=float, default=0.0)
    ap.add_argument("--start-date", help="first rebalance not before this date")
    ap.add_argument("--max-rebalances", type=int, help="stop after this many rebalances (testing)")
    ap.add_argument("--naive", action="store_true"), ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out-dir", default="."), ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.vol_cap <= 0:
        ap.error("--vol-cap must be positive (write 6000000, not $6000000: the shell expands $6000000 to 000000)")
    os.makedirs(a.out_dir, exist_ok=True)

    proxy_weeks = None
    if a.demo:
        rets, price, mkt, fv = demo_data()
        a.lookback_weeks, a.vol_weeks = min(a.lookback_weeks, 104), min(a.vol_weeks, 26)
    else:
        if not (a.prices_csv and a.inputs):
            ap.error("--prices-csv and --inputs are required (or use --demo)")
        start = (pd.Timestamp(a.asof) - pd.Timedelta(days=int(a.years * 365))).strftime("%Y-%m-%d")
        rets, price, mkt, proxy_weeks = load_weekly(a.prices_csv, start, a.asof)
        fv = pd.read_csv(a.inputs).set_index("issuer")["fair_value"].reindex(rets.columns)
        if mkt is None:
            print("WARNING: SPY not in the price snapshot: betas will be to an equal-weight basket (SECTOR beta).")
    rets = rets.dropna()
    price, fv = price.reindex(rets.index), fv.reindex(rets.columns)
    ratio = fv / price.iloc[-1]
    for n in ratio.index[(ratio < 0.3) | (ratio > 3)]:
        print(f"WARNING: {n}: fair value / latest price = {ratio[n]:.2f}: check units and currency")

    print("\n*** HINDSIGHT WARNING: current fair values are applied to past dates. Results are IN-SAMPLE and "
          "test the mechanics only, not the skill of the signal. ***\n")
    dates = [d for d in month_end_weeks(rets.index) if rets.index.get_loc(d) >= a.lookback_weeks - 1]
    if a.start_date:
        dates = [d for d in dates if d >= pd.Timestamp(a.start_date)]
    dates = dates[:a.max_rebalances] if a.max_rebalances else dates[:-1] if len(dates) > 1 else dates
    if not dates:
        raise SystemExit("No rebalance dates: need more weekly history than --lookback-weeks.")

    targets, naive_targets, log = {}, {}, []
    for k, t in enumerate(dates, 1):
        try:
            V, row = decide(t, rets, price, mkt, fv, a)
            targets[t] = V
        except mrb.InfeasibleError as exc:
            row = {"date": t, "status": f"INFEASIBLE, prior holdings kept ({str(exc)[:60]})"}
            if targets:
                targets[t] = pd.Series(np.nan, index=rets.columns)       # replaced below by prior dollars
        log.append(row)
        if a.naive:
            naive_targets[t] = naive_book(t, price, fv)
        print(f"[{k}/{len(dates)}] {t.date()}  {row['status']}", flush=True)

    # an infeasible month keeps the previous target dollars
    prev = None
    for t in sorted(targets):
        if targets[t].isna().any() and prev is not None:
            targets[t] = prev
        prev = targets[t]
    targets = {t: v for t, v in targets.items() if not v.isna().any()}
    pnl, turn = simulate(targets, rets, a.borrow_fee, a.funding_rate, a.rebate_rate)
    gmv0 = 100e6
    results, series = {}, {}
    stats, cum, dd = perf_stats(pnl, gmv0)
    results["optimized, monthly"], series["optimized, monthly"] = stats, pd.DataFrame(
        {"pnl_usd": pnl, "cum_pnl_usd": cum, "drawdown_usd": dd})
    if a.naive:
        npnl, nturn = simulate(naive_targets, rets, a.borrow_fee, a.funding_rate, a.rebate_rate)
        s2, c2, d2 = perf_stats(npnl, gmv0)
        results["naive gap-rank, monthly"] = s2
        series["naive gap-rank, monthly"] = pd.DataFrame({"pnl_usd": npnl, "cum_pnl_usd": c2, "drawdown_usd": d2})
    table = pd.DataFrame(results)
    table.loc["avg_turnover_per_rebalance_usd"] = [turn.mean()] + ([nturn.mean()] if a.naive else [])
    if mkt is not None:                                                 # realized market exposure of the weekly P&L
        m = mkt.reindex(pnl.index)
        table.loc["realized_beta_exposure_usd"] = [float(np.cov(pnl, m)[0, 1] / m.var())] + (
            [float(np.cov(npnl, m)[0, 1] / m.var())] if a.naive else [])

    logdf = pd.DataFrame(log).set_index("date")
    nxt = pnl.groupby(pd.cut(pnl.index, list(logdf.index) + [rets.index.max() + pd.Timedelta(days=1)],
                             labels=list(logdf.index), right=False), observed=False).sum()
    pd.set_option("display.width", 160, "display.float_format", "{:,.2f}".format)
    print(f"\nWalk-forward, IN-SAMPLE (hindsight fair values): first rebalance {dates[0].date()}, "
          f"{len(dates)} monthly rebalances, {len(pnl)} weekly P&L observations")
    if proxy_weeks is not None:
        print(f"SK hynix: {proxy_weeks} weeks of the data use the KRX-based proxy.")
    print(table.to_string())
    fb = int(logdf["status"].str.contains("FALLBACK|INFEASIBLE").sum())
    print(f"\nMonths on the minimum-variance fallback or infeasible: {fb} of {len(logdf)} "
          f"(vol cap ${a.vol_cap:,.0f} may be below what realized volatility allows)")
    print(f"Average expected monthly P&L at rebalance: ${logdf['expected_pnl_usd'].mean():,.0f}; "
          f"average realized P&L over the following month: ${nxt.mean():,.0f}")
    print("\nIn-sample, hindsight fair values: not out-of-sample evidence.")

    table.to_csv(os.path.join(a.out_dir, "wf_stats.csv"))
    logdf.assign(turnover_usd=turn, next_month_pnl_usd=nxt.values).to_csv(os.path.join(a.out_dir, "wf_rebalances.csv"))
    pd.DataFrame(targets).T.rename_axis("date").to_csv(os.path.join(a.out_dir, "wf_weights.csv"))
    pd.concat(series, axis=1).to_csv(os.path.join(a.out_dir, "wf_pnl.csv"))
    if not a.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            for name, df in series.items():
                ax[0].plot(df.index, df["cum_pnl_usd"] / 1e6, label=name)
                ax[1].plot(df.index, -df["drawdown_usd"] / 1e6, label=name)
            ax[0].set_ylabel("Cumulative P&L ($m)"), ax[1].set_ylabel("Drawdown ($m)"), ax[0].legend()
            ax[0].set_title("Walk-forward, monthly re-optimization (IN-SAMPLE: hindsight fair values)")
            fig.tight_layout(), fig.savefig(os.path.join(a.out_dir, "wf_pnl.png"), dpi=150)
        except ImportError:
            pass
    print("Saved wf_stats.csv, wf_rebalances.csv, wf_weights.csv, wf_pnl.csv" + ("" if a.no_plot else ", wf_pnl.png"))


if __name__ == "__main__":
    main()