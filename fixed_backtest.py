"""
fixed_book_backtest.py -- historical IN-SAMPLE fixed-book analysis (assignment B8).

WHAT IT DOES
  Applies the target dollar positions V (from book_positions.csv) to past WEEKLY USD
  returns, as if the book had been restored to the same dollars every week:
        weekly P&L_t = sum_i V_i * r_{i,t}  -  carry_t
  and reports Sharpe, Sortino, maximum drawdown and dollar volatility.

THIS IS NOT A BACKTEST OF THE STRATEGY. Morningstar fair values exist only as of today,
so past signals cannot be recreated: the weights are chosen with hindsight and the
results are IN-SAMPLE. Do not present them as out-of-sample evidence.

DEFINITIONS (same as the memo)
  r_t       = weekly P&L_t / GMV_0, GMV_0 = |V| summed (fixed initial gross, $100m)
  Sharpe    = mean(r)/std(r) * sqrt(52)            (zero hurdle; sample std, ddof=1)
  Sortino   = mean(r)/DD * sqrt(52), DD = sqrt(mean(min(r,0)^2))  (zero MAR, all weeks)
  Drawdown_t= max_{s<=t} C_s - C_t, C_t = cumulative dollar P&L, C_0 = 0 (not compounded)
  Dollar volatility = std of weekly dollar P&L; also annualized (x sqrt(52)) and scaled
  to a 30-day horizon (x sqrt(30/7)) to compare with the forecast volatility in B7.
  Returns are total returns (dividends included; shorts therefore owe them). Carry is
  optional: --borrow-fee, --funding-rate, --rebate-rate (annual, default 0 = an
  assumption you must disclose); weekly carry =
  (-fee*|short| + rebate*|short| - funding*long) * 7/360.

INPUTS (one of)
  --positions  CSV: issuer, target_usd
  --prices-csv daily closes saved by inputs.py (inputs_prices_*.csv, about 5 years);
               weekly USD returns are built with weekly_correlation.py (SK hynix proxy
               splice before its ADS listing; Samsung converted from KRW), or
  --returns-csv weekly USD returns, dates in the first column, one column per issuer.
  Optional --baseline also reports an equal-dollar-within-side book (sides as in the
  positions file) for comparison with the optimized book.

Run:   python backtest.py --positions book_positions.csv --prices-csv prices.csv --asof 2026-11-13
Demo:  python backtest.py --demo        (SYNTHETIC data, for testing only)
"""
import argparse
import os
from datetime import date

import numpy as np
import pandas as pd

PERIODS = 52                                       # weekly observations per year


# ------------------------------- core maths -----------------------------------
def weekly_pnl(V, rets, borrow_fee=0.0, funding=0.0, rebate=0.0):
    """Weekly dollar P&L of the fixed book, net of optional carry. V: Series by issuer."""
    pnl = pd.Series(rets[list(V.index)].values @ V.values, index=rets.index)
    long_mv, short_mv = V[V > 0].sum(), -V[V < 0].sum()
    carry = (-borrow_fee * short_mv + rebate * short_mv - funding * long_mv) * 7 / 360
    return pnl + carry, carry


def perf_stats(pnl, gmv0):
    """Sharpe, Sortino, maximum drawdown and dollar volatility of a weekly P&L series."""
    r = pnl / gmv0
    cum = pnl.cumsum()
    peak = np.maximum.accumulate(np.r_[0.0, cum.values])[1:]        # C_0 = 0 is the first peak
    dd = peak - cum.values
    trough = int(np.argmax(dd))
    downside = np.sqrt(np.mean(np.minimum(r.values, 0.0) ** 2))
    vol = float(pnl.std(ddof=1))
    return {
        "weeks": len(pnl),
        "total_pnl_usd": float(cum.iloc[-1]),
        "total_pnl_pct_gmv0": float(100 * cum.iloc[-1] / gmv0),
        "mean_weekly_pnl_usd": float(pnl.mean()),
        "dollar_vol_weekly_usd": vol,
        "dollar_vol_annualized_usd": vol * np.sqrt(PERIODS),
        "dollar_vol_30day_usd": vol * np.sqrt(30 / 7),
        "sharpe_zero_hurdle": float(r.mean() / r.std(ddof=1) * np.sqrt(PERIODS)),
        "sortino_zero_mar": float(r.mean() / downside * np.sqrt(PERIODS)) if downside > 0 else np.nan,
        "max_drawdown_usd": float(dd.max()),
        "max_drawdown_pct_gmv0": float(100 * dd.max() / gmv0),
        "max_drawdown_trough": str(pnl.index[trough].date()) if hasattr(pnl.index[trough], "date") else str(trough),
        "best_week_usd": float(pnl.max()), "worst_week_usd": float(pnl.min()),
        "hit_rate_pct": float(100 * (pnl > 0).mean()),
    }, cum, pd.Series(dd, index=pnl.index)


def baseline_book(V, gmv0):
    """Equal dollars within each side, sides as in V (B9 convention)."""
    longs, shorts = V[V > 0].index, V[V < 0].index
    out = pd.Series(0.0, index=V.index)
    out[longs], out[shorts] = gmv0 / 2 / len(longs), -gmv0 / 2 / len(shorts)
    return out


# ------------------------------- data -----------------------------------------
def returns_from_prices(prices_csv, start, asof):
    try:
        import weekly_correlation as wc
    except ImportError:
        import correlation as wc
    px = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    wc.START, wc.END_INCLUSIVE = start, asof                      # weekly_correlation reads these globals
    rets, proxy_weeks = wc.weekly_usd_returns(px)
    return rets[rets.index <= pd.Timestamp(asof)], proxy_weeks    # drop an incomplete final week


def demo_inputs():
    """SYNTHETIC weekly returns and a dollar-neutral book. Not market data."""
    rng = np.random.default_rng(21)
    names = ["NVIDIA", "AMD", "Intel", "Micron", "ASML", "TSMC", "SK hynix", "Samsung"]
    idx = pd.date_range("2021-11-19", periods=260, freq="W-FRI")
    mkt = rng.normal(0.002, 0.025, len(idx))
    load = rng.uniform(0.9, 1.6, len(names))
    rets = pd.DataFrame({n: load[i] * mkt + rng.standard_t(5, len(idx)) * 0.03
                         for i, n in enumerate(names)}, index=idx)
    V = pd.Series([15, -20, 10, -10, 10, -10, 15, -10.0], index=names) * 1e6
    return V, rets


# ------------------------------- driver ---------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions"), ap.add_argument("--prices-csv"), ap.add_argument("--returns-csv")
    ap.add_argument("--asof", default=date.today().isoformat()), ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--borrow-fee", type=float, default=0.0), ap.add_argument("--funding-rate", type=float, default=0.0)
    ap.add_argument("--rebate-rate", type=float, default=0.0)
    ap.add_argument("--baseline", action="store_true"), ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out-dir", default="."), ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    proxy_weeks = None
    if a.demo:
        V, rets = demo_inputs()
    else:
        if not a.positions or not (a.prices_csv or a.returns_csv):
            ap.error("--positions and one of --prices-csv / --returns-csv are required (or use --demo)")
        V = pd.read_csv(a.positions).set_index("issuer")["target_usd"]
        start = (pd.Timestamp(a.asof) - pd.Timedelta(days=int(a.years * 365))).strftime("%Y-%m-%d")
        if a.prices_csv:
            rets, proxy_weeks = returns_from_prices(a.prices_csv, start, a.asof)
        else:
            rets = pd.read_csv(a.returns_csv, index_col=0, parse_dates=True)
            rets = rets[(rets.index >= start) & (rets.index <= pd.Timestamp(a.asof))]
    rets = rets[list(V.index)].dropna()
    gmv0 = float(V.abs().sum())
    if abs(V.sum()) > 1 or abs(gmv0 - 100e6) > 1:
        print(f"NOTE: book has GMV ${gmv0:,.0f} and net ${V.sum():,.0f}; ratios use this GMV as GMV_0.")
    if a.borrow_fee == a.funding_rate == a.rebate_rate == 0:
        print("NOTE: carry (borrow fee, funding, rebate) assumed ZERO: disclose this assumption.")

    books = {"optimized book": V}
    if a.baseline:
        books["equal-weight baseline"] = baseline_book(V, gmv0)
    results, series = {}, {}
    for name, book in books.items():
        pnl, carry = weekly_pnl(book, rets, a.borrow_fee, a.funding_rate, a.rebate_rate)
        stats, cum, dd = perf_stats(pnl, gmv0)
        results[name], series[name] = stats, pd.DataFrame({"pnl_usd": pnl, "cum_pnl_usd": cum, "drawdown_usd": dd})

    table = pd.DataFrame(results)
    pd.set_option("display.width", 150, "display.float_format", "{:,.2f}".format)
    print(f"\nFixed-book analysis, IN-SAMPLE: {rets.index.min().date()} to {rets.index.max().date()} "
          f"({len(rets)} weekly observations)")
    if proxy_weeks is not None:
        print(f"SK hynix: {proxy_weeks} weeks use the KRX-based proxy; the rest use actual ADS returns.")
    print(table.to_string())

    main_pnl = series["optimized book"]["pnl_usd"]
    yearly = main_pnl.groupby(main_pnl.index.year).agg(weeks="count", pnl_usd="sum")
    yearly["sharpe"] = main_pnl.groupby(main_pnl.index.year).apply(
        lambda s: s.mean() / s.std(ddof=1) * np.sqrt(PERIODS) if len(s) > 2 else np.nan)
    print("\nBy calendar year (optimized book); partial years have few weeks")
    print(yearly.to_string())

    table.to_csv(os.path.join(a.out_dir, "fixed_book_stats.csv"))
    pd.concat(series, axis=1).to_csv(os.path.join(a.out_dir, "fixed_book_pnl.csv"))
    if not a.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            for name, df in series.items():
                ax[0].plot(df.index, df["cum_pnl_usd"] / 1e6, label=name)
                ax[1].plot(df.index, -df["drawdown_usd"] / 1e6, label=name)
            ax[0].set_ylabel("Cumulative P&L ($m)"), ax[1].set_ylabel("Drawdown ($m)")
            ax[0].legend(), ax[0].set_title("Fixed-book analysis (in-sample, hindsight-selected weights)")
            fig.tight_layout(), fig.savefig(os.path.join(a.out_dir, "fixed_book_pnl.png"), dpi=150)
        except ImportError:
            print("matplotlib not installed: skipping the chart")
    print("\nIn-sample and hindsight-selected: not out-of-sample evidence. Saved fixed_book_stats.csv, "
          "fixed_book_pnl.csv" + ("" if a.no_plot else ", fixed_book_pnl.png"))


if __name__ == "__main__":
    main()