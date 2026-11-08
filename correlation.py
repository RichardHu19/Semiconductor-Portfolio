"""
weekly_correlation.py -- weekly return correlations for the eight-stock universe,
shown as a table and a heat-map grid.

Data: Yahoo Finance daily closes via the unofficial `yfinance` package.
All returns are in USD, because the portfolio is valued in USD.

Conventions:
* Weekly return = Friday-to-Friday change in the last available close of the week
  (if Friday is a holiday, the last earlier close of that week is used).
* Prices are adjusted for splits and dividends (total return).
* Samsung (005930.KS) is converted from KRW to USD with Yahoo's KRW=X (KRW per USD).
* SK hynix: the ADS (SKHY) only started trading in July 2026. For earlier weeks we
  use a PROXY: 0.1 x KRX share price (000660.KS) converted to USD (1 ADS = 0.1 share).
  Weekly returns are spliced: proxy up to and including the listing week, actual
  ADS returns after that. The number of proxy weeks is printed.
* Limitation: Korean closes occur before New York's, so Korean-based series are
  non-synchronous with the US names. Weekly returns reduce this but do not remove it.

Install:  pip install yfinance pandas numpy matplotlib
Run:      python correlation.py                      (download, save snapshot)
Replay:   python correlation.py prices_snapshot_XXXX.csv   (no download)
Demo:     python correlation.py --demo               (SYNTHETIC data, test only)
"""
import argparse
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

START, END_INCLUSIVE = "2026-01-01", "2026-11-6"
RAW_TICKERS = ["NVDA", "AMD", "INTC", "MU", "ASML", "TSM",
               "SKHY", "000660.KS", "005930.KS", "KRW=X"]
US_NAMES = {"NVIDIA": "NVDA", "AMD": "AMD", "Intel": "INTC",
            "Micron": "MU", "ASML": "ASML", "TSMC": "TSM"}
SKHY_ADS_PER_SHARE = 0.1          # 1 ADS = 0.1 ordinary share
ORDER = ["NVIDIA", "AMD", "Intel", "Micron", "ASML", "TSMC", "SK hynix", "Samsung"]


# ------------------------------- data ----------------------------------------
def download_prices() -> pd.DataFrame:
    import yfinance as yf                       # imported here so replay needs no network
    end_exclusive = (pd.Timestamp(END_INCLUSIVE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(RAW_TICKERS, start=START, end=end_exclusive,
                      auto_adjust=True, progress=False)
    close = raw["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def demo_prices() -> pd.DataFrame:
    """SYNTHETIC prices to test the pipeline. Not real market data."""
    rng = np.random.default_rng(7)
    days = pd.bdate_range(START, END_INCLUSIVE)
    common = rng.normal(0, 0.012, len(days))
    out = {}
    for t in RAW_TICKERS[:-1]:
        r = 0.6 * common + rng.normal(0, 0.018, len(days))
        out[t] = 100 * np.exp(np.cumsum(r))
    out["KRW=X"] = 1400 * np.exp(np.cumsum(rng.normal(0, 0.003, len(days))))
    df = pd.DataFrame(out, index=days)
    df.loc[df.index < "2026-07-01", "SKHY"] = np.nan      # ADS lists in July
    return df


def save_snapshot(close: pd.DataFrame) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"prices_snapshot_{stamp}.csv"
    close.to_csv(path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    open(path + ".sha256", "w").write(digest + "\n")
    print(f"Saved raw prices to {path}  (sha256 {digest[:16]}...)")
    return path


# ------------------------------- returns -------------------------------------
def weekly_usd_returns(close: pd.DataFrame):
    """Return (weekly USD returns for the 8 issuers, number of SK hynix proxy weeks)."""
    close = close.loc[START:END_INCLUSIVE]
    fx = close["KRW=X"].ffill()                                  # KRW per USD
    usd = pd.DataFrame({name: close[tkr] for name, tkr in US_NAMES.items()})
    usd["Samsung"] = close["005930.KS"] / fx
    usd["SKHY proxy"] = SKHY_ADS_PER_SHARE * close["000660.KS"] / fx
    usd["SKHY ADS"] = close["SKHY"]

    weekly = usd.resample("W-FRI").last()                        # last close of each week
    keep_nan = ["SKHY ADS"]                                      # NaN before listing is real
    fill_cols = [c for c in weekly.columns if c not in keep_nan]
    weekly[fill_cols] = weekly[fill_cols].ffill(limit=1)         # bridge a missing week only
    ret = weekly / weekly.shift(1) - 1

    # Splice: ADS return where it exists (2nd listed week onward), proxy before that.
    ret["SK hynix"] = ret["SKHY ADS"].where(ret["SKHY ADS"].notna(), ret["SKHY proxy"])
    out = ret[ORDER].dropna(how="any")                           # same weeks for every pair
    proxy_weeks = int(ret.loc[out.index, "SKHY ADS"].isna().sum())
    return out, proxy_weeks


# ------------------------------- output --------------------------------------
def show_grid(corr: pd.DataFrame, n_weeks: int, title: str, show: bool) -> str:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    k = len(corr)
    ax.set_xticks(range(k), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(k), corr.index)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title(f"{title}\nWeekly USD returns, {n_weeks} weeks, 1 Jan - 31 Oct 2026")
    fig.tight_layout()
    path = "weekly_correlation_grid.png"
    fig.savefig(path, dpi=160)
    if show:
        plt.show()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", nargs="?", help="saved prices CSV to replay (no download)")
    ap.add_argument("--demo", action="store_true", help="use SYNTHETIC data to test the code")
    ap.add_argument("--no-show", action="store_true", help="save the figure without opening it")
    args = ap.parse_args()

    if args.demo:
        close = demo_prices()
    elif args.snapshot:
        close = pd.read_csv(args.snapshot, index_col=0, parse_dates=True)
    else:
        close = download_prices()
        save_snapshot(close)

    rets, proxy_weeks = weekly_usd_returns(close)
    corr = rets.corr()
    n = len(rets)

    pd.set_option("display.width", 160)
    print(f"\nWeekly return correlation matrix ({n} weekly observations)\n")
    print(corr.round(2).to_string())
    names = list(corr.columns)
    iu = np.triu_indices(len(names), k=1)                        # each pair once, no diagonal
    pairs = sorted((corr.values[i, j], names[i], names[j]) for i, j in zip(*iu))
    print("\nLowest-correlated pairs: ", ", ".join(f"{a}/{b} {v:.2f}" for v, a, b in pairs[:3]))
    print("Highest-correlated pairs:", ", ".join(f"{a}/{b} {v:.2f}" for v, a, b in pairs[-3:][::-1]))
    print(f"\nSK hynix: {proxy_weeks} of {n} weeks use the KRX-based proxy; the rest use actual SKHY ADS returns.")
    print(f"Sampling error: with n={n}, a correlation of 0.6 has a standard error of about "
          f"{(1 - 0.6 ** 2) / np.sqrt(n):.2f}, so small differences between pairs are not meaningful.")

    rets.to_csv("weekly_returns.csv")
    corr.to_csv("weekly_correlation_matrix.csv")
    title = "SYNTHETIC DEMO - not market data" if args.demo else "Correlation of weekly returns"
    png = show_grid(corr, n, title, show=not args.no_show)
    print(f"\nSaved: weekly_returns.csv, weekly_correlation_matrix.csv, {png}")


if __name__ == "__main__":
    main()