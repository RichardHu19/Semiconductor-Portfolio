"""
inputs.py -- build inputs.csv and cov.csv for max_return_book.py.

Uses the programs you already have (same folder):
    correlation.py                             weekly USD returns, SK hynix proxy splice
    implied_vol.py                             ATM implied vols from Yahoo option chains
    samsung_vol.py                             Samsung implied-vol proxy from peers

What it produces
  inputs.csv  issuer, ticker, price, price_usd, price_date, fair_value, [fv_date],
              implied_vol, iv_source          (annualized implied vol, decimals)
  cov.csv     weekly covariance of the 8 issuers' USD returns plus a MARKET column (SPY),
              so max_return_book.py gets market betas. NOTE: it is in WEEKLY units.
              max_return_book.py uses only its correlations (it rescales with implied
              vol), so do NOT pass --cov-as-is.

Fair values: Morningstar's are not available through a free API, so YOU supply them.
  First run (no fair-values file): writes fair_values_template.csv pre-filled with
  current prices; fill in `fair_value` (and `fv_date`), save as fair_values.csv, rerun.
  Units: fair_value per LISTED share in the listing currency (NVDA, AMD, INTC, MU,
  ASML, TSM and SK hynix's ADS in USD; Samsung in KRW). Exceptions handled for you:
  SK hynix quoted in KRW per ordinary share (fv_currency = KRW) and Samsung quoted in
  USD (fv_currency = USD). Any other currency must be converted by you first.

Volatility sources, in order: observed ATM implied vol -> Samsung proxy (samsung_vol.py)
-> FALLBACK EWMA realized vol. The source of every number is written to iv_source;
do not hide a fallback in your memo.

Run:     python inputs.py --asof 2026-11-16
Replay:  python inputs.py --prices-csv prices.csv --iv-csv inputs_iv_X.csv --asof 2026-11-16
         (a live run saves both files; keep them, and fair_values.csv, with your submission)
Run after the close of the as-of date, or use the prior trading day.
"""
import argparse
import hashlib
import os
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

try:
    import correlation as wc
except ImportError:                              # you may have renamed it
    import correlation as wc
import implied_vol as ivm
try:
    import samsung_vol as sv
except ImportError:
    sv = None

MARKET = "SPY"
LISTING = {**wc.US_NAMES, "SK hynix": "SKHY", "Samsung": "005930.KS"}
ORDER = wc.ORDER
FV_RATIO_RANGE = (0.3, 3.0)                      # sanity band for fair_value / price


# ------------------------------- prices --------------------------------------
def load_prices(path, start, end):
    if path:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    tickers = sorted(set(wc.RAW_TICKERS + [MARKET]))
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    px = yf.download(tickers, start=start, end=end_excl, auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = f"inputs_prices_{stamp}.csv"
    px.to_csv(out)
    digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
    open(out + ".sha256", "w").write(digest + "\n")
    print(f"Saved prices to {out} (sha256 {digest[:16]}...)")
    return px


def current_prices(px, asof, warnings):
    px = px.loc[:asof]
    fx = px["KRW=X"].ffill()
    rows = []
    for issuer, tkr in LISTING.items():
        s = px[tkr].dropna()
        if s.empty:
            raise ValueError(f"No price history for {tkr} up to {asof}")
        d, p = s.index[-1], float(s.iloc[-1])
        if (pd.Timestamp(asof) - d).days > 5:
            warnings.append(f"{issuer}: last close {d.date()} is more than 5 days before {asof}")
        p_usd = p / float(fx.loc[:d].iloc[-1]) if tkr.endswith(".KS") else p
        rows.append({"issuer": issuer, "ticker": tkr, "price": p, "price_usd": p_usd,
                     "price_date": d.date().isoformat()})
    return pd.DataFrame(rows).set_index("issuer"), float(fx.iloc[-1])


# ------------------------------- fair values ---------------------------------
def fv_in_price_basis(issuer, fv, fv_ccy, fx_krw_per_usd):
    """Convert a fair value to the same currency and share basis as the listing price."""
    ccy = (fv_ccy or "").strip().upper()
    if issuer == "SK hynix" and ccy == "KRW":        # per ordinary share -> USD per ADS
        return fv * wc.SKHY_ADS_PER_SHARE / fx_krw_per_usd
    if issuer == "Samsung" and ccy == "USD":         # USD per share -> KRW per share
        return fv * fx_krw_per_usd
    expected = "KRW" if issuer == "Samsung" else "USD"
    if ccy not in ("", expected):
        raise ValueError(f"{issuer}: fv_currency {ccy} not supported; convert to {expected} first.")
    return fv


def apply_fair_values(prices, fv_path, fx, warnings):
    fv = pd.read_csv(fv_path).set_index("issuer")
    if fv["fair_value"].isna().any():
        raise ValueError("fair_values file has blank fair_value entries: "
                         + ", ".join(fv.index[fv["fair_value"].isna()]))
    out = prices.copy()
    out["fair_value"] = [fv_in_price_basis(i, float(fv.loc[i, "fair_value"]),
                                           fv.loc[i, "fv_currency"] if "fv_currency" in fv else "", fx)
                         for i in out.index]
    if "fv_date" in fv:
        out["fv_date"] = fv.loc[out.index, "fv_date"]
    lo, hi = FV_RATIO_RANGE
    for i in out.index:
        r = out.loc[i, "fair_value"] / out.loc[i, "price"]
        if not lo <= r <= hi:
            warnings.append(f"{i}: fair_value/price = {r:.2f}: check currency, ADS ratio and units")
    return out


# ------------------------------- volatility ----------------------------------
def daily_usd_logret(px, issuer):
    fx = px["KRW=X"].ffill()
    if issuer == "Samsung":
        return np.log(px["005930.KS"] / fx).diff()
    if issuer == "SK hynix":                          # ADS where it exists, KRX proxy before
        adr = np.log(px["SKHY"]).diff()
        proxy = np.log(wc.SKHY_ADS_PER_SHARE * px["000660.KS"] / fx).diff()
        return adr.where(adr.notna(), proxy)
    return np.log(px[LISTING[issuer]]).diff()


def ewma_vol(r, lam=0.94, window=250):
    r = r.dropna().tail(window)
    var = r.head(30).var()
    for x in r:
        var = lam * var + (1 - lam) * x * x
    return float(np.sqrt(var * 252))


def get_ivs(iv_csv, out_dir="."):
    """Replay from a saved CSV (ticker, iv_annual), or pull live and SAVE what was pulled
    so the run can be replayed exactly."""
    if iv_csv:
        return pd.read_csv(iv_csv).set_index("ticker")["iv_annual"]
    ivs = {}
    for sym in ivm.TICKERS.values():
        try:
            ivs[sym] = ivm.get_implied_vol(sym).get("iv_annual", np.nan)
        except Exception as exc:
            print(f"WARNING: implied vol failed for {sym}: {exc}")
            ivs[sym] = np.nan
    s = pd.Series(ivs, name="iv_annual").rename_axis("ticker")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"inputs_iv_{stamp}.csv")
    s.reset_index().assign(pulled_utc=datetime.now(timezone.utc).isoformat(timespec="seconds")).to_csv(path, index=False)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    open(path + ".sha256", "w").write(digest + "\n")
    print(f"Saved implied vols to {path} (sha256 {digest[:16]}...)")
    return s


def build_vols(px, ivs, warnings):
    vol, src = {}, {}
    for issuer, tkr in LISTING.items():
        v = ivs.get(tkr, np.nan)
        if np.isfinite(v):
            vol[issuer], src[issuer] = float(v), "observed ATM implied vol (Yahoo options)"
    if "Samsung" not in vol and sv is not None:
        try:
            _, _, est = sv.estimate(px, ivs.reindex(sv.PEERS))
            vol["Samsung"] = float(est.loc["median", "iv_usd_annual"])
            src["Samsung"] = "PROXY: peer implied/realized ratio, median (samsung_vol.py)"
        except Exception as exc:
            warnings.append(f"Samsung proxy failed ({exc}); using EWMA fallback")
    for issuer in ORDER:
        if issuer not in vol:
            vol[issuer], src[issuer] = ewma_vol(daily_usd_logret(px, issuer)), \
                "FALLBACK: EWMA realized vol (NOT implied)"
    for issuer, v in vol.items():
        if not 0.15 <= v <= 1.5:
            warnings.append(f"{issuer}: annual vol {v:.2f} looks implausible; check the source")
    return pd.Series(vol), pd.Series(src)


# ------------------------------- covariance ----------------------------------
def build_cov(px, start, asof):
    wc.START, wc.END_INCLUSIVE = start, asof          # weekly_correlation reads these globals
    rets, proxy_weeks = wc.weekly_usd_returns(px)
    rets = rets[rets.index <= pd.Timestamp(asof)]      # drop an incomplete final week
    spy = px.loc[start:asof, MARKET].resample("W-FRI").last()
    mkt = (spy / spy.shift(1) - 1).rename("MARKET")
    both = rets.join(mkt, how="inner").dropna()
    return both.cov(), len(both), proxy_weeks


# ------------------------------- main ----------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", default=date.today().isoformat(), help="as-of date, YYYY-MM-DD")
    ap.add_argument("--years", type=float, default=5.0, help="history for the covariance")
    ap.add_argument("--fair-values", default="fair_values.csv")
    ap.add_argument("--prices-csv"), ap.add_argument("--iv-csv")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    warnings = []
    os.makedirs(a.out_dir, exist_ok=True)

    start = (pd.Timestamp(a.asof) - pd.Timedelta(days=int(a.years * 365))).strftime("%Y-%m-%d")
    px = load_prices(a.prices_csv, start, a.asof)
    prices, fx = current_prices(px, a.asof, warnings)

    if not os.path.exists(a.fair_values):
        tpl = prices[["ticker", "price"]].rename(columns={"price": "current_price"})
        tpl["fair_value"], tpl["fv_date"] = "", ""
        tpl["fv_currency"] = ["KRW" if i == "Samsung" else "USD" for i in tpl.index]
        path = os.path.join(a.out_dir, "fair_values_template.csv")
        tpl.to_csv(path)
        print(f"\n{a.fair_values} not found. Wrote {path} with current prices.\n"
              "Fill in fair_value (and fv_date), save as fair_values.csv, then rerun.\n"
              "SK hynix: use fv_currency KRW for a per-ordinary-share KRW fair value, or USD per ADS.")
        return

    inp = apply_fair_values(prices, a.fair_values, fx, warnings)
    vols, src = build_vols(px, get_ivs(a.iv_csv, a.out_dir), warnings)
    inp["implied_vol"], inp["iv_source"] = vols.reindex(inp.index), src.reindex(inp.index)
    cov, n_weeks, proxy_weeks = build_cov(px, start, a.asof)
    inp = inp.loc[ORDER]
    cov = cov.loc[ORDER + ["MARKET"], ORDER + ["MARKET"]]

    inp.rename_axis("issuer").to_csv(os.path.join(a.out_dir, "inputs.csv"))
    cov.to_csv(os.path.join(a.out_dir, "cov.csv"))

    pd.set_option("display.width", 170, "display.max_colwidth", 60)
    print(f"\nAs of {a.asof}; covariance from {n_weeks} weekly observations "
          f"({proxy_weeks} SK hynix weeks use the KRX-based proxy)\n")
    print(inp[["ticker", "price", "fair_value", "implied_vol", "iv_source"]].round(3).to_string())
    for w in warnings:
        print("WARNING:", w)
    print("\nWrote inputs.csv and cov.csv. Next:")
    print("  python max_return_book.py --inputs inputs.csv --cov cov.csv --vol-cap <dollars>")
    print("Remember: cov.csv is in weekly units (do not use --cov-as-is); implied vols are for the "
          f"{ivm.TARGET_DAYS}-day horizon used by implied_vol.py (default --horizon-days 30).")


if __name__ == "__main__":
    main()