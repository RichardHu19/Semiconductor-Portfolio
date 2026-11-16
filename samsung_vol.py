"""
samsung_vol.py -- estimate Samsung Electronics' implied volatility 
from its peers' implied-to-realized volatility ratio.

Method
1. For each peer with listed options, ratio_i = implied vol_i / realized vol_i
   (implied vol: ATM, interpolated to the target horizon; realized: last WINDOW
   trading days of daily log returns, annualized with sqrt(252)).
2. Samsung local-currency (KRW) proxy:  IV_local = ratio * RV_Samsung_local,
   using the lowest, median and highest peer ratio to give a range.
3. Convert to USD (your book is valued in USD). In log returns, USD return = local
   return + won return, so
        sigma_USD^2 = sigma_local^2 + sigma_FX^2 + 2 * rho * sigma_local * sigma_FX
   with sigma_FX and rho estimated from realized data over the same window.
4. Horizon vol = annual vol * sqrt(TARGET_DAYS / 365).

Assumptions: the implied/realized premium of the peers carries
over to Samsung; the FX part is valued at realized vol (no free FX implied vol);
RV window and peer set are judgment calls. 

Run (live):    python samsung_vol.py
Replay:        python samsung_vol.py --prices-csv prices_X.csv --iv-csv implied_vol_snapshot_X.csv
Needs:         pip install yfinance pandas numpy   (and implied_vol.py in the same folder)
"""
import argparse
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PEERS = ["MU", "TSM", "ASML"]       # peers with listed US options (a judgment call)
SAMSUNG, FX = "005930.KS", "KRW=X"  # FX = KRW per USD on Yahoo
WINDOW = 60                         # trading days for realized vol and correlation
TARGET_DAYS = 30                    # evaluation-horizon in calendar days


# ------------------------------- inputs --------------------------------------
def load_prices(path=None, calendar_days=300):
    if path:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf                                   # imported only when downloading
    start = (pd.Timestamp.today() - pd.Timedelta(days=calendar_days)).strftime("%Y-%m-%d")
    px = yf.download(PEERS + [SAMSUNG, FX], start=start, auto_adjust=True,
                     progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = f"samsung_prices_{stamp}.csv"
    px.to_csv(out)
    digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
    open(out + ".sha256", "w").write(digest + "\n")
    print(f"Saved prices to {out} (sha256 {digest[:16]}...)")
    return px


def load_peer_ivs(path=None):
    """Annualized ATM implied vols. From a saved implied_vol.py snapshot, or live."""
    if path:
        df = pd.read_csv(path)
        return df.set_index("ticker")["iv_annual"].reindex(PEERS)
    from implied_vol import get_implied_vol                  # live option-chain pull
    ivs = {}
    for t in PEERS:
        try:
            ivs[t] = get_implied_vol(t, TARGET_DAYS).get("iv_annual", np.nan)
        except Exception as exc:
            print(f"WARNING: no implied vol for {t}: {exc}")
            ivs[t] = np.nan
    return pd.Series(ivs)


# ------------------------------- maths ---------------------------------------
def realized_vol(logret, window=WINDOW):
    return float(logret.dropna().tail(window).std(ddof=1) * np.sqrt(252))


def estimate(prices, ivs, window=WINDOW):
    logret = np.log(prices).diff()
    peers = pd.DataFrame({"iv": ivs})
    peers["rv"] = [realized_vol(logret[t], window) for t in peers.index]
    peers["ratio"] = peers["iv"] / peers["rv"]
    ratios = peers["ratio"].dropna()
    if ratios.empty:
        raise ValueError("No peer has a usable implied vol: use a realized-vol fallback.")

    # Samsung and the won on COMMON dates only, so both returns cover the same interval
    pair = prices[[SAMSUNG, FX]].dropna()
    r = np.log(pair).diff().dropna().tail(window)
    r_loc = r[SAMSUNG]
    r_krw = -r[FX]                                           # log return of the won's USD value
    s_loc_rv = float(r_loc.std(ddof=1) * np.sqrt(252))
    s_fx = float(r_krw.std(ddof=1) * np.sqrt(252))
    rho = float(r_loc.corr(r_krw))
    s_usd_rv = float((r_loc + r_krw).std(ddof=1) * np.sqrt(252))   # direct realized USD vol

    rows = {}
    for label, ratio in (("low", ratios.min()), ("median", ratios.median()), ("high", ratios.max())):
        s_loc = ratio * s_loc_rv
        s_usd = float(np.sqrt(s_loc ** 2 + s_fx ** 2 + 2 * rho * s_loc * s_fx))
        rows[label] = {"ratio": ratio, "iv_local_annual": s_loc, "iv_usd_annual": s_usd,
                       "iv_usd_horizon": s_usd * np.sqrt(TARGET_DAYS / 365),
                       "simple_ratio_x_rv_usd": ratio * s_usd_rv}
    stats = {"rv_local": s_loc_rv, "rv_fx": s_fx, "rho": rho, "rv_usd": s_usd_rv}
    return peers, stats, pd.DataFrame(rows).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices-csv", help="saved daily closes (replay)")
    ap.add_argument("--iv-csv", help="implied_vol.py snapshot with columns ticker, iv_annual")
    args = ap.parse_args()

    prices = load_prices(args.prices_csv)
    ivs = load_peer_ivs(args.iv_csv)
    peers, stats, est = estimate(prices, ivs)

    pd.set_option("display.width", 150)
    print(f"\nPeers: implied vs realized vol ({WINDOW}-day window, annualized)")
    print(peers.round(3).to_string())
    print(f"\nSamsung realized (last {WINDOW} common trading days): local vol {stats['rv_local']:.3f}, "
          f"won vol {stats['rv_fx']:.3f}, corr(local, won) {stats['rho']:.2f}, USD vol {stats['rv_usd']:.3f}")
    print("\nSamsung implied-vol PROXY (annualized unless stated)")
    print(est.round(3).to_string())
    print("\n'iv_usd_*' applies the ratio to the local equity vol and adds FX at realized vol (preferred).")
    print("'simple_ratio_x_rv_usd' applies the ratio to total USD realized vol (comparison only).")
    est.to_csv("samsung_vol_estimate.csv")
    print("Saved samsung_vol_estimate.csv")


if __name__ == "__main__":
    main()