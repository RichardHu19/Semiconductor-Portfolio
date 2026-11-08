"""
implied_vol.py -- at-the-money (ATM) implied volatility from Yahoo Finance option
chains (via the unofficial `yfinance` package), interpolated to a target horizon.

How it works
1. For each stock, list the option expiries and pick the two that bracket TARGET_DAYS.
2. In each expiry, take the call and the put whose strike is closest to the spot price
   (ignoring quotes with no bid/ask or junk implied vol) and average their IVs.
3. Interpolate between the two expiries in TOTAL VARIANCE (sigma^2 * T), which is
   linear in time, then convert back to an annualized vol.
4. Convert to the horizon: sigma_horizon = sigma_annual * sqrt(TARGET_DAYS / 365).

Limitations (put these in your memo)
* Yahoo computes `impliedVolatility` itself, and its method is not documented.
  Illiquid or stale quotes can give junk values; the filters below reduce this.
* IV is risk-neutral and usually exceeds realized vol (variance risk premium).
* An expiry that spans an earnings date (NVDA and AMD report in your window) has
  event variance built in, so the IV is higher than "normal" days.
* 005930.KS (Samsung, KRX) has no US options on Yahoo. Those come back NaN -> 
  use EWMA/realized vol as a fallback.

Install:  pip install yfinance pandas numpy
Run:      python implied_vol.py
"""
import hashlib
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

TICKERS = {"NVIDIA": "NVDA", "AMD": "AMD", "Intel": "INTC", "Micron": "MU",
           "ASML": "ASML", "TSMC": "TSM", "SK hynix": "SKHY"}   # Samsung: no US options
TARGET_DAYS = 30          # 1 month window
MIN_IV = 0.05             # drop junk implied vols (Yahoo shows ~0 for dead quotes)


def atm_iv(chain, spot):
    """Average IV of the call and put nearest to spot. Returns (iv, strikes used)."""
    ivs, strikes = [], []
    for side in (chain.calls, chain.puts):
        d = side[(side["bid"] > 0) & (side["ask"] > 0) & (side["impliedVolatility"] > MIN_IV)]
        if d.empty:
            continue
        row = d.iloc[int((d["strike"] - spot).abs().to_numpy().argmin())]
        ivs.append(float(row["impliedVolatility"]))
        strikes.append(float(row["strike"]))
    return (float(np.mean(ivs)), strikes) if ivs else (np.nan, [])


def interpolate_iv(points, target_days):
    """points: list of (days_to_expiry, iv). Linear in total variance between the
    bracketing expiries. If the target is outside the range, use the nearest expiry."""
    pts = sorted(p for p in points if p[0] > 0 and not np.isnan(p[1]))
    if not pts:
        return np.nan, (None, None)
    lo = [p for p in pts if p[0] <= target_days]
    hi = [p for p in pts if p[0] >= target_days]
    if not lo:
        return hi[0][1], (hi[0][0], hi[0][0])
    if not hi:
        return lo[-1][1], (lo[-1][0], lo[-1][0])
    (d1, s1), (d2, s2) = lo[-1], hi[0]
    if d1 == d2:
        return s1, (d1, d2)
    w1, w2 = s1 ** 2 * d1, s2 ** 2 * d2
    w = w1 + (w2 - w1) * (target_days - d1) / (d2 - d1)
    return float(np.sqrt(w / target_days)), (d1, d2)


def get_implied_vol(ticker_symbol, target_days=TARGET_DAYS):
    import yfinance as yf                                  # lazy import
    tk = yf.Ticker(ticker_symbol)
    expiries = list(tk.options)                            # tuple of 'YYYY-MM-DD'
    if not expiries:
        return {"status": "no listed options"}
    spot = float(tk.history(period="5d")["Close"].dropna().iloc[-1])
    today = date.today()
    days = {e: (date.fromisoformat(e) - today).days for e in expiries}
    days = {e: d for e, d in days.items() if d >= 5}       # skip expiries about to lapse
    below = [e for e, d in days.items() if d <= target_days]
    above = [e for e, d in days.items() if d >= target_days]
    chosen = []
    if below:
        chosen.append(max(below, key=days.get))            # nearest expiry at/below target
    if above:
        chosen.append(min(above, key=days.get))            # nearest expiry at/above target
    points = []
    for e in dict.fromkeys(chosen):                        # unique, order kept
        iv, _ = atm_iv(tk.option_chain(e), spot)
        points.append((days[e], iv))
    iv, (d1, d2) = interpolate_iv(points, target_days)
    return {"spot": spot, "days_lo": d1, "days_hi": d2, "iv_annual": iv,
            "iv_horizon": iv * np.sqrt(target_days / 365) if not np.isnan(iv) else np.nan,
            "status": "ok" if not np.isnan(iv) else "no usable quotes"}


def main():
    rows = []
    for name, sym in TICKERS.items():
        try:
            res = get_implied_vol(sym)
        except Exception as exc:                           # network error, bad ticker, etc.
            res = {"status": f"error: {exc}"}
        rows.append({"issuer": name, "ticker": sym, **res})
    df = pd.DataFrame(rows)
    df["pulled_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with pd.option_context("display.width", 150, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"implied_vol_snapshot_{stamp}.csv"
    df.to_csv(path, index=False)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    open(path + ".sha256", "w").write(digest + "\n")
    print(f"\nSaved {path} (sha256 {digest[:16]}...). Samsung (KRX) has no US options: use a fallback.")


if __name__ == "__main__":
    main()