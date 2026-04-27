"""
03_merge_process.py
-------------------
Merge NYC weather and MTA delay datasets into a single analysis-ready table.

Key outputs
-----------
data/analysis_daily.csv      — one row per (date, line)
data/analysis_systemwide.csv — one row per date (all lines aggregated)

New columns added:
  lag1_precip_mm         : precipitation the previous day
  lag2_precip_mm         : precipitation 2 days ago
  roll3_precip_mm        : 3-day rolling total precipitation
  roll7_precip_mm        : 7-day rolling total precipitation
  prev_extreme_heat      : extreme_heat flag the previous day
  heat_index             : simplified heat index (°C) for humidity proxy
  incidents_per_train    : total_incidents / scheduled_trains (intensity)
  pct_weather_incidents  : weather_related_incidents / total_incidents
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    weather_path = BASE / "data" / "nyc_weather_daily.csv"
    mta_path     = BASE / "data" / "mta_delays_daily.csv"

    if not weather_path.exists():
        raise FileNotFoundError(
            f"Weather data not found. Run 01_download_weather.py first.\n{weather_path}"
        )
    if not mta_path.exists():
        raise FileNotFoundError(
            f"MTA data not found. Run 02_download_mta.py first.\n{mta_path}"
        )

    weather = pd.read_csv(weather_path, parse_dates=["date"])
    mta     = pd.read_csv(mta_path,     parse_dates=["date"])
    return weather, mta


# ---------------------------------------------------------------------------
# Build system-wide daily MTA (sum across all lines)
# ---------------------------------------------------------------------------
def aggregate_systemwide(mta: pd.DataFrame) -> pd.DataFrame:
    agg = (
        mta.groupby("date")
        .agg(
            total_incidents            = ("total_incidents",            "sum"),
            weather_related_incidents  = ("weather_related_incidents",  "sum"),
            avg_delay_min              = ("avg_delay_min",              "mean"),
            disrupted_trains           = ("disrupted_trains",           "sum"),
        )
        .reset_index()
    )
    agg["pct_weather_incidents"] = np.where(
        agg["total_incidents"] > 0,
        agg["weather_related_incidents"] / agg["total_incidents"],
        0,
    )
    return agg


# ---------------------------------------------------------------------------
# Add weather lag / rolling features
# ---------------------------------------------------------------------------
def add_weather_lags(weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.sort_values("date").copy()
    w["lag1_precip_mm"]    = w["precip_mm"].shift(1)
    w["lag2_precip_mm"]    = w["precip_mm"].shift(2)
    w["lag3_precip_mm"]    = w["precip_mm"].shift(3)
    w["roll3_precip_mm"]   = w["precip_mm"].rolling(3).sum()
    w["roll7_precip_mm"]   = w["precip_mm"].rolling(7).sum()
    w["prev_extreme_heat"] = w["extreme_heat"].shift(1).fillna(0).astype(int)
    w["prev_heavy_rain"]   = w["heavy_rain"].shift(1).fillna(0).astype(int)

    # Simplified heat index proxy (Steadman 1979 simplified):
    # HI ≈ T + 0.33 × (dewpoint proxy) − 4.0  — we approximate dewpoint from Tmin
    # This is a rough proxy; real dewpoint not available from Open-Meteo daily aggregates
    w["heat_index_approx"] = w["tmax_c"] + 0.33 * (w["tmin_c"] * 0.7) - 4.0

    # Cumulative heat stress: days into heat wave
    streak, streaks = 0, []
    for v in w["extreme_heat"]:
        streak = streak + 1 if v else 0
        streaks.append(streak)
    w["heat_wave_streak"] = streaks

    return w


# ---------------------------------------------------------------------------
# Scheduled trains lookup (approximate — from MTA published timetables)
# Used to normalise incident rates by service volume
# ---------------------------------------------------------------------------
SCHEDULED_TRAINS_PER_LINE = {
    "1": 370, "2": 340, "3": 290, "4": 410, "5": 310, "6": 450, "7": 330,
    "A": 390, "B": 220, "C": 260, "D": 250, "E": 330, "F": 380, "G": 180,
    "J": 210, "L": 290, "M": 200, "N": 310, "Q": 290, "R": 340, "W": 170,
    "Z": 140, "SIR": 70,
}


# ---------------------------------------------------------------------------
# Merge and enrich
# ---------------------------------------------------------------------------
def merge_and_enrich(
    weather: pd.DataFrame,
    mta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    weather_lag = add_weather_lags(weather)
    mta_sys     = aggregate_systemwide(mta)

    # --- Per-line daily dataset ---
    line_daily = mta.merge(weather_lag, on="date", how="inner")
    line_daily["scheduled_trains"]   = line_daily["line"].map(SCHEDULED_TRAINS_PER_LINE)
    line_daily["incidents_per_train"] = np.where(
        line_daily["scheduled_trains"] > 0,
        line_daily["total_incidents"] / line_daily["scheduled_trains"],
        np.nan,
    )
    line_daily["pct_weather_incidents"] = np.where(
        line_daily["total_incidents"] > 0,
        line_daily["weather_related_incidents"] / line_daily["total_incidents"],
        0,
    )
    line_daily["delay_weighted"] = (
        line_daily["avg_delay_min"] * line_daily["total_incidents"]
    )

    # --- System-wide daily dataset ---
    sys_daily = mta_sys.merge(weather_lag, on="date", how="inner")

    return line_daily, sys_daily


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def print_summary(sys_daily: pd.DataFrame) -> None:
    print("\n── Dataset Summary ──────────────────────────────────────")
    print(f"  Rows          : {len(sys_daily):,}")
    print(f"  Date range    : {sys_daily['date'].min().date()} → {sys_daily['date'].max().date()}")

    for label, mask in [
        ("Normal days",       (~sys_daily["heavy_rain"].astype(bool)) & (~sys_daily["extreme_heat"].astype(bool))),
        ("Heavy rain days",   sys_daily["heavy_rain"].astype(bool)),
        ("Extreme heat days", sys_daily["extreme_heat"].astype(bool)),
        ("Extreme rain days", sys_daily["extreme_rain"].astype(bool)),
    ]:
        sub = sys_daily[mask]
        if not sub.empty:
            print(
                f"  {label:<22}: n={len(sub):>4}  "
                f"avg incidents={sub['total_incidents'].mean():.1f}  "
                f"avg delay={sub['avg_delay_min'].mean():.1f} min"
            )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 60)
    print("  Step 3 — Merge & Process Data")
    print("=" * 60)

    weather, mta = load_datasets()
    line_daily, sys_daily = merge_and_enrich(weather, mta)

    # Save
    out_line = BASE / "data" / "analysis_daily.csv"
    out_sys  = BASE / "data" / "analysis_systemwide.csv"

    line_daily.to_csv(out_line, index=False)
    sys_daily.to_csv(out_sys,  index=False)

    print(f"\n  Saved line-level  → {out_line.relative_to(BASE)}")
    print(f"  Saved system-wide → {out_sys.relative_to(BASE)}")

    print_summary(sys_daily)
    return line_daily, sys_daily


if __name__ == "__main__":
    main()
