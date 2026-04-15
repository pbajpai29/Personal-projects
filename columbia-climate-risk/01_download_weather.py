"""
01_download_weather.py
----------------------
Download daily weather data for New York City (2020-2024) using the
Open-Meteo Archive API — no API key required.

Variables fetched:
  - precipitation_sum    mm/day
  - temperature_2m_max   °C  (converted to °F)
  - temperature_2m_min   °C
  - temperature_2m_mean  °C
  - windspeed_10m_max    km/h

Extreme-event flags added (based on NOAA / NYC Emergency Management thresholds):
  - heavy_rain      : precip ≥ 25.4 mm (1 in)
  - extreme_rain    : precip ≥ 50.8 mm (2 in) — flash-flood risk
  - extreme_heat    : tmax ≥ 32.2 °C (90 °F) — NOAA heat threshold
  - heat_emergency  : tmax ≥ 35.0 °C (95 °F) — NYC emergency declaration threshold
  - heat_wave_day   : 3 or more consecutive extreme_heat days

Output: data/nyc_weather_daily.csv
"""

import sys
import requests
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NYC_LAT    = 40.7128
NYC_LON    = -74.0060
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "precipitation_sum",           # mm/day
    "temperature_2m_max",          # °C
    "temperature_2m_min",          # °C
    "temperature_2m_mean",         # °C
    "windspeed_10m_max",           # km/h
    "precipitation_hours",         # hrs/day with measurable rain
    "et0_fao_evapotranspiration",  # mm — proxy for drought stress
]

# Extreme-event thresholds
HEAVY_RAIN_MM    = 25.4   # 1 inch
EXTREME_RAIN_MM  = 50.8   # 2 inches
EXTREME_HEAT_C   = 32.2   # 90 °F
HEAT_EMERGENCY_C = 35.0   # 95 °F
HEAT_WAVE_DAYS   = 3      # consecutive days ≥ EXTREME_HEAT_C


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def fetch_weather() -> dict:
    params = {
        "latitude":   NYC_LAT,
        "longitude":  NYC_LON,
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "daily":      ",".join(DAILY_VARS),
        "timezone":   "America/New_York",
    }
    print(f"  GET {ARCHIVE_URL}")
    resp = requests.get(ARCHIVE_URL, params=params, timeout=90)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------
def build_dataframe(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame(raw["daily"])
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    # Rename to cleaner column names
    df.rename(columns={
        "precipitation_sum":          "precip_mm",
        "temperature_2m_max":         "tmax_c",
        "temperature_2m_min":         "tmin_c",
        "temperature_2m_mean":        "tmean_c",
        "windspeed_10m_max":          "windspeed_kmh",
        "precipitation_hours":        "precip_hours",
        "et0_fao_evapotranspiration": "et0_mm",
    }, inplace=True)

    # Unit conversions
    df["precip_in"] = df["precip_mm"] / 25.4
    df["tmax_f"]    = df["tmax_c"] * 9 / 5 + 32
    df["tmin_f"]    = df["tmin_c"] * 9 / 5 + 32
    df["tmean_f"]   = df["tmean_c"] * 9 / 5 + 32

    # Fill any missing values with 0 for precipitation
    df["precip_mm"] = df["precip_mm"].fillna(0)

    # --- Extreme event flags ---
    df["heavy_rain"]     = (df["precip_mm"] >= HEAVY_RAIN_MM).astype(int)
    df["extreme_rain"]   = (df["precip_mm"] >= EXTREME_RAIN_MM).astype(int)
    df["extreme_heat"]   = (df["tmax_c"] >= EXTREME_HEAT_C).astype(int)
    df["heat_emergency"] = (df["tmax_c"] >= HEAT_EMERGENCY_C).astype(int)

    # Heat wave: 3+ consecutive extreme_heat days
    wave, streak = [], 0
    for v in df["extreme_heat"]:
        streak = streak + 1 if v else 0
        wave.append(int(streak >= HEAT_WAVE_DAYS))
    df["heat_wave_day"] = wave

    # Combined "weather stress" score: 0-3
    df["weather_stress"] = (
        df["heavy_rain"] + df["extreme_rain"] + df["extreme_heat"]
    ).clip(upper=3)

    # Calendar helpers
    df["year"]   = df["date"].dt.year
    df["month"]  = df["date"].dt.month
    df["dow"]    = df["date"].dt.dayofweek   # 0=Mon, 6=Sun
    df["season"] = df["month"].map(
        lambda m: "Winter" if m in (12, 1, 2)
        else "Spring" if m in (3, 4, 5)
        else "Summer" if m in (6, 7, 8)
        else "Autumn"
    )

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> pd.DataFrame:
    print("=" * 60)
    print("  Step 1 — Download NYC Weather Data (Open-Meteo)")
    print("=" * 60)

    raw = fetch_weather()
    df  = build_dataframe(raw)

    out = Path(__file__).parent / "data" / "nyc_weather_daily.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n  Saved {len(df)} rows → {out.relative_to(Path(__file__).parent)}")
    print(f"  Date range      : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Heavy rain days : {df['heavy_rain'].sum()} ({df['heavy_rain'].mean():.1%})")
    print(f"  Extreme rain    : {df['extreme_rain'].sum()} ({df['extreme_rain'].mean():.1%})")
    print(f"  Extreme heat    : {df['extreme_heat'].sum()} ({df['extreme_heat'].mean():.1%})")
    print(f"  Heat wave days  : {df['heat_wave_day'].sum()}")
    print()
    return df


if __name__ == "__main__":
    main()
