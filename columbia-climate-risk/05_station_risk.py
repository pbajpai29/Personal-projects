"""
05_station_risk.py
------------------
Fetch MTA subway station locations from data.ny.gov and compute
climate risk scores per station and per line.

Risk dimensions (0–10 each):
  flood_risk        — rainfall/flood exposure (geography + real incident uplift)
  heat_risk         — extreme heat exposure (UHI + underground + real uplift)
  vulnerability     — infrastructure age & structure type
  economic_exposure — ridership-weighted loss potential

Sources:
  Stations  : data.ny.gov/resource/39hk-dx4f (MTA Subway Stations)
  Risk model: MTA post-Sandy reports, NYC climate vulnerability literature

Output:
  data/station_risk.csv
  data/line_risk.csv
"""

import re
import requests
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

# ---------------------------------------------------------------------------
# MTA official line colours
# ---------------------------------------------------------------------------
LINE_COLORS = {
    "1":"#EE352E","2":"#EE352E","3":"#EE352E",
    "4":"#00933C","5":"#00933C","6":"#00933C",
    "7":"#B933AD",
    "A":"#0039A6","C":"#0039A6","E":"#0039A6",
    "B":"#FF6319","D":"#FF6319","F":"#FF6319","M":"#FF6319",
    "G":"#6CBE45",
    "J":"#996633","Z":"#996633","JZ":"#996633",
    "L":"#A7A9AC",
    "N":"#FCCC0A","Q":"#FCCC0A","R":"#FCCC0A","W":"#FCCC0A",
    "S 42ND":"#808183","S FKLN":"#808183","S ROCK":"#808183",
    "SIR":"#1D6363",
}

# ---------------------------------------------------------------------------
# Expert-coded geographic risk by line
# Sources: MTA Sandy After-Action 2013, NYC Climate Risk 2023, FEMA FIRM maps
# ---------------------------------------------------------------------------
GEO_FLOOD = {
    "A":8.5,"C":7.5,"E":6.0,
    "B":5.0,"D":5.5,"F":7.0,"M":5.0,
    "G":7.5,"J":5.5,"Z":5.5,"JZ":5.5,
    "L":8.0,
    "N":6.5,"Q":6.0,"R":6.5,"W":6.0,
    "1":6.0,"2":5.5,"3":5.0,
    "4":5.0,"5":5.0,"6":5.5,"7":4.5,
    "S 42ND":4.0,"S FKLN":4.5,"S ROCK":4.0,"SIR":5.0,
}
GEO_HEAT = {
    "A":6.0,"C":6.5,"E":6.0,
    "B":5.5,"D":5.5,"F":6.0,"M":6.0,
    "G":6.5,"J":5.5,"Z":5.5,"JZ":5.5,
    "L":7.0,
    "N":5.5,"Q":5.0,"R":6.0,"W":5.0,
    "1":6.5,"2":7.0,"3":6.0,
    "4":6.5,"5":7.5,"6":7.0,"7":5.5,
    "S 42ND":5.0,"S FKLN":5.0,"S ROCK":5.0,"SIR":4.5,
}
GEO_VULN = {
    "A":7.0,"C":7.5,"E":7.0,
    "B":6.0,"D":6.0,"F":6.5,"M":6.0,
    "G":7.0,"J":5.5,"Z":5.5,"JZ":5.5,
    "L":8.5,
    "N":6.5,"Q":6.0,"R":6.5,"W":5.5,
    "1":7.5,"2":7.0,"3":7.0,
    "4":7.0,"5":6.5,"6":7.0,"7":5.5,
    "S 42ND":5.0,"S FKLN":5.0,"S ROCK":5.0,"SIR":5.5,
}
LINE_RIDERSHIP = {  # avg weekday riders (thousands) — MTA 2023
    "A":250,"C":120,"E":195,"B":135,"D":155,"F":295,"M":105,
    "G":90,"J":80,"Z":35,"JZ":80,"L":205,
    "N":165,"Q":175,"R":180,"W":60,
    "1":330,"2":290,"3":185,"4":335,"5":200,"6":400,"7":290,
    "S 42ND":15,"S FKLN":10,"S ROCK":8,"SIR":20,
}
MAX_RIDERSHIP = max(LINE_RIDERSHIP.values())

# Structure vulnerability bonus
STRUCT_VULN = {"underground": 2.0, "elevated": 0.5, "open cut": 1.0, "at grade": 0.0}


# ---------------------------------------------------------------------------
# Fetch stations from data.ny.gov
# ---------------------------------------------------------------------------
def fetch_stations() -> pd.DataFrame:
    print("  Fetching MTA stations from data.ny.gov…")
    url  = "https://data.ny.gov/resource/39hk-dx4f.json"
    resp = requests.get(url, params={"$limit": 600}, timeout=30)
    resp.raise_for_status()
    raw  = pd.DataFrame(resp.json())
    print(f"  → {len(raw)} station records")

    df = pd.DataFrame({
        "station":   raw["stop_name"].fillna("Unknown"),
        "line":      raw.get("line", pd.Series([""] * len(raw))).fillna(""),
        "routes":    raw.get("daytime_routes", pd.Series([""] * len(raw))).fillna(""),
        "borough":   raw.get("borough", pd.Series([""] * len(raw))).fillna(""),
        "structure": raw.get("structure", pd.Series([""] * len(raw))).fillna("").str.lower(),
        "lat":       pd.to_numeric(raw["gtfs_latitude"],  errors="coerce"),
        "lon":       pd.to_numeric(raw["gtfs_longitude"], errors="coerce"),
    })
    return df.dropna(subset=["lat", "lon"])


# ---------------------------------------------------------------------------
# Compute per-station risk
# ---------------------------------------------------------------------------
def compute_station_risk(
    stations: pd.DataFrame,
    line_daily: pd.DataFrame | None,
) -> pd.DataFrame:

    # Data-driven uplift from real MTA incident data
    rain_uplift: dict[str, float] = {}
    heat_uplift: dict[str, float] = {}
    if line_daily is not None and "heavy_rain" in line_daily.columns:
        for ln, grp in line_daily.groupby("line"):
            norm = grp.loc[(grp["heavy_rain"]==0)&(grp["extreme_heat"]==0),
                           "total_incidents"].mean()
            rain = grp.loc[grp["heavy_rain"]==1,   "total_incidents"].mean()
            heat = grp.loc[grp["extreme_heat"]==1, "total_incidents"].mean()
            rain_uplift[ln] = max(0, (rain / norm - 1)) if norm > 0 else 0
            heat_uplift[ln] = max(0, (heat / norm - 1)) if norm > 0 else 0

    records = []
    for _, row in stations.iterrows():
        # Parse which lines this station serves (use daytime_routes if available)
        routes_str = row["routes"] if row["routes"] else row["line"]
        lines = [t.strip() for t in re.split(r"[\s,]+", routes_str.upper()) if t.strip()]
        if not lines:
            continue

        primary = lines[0]

        # Geographic base scores averaged across serving lines
        flood_geo = np.mean([GEO_FLOOD.get(l, 5.0) for l in lines])
        heat_geo  = np.mean([GEO_HEAT.get(l,  5.0) for l in lines])
        vuln_base = np.mean([GEO_VULN.get(l,  5.0) for l in lines])

        # Structure bonus to vulnerability
        struct_bonus = STRUCT_VULN.get(row["structure"], 0.0)
        vuln = min(10, vuln_base + struct_bonus)

        # Data-driven uplift (max +2 pts)
        rain_bonus = min(2.0, np.mean([rain_uplift.get(l, 0) for l in lines]) * 4)
        heat_bonus = min(2.0, np.mean([heat_uplift.get(l, 0) for l in lines]) * 4)

        flood_score = min(10, flood_geo + rain_bonus)
        heat_score  = min(10, heat_geo  + heat_bonus)
        ridership   = np.mean([LINE_RIDERSHIP.get(l, 50) for l in lines])
        econ_score  = round((ridership / MAX_RIDERSHIP) * 10, 2)

        composite = round(0.40*flood_score + 0.30*heat_score + 0.20*vuln + 0.10*econ_score, 2)

        records.append({
            "station":           row["station"],
            "lat":               row["lat"],
            "lon":               row["lon"],
            "borough":           row["borough"],
            "structure":         row["structure"],
            "lines":             " ".join(lines),
            "primary_line":      primary,
            "line_color":        LINE_COLORS.get(primary, "#888888"),
            "flood_risk":        round(flood_score, 2),
            "heat_risk":         round(heat_score, 2),
            "vulnerability":     round(vuln, 2),
            "economic_exposure": econ_score,
            "composite_risk":    composite,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Aggregate to line level
# ---------------------------------------------------------------------------
def compute_line_risk(station_risk: pd.DataFrame) -> pd.DataFrame:
    records = []
    for line in GEO_FLOOD.keys():
        mask = station_risk["lines"].str.contains(
            r"(?<![A-Z])" + re.escape(line) + r"(?![A-Z0-9])", regex=True, na=False
        )
        sub = station_risk[mask]

        if sub.empty:
            flood = GEO_FLOOD.get(line, 5.0)
            heat  = GEO_HEAT.get(line, 5.0)
            vuln  = GEO_VULN.get(line, 5.0)
            econ  = (LINE_RIDERSHIP.get(line, 50) / MAX_RIDERSHIP) * 10
        else:
            flood = sub["flood_risk"].mean()
            heat  = sub["heat_risk"].mean()
            vuln  = sub["vulnerability"].mean()
            econ  = sub["economic_exposure"].mean()

        composite = round(0.40*flood + 0.30*heat + 0.20*vuln + 0.10*econ, 2)
        risk_label = (
            "Critical" if composite >= 8 else
            "High"     if composite >= 6 else
            "Medium"   if composite >= 4 else "Low"
        )

        records.append({
            "line":             line,
            "line_color":       LINE_COLORS.get(line, "#888888"),
            "flood_risk":       round(flood, 2),
            "heat_risk":        round(heat, 2),
            "vulnerability":    round(vuln, 2),
            "economic_exposure":round(econ, 2),
            "composite_risk":   composite,
            "risk_label":       risk_label,
            "ridership_k":      LINE_RIDERSHIP.get(line, 0),
        })

    return pd.DataFrame(records).sort_values("composite_risk", ascending=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Step 5 — Station & Line Risk Scores")
    print("=" * 60)

    line_path = BASE / "data" / "analysis_daily.csv"
    line_daily = pd.read_csv(line_path, parse_dates=["date"]) if line_path.exists() else None

    stations     = fetch_stations()
    station_risk = compute_station_risk(stations, line_daily)
    line_risk    = compute_line_risk(station_risk)

    out_s = BASE / "data" / "station_risk.csv"
    out_l = BASE / "data" / "line_risk.csv"
    station_risk.to_csv(out_s, index=False)
    line_risk.to_csv(out_l, index=False)

    print(f"\n  Stations scored : {len(station_risk)}")
    print(f"  Lines scored    : {len(line_risk)}")
    print(f"\n  Top 5 highest-risk lines:")
    print(line_risk[["line","composite_risk","risk_label","flood_risk","heat_risk"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
