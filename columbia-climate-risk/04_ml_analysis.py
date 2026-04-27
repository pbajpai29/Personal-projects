"""
04_ml_analysis.py
-----------------
ML analysis of weather → MTA delay correlations and economic loss estimation.

Steps
-----
1. Correlation analysis   — Pearson + Spearman between weather vars and delay metrics
2. Linear regression      — baseline interpretable model
3. Random Forest          — captures nonlinear weather × service interactions
4. SHAP values            — feature importance for the Random Forest
5. Economic loss model    — $ cost of weather-caused MTA disruptions

Economic loss methodology
-------------------------
  Loss ($/day) = Riders_affected × Value_of_Time ($/hr) × Excess_Delay (hr)

  - Riders affected = disrupted_trains × avg_load_factor × p(affected)
  - Value of time   = $18.50/hr  (NYC DOT transit VOT, 2023 update)
  - Excess delay    = predicted_delay(weather day) − predicted_delay(normal day)

  Sources:
    - NYC DOT Measuring the Street (2023): transit VOT = $18.50/hr
    - MTA Blue Book 2023: avg load factor 85–110 pax/train at peak
    - Cambridge Systematics study: p(affected) ≈ 0.35 on incident day

Outputs (all saved to data/)
-------------------------------
  ml_results.json          — model metrics, top features, loss estimates
  correlation_matrix.csv   — full Pearson correlation matrix
  feature_importance.csv   — RF feature importances + SHAP means
  economic_loss.csv        — per-day economic loss estimates
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("  Note: shap not installed — skipping SHAP values. Run: pip install shap")

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Economic constants
# ---------------------------------------------------------------------------
VALUE_OF_TIME_PER_HR   = 18.50   # NYC DOT 2023 transit VOT ($/hr)
AVG_LOAD_FACTOR        = 95      # passengers per train (MTA avg)
P_AFFECTED             = 0.35    # probability a rider is affected by an incident


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
WEATHER_FEATURES = [
    "precip_mm", "precip_in", "tmax_c", "tmax_f",
    "heavy_rain", "extreme_rain", "extreme_heat", "heat_emergency",
    "heat_wave_day", "weather_stress",
    "lag1_precip_mm", "lag2_precip_mm", "roll3_precip_mm", "roll7_precip_mm",
    "prev_extreme_heat", "prev_heavy_rain", "heat_index_approx",
    "windspeed_kmh", "precip_hours",
]
CALENDAR_FEATURES = ["month", "dow", "year"]
TARGET_INCIDENTS  = "total_incidents"
TARGET_DELAY      = "avg_delay_min"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    line_path = BASE / "data" / "analysis_daily.csv"
    sys_path  = BASE / "data" / "analysis_systemwide.csv"

    if not sys_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run 03_merge_process.py first."
        )

    line_df = pd.read_csv(line_path, parse_dates=["date"])
    sys_df  = pd.read_csv(sys_path,  parse_dates=["date"])
    return line_df, sys_df


# ---------------------------------------------------------------------------
# 1. Correlation analysis
# ---------------------------------------------------------------------------
def run_correlations(sys_df: pd.DataFrame) -> dict:
    print("  1. Correlation analysis…")

    avail_features = [f for f in WEATHER_FEATURES + CALENDAR_FEATURES if f in sys_df.columns]
    targets = [t for t in [TARGET_INCIDENTS, TARGET_DELAY] if t in sys_df.columns]

    sub = sys_df[avail_features + targets].dropna()

    results = {}
    for target in targets:
        pearson, spearman = {}, {}
        for feat in avail_features:
            if feat not in sub.columns:
                continue
            p_r, p_p   = stats.pearsonr(sub[feat],  sub[target])
            sp_r, sp_p = stats.spearmanr(sub[feat], sub[target])
            pearson[feat]  = {"r": round(p_r, 4),  "p": round(p_p, 6)}
            spearman[feat] = {"r": round(sp_r, 4), "p": round(sp_p, 6)}
        results[target] = {"pearson": pearson, "spearman": spearman}

    # Save full Pearson matrix
    corr_cols = avail_features + targets
    corr_matrix = sub[corr_cols].corr(method="pearson")
    corr_matrix.to_csv(BASE / "data" / "correlation_matrix.csv")

    # Print top correlates with total_incidents
    if TARGET_INCIDENTS in results:
        sorted_p = sorted(
            results[TARGET_INCIDENTS]["pearson"].items(),
            key=lambda x: abs(x[1]["r"]),
            reverse=True,
        )
        print(f"\n  Top correlates with {TARGET_INCIDENTS} (Pearson r):")
        for feat, vals in sorted_p[:8]:
            sig = "**" if vals["p"] < 0.01 else ("*" if vals["p"] < 0.05 else "")
            print(f"    {feat:<28} r={vals['r']:>7.4f}  p={vals['p']:.4f} {sig}")

    return results


# ---------------------------------------------------------------------------
# 2 & 3. Regression models
# ---------------------------------------------------------------------------
def run_models(sys_df: pd.DataFrame) -> dict:
    print("\n  2–3. Regression models…")

    avail = [f for f in WEATHER_FEATURES + CALENDAR_FEATURES if f in sys_df.columns]
    sub   = sys_df[avail + [TARGET_INCIDENTS, TARGET_DELAY]].dropna()
    sub   = sub.sort_values("date") if "date" in sub.columns else sub

    X = sub[avail].values
    tscv = TimeSeriesSplit(n_splits=5)

    model_results = {}

    for target_name in [TARGET_INCIDENTS, TARGET_DELAY]:
        y = sub[target_name].values
        print(f"\n    Target: {target_name}")

        # --- Linear (Ridge) ---
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X)
        ridge   = Ridge(alpha=1.0)
        cv_r2   = cross_val_score(ridge, X_sc, y, cv=tscv, scoring="r2")
        cv_mae  = cross_val_score(ridge, X_sc, y, cv=tscv, scoring="neg_mean_absolute_error")
        ridge.fit(X_sc, y)
        y_pred_ridge = ridge.predict(X_sc)

        print(f"    Ridge R²={np.mean(cv_r2):.3f} (±{np.std(cv_r2):.3f})  "
              f"MAE={-np.mean(cv_mae):.2f}")

        # --- Random Forest ---
        rf = RandomForestRegressor(n_estimators=300, max_depth=10,
                                   min_samples_leaf=5, n_jobs=-1, random_state=42)
        cv_r2_rf  = cross_val_score(rf, X, y, cv=tscv, scoring="r2")
        cv_mae_rf = cross_val_score(rf, X, y, cv=tscv, scoring="neg_mean_absolute_error")
        rf.fit(X, y)
        y_pred_rf = rf.predict(X)

        print(f"    RF    R²={np.mean(cv_r2_rf):.3f} (±{np.std(cv_r2_rf):.3f})  "
              f"MAE={-np.mean(cv_mae_rf):.2f}")

        model_results[target_name] = {
            "ridge": {
                "cv_r2_mean":  round(float(np.mean(cv_r2)), 4),
                "cv_r2_std":   round(float(np.std(cv_r2)), 4),
                "cv_mae_mean": round(float(-np.mean(cv_mae)), 4),
                "coefs": {f: round(float(c), 6) for f, c in zip(avail, ridge.coef_)},
            },
            "random_forest": {
                "cv_r2_mean":  round(float(np.mean(cv_r2_rf)), 4),
                "cv_r2_std":   round(float(np.std(cv_r2_rf)), 4),
                "cv_mae_mean": round(float(-np.mean(cv_mae_rf)), 4),
                "feature_importances": {
                    f: round(float(imp), 6)
                    for f, imp in sorted(
                        zip(avail, rf.feature_importances_),
                        key=lambda x: -x[1],
                    )[:15]
                },
            },
            "predictions": {
                "ridge": y_pred_ridge.tolist(),
                "rf":    y_pred_rf.tolist(),
                "actual": y.tolist(),
            },
        }

        # --- SHAP ---
        if HAS_SHAP and target_name == TARGET_INCIDENTS:
            print("    Computing SHAP values…")
            explainer   = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X)
            shap_means  = np.abs(shap_values).mean(axis=0)
            model_results[target_name]["shap_mean_abs"] = {
                f: round(float(v), 6) for f, v in zip(avail, shap_means)
            }

    # Save feature importance CSV
    if TARGET_INCIDENTS in model_results:
        rf_imp = model_results[TARGET_INCIDENTS]["random_forest"]["feature_importances"]
        fi_df  = pd.DataFrame(list(rf_imp.items()), columns=["feature", "importance"])
        if "shap_mean_abs" in model_results.get(TARGET_INCIDENTS, {}):
            shap_dict = model_results[TARGET_INCIDENTS]["shap_mean_abs"]
            fi_df["shap_mean_abs"] = fi_df["feature"].map(shap_dict)
        fi_df.to_csv(BASE / "data" / "feature_importance.csv", index=False)

    return model_results


# ---------------------------------------------------------------------------
# 4. Economic loss model
# ---------------------------------------------------------------------------
def estimate_economic_loss(sys_df: pd.DataFrame, model_results: dict) -> pd.DataFrame:
    """
    Estimate daily $ economic loss attributable to extreme weather events.

    Method:
      excess_incidents = actual − predicted_baseline(no-weather-day)
      riders_affected  = excess_incidents × AVG_LOAD_FACTOR × P_AFFECTED
      excess_delay_hr  = avg_delay_min / 60  (on weather days vs baseline)
      loss_usd         = riders_affected × VALUE_OF_TIME_PER_HR × excess_delay_hr
    """
    print("\n  4. Economic loss estimation…")

    # Average incidents on a "normal" day (no extreme weather, weekday)
    normal_mask = (
        (sys_df["heavy_rain"] == 0) &
        (sys_df["extreme_heat"] == 0) &
        (sys_df["dow"].between(0, 4))   # weekday
    )
    baseline_incidents = sys_df.loc[normal_mask, "total_incidents"].mean()
    baseline_delay_min = sys_df.loc[normal_mask, "avg_delay_min"].mean()

    records = []
    for _, row in sys_df.iterrows():
        is_weather_day = bool(row.get("heavy_rain", 0)) or bool(row.get("extreme_heat", 0))

        excess_incidents = max(0, row["total_incidents"] - baseline_incidents)
        riders_affected  = excess_incidents * AVG_LOAD_FACTOR * P_AFFECTED
        excess_delay_min = max(0, row["avg_delay_min"] - baseline_delay_min)
        excess_delay_hr  = excess_delay_min / 60

        loss_usd = riders_affected * VALUE_OF_TIME_PER_HR * excess_delay_hr if is_weather_day else 0

        event = "normal"
        if row.get("extreme_rain", 0):   event = "extreme_rain"
        elif row.get("heavy_rain", 0):   event = "heavy_rain"
        elif row.get("heat_emergency", 0): event = "heat_emergency"
        elif row.get("extreme_heat", 0): event = "extreme_heat"

        records.append({
            "date":               row["date"],
            "event_type":         event,
            "total_incidents":    row["total_incidents"],
            "excess_incidents":   round(excess_incidents, 1),
            "riders_affected":    round(riders_affected),
            "excess_delay_min":   round(excess_delay_min, 1),
            "economic_loss_usd":  round(loss_usd, 2),
            "heavy_rain":         int(row.get("heavy_rain", 0)),
            "extreme_heat":       int(row.get("extreme_heat", 0)),
            "precip_mm":          row.get("precip_mm", 0),
            "tmax_f":             row.get("tmax_f", np.nan),
        })

    loss_df = pd.DataFrame(records)
    loss_df.to_csv(BASE / "data" / "economic_loss.csv", index=False)

    # Summary
    total_loss = loss_df["economic_loss_usd"].sum()
    rain_loss  = loss_df.loc[loss_df["heavy_rain"] == 1, "economic_loss_usd"].sum()
    heat_loss  = loss_df.loc[loss_df["extreme_heat"] == 1, "economic_loss_usd"].sum()

    print(f"\n  ── Economic Loss Summary (2020–2024) ──────────────────")
    print(f"  Total estimated loss   : ${total_loss:>12,.0f}")
    print(f"  Due to heavy rain      : ${rain_loss:>12,.0f}  ({rain_loss/total_loss:.1%})")
    print(f"  Due to extreme heat    : ${heat_loss:>12,.0f}  ({heat_loss/total_loss:.1%})")
    print(f"  Avg loss per rain day  : ${loss_df.loc[loss_df['heavy_rain']==1,'economic_loss_usd'].mean():>12,.0f}")
    print(f"  Avg loss per heat day  : ${loss_df.loc[loss_df['extreme_heat']==1,'economic_loss_usd'].mean():>12,.0f}")
    print(f"  Worst single day       : ${loss_df['economic_loss_usd'].max():>12,.0f}")
    print(f"    → {loss_df.loc[loss_df['economic_loss_usd'].idxmax(), 'date']}")

    return loss_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> dict:
    print("=" * 60)
    print("  Step 4 — ML Analysis & Economic Loss Estimation")
    print("=" * 60)

    line_df, sys_df = load_data()

    correlations = run_correlations(sys_df)
    model_results = run_models(sys_df)
    loss_df = estimate_economic_loss(sys_df, model_results)

    # Compile summary JSON
    summary = {
        "correlations": correlations,
        "models":       model_results,
        "economic_loss": {
            "total_usd":         round(loss_df["economic_loss_usd"].sum(), 2),
            "rain_usd":          round(loss_df.loc[loss_df["heavy_rain"]==1, "economic_loss_usd"].sum(), 2),
            "heat_usd":          round(loss_df.loc[loss_df["extreme_heat"]==1, "economic_loss_usd"].sum(), 2),
            "avg_per_rain_day":  round(loss_df.loc[loss_df["heavy_rain"]==1, "economic_loss_usd"].mean(), 2),
            "avg_per_heat_day":  round(loss_df.loc[loss_df["extreme_heat"]==1, "economic_loss_usd"].mean(), 2),
            "methodology": {
                "value_of_time_per_hr": VALUE_OF_TIME_PER_HR,
                "avg_load_factor":      AVG_LOAD_FACTOR,
                "p_affected":           P_AFFECTED,
                "source": "NYC DOT 2023 Transit VOT; MTA Blue Book 2023",
            },
        },
    }

    out = BASE / "data" / "ml_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Results saved → data/ml_results.json")
    print(f"                  data/correlation_matrix.csv")
    print(f"                  data/feature_importance.csv")
    print(f"                  data/economic_loss.csv")
    print()

    return summary


if __name__ == "__main__":
    main()
