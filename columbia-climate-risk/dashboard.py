"""
dashboard.py — Columbia Climate Risk Dashboard
------------------------------------------------
Streamlit web app: NYC extreme weather × MTA subway disruptions + economic cost.

Run locally:
    streamlit run dashboard.py

Deploy on Streamlit Community Cloud:
    Push this folder to GitHub → connect at https://share.streamlit.io

If data files are missing the app will run the pipeline automatically.
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

BASE     = Path(__file__).parent
DATA_DIR = BASE / "data"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NYC Climate Risk × MTA",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-header {
      background: #061717; color: white;
      padding: 28px 36px 22px; border-radius: 8px; margin-bottom: 24px;
  }
  .main-header h1 { margin: 0 0 6px; font-size: 28px; font-weight: 500; }
  .main-header p  { margin: 0; color: rgba(255,255,255,0.55); font-size: 13px; }
  .kpi-card {
      background: white; border: 1px solid #E5E7EB;
      border-radius: 8px; padding: 18px 20px; text-align: center;
  }
  .kpi-label { font-size: 11px; color: #6B7280; text-transform: uppercase;
               letter-spacing: 0.08em; margin-bottom: 6px; }
  .kpi-value { font-size: 28px; font-weight: 600; color: #111827; }
  .kpi-sub   { font-size: 12px; color: #9CA3AF; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

RAIN_COLOR  = "#1D4ED8"
HEAT_COLOR  = "#DC2626"
DELAY_COLOR = "#15803D"
ACCENT      = "#16A34A"
NEUTRAL     = "#6B7280"

# ---------------------------------------------------------------------------
# Auto-run pipeline if data is missing
# ---------------------------------------------------------------------------
def run_pipeline_if_needed():
    needed = [
        DATA_DIR / "nyc_weather_daily.csv",
        DATA_DIR / "mta_delays_daily.csv",
        DATA_DIR / "analysis_systemwide.csv",
        DATA_DIR / "ml_results.json",
    ]
    missing = [f for f in needed if not f.exists()]
    if not missing:
        return

    st.info("First run — fetching data and running pipeline. This takes ~2 minutes…")
    steps = [
        ("Downloading NYC weather (Open-Meteo)…",  ["python", str(BASE / "01_download_weather.py")]),
        ("Downloading real MTA incident data…",     ["python", str(BASE / "02_download_mta.py")]),
        ("Merging datasets…",                       ["python", str(BASE / "03_merge_process.py")]),
        ("Running ML analysis…",                    ["python", str(BASE / "04_ml_analysis.py")]),
        ("Scoring station & line risk…",            ["python", str(BASE / "05_station_risk.py")]),
    ]
    progress = st.progress(0)
    for i, (msg, cmd) in enumerate(steps):
        with st.spinner(msg):
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE))
            if result.returncode != 0:
                st.error(f"Pipeline step failed:\n{result.stderr[-2000:]}")
                st.stop()
        progress.progress((i + 1) / len(steps))

    st.success("Pipeline complete! Loading dashboard…")
    st.rerun()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_weather():
    p = DATA_DIR / "nyc_weather_daily.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None

@st.cache_data
def load_mta():
    p = DATA_DIR / "mta_delays_daily.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None

@st.cache_data
def load_systemwide():
    p = DATA_DIR / "analysis_systemwide.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None

@st.cache_data
def load_line_daily():
    p = DATA_DIR / "analysis_daily.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None

@st.cache_data
def load_economic_loss():
    p = DATA_DIR / "economic_loss.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None

@st.cache_data
def load_feature_importance():
    p = DATA_DIR / "feature_importance.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_ml_results():
    p = DATA_DIR / "ml_results.json"
    return json.load(open(p)) if p.exists() else None

@st.cache_data
def load_corr_matrix():
    p = DATA_DIR / "correlation_matrix.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None

@st.cache_data
def load_station_risk():
    p = DATA_DIR / "station_risk.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_line_risk():
    p = DATA_DIR / "line_risk.csv"
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
def sidebar_filters(sys_df):
    st.sidebar.markdown("### Filters")
    min_d = sys_df["date"].min().date()
    max_d = sys_df["date"].max().date()

    dr = st.sidebar.date_input("Date range", value=(min_d, max_d),
                                min_value=min_d, max_value=max_d)
    d_start, d_end = (dr[0], dr[1]) if len(dr) == 2 else (min_d, max_d)

    mta_df = load_mta()
    lines_all = sorted(mta_df["line"].unique()) if mta_df is not None else []
    selected = st.sidebar.multiselect("Subway lines", lines_all, default=lines_all)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Extreme event thresholds**")
    rain_t = st.sidebar.slider("Heavy rain (mm)", 10, 75, 25, 5)
    heat_t = st.sidebar.slider("Extreme heat (°F)", 85, 100, 90, 1)
    return d_start, d_end, selected, rain_t, heat_t


# ---------------------------------------------------------------------------
# KPI card helper
# ---------------------------------------------------------------------------
def kpi(col, label, value, sub):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------
def tab_overview(sys_df, weather, d_start, d_end, rain_t, heat_t):
    mask = (sys_df["date"].dt.date >= d_start) & (sys_df["date"].dt.date <= d_end)
    df   = sys_df[mask].copy()
    df["_rain"] = (df["precip_mm"] >= rain_t).astype(int)
    df["_heat"] = (df["tmax_f"]    >= heat_t).astype(int)

    n      = len(df)
    r_days = int(df["_rain"].sum())
    h_days = int(df["_heat"].sum())
    norm_i = df.loc[(df["_rain"]==0)&(df["_heat"]==0), "total_incidents"].mean()
    rain_i = df.loc[df["_rain"]==1, "total_incidents"].mean()
    heat_i = df.loc[df["_heat"]==1, "total_incidents"].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1, "Days analysed",       f"{n:,}",          f"{d_start} → {d_end}")
    kpi(c2, "Heavy rain days",     f"{r_days}",        f"{r_days/n:.1%} of period")
    kpi(c3, "Extreme heat days",   f"{h_days}",        f"{h_days/n:.1%} of period")
    kpi(c4, "Avg daily incidents", f"{norm_i:.0f}",    "normal days")
    kpi(c5, "Rain uplift",         f"+{(rain_i/norm_i-1)*100:.0f}%", "incidents on rain days")
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Bar: incidents by weather condition
        fig = go.Figure()
        for label, val, color in [
            ("Normal",       norm_i, NEUTRAL),
            ("Heavy Rain",   rain_i, RAIN_COLOR),
            ("Extreme Heat", heat_i, HEAT_COLOR),
        ]:
            fig.add_trace(go.Bar(x=[label], y=[val], marker_color=color, name=label))
        fig.update_layout(title="Avg Daily Incidents by Weather Condition",
                          yaxis_title="Incidents/day", showlegend=False,
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Monthly precip vs incidents
        df["ym"] = df["date"].dt.to_period("M").dt.to_timestamp()
        monthly  = df.groupby("ym").agg(precip=("precip_mm","sum"),
                                         incidents=("total_incidents","sum")).reset_index()
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=monthly["ym"], y=monthly["precip"],
                              name="Precip (mm)", marker_color=RAIN_COLOR, opacity=0.55),
                       secondary_y=False)
        fig2.add_trace(go.Scatter(x=monthly["ym"], y=monthly["incidents"],
                                   name="Incidents", line=dict(color=DELAY_COLOR, width=2)),
                       secondary_y=True)
        fig2.update_layout(title="Monthly Precipitation vs Incidents",
                           plot_bgcolor="white", paper_bgcolor="white",
                           legend=dict(orientation="h", y=-0.2))
        fig2.update_yaxes(title_text="Precip (mm)", secondary_y=False)
        fig2.update_yaxes(title_text="Incidents",   secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)

    # Weather timeline
    st.markdown("##### Daily Weather — NYC 2020–2024")
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          subplot_titles=("Precipitation (mm)", "Max Temperature (°F)"),
                          vertical_spacing=0.08)
    fig3.add_trace(go.Bar(x=df["date"], y=df["precip_mm"],
                           marker_color=np.where(df["precip_mm"]>=rain_t, RAIN_COLOR,"#93C5FD"),
                           name="Precip"), row=1, col=1)
    fig3.add_hline(y=rain_t, line_dash="dash", line_color=RAIN_COLOR,
                   annotation_text=f"{rain_t} mm threshold", row=1, col=1)
    fig3.add_trace(go.Scatter(x=df["date"], y=df["tmax_f"],
                               line=dict(color=HEAT_COLOR, width=1), name="Max °F"),
                   row=2, col=1)
    fig3.add_hline(y=heat_t, line_dash="dash", line_color=HEAT_COLOR,
                   annotation_text=f"{heat_t}°F threshold", row=2, col=1)
    fig3.update_layout(height=400, showlegend=False,
                       plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — MTA Performance
# ---------------------------------------------------------------------------
def tab_mta(line_df, d_start, d_end, selected_lines):
    mask = ((line_df["date"].dt.date >= d_start) &
            (line_df["date"].dt.date <= d_end) &
            (line_df["line"].isin(selected_lines)))
    df = line_df[mask].copy()

    c1, c2 = st.columns(2)
    with c1:
        # Heatmap: incidents by line × year — using go.Heatmap (no xarray)
        pivot = (df.groupby(["line", df["date"].dt.year])["total_incidents"]
                   .sum().reset_index()
                   .pivot(index="line", columns="date", values="total_incidents")
                   .fillna(0))
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="Blues",
            colorbar=dict(title="Incidents"),
        ))
        fig.update_layout(title="Total Incidents by Line × Year",
                          xaxis_title="Year", yaxis_title="Line",
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        avg_delay = (df.groupby("line")["avg_delay_min"].mean()
                       .reset_index().sort_values("avg_delay_min", ascending=True))
        fig2 = go.Figure(go.Bar(
            x=avg_delay["avg_delay_min"], y=avg_delay["line"],
            orientation="h",
            marker=dict(color=avg_delay["avg_delay_min"],
                        colorscale="Reds", showscale=False),
        ))
        fig2.update_layout(title="Avg Delay (min) by Line",
                           xaxis_title="Avg delay (min)", yaxis_title="Line",
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    # Uplift bar
    st.markdown("##### Incident Uplift on Heavy Rain Days vs Normal Days")
    normal_g = df[df["heavy_rain"]==0].groupby("line")["total_incidents"].mean()
    rain_g   = df[df["heavy_rain"]==1].groupby("line")["total_incidents"].mean()
    uplift   = ((rain_g - normal_g) / normal_g * 100).dropna().reset_index()
    uplift.columns = ["line", "pct"]
    uplift   = uplift.sort_values("pct", ascending=False)

    colors = [RAIN_COLOR if v >= 0 else NEUTRAL for v in uplift["pct"]]
    fig3 = go.Figure(go.Bar(x=uplift["line"], y=uplift["pct"], marker_color=colors))
    fig3.add_hline(y=0, line_color="#374151")
    fig3.update_layout(title="% More Incidents on Heavy Rain Days",
                       yaxis_title="Uplift (%)", xaxis_title="Line",
                       plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 — Correlations
# ---------------------------------------------------------------------------
def tab_correlations(sys_df, corr_matrix, d_start, d_end):
    mask = (sys_df["date"].dt.date >= d_start) & (sys_df["date"].dt.date <= d_end)
    df   = sys_df[mask].copy()

    c1, c2 = st.columns([3, 2])
    with c1:
        if corr_matrix is not None:
            wx_cols  = [c for c in ["precip_mm","tmax_f","heavy_rain","extreme_heat",
                                     "roll7_precip_mm","lag1_precip_mm","weather_stress"]
                        if c in corr_matrix.columns]
            tgt_cols = [c for c in ["total_incidents","avg_delay_min"]
                        if c in corr_matrix.columns]
            sub = corr_matrix.loc[
                [r for r in wx_cols+tgt_cols if r in corr_matrix.index],
                [c for c in wx_cols+tgt_cols if c in corr_matrix.columns],
            ].round(3)

            # go.Heatmap — no xarray dependency
            fig = go.Figure(go.Heatmap(
                z=sub.values,
                x=sub.columns.tolist(),
                y=sub.index.tolist(),
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                text=sub.values.round(2),
                texttemplate="%{text}",
                colorbar=dict(title="r"),
            ))
            fig.update_layout(title="Pearson Correlation Matrix",
                              paper_bgcolor="white",
                              xaxis=dict(tickangle=30))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run 04_ml_analysis.py to generate the correlation matrix.")

    with c2:
        wx_opt  = st.selectbox("Weather variable",
                               [c for c in ["precip_mm","tmax_f","roll7_precip_mm","lag1_precip_mm"]
                                if c in df.columns])
        tgt_opt = st.selectbox("MTA metric",
                               [c for c in ["total_incidents","avg_delay_min"] if c in df.columns])
        sub2 = df[[wx_opt, tgt_opt]].dropna()
        coefs = np.polyfit(sub2[wx_opt], sub2[tgt_opt], 1)
        x_line = np.linspace(sub2[wx_opt].min(), sub2[wx_opt].max(), 50)

        r_val, p_val = pearsonr(sub2[wx_opt], sub2[tgt_opt])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sub2[wx_opt], y=sub2[tgt_opt], mode="markers",
                                   marker=dict(color=RAIN_COLOR, opacity=0.35, size=5),
                                   name="Daily obs"))
        fig2.add_trace(go.Scatter(x=x_line, y=np.polyval(coefs, x_line),
                                   mode="lines", line=dict(color="#1F2937", width=2, dash="dash"),
                                   name="OLS trend"))
        fig2.update_layout(
            title=f"{wx_opt} vs {tgt_opt}<br><sup>r = {r_val:.3f}  p = {p_val:.4f}</sup>",
            xaxis_title=wx_opt, yaxis_title=tgt_opt,
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Seasonal grouped bar
    st.markdown("##### Mean Incidents by Season × Weather")
    season_order = ["Winter","Spring","Summer","Autumn"]
    rows = []
    for s in season_order:
        sd = df[df["season"]==s]
        rows.append({
            "season": s,
            "Normal":       sd.loc[(sd["heavy_rain"]==0)&(sd["extreme_heat"]==0), "total_incidents"].mean(),
            "Heavy Rain":   sd.loc[sd["heavy_rain"]==1,   "total_incidents"].mean(),
            "Extreme Heat": sd.loc[sd["extreme_heat"]==1, "total_incidents"].mean(),
        })
    seas = pd.DataFrame(rows).fillna(0)
    fig3 = go.Figure()
    for label, color in [("Normal",NEUTRAL),("Heavy Rain",RAIN_COLOR),("Extreme Heat",HEAT_COLOR)]:
        fig3.add_trace(go.Bar(x=seas["season"], y=seas[label], name=label, marker_color=color))
    fig3.update_layout(barmode="group", title="Avg Daily Incidents: Season × Weather",
                       yaxis_title="Avg incidents", plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — ML Model
# ---------------------------------------------------------------------------
def tab_ml(ml_results, fi_df, sys_df):
    if ml_results is None:
        st.info("Run 04_ml_analysis.py to generate ML results.")
        return

    target = "total_incidents"
    models = ml_results.get("models", {})

    if target in models:
        ridge_m = models[target]["ridge"]
        rf_m    = models[target]["random_forest"]
        c1,c2,c3,c4 = st.columns(4)
        kpi(c1, "Ridge R²",          f"{ridge_m['cv_r2_mean']:.3f}",  "5-fold time-series CV")
        kpi(c2, "Ridge MAE",         f"{ridge_m['cv_mae_mean']:.1f}", "incidents/day")
        kpi(c3, "Random Forest R²",  f"{rf_m['cv_r2_mean']:.3f}",    "5-fold time-series CV")
        kpi(c4, "RF MAE",            f"{rf_m['cv_mae_mean']:.1f}",   "incidents/day")
        st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if fi_df is not None and not fi_df.empty:
            col = "importance"
            top = fi_df.nlargest(12, col).sort_values(col)
            fig = go.Figure(go.Bar(
                x=top[col], y=top["feature"], orientation="h",
                marker=dict(color=top[col], colorscale=[[0,"#D1FAE5"],[1,ACCENT]],
                            showscale=False),
            ))
            fig.update_layout(title="Feature Importance (Random Forest)",
                              xaxis_title="Importance", yaxis_title="",
                              plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if target in models:
            preds = models[target].get("predictions", {})
            if preds:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=preds["actual"], y=preds["rf"],
                                          mode="markers",
                                          marker=dict(color=RAIN_COLOR, opacity=0.4, size=5),
                                          name="RF prediction"))
                lim = max(max(preds["actual"]), max(preds["rf"]))
                fig2.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                               line=dict(dash="dash", color="#9CA3AF"))
                fig2.update_layout(title="Predicted vs Actual Incidents (RF)",
                                   xaxis_title="Actual", yaxis_title="Predicted",
                                   plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig2, use_container_width=True)

    # Ridge coefficients
    if target in models:
        coefs = models[target]["ridge"].get("coefs", {})
        wx_keys = ["precip_mm","tmax_f","heavy_rain","extreme_heat",
                   "roll7_precip_mm","weather_stress","heat_wave_day"]
        wx_coefs = {k: v for k, v in coefs.items() if k in wx_keys}
        if wx_coefs:
            cdf = pd.DataFrame(list(wx_coefs.items()), columns=["feature","coef"])
            cdf = cdf.sort_values("coef")
            fig3 = go.Figure(go.Bar(
                x=cdf["coef"], y=cdf["feature"], orientation="h",
                marker_color=[RAIN_COLOR if v>0 else NEUTRAL for v in cdf["coef"]],
            ))
            fig3.add_vline(x=0, line_color="#374151")
            fig3.update_layout(title="Ridge Coefficients — Weather Features (standardised)",
                               xaxis_title="Coefficient",
                               plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5 — Economic Impact
# ---------------------------------------------------------------------------
def tab_economic(loss_df, d_start, d_end):
    if loss_df is None:
        st.info("Run 04_ml_analysis.py to generate economic loss estimates.")
        return

    mask = (loss_df["date"].dt.date >= d_start) & (loss_df["date"].dt.date <= d_end)
    df   = loss_df[mask].copy()

    total     = df["economic_loss_usd"].sum()
    rain_loss = df.loc[df["heavy_rain"]==1,   "economic_loss_usd"].sum()
    heat_loss = df.loc[df["extreme_heat"]==1, "economic_loss_usd"].sum()
    avg_rain  = df.loc[df["heavy_rain"]==1,   "economic_loss_usd"].mean()
    avg_heat  = df.loc[df["extreme_heat"]==1, "economic_loss_usd"].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1, "Total estimated loss",  f"${total/1e6:.2f}M", "2020–2024")
    kpi(c2, "Rain-caused loss",      f"${rain_loss/1e6:.2f}M", f"{rain_loss/total:.0%} of total")
    kpi(c3, "Heat-caused loss",      f"${heat_loss/1e6:.2f}M", f"{heat_loss/total:.0%} of total")
    kpi(c4, "Avg loss / rain day",   f"${avg_rain:,.0f}",  "heavy rain (≥1 in)")
    kpi(c5, "Avg loss / heat day",   f"${avg_heat:,.0f}",  "extreme heat (≥90°F)")
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        annual = df.groupby(df["date"].dt.year)["economic_loss_usd"].sum().reset_index()
        annual.columns = ["year","loss"]
        fig = go.Figure(go.Bar(
            x=annual["year"], y=annual["loss"],
            marker=dict(color=annual["loss"], colorscale=[[0,"#FEF9C3"],[1,HEAT_COLOR]],
                        showscale=False),
            text=[f"${v/1e6:.2f}M" for v in annual["loss"]],
            textposition="outside",
        ))
        fig.update_layout(title="Annual Economic Loss ($)",
                          xaxis_title="Year", yaxis_title="Loss ($)",
                          plot_bgcolor="white", paper_bgcolor="white",
                          uniformtext_minsize=8)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        event_loss = (df[df["economic_loss_usd"]>0]
                      .groupby("event_type")["economic_loss_usd"].sum().reset_index())
        color_map = {"extreme_rain": RAIN_COLOR, "heavy_rain": "#60A5FA",
                     "heat_emergency": "#7F1D1D", "extreme_heat": HEAT_COLOR}
        fig2 = go.Figure(go.Pie(
            labels=event_loss["event_type"],
            values=event_loss["economic_loss_usd"],
            marker_colors=[color_map.get(e, NEUTRAL) for e in event_loss["event_type"]],
            hole=0.4,
        ))
        fig2.update_layout(title="Loss by Event Type", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    # Cumulative timeline
    dfs = df.sort_values("date")
    dfs["cumulative"] = dfs["economic_loss_usd"].cumsum()
    major = dfs[dfs["event_type"].isin(["extreme_rain","heat_emergency"])]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=dfs["date"], y=dfs["cumulative"],
                               fill="tozeroy", line=dict(color=HEAT_COLOR, width=2),
                               name="Cumulative loss"))
    fig3.add_trace(go.Scatter(x=major["date"], y=major["cumulative"],
                               mode="markers",
                               marker=dict(color=RAIN_COLOR, size=9, symbol="circle"),
                               name="Major event"))
    fig3.update_layout(title="Cumulative Economic Loss Over Time",
                       xaxis_title="Date", yaxis_title="Cumulative Loss ($)",
                       plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Methodology & assumptions"):
        st.markdown("""
**Formula:** `Loss = Excess Incidents × Avg Load Factor × P(Affected) × VOT × Excess Delay (hr)`

| Parameter | Value | Source |
|-----------|-------|--------|
| Value of Time (transit) | $18.50/hr | NYC DOT 2023 |
| Avg load factor | 95 pax/train | MTA Blue Book 2023 |
| P(rider affected per incident) | 35% | Cambridge Systematics / TCRP 86 |
| MTA data source | Real monthly totals, weather-downscaled | data.ny.gov g937-7k7c, 9zbp-wz3y |

**Limitations:** Conservative lower bound — excludes indirect costs (business disruption,
emergency services, infrastructure damage). Real losses are likely 2–5× higher when full
economic multipliers are applied.
        """)


# ---------------------------------------------------------------------------
# Tab 6 — Station Risk Map
# ---------------------------------------------------------------------------
def tab_station_risk(station_df, line_df):
    if station_df is None or line_df is None:
        st.info("Run 05_station_risk.py to generate station and line risk scores.")
        return

    dim_choice = st.radio(
        "Risk dimension",
        ["composite_risk", "flood_risk", "heat_risk", "vulnerability", "economic_exposure"],
        horizontal=True, index=0,
    )

    n_stations   = len(station_df)
    n_critical   = int((station_df["composite_risk"] >= 8).sum())
    n_high       = int(((station_df["composite_risk"] >= 6) & (station_df["composite_risk"] < 8)).sum())
    top_line     = line_df.iloc[0]
    avg_flood    = station_df["flood_risk"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Stations scored",    f"{n_stations}",            "MTA subway system")
    kpi(c2, "Critical stations",  f"{n_critical}",            "composite ≥ 8")
    kpi(c3, "High-risk stations", f"{n_high}",                "composite 6–8")
    kpi(c4, "Highest-risk line",  f"{top_line['line']}",      f"score {top_line['composite_risk']:.2f}")
    kpi(c5, "Avg flood risk",     f"{avg_flood:.2f}",         "system-wide (0–10)")
    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Map ----
    st.markdown(f"##### Station Map — colored by {dim_choice.replace('_',' ')}")
    fig = go.Figure(go.Scattermapbox(
        lat=station_df["lat"], lon=station_df["lon"],
        mode="markers",
        marker=dict(
            size=8,
            color=station_df[dim_choice],
            colorscale="RdYlGn_r",
            cmin=float(station_df[dim_choice].min()),
            cmax=float(station_df[dim_choice].max()),
            colorbar=dict(title=dim_choice.replace("_", " ")),
            opacity=0.85,
        ),
        text=[
            f"<b>{r['station']}</b><br>"
            f"Line: {r['primary_line']} ({r['borough']})<br>"
            f"Structure: {r['structure']}<br>"
            f"Composite: {r['composite_risk']:.2f}<br>"
            f"Flood: {r['flood_risk']:.2f}  Heat: {r['heat_risk']:.2f}<br>"
            f"Vuln: {r['vulnerability']:.2f}  Econ: {r['economic_exposure']:.2f}"
            for _, r in station_df.iterrows()
        ],
        hoverinfo="text",
    ))
    fig.update_layout(
        mapbox=dict(style="carto-positron",
                    center=dict(lat=40.72, lon=-73.95), zoom=10),
        height=560, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Line-level table + chart ----
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("##### Line Risk Ranking")
        lr = line_df.sort_values("composite_risk", ascending=True)
        fig2 = go.Figure(go.Bar(
            x=lr["composite_risk"], y=lr["line"], orientation="h",
            marker=dict(color=lr["line_color"]),
            text=[f"{v:.2f}" for v in lr["composite_risk"]],
            textposition="outside",
        ))
        fig2.update_layout(
            title="Composite Risk by Line (0–10)",
            xaxis_title="Composite risk", yaxis_title="",
            height=560, plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("##### Top 15 Highest-Risk Stations")
        top = station_df.nlargest(15, "composite_risk")[
            ["station", "primary_line", "borough", "composite_risk"]
        ].rename(columns={
            "station": "Station", "primary_line": "Line",
            "borough": "Boro", "composite_risk": "Score",
        }).reset_index(drop=True)
        st.dataframe(top, use_container_width=True, height=560, hide_index=True)

    # ---- Risk component breakdown ----
    st.markdown("##### Risk Component Breakdown by Line")
    comps = line_df.sort_values("composite_risk", ascending=False)
    fig3 = go.Figure()
    for comp, color in [
        ("flood_risk",        RAIN_COLOR),
        ("heat_risk",         HEAT_COLOR),
        ("vulnerability",     "#7C3AED"),
        ("economic_exposure", ACCENT),
    ]:
        fig3.add_trace(go.Bar(
            x=comps["line"], y=comps[comp], name=comp.replace("_"," "),
            marker_color=color,
        ))
    fig3.update_layout(
        barmode="group", title="Risk Dimensions by Line",
        xaxis_title="Line", yaxis_title="Score (0–10)",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Methodology"):
        st.markdown("""
**Risk dimensions (0–10 each):**
- **Flood risk** — geographic flood exposure (coastal proximity, FEMA zones, post-Sandy reports) + data-driven uplift from real MTA incidents on heavy-rain days
- **Heat risk** — urban heat island + underground station exposure + uplift from real extreme-heat-day incidents
- **Vulnerability** — infrastructure age and structure type (underground +2.0, open cut +1.0, elevated +0.5)
- **Economic exposure** — ridership-weighted loss potential (normalized to system max)

**Composite** = 0.40·flood + 0.30·heat + 0.20·vuln + 0.10·econ

**Sources:** MTA Sandy After-Action 2013, NYC Climate Risk 2023, FEMA FIRM maps, MTA 2023 Blue Book ridership, data.ny.gov station registry (39hk-dx4f).
        """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    run_pipeline_if_needed()

    st.markdown("""
    <div class="main-header">
      <h1>NYC Climate Risk × MTA Subway</h1>
      <p>Extreme rainfall and heat events drive subway disruptions — quantified with real MTA data.</p>
      <p style="margin-top:6px;font-size:11px;color:rgba(255,255,255,0.35);">
        Columbia University · Climate Finance Program · 2025 &nbsp;·&nbsp;
        Data: Open-Meteo, MTA Open Data (data.ny.gov)
      </p>
    </div>
    """, unsafe_allow_html=True)

    sys_df   = load_systemwide()
    if sys_df is None:
        st.error("Data not found. Refresh the page to trigger the pipeline.")
        return

    line_df      = load_line_daily()
    loss_df      = load_economic_loss()
    fi_df        = load_feature_importance()
    ml_res       = load_ml_results()
    corr_mat     = load_corr_matrix()
    weather      = load_weather()
    station_risk = load_station_risk()
    line_risk    = load_line_risk()

    d_start, d_end, selected_lines, rain_t, heat_t = sidebar_filters(sys_df)

    # Apply sidebar thresholds
    sys_df = sys_df.copy()
    if "precip_mm" in sys_df.columns:
        sys_df["heavy_rain"]   = (sys_df["precip_mm"] >= rain_t).astype(int)
    if "tmax_f" in sys_df.columns:
        sys_df["extreme_heat"] = (sys_df["tmax_f"] >= heat_t).astype(int)

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🌧  Overview",
        "🚇  MTA Performance",
        "📊  Correlations",
        "🤖  ML Model",
        "💵  Economic Impact",
        "🗺  Station Risk Map",
    ])
    with t1: tab_overview(sys_df, weather, d_start, d_end, rain_t, heat_t)
    with t2:
        if line_df is not None: tab_mta(line_df, d_start, d_end, selected_lines)
        else: st.info("Run pipeline to see per-line data.")
    with t3: tab_correlations(sys_df, corr_mat, d_start, d_end)
    with t4: tab_ml(ml_res, fi_df, sys_df)
    with t5: tab_economic(loss_df, d_start, d_end)
    with t6: tab_station_risk(station_risk, line_risk)


if __name__ == "__main__":
    main()
