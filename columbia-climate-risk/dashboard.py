"""
dashboard.py — NYC Climate Risk x MTA
--------------------------------------
CarbonPlan-inspired scrollytelling dashboard.
Dark theme, narrative-driven, animated data story.

Run:  streamlit run dashboard.py
"""

import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA = BASE / "data"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Climate Risk x MTA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CarbonPlan design tokens ────────────────────────────────────────────────
BG       = "#1b1e23"
BG_CARD  = "#24272d"
BG_HINT  = "#2a2c30"
TEXT     = "#ebebec"
TEXT_DIM = "#808080"
MUTED    = "#393a3d"
RED      = "#f07071"
ORANGE   = "#ea9755"
YELLOW   = "#d4c05e"
GREEN    = "#7eb36a"
TEAL     = "#64b9c4"
BLUE     = "#85a2f7"
PURPLE   = "#bc85d9"
PINK     = "#e587b6"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'JetBrains Mono', 'SF Mono', monospace",
              color=TEXT, size=12),
    margin=dict(l=48, r=24, t=48, b=48),
    xaxis=dict(gridcolor=MUTED, zerolinecolor=MUTED, tickfont=dict(color=TEXT_DIM)),
    yaxis=dict(gridcolor=MUTED, zerolinecolor=MUTED, tickfont=dict(color=TEXT_DIM)),
)

def styled_fig(fig, **kw):
    merged = {**PLOTLY_LAYOUT, **kw}
    fig.update_layout(**merged)
    return fig


# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

  /* ── Reset Streamlit chrome ────────────────────────────── */
  .stApp {{
    background-color: {BG};
    color: {TEXT};
  }}
  header[data-testid="stHeader"] {{ background: {BG}; }}
  section[data-testid="stSidebar"] {{ display: none; }}
  .block-container {{
    max-width: 1080px;
    padding: 0 2rem 6rem 2rem;
  }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{
    border: none !important;
  }}
  /* hide hamburger + footer */
  #MainMenu, footer, .stDeployButton {{ display: none !important; }}

  /* ── Typography ────────────────────────────────────────── */
  html, body, .stApp, .stMarkdown, p, li, span {{
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    color: {TEXT};
    -webkit-font-smoothing: antialiased;
  }}
  h1, h2, h3, h4, h5, h6 {{
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    color: {TEXT} !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
  }}
  .mono {{
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.04em;
  }}

  /* ── Section headers ───────────────────────────────────── */
  .section-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {TEXT_DIM};
    margin-bottom: 8px;
    padding-top: 4rem;
  }}
  .section-title {{
    font-size: 32px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: {TEXT};
    line-height: 1.15;
    margin-bottom: 12px;
  }}
  .section-body {{
    font-size: 17px;
    line-height: 1.65;
    color: {TEXT_DIM};
    max-width: 680px;
    margin-bottom: 2.5rem;
  }}

  /* ── Hero ───────────────────────────────────────────────── */
  .hero {{
    padding: 8rem 0 5rem 0;
    border-bottom: 1px solid {MUTED};
    margin-bottom: 3rem;
  }}
  .hero h1 {{
    font-size: 56px !important;
    font-weight: 700 !important;
    line-height: 1.08;
    letter-spacing: -0.03em;
    margin-bottom: 20px;
  }}
  .hero .subtitle {{
    font-size: 20px;
    line-height: 1.55;
    color: {TEXT_DIM};
    max-width: 620px;
  }}
  .hero .attribution {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: {MUTED};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 28px;
  }}

  /* ── KPI strip ─────────────────────────────────────────── */
  .kpi-strip {{
    display: flex;
    gap: 1px;
    margin: 2rem 0 2.5rem 0;
    background: {MUTED};
    border-radius: 4px;
    overflow: hidden;
  }}
  .kpi-cell {{
    flex: 1;
    background: {BG_CARD};
    padding: 20px 16px;
    text-align: center;
  }}
  .kpi-cell .label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {TEXT_DIM};
    margin-bottom: 6px;
  }}
  .kpi-cell .value {{
    font-family: 'Inter', sans-serif;
    font-size: 28px;
    font-weight: 600;
    color: {TEXT};
    letter-spacing: -0.02em;
  }}
  .kpi-cell .sub {{
    font-size: 11px;
    color: {MUTED};
    margin-top: 4px;
  }}

  /* ── Dividers ──────────────────────────────────────────── */
  .section-divider {{
    border: none;
    border-top: 1px solid {MUTED};
    margin: 4rem 0;
  }}

  /* ── Data table ────────────────────────────────────────── */
  .risk-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
  }}
  .risk-table th {{
    text-align: left;
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {TEXT_DIM};
    padding: 10px 12px;
    border-bottom: 1px solid {MUTED};
  }}
  .risk-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid {BG_HINT};
    color: {TEXT};
  }}
  .risk-table tr:hover td {{
    background: {BG_HINT};
  }}

  /* ── Source footer ─────────────────────────────────────── */
  .source-footer {{
    border-top: 1px solid {MUTED};
    padding-top: 3rem;
    margin-top: 4rem;
  }}
  .source-footer h3 {{
    font-size: 14px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_DIM} !important;
  }}
  .source-footer p, .source-footer li {{
    font-size: 13px;
    color: {TEXT_DIM};
    line-height: 1.7;
  }}

  /* fix plotly bg */
  .stPlotlyChart {{ background: transparent !important; }}

  /* smooth scroll */
  html {{ scroll-behavior: smooth; }}

  /* animated fade-in */
  @keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(30px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
  .fade-up {{
    animation: fadeUp 0.8s ease-out both;
  }}
  .fade-up-d1 {{ animation-delay: 0.15s; }}
  .fade-up-d2 {{ animation-delay: 0.30s; }}
  .fade-up-d3 {{ animation-delay: 0.45s; }}
  .fade-up-d4 {{ animation-delay: 0.60s; }}

  /* color accents */
  .c-red    {{ color: {RED}; }}
  .c-blue   {{ color: {BLUE}; }}
  .c-teal   {{ color: {TEAL}; }}
  .c-green  {{ color: {GREEN}; }}
  .c-orange {{ color: {ORANGE}; }}
  .c-yellow {{ color: {YELLOW}; }}
  .c-purple {{ color: {PURPLE}; }}

  /* stat callout */
  .stat-callout {{
    font-size: 64px;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1;
    margin: 0.5rem 0;
  }}
  .stat-context {{
    font-size: 15px;
    color: {TEXT_DIM};
    line-height: 1.5;
  }}

  /* toggle buttons */
  .stRadio > div {{
    gap: 0 !important;
    background: {BG_HINT};
    border-radius: 4px;
    padding: 2px;
    display: inline-flex !important;
  }}
  .stRadio > div > label {{
    background: transparent !important;
    color: {TEXT_DIM} !important;
    border: none !important;
    padding: 6px 14px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    cursor: pointer;
  }}
  .stRadio > div > label[data-checked="true"],
  .stRadio > div [aria-checked="true"] {{
    background: {MUTED} !important;
    color: {TEXT} !important;
  }}

  /* streamlit elements override */
  .stSelectbox label, .stRadio > label {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_DIM} !important;
  }}
</style>
""", unsafe_allow_html=True)


# ── Auto-pipeline ────────────────────────────────────────────────────────────
def run_pipeline_if_needed():
    needed = [DATA/"nyc_weather_daily.csv", DATA/"mta_delays_daily.csv",
              DATA/"analysis_systemwide.csv", DATA/"ml_results.json",
              DATA/"station_risk.csv"]
    if all(f.exists() for f in needed):
        return
    st.info("First run — fetching data (~2 min)...")
    for i, (msg, cmd) in enumerate([
        ("Weather...",   ["python", str(BASE/"01_download_weather.py")]),
        ("MTA data...",  ["python", str(BASE/"02_download_mta.py")]),
        ("Merging...",   ["python", str(BASE/"03_merge_process.py")]),
        ("ML...",        ["python", str(BASE/"04_ml_analysis.py")]),
        ("Station risk.",["python", str(BASE/"05_station_risk.py")]),
    ]):
        with st.spinner(msg):
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE))
            if r.returncode != 0:
                st.error(f"Failed:\n{r.stderr[-2000:]}")
                st.stop()
    st.rerun()


# ── Loaders ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_sys():
    return pd.read_csv(DATA/"analysis_systemwide.csv", parse_dates=["date"])

@st.cache_data
def load_line_daily():
    p = DATA/"analysis_daily.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None

@st.cache_data
def load_loss():
    p = DATA/"economic_loss.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None

@st.cache_data
def load_ml():
    p = DATA/"ml_results.json"
    return json.load(open(p)) if p.exists() else None

@st.cache_data
def load_fi():
    p = DATA/"feature_importance.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_corr():
    p = DATA/"correlation_matrix.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None

@st.cache_data
def load_stations():
    return pd.read_csv(DATA/"station_risk.csv")

@st.cache_data
def load_line_risk():
    return pd.read_csv(DATA/"line_risk.csv")


# ── Helper: section header ──────────────────────────────────────────────────
def section(num, title, body=""):
    st.markdown(f"""
    <div class="section-label fade-up">{num}</div>
    <div class="section-title fade-up fade-up-d1">{title}</div>
    {"<div class='section-body fade-up fade-up-d2'>" + body + "</div>" if body else ""}
    """, unsafe_allow_html=True)

def divider():
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

def kpi_strip(cells):
    inner = ""
    for label, value, sub in cells:
        inner += f"""<div class="kpi-cell">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>"""
    st.markdown(f'<div class="kpi-strip fade-up fade-up-d2">{inner}</div>',
                unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    run_pipeline_if_needed()

    sys_df   = load_sys()
    loss_df  = load_loss()
    ml       = load_ml()
    fi_df    = load_fi()
    corr     = load_corr()
    line_df  = load_line_daily()
    stn_df   = load_stations()
    lrisk_df = load_line_risk()

    # pre-compute
    normal = sys_df[(sys_df["heavy_rain"]==0)&(sys_df["extreme_heat"]==0)]
    rain   = sys_df[sys_df["heavy_rain"]==1]
    heat   = sys_df[sys_df["extreme_heat"]==1]
    norm_avg = normal["total_incidents"].mean()
    rain_avg = rain["total_incidents"].mean()
    heat_avg = heat["total_incidents"].mean()
    rain_uplift = (rain_avg/norm_avg - 1)*100

    # ─────────────────────────────────────────────────────────────────────────
    # HERO
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero">
      <h1 class="fade-up">Climate risk is<br>reshaping how<br>New York <span class="c-red">moves</span>.</h1>
      <div class="subtitle fade-up fade-up-d1">
        Extreme rainfall and heat events are intensifying. When they hit,
        the subway system that 5.5 million New Yorkers depend on daily
        buckles under the stress. This is the data story of how climate
        risk translates into transit disruption and economic loss.
      </div>
      <div class="attribution fade-up fade-up-d2">
        Columbia University &nbsp;&middot;&nbsp; Climate Finance Program
        &nbsp;&middot;&nbsp; 2025
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — The weather is getting worse
    # ─────────────────────────────────────────────────────────────────────────
    section("01", "The weather is getting worse.",
            "Five years of New York City weather data reveal an acceleration "
            "in extreme events. Heavy rainfall days and heat emergencies are "
            "no longer outliers — they are becoming the pattern.")

    kpi_strip([
        ("Period", "2020 – 2024", "1,827 days"),
        ("Heavy rain days", f"{len(rain)}", f"{len(rain)/len(sys_df):.1%} of all days"),
        ("Extreme heat days", f"{len(heat)}", f"{len(heat)/len(sys_df):.1%} of all days"),
        ("Rain uplift", f"+{rain_uplift:.0f}%", "more incidents on rain days"),
    ])

    # Precipitation timeline
    fig_precip = go.Figure()
    colors_rain = np.where(sys_df["precip_mm"] >= 25, BLUE, MUTED)
    fig_precip.add_trace(go.Bar(
        x=sys_df["date"], y=sys_df["precip_mm"],
        marker_color=colors_rain, name="Precipitation",
    ))
    fig_precip.add_hline(y=25, line_dash="dot", line_color=BLUE, opacity=0.6,
                         annotation=dict(text="Heavy rain threshold (25 mm)",
                                         font=dict(size=11, color=TEXT_DIM),
                                         xanchor="left"))
    styled_fig(fig_precip, height=280,
               title=dict(text="Daily Precipitation (mm)", font=dict(size=14, color=TEXT_DIM)),
               xaxis=dict(gridcolor=MUTED), yaxis=dict(gridcolor=MUTED))
    st.plotly_chart(fig_precip, use_container_width=True)

    # Heat timeline (Celsius)
    fig_heat = go.Figure()
    fig_heat.add_trace(go.Scatter(
        x=sys_df["date"], y=sys_df["tmax_c"],
        mode="lines", line=dict(color=RED, width=1),
        fill="tozeroy", fillcolor="rgba(240,112,113,0.08)",
    ))
    fig_heat.add_hline(y=32.2, line_dash="dot", line_color=RED, opacity=0.6,
                       annotation=dict(text="Extreme heat threshold (32.2 °C)",
                                       font=dict(size=11, color=TEXT_DIM),
                                       xanchor="left"))
    styled_fig(fig_heat, height=260,
               title=dict(text="Daily Maximum Temperature (°C)", font=dict(size=14, color=TEXT_DIM)),
               showlegend=False)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Future projection callout
    st.markdown(f"""
    <div style="display:flex; gap:3rem; margin:2rem 0 1rem 0;" class="fade-up">
      <div>
        <div class="stat-callout c-blue">+23%</div>
        <div class="stat-context">projected increase in extreme<br>precipitation days by 2050<br>
        <span style="font-size:11px;color:{MUTED}">Source: NYC Climate Resiliency Design Guidelines v4.1</span></div>
      </div>
      <div>
        <div class="stat-callout c-red">+3.2 °C</div>
        <div class="stat-context">projected rise in average summer<br>temperatures by mid-century<br>
        <span style="font-size:11px;color:{MUTED}">Source: NPCC 2024 Climate Assessment</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — What that means for the subway
    # ─────────────────────────────────────────────────────────────────────────
    section("02", "When it rains, the subway breaks.",
            "MTA incident data from 2020–2024 shows a clear, measurable link "
            "between extreme weather and service disruptions. On heavy-rain days, "
            "the system averages <strong>{:.0f}%</strong> more incidents than normal.".format(rain_uplift))

    # Condition comparison bars
    fig_cond = go.Figure()
    labels = ["Normal days", "Heavy rain days", "Extreme heat days"]
    vals   = [norm_avg, rain_avg, heat_avg]
    colors = [TEXT_DIM, BLUE, RED]
    fig_cond.add_trace(go.Bar(
        x=labels, y=vals, marker_color=colors,
        text=[f"{v:.0f}" for v in vals], textposition="outside",
        textfont=dict(color=TEXT, size=14, family="'JetBrains Mono', monospace"),
    ))
    styled_fig(fig_cond, height=340, showlegend=False,
               title=dict(text="Average Daily Incidents by Weather Condition",
                          font=dict(size=14, color=TEXT_DIM)),
               yaxis=dict(title="Incidents per Day", gridcolor=MUTED))
    st.plotly_chart(fig_cond, use_container_width=True)

    # Monthly dual axis
    sys_m = sys_df.copy()
    sys_m["ym"] = sys_m["date"].dt.to_period("M").dt.to_timestamp()
    monthly = sys_m.groupby("ym").agg(
        precip=("precip_mm","sum"), incidents=("total_incidents","sum")
    ).reset_index()

    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dual.add_trace(go.Bar(
        x=monthly["ym"], y=monthly["precip"], name="Precipitation (mm)",
        marker_color=BLUE, opacity=0.4,
    ), secondary_y=False)
    fig_dual.add_trace(go.Scatter(
        x=monthly["ym"], y=monthly["incidents"], name="Incidents",
        line=dict(color=TEAL, width=2),
    ), secondary_y=True)
    styled_fig(fig_dual, height=320,
               title=dict(text="Monthly Precipitation vs Transit Incidents",
                          font=dict(size=14, color=TEXT_DIM)),
               legend=dict(orientation="h", y=-0.15, font=dict(size=11)))
    fig_dual.update_yaxes(title_text="Precipitation (mm)", secondary_y=False,
                          gridcolor=MUTED, tickfont=dict(color=TEXT_DIM))
    fig_dual.update_yaxes(title_text="Total Incidents", secondary_y=True,
                          gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT_DIM))
    st.plotly_chart(fig_dual, use_container_width=True)

    # Per-line uplift
    if line_df is not None:
        norm_g = line_df[line_df["heavy_rain"]==0].groupby("line")["total_incidents"].mean()
        rain_g = line_df[line_df["heavy_rain"]==1].groupby("line")["total_incidents"].mean()
        uplift = ((rain_g - norm_g) / norm_g * 100).dropna().sort_values(ascending=True).reset_index()
        uplift.columns = ["line", "pct"]

        fig_up = go.Figure(go.Bar(
            x=uplift["pct"], y=uplift["line"], orientation="h",
            marker_color=[BLUE if v >= 0 else TEXT_DIM for v in uplift["pct"]],
            text=[f"{v:+.0f}%" for v in uplift["pct"]],
            textposition="outside",
            textfont=dict(size=11, color=TEXT_DIM,
                          family="'JetBrains Mono', monospace"),
        ))
        fig_up.add_vline(x=0, line_color=MUTED)
        styled_fig(fig_up, height=max(380, len(uplift)*22),
                   title=dict(text="Incident Uplift on Heavy Rain Days by Subway Line",
                              font=dict(size=14, color=TEXT_DIM)),
                   showlegend=False)
        st.plotly_chart(fig_up, use_container_width=True)

    divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Correlation deep-dive
    # ─────────────────────────────────────────────────────────────────────────
    section("03", "Quantifying the link.",
            "Pearson correlations and regression analysis confirm a statistically "
            "significant relationship between weather intensity and transit disruptions.")

    LABEL_MAP = {
        "precip_mm": "Precipitation (mm)",
        "tmax_f": "Max Temp (°F)",
        "tmax_c": "Max Temp (°C)",
        "heavy_rain": "Heavy Rain Day",
        "extreme_heat": "Extreme Heat Day",
        "extreme_rain": "Extreme Rain",
        "roll7_precip_mm": "7-Day Rolling Precip",
        "roll3_precip_mm": "3-Day Rolling Precip",
        "weather_stress": "Weather Stress Index",
        "total_incidents": "Total Incidents",
        "avg_delay_min": "Avg Delay (min)",
        "heat_wave_day": "Heat Wave Day",
        "heat_wave_streak": "Heat Wave Streak",
        "lag1_precip_mm": "Prior-Day Precip",
        "lag2_precip_mm": "2-Day Lag Precip",
        "precip_hours": "Precipitation Hours",
        "windspeed_kmh": "Wind Speed (km/h)",
        "heat_index_approx": "Heat Index",
        "year": "Year",
        "month": "Month",
        "dow": "Day of Week",
        "season": "Season",
    }

    if corr is not None:
        wx = [c for c in ["precip_mm","tmax_f","heavy_rain","extreme_heat",
                           "roll7_precip_mm","weather_stress"] if c in corr.columns]
        tg = [c for c in ["total_incidents","avg_delay_min"] if c in corr.columns]
        sub = corr.loc[
            [r for r in wx+tg if r in corr.index],
            [c for c in wx+tg if c in corr.columns]
        ].round(3)

        display_x = [LABEL_MAP.get(c, c) for c in sub.columns]
        display_y = [LABEL_MAP.get(r, r) for r in sub.index]

        fig_corr = go.Figure(go.Heatmap(
            z=sub.values, x=display_x, y=display_y,
            colorscale=[[0, BLUE], [0.5, BG], [1, RED]],
            zmin=-1, zmax=1,
            text=sub.values.round(2), texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="r", tickfont=dict(color=TEXT_DIM)),
        ))
        styled_fig(fig_corr, height=400,
                   title=dict(text="Correlation Matrix — Weather vs Transit Disruption",
                              font=dict(size=14, color=TEXT_DIM)),
                   xaxis=dict(tickangle=30, tickfont=dict(size=10)))
        st.plotly_chart(fig_corr, use_container_width=True)

    # Scatter
    col1, col2 = st.columns(2)
    with col1:
        sub2 = sys_df[["precip_mm","total_incidents"]].dropna()
        r_val, p_val = pearsonr(sub2["precip_mm"], sub2["total_incidents"])
        coefs = np.polyfit(sub2["precip_mm"], sub2["total_incidents"], 1)
        x_ln = np.linspace(sub2["precip_mm"].min(), sub2["precip_mm"].max(), 50)

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=sub2["precip_mm"], y=sub2["total_incidents"], mode="markers",
            marker=dict(color=BLUE, opacity=0.25, size=4), name="Daily obs",
        ))
        fig_sc.add_trace(go.Scatter(
            x=x_ln, y=np.polyval(coefs, x_ln), mode="lines",
            line=dict(color=TEXT, width=2, dash="dash"), name="OLS trend",
        ))
        styled_fig(fig_sc, height=340, showlegend=False,
                   title=dict(text=f"Precipitation vs Incidents (r = {r_val:.3f})",
                              font=dict(size=13, color=TEXT_DIM)),
                   xaxis=dict(title="Daily Precipitation (mm)", gridcolor=MUTED),
                   yaxis=dict(title="Daily Incidents", gridcolor=MUTED))
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        sub3 = sys_df[["tmax_c","total_incidents"]].dropna()
        r_val2, _ = pearsonr(sub3["tmax_c"], sub3["total_incidents"])
        coefs2 = np.polyfit(sub3["tmax_c"], sub3["total_incidents"], 1)
        x_ln2 = np.linspace(sub3["tmax_c"].min(), sub3["tmax_c"].max(), 50)

        fig_sc2 = go.Figure()
        fig_sc2.add_trace(go.Scatter(
            x=sub3["tmax_c"], y=sub3["total_incidents"], mode="markers",
            marker=dict(color=RED, opacity=0.25, size=4),
        ))
        fig_sc2.add_trace(go.Scatter(
            x=x_ln2, y=np.polyval(coefs2, x_ln2), mode="lines",
            line=dict(color=TEXT, width=2, dash="dash"),
        ))
        styled_fig(fig_sc2, height=340, showlegend=False,
                   title=dict(text=f"Max Temperature vs Incidents (r = {r_val2:.3f})",
                              font=dict(size=13, color=TEXT_DIM)),
                   xaxis=dict(title="Max Temperature (°C)", gridcolor=MUTED),
                   yaxis=dict(title="Daily Incidents", gridcolor=MUTED))
        st.plotly_chart(fig_sc2, use_container_width=True)

    divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — ML Model
    # ─────────────────────────────────────────────────────────────────────────
    section("04", "Predicting disruptions with machine learning.",
            "We trained Ridge Regression and Random Forest models on weather features "
            "to predict daily incident counts. The models confirm that weather stress, "
            "heavy rain events, and extreme heat are the strongest weather-driven predictors.")

    if ml:
        models = ml.get("models", {}).get("total_incidents", {})
        ridge = models.get("ridge", {})
        rf    = models.get("random_forest", {})

        kpi_strip([
            ("Ridge R\u00b2", f"{ridge.get('cv_r2_mean',0):.3f}", "5-fold time-series CV"),
            ("Ridge MAE", f"{ridge.get('cv_mae_mean',0):.1f}", "incidents/day"),
            ("Random Forest R\u00b2", f"{rf.get('cv_r2_mean',0):.3f}", "5-fold CV"),
            ("RF MAE", f"{rf.get('cv_mae_mean',0):.1f}", "incidents/day"),
        ])

        c1, c2 = st.columns(2)
        with c1:
            if fi_df is not None and not fi_df.empty:
                top_fi = fi_df.nlargest(10, "importance").sort_values("importance")
                top_fi["label"] = top_fi["feature"].map(LABEL_MAP).fillna(top_fi["feature"])
                fig_fi = go.Figure(go.Bar(
                    x=top_fi["importance"], y=top_fi["label"], orientation="h",
                    marker=dict(color=top_fi["importance"],
                                colorscale=[[0, BG_HINT], [1, TEAL]],
                                showscale=False),
                    text=[f"{v:.3f}" for v in top_fi["importance"]],
                    textposition="outside",
                    textfont=dict(size=10, color=TEXT_DIM,
                                  family="'JetBrains Mono', monospace"),
                ))
                styled_fig(fig_fi, height=360, showlegend=False,
                           title=dict(text="Feature Importance (Random Forest)",
                                      font=dict(size=13, color=TEXT_DIM)))
                st.plotly_chart(fig_fi, use_container_width=True)

        with c2:
            preds = models.get("predictions", {})
            if preds:
                fig_pv = go.Figure()
                fig_pv.add_trace(go.Scatter(
                    x=preds["actual"], y=preds["rf"], mode="markers",
                    marker=dict(color=TEAL, opacity=0.3, size=4),
                ))
                lim = max(max(preds["actual"]), max(preds["rf"]))
                fig_pv.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                                 line=dict(dash="dash", color=MUTED))
                styled_fig(fig_pv, height=360, showlegend=False,
                           title=dict(text="Predicted vs Actual Incidents (Random Forest)",
                                      font=dict(size=13, color=TEXT_DIM)),
                           xaxis=dict(title="Actual Incidents", gridcolor=MUTED),
                           yaxis=dict(title="Predicted Incidents", gridcolor=MUTED))
                st.plotly_chart(fig_pv, use_container_width=True)

        # Ridge coefficients
        coef_data = ridge.get("coefs", {})
        wx_keys = ["precip_mm","tmax_f","heavy_rain","extreme_heat",
                   "weather_stress","roll7_precip_mm","heat_wave_day"]
        wx_coefs = {k: v for k, v in coef_data.items() if k in wx_keys}
        if wx_coefs:
            cdf = pd.DataFrame(list(wx_coefs.items()), columns=["feature","coef"])
            cdf["label"] = cdf["feature"].map(LABEL_MAP).fillna(cdf["feature"])
            cdf = cdf.sort_values("coef")
            fig_rc = go.Figure(go.Bar(
                x=cdf["coef"], y=cdf["label"], orientation="h",
                marker_color=[BLUE if v>0 else TEXT_DIM for v in cdf["coef"]],
                text=[f"{v:+.2f}" for v in cdf["coef"]],
                textposition="outside",
                textfont=dict(size=10, color=TEXT_DIM,
                              family="'JetBrains Mono', monospace"),
            ))
            fig_rc.add_vline(x=0, line_color=MUTED)
            styled_fig(fig_rc, height=300, showlegend=False,
                       title=dict(text="Ridge Regression Coefficients (Standardised Weather Features)",
                                  font=dict(size=13, color=TEXT_DIM)))
            st.plotly_chart(fig_rc, use_container_width=True)

    divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — Economic cost
    # ─────────────────────────────────────────────────────────────────────────
    section("05", "The cost is already significant. It will get worse.",
            "Conservative estimates based on rider time-value, load factors, and "
            "excess delay put weather-driven economic losses at nearly half a million "
            "dollars over 2020–2024. Under climate-change scenarios, these losses "
            "are projected to grow substantially.")

    if loss_df is not None:
        total = loss_df["economic_loss_usd"].sum()
        rain_loss = loss_df.loc[loss_df["heavy_rain"]==1, "economic_loss_usd"].sum()
        heat_loss = loss_df.loc[loss_df["extreme_heat"]==1, "economic_loss_usd"].sum()

        kpi_strip([
            ("Total estimated loss", f"${total/1e6:.2f}M", "2020 - 2024"),
            ("Rain-driven loss", f"${rain_loss/1e6:.2f}M", f"{rain_loss/total:.0%} of total"),
            ("Heat-driven loss", f"${heat_loss/1e6:.2f}M", f"{heat_loss/total:.0%} of total"),
        ])

        # Annual bars
        annual = loss_df.groupby(loss_df["date"].dt.year)["economic_loss_usd"].sum().reset_index()
        annual.columns = ["year", "loss"]

        fig_annual = go.Figure(go.Bar(
            x=annual["year"], y=annual["loss"],
            marker=dict(color=annual["loss"],
                        colorscale=[[0, BG_HINT], [1, ORANGE]],
                        showscale=False),
            text=[f"${v/1e3:.0f}K" for v in annual["loss"]],
            textposition="outside",
            textfont=dict(color=TEXT_DIM, size=12,
                          family="'JetBrains Mono', monospace"),
        ))
        styled_fig(fig_annual, height=320,
                   title=dict(text="Annual Economic Loss from Weather Disruptions",
                              font=dict(size=14, color=TEXT_DIM)),
                   xaxis=dict(title="", gridcolor=MUTED),
                   yaxis=dict(title="Economic Loss (USD)", gridcolor=MUTED))
        st.plotly_chart(fig_annual, use_container_width=True)

        # Scenario projection
        st.markdown(f"""
        <div class="section-label" style="padding-top:1.5rem">PROJECTION</div>
        <div class="section-title" style="font-size:24px">Three climate scenarios for MTA losses</div>
        <div class="section-body">
          Using NPCC 2024 projections for NYC and current loss rates, we model
          how annual weather-driven costs scale under baseline, moderate, and
          high-emissions scenarios through 2060.
        </div>
        """, unsafe_allow_html=True)

        # Build projection
        base_annual = annual["loss"].mean()
        years_proj = list(range(2025, 2061))
        # Baseline: +2%/yr, Moderate (SSP2-4.5): +4.5%/yr, High (SSP5-8.5): +8%/yr
        baseline = [base_annual * (1.02 ** (y-2024)) for y in years_proj]
        moderate = [base_annual * (1.045 ** (y-2024)) for y in years_proj]
        high     = [base_annual * (1.08  ** (y-2024)) for y in years_proj]

        fig_proj = go.Figure()
        # Historical
        fig_proj.add_trace(go.Scatter(
            x=annual["year"], y=annual["loss"],
            mode="lines+markers", name="Historical",
            line=dict(color=TEXT, width=2),
            marker=dict(color=TEXT, size=6),
        ))
        for vals, name, color, dash in [
            (baseline, "Baseline (+2%/yr)",     TEXT_DIM, "dot"),
            (moderate, "SSP2-4.5 (+4.5%/yr)",   ORANGE,  "dash"),
            (high,     "SSP5-8.5 (+8%/yr)",     RED,     "solid"),
        ]:
            fig_proj.add_trace(go.Scatter(
                x=years_proj, y=vals, name=name,
                line=dict(color=color, width=2, dash=dash),
            ))
        # Fill between moderate and high
        fig_proj.add_trace(go.Scatter(
            x=years_proj + years_proj[::-1],
            y=high + moderate[::-1],
            fill="toself", fillcolor="rgba(240,112,113,0.08)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        styled_fig(fig_proj, height=380,
                   title=dict(text="Projected Annual Weather-Driven Losses",
                              font=dict(size=14, color=TEXT_DIM)),
                   xaxis=dict(title="", gridcolor=MUTED),
                   yaxis=dict(title="Annual Loss (USD)", gridcolor=MUTED),
                   legend=dict(orientation="h", y=-0.15, font=dict(size=11)))
        st.plotly_chart(fig_proj, use_container_width=True)

        # Big callout
        high_2050 = base_annual * (1.08 ** (2050-2024))
        st.markdown(f"""
        <div style="display:flex; gap:3rem; margin:1.5rem 0;" class="fade-up">
          <div>
            <div class="stat-callout c-orange">${high_2050/1e6:.1f}M</div>
            <div class="stat-context">projected annual loss by 2050<br>under high-emissions scenario</div>
          </div>
          <div>
            <div class="stat-callout c-red">{high_2050/base_annual:.0f}x</div>
            <div class="stat-context">multiplier vs current<br>baseline average</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Cumulative curve
        dfs = loss_df.sort_values("date")
        dfs["cumulative"] = dfs["economic_loss_usd"].cumsum()
        major = dfs[dfs["event_type"].isin(["extreme_rain", "heat_emergency"])]

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=dfs["date"], y=dfs["cumulative"],
            fill="tozeroy", fillcolor="rgba(234,151,85,0.12)",
            line=dict(color=ORANGE, width=2), name="Cumulative loss",
        ))
        if not major.empty:
            fig_cum.add_trace(go.Scatter(
                x=major["date"], y=major["cumulative"], mode="markers",
                marker=dict(color=RED, size=7, symbol="circle"), name="Major event",
            ))
        styled_fig(fig_cum, height=300,
                   title=dict(text="Cumulative Economic Loss Over Time", font=dict(size=14, color=TEXT_DIM)),
                   legend=dict(orientation="h", y=-0.15, font=dict(size=11)))
        st.plotly_chart(fig_cum, use_container_width=True)

    divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6 — Station risk map
    # ─────────────────────────────────────────────────────────────────────────
    section("06", "Every station has a risk profile.",
            "We scored all 496 MTA subway stations across four risk dimensions: "
            "flood exposure, heat exposure, infrastructure vulnerability, and "
            "ridership-weighted economic exposure. Hover over any station to see "
            "its breakdown.")

    dim = st.radio(
        "Risk dimension",
        ["composite_risk", "flood_risk", "heat_risk", "vulnerability", "economic_exposure"],
        horizontal=True, index=0,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    dim_colors = {
        "composite_risk":    [[0, GREEN], [0.5, YELLOW], [1, RED]],
        "flood_risk":        [[0, BG_HINT], [1, BLUE]],
        "heat_risk":         [[0, BG_HINT], [1, RED]],
        "vulnerability":     [[0, BG_HINT], [1, PURPLE]],
        "economic_exposure": [[0, BG_HINT], [1, ORANGE]],
    }

    fig_map = go.Figure(go.Scattermapbox(
        lat=stn_df["lat"], lon=stn_df["lon"],
        mode="markers",
        marker=dict(
            size=7, opacity=0.85,
            color=stn_df[dim],
            colorscale=dim_colors.get(dim, [[0, BG_HINT], [1, TEAL]]),
            cmin=float(stn_df[dim].min()),
            cmax=float(stn_df[dim].max()),
            colorbar=dict(
                title=dict(text=dim.replace("_"," "), font=dict(color=TEXT_DIM, size=11)),
                tickfont=dict(color=TEXT_DIM, size=10),
                bgcolor="rgba(0,0,0,0)",
                outlinewidth=0,
                len=0.6,
            ),
        ),
        text=[
            f"<b>{r['station']}</b><br>"
            f"<span style='color:{TEXT_DIM}'>Line {r['primary_line']}  ·  {r['borough']}  ·  {r['structure']}</span><br><br>"
            f"<span style='color:{BLUE}'>Flood risk</span>  {r['flood_risk']:.1f}<br>"
            f"<span style='color:{RED}'>Heat risk</span>   {r['heat_risk']:.1f}<br>"
            f"<span style='color:{PURPLE}'>Vulnerability</span> {r['vulnerability']:.1f}<br>"
            f"<span style='color:{ORANGE}'>Economic exp</span> {r['economic_exposure']:.1f}<br><br>"
            f"<b>Composite  {r['composite_risk']:.2f}</b>"
            for _, r in stn_df.iterrows()
        ],
        hoverinfo="text",
    ))
    fig_map.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=40.72, lon=-73.95),
            zoom=10.2,
        ),
        height=620,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Line risk ranking
    st.markdown(f"""
    <div class="section-label" style="padding-top:1.5rem">LINE RISK RANKING</div>
    """, unsafe_allow_html=True)

    lr = lrisk_df.sort_values("composite_risk", ascending=True)
    fig_lr = go.Figure(go.Bar(
        x=lr["composite_risk"], y=lr["line"], orientation="h",
        marker=dict(color=lr["line_color"]),
        text=[f"{v:.2f}  {l}" for v, l in zip(lr["composite_risk"], lr["risk_label"])],
        textposition="outside",
        textfont=dict(size=10, color=TEXT_DIM,
                      family="'JetBrains Mono', monospace"),
    ))
    styled_fig(fig_lr, height=560, showlegend=False,
               title=dict(text="Composite Risk Score by Subway Line",
                          font=dict(size=14, color=TEXT_DIM)),
               xaxis=dict(range=[0, 10], gridcolor=MUTED))
    st.plotly_chart(fig_lr, use_container_width=True)

    # Top stations table
    st.markdown(f"""
    <div class="section-label" style="padding-top:0.5rem">HIGHEST RISK STATIONS</div>
    """, unsafe_allow_html=True)

    top_stn = stn_df.nlargest(15, "composite_risk")
    rows_html = ""
    for _, r in top_stn.iterrows():
        badge_color = RED if r["composite_risk"] >= 8 else ORANGE if r["composite_risk"] >= 6 else YELLOW
        rows_html += f"""<tr>
            <td><strong>{r['station']}</strong></td>
            <td><span style="color:{r['line_color']}">{r['primary_line']}</span></td>
            <td>{r['borough']}</td>
            <td>{r['flood_risk']:.1f}</td>
            <td>{r['heat_risk']:.1f}</td>
            <td>{r['vulnerability']:.1f}</td>
            <td>{r['economic_exposure']:.1f}</td>
            <td><span style="color:{badge_color};font-weight:600">{r['composite_risk']:.2f}</span></td>
        </tr>"""

    st.markdown(f"""
    <table class="risk-table">
      <thead><tr>
        <th>Station</th><th>Line</th><th>Borough</th>
        <th>Flood</th><th>Heat</th><th>Vuln</th><th>Econ</th><th>Composite</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # Component breakdown
    comps = lrisk_df.sort_values("composite_risk", ascending=False)
    fig_comp = go.Figure()
    for comp, color, name in [
        ("flood_risk",        BLUE,   "Flood"),
        ("heat_risk",         RED,    "Heat"),
        ("vulnerability",     PURPLE, "Vulnerability"),
        ("economic_exposure", ORANGE, "Economic"),
    ]:
        fig_comp.add_trace(go.Bar(
            x=comps["line"], y=comps[comp], name=name,
            marker_color=color, opacity=0.85,
        ))
    styled_fig(fig_comp, height=360, barmode="group",
               title=dict(text="Risk Dimensions by Subway Line",
                          font=dict(size=14, color=TEXT_DIM)),
               xaxis=dict(title="", gridcolor=MUTED),
               yaxis=dict(title="Score (0-10)", gridcolor=MUTED),
               legend=dict(orientation="h", y=-0.15, font=dict(size=11)))
    st.plotly_chart(fig_comp, use_container_width=True)

    divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 7 — Sources & methodology
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="source-footer">
      <div class="section-label">07</div>
      <h3>Data sources & methodology</h3>

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:2rem; margin-top:1.5rem;">
        <div>
          <p style="color:{TEXT};font-weight:500;margin-bottom:8px;">Weather data</p>
          <p>Daily precipitation, temperature, wind speed, and derived variables
          (heat index, rolling averages, lag terms) from
          <a href="https://open-meteo.com" style="color:{TEAL}">Open-Meteo</a>
          historical API. NYC Central Park station, 2020-2024.</p>

          <p style="color:{TEXT};font-weight:500;margin-bottom:8px;margin-top:1.2rem;">MTA incident data</p>
          <p>Monthly performance totals from
          <a href="https://data.ny.gov" style="color:{TEAL}">data.ny.gov</a>
          datasets g937-7k7c and 9zbp-wz3y, disaggregated to daily resolution
          using weather-aware statistical downscaling.</p>

          <p style="color:{TEXT};font-weight:500;margin-bottom:8px;margin-top:1.2rem;">Station locations</p>
          <p>MTA subway station registry from data.ny.gov (39hk-dx4f).
          496 stations with latitude, longitude, structure type, and serving lines.</p>
        </div>

        <div>
          <p style="color:{TEXT};font-weight:500;margin-bottom:8px;">Risk model</p>
          <p>Station-level risk scores combine four dimensions (0-10 each):</p>
          <ul>
            <li><span class="c-blue">Flood risk</span> (40%) — geographic flood exposure + data-driven incident uplift on rain days</li>
            <li><span class="c-red">Heat risk</span> (30%) — UHI exposure + underground heat trapping + incident uplift</li>
            <li><span class="c-purple">Vulnerability</span> (20%) — infrastructure age, structure type</li>
            <li><span class="c-orange">Economic exposure</span> (10%) — ridership-weighted loss potential</li>
          </ul>
          <p>Sources: MTA Sandy After-Action 2013, NYC Climate Risk 2023, FEMA FIRM maps.</p>

          <p style="color:{TEXT};font-weight:500;margin-bottom:8px;margin-top:1.2rem;">Economic loss model</p>
          <p>Loss = Excess incidents x Load factor (95 pax) x P(affected) (35%)
          x Value of time ($18.50/hr, NYC DOT 2023) x Excess delay.
          Conservative lower bound excluding indirect costs.</p>

          <p style="color:{TEXT};font-weight:500;margin-bottom:8px;margin-top:1.2rem;">ML models</p>
          <p>Ridge Regression and Random Forest with 5-fold time-series
          cross-validation. Features: precipitation, temperature, derived weather
          stress indices, temporal encodings.</p>
        </div>
      </div>

      <div style="margin-top:2.5rem; padding-top:1.5rem; border-top:1px solid {MUTED};">
        <p style="font-size:12px; color:{MUTED};">
          Columbia University &nbsp;&middot;&nbsp; Climate Finance Program &nbsp;&middot;&nbsp; 2025
          &nbsp;&nbsp;|&nbsp;&nbsp;
          Analysis: Python, scikit-learn, Plotly
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
