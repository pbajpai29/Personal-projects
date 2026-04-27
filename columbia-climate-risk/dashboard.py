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
    st.info("First run: fetching data (~2 min)...")
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

    # pre-compute
    normal = sys_df[(sys_df["heavy_rain"]==0)&(sys_df["extreme_heat"]==0)]
    rain   = sys_df[sys_df["heavy_rain"]==1]
    heat   = sys_df[sys_df["extreme_heat"]==1]
    norm_avg = normal["total_incidents"].mean()
    rain_avg = rain["total_incidents"].mean()
    heat_avg = heat["total_incidents"].mean()
    rain_uplift = (rain_avg/norm_avg - 1)*100

    # Economic pre-compute
    COST_MULTIPLIER = 27.5
    if loss_df is not None:
        loss_df = loss_df.copy()
        loss_df["economic_loss_usd"] = loss_df["economic_loss_usd"] * COST_MULTIPLIER
        total_loss = loss_df["economic_loss_usd"].sum()
        avg_annual_loss = total_loss / 5
        annual = loss_df.groupby(loss_df["date"].dt.year)["economic_loss_usd"].sum().reset_index()
        annual.columns = ["year", "loss"]
        base_annual = annual["loss"].mean()
    else:
        total_loss = avg_annual_loss = base_annual = 0
        annual = pd.DataFrame(columns=["year", "loss"])

    # ─────────────────────────────────────────────────────────────────────────
    # HERO
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero">
      <h1 class="fade-up">The MTA is not<br>prepared for<br><span class="c-red">wildcard</span> scenarios.</h1>
      <div class="subtitle fade-up fade-up-d1">
        A 100-year flood every five years. Simultaneous failure across multiple lines.
        Emergency shutdowns lasting days. These aren't hypotheticals. They're the
        scenarios the MTA's century-old infrastructure was never built to handle,
        and climate change is making them inevitable.
      </div>
      <div class="attribution fade-up fade-up-d2">
        Columbia University &nbsp;&middot;&nbsp; MS in Climate Finance
        &nbsp;&middot;&nbsp; Climate Risk &nbsp;&middot;&nbsp; 2026
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — Wildcard Scenario Modeling
    # ─────────────────────────────────────────────────────────────────────────
    section("01", "The scenarios nobody is budgeting for.",
            "Not routine bad weather. Compound failures. A 100-year flood that now "
            "hits every decade, a power grid that collapses mid-heatwave, flash flooding "
            "and storm surge arriving together. Four wildcard scenarios, modeled to 2060.")

    # Scenario data: name -> (direct_cost, color, annual_prob_by_decade)
    # prob_by_decade = [2025, 2030, 2040, 2050, 2060]
    SCENARIOS = {
        "100-Year Flood":       (5.0e9,  BLUE,   [0.15, 0.18, 0.22, 0.26, 0.30]),
        "Cascading Power Fail": (0.8e9,  RED,    [0.08, 0.11, 0.15, 0.20, 0.25]),
        "Heat Emergency":       (1.2e9,  ORANGE, [0.10, 0.14, 0.19, 0.25, 0.32]),
        "Flash Flood + Surge":  (3.5e9,  TEAL,   [0.12, 0.15, 0.19, 0.24, 0.30]),
    }
    proj_years = [2025, 2030, 2040, 2050, 2060]

    # --- Dramatic line chart: all 4 scenarios + stacked area for combined ---
    fig_wc = go.Figure()

    # Compute combined total EAL per year for stacked area
    combined_eal = [0.0] * len(proj_years)
    scenario_eals = {}
    for name, (cost, color, probs) in SCENARIOS.items():
        eal_line = [cost * p / 1e6 for p in probs]
        scenario_eals[name] = eal_line
        combined_eal = [c + e for c, e in zip(combined_eal, eal_line)]

    # Stacked area for combined total (background)
    fig_wc.add_trace(go.Scatter(
        x=proj_years, y=combined_eal,
        fill="tozeroy", fillcolor="rgba(240,112,113,0.08)",
        line=dict(width=1, color=MUTED, dash="dot"),
        name="Combined Total", hovertemplate="Combined: $%{y:.0f}M/yr<extra></extra>",
    ))

    # Individual scenario lines on top
    for name, (cost, color, probs) in SCENARIOS.items():
        eal_line = scenario_eals[name]
        fig_wc.add_trace(go.Scatter(
            x=proj_years, y=eal_line,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=3),
            marker=dict(size=9, color=color, line=dict(width=2, color=BG)),
            hovertemplate=f"<b>{name}</b><br>%{{x}}: $%{{y:.0f}}M/yr<extra></extra>",
        ))

    # Current weather losses reference
    fig_wc.add_hline(y=avg_annual_loss/1e6, line_dash="dash", line_color=YELLOW, opacity=0.7,
                     annotation=dict(
                         text=f"Today's weather losses: ${avg_annual_loss/1e6:.1f}M/yr",
                         font=dict(size=11, color=YELLOW, family="'JetBrains Mono', monospace"),
                         xanchor="left", x=0.01, bgcolor=BG_CARD, borderpad=4))

    # Annotate the 2060 endpoints
    for name, (cost, color, probs) in SCENARIOS.items():
        val_2060 = cost * probs[-1] / 1e6
        fig_wc.add_annotation(
            x=2060, y=val_2060,
            text=f"${val_2060:.0f}M",
            font=dict(size=11, color=color, family="'JetBrains Mono', monospace",
                      weight="bold" if val_2060 > 500 else "normal"),
            showarrow=True, arrowhead=0, arrowcolor=color, arrowwidth=1,
            ax=45, ay=0, bgcolor=BG_CARD, borderpad=3,
        )

    # Combined 2060 annotation
    fig_wc.add_annotation(
        x=2060, y=combined_eal[-1],
        text=f"<b>TOTAL: ${combined_eal[-1]:.0f}M/yr</b>",
        font=dict(size=12, color=RED, family="'JetBrains Mono', monospace"),
        showarrow=True, arrowhead=2, arrowcolor=RED, arrowwidth=2,
        ax=50, ay=-30, bgcolor=BG_CARD, borderpad=4,
    )

    styled_fig(fig_wc, height=520,
               title=dict(text="Wildcard Scenarios: Expected Annual Loss Projection to 2060",
                          font=dict(size=16, color=TEXT)),
               xaxis=dict(title="", gridcolor=MUTED, dtick=5,
                          range=[2023, 2063]),
               yaxis=dict(title="Expected Annual Loss ($M)", gridcolor=MUTED),
               legend=dict(orientation="h", y=-0.12, font=dict(size=11),
                           bgcolor="rgba(0,0,0,0)"),
               margin=dict(l=48, r=80, t=56, b=56))
    st.plotly_chart(fig_wc, use_container_width=True)

    # --- Callout numbers ---
    total_eal_2025 = sum(cost * probs[0] for _, (cost, _, probs) in SCENARIOS.items())
    total_eal_2060 = sum(cost * probs[-1] for _, (cost, _, probs) in SCENARIOS.items())

    st.markdown(f"""
    <div style="display:flex; gap:3rem; margin:1.5rem 0;" class="fade-up">
      <div>
        <div class="stat-callout c-red">${total_eal_2060/1e9:.1f}B</div>
        <div class="stat-context">combined expected annual loss<br>across all wildcards by 2060</div>
      </div>
      <div>
        <div class="stat-callout c-orange">{total_eal_2060/avg_annual_loss:.0f}x</div>
        <div class="stat-context">vs current annual<br>weather disruption costs</div>
      </div>
      <div>
        <div class="stat-callout c-teal">{total_eal_2060/total_eal_2025:.1f}x</div>
        <div class="stat-context">growth from today's<br>wildcard exposure</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Beyond Repair: Interactive Visioning ---
    st.markdown(f"""
    <div style="margin:2.5rem 0 1rem 0;" class="fade-up">
      <p style="color:{TEAL};font-weight:700;font-size:24px;letter-spacing:-0.02em;margin-bottom:6px;">
        Beyond repair: visioning the future</p>
      <p style="color:{TEXT_DIM};font-size:15px;line-height:1.7;max-width:680px;">
        Wildcard scenarios demand <strong style="color:{TEXT}">visioning</strong>, not just planning.
        Explore how NYC transit must evolve through four phases of climate adaptation.</p>
    </div>
    """, unsafe_allow_html=True)

    # Phase data for the interactive timeline
    PHASES = {
        "React (Now to 2030)": {
            "color": RED,
            "headline": "Climate migration begins",
            "body": (
                "100-year floods hitting every 5 to 10 years force full or partial retreat "
                "from flood-prone areas. Residents shift toward outer boroughs based on "
                "transit reliability, permanently changing where people live and work."
            ),
            "transit_mix": {"Subway": 62, "Bus": 22, "Ferry": 3, "Elevated": 0, "Micro-transit": 3, "Walking/Cycling": 10},
            "metric_label": "Flood disruptions/yr", "metric_val": "75+",
        },
        "Adapt (2030 to 2040)": {
            "color": ORANGE,
            "headline": "Hybrid mobility takes hold",
            "body": (
                "Heat risk hits vulnerable populations hardest. Long walks and waits become "
                "dangerous. Transit users shift to hybrid mobility: more buses, ferries, and "
                "elevated lines. The MTA begins diversifying beyond subways."
            ),
            "transit_mix": {"Subway": 48, "Bus": 25, "Ferry": 8, "Elevated": 5, "Micro-transit": 6, "Walking/Cycling": 8},
            "metric_label": "Extreme heat days/yr", "metric_val": "40+",
        },
        "Transform (2040 to 2050)": {
            "color": TEAL,
            "headline": "Infrastructure reimagined",
            "body": (
                "Resilience bonds tied to congestion pricing fund massive rebuilds. Underground "
                "lines shift to elevated systems, slashing flood exposure. Green stormwater "
                "infrastructure addresses root causes, not just symptoms."
            ),
            "transit_mix": {"Subway": 35, "Bus": 22, "Ferry": 12, "Elevated": 15, "Micro-transit": 8, "Walking/Cycling": 8},
            "metric_label": "Lines converted to elevated", "metric_val": "6",
        },
        "Thrive (2050 to 2060)": {
            "color": GREEN,
            "headline": "Climate-positive transit",
            "body": (
                "Solar-powered cooling protects underserved neighborhoods. Nature-based solutions "
                "are standard. The network is diversified, resilient, and equitable. Designed for "
                "the climate that actually exists."
            ),
            "transit_mix": {"Subway": 28, "Bus": 18, "Ferry": 15, "Elevated": 20, "Micro-transit": 10, "Walking/Cycling": 9},
            "metric_label": "Annual savings vs. inaction", "metric_val": "$4.2B",
        },
    }

    phase_names = list(PHASES.keys())
    selected_phase = st.radio("ADAPTATION TIMELINE", phase_names, horizontal=True, index=0,
                              key="visioning_phase")
    phase = PHASES[selected_phase]

    # --- Phase detail + metric side-by-side ---
    st.markdown(f"""
    <div style="display:flex; gap:1.5rem; margin:1rem 0 1.5rem 0;" class="fade-up">
      <div style="flex:1; padding:1.6rem 2rem; background:{BG_CARD}; border-radius:8px;
                  border-left:4px solid {phase['color']};">
        <p style="color:{phase['color']};font-weight:700;font-size:18px;margin:0 0 4px 0;
                  letter-spacing:-0.01em;">{phase['headline']}</p>
        <p style="color:{TEXT_DIM};font-size:11px;margin:0 0 12px 0;font-family:'JetBrains Mono',monospace;
           letter-spacing:0.06em;text-transform:uppercase;">{selected_phase}</p>
        <p style="color:{TEXT_DIM};font-size:14px;line-height:1.75;margin:0;">
          {phase['body']}</p>
      </div>
      <div style="width:180px; padding:1.6rem 1.2rem; background:{BG_CARD}; border-radius:8px;
                  display:flex; flex-direction:column; align-items:center; justify-content:center;
                  text-align:center;">
        <div style="font-size:38px; font-weight:700; color:{phase['color']};
                    letter-spacing:-0.03em; line-height:1;">{phase['metric_val']}</div>
        <div style="font-size:10px; color:{TEXT_DIM}; margin-top:8px;
                    font-family:'JetBrains Mono',monospace; letter-spacing:0.04em;
                    text-transform:uppercase;">{phase['metric_label']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Transit mode evolution chart ---
    mode_colors = {
        "Subway": BLUE, "Bus": ORANGE, "Ferry": TEAL,
        "Elevated": PURPLE, "Micro-transit": PINK, "Walking/Cycling": GREEN,
    }

    fig_vision = go.Figure()
    modes = list(mode_colors.keys())
    for mode in modes:
        vals = [PHASES[p]["transit_mix"][mode] for p in phase_names]
        fig_vision.add_trace(go.Bar(
            x=phase_names, y=vals, name=mode,
            marker=dict(color=mode_colors[mode],
                        opacity=[1.0 if p == selected_phase else 0.3 for p in phase_names]),
            hovertemplate=f"<b>{mode}</b><br>%{{x}}: %{{y}}%<extra></extra>",
            text=[f"{v}%" if p == selected_phase else "" for v, p in zip(vals, phase_names)],
            textposition="inside", textfont=dict(size=10, color=TEXT),
        ))
    styled_fig(fig_vision, height=360, barmode="stack",
               title=dict(text="Transit Mode Share Evolution",
                          font=dict(size=14, color=TEXT_DIM)),
               xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11, color=TEXT)),
               yaxis=dict(title="Mode Share (%)", gridcolor=MUTED, range=[0, 105]),
               legend=dict(orientation="h", y=-0.18, font=dict(size=10),
                           bgcolor="rgba(0,0,0,0)"),
               margin=dict(l=48, r=24, t=48, b=70))
    st.plotly_chart(fig_vision, use_container_width=True)

    # --- Solution opportunities ---
    solutions = [
        ("Resilience Bonds", "$330M", "Tied to congestion pricing revenue", BLUE),
        ("Elevated Lines", "6 routes", "Convert flood-prone underground segments", PURPLE),
        ("Green Infrastructure", "40% runoff reduction", "Bioswales, permeable surfaces, green roofs", GREEN),
        ("Solar Cooling", "472 stations", "Platform cooling for underserved areas", ORANGE),
    ]

    st.markdown(f"""
    <p style="color:{TEXT_DIM};font-size:11px;font-family:'JetBrains Mono',monospace;
       letter-spacing:0.12em;text-transform:uppercase;margin:1rem 0 0.8rem 0;">
       Opportunity portfolio</p>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    for col, (title, val, desc, color) in zip(cols, solutions):
        with col:
            st.markdown(f"""
            <div style="padding:1.2rem; background:{BG_CARD}; border-radius:8px; height:100%;
                        border-top:3px solid {color};" class="fade-up">
              <p style="color:{color};font-weight:600;font-size:13px;margin:0 0 4px 0;">{title}</p>
              <p style="color:{TEXT};font-size:20px;font-weight:700;margin:2px 0 8px 0;
                        letter-spacing:-0.02em;">{val}</p>
              <p style="color:{TEXT_DIM};font-size:12px;line-height:1.5;margin:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Economic cost (moved up)
    # ─────────────────────────────────────────────────────────────────────────
    section("02", "Weather already costs the MTA $2.5 million a year.",
            "Direct rider delays, service rerouting, crew overtime, bus bridges. "
            "That number excludes infrastructure damage and long-term ridership loss, "
            "which push the real figure 3 to 5x higher.")

    if loss_df is not None:
        rain_loss = loss_df.loc[loss_df["heavy_rain"]==1, "economic_loss_usd"].sum()
        heat_loss = loss_df.loc[loss_df["extreme_heat"]==1, "economic_loss_usd"].sum()

        kpi_strip([
            ("Total losses (2020-2024)", f"${total_loss/1e6:.1f}M", "direct + indirect costs"),
            ("Annual average", f"${avg_annual_loss/1e6:.1f}M", "per year"),
            ("Rain-driven", f"${rain_loss/1e6:.1f}M", f"{rain_loss/total_loss:.0%} of total"),
            ("Heat-driven", f"${heat_loss/1e6:.1f}M", f"{heat_loss/total_loss:.0%} of total"),
        ])

        fig_annual = go.Figure(go.Bar(
            x=annual["year"], y=annual["loss"],
            marker=dict(color=annual["loss"],
                        colorscale=[[0, BG_HINT], [1, ORANGE]],
                        showscale=False),
            text=[f"${v/1e6:.1f}M" for v in annual["loss"]],
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

        # Projection
        st.markdown(f"""
        <div class="section-label" style="padding-top:1.5rem">PROJECTION</div>
        <div class="section-title" style="font-size:24px">This gets much worse from here</div>
        <div class="section-body">
          Three scenarios based on NPCC 2024 climate projections for NYC.
          Even the baseline assumption shows costs doubling by 2040.
        </div>
        """, unsafe_allow_html=True)

        years_proj = list(range(2025, 2061))
        baseline_proj = [base_annual * (1.02 ** (y-2024)) for y in years_proj]
        moderate_proj = [base_annual * (1.045 ** (y-2024)) for y in years_proj]
        high_proj     = [base_annual * (1.08  ** (y-2024)) for y in years_proj]

        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(
            x=annual["year"], y=annual["loss"],
            mode="lines+markers", name="Historical",
            line=dict(color=TEXT, width=2), marker=dict(color=TEXT, size=6),
        ))
        for vals, name, color, dash in [
            (baseline_proj, "Baseline (+2%/yr)",   TEXT_DIM, "dot"),
            (moderate_proj, "SSP2-4.5 (+4.5%/yr)", ORANGE,  "dash"),
            (high_proj,     "SSP5-8.5 (+8%/yr)",   RED,     "solid"),
        ]:
            fig_proj.add_trace(go.Scatter(
                x=years_proj, y=vals, name=name,
                line=dict(color=color, width=2, dash=dash),
            ))
        fig_proj.add_trace(go.Scatter(
            x=years_proj + years_proj[::-1],
            y=high_proj + moderate_proj[::-1],
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
                   title=dict(text="Cumulative Economic Loss Over Time",
                              font=dict(size=14, color=TEXT_DIM)),
                   legend=dict(orientation="h", y=-0.15, font=dict(size=11)))
        st.plotly_chart(fig_cum, use_container_width=True)

    divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — ML Model
    # ─────────────────────────────────────────────────────────────────────────
    section("03", "ML confirms what riders already know.",
            "Ridge regression and Random Forest trained on 5 years of daily weather "
            "to predict subway incidents. Top predictors: cumulative rainfall, "
            "heat stress, and whether it rained the day before.")

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
                           title=dict(text="Predicted vs Actual Incidents (RF)",
                                      font=dict(size=13, color=TEXT_DIM)),
                           xaxis=dict(title="Actual Incidents", gridcolor=MUTED),
                           yaxis=dict(title="Predicted Incidents", gridcolor=MUTED))
                st.plotly_chart(fig_pv, use_container_width=True)

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
                       title=dict(text="Ridge Coefficients (Standardised Weather Features)",
                                  font=dict(size=13, color=TEXT_DIM)))
            st.plotly_chart(fig_rc, use_container_width=True)

    divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Global Transit Comparison
    # ─────────────────────────────────────────────────────────────────────────
    section("04", "Other cities are already ahead.",
            "Tokyo built a $4B underground reservoir. Singapore budgets $100B for "
            "coastal defense. London published 477 climate risks and is acting on them. "
            "NYC has the exposure and the spending, but not the execution speed.")

    # Comparison data
    CITIES = {
        "New York City": {
            "system": "MTA", "flood": "Extreme", "heat": "High",
            "flood_score": 9.5, "heat_score": 7.5, "color": RED,
            "resilience_spend": 7600, "stations": 472, "age": 120,
            "key_threats": [
                "Aging 120-year sewer system designed for 44mm/hr; storms now exceed 63mm/hr",
                "200 of 472 stations flooded in recent years",
                "Sandy (2012) inundated 150 stations, causing $5B in damage",
                "75+ flood disruptions from Jan 2020 to Sep 2025",
                "Underground platforms trap heat; 6 heatwaves in 18 months (2024-25)",
            ],
            "preparedness": [
                "$7.6B post-Sandy federal coastal hardening program",
                "2024 Climate Resilience Roadmap: $6B over 10 years",
                "$1.5B committed 2025-29: $700M flood + $800M Metro-North",
                "Geothermal cooling, platform fans, white rails under exploration",
            ],
            "verdict": "Biggest spender, biggest exposure. Leading on planning, lagging on execution speed.",
        },
        "Tokyo": {
            "system": "Metro / Toei", "flood": "Low (managed)", "heat": "Moderate",
            "flood_score": 3.0, "heat_score": 5.5, "color": GREEN,
            "resilience_spend": 4200, "stations": 285, "age": 97,
            "key_threats": [
                "Typhoon and intense rainfall risk exceeding 75mm/hr increasing",
                "Record 41.1 °C nationally (2020); heatstroke emergencies surging",
                "Underground flooding risk to older metro lines during extreme events",
            ],
            "preparedness": [
                "G-Cans underground reservoir: prevents ~$930M flood damage annually",
                "Triggered ~7 times/year; $4.1B expansion launching 2024",
                "World's first certified resilience bond ($330M, Oct 2025)",
                "Full AC across entire metro network",
            ],
            "verdict": "Gold standard. Invests proactively, designs for worst case, finances innovatively.",
        },
        "London": {
            "system": "TfL / Underground", "flood": "High", "heat": "Extreme",
            "flood_score": 7.5, "heat_score": 9.0, "color": TEAL,
            "resilience_spend": 2800, "stations": 272, "age": 161,
            "key_threats": [
                "477 climate risks identified in 2024 ARP4 assessment",
                "52% of risks are precipitation-related",
                "Deep tube tunnels exceed 30 °C; London hit 40 °C in 2022",
                "Heat causes signal failures and track faults in moisture-sensitive clay",
            ],
            "preparedness": [
                "First Climate Adaptation Plan (2023); budget doubled to $2.5M",
                "AC trains for Piccadilly line from 2025; cooling panels at Holborn",
                "Thames Estuary 2100 plan: 330km flood assets, 1.4M residents protected",
                "London Climate Resilience Review: 50 recommendations pledged",
            ],
            "verdict": "Heat risk rivals NYC. Systematic planning, but oldest system and underfunded.",
        },
        "Paris": {
            "system": "RATP", "flood": "High", "heat": "High",
            "flood_score": 7.0, "heat_score": 7.5, "color": ORANGE,
            "resilience_spend": 800, "stations": 303, "age": 124,
            "key_threats": [
                "Seine flooding threatens central Metro lines",
                "Most tunnels not flood-gated; 1910-level floods could submerge large areas",
                "Only 10% green space; zinc rooftops trap heat",
                "Most Metro cars and stations lack AC",
            ],
            "preparedness": [
                "No comprehensive metro-specific climate roadmap",
                "Relies on citywide Seine flood plans",
                "Olympic 2024 prompted heat investment: green space, roof gardens",
                "New AC rolling stock being phased in slowly",
            ],
            "verdict": "Cautionary tale. Similar exposure to NYC but no transit-specific climate strategy.",
        },
        "Singapore": {
            "system": "SMRT / MRT", "flood": "Moderate", "heat": "High",
            "flood_score": 5.0, "heat_score": 8.5, "color": BLUE,
            "resilience_spend": 12500, "stations": 187, "age": 37,
            "key_threats": [
                "Extreme daily rainfall (99th pct) projected to rise 6-92%",
                "Very hot days (>35 °C) projected 4/yr to 351/yr by 2100",
                "Warm nights nearly year-round; tropical humidity compounds heat stress",
            ],
            "preparedness": [
                "Marina Barrage (S$226M, 2008) for downtown flood control",
                "S$125M Coastal Research Programme (2023)",
                "2026 declared Year of Climate Adaptation; $100B long-term coastal budget",
                "MRT fully air-conditioned; newest system of all comparators",
            ],
            "verdict": "Highest per-station spending. Newest system, most proactive long-term planning.",
        },
        "Boston": {
            "system": "MBTA", "flood": "High", "heat": "Moderate",
            "flood_score": 7.0, "heat_score": 6.0, "color": YELLOW,
            "resilience_spend": 1200, "stations": 155, "age": 127,
            "key_threats": [
                "Oldest subway in the US (1897); infrastructure at end of life",
                "Coastal storm surge threatens Blue Line and waterfront stations",
                "2022 flooding shut down entire Orange Line for 30 days",
                "Sea level rise of 1.5 ft by 2050 puts downtown tunnels at risk",
            ],
            "preparedness": [
                "$2.3B Resilience Program announced 2024",
                "Flood barriers at Aquarium and Airport stations",
                "Climate Change Vulnerability Assessment completed 2023",
                "Emergency slow zones and speed restrictions during heat events",
            ],
            "verdict": "Oldest US system, chronically underfunded. High awareness but slow execution.",
        },
        "Washington DC": {
            "system": "WMATA / Metro", "flood": "Moderate", "heat": "High",
            "flood_score": 5.5, "heat_score": 7.0, "color": PINK,
            "resilience_spend": 2100, "stations": 98, "age": 49,
            "key_threats": [
                "Potomac River flooding threatens low-lying stations",
                "Intense summer heat (35+ °C days increasing 40% since 2000)",
                "Aging electrical systems vulnerable to storm-related outages",
                "Flash floods overwhelmed drainage at 12 stations in 2023-24",
            ],
            "preparedness": [
                "$1.6B SafeTrack infrastructure renewal program",
                "Flood sensor network installed at 40 vulnerable stations",
                "Heat mitigation plan with cooling centers at major hubs",
                "Federal funding advantage as the nation's capital system",
            ],
            "verdict": "Younger system but politically complex. Federal funding is a double-edged sword.",
        },
    }

    # --- Scatter plot: flood vs heat ---
    city_names = list(CITIES.keys())
    flood_scores = [CITIES[c]["flood_score"] for c in city_names]
    heat_scores = [CITIES[c]["heat_score"] for c in city_names]
    city_colors = [CITIES[c]["color"] for c in city_names]
    city_sizes = [22 if c == "New York City" else 14 for c in city_names]

    fig_global = go.Figure()
    for i, c in enumerate(city_names):
        fig_global.add_trace(go.Scatter(
            x=[flood_scores[i]], y=[heat_scores[i]],
            mode="markers+text",
            marker=dict(color=city_colors[i], size=city_sizes[i], opacity=0.9,
                        line=dict(width=2, color=TEXT if c == "New York City" else "rgba(0,0,0,0)")),
            text=[c], textposition="top center",
            textfont=dict(size=11, color=TEXT),
            name=c, showlegend=False,
            hovertemplate=f"<b>{c}</b><br>Flood: {flood_scores[i]}/10<br>Heat: {heat_scores[i]}/10<extra></extra>",
        ))
    # Quadrant shading
    fig_global.add_shape(type="rect", x0=7, y0=7, x1=10.5, y1=10.5,
                         fillcolor="rgba(240,112,113,0.06)", line=dict(width=0))
    fig_global.add_shape(type="rect", x0=0, y0=0, x1=5, y1=5,
                         fillcolor="rgba(126,179,106,0.06)", line=dict(width=0))
    fig_global.add_hline(y=7, line_dash="dot", line_color=MUTED, opacity=0.4)
    fig_global.add_vline(x=7, line_dash="dot", line_color=MUTED, opacity=0.4)
    fig_global.add_annotation(x=9, y=10, text="HIGHEST RISK", showarrow=False,
                              font=dict(size=9, color=RED, family="'JetBrains Mono', monospace"))
    fig_global.add_annotation(x=2.5, y=3.5, text="BEST MANAGED", showarrow=False,
                              font=dict(size=9, color=GREEN, family="'JetBrains Mono', monospace"))
    styled_fig(fig_global, height=480, showlegend=False,
               title=dict(text="Global Transit Climate Risk: Flood vs Heat Exposure",
                          font=dict(size=14, color=TEXT_DIM)),
               xaxis=dict(title="Flood Risk Score", range=[0, 10.5], gridcolor=MUTED, dtick=2),
               yaxis=dict(title="Heat Risk Score", range=[0, 10.5], gridcolor=MUTED, dtick=2))
    st.plotly_chart(fig_global, use_container_width=True)

    # --- Interactive city selector ---
    city_choice = st.radio(
        "Select city to compare with NYC",
        [c for c in city_names if c != "New York City"],
        horizontal=True, index=0,
    )
    nyc = CITIES["New York City"]
    comp = CITIES[city_choice]

    # Side-by-side radar chart
    categories = ["Flood Risk", "Heat Risk", "System Age", "Stations", "Resilience\nSpend ($M)"]
    # Normalize to 0-10
    max_age = max(c["age"] for c in CITIES.values())
    max_stations = max(c["stations"] for c in CITIES.values())
    max_spend = max(c["resilience_spend"] for c in CITIES.values())

    nyc_vals = [nyc["flood_score"], nyc["heat_score"],
                nyc["age"]/max_age*10, nyc["stations"]/max_stations*10,
                nyc["resilience_spend"]/max_spend*10]
    comp_vals = [comp["flood_score"], comp["heat_score"],
                 comp["age"]/max_age*10, comp["stations"]/max_stations*10,
                 comp["resilience_spend"]/max_spend*10]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=nyc_vals + [nyc_vals[0]], theta=categories + [categories[0]],
        fill="toself", fillcolor="rgba(240,112,113,0.15)",
        line=dict(color=RED, width=2), name="New York City",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=comp_vals + [comp_vals[0]], theta=categories + [categories[0]],
        fill="toself", fillcolor=f"rgba({int(comp['color'][1:3],16)},{int(comp['color'][3:5],16)},{int(comp['color'][5:7],16)},0.15)",
        line=dict(color=comp["color"], width=2), name=city_choice,
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 10], gridcolor=MUTED,
                            tickfont=dict(size=9, color=TEXT_DIM)),
            angularaxis=dict(gridcolor=MUTED, tickfont=dict(size=11, color=TEXT)),
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Space Grotesk', sans-serif", color=TEXT),
        height=420, showlegend=True,
        legend=dict(orientation="h", y=-0.1, font=dict(size=12),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=60, t=40, b=60),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Side-by-side detail cards
    col_nyc, col_comp = st.columns(2)
    with col_nyc:
        st.markdown(f"""
        <div style="padding:1.5rem; background:{BG_CARD}; border-radius:8px;
                    border-top:3px solid {RED};">
          <p style="color:{RED};font-weight:600;font-size:16px;margin-bottom:4px;">New York City</p>
          <p style="color:{TEXT_DIM};font-size:12px;margin-bottom:12px;">
            {nyc['system']} &nbsp;|&nbsp; {nyc['stations']} stations &nbsp;|&nbsp; {nyc['age']} years old
            &nbsp;|&nbsp; ${nyc['resilience_spend']/1e3:.1f}B resilience spend</p>
          <p style="color:{TEXT};font-weight:500;font-size:13px;margin-bottom:6px;">Key Threats</p>
          <ul style="font-size:12px;color:{TEXT_DIM};line-height:1.7;margin-bottom:12px;">
            {"".join(f"<li>{t}</li>" for t in nyc['key_threats'])}
          </ul>
          <p style="color:{TEXT};font-weight:500;font-size:13px;margin-bottom:6px;">Preparedness</p>
          <ul style="font-size:12px;color:{TEXT_DIM};line-height:1.7;margin-bottom:12px;">
            {"".join(f"<li>{p}</li>" for p in nyc['preparedness'])}
          </ul>
          <p style="font-size:12px;color:{RED};font-style:italic;margin:0;">{nyc['verdict']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_comp:
        st.markdown(f"""
        <div style="padding:1.5rem; background:{BG_CARD}; border-radius:8px;
                    border-top:3px solid {comp['color']};">
          <p style="color:{comp['color']};font-weight:600;font-size:16px;margin-bottom:4px;">{city_choice}</p>
          <p style="color:{TEXT_DIM};font-size:12px;margin-bottom:12px;">
            {comp['system']} &nbsp;|&nbsp; {comp['stations']} stations &nbsp;|&nbsp; {comp['age']} years old
            &nbsp;|&nbsp; ${comp['resilience_spend']/1e3:.1f}B resilience spend</p>
          <p style="color:{TEXT};font-weight:500;font-size:13px;margin-bottom:6px;">Key Threats</p>
          <ul style="font-size:12px;color:{TEXT_DIM};line-height:1.7;margin-bottom:12px;">
            {"".join(f"<li>{t}</li>" for t in comp['key_threats'])}
          </ul>
          <p style="color:{TEXT};font-weight:500;font-size:13px;margin-bottom:6px;">Preparedness</p>
          <ul style="font-size:12px;color:{TEXT_DIM};line-height:1.7;margin-bottom:12px;">
            {"".join(f"<li>{p}</li>" for p in comp['preparedness'])}
          </ul>
          <p style="font-size:12px;color:{comp['color']};font-style:italic;margin:0;">{comp['verdict']}</p>
        </div>
        """, unsafe_allow_html=True)

    divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — The weather is getting worse (moved to later)
    # ─────────────────────────────────────────────────────────────────────────
    section("05", "Five years of weather data. The trend is obvious.",
            "More heavy rain days, more extreme heat days, more of both in the same "
            "month. The 2020 to 2024 record is not an anomaly. It is the baseline now.")

    kpi_strip([
        ("Period", "2020 - 2024", "1,827 days"),
        ("Heavy rain days", f"{len(rain)}", f"{len(rain)/len(sys_df):.1%} of all days"),
        ("Extreme heat days", f"{len(heat)}", f"{len(heat)/len(sys_df):.1%} of all days"),
        ("Rain uplift", f"+{rain_uplift:.0f}%", "more incidents on rain days"),
    ])

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
    # SECTION 6 — When it rains, the subway breaks
    # ─────────────────────────────────────────────────────────────────────────
    section("06", "Rain days mean broken service.",
            "MTA data from 2020 to 2024: heavy rain days produce {:.0f}% more incidents "
            "than dry days. Not a surprise to anyone who commutes, but the "
            "correlation is tighter than most people assume.".format(rain_uplift))

    fig_cond = go.Figure()
    cond_labels = ["Normal days", "Heavy rain days", "Extreme heat days"]
    cond_vals   = [norm_avg, rain_avg, heat_avg]
    cond_colors = [TEXT_DIM, BLUE, RED]
    fig_cond.add_trace(go.Bar(
        x=cond_labels, y=cond_vals, marker_color=cond_colors,
        text=[f"{v:.0f}" for v in cond_vals], textposition="outside",
        textfont=dict(color=TEXT, size=14, family="'JetBrains Mono', monospace"),
    ))
    styled_fig(fig_cond, height=340, showlegend=False,
               title=dict(text="Average Daily Incidents by Weather Condition",
                          font=dict(size=14, color=TEXT_DIM)),
               yaxis=dict(title="Incidents per Day", gridcolor=MUTED))
    st.plotly_chart(fig_cond, use_container_width=True)

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

    # Correlation scatters
    st.markdown(f"""
    <div class="section-label" style="padding-top:1.5rem">CORRELATION</div>
    <div class="section-title" style="font-size:24px">Quantifying the link</div>
    <div class="section-body">
      Statistical analysis confirms what all of us who have had the misfortune
      of taking the trains on rainy days already know. More rain means more delays.
      Higher temps mean more breakdowns. The correlations are significant and
      consistent across all lines.
    </div>
    """, unsafe_allow_html=True)

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
                   title=dict(text="Correlation Matrix: Weather vs Transit Disruption",
                              font=dict(size=14, color=TEXT_DIM)),
                   xaxis=dict(tickangle=30, tickfont=dict(size=10)))
        st.plotly_chart(fig_corr, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        sub2 = sys_df[["precip_mm","total_incidents"]].dropna()
        r_val, _ = pearsonr(sub2["precip_mm"], sub2["total_incidents"])
        coefs = np.polyfit(sub2["precip_mm"], sub2["total_incidents"], 1)
        x_ln = np.linspace(sub2["precip_mm"].min(), sub2["precip_mm"].max(), 50)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=sub2["precip_mm"], y=sub2["total_incidents"], mode="markers",
            marker=dict(color=BLUE, opacity=0.25, size=4),
        ))
        fig_sc.add_trace(go.Scatter(
            x=x_ln, y=np.polyval(coefs, x_ln), mode="lines",
            line=dict(color=TEXT, width=2, dash="dash"),
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
    # SECTION 7 — Station risk map
    # ─────────────────────────────────────────────────────────────────────────
    section("07", "496 stations, three risk dimensions.",
            "Flood risk, heat risk, economic exposure. Each scored 0 to 10 for every "
            "station in the system. Toggle the map and see which lines and "
            "boroughs carry the most risk.")

    DIM_LABELS = {
        "flood_risk": "Flood Risk",
        "heat_risk": "Heat Risk",
        "economic_exposure": "Est. Annual Losses",
    }
    dim = st.radio(
        "Risk dimension",
        ["flood_risk", "heat_risk", "economic_exposure"],
        horizontal=True, index=0,
        format_func=lambda x: DIM_LABELS.get(x, x),
    )
    dim_colors = {
        "flood_risk":        [[0, BG_HINT], [0.5, TEAL], [1, BLUE]],
        "heat_risk":         [[0, BG_HINT], [0.5, ORANGE], [1, RED]],
        "economic_exposure": [[0, BG_HINT], [0.5, YELLOW], [1, ORANGE]],
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
                bgcolor="rgba(0,0,0,0)", outlinewidth=0, len=0.6,
            ),
        ),
        text=[
            f"<b>{r['station']}</b><br>"
            f"<span style='color:{TEXT_DIM}'>Line {r['primary_line']}  ·  {r['borough']}  ·  {r['structure']}</span><br><br>"
            f"<span style='color:{BLUE}'>Flood Risk</span>  {r['flood_risk']:.1f} / 10<br>"
            f"<span style='color:{RED}'>Heat Risk</span>   {r['heat_risk']:.1f} / 10<br>"
            f"<span style='color:{ORANGE}'>Est. Losses</span>  {r['economic_exposure']:.1f} / 10"
            for _, r in stn_df.iterrows()
        ],
        hoverinfo="text",
    ))
    fig_map.update_layout(
        mapbox=dict(style="carto-darkmatter",
                    center=dict(lat=40.72, lon=-73.95), zoom=10.2),
        height=620, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Risk dimensions grouped bar
    comps = lrisk_df.sort_values("flood_risk", ascending=False)
    fig_comp = go.Figure()
    for comp, color, name in [
        ("flood_risk", TEAL, "Flood Risk"),
        ("heat_risk", RED, "Heat Risk"),
        ("economic_exposure", YELLOW, "Est. Annual Losses"),
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
    # SECTION 8 — Mitigation Strategy
    # ─────────────────────────────────────────────────────────────────────────
    section("08", "$5.15B to cut annual losses by 84%.",
            "React by 2030, adapt by 2040, transform by 2050. "
            "Spend $5.15B over 35 years, avoid $3.1B in losses every year after that.")

    # --- Headline KPIs ---
    kpi_strip([
        ("2060 loss if we do nothing", "$3.1B/yr", "expected annual loss"),
        ("After mitigation", "$510M/yr", "84% reduction"),
        ("Total investment", "$5.15B", "over 35 years"),
        ("ROI", "6:1", "$3.1B avoided per year"),
    ])

    # --- Three phases ---
    st.markdown(f"""
    <p style="color:{TEXT_DIM};font-size:11px;font-family:'JetBrains Mono',monospace;
       letter-spacing:0.12em;text-transform:uppercase;margin:2rem 0 1rem 0;">
       Phased implementation</p>
    """, unsafe_allow_html=True)

    phase_col1, phase_col2, phase_col3 = st.columns(3)
    with phase_col1:
        st.markdown(f"""
        <div style="padding:1.5rem; background:{BG_CARD}; border-radius:8px;
                    border-top:3px solid {RED};" class="fade-up">
          <p style="color:{RED};font-weight:700;font-size:18px;margin:0 0 2px 0;">REACT</p>
          <p style="color:{TEXT_DIM};font-size:12px;margin:0 0 16px 0;
             font-family:'JetBrains Mono',monospace;">Now – 2030 &nbsp;&middot;&nbsp; $930M</p>
          <p style="color:{TEXT};font-size:13px;font-weight:500;margin:0 0 6px 0;">Flood Defense</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 12px 0;">
            Emergency pumps at 50 stations. Flood sensors at 200 stations.
            Barriers deployable in under 15 minutes.</p>
          <p style="color:{TEXT};font-size:13px;font-weight:500;margin:0 0 6px 0;">Heat Resilience</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 12px 0;">
            Fans + misting at 30 hottest stations. Heat protocols and speed limits.
            Sensor network deployed.</p>
          <p style="color:{TEXT};font-size:13px;font-weight:500;margin:0 0 6px 0;">Power & Cascades</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 12px 0;">
            Backup power at all 26 signal nodes. Cascade detection software.
            Auto-isolation protocol.</p>
          <p style="color:{TEXT};font-size:13px;font-weight:500;margin:0 0 6px 0;">Finance & Governance</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0;">
            $330M Resilience Bond (congestion pricing). Climate Risk Office established.
            TCFD disclosure committed.</p>
        </div>
        """, unsafe_allow_html=True)
    with phase_col2:
        st.markdown(f"""
        <div style="padding:1.5rem; background:{BG_CARD}; border-radius:8px;
                    border-top:3px solid {ORANGE};" class="fade-up fade-up-d1">
          <p style="color:{ORANGE};font-weight:700;font-size:18px;margin:0 0 2px 0;">ADAPT</p>
          <p style="color:{TEXT_DIM};font-size:12px;margin:0 0 16px 0;
             font-family:'JetBrains Mono',monospace;">2031 – 2040 &nbsp;&middot;&nbsp; $1.95B</p>
          <p style="color:{TEXT};font-size:13px;font-weight:500;margin:0 0 6px 0;">Flood Defense</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 12px 0;">
            Raise entrances +18" at 120 coastal stations.
            40% runoff reduction via green infrastructure.</p>
          <p style="color:{TEXT};font-size:13px;font-weight:500;margin:0 0 6px 0;">Heat Resilience</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 12px 0;">
            Solar cooling at 150 underground stations. Green roofs at 80 stations.
            Ventilation shaft redesign.</p>
          <p style="color:{TEXT};font-size:13px;font-weight:500;margin:0 0 6px 0;">Power & Cascades</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 12px 0;">
            Microgrids at 25 transfer hubs. BESS with 4-hour island capacity.
            Solar canopy at 40 stations.</p>
          <p style="color:{TEXT};font-size:13px;font-weight:500;margin:0 0 6px 0;">Finance & Governance</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0;">
            $1B+ Green Bond issuance. Parametric insurance with flood trigger.
            DEP/ConEd/OEM MOUs live.</p>
        </div>
        """, unsafe_allow_html=True)
    with phase_col3:
        st.markdown(f"""
        <div style="padding:1.5rem; background:{BG_CARD}; border-radius:8px;
                    border-top:3px solid {TEAL};" class="fade-up fade-up-d2">
          <p style="color:{TEAL};font-weight:700;font-size:18px;margin:0 0 2px 0;">TRANSFORM</p>
          <p style="color:{TEXT_DIM};font-size:12px;margin:0 0 16px 0;
             font-family:'JetBrains Mono',monospace;">2041 – 2050 &nbsp;&middot;&nbsp; $4.0B</p>
          <p style="color:{TEXT};font-size:13px;font-weight:500;margin:0 0 6px 0;">Flood Defense</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 12px 0;">
            Convert 6 underground segments to elevated rail (G-Cans model).
            Underground reservoir at 3 hubs.</p>
          <p style="color:{TEXT};font-size:13px;font-weight:500;margin:0 0 6px 0;">Heat Resilience</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 12px 0;">
            Geothermal pilot at 10 stations. Full HVAC at 200 underground stations.
            White-rail treatment system-wide.</p>
          <p style="color:{TEXT};font-size:13px;font-weight:500;margin:0 0 6px 0;">Power & Cascades</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 12px 0;">
            Full network islanding with 72-hour capacity.
            AI load-shedding in real-time. Redundant control centers.</p>
          <p style="color:{TEXT};font-size:13px;font-weight:500;margin:0 0 6px 0;">Finance & Governance</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0;">
            Blended finance platform. Station risk score tied to budget formula.
            Performance bond structure.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Mitigation impact chart ---
    mit_years = [2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]
    loss_no_action = [0.48, 0.72, 1.05, 1.50, 1.95, 2.40, 2.75, 3.10]
    loss_mitigated = [0.48, 0.60, 0.55, 0.52, 0.51, 0.51, 0.51, 0.51]

    fig_mit = go.Figure()
    fig_mit.add_trace(go.Scatter(
        x=mit_years, y=loss_no_action, mode="lines+markers",
        name="No action", line=dict(color=RED, width=3),
        marker=dict(size=8, color=RED, line=dict(width=2, color=BG)),
        fill="tozeroy", fillcolor="rgba(240,112,113,0.08)",
        hovertemplate="No action: $%{y:.2f}B/yr<extra></extra>",
    ))
    fig_mit.add_trace(go.Scatter(
        x=mit_years, y=loss_mitigated, mode="lines+markers",
        name="With mitigation", line=dict(color=GREEN, width=3),
        marker=dict(size=8, color=GREEN, line=dict(width=2, color=BG)),
        fill="tozeroy", fillcolor="rgba(126,179,106,0.08)",
        hovertemplate="Mitigated: $%{y:.2f}B/yr<extra></extra>",
    ))
    # Shade the gap
    fig_mit.add_trace(go.Scatter(
        x=mit_years + mit_years[::-1],
        y=loss_no_action + loss_mitigated[::-1],
        fill="toself", fillcolor="rgba(240,112,113,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_mit.add_annotation(
        x=2055, y=(loss_no_action[6] + loss_mitigated[6]) / 2,
        text="<b>$2.6B/yr avoided</b>",
        font=dict(size=12, color=TEXT, family="'JetBrains Mono', monospace"),
        showarrow=True, arrowhead=0, arrowcolor=TEXT_DIM, arrowwidth=1,
        ax=-60, ay=0, bgcolor=BG_CARD, borderpad=4,
    )
    styled_fig(fig_mit, height=420,
               title=dict(text="Expected Annual Loss: No Action vs Mitigation Strategy",
                          font=dict(size=14, color=TEXT_DIM)),
               xaxis=dict(title="", gridcolor=MUTED, dtick=5),
               yaxis=dict(title="Expected Annual Loss ($B)", gridcolor=MUTED),
               legend=dict(orientation="h", y=-0.12, font=dict(size=11),
                           bgcolor="rgba(0,0,0,0)"),
               margin=dict(l=48, r=40, t=56, b=48))
    st.plotly_chart(fig_mit, use_container_width=True)

    # --- 30/60/90 day execution timeline ---
    st.markdown(f"""
    <p style="color:{TEXT_DIM};font-size:11px;font-family:'JetBrains Mono',monospace;
       letter-spacing:0.12em;text-transform:uppercase;margin:1.5rem 0 1rem 0;">
       Immediate execution milestones</p>
    <div style="display:flex; gap:1px; background:{MUTED}; border-radius:4px; overflow:hidden;
                margin-bottom:2rem;" class="fade-up">
      <div style="flex:1; background:{BG_CARD}; padding:1.2rem 1rem;">
        <p style="color:{RED};font-weight:700;font-size:14px;margin:0 0 4px 0;">30 days</p>
        <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0;">
          Appoint Chief Resilience Officer. Issue Resilience Bond RFP.</p>
      </div>
      <div style="flex:1; background:{BG_CARD}; padding:1.2rem 1rem;">
        <p style="color:{ORANGE};font-weight:700;font-size:14px;margin:0 0 4px 0;">60 days</p>
        <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0;">
          Cascade protocol live. DEP/ConEd MOUs signed.</p>
      </div>
      <div style="flex:1; background:{BG_CARD}; padding:1.2rem 1rem;">
        <p style="color:{GREEN};font-weight:700;font-size:14px;margin:0 0 4px 0;">90 days</p>
        <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0;">
          Fan pilot at 10 stations. Backup power pilot at 5 signal nodes.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 9 — Conclusion & Priorities
    # ─────────────────────────────────────────────────────────────────────────
    section("09", "What to do about it.",
            "The bottleneck is not technical solutions. It is governance, funding "
            "coordination, and execution speed.")

    # --- Two-column: Key Conclusions + Recommended Next Steps ---
    col_conc, col_next = st.columns(2)
    with col_conc:
        st.markdown(f"""
        <div style="padding:1.5rem; background:{BG_CARD}; border-radius:8px;
                    border-top:3px solid {PURPLE};" class="fade-up">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
            <p style="color:{TEXT};font-weight:700;font-size:18px;margin:0;">Key Conclusion</p>
            <span style="background:{PURPLE};color:{BG};font-size:10px;font-weight:700;
                         padding:4px 10px;border-radius:3px;font-family:'JetBrains Mono',monospace;
                         letter-spacing:0.06em;">WHY THIS MATTERS</span>
          </div>
          <div style="border-left:3px solid {MUTED}; padding-left:14px; margin-bottom:14px;">
            <p style="color:{PURPLE};font-weight:600;font-size:12px;margin:0 0 4px 0;">Risk</p>
            <p style="color:{TEXT_DIM};font-size:13px;line-height:1.6;margin:0;">
              MTA's climate exposure is becoming a balance-sheet problem,
              not just an operations issue.</p>
          </div>
          <div style="border-left:3px solid {MUTED}; padding-left:14px; margin-bottom:14px;">
            <p style="color:{ORANGE};font-weight:600;font-size:12px;margin:0 0 4px 0;">Evidence</p>
            <p style="color:{TEXT_DIM};font-size:13px;line-height:1.6;margin:0;">
              Weather disruption already costs about $2.5M per year today; under wildcard
              scenarios, expected annual losses could reach $3.1B by 2060.</p>
          </div>
          <div style="border-left:3px solid {MUTED}; padding-left:14px; margin-bottom:14px;">
            <p style="color:{GREEN};font-weight:600;font-size:12px;margin:0 0 4px 0;">Economics</p>
            <p style="color:{TEXT_DIM};font-size:13px;line-height:1.6;margin:0;">
              A $5.15B multi-decade adaptation plan can reduce projected 2060 annual
              losses from $3.1B to about $510M per year.</p>
          </div>
          <div style="border-left:3px solid {MUTED}; padding-left:14px; margin-bottom:14px;">
            <p style="color:{RED};font-weight:600;font-size:12px;margin:0 0 4px 0;">Priority</p>
            <p style="color:{TEXT_DIM};font-size:13px;line-height:1.6;margin:0;">
              The question is not whether MTA can afford to adapt,
              but whether it can afford not to.</p>
          </div>
          <div style="border-left:3px solid {MUTED}; padding-left:14px;">
            <p style="color:{TEAL};font-weight:600;font-size:12px;margin:0 0 4px 0;">Execution</p>
            <p style="color:{TEXT_DIM};font-size:13px;line-height:1.6;margin:0;">
              Move from reactive repair toward station-level risk management,
              financed and governed as a long-term resilience program.</p>
          </div>
          <p style="font-size:12px;color:{TEAL};font-style:italic;margin:16px 0 0 0;">
            Main takeaway: adaptation is financially rational and operationally necessary.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_next:
        st.markdown(f"""
        <div style="padding:1.5rem; background:{BG_CARD}; border-radius:8px;
                    border-top:3px solid {ORANGE};" class="fade-up fade-up-d1">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
            <p style="color:{TEXT};font-weight:700;font-size:18px;margin:0;">Recommended Next Steps</p>
            <span style="background:{ORANGE};color:{BG};font-size:10px;font-weight:700;
                         padding:4px 10px;border-radius:3px;font-family:'JetBrains Mono',monospace;
                         letter-spacing:0.06em;">ACTIONABLE NOW</span>
          </div>
          <div style="border-left:3px solid {MUTED}; padding-left:14px; margin-bottom:14px;">
            <p style="color:{BLUE};font-weight:600;font-size:12px;margin:0 0 4px 0;">Capital</p>
            <p style="color:{TEXT_DIM};font-size:13px;line-height:1.6;margin:0;">
              Prioritize capex by avoided loss per dollar, starting with
              the highest-risk flood and heat stations.</p>
          </div>
          <div style="border-left:3px solid {MUTED}; padding-left:14px; margin-bottom:14px;">
            <p style="color:{GREEN};font-weight:600;font-size:12px;margin:0 0 4px 0;">Finance</p>
            <p style="color:{TEXT_DIM};font-size:13px;line-height:1.6;margin:0;">
              Layer resilience bonds, green bonds, federal grants,
              and parametric insurance to diversify funding.</p>
          </div>
          <div style="border-left:3px solid {MUTED}; padding-left:14px; margin-bottom:14px;">
            <p style="color:{PURPLE};font-weight:600;font-size:12px;margin:0 0 4px 0;">Governance</p>
            <p style="color:{TEXT_DIM};font-size:13px;line-height:1.6;margin:0;">
              Create a dedicated Climate Risk Office / Resilience PMO
              linking data, budgeting, engineering, and emergency response.</p>
          </div>
          <div style="border-left:3px solid {MUTED}; padding-left:14px; margin-bottom:14px;">
            <p style="color:{ORANGE};font-weight:600;font-size:12px;margin:0 0 4px 0;">Targeting</p>
            <p style="color:{TEXT_DIM};font-size:13px;line-height:1.6;margin:0;">
              Use station-level flood risk, heat risk, and economic exposure
              scores to allocate funding more efficiently.</p>
          </div>
          <div style="border-left:3px solid {MUTED}; padding-left:14px;">
            <p style="color:{RED};font-weight:600;font-size:12px;margin:0 0 4px 0;">Timing</p>
            <p style="color:{TEXT_DIM};font-size:13px;line-height:1.6;margin:0;">
              Act before the next storm season: start with
              30 / 60 / 90-day execution milestones and pilot projects.</p>
          </div>
          <p style="font-size:12px;color:{ORANGE};font-style:italic;margin:16px 0 0 0;">
            Execution bottleneck: governance and funding coordination, not lack of technical solutions.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Priority matrix: 3 columns ---
    st.markdown(f"""
    <p style="color:{TEXT_DIM};font-size:11px;font-family:'JetBrains Mono',monospace;
       letter-spacing:0.12em;text-transform:uppercase;margin:2.5rem 0 1rem 0;">
       Priority matrix</p>
    """, unsafe_allow_html=True)

    pri1, pri2, pri3 = st.columns(3)
    with pri1:
        st.markdown(f"""
        <div style="padding:1.5rem; background:{BG_CARD}; border-radius:8px;
                    border-top:3px solid {GREEN};" class="fade-up">
          <p style="color:{GREEN};font-weight:700;font-size:14px;letter-spacing:0.06em;
                    text-transform:uppercase;margin:0 0 16px 0;">Engage now</p>
          <p style="color:{GREEN};font-weight:600;font-size:13px;margin:0 0 4px 0;">Climate Risk Office</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 14px 0;">
            Stand up a cross-functional unit to connect risk data,
            capital planning, insurance, and operations.</p>
          <p style="color:{GREEN};font-weight:600;font-size:13px;margin:0 0 4px 0;">Station-Level Prioritization</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 14px 0;">
            Use flood, heat, and economic exposure scores to rank
            projects by avoided loss rather than political visibility.</p>
          <p style="color:{GREEN};font-weight:600;font-size:13px;margin:0 0 4px 0;">Pilot Projects</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0;">
            Launch high-visibility pilots on pumps, cooling, sensors,
            and backup power before the next storm season.</p>
        </div>
        """, unsafe_allow_html=True)
    with pri2:
        st.markdown(f"""
        <div style="padding:1.5rem; background:{BG_CARD}; border-radius:8px;
                    border-top:3px solid {YELLOW};" class="fade-up fade-up-d1">
          <p style="color:{YELLOW};font-weight:700;font-size:14px;letter-spacing:0.06em;
                    text-transform:uppercase;margin:0 0 16px 0;">Watch closely</p>
          <p style="color:{YELLOW};font-weight:600;font-size:13px;margin:0 0 4px 0;">Funding Mix</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 14px 0;">
            Track resilience bond feasibility, green bond sizing,
            and the availability of federal or state co-funding.</p>
          <p style="color:{YELLOW};font-weight:600;font-size:13px;margin:0 0 4px 0;">Data Quality</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 14px 0;">
            Improve station-level weather and incident data to
            sharpen risk scoring and future capital allocation.</p>
          <p style="color:{YELLOW};font-weight:600;font-size:13px;margin:0 0 4px 0;">Technology Integration</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0;">
            Monitor how flood defense, cooling, microgrids,
            and control systems interact across the network.</p>
        </div>
        """, unsafe_allow_html=True)
    with pri3:
        st.markdown(f"""
        <div style="padding:1.5rem; background:{BG_CARD}; border-radius:8px;
                    border-top:3px solid {RED};" class="fade-up fade-up-d2">
          <p style="color:{RED};font-weight:700;font-size:14px;letter-spacing:0.06em;
                    text-transform:uppercase;margin:0 0 16px 0;">Do not underestimate</p>
          <p style="color:{RED};font-weight:600;font-size:13px;margin:0 0 4px 0;">Execution Risk</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 14px 0;">
            The biggest threat is slow implementation: planning
            progress without delivery leaves exposure unchanged.</p>
          <p style="color:{RED};font-weight:600;font-size:13px;margin:0 0 4px 0;">Compound Events</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0 0 14px 0;">
            Flood, heat, and power failures can cascade across
            lines, so single-hazard planning is not enough.</p>
          <p style="color:{RED};font-weight:600;font-size:13px;margin:0 0 4px 0;">Public Value</p>
          <p style="color:{TEXT_DIM};font-size:12px;line-height:1.6;margin:0;">
            Resilience is not only about asset protection; it
            protects mobility access, rider safety, and trust in public transit.</p>
        </div>
        """, unsafe_allow_html=True)

    divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 10 — Data, Sources & Methodology
    # ─────────────────────────────────────────────────────────────────────────
    section("10", "Data, Sources & Methodology", "")

    # --- Data Sources ---
    st.markdown("### Data Sources")

    st.markdown(f"""
**Weather Data** --- Daily precipitation, temperature (max/min), wind speed, and derived
variables sourced from the [Open-Meteo](https://open-meteo.com) historical weather API.
All readings are from the NYC Central Park weather station for the period January 2020
through December 2024 (1,827 days). Derived variables include heat index approximation,
3-day and 7-day rolling precipitation totals, 1-day and 2-day lagged precipitation,
precipitation hours, and a composite weather stress index.

**MTA Performance Data** --- Monthly subway performance totals from
[data.ny.gov](https://data.ny.gov) dataset g937-7k7c (MTA Subway Major Incidents)
and service alert data from dataset 9zbp-wz3y. Monthly totals were disaggregated to
daily resolution using a weather-aware statistical downscaling model that distributes
monthly incident counts proportionally based on daily weather severity. Station
coordinates and line assignments come from the MTA GTFS static feed registry (dataset
39hk-dx4f), covering 496 subway stations across 27 lines.

**Station Risk Scoring** --- Each of the 496 stations is scored from 0 to 10 on three
dimensions: flood risk (FEMA FIRM flood zone exposure, coastal proximity, post-Hurricane
Sandy flood history, and empirically observed incident uplift on heavy rain days), heat
risk (urban heat island intensity, underground station heat trapping characteristics,
and incident uplift on extreme heat days), and economic exposure (ridership-weighted loss
potential normalized to the system maximum using MTA 2023 Blue Book ridership data).
Risk scores are expert-coded at the line level using published vulnerability assessments
and then averaged across all lines serving each station.
""")

    # --- Methodology ---
    st.markdown("### Methodology")

    st.markdown(f"""
**Economic Loss Model** --- Direct economic losses are estimated using the formula:
excess incidents above the normal-weather baseline multiplied by average train load
factor (95 passengers per affected train), multiplied by the probability a rider is
materially delayed (35%), multiplied by the NYC DOT 2023 value of time ($18.50 per hour),
multiplied by excess delay duration per incident. The resulting direct rider-delay cost
is then scaled by a factor of 27.5x to capture the full economic impact, including
operational disruption costs (crew overtime, service rerouting, bus bridge deployment),
accelerated infrastructure wear from water damage and thermal stress, emergency response
expenditures, and long-term ridership erosion effects. This multiplier is calibrated
against transit industry benchmarks from TCRP Report 86 and Cambridge Systematics
transit cost models. The resulting estimate of approximately $2.5 million per year
represents a conservative total-system cost figure.

**Machine Learning Models** --- Two models are trained to predict daily system-wide
incident counts from weather features: Ridge Regression (L2-regularized linear model)
and Random Forest (500 estimators). Both use 5-fold expanding-window time-series
cross-validation to prevent data leakage. The feature set includes daily precipitation,
maximum temperature, binary extreme event flags (heavy rain and extreme heat), the
weather stress composite index, 3-day and 7-day rolling precipitation, 1-day and 2-day
lagged precipitation, and temporal encodings (month, day-of-week, year trend).

**Climate Projections** --- Future loss projections are based on the New York City Panel
on Climate Change (NPCC) 2024 Assessment and NYC Climate Resiliency Design Guidelines
v4.1. Three compound annual growth scenarios are applied to current average annual losses:
Baseline (+2% per year, reflecting historical trend continuation), SSP2-4.5 (+4.5% per
year, moderate emissions pathway), and SSP5-8.5 (+8% per year, high emissions pathway).

**Wildcard Scenario Modeling** --- Wildcard scenarios are modeled using direct cost
estimates from comparable historical events (primarily Hurricane Sandy for the 100-year
flood scenario) combined with expert-assessed annual occurrence probabilities that reflect
the increasing frequency of extreme events under climate change. Expected annual loss
for each scenario equals direct cost multiplied by annual probability.
""")

    # --- Assumptions ---
    st.markdown("### Key Assumptions")

    st.markdown(f"""
1. **Single weather station.** All weather data comes from Central Park. Actual
   conditions at individual subway stations will vary, particularly for stations in
   coastal flood zones or urban heat island hotspots. This is a simplification that
   likely understates localized extremes.

2. **Statistical downscaling of MTA data.** The MTA publishes monthly performance
   totals, not daily counts. We disaggregate to daily resolution using a statistical
   model that allocates incidents proportionally based on daily weather severity. This
   means daily figures are modeled estimates, not direct observations.

3. **Economic loss multiplier.** The 27.5x scaling factor converts direct rider-delay
   costs into full system costs. This is based on transit industry literature (TCRP,
   Cambridge Systematics) but involves judgment. Direct rider-delay costs alone are
   approximately $90,000 per year. The scaled figure of $2.5 million per year captures
   operational, infrastructure, and ridership erosion costs that are real but harder
   to measure precisely.

4. **Expert-coded risk scores.** Station-level risk dimensions are derived from
   line-level expert assessments using published vulnerability literature (MTA Sandy
   After-Action Report, FEMA FIRM maps, NPCC assessments), not from station-level
   sensor data or engineering inspections. Individual station conditions may differ
   from the line-level average.

5. **Climate projection linearity.** Future loss projections assume compound annual
   growth, which produces smooth exponential curves. In reality, climate impacts are
   likely to be nonlinear, with tipping points and threshold effects. Actual trajectories
   depend heavily on emissions policy, adaptation investment, and the frequency of
   tail-risk events.

6. **Model generalization.** ML models are trained on five years of data (2020-2024).
   Performance may degrade for weather conditions outside the observed historical range,
   particularly for the extreme scenarios that matter most for planning purposes.
""")

    # --- Bibliography ---
    st.markdown("### Bibliography")

    st.markdown(f"""
1. MTA. *Sandy After-Action Report*. Metropolitan Transportation Authority, 2013.
2. MTA. *Climate Resilience Roadmap Update*. Metropolitan Transportation Authority, October 2025.
3. NPCC. *New York City Panel on Climate Change 2024 Assessment*. Annals of the New York Academy of Sciences, 2024.
4. NYC Mayor's Office. *Climate Resiliency Design Guidelines*, v4.1. City of New York, 2023.
5. FEMA. *Flood Insurance Rate Maps (FIRM)*, New York City. Federal Emergency Management Agency.
6. NYC DOT. *Value of Time in Transit Analysis*. NYC Department of Transportation, 2023.
7. MTA. *Annual Subway Ridership Data (Blue Book)*. Metropolitan Transportation Authority, 2023.
8. Cambridge Systematics. *TCRP Report 86: Public Transportation Peer-to-Peer Knowledge Sharing*. Transit Cooperative Research Program, Transportation Research Board.
9. Zscheischler, J. et al. "Future climate risk from compound events." *Nature Climate Change*, 8, 469-477, 2018.
10. Rosenzweig, C. et al. "Climate risk information for New York City infrastructure planning." *NPCC Technical Report*, 2019.
11. TfL. *Climate Change Adaptation Plan (ARP4)*. Transport for London, 2024.
12. London Climate Resilience Review. *50 Recommendations for a Climate-Resilient London*. Greater London Authority, July 2024.
13. CMAP. *Transportation Resilience Improvement Plan (TRIP) Assessment*. Chicago Metropolitan Agency for Planning, December 2024.
14. Naturanal. *Tokyo Flood Resilience and G-Cans System*. September 2025.
15. Singapore MSE. *Year of Climate Adaptation 2026*. Ministry of Sustainability and the Environment, March 2026.
""")

    # Footer
    st.markdown("---")
    st.markdown(f"""
<p style="font-size:12px; color:{MUTED}; text-align:center; padding:1rem 0;">
  Columbia University &nbsp;&middot;&nbsp; MS in Climate Finance
  &nbsp;&middot;&nbsp; Climate Risk &nbsp;&middot;&nbsp; 2026
  &nbsp;&nbsp;|&nbsp;&nbsp;
  Built with Python, scikit-learn, Plotly, Streamlit
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://github.com/pbajpai29/Personal-projects" style="color:{MUTED};">Source on GitHub</a>
</p>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
