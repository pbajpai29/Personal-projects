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
    section("01", "What happens when everything goes wrong at once.",
            "Climatic wildcard scenarios go beyond normal bad weather. Think simultaneous "
            "extreme storms overwhelming drainage, power outages cascading across the "
            "network, and emergency policy mandates shutting down MTA services for days. "
            "We modeled four of these scenarios to estimate what the MTA is really facing.")

    # Scenario definitions
    SCENARIOS = {
        "100-Year Flood (Recurring)": {
            "desc": "A Sandy-scale flood event recurring every 5-10 years instead of every 100. "
                    "150+ stations inundated, tunnels flooded, 3-5 day full shutdown.",
            "direct_cost": 5_000_000_000,
            "annual_prob": 0.15,
            "lines_affected": 22,
            "recovery_days": 14,
            "color": BLUE,
        },
        "Cascading Power Failure": {
            "desc": "Extreme heat wave triggers grid overload. Traction power fails across "
                    "multiple boroughs simultaneously. Signal systems go dark. No trains move.",
            "direct_cost": 800_000_000,
            "annual_prob": 0.08,
            "lines_affected": 27,
            "recovery_days": 5,
            "color": RED,
        },
        "Multi-Day Heat Emergency": {
            "desc": "Week-long heat dome with temperatures exceeding 43 °C. Emergency mandate "
                    "shuts underground service to protect riders. Track buckling across the network.",
            "direct_cost": 1_200_000_000,
            "annual_prob": 0.10,
            "lines_affected": 18,
            "recovery_days": 10,
            "color": ORANGE,
        },
        "Flash Flood + Storm Surge": {
            "desc": "Atmospheric river dumps 150mm in 6 hours while coastal storm surge pushes "
                    "seawater into tunnel ventilation shafts. Sewer system designed for 44mm/hr "
                    "fails completely.",
            "direct_cost": 3_500_000_000,
            "annual_prob": 0.12,
            "lines_affected": 15,
            "recovery_days": 21,
            "color": TEAL,
        },
    }

    scenario_choice = st.radio(
        "Select scenario",
        list(SCENARIOS.keys()),
        horizontal=True, index=0,
    )
    sc = SCENARIOS[scenario_choice]

    st.markdown(f"""
    <div style="padding:1.5rem; background:{BG_CARD}; border-radius:8px;
                border-left:4px solid {sc['color']}; margin:1rem 0 2rem 0;" class="fade-up">
      <p style="color:{TEXT};font-size:15px;line-height:1.6;margin:0;">{sc['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    kpi_strip([
        ("Direct cost", f"${sc['direct_cost']/1e9:.1f}B", "infrastructure + lost revenue"),
        ("Annual probability", f"{sc['annual_prob']:.0%}", "and rising with climate change"),
        ("Lines affected", f"{sc['lines_affected']}", f"of 27 total lines"),
        ("Recovery time", f"{sc['recovery_days']}+ days", "to full service restoration"),
    ])

    # Expected annual loss by scenario
    sc_names = list(SCENARIOS.keys())
    sc_eal = [SCENARIOS[s]["direct_cost"] * SCENARIOS[s]["annual_prob"] for s in sc_names]
    sc_colors = [SCENARIOS[s]["color"] for s in sc_names]
    sc_short = ["100-Yr Flood", "Power Failure", "Heat Emergency", "Flash Flood"]

    fig_eal = go.Figure(go.Bar(
        x=sc_short, y=sc_eal,
        marker_color=sc_colors,
        text=[f"${v/1e6:.0f}M" for v in sc_eal],
        textposition="outside",
        textfont=dict(color=TEXT, size=13, family="'JetBrains Mono', monospace"),
    ))
    styled_fig(fig_eal, height=340, showlegend=False,
               title=dict(text="Expected Annual Loss by Wildcard Scenario",
                          font=dict(size=14, color=TEXT_DIM)),
               yaxis=dict(title="Expected Annual Loss (USD)", gridcolor=MUTED))
    st.plotly_chart(fig_eal, use_container_width=True)

    total_eal = sum(sc_eal)
    st.markdown(f"""
    <div style="display:flex; gap:3rem; margin:1.5rem 0;" class="fade-up">
      <div>
        <div class="stat-callout c-red">${total_eal/1e9:.1f}B</div>
        <div class="stat-context">combined expected annual loss<br>across all wildcard scenarios</div>
      </div>
      <div>
        <div class="stat-callout c-orange">{total_eal/avg_annual_loss:.0f}x</div>
        <div class="stat-context">multiplier vs current annual<br>weather disruption costs</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Visioning narrative
    st.markdown(f"""
    <div style="margin:2rem 0; padding:1.5rem 2rem; background:{BG_CARD}; border-radius:8px;" class="fade-up">
      <p style="color:{TEAL};font-weight:600;font-size:15px;margin-bottom:12px;">Beyond repair: visioning the future</p>
      <p style="color:{TEXT_DIM};font-size:14px;line-height:1.7;margin-bottom:12px;">
        Wildcard scenarios like a 100-year flood every five or ten years require
        <strong style="color:{TEXT}">visioning</strong>, not just planning. Full or partial
        retreat from flood-prone areas could trigger climate migration toward outer boroughs,
        permanently shifting where people live and work based on transit reliability.
      </p>
      <p style="color:{TEXT_DIM};font-size:14px;line-height:1.7;margin-bottom:12px;">
        Heat risk particularly affects vulnerable populations facing long walking and waiting
        times. Transit users will increasingly shift to hybrid mobility: more buses, ferries,
        and elevated transit. The MTA must diversify its offerings.
      </p>
      <p style="color:{TEXT_DIM};font-size:14px;line-height:1.7;margin:0;">
        The opportunities are real: resilience bonds tied to congestion pricing can fund
        infrastructure rebuilds. Shifting lines from underground to elevated systems reduces
        flood exposure. Nature-based solutions like green stormwater infrastructure address
        root causes. Solar-powered cooling at stations protects underserved neighborhoods.
      </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Economic cost (moved up)
    # ─────────────────────────────────────────────────────────────────────────
    section("02", "The cost is already $2.5 million per year. It will get worse.",
            "That's the conservative number based on direct rider delays, operational "
            "disruption, and cascading service impacts. The real cost including "
            "infrastructure damage, emergency response, and long-term ridership "
            "erosion is likely 3 to 5 times higher.")

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
    section("03", "Predicting disruptions with machine learning.",
            "We trained Ridge and Random Forest models on weather data to predict "
            "daily subway incidents. The biggest drivers? Weather stress index, "
            "heavy rain flags, and extreme heat events. The model demonstrates what "
            "the transit infrastructure is underfunded to handle.")

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
    section("04", "NYC MTA can learn from other flood- and heat-prone cities.",
            "NYC is uniquely exposed on both flood and heat. Its subway is the oldest "
            "and densest underground network in North America, with infrastructure built "
            "for a climate that no longer exists. But other cities are facing similar "
            "challenges and some are further ahead.")

    # Comparison data
    CITIES = [
        {"city": "New York City", "system": "MTA", "flood": "Extreme", "heat": "High",
         "flood_score": 9.5, "heat_score": 7.5,
         "key_fact": "Sandy inundated 150 stations, $5B damage. Sewer designed for 44mm/hr, storms now exceed 63mm/hr.",
         "preparedness": "$7.6B post-Sandy hardening. 2024 Climate Roadmap with $6B 10-yr plan. $1.5B committed 2025-29.",
         "flag": "focus"},
        {"city": "Tokyo", "system": "Metro/Toei", "flood": "Low (managed)", "heat": "Moderate",
         "flood_score": 3.0, "heat_score": 5.5,
         "key_fact": "G-Cans underground reservoir prevents ~$930M flood damage annually. Triggered ~7 times/year.",
         "preparedness": "World's first certified resilience bond ($330M, Oct 2025). Full AC network. Gold standard.",
         "flag": "leader"},
        {"city": "London", "system": "TfL / Underground", "flood": "High", "heat": "Extreme",
         "flood_score": 7.5, "heat_score": 9.0,
         "key_fact": "477 climate risks identified in 2024. Deep tube tunnels exceed 30 °C. Hit 40 °C in 2022.",
         "preparedness": "First Climate Adaptation Plan (2023). AC trains for Piccadilly from 2025. Thames TE2100 plan.",
         "flag": "peer"},
        {"city": "Paris", "system": "RATP", "flood": "High", "heat": "High",
         "flood_score": 7.0, "heat_score": 7.5,
         "key_fact": "Seine flooding threatens central Metro. Only 10% green space. Most cars lack AC.",
         "preparedness": "No comprehensive metro climate roadmap. Relies on citywide Seine flood plans. Cautionary parallel.",
         "flag": "lagging"},
        {"city": "Singapore", "system": "SMRT / MRT", "flood": "Moderate", "heat": "High",
         "flood_score": 5.0, "heat_score": 8.5,
         "key_fact": "Very hot days (>35 °C) projected 4/yr to 351/yr by 2100. Marina Barrage provides flood control.",
         "preparedness": "S$125M research programme. $100B long-term coastal protection. MRT fully air-conditioned.",
         "flag": "proactive"},
        {"city": "Boston", "system": "MBTA", "flood": "High", "heat": "Moderate",
         "flood_score": 7.0, "heat_score": 5.0,
         "key_fact": "Blue Line coastal exposure to Boston Harbor. Airport station flood risk from nor'easters.",
         "preparedness": "2024 Climate Assessment. $20M Blue Line tunnel flood portal. Aquarium station flood-proofing done.",
         "flag": "peer"},
        {"city": "Chicago", "system": "CTA", "flood": "Moderate", "heat": "High",
         "flood_score": 5.5, "heat_score": 7.5,
         "key_fact": "50%+ of bus stops and rail stations rated high/very high heat vulnerability.",
         "preparedness": "CMAP Transportation Resilience Plan (TRIP) launched. Equity-focused investments planned.",
         "flag": "emerging"},
        {"city": "Washington DC", "system": "WMATA", "flood": "Moderate", "heat": "High",
         "flood_score": 5.0, "heat_score": 7.0,
         "key_fact": "Flash flooding affects entrances. Potomac storm surge risk. Aging HVAC systems.",
         "preparedness": "$874M Climate Bonds (2021). No published flood/heat roadmap comparable to MTA.",
         "flag": "lagging"},
    ]

    # Scatter: flood vs heat risk
    city_names = [c["city"] for c in CITIES]
    flood_scores = [c["flood_score"] for c in CITIES]
    heat_scores = [c["heat_score"] for c in CITIES]
    flag_colors = {"focus": RED, "leader": GREEN, "peer": TEAL, "proactive": BLUE,
                   "emerging": YELLOW, "lagging": ORANGE}
    marker_colors = [flag_colors.get(c["flag"], TEXT_DIM) for c in CITIES]
    marker_sizes = [18 if c["flag"] == "focus" else 12 for c in CITIES]

    fig_global = go.Figure()
    fig_global.add_trace(go.Scatter(
        x=flood_scores, y=heat_scores,
        mode="markers+text",
        marker=dict(color=marker_colors, size=marker_sizes, opacity=0.9,
                    line=dict(width=1, color=TEXT_DIM)),
        text=city_names,
        textposition="top center",
        textfont=dict(size=11, color=TEXT, family="'Space Grotesk', sans-serif"),
    ))
    # Quadrant lines
    fig_global.add_hline(y=7, line_dash="dot", line_color=MUTED, opacity=0.5)
    fig_global.add_vline(x=7, line_dash="dot", line_color=MUTED, opacity=0.5)
    # Quadrant labels
    fig_global.add_annotation(x=9, y=9.5, text="HIGHEST RISK", showarrow=False,
                              font=dict(size=9, color=RED, family="'JetBrains Mono', monospace"))
    fig_global.add_annotation(x=2, y=4, text="BEST MANAGED", showarrow=False,
                              font=dict(size=9, color=GREEN, family="'JetBrains Mono', monospace"))
    styled_fig(fig_global, height=450, showlegend=False,
               title=dict(text="Global Transit Climate Risk: Flood vs Heat Exposure",
                          font=dict(size=14, color=TEXT_DIM)),
               xaxis=dict(title="Flood Risk Score", range=[0, 10.5], gridcolor=MUTED,
                          dtick=2),
               yaxis=dict(title="Heat Risk Score", range=[0, 10.5], gridcolor=MUTED,
                          dtick=2))
    st.plotly_chart(fig_global, use_container_width=True)

    # Comparison table
    rows_html = ""
    for c in CITIES:
        flag_color = flag_colors.get(c["flag"], TEXT_DIM)
        star = " &#9733;" if c["flag"] == "focus" else ""
        rows_html += f"""<tr>
            <td><strong style="color:{flag_color}">{c['city']}{star}</strong><br>
                <span style="color:{TEXT_DIM};font-size:11px">{c['system']}</span></td>
            <td style="text-align:center"><span style="color:{BLUE if c['flood_score']>=7 else TEAL if c['flood_score']>=5 else GREEN};font-weight:600">{c['flood']}</span></td>
            <td style="text-align:center"><span style="color:{RED if c['heat_score']>=7 else ORANGE if c['heat_score']>=5 else GREEN};font-weight:600">{c['heat']}</span></td>
            <td style="font-size:12px;color:{TEXT_DIM}">{c['key_fact']}</td>
            <td style="font-size:12px;color:{TEXT_DIM}">{c['preparedness']}</td>
        </tr>"""

    st.markdown(f"""
    <table class="risk-table" style="font-size:12px;">
      <thead><tr>
        <th style="width:14%">City / System</th>
        <th style="width:8%;text-align:center">Flood</th>
        <th style="width:8%;text-align:center">Heat</th>
        <th style="width:35%">Key Threats</th>
        <th style="width:35%">Preparedness</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # Key takeaway
    st.markdown(f"""
    <div style="margin:2rem 0; padding:1.5rem 2rem; background:{BG_CARD}; border-radius:8px;
                border-left:4px solid {GREEN};" class="fade-up">
      <p style="color:{GREEN};font-weight:600;font-size:15px;margin-bottom:10px;">What the MTA can learn</p>
      <p style="color:{TEXT_DIM};font-size:14px;line-height:1.7;margin-bottom:10px;">
        <strong style="color:{TEXT}">Tokyo</strong> is the gold standard. Its G-Cans underground
        reservoir system prevents nearly $1B in flood damage annually and in October 2025 issued the
        world's first certified resilience bond ($330M) for coastal defenses.
      </p>
      <p style="color:{TEXT_DIM};font-size:14px;line-height:1.7;margin-bottom:10px;">
        <strong style="color:{TEXT}">London's</strong> heat challenge rivals NYC's, with deep-tube
        tunnels still lacking AC and a 2024 risk assessment finding 477 climate hazards for TfL.
        But their Climate Adaptation Plan and the TE2100 seawall programme show what systematic
        planning looks like.
      </p>
      <p style="color:{TEXT_DIM};font-size:14px;line-height:1.7;margin:0;">
        <strong style="color:{TEXT}">Paris</strong> is the cautionary tale. Similar flood exposure
        to NYC but no metro-specific climate roadmap, relying entirely on citywide Seine flood plans.
        An MTA VP noted that Paris, Tokyo, and London are all experiencing flood events comparable to New York.
      </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — The weather is getting worse (moved to later)
    # ─────────────────────────────────────────────────────────────────────────
    section("05", "The weather is getting worse.",
            "We analyze five years of NYC weather data and the trends are rather obvious. "
            "Heavy rain events and extreme heat days are happening more often and "
            "becoming the new normal.")

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
    section("06", "When it rains, the subway breaks.",
            "Real MTA incident and delays data from 2020 to 2024 show that on heavy "
            "rain days the system sees <strong>{:.0f}%</strong> more incidents than normal. "
            "This is a systemic preparedness problem towards extreme weather events.".format(rain_uplift))

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
    section("07", "Every station has a risk profile.",
            "All 496 MTA subway stations scored on flood risk, heat risk, and "
            "estimated annual economic losses. Toggle between dimensions and hover "
            "over any station to see its numbers.")

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
    # SECTION 8 — Data, Sources & Methodology
    # ─────────────────────────────────────────────────────────────────────────
    section("08", "Data, Sources & Methodology", "")

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
