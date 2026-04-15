"""
Climate Finance Job Digest
--------------------------
Searches 5 job categories via DuckDuckGo (last 7 days), curates each with
Groq/Llama 3.3 (free), renders a segmented HTML email digest, and sends it
via Gmail SMTP.

Usage:
    python main.py
"""

import os
import smtplib
import textwrap
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from ddgs import DDGS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------------------
# Candidate profile
# ---------------------------------------------------------------------------
CANDIDATE_PROFILE = textwrap.dedent("""\
    Climate finance specialist, 7 years experience, Columbia MS Climate Finance.
    Targeting: World Bank, ADB, GCF, UN agencies, IDB Invest, FRDL, CIF,
    blended finance funds, climate risk roles in New York, impact investing
    in DC/NYC/Boston, climate policy in DC/NYC, and senior C-suite at climate
    tech/policy firms in India.
    Skills: blended finance, green bonds, MDB operations, climate risk
    assessment, adaptation & resilience finance, carbon markets, ESG,
    impact investing, climate policy analysis.
""")

# ---------------------------------------------------------------------------
# Job categories with their search queries
# ---------------------------------------------------------------------------
JOB_CATEGORIES = {
    "Multilateral Banks & Climate Funds": {
        "label": "MULTILATERAL BANKS & CLIMATE FUNDS",
        "accent": "#1D4ED8",
        "queries": [
            "World Bank climate finance adaptation resilience jobs 2025",
            "Asian Development Bank ADB climate finance data analyst 2025",
            "Green Climate Fund GCF job openings 2025",
            "UN UNDP UNEP IDB climate finance jobs 2025",
            "FRDL CIF Climate Investment Funds jobs openings 2025",
            "IDB Invest climate adaptation resilience finance jobs 2025",
        ],
    },
    "Climate Risk — New York": {
        "label": "CLIMATE RISK — NEW YORK",
        "accent": "#B91C1C",
        "queries": [
            "climate risk analyst jobs New York City 2025",
            "climate financial risk manager NYC 2025",
            "ESG climate risk quantitative analyst New York 2025",
            "physical climate risk transition risk jobs New York 2025",
        ],
    },
    "Impact Investing & Family Offices — NYC / Boston": {
        "label": "IMPACT INVESTING & FAMILY OFFICES — NYC / BOSTON",
        "accent": "#15803D",
        "queries": [
            "site:impactalpha.com impact investing jobs New York Boston 2025",
            "site:thegiin.org impact investing jobs New York Boston 2025",
            "GIIN impact investing senior associate director New York 2025",
            "ImpactAlpha impact fund manager New York Boston 2025",
            "family office impact investing director VP New York 2025",
            "family office ESG sustainable investing analyst associate New York Boston 2025",
            "impact investing principal senior associate New York Boston 2025",
        ],
    },
    "Climate Policy — DC / NYC": {
        "label": "CLIMATE POLICY — DC / NYC",
        "accent": "#6D28D9",
        "queries": [
            "climate policy analyst director Washington DC 2025",
            "climate policy jobs New York 2025",
            "climate advocacy senior policy DC 2025",
        ],
    },
    "C-Suite — Climate Tech & Policy, India": {
        "label": "C-SUITE — CLIMATE TECH & POLICY, INDIA",
        "accent": "#C2410C",
        "queries": [
            "climate tech CEO CTO VP India 2025",
            "clean energy director senior India 2025",
            "climate policy senior director India 2025",
            "climate finance head director India 2025",
        ],
    },
}

RESULTS_PER_QUERY        = 3
MAX_CHARS_PER_CATEGORY   = 2_500   # keeps each Groq call within free-tier limits


# ---------------------------------------------------------------------------
# Phase 1: search each category
# ---------------------------------------------------------------------------
def search_category(category: str, queries: list[str]) -> str:
    """Run DDG queries for one category; return raw results as text."""
    seen: set[str] = set()
    blocks: list[str] = []

    with DDGS() as ddgs:
        for query in queries:
            try:
                # timelimit='w' = last 7 days — captures the 48-hour window
                hits = list(ddgs.text(query, max_results=RESULTS_PER_QUERY, timelimit="w"))
            except Exception as exc:
                print(f"    ⚠ DDG error on '{query}': {exc}")
                hits = []

            for h in hits:
                url = h.get("href", "")
                if url in seen:
                    continue
                seen.add(url)
                blocks.append(
                    f"Title: {h.get('title', '')}\n"
                    f"URL:   {url}\n"
                    f"Snippet: {h.get('body', '')}\n"
                )

            time.sleep(1.0)

    raw = "\n".join(blocks)
    if len(raw) > MAX_CHARS_PER_CATEGORY:
        raw = raw[:MAX_CHARS_PER_CATEGORY] + "\n[truncated]"
    return raw


def search_all_categories() -> dict[str, str]:
    """Return {category_name: raw_results_text} for all categories."""
    results: dict[str, str] = {}
    total_categories = len(JOB_CATEGORIES)
    for i, (cat_name, cat_cfg) in enumerate(JOB_CATEGORIES.items(), 1):
        print(f"  [{i}/{total_categories}] {cat_name}")
        results[cat_name] = search_category(cat_name, cat_cfg["queries"])
        count = results[cat_name].count("Title:")
        print(f"    → {count} results")
    return results


# ---------------------------------------------------------------------------
# Phase 2: curate each category with Groq
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are an expert climate-finance career coach. "
    "You return ONLY valid HTML fragments — no markdown, no explanation."
)

def curate_category(client: Groq, cat_name: str, cat_cfg: dict, raw: str, today: str) -> str:
    """Return 2-3 HTML job entries for one category."""
    if not raw.strip():
        return ""

    accent = cat_cfg["accent"]

    prompt = f"""\
Today is {today}.

CANDIDATE:
{CANDIDATE_PROFILE}

CATEGORY: {cat_name}

RAW SEARCH RESULTS (last 7 days):
{raw}

TASK:
Select the 2-3 most relevant open roles for this candidate. For each, produce one HTML
entry using EXACTLY this template (inline CSS only, no class attributes):

<div style="padding:28px 0;border-bottom:1px solid rgba(6,23,23,0.08);">
  <table width="100%" cellpadding="0" cellspacing="0"><tr>
    <td>
      <span style="font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:{accent};font-family:'DM Sans',Arial,sans-serif;">ORG &nbsp;·&nbsp; LOCATION</span>
    </td>
    <td align="right" style="white-space:nowrap;">
      <span style="font-size:10px;font-weight:600;padding:3px 10px;border-radius:20px;background:BADGE_BG;color:BADGE_COLOR;font-family:'DM Sans',Arial,sans-serif;">BADGE_TEXT</span>
    </td>
  </tr></table>
  <h3 style="margin:8px 0 10px;font-size:20px;font-weight:400;color:#061717;line-height:1.3;letter-spacing:-0.3px;font-family:'DM Serif Display',Georgia,serif;">JOB TITLE</h3>
  <p style="margin:0 0 16px;font-size:14px;color:#4B5563;line-height:1.75;font-family:'DM Sans',Arial,sans-serif;">Write 1–2 sentences: what the role involves and precisely why it matches this candidate's climate finance background and target institutions.</p>
  <span style="font-size:12px;color:#9CA3AF;font-family:'DM Sans',Arial,sans-serif;">POSTED_DATE_OR_DEADLINE</span>
  &nbsp;&nbsp;
  <a href="URL" style="font-size:13px;font-weight:500;color:{accent};text-decoration:none;font-family:'DM Sans',Arial,sans-serif;">View role &#8594;</a>
</div>

Fit badges — pick one, use exact values:
  Strong Match  → background:#ECFDF5  color:#065F46
  Good Match    → background:#FEF9C3  color:#854D0E
  Worth Noting  → background:#F1F5F9  color:#475569
  NEW (prepend to JOB TITLE only if the posting appears to be ≤48 hours old)

Rules:
- Fill in all placeholders with real data from the search results.
- POSTED_DATE_OR_DEADLINE: use the posting date if visible, otherwise write "Open until filled".
- No bullet lists, no markdown, no text outside the template divs.
- Return ONLY the entry divs.
"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"    ⚠ Groq error for '{cat_name}': {exc}")
        return f"<p style='color:#999;font-size:13px;'>Could not retrieve results for this category.</p>"


# ---------------------------------------------------------------------------
# Phase 3: assemble full HTML
# ---------------------------------------------------------------------------
def build_html(curated: dict[str, str], today: str) -> str:
    """Wrap all category sections into a single HTML email document."""
    SF = "'DM Sans',Arial,Helvetica,sans-serif"
    TF = "'DM Serif Display',Georgia,'Times New Roman',serif"

    sections_html = ""
    for cat_name, entries_html in curated.items():
        if not entries_html.strip():
            continue
        cfg    = JOB_CATEGORIES[cat_name]
        label  = cfg["label"]
        accent = cfg["accent"]
        sections_html += f"""
  <div style="margin-bottom:52px;">
    <p style="margin:0 0 14px;font-family:{SF};font-size:10px;font-weight:700;
      letter-spacing:0.14em;text-transform:uppercase;color:{accent};">{label}</p>
    <div style="border-top:1px solid rgba(6,23,23,0.12);margin-bottom:0;">{entries_html}
      <div style="height:1px;"></div>
    </div>
  </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Climate Finance — Job Digest, {today}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;1,400&family=DM+Serif+Display&display=swap" rel="stylesheet">
<style>
  body {{margin:0;padding:0;background:#F5F4F0;}}
  a {{text-decoration:none;}}
</style>
</head>
<body>
<table width="100%" cellpadding="0" cellspacing="0" style="background:#F5F4F0;">
<tr><td align="center" style="padding:36px 16px 56px;">
<table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;">

  <!-- Pre-header label -->
  <tr><td style="padding-bottom:18px;">
    <p style="margin:0;font-family:{SF};font-size:11px;font-weight:600;
      letter-spacing:0.12em;text-transform:uppercase;color:#9CA3AF;text-align:center;">
      Climate Finance &nbsp;/&nbsp; Weekly Job Digest
    </p>
  </td></tr>

  <!-- Masthead -->
  <tr><td style="background:#061717;padding:44px 48px 40px;">
    <p style="margin:0 0 16px;font-family:{SF};font-size:11px;font-weight:500;
      letter-spacing:0.12em;text-transform:uppercase;color:#4ADE80;">
      {today}
    </p>
    <h1 style="margin:0 0 14px;font-family:{TF};font-size:36px;font-weight:400;
      color:#FFFFFF;line-height:1.15;letter-spacing:-0.5px;">
      The Climate Finance<br>Job Digest
    </h1>
    <p style="margin:0;font-family:{SF};font-size:13px;color:rgba(255,255,255,0.45);
      line-height:1.6;">
      Curated roles across multilateral finance, climate risk, impact investing,
      policy, and clean energy leadership.
      &nbsp;&#8212;&nbsp; NEW = posted within 48 hours.
    </p>
  </td></tr>

  <!-- Green rule -->
  <tr><td style="background:#16A34A;height:2px;"></td></tr>

  <!-- Body -->
  <tr><td style="background:#FFFFFF;padding:44px 48px 36px;">

    {sections_html}

    <!-- Footer -->
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr><td style="border-top:1px solid rgba(6,23,23,0.08);padding-top:28px;">
        <p style="margin:0 0 6px;font-family:{SF};font-size:12px;
          color:#9CA3AF;line-height:1.6;">
          Sourced via web search (last 7 days) and AI-curated by Groq / Llama 3.3.
        </p>
        <p style="margin:0;font-family:{SF};font-size:11px;color:#D1D5DB;">
          Always verify postings on the employer's website before applying.
        </p>
      </td></tr>
    </table>

  </td></tr>

</table>
</td></tr>
</table>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Phase 4: send email
# ---------------------------------------------------------------------------
def send_email(html: str) -> None:
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digest.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"HTML digest saved → {out_path}")

    smtp_user = os.getenv("GMAIL_USER", "").strip()
    smtp_pass = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    recipient = os.getenv("RECIPIENT_EMAIL", smtp_user).strip() or smtp_user

    if not smtp_user or smtp_pass in ("xxxx_xxxx_xxxx_xxxx", ""):
        print("\nGmail credentials not set — skipping email send.")
        return

    today = datetime.now().strftime("%B %d, %Y")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🌿 Climate Finance Job Digest — {today}"
    msg["From"]    = smtp_user
    msg["To"]      = recipient
    msg.attach(MIMEText("Please view this in an HTML-capable email client.", "plain"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    print(f"Sending digest to {recipient}…")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, recipient, msg.as_string())
    print(f"✅ Digest sent to {recipient}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key or groq_key == "your_groq_api_key_here":
        raise SystemExit("ERROR: Set GROQ_API_KEY in .env\nFree key at https://console.groq.com")

    client = Groq(api_key=groq_key)
    today  = datetime.now().strftime("%B %d, %Y")

    print("=" * 60)
    print("  Climate Finance Job Digest")
    print("=" * 60)
    print()

    # Phase 1 — search
    print("── Phase 1: Searching job boards (last 7 days) ──")
    raw_by_cat = search_all_categories()
    print()

    # Phase 2 — curate each category
    print("── Phase 2: Curating with Groq (one call per category) ──")
    curated: dict[str, str] = {}
    for cat_name, raw in raw_by_cat.items():
        print(f"  {cat_name}…")
        curated[cat_name] = curate_category(client, cat_name, JOB_CATEGORIES[cat_name], raw, today)
        time.sleep(2)   # stay under Groq RPM limit
    print()

    # Phase 3 — assemble HTML
    html = build_html(curated, today)

    # Phase 4 — send
    print("── Phase 3: Sending ──")
    send_email(html)

    print()
    print("=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
