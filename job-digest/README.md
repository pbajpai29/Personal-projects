# Climate Finance Job Digest

Searches the web for open climate finance roles, curates the top 5-7 matches
against a candidate profile, renders a styled HTML email digest, and sends it
via Gmail. **100% free — no paid APIs required.**

## What it does

| Step | Details |
|------|---------|
| **Search** | Runs 10 targeted DuckDuckGo queries across ADB, World Bank, CIF, GCF, UN careers, Devex, and LinkedIn — no API key needed |
| **Curate** | Google Gemini (`gemini-1.5-flash`, free tier) selects the best 5-7 matches, scores each one (⭐ Strong / ✅ Good Match), and explains relevance |
| **Render** | Results are formatted as a responsive HTML email with job cards, fit badges, and apply buttons |
| **Send** | Delivered via Gmail SMTP using an App Password; a `digest.html` copy is always saved locally |

## Setup

### 1. Install dependencies

```bash
cd job-digest
pip install -r requirements.txt
```

### 2. Configure `.env`

Edit `.env`:

```
GOOGLE_API_KEY=AIza...                   # free — see below
GMAIL_USER=pulkit.bajpei@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx   # 16-char App Password — see below
RECIPIENT_EMAIL=pulkit.bajpei@gmail.com
```

**How to get a free Google Gemini API key**
1. Go to <https://aistudio.google.com/app/apikey>
2. Sign in with your Google account (no credit card needed)
3. Click **Get API key** → **Create API key**
4. Paste it into `GOOGLE_API_KEY`

**How to get a Gmail App Password**
1. Enable 2-Step Verification on your Google account
2. Go to <https://myaccount.google.com/apppasswords>
3. Create a new app password (name it anything, e.g. "job-digest")
4. Copy the 16-character code into `GMAIL_APP_PASSWORD`

### 3. Run

```bash
python main.py
```

Progress is printed to the terminal. `digest.html` is written to the project
folder. If Gmail credentials are configured, the digest is also emailed.

## File structure

```
job-digest/
├── main.py           ← entry point
├── .env              ← secrets (never commit this)
├── requirements.txt
├── README.md
└── digest.html       ← output — created on each run
```

## Notes

- DuckDuckGo search is free and requires no account or API key.
- Gemini free tier: 15 requests/min, 1 500 requests/day — more than enough.
- No Gmail App Password? The script skips sending and just saves `digest.html`.
- Edit `CANDIDATE_PROFILE` and `SEARCH_QUERIES` in `main.py` to customise.
