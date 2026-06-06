# WC 2026 Survivor League Simulator

Monte Carlo tournament simulator with a React web UI.
Runs 50,000 simulations daily via GitHub Actions and deploys to GitHub Pages.

## Architecture

```
GitHub Actions (daily cron 6am UTC)
  → simulator/export_json.py   runs 50k sims, writes public/data/*.json
  → commits JSON back to main
  → builds React app (web/)
  → deploys to GitHub Pages
```

Zero servers. Zero cost.

## Quick Start

### 1. Fork / clone this repo

```bash
git clone https://github.com/yourname/wc2026-survivor
cd wc2026-survivor
```

### 2. Run the simulator locally

```bash
cd simulator
pip install -r requirements.txt
python export_json.py --sims 50000 --out-dir ../public/data
```

### 3. Run the web app locally

```bash
cd web
npm install
# Copy data into web's public dir
cp ../public/data/*.json public/data/
npm run dev
# → http://localhost:5173
```

### 4. Deploy to GitHub Pages

**One-time setup:**

1. Go to your repo → **Settings → Pages**
2. Set Source to **GitHub Actions**
3. Go to **Settings → Variables → Actions** and add:
   - `VITE_BASE_PATH` = `/wc2026-survivor/`  
     (replace with your actual repo name, including leading and trailing slashes)

**Then push to main** — the workflow triggers automatically.

Or trigger manually: **Actions → Simulate & Deploy → Run workflow**

### 5. (Optional) Enable FBref live ratings

Once you have the scraper set up:

```bash
cd simulator
python fbref_scraper.py           # pulls qualifying xG, patches simulator
python export_json.py --sims 50000 --out-dir ../public/data
```

To run this in CI, add a `FBREF_ENABLED` secret and uncomment the scraper
step in `.github/workflows/simulate.yml`.

## Updating ratings mid-tournament

After each matchday, edit the `att` / `defe` values in `simulator/wc2026_simulator.py`
for teams whose form has changed, or re-run `fbref_scraper.py` to pull fresh xG data.
Push to main and the workflow will re-simulate and redeploy automatically.

## Adjusting pick% model

In `wc2026_simulator.py`, find `MEDIA_SALIENCE` and `PICK_TEMPERATURE`:

```python
MEDIA_SALIENCE = {
    "Argentina": 1.00,   # defending champion — will be heavily over-picked
    "Brazil":    1.00,   # global brand
    ...
}
PICK_TEMPERATURE = 0.75  # lower = more chalk-heavy field
```

These are the two biggest levers for tuning the game theory model.

## Files

```
simulator/
  wc2026_simulator.py     Core Monte Carlo engine (Dixon-Coles Poisson)
  fbref_scraper.py        FBref xG scraper with MLE + SOS calibration
  export_json.py          Runs sim → writes JSON for web app
  requirements.txt

web/src/
  App.jsx                 Main app + tab navigation
  components/
    SurvivorPicks.jsx     EV/Chalk/Contrarian pick sheets
    GroupStage.jsx        Group tables + match odds
    KnockoutBracket.jsx   Sortable knockout probability table + chart
    PickIntelligence.jsx  Scatter plot + EV table per stage

public/data/              Pre-computed JSON (committed by CI)
  meta.json
  groups.json
  knockout.json
  survivor.json

.github/workflows/
  simulate.yml            Daily cron + deploy workflow
```
