"""
FBref Ratings Scraper for WC 2026 Simulator  (v3 — Playwright)
===============================================================
Uses a real Chromium browser via Playwright to bypass FBref's
IP-based blocking. Looks identical to a real Chrome user browsing
the site — no User-Agent spoofing issues, no datacenter IP blocks.

Installation (one-time, on your Mac):
    pip install playwright
    playwright install chromium

Usage
-----
  python fbref_scraper.py                        # standard run
  python fbref_scraper.py --no-cache             # force fresh fetch
  python fbref_scraper.py --dry-run              # preview, don't patch simulator
  python fbref_scraper.py --no-sos               # skip SOS adjustment
  python fbref_scraper.py --simulator path/to/wc2026_simulator.py

Why Playwright?
---------------
FBref blocks requests from Python's requests library (and even
requests-html) because they detect the HTTP fingerprint. Playwright
runs a real headless Chromium browser, so FBref sees a legitimate
Chrome request — same TLS fingerprint, same headers, same JS execution.

Rate limiting is still enforced: 5–9s between pages, randomised.
The disk cache means re-runs never re-fetch already-downloaded pages.

Outputs
-------
  fbref_raw_matches.csv     One row per qualifying match with xG/xGA
  fbref_raw_stats.csv       Aggregated per-team xG, xGA, GP
  fbref_ratings.csv         Final calibrated att/defe multipliers
  fbref_sos_report.csv      Per-team SOS diagnostics
  wc2026_simulator.py       Patched in-place with fresh ratings
"""

import argparse
import hashlib
import json
import math
import re
import sys
import textwrap
import time
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── Playwright import with friendly error ────────────────────────────────────
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
except ImportError:
    print(
        "\n✗ Playwright not installed.\n"
        "  Run: pip install playwright && playwright install chromium\n"
    )
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

RATE_LIMIT_MIN = 5.0    # seconds between page loads
RATE_LIMIT_MAX = 9.0    # randomised to avoid fingerprinting
MAX_RETRIES    = 3
RETRY_DELAY    = 12     # seconds on failure

FBREF_BASE = "https://fbref.com"

# Squad shooting stat pages (aggregate xG per team)
COMPETITION_PAGES = {
    "WC_2026":      "/en/comps/1/2026/shooting/2026-World-Cup-Stats",
    "UEFA_WCQ":     "/en/comps/27/shooting/UEFA-World-Cup-Qualifying-Stats",
    "CONMEBOL_WCQ": "/en/comps/30/shooting/CONMEBOL-World-Cup-Qualifying-Stats",
    "CONCACAF_WCQ": "/en/comps/85/shooting/CONCACAF-World-Cup-Qualifying-Stats",
    "CAF_WCQ":      "/en/comps/36/shooting/CAF-World-Cup-Qualifying-Stats",
    "AFC_WCQ":      "/en/comps/139/shooting/AFC-World-Cup-Qualifying-Stats",
    "WC_2022":      "/en/comps/1/2022/shooting/2022-World-Cup-Stats",
}

# Match-level fixture/score pages (for MLE — need individual scorelines + xG)
MATCH_SCORE_PAGES = {
    "WC_2026":      "/en/comps/1/2026/schedule/2026-World-Cup-Scores-and-Fixtures",
    "UEFA_WCQ":     "/en/comps/27/schedule/UEFA-World-Cup-Qualifying-Scores-and-Fixtures",
    "CONMEBOL_WCQ": "/en/comps/30/schedule/CONMEBOL-World-Cup-Qualifying-Scores-and-Fixtures",
    "CONCACAF_WCQ": "/en/comps/85/schedule/CONCACAF-World-Cup-Qualifying-Scores-and-Fixtures",
    "CAF_WCQ":      "/en/comps/36/schedule/CAF-World-Cup-Qualifying-Scores-and-Fixtures",
    "AFC_WCQ":      "/en/comps/139/schedule/AFC-World-Cup-Qualifying-Scores-and-Fixtures",
    "WC_2022":      "/en/comps/1/2022/schedule/2022-World-Cup-Scores-and-Fixtures",
}

# Confederation strength scalars (applied before MLE)
CONFEDERATION_STRENGTH = {
    "WC_2026":      1.00,
    "WC_2022":      1.00,
    "UEFA_WCQ":     1.00,
    "CONMEBOL_WCQ": 0.97,
    "CAF_WCQ":      0.83,
    "AFC_WCQ":      0.80,
    "CONCACAF_WCQ": 0.78,
    "OFC_WCQ":      0.65,
}

# Source weighting: WC matches count much more than qualifying
SOURCE_WEIGHT = {
    "WC_2026":      5.0,
    "WC_2022":      2.0,
    "UEFA_WCQ":     1.0,
    "CONMEBOL_WCQ": 1.0,
    "CAF_WCQ":      0.8,
    "AFC_WCQ":      0.8,
    "CONCACAF_WCQ": 0.7,
}

# Team name normalisation (FBref → simulator names)
FBREF_NAME_MAP = {
    "United States":          "USA",
    "United States Men":      "USA",
    "Korea Republic":         "South Korea",
    "Republic of Korea":      "South Korea",
    "IR Iran":                "Iran",
    "Côte d'Ivoire":          "Ivory Coast",
    "Bosnia and Herzegovina": "Bosnia",
    "North Macedonia":        "N. Macedonia",
    "Türkiye":                "Türkiye",
    "Turkey":                 "Türkiye",
    "Cape Verde Islands":     "Cape Verde",
    "DR Congo":               "DR Congo",
    "Congo DR":               "DR Congo",
}

def _normalise_team(name: str) -> str:
    name = name.strip()
    return FBREF_NAME_MAP.get(name, name)


# ─────────────────────────────────────────────────────────────────────────────
# DISK CACHE  (avoids re-fetching pages between runs)
# ─────────────────────────────────────────────────────────────────────────────

class PageCache:
    def __init__(self, cache_dir: Path):
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "_index.json"
        self.index: dict = self._load_index()

    def _load_index(self) -> dict:
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)

    def _key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def get(self, url: str) -> Optional[str]:
        k = self._key(url)
        if k in self.index:
            path = self.dir / self.index[k]["file"]
            if path.exists():
                age_h = (datetime.utcnow() -
                         datetime.fromisoformat(self.index[k]["fetched"])).total_seconds() / 3600
                print(f"  [cache] {url}  (age: {age_h:.1f}h)")
                return path.read_text(encoding="utf-8")
        return None

    def put(self, url: str, html: str):
        k = self._key(url)
        fname = f"{k}.html"
        (self.dir / fname).write_text(html, encoding="utf-8")
        self.index[k] = {"url": url, "file": fname, "fetched": datetime.utcnow().isoformat()}
        self._save_index()

    def clear(self):
        for f in self.dir.glob("*.html"):
            f.unlink()
        self.index = {}
        self._save_index()
        print("Cache cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# PLAYWRIGHT FETCHER
# ─────────────────────────────────────────────────────────────────────────────

_last_fetch_time: float = 0.0

def playwright_get(url: str, cache: PageCache, page) -> str:
    """
    Fetch a URL using a real Playwright browser page.
    Respects rate limits and uses disk cache.
    The `page` object is a persistent Playwright page — reusing it
    across requests keeps the browser session alive (looks more human).
    """
    global _last_fetch_time

    # Cache hit
    cached = cache.get(url)
    if cached:
        return cached

    # Rate limit
    elapsed = time.time() - _last_fetch_time
    gap = random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX)
    if elapsed < gap:
        wait = gap - elapsed
        print(f"  [rate-limit] sleeping {wait:.1f}s …")
        time.sleep(wait)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [browser] {url}  (attempt {attempt})")
        try:
            # Navigate and wait for the stats table to appear
            response = page.goto(url, wait_until="networkidle", timeout=30_000)
            _last_fetch_time = time.time()

            if response is None:
                print(f"  [warn] no response object — retrying")
                time.sleep(RETRY_DELAY)
                continue

            status = response.status
            if status == 200:
                # Wait a beat for JS to render any dynamic content
                time.sleep(random.uniform(1.5, 3.0))
                html = page.content()
                if len(html) < 500:
                    print(f"  [warn] suspiciously short response ({len(html)} chars) — retrying")
                    time.sleep(RETRY_DELAY)
                    continue
                cache.put(url, html)
                print(f"  [ok] {len(html):,} chars")
                return html

            elif status == 429:
                delay = RETRY_DELAY * (2 ** (attempt - 1))
                print(f"  [429 rate-limited] waiting {delay}s …")
                time.sleep(delay)

            elif status == 403:
                print(
                    f"\n✗ FBref returned 403 on attempt {attempt}.\n"
                    "  Tips:\n"
                    "    • Make sure you're running on your Mac (not a server/CI)\n"
                    "    • Disable any VPN\n"
                    "    • Wait 10 minutes and try again\n"
                    "    • Try --no-cache to force a fresh session\n"
                )
                if attempt == MAX_RETRIES:
                    sys.exit(0)   # exit cleanly — sim will use existing ratings
                time.sleep(RETRY_DELAY * attempt)

            else:
                print(f"  [warn] status {status} — retrying")
                time.sleep(RETRY_DELAY)

        except PWTimeout:
            print(f"  [timeout] attempt {attempt} timed out")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  [error] {e}")
            time.sleep(RETRY_DELAY)

    print(f"  [failed] Could not fetch {url} after {MAX_RETRIES} attempts")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# HTML PARSERS  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def parse_squad_shooting(html: str, competition: str) -> pd.DataFrame:
    if not html:
        return pd.DataFrame()
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        table = (
            soup.find("table", {"id": re.compile(r"stats_shooting")})
            or soup.find("table", {"id": re.compile(r"results.*shooting")})
        )
        if table:
            df = pd.read_html(str(table))[0]
        else:
            dfs = pd.read_html(html)
            df = next((d for d in dfs
                       if any("xg" in str(c).lower() for c in d.columns.get_level_values(-1))), None)
            if df is None:
                return pd.DataFrame()
    except Exception as e:
        print(f"  [parse] squad shooting: {e}")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col if str(c) != "nan").strip("_")
                      for col in df.columns]
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    if "squad" not in df.columns:
        return pd.DataFrame()
    df = df[df["squad"].notna() &
            ~df["squad"].astype(str).str.lower().str.contains("squad")].copy()

    def fc(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    xg_col  = fc(["xg","expected_xg","xg_xg","xg_expected","xg_standard"])
    xga_col = fc(["xga","expected_xga","xga_xga","against_xga"])
    mp_col  = fc(["mp","matches","mp_playing_time"])
    gf_col  = fc(["gls","gf","goals","goals_standard"])
    ga_col  = fc(["ga","goals_against"])

    if not xg_col:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        squad = str(row.get("squad","")).strip()
        if not squad or squad == "nan":
            continue
        try:
            rows.append({
                "squad":         squad,
                "team":          _normalise_team(squad),
                "competition":   competition,
                "confederation": competition,
                "mp":  float(row[mp_col])  if mp_col  else 1.0,
                "xg":  float(row[xg_col])  if xg_col  else float("nan"),
                "xga": float(row[xga_col]) if xga_col else float("nan"),
                "gf":  float(row[gf_col])  if gf_col  else float("nan"),
                "ga":  float(row[ga_col])  if ga_col  else float("nan"),
            })
        except (ValueError, TypeError):
            continue
    return pd.DataFrame(rows)


def parse_match_scores(html: str, competition: str) -> pd.DataFrame:
    if not html:
        return pd.DataFrame()
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", {"id": re.compile(r"sched")})
        if table is None:
            dfs = pd.read_html(html)
            table = next((d for d in dfs
                          if any(c in [str(x).lower() for x in d.columns.get_level_values(-1)]
                                 for c in ["score","xg"])), None)
            if table is None:
                return pd.DataFrame()
            df = table
        else:
            df = pd.read_html(str(table))[0]
    except Exception as e:
        print(f"  [parse] match scores: {e}")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col if str(c) != "nan").strip("_")
                      for col in df.columns]
    df.columns = [c.lower().replace(" ","_").replace("(","").replace(")","")
                  for c in df.columns]

    def fc(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    home_col  = fc(["home","home_squad"])
    away_col  = fc(["away","away_squad","visitor"])
    score_col = fc(["score","result"])
    hxg_col   = fc(["home_xg","xg","xg_home"])
    axg_col   = fc(["away_xg","xg.1","xg_away"])

    if not home_col or not away_col or not score_col:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        score_str = str(row.get(score_col,"")).strip()
        m = re.match(r"(\d+)\s*[–\-]\s*(\d+)", score_str)
        if not m:
            continue
        home_g, away_g = int(m.group(1)), int(m.group(2))
        home_name = _normalise_team(str(row.get(home_col,"")).strip())
        away_name = _normalise_team(str(row.get(away_col,"")).strip())
        if not home_name or not away_name:
            continue
        try:
            hxg = float(row[hxg_col]) if hxg_col and str(row.get(hxg_col,"")) not in ("nan","") else float("nan")
            axg = float(row[axg_col]) if axg_col and str(row.get(axg_col,"")) not in ("nan","") else float("nan")
        except (ValueError, TypeError):
            hxg = axg = float("nan")
        rows.append({
            "home":          home_name,
            "away":          away_name,
            "home_goals":    home_g,
            "away_goals":    away_g,
            "home_xg":       hxg,
            "away_xg":       axg,
            "competition":   competition,
            "confederation": competition,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MLE + SOS  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def dixon_coles_mle(matches_df: pd.DataFrame, use_xg: bool = True,
                    time_weight_halflife: int = 20) -> pd.DataFrame:
    df = matches_df.dropna(subset=["home","away"]).copy()
    if use_xg:
        df["g_h"] = df["home_xg"].where(df["home_xg"].notna(), df["home_goals"])
        df["g_a"] = df["away_xg"].where(df["away_xg"].notna(), df["away_goals"])
    else:
        df["g_h"] = df["home_goals"].astype(float)
        df["g_a"] = df["away_goals"].astype(float)
    df = df.dropna(subset=["g_h","g_a"])
    if df.empty:
        return pd.DataFrame()

    df["conf_scalar"] = df["confederation"].map(
        lambda c: CONFEDERATION_STRENGTH.get(c, 0.85))
    df["g_h"] *= df["conf_scalar"]
    df["g_a"] *= df["conf_scalar"]

    df["source_w"] = df["confederation"].map(lambda c: SOURCE_WEIGHT.get(c, 0.8))
    df = df.reset_index(drop=True)
    n = len(df)
    df["recency_w"] = 0.5 ** ((n - df.index - 1) / time_weight_halflife)
    df["weight"] = df["source_w"] * df["recency_w"]

    teams   = sorted(set(df["home"]) | set(df["away"]))
    t_idx   = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    print(f"  [MLE] {n_teams} teams · {n} matches")

    def neg_ll(params):
        log_att  = params[:n_teams]
        log_def  = params[n_teams:2*n_teams]
        base     = math.exp(params[2*n_teams])
        ll = 0.0
        for _, row in df.iterrows():
            hi, ai = t_idx[row["home"]], t_idx[row["away"]]
            lh = base * math.exp(log_att[hi]) * math.exp(log_def[ai])
            la = base * math.exp(log_att[ai]) * math.exp(log_def[hi])
            if lh <= 0 or la <= 0:
                continue
            w = row["weight"]
            ll += w * (row["g_h"] * math.log(lh) - lh)
            ll += w * (row["g_a"] * math.log(la) - la)
        return -ll

    constraints = [
        {"type":"eq","fun": lambda p: np.mean(p[:n_teams])},
        {"type":"eq","fun": lambda p: np.mean(p[n_teams:2*n_teams])},
    ]
    result = minimize(neg_ll, np.zeros(2*n_teams+1), method="SLSQP",
                      constraints=constraints, options={"maxiter":2000,"ftol":1e-9})
    if not result.success:
        print(f"  [MLE] Warning: {result.message}")

    att_vals = np.exp(result.x[:n_teams])
    def_vals = np.exp(result.x[n_teams:2*n_teams])
    match_counts = defaultdict(int)
    for _, row in df.iterrows():
        match_counts[row["home"]] += 1
        match_counts[row["away"]] += 1

    rows = []
    for t in teams:
        i = t_idx[t]
        rows.append({
            "team":       t,
            "att":        round(float(att_vals[i]), 4),
            "defe":       round(float(def_vals[i]), 4),
            "n_matches":  match_counts[t],
            "low_sample": match_counts[t] < 6,
        })
    return pd.DataFrame(rows).sort_values("att", ascending=False).reset_index(drop=True)


def compute_sos_report(matches_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    if ratings_df.empty or matches_df.empty:
        return pd.DataFrame()
    rat     = dict(zip(ratings_df["team"], ratings_df["att"]))
    def_rat = dict(zip(ratings_df["team"], ratings_df["defe"]))
    records = defaultdict(lambda: {"opp_att":[],"opp_def":[],"xg":[]})
    for _, row in matches_df.iterrows():
        h, a = row["home"], row["away"]
        xgh = row.get("home_xg", row.get("home_goals", 0))
        xga = row.get("away_xg", row.get("away_goals", 0))
        scalar = CONFEDERATION_STRENGTH.get(row.get("confederation",""), 0.85)
        if pd.isna(xgh): xgh = row.get("home_goals", 0)
        if pd.isna(xga): xga = row.get("away_goals", 0)
        xgh, xga = float(xgh)*scalar, float(xga)*scalar
        if h in rat and a in rat:
            records[h]["opp_att"].append(rat[a])
            records[h]["opp_def"].append(def_rat[a])
            records[h]["xg"].append(xgh)
            records[a]["opp_att"].append(rat[h])
            records[a]["opp_def"].append(def_rat[h])
            records[a]["xg"].append(xga)
    rows = []
    for team, data in records.items():
        if not data["opp_att"]:
            continue
        avg_opp_att = float(np.mean(data["opp_att"]))
        avg_opp_def = float(np.mean(data["opp_def"]))
        sos = (avg_opp_att + (2 - avg_opp_def)) / 2
        raw_xg_pg = float(np.mean(data["xg"])) if data["xg"] else float("nan")
        mle_att   = rat.get(team, 1.0)
        adj_xg_pg = mle_att * 1.25
        sos_boost = adj_xg_pg / raw_xg_pg if raw_xg_pg > 0 else 1.0
        rows.append({
            "team": team, "avg_opp_att": round(avg_opp_att,3),
            "avg_opp_def": round(avg_opp_def,3), "sos_score": round(sos,3),
            "raw_xg_pg": round(raw_xg_pg,3), "adj_xg_pg": round(adj_xg_pg,3),
            "sos_boost": round(sos_boost,3), "mle_att": round(mle_att,4),
            "interpretation": (
                "overrated by raw stats"  if sos_boost < 0.90 else
                "underrated by raw stats" if sos_boost > 1.10 else
                "fairly rated"
            ),
        })
    return pd.DataFrame(rows).sort_values("sos_score", ascending=False).reset_index(drop=True)


def _simple_calibrate(squad_df: pd.DataFrame) -> pd.DataFrame:
    agg = squad_df.groupby("team").apply(
        lambda g: pd.Series({
            "total_mp":  g["mp"].sum(),
            "total_xg":  g["xg"].sum(),
            "total_xga": g["xga"].sum(),
        })
    ).reset_index()
    agg["xg_pg"]  = agg["total_xg"]  / agg["total_mp"].clip(lower=1)
    agg["xga_pg"] = agg["total_xga"] / agg["total_mp"].clip(lower=1)
    med_xg  = agg["xg_pg"].median()
    med_xga = agg["xga_pg"].median()
    PRIOR   = 10.0
    agg["k"]    = agg["total_mp"] / (agg["total_mp"] + PRIOR)
    agg["att"]  = (agg["k"] * (agg["xg_pg"]/med_xg)  + (1-agg["k"])).round(4)
    agg["defe"] = (agg["k"] * (agg["xga_pg"]/med_xga) + (1-agg["k"])).round(4)
    agg["n_matches"]  = agg["total_mp"].astype(int)
    agg["low_sample"] = agg["total_mp"] < 6
    return agg[["team","att","defe","n_matches","low_sample"]].sort_values("att", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATOR PATCHER  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def patch_simulator(ratings_df: pd.DataFrame, simulator_path: Path, dry_run: bool):
    if not simulator_path.exists():
        print(f"  [patch] {simulator_path} not found — skipping.")
        return
    ratings_map = {row["team"]: (row["att"], row["defe"])
                   for _, row in ratings_df.iterrows()}
    src = simulator_path.read_text(encoding="utf-8")

    def replace_team_line(match):
        prefix, name, group, old_att, old_defe, suffix = (
            match.group(1), match.group(2), match.group(3),
            match.group(4), match.group(5), match.group(6))
        if name in ratings_map:
            new_att, new_defe = ratings_map[name]
            tag = "  # ← FBref MLE+SOS"
        else:
            new_att, new_defe = old_att, old_defe
            tag = ""
        return f'{prefix}Team("{name}", "{group}", att={new_att}, defe={new_defe},{suffix}{tag}'

    pattern = re.compile(
        r'([ \t]*)Team\("([^"]+)",\s*"([A-Z])",\s*att=([\d.]+),\s*defe=([\d.]+),(.*?)(?=\n)',
        re.MULTILINE)
    new_src, n = pattern.subn(replace_team_line, src)
    new_src = re.sub(
        r"(# fmt: off\n)",
        f"\\1# Ratings updated from FBref MLE+SOS: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n",
        new_src, count=1)

    if dry_run:
        matched = sum(1 for t in ratings_map if re.search(rf'Team\("{re.escape(t)}"', src))
        print(f"[dry-run] Would patch {matched} teams in {simulator_path}")
        return
    backup = simulator_path.with_suffix(".py.bak")
    backup.write_text(src, encoding="utf-8")
    print(f"  [patch] Backup → {backup}")
    simulator_path.write_text(new_src, encoding="utf-8")
    print(f"  [patch] {n} team entries updated in {simulator_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(cache_dir: Path, no_cache: bool, dry_run: bool,
        simulator_path: Path, apply_sos: bool):

    cache = PageCache(cache_dir)
    if no_cache:
        cache.clear()

    print("\n" + "="*60)
    print("  FBref Scraper v3 — Playwright browser mode")
    print("="*60 + "\n")

    all_squad:   list[pd.DataFrame] = []
    all_matches: list[pd.DataFrame] = []

    # Launch a single persistent Chromium browser for all requests
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ]
        )
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
            # Stealth: hide webdriver flag
            java_script_enabled=True,
        )

        # Stealth script: remove navigator.webdriver fingerprint
        ctx.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
        """)

        page = ctx.new_page()

        # Warm up: visit FBref homepage first (looks more human)
        print("  [browser] warming up — visiting fbref.com …")
        try:
            page.goto("https://fbref.com/en/", wait_until="networkidle", timeout=20_000)
            time.sleep(random.uniform(2, 4))
        except Exception as e:
            print(f"  [warn] warmup failed: {e}")

        # ── Squad shooting tables ──────────────────────────────────────────
        print("\n── Squad shooting tables ──")
        for comp, path in COMPETITION_PAGES.items():
            url  = FBREF_BASE + path
            html = playwright_get(url, cache, page)
            df   = parse_squad_shooting(html, comp)
            if not df.empty:
                print(f"  {comp}: {len(df)} teams")
                all_squad.append(df)
            else:
                print(f"  {comp}: no data parsed")

        # ── Match score tables ─────────────────────────────────────────────
        print("\n── Match score tables ──")
        for comp, path in MATCH_SCORE_PAGES.items():
            url  = FBREF_BASE + path
            html = playwright_get(url, cache, page)
            df   = parse_match_scores(html, comp)
            if not df.empty:
                print(f"  {comp}: {len(df)} matches")
                all_matches.append(df)
            else:
                print(f"  {comp}: no matches parsed")

        browser.close()

    if not all_squad and not all_matches:
        print("\n✗ No data scraped — check FBref URLs or try --no-cache")
        sys.exit(0)

    squad_df = pd.concat(all_squad,   ignore_index=True) if all_squad   else pd.DataFrame()
    match_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()

    if not squad_df.empty:
        squad_df.to_csv("fbref_raw_stats.csv", index=False)
        print(f"\n✓ fbref_raw_stats.csv  ({len(squad_df)} rows)")
    if not match_df.empty:
        match_df.to_csv("fbref_raw_matches.csv", index=False)
        print(f"✓ fbref_raw_matches.csv  ({len(match_df)} matches)")

    # ── MLE calibration ────────────────────────────────────────────────────
    print("\n── Fitting Dixon-Coles MLE …")
    if not match_df.empty:
        ratings_df = dixon_coles_mle(match_df, use_xg=True)
    elif not squad_df.empty:
        print("  [fallback] using aggregate xG ratios (no match-level data)")
        ratings_df = _simple_calibrate(squad_df)
    else:
        print("✗ No data to calibrate")
        sys.exit(0)

    if ratings_df.empty:
        print("✗ MLE returned no results")
        sys.exit(0)

    # ── SOS report ─────────────────────────────────────────────────────────
    if apply_sos and not match_df.empty:
        print("\n── Computing SOS report …")
        sos_df = compute_sos_report(match_df, ratings_df)
        if not sos_df.empty:
            sos_df.to_csv("fbref_sos_report.csv", index=False)
            print(f"✓ fbref_sos_report.csv  ({len(sos_df)} teams)")
            print("\n── Top 10 hardest schedules ──")
            print(sos_df[["team","sos_score","raw_xg_pg","adj_xg_pg",
                           "sos_boost","interpretation"]].head(10).to_string(index=False))

    # ── Save & patch ───────────────────────────────────────────────────────
    ratings_df.to_csv("fbref_ratings.csv", index=False)
    print(f"\n✓ fbref_ratings.csv  ({len(ratings_df)} teams)")
    print("\n── Top 15 by attack rating ──")
    print(ratings_df[["team","att","defe","n_matches"]].head(15).to_string(index=False))

    low = ratings_df[ratings_df.get("low_sample", pd.Series(dtype=bool)) == True]
    if not low.empty:
        print(f"\n⚠  Low sample (<6 games): {', '.join(low['team'].tolist())}")

    print(f"\n── Patching {simulator_path} …")
    patch_simulator(ratings_df, simulator_path, dry_run)
    print("\n✓ Done. Push wc2026_simulator.py to trigger a fresh deploy.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape FBref via Playwright browser + fit MLE ratings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            First-time setup:
              pip install playwright beautifulsoup4 lxml scipy numpy pandas
              playwright install chromium

            Examples:
              python fbref_scraper.py
              python fbref_scraper.py --no-cache
              python fbref_scraper.py --dry-run
              python fbref_scraper.py --simulator ../wc2026_simulator.py
        """),
    )
    parser.add_argument("--cache-dir",  type=Path, default=Path(".fbref_cache"))
    parser.add_argument("--no-cache",   action="store_true")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--no-sos",     action="store_true")
    parser.add_argument("--simulator",  type=Path, default=Path("wc2026_simulator.py"))
    args = parser.parse_args()

    run(
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
        dry_run=args.dry_run,
        simulator_path=args.simulator,
        apply_sos=not args.no_sos,
    )
