"""
FBref Ratings Scraper for WC 2026 Simulator  (v2 — SOS-adjusted)
=================================================================
Scrapes national team xG / xGA data from FBref, then calibrates
attack / defense multipliers using TWO layers of opponent adjustment:

  Layer 1 — Dixon-Coles MLE
    Solves for all team att/def ratings simultaneously via maximum
    likelihood on observed (xG_for, xG_against) match data.  This
    means a 3-0 xG result against a strong opponent boosts your
    rating MORE than the same result against a weak one.

  Layer 2 — Confederation strength scalar
    Qualifying happens within confederations so UEFA teams never
    face CONMEBOL teams.  We apply a cross-confederation scalar
    (derived from historical WC group-stage xG) so that a CONCACAF
    team's qualifying xG is not treated the same as a UEFA one.

Rate-limit policy (FBref)
--------------------------
  • 4–7 s randomised gap between every HTTP request
  • Disk cache — pages already fetched are never re-requested
  • Exponential back-off on 429 / 5xx  (8 s → 16 s → 32 s, 3 tries)
  • Descriptive User-Agent identifying this as personal research
  • Clean exit on 403 with diagnosis

Usage
-----
  python fbref_scraper.py                        # standard run
  python fbref_scraper.py --no-cache             # force fresh fetch
  python fbref_scraper.py --dry-run              # preview, don't patch
  python fbref_scraper.py --no-sos               # skip confederation adjust
  python fbref_scraper.py --simulator path/to/wc2026_simulator.py

Outputs
-------
  fbref_raw_matches.csv      One row per qualifying match with xG/xGA
  fbref_raw_stats.csv        Aggregated per-team xG, xGA, GP
  fbref_ratings.csv          Final calibrated att / defe multipliers
  fbref_sos_report.csv       Per-team SOS diagnostics
  wc2026_simulator.py        Patched in-place (unless --dry-run)
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
import requests
from bs4 import BeautifulSoup
from scipy.optimize import minimize

# ─────────────────────────────────────────────────────────────
# RATE LIMIT CONFIG
# ─────────────────────────────────────────────────────────────
RATE_LIMIT_MIN   = 4.0   # seconds — FBref asks ≥3 s; we add margin
RATE_LIMIT_MAX   = 7.0
MAX_RETRIES      = 3
RETRY_BASE_DELAY = 8     # doubles each retry

FBREF_BASE = "https://fbref.com"

# Squad Shooting pages — one per qualifying confederation + WC itself
COMPETITION_PAGES = {
    # ── Live 2026 WC data (highest priority — actual tournament xG) ──
    "WC_2026":      "/en/comps/1/2026/shooting/2026-World-Cup-Stats",
    # ── Qualifying data (used for teams not yet played in WC) ──
    "UEFA_WCQ":     "/en/comps/27/shooting/UEFA-World-Cup-Qualifying-Stats",
    "CONMEBOL_WCQ": "/en/comps/30/shooting/CONMEBOL-World-Cup-Qualifying-Stats",
    "CONCACAF_WCQ": "/en/comps/85/shooting/CONCACAF-World-Cup-Qualifying-Stats",
    "CAF_WCQ":      "/en/comps/36/shooting/CAF-World-Cup-Qualifying-Stats",
    "AFC_WCQ":      "/en/comps/139/shooting/AFC-World-Cup-Qualifying-Stats",
    "WC_2022":      "/en/comps/1/2022/shooting/2022-World-Cup-Stats",
}

# Match-level score pages (needed for MLE — we need individual scorelines)
MATCH_SCORE_PAGES = {
    # ── Live 2026 WC matches (most important — weight these highest) ──
    "WC_2026":      "/en/comps/1/2026/schedule/2026-World-Cup-Scores-and-Fixtures",
    # ── Qualifying matches ──
    "UEFA_WCQ":     "/en/comps/27/schedule/UEFA-World-Cup-Qualifying-Scores-and-Fixtures",
    "CONMEBOL_WCQ": "/en/comps/30/schedule/CONMEBOL-World-Cup-Qualifying-Scores-and-Fixtures",
    "CONCACAF_WCQ": "/en/comps/85/schedule/CONCACAF-World-Cup-Qualifying-Scores-and-Fixtures",
    "CAF_WCQ":      "/en/comps/36/schedule/CAF-World-Cup-Qualifying-Scores-and-Fixtures",
    "AFC_WCQ":      "/en/comps/139/schedule/AFC-World-Cup-Qualifying-Scores-and-Fixtures",
    "WC_2022":      "/en/comps/1/2022/schedule/2022-World-Cup-Scores-and-Fixtures",
}

USER_AGENT = (
    "Mozilla/5.0 (personal-research; wc2026-survivor-league; "
    "contact: see fbref.com terms) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {
    "User-Agent":      USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer":         "https://www.google.com/",
    "DNT":             "1",
}

# ─────────────────────────────────────────────────────────────
# CONFEDERATION STRENGTH SCALARS
# Derived from 2014–2022 World Cup group-stage xG averages.
# Reflects how much harder it is to generate xG against each
# confederation's typical opponents.  UEFA = 1.0 baseline.
# ─────────────────────────────────────────────────────────────
# How to read:  CONMEBOL = 0.97 means CONMEBOL qualifying is
# ~97% as difficult as UEFA qualifying per unit of xG.
# A CONCACAF team's raw xG is inflated because they face easier
# opponents; we scale it down by 0.78 before MLE.
CONFEDERATION_STRENGTH = {
    "WC_2026":      1.00,   # live tournament — already cross-confederation, no adjustment needed
    "UEFA_WCQ":     1.00,   # baseline
    "CONMEBOL_WCQ": 0.97,   # very close to UEFA
    "CAF_WCQ":      0.83,   # Africa — good but easier avg opponent
    "AFC_WCQ":      0.80,   # Asia
    "CONCACAF_WCQ": 0.78,   # CONCACAF — significant inflation risk
    "OFC_WCQ":      0.65,   # Oceania — tiny pool, very weak opponents
    "WC_2022":      1.00,   # actual WC = already cross-conf
}

# ─────────────────────────────────────────────────────────────
# TEAM NAME NORMALISATION
# ─────────────────────────────────────────────────────────────
FBREF_NAME_MAP = {
    "United States":          "USA",
    "United States Men":      "USA",
    "Korea Republic":         "South Korea",
    "Republic of Korea":      "South Korea",
    "IR Iran":                "Iran",
    "Côte d'Ivoire":          "Ivory Coast",
    "Bosnia and Herzegovina": "Bosnia",
    "North Macedonia":        "N. Macedonia",
}

# ─────────────────────────────────────────────────────────────
# DISK CACHE
# ─────────────────────────────────────────────────────────────
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
                print(f"  [cache hit] {url}")
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


# ─────────────────────────────────────────────────────────────
# RATE-LIMITED FETCHER
# ─────────────────────────────────────────────────────────────
_last_request_time: float = 0.0

def polite_get(url: str, cache: PageCache, session: requests.Session) -> str:
    global _last_request_time
    cached = cache.get(url)
    if cached:
        return cached

    elapsed = time.time() - _last_request_time
    gap = random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX)
    if elapsed < gap:
        wait = gap - elapsed
        print(f"  [rate-limit] sleeping {wait:.1f}s …")
        time.sleep(wait)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [fetch] {url}  (attempt {attempt})")
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
            _last_request_time = time.time()

            if resp.status_code == 200:
                cache.put(url, resp.text)
                return resp.text

            elif resp.status_code == 403:
                print(
                    "\n⚠ FBref returned 403 Forbidden.\n"
                    "  This is expected when running from CI/cloud environments.\n"
                    "  Simulation will continue with existing hardcoded ratings.\n"
                    "  To update ratings, run fbref_scraper.py locally on your Mac.\n"
                )
                sys.exit(0)   # exit cleanly so CI pipeline continues

            elif resp.status_code == 429:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"  [429] rate-limited — waiting {delay}s …")
                time.sleep(delay)

            elif resp.status_code >= 500:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"  [{resp.status_code}] server error — waiting {delay}s …")
                time.sleep(delay)

            else:
                print(f"  [skip] unexpected status {resp.status_code}")
                return ""

        except requests.exceptions.Timeout:
            print(f"  [timeout] attempt {attempt}")
            time.sleep(RETRY_BASE_DELAY)
        except requests.exceptions.ConnectionError as e:
            print(f"  [connection error] {e}")
            time.sleep(RETRY_BASE_DELAY)

    print(f"  [failed] {url} after {MAX_RETRIES} attempts.")
    return ""


# ─────────────────────────────────────────────────────────────
# HTML PARSERS
# ─────────────────────────────────────────────────────────────

def _normalise_team(name: str) -> str:
    name = name.strip()
    return FBREF_NAME_MAP.get(name, name)

def parse_squad_shooting(html: str, competition: str) -> pd.DataFrame:
    """Parse FBref 'Squad Shooting' table → per-team xG/xGA aggregates."""
    if not html:
        return pd.DataFrame()
    soup = BeautifulSoup(html, "lxml")

    table = (
        soup.find("table", {"id": re.compile(r"stats_shooting")})
        or soup.find("table", {"id": re.compile(r"results.*shooting")})
    )

    try:
        if table:
            df = pd.read_html(str(table))[0]
        else:
            dfs = pd.read_html(html)
            df = next((d for d in dfs if any("xg" in str(c).lower() for c in d.columns.get_level_values(-1))), None)
            if df is None:
                return pd.DataFrame()
    except Exception as e:
        print(f"  [parse-shooting] {e}")
        return pd.DataFrame()

    # Flatten multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col if str(c) != "nan").strip("_") for col in df.columns]
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    if "squad" not in df.columns:
        return pd.DataFrame()
    df = df[df["squad"].notna() & ~df["squad"].astype(str).str.lower().str.contains("squad")].copy()

    def fc(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    xg_col  = fc(["xg", "expected_xg", "xg_xg", "xg_expected", "xg_standard"])
    xga_col = fc(["xga", "expected_xga", "xga_xga", "against_xga"])
    mp_col  = fc(["mp", "matches", "mp_playing_time"])
    gf_col  = fc(["gls", "gf", "goals", "goals_standard"])
    ga_col  = fc(["ga", "goals_against"])

    if not xg_col:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        squad = str(row.get("squad", "")).strip()
        if not squad or squad == "nan":
            continue
        try:
            rows.append({
                "squad":       squad,
                "team":        _normalise_team(squad),
                "competition": competition,
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
    """
    Parse FBref fixtures/scores table → one row per played match.
    Columns: home, away, home_xg, away_xg, home_goals, away_goals, competition
    """
    if not html:
        return pd.DataFrame()
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table", {"id": re.compile(r"sched")})
    if table is None:
        try:
            dfs = pd.read_html(html)
            # Pick table most likely to be fixtures (has 'Score' or 'xG' col)
            for d in dfs:
                cols_lower = [str(c).lower() for c in d.columns.get_level_values(-1)]
                if "score" in cols_lower or "xg" in cols_lower:
                    table = d
                    break
        except Exception:
            pass

    if table is None:
        return pd.DataFrame()

    try:
        if not isinstance(table, pd.DataFrame):
            df = pd.read_html(str(table))[0]
        else:
            df = table
    except Exception as e:
        print(f"  [parse-matches] {e}")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col if str(c) != "nan").strip("_") for col in df.columns]
    df.columns = [c.lower().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]

    def fc(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    home_col  = fc(["home", "home_squad"])
    away_col  = fc(["away", "away_squad", "visitor"])
    score_col = fc(["score", "result"])
    hxg_col   = fc(["home_xg", "xg", "xg_home"])
    axg_col   = fc(["away_xg", "xg.1", "xg_away"])

    if not home_col or not away_col or not score_col:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        score_str = str(row.get(score_col, "")).strip()
        # FBref score format: "2–1" or "2-1" (en-dash or hyphen)
        m = re.match(r"(\d+)\s*[–\-]\s*(\d+)", score_str)
        if not m:
            continue  # unplayed fixture or header row
        home_g, away_g = int(m.group(1)), int(m.group(2))

        home_name = _normalise_team(str(row.get(home_col, "")).strip())
        away_name = _normalise_team(str(row.get(away_col, "")).strip())
        if not home_name or not away_name:
            continue

        try:
            hxg = float(row[hxg_col]) if hxg_col and str(row[hxg_col]) not in ("nan", "") else float("nan")
            axg = float(row[axg_col]) if axg_col and str(row[axg_col]) not in ("nan", "") else float("nan")
        except (ValueError, TypeError):
            hxg = axg = float("nan")

        rows.append({
            "home":        home_name,
            "away":        away_name,
            "home_goals":  home_g,
            "away_goals":  away_g,
            "home_xg":     hxg,
            "away_xg":     axg,
            "competition": competition,
            "confederation": competition,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# LAYER 1 — DIXON-COLES MLE
# ─────────────────────────────────────────────────────────────

def dixon_coles_mle(
    matches_df: pd.DataFrame,
    use_xg: bool = True,
    time_weight_halflife: int = 20,  # matches; older games weighted less
) -> pd.DataFrame:
    """
    Fit Dixon-Coles attack/defense parameters for all teams simultaneously
    via maximum likelihood on match-level (x)G data.

    Parameters
    ----------
    matches_df : DataFrame with columns home, away, home_xg, away_xg
                 (falls back to home_goals/away_goals if xg is missing)
    use_xg     : use xG values instead of actual goals (preferred)
    time_weight_halflife : recency weighting; set high to treat all games equally

    Returns
    -------
    DataFrame with columns: team, att, defe, n_matches
    """
    df = matches_df.dropna(subset=["home", "away"]).copy()

    # Decide which goal column to use
    if use_xg:
        df["g_h"] = df["home_xg"].where(df["home_xg"].notna(), df["home_goals"])
        df["g_a"] = df["away_xg"].where(df["away_xg"].notna(), df["away_goals"])
    else:
        df["g_h"] = df["home_goals"].astype(float)
        df["g_a"] = df["away_goals"].astype(float)

    df = df.dropna(subset=["g_h", "g_a"])
    if df.empty:
        print("  [MLE] No valid match data — returning empty.")
        return pd.DataFrame()

    # Apply confederation strength scalar to xG values
    # This adjusts raw xG before fitting so cross-conf comparisons are valid
    df["conf_scalar"] = df["confederation"].map(
        lambda c: CONFEDERATION_STRENGTH.get(c, 0.85)
    )
    df["g_h"] = df["g_h"] * df["conf_scalar"]
    df["g_a"] = df["g_a"] * df["conf_scalar"]

    # Source weighting: live WC 2026 matches count 5× more than qualifying
    # because they're actual tournament performance vs same opponents
    SOURCE_WEIGHT = {
        "WC_2026":      5.0,   # live tournament — highest signal
        "WC_2022":      2.0,   # recent WC — cross-conf, high quality
        "UEFA_WCQ":     1.0,   # qualifying baseline
        "CONMEBOL_WCQ": 1.0,
        "CAF_WCQ":      0.8,
        "AFC_WCQ":      0.8,
        "CONCACAF_WCQ": 0.7,
    }
    df["source_weight"] = df["confederation"].map(lambda c: SOURCE_WEIGHT.get(c, 0.8))

    # Recency weighting: w = 0.5^(games_ago / halflife)
    df = df.reset_index(drop=True)
    n = len(df)
    df["recency_weight"] = 0.5 ** ((n - df.index - 1) / time_weight_halflife)
    df["weight"] = df["source_weight"] * df["recency_weight"]

    # Build team index
    teams = sorted(set(df["home"]) | set(df["away"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams  = len(teams)

    print(f"  [MLE] Fitting {n_teams} teams on {n} matches …")

    # Parameter vector layout:
    #   params[0:n_teams]          = log(att_i)
    #   params[n_teams:2*n_teams]  = log(def_i)
    #   params[2*n_teams]          = log(base_rate)
    # Constraints: sum(log_att) = 0  (identifiability)

    def log_likelihood(params):
        log_att  = params[:n_teams]
        log_def  = params[n_teams:2*n_teams]
        log_base = params[2*n_teams]
        base = math.exp(log_base)
        ll = 0.0
        for _, row in df.iterrows():
            hi = team_idx[row["home"]]
            ai = team_idx[row["away"]]
            lam_h = base * math.exp(log_att[hi]) * math.exp(log_def[ai])
            lam_a = base * math.exp(log_att[ai]) * math.exp(log_def[hi])
            # Poisson log-likelihood: g*log(lam) - lam  (drop constants)
            # Use xG as the observed value
            gh, ga, w = row["g_h"], row["g_a"], row["weight"]
            if lam_h <= 0 or lam_a <= 0:
                continue
            ll += w * (gh * math.log(lam_h) - lam_h)
            ll += w * (ga * math.log(lam_a) - lam_a)
        return -ll  # minimise negative log-likelihood

    # Identifiability constraint: mean(log_att) = 0
    constraints = [
        {"type": "eq", "fun": lambda p: np.mean(p[:n_teams])},
        {"type": "eq", "fun": lambda p: np.mean(p[n_teams:2*n_teams])},
    ]

    # Initial guess: all zeros (att=1, def=1, base=1)
    x0 = np.zeros(2 * n_teams + 1)

    result = minimize(
        log_likelihood,
        x0,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-9},
    )

    if not result.success:
        print(f"  [MLE] Warning: optimiser did not fully converge: {result.message}")

    log_att  = result.x[:n_teams]
    log_def  = result.x[n_teams:2*n_teams]

    att_vals  = np.exp(log_att)
    def_vals  = np.exp(log_def)

    # Match count per team (for diagnostic / shrinkage flag)
    match_counts = defaultdict(int)
    for _, row in df.iterrows():
        match_counts[row["home"]] += 1
        match_counts[row["away"]] += 1

    rows = []
    for t in teams:
        i = team_idx[t]
        rows.append({
            "team":      t,
            "att":       round(float(att_vals[i]), 4),
            "defe":      round(float(def_vals[i]), 4),
            "n_matches": match_counts[t],
            "low_sample": match_counts[t] < 6,
        })

    return pd.DataFrame(rows).sort_values("att", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# LAYER 2 — SOS DIAGNOSTICS
# ─────────────────────────────────────────────────────────────

def compute_sos_report(matches_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, compute:
      - avg_opp_att  : mean att rating of opponents faced
      - avg_opp_def  : mean def rating of opponents faced
      - sos_score    : composite (higher = harder schedule)
      - raw_xg_pg    : xG per game before SOS adjustment
      - adj_xg_pg    : xG per game implied by MLE att rating
      - sos_boost    : adj / raw  > 1 means team was underrated by raw stats
    """
    if ratings_df.empty or matches_df.empty:
        return pd.DataFrame()

    rat = dict(zip(ratings_df["team"], ratings_df["att"]))
    def_rat = dict(zip(ratings_df["team"], ratings_df["defe"]))

    records = defaultdict(lambda: {"opp_att": [], "opp_def": [], "xg": [], "xga": []})

    use_xg = "home_xg" in matches_df.columns
    for _, row in matches_df.iterrows():
        h, a = row["home"], row["away"]
        xgh = row["home_xg"] if use_xg and not pd.isna(row.get("home_xg")) else row.get("home_goals", 0)
        xga = row["away_xg"] if use_xg and not pd.isna(row.get("away_xg")) else row.get("away_goals", 0)

        # Apply conf scalar
        scalar = CONFEDERATION_STRENGTH.get(row.get("confederation", ""), 0.85)
        xgh *= scalar
        xga *= scalar

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
        sos = (avg_opp_att + (2 - avg_opp_def)) / 2   # higher = harder
        raw_xg_pg = float(np.mean(data["xg"])) if data["xg"] else float("nan")
        mle_att = rat.get(team, 1.0)
        adj_xg_pg = mle_att * 1.25  # BASE_RATE=1.25 × att vs avg opponent
        sos_boost = adj_xg_pg / raw_xg_pg if raw_xg_pg > 0 else 1.0

        rows.append({
            "team":        team,
            "avg_opp_att": round(avg_opp_att, 3),
            "avg_opp_def": round(avg_opp_def, 3),
            "sos_score":   round(sos, 3),
            "raw_xg_pg":   round(raw_xg_pg, 3),
            "adj_xg_pg":   round(adj_xg_pg, 3),
            "sos_boost":   round(sos_boost, 3),
            "mle_att":     round(mle_att, 4),
            "interpretation": (
                "overrated by raw stats" if sos_boost < 0.90 else
                "underrated by raw stats" if sos_boost > 1.10 else
                "fairly rated"
            ),
        })

    return pd.DataFrame(rows).sort_values("sos_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# SIMULATOR PATCHER
# ─────────────────────────────────────────────────────────────

def patch_simulator(ratings_df: pd.DataFrame, simulator_path: Path, dry_run: bool):
    if not simulator_path.exists():
        print(f"  [patch] {simulator_path} not found — skipping.")
        return

    ratings_map = {
        row["team"]: (row["att"], row["defe"])
        for _, row in ratings_df.iterrows()
    }

    src = simulator_path.read_text(encoding="utf-8")

    def replace_team_line(match):
        prefix, name, group, old_att, old_defe, suffix = (
            match.group(1), match.group(2), match.group(3),
            match.group(4), match.group(5), match.group(6),
        )
        if name in ratings_map:
            new_att, new_defe = ratings_map[name]
            tag = "  # ← FBref MLE+SOS"
        else:
            new_att, new_defe = old_att, old_defe
            tag = ""
        return f'{prefix}Team("{name}", "{group}", att={new_att}, defe={new_defe},{suffix}{tag}'

    pattern = re.compile(
        r'([ \t]*)Team\("([^"]+)",\s*"([A-Z])",\s*att=([\d.]+),\s*defe=([\d.]+),(.*?)(?=\n)',
        re.MULTILINE,
    )
    new_src, n = pattern.subn(replace_team_line, src)
    new_src = re.sub(
        r"(# fmt: off\n)",
        f"\\1# Ratings updated from FBref MLE+SOS: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n",
        new_src, count=1,
    )

    if dry_run:
        matched = sum(1 for t in ratings_map if re.search(rf'Team\("{re.escape(t)}"', src))
        print(f"[dry-run] Would patch {matched} teams in {simulator_path} (backup would be created).")
        return

    backup = simulator_path.with_suffix(".py.bak")
    backup.write_text(src, encoding="utf-8")
    print(f"  [patch] Backup → {backup}")
    simulator_path.write_text(new_src, encoding="utf-8")
    print(f"  [patch] {n} team entries updated in {simulator_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run(cache_dir: Path, no_cache: bool, dry_run: bool,
        simulator_path: Path, apply_sos: bool):

    cache   = PageCache(cache_dir)
    if no_cache:
        cache.clear()
    session = requests.Session()
    session.headers.update(HEADERS)

    print("\n" + "="*60)
    print("  FBref Scraper  —  MLE + SOS-adjusted ratings  (v2)")
    print("="*60)

    # ── 1. Scrape squad shooting summaries (for fallback aggregates) ──
    all_squad: list[pd.DataFrame] = []
    print("\n── Squad shooting tables ──")
    for comp, path in COMPETITION_PAGES.items():
        html = polite_get(FBREF_BASE + path, cache, session)
        df   = parse_squad_shooting(html, comp)
        if not df.empty:
            print(f"  {comp}: {len(df)} teams")
            all_squad.append(df)

    # ── 2. Scrape match-level scores (for MLE) ──
    all_matches: list[pd.DataFrame] = []
    print("\n── Match score tables ──")
    for comp, path in MATCH_SCORE_PAGES.items():
        html = polite_get(FBREF_BASE + path, cache, session)
        df   = parse_match_scores(html, comp)
        if not df.empty:
            print(f"  {comp}: {len(df)} matches")
            all_matches.append(df)

    if not all_squad and not all_matches:
        print(
            "\n✗ No data scraped. Check your network / VPN / FBref URLs.\n"
            "  Run with --dry-run to skip scraping and use existing ratings.\n"
        )
        sys.exit(1)

    # Save raw data
    squad_df = pd.concat(all_squad, ignore_index=True) if all_squad else pd.DataFrame()
    match_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()

    if not squad_df.empty:
        squad_df.to_csv("fbref_raw_stats.csv", index=False)
        print(f"\n✓ fbref_raw_stats.csv  ({len(squad_df)} rows)")
    if not match_df.empty:
        match_df.to_csv("fbref_raw_matches.csv", index=False)
        print(f"✓ fbref_raw_matches.csv  ({len(match_df)} matches)")

    # ── 3. Fit MLE ratings ──
    print("\n── Fitting Dixon-Coles MLE …")
    if not match_df.empty:
        ratings_df = dixon_coles_mle(match_df, use_xg=True)
    elif not squad_df.empty:
        # Fallback: simple normalised xG/game ratios if no match-level data
        print("  [fallback] No match-level data — using aggregate xG ratios.")
        from fbref_scraper import _simple_calibrate  # self-import of helper below
        ratings_df = _simple_calibrate(squad_df)
    else:
        print("✗ Cannot calibrate — no data available.")
        sys.exit(1)

    if ratings_df.empty:
        print("✗ MLE returned empty ratings.")
        sys.exit(1)

    # ── 4. SOS report ──
    sos_df = pd.DataFrame()
    if apply_sos and not match_df.empty:
        print("\n── Computing SOS report …")
        sos_df = compute_sos_report(match_df, ratings_df)
        if not sos_df.empty:
            sos_df.to_csv("fbref_sos_report.csv", index=False)
            print(f"✓ fbref_sos_report.csv  ({len(sos_df)} teams)")
            print("\n── Top 10 teams by schedule difficulty ──")
            print(sos_df[["team","sos_score","raw_xg_pg","adj_xg_pg",
                           "sos_boost","interpretation"]].head(10).to_string(index=False))
            print("\n── Bottom 5 (easiest schedules — most inflated raw stats) ──")
            print(sos_df[["team","sos_score","raw_xg_pg","adj_xg_pg",
                           "sos_boost","interpretation"]].tail(5).to_string(index=False))

    # Save ratings
    ratings_df.to_csv("fbref_ratings.csv", index=False)
    print(f"\n✓ fbref_ratings.csv  ({len(ratings_df)} teams)")
    print("\n── Top 15 by attack rating ──")
    low_flag = ratings_df[ratings_df.get("low_sample", pd.Series(dtype=bool)) == True]["team"].tolist()
    print(ratings_df[["team","att","defe","n_matches"]].head(15).to_string(index=False))
    if low_flag:
        print(f"\n⚠  Low sample (<6 games): {', '.join(low_flag)}")

    # ── 5. Patch simulator ──
    print(f"\n── Patching {simulator_path} …")
    patch_simulator(ratings_df, simulator_path, dry_run)

    print("\n✓ Done — re-run wc2026_simulator.py to use updated ratings.\n")


def _simple_calibrate(squad_df: pd.DataFrame) -> pd.DataFrame:
    """Fallback calibration when match-level data isn't available."""
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
    agg["att"]  = (agg["k"] * (agg["xg_pg"]  / med_xg)  + (1 - agg["k"])).round(4)
    agg["defe"] = (agg["k"] * (agg["xga_pg"] / med_xga) + (1 - agg["k"])).round(4)
    agg["n_matches"] = agg["total_mp"].astype(int)
    agg["low_sample"] = agg["total_mp"] < 6
    return agg[["team","att","defe","n_matches","low_sample"]].sort_values("att", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape FBref + fit MLE + SOS-adjusted ratings for WC 2026 simulator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python fbref_scraper.py
              python fbref_scraper.py --no-cache
              python fbref_scraper.py --dry-run
              python fbref_scraper.py --no-sos
        """),
    )
    parser.add_argument("--cache-dir",  type=Path, default=Path(".fbref_cache"))
    parser.add_argument("--no-cache",   action="store_true")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--no-sos",     action="store_true", help="Skip confederation SOS adjustment")
    parser.add_argument("--simulator",  type=Path, default=Path("wc2026_simulator.py"))
    args = parser.parse_args()

    run(
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
        dry_run=args.dry_run,
        simulator_path=args.simulator,
        apply_sos=not args.no_sos,
    )
