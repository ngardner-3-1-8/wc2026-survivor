"""
FIFA World Cup 2026 Monte Carlo Simulator
==========================================
Model: Dixon-Coles Poisson process
  - Each team has Attack (att) and Defense (def) ratings
  - Goals scored by team i vs team j ~ Poisson(λ)
    where λ_i = base_rate * att_i * def_j * home_adv (if applicable)
  - Low-score correction (Dixon-Coles rho) adjusts 0-0, 1-0, 0-1, 1-1 probabilities
  - Knockout ties resolved by penalty shootout (each team 60% chance if equal quality)

Outputs
-------
  group_results.csv        - Points, GF, GA, GD, advancement probability per team
  match_results.csv        - Win%, Draw%, Loss%, xGF, xGA per group match
  knockout_results.csv     - Win probability at each knockout round per team
  survivor_picks.csv       - Best survivor picks at each stage
  simulation_summary.txt   - Human-readable narrative summary

Usage
-----
  python wc2026_simulator.py                  # run with default 5000 sims
  python wc2026_simulator.py --sims 10000     # custom sim count
  python wc2026_simulator.py --seed 42        # reproducible
"""



import argparse
import json
from pathlib import Path
import itertools
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1.  TEAM RATINGS  (Dixon-Coles parameterisation)
#     att  = attacking strength multiplier  (1.0 = average)
#     defe = defensive strength multiplier  (lower = better defence)
#     These are calibrated against FIFA rankings + recent tournament data.
#     Adjust freely — the model is entirely rating-driven.
# ---------------------------------------------------------------------------

@dataclass
class Team:
    name: str
    group: str
    att: float   # attacking multiplier
    defe: float  # defensive multiplier (lower = stingier)
    fifa_rank: int = 999

# Base scoring rate: average top-tournament match produces ~2.5 goals total
BASE_RATE = 1.25   # each team's "neutral" lambda before modifiers

# fmt: off
# Ratings last updated: 2026-06-06 — based on FIFA April 2026 rankings + qualifying xG priors
TEAMS = [
    Team("Mexico", "A", att=1.11, defe=0.81, fifa_rank=15),
    Team("South Africa", "A", att=0.91, defe=1.05, fifa_rank=60),
    Team("South Korea", "A", att=1.09, defe=1.08, fifa_rank=25),
    Team("Czechia", "A", att=0.95, defe=1.04, fifa_rank=41),
    Team("Switzerland", "B", att=1.21, defe=0.83, fifa_rank=19),
    Team("Canada", "B", att=1.05, defe=0.92, fifa_rank=30),
    Team("Qatar", "B", att=0.84, defe=1.09, fifa_rank=55),
    Team("Bosnia", "B", att=0.89, defe=1.09, fifa_rank=65),
    Team("Brazil", "C", att=1.29, defe=0.74, fifa_rank=6),
    Team("Morocco", "C", att=1.35, defe=0.75, fifa_rank=8),
    Team("Scotland", "C", att=1.01, defe=1.07, fifa_rank=43),
    Team("Haiti", "C", att=0.81, defe=1.17, fifa_rank=83),
    Team("USA", "D", att=1.14, defe=0.9, fifa_rank=16),
    Team("Türkiye", "D", att=1.1, defe=0.93, fifa_rank=22),
    Team("Australia", "D", att=1.06, defe=0.9, fifa_rank=27),
    Team("Paraguay", "D", att=1.0, defe=0.94, fifa_rank=40),
    Team("Germany", "E", att=1.34, defe=0.78, fifa_rank=10),
    Team("Ecuador", "E", att=1.08, defe=0.81, fifa_rank=23),
    Team("Ivory Coast", "E", att=1.17, defe=0.83, fifa_rank=34),
    Team("Curaçao", "E", att=0.84, defe=1.12, fifa_rank=82),
    Team("Netherlands", "F", att=1.33, defe=0.8, fifa_rank=7),
    Team("Japan", "F", att=1.26, defe=0.96, fifa_rank=18),
    Team("Sweden", "F", att=0.93, defe=1.09, fifa_rank=38),
    Team("Tunisia", "F", att=0.98, defe=0.95, fifa_rank=44),
    Team("Belgium", "G", att=1.27, defe=0.81, fifa_rank=9),
    Team("Iran", "G", att=1.09, defe=0.98, fifa_rank=21),
    Team("Egypt", "G", att=1.1, defe=0.87, fifa_rank=29),
    Team("New Zealand", "G", att=0.79, defe=1.16, fifa_rank=85),
    Team("Spain", "H", att=1.51, defe=0.63, fifa_rank=2),
    Team("Uruguay", "H", att=1.16, defe=0.83, fifa_rank=17),
    Team("Saudi Arabia", "H", att=0.8, defe=1.05, fifa_rank=61),
    Team("Cape Verde", "H", att=0.84, defe=1.12, fifa_rank=69),
    Team("France", "I", att=1.38, defe=0.65, fifa_rank=1),
    Team("Senegal", "I", att=1.24, defe=0.79, fifa_rank=14),
    Team("Norway", "I", att=1.53, defe=0.83, fifa_rank=31),
    Team("Iraq", "I", att=0.98, defe=1.18, fifa_rank=57),
    Team("Argentina", "J", att=1.4, defe=0.63, fifa_rank=3),
    Team("Algeria", "J", att=1.16, defe=0.93, fifa_rank=28),
    Team("Austria", "J", att=1.19, defe=0.92, fifa_rank=24),
    Team("Jordan", "J", att=0.92, defe=1.11, fifa_rank=63),
    Team("Portugal", "K", att=1.35, defe=0.78, fifa_rank=5),
    Team("Colombia", "K", att=1.24, defe=0.81, fifa_rank=13),
    Team("Uzbekistan", "K", att=0.89, defe=1.01, fifa_rank=50),
    Team("DR Congo", "K", att=0.96, defe=1.03, fifa_rank=46),
    Team("England", "L", att=1.36, defe=0.67, fifa_rank=4),
    Team("Croatia", "L", att=1.22, defe=0.91, fifa_rank=11),
    Team("Panama", "L", att=1.05, defe=1.03, fifa_rank=33),
    Team("Ghana", "L", att=1.01, defe=1.01, fifa_rank=74),
]
# fmt: on

TEAM_MAP: dict[str, Team] = {t.name: t for t in TEAMS}

# ---------------------------------------------------------------------------
# 2.  TOURNAMENT STRUCTURE
# ---------------------------------------------------------------------------

# Group matchday schedule (each pair plays once)
def build_group_fixtures(teams: list[Team]) -> list[tuple[str, str]]:
    names = [t.name for t in teams]
    return list(itertools.combinations(names, 2))

# WC 2026: 12 groups of 4 → top 2 advance + 8 best 3rd-place teams (32 total)
GROUPS: dict[str, list[Team]] = defaultdict(list)
for t in TEAMS:
    GROUPS[t.group].append(t)

# 3rd-place advancement: 8 of 12 third-place teams qualify
THIRD_PLACE_ADVANCE_COUNT = 8

# ---------------------------------------------------------------------------
# 3.  DIXON-COLES POISSON MATCH SIMULATOR
# ---------------------------------------------------------------------------

RHO = -0.13   # low-score correlation parameter (standard calibrated value)

def _dc_correction(g1: int, g2: int, l1: float, l2: float, rho: float) -> float:
    """Dixon-Coles correction for {0-0, 1-0, 0-1, 1-1}."""
    if g1 == 0 and g2 == 0:
        return 1 - l1 * l2 * rho
    elif g1 == 1 and g2 == 0:
        return 1 + l2 * rho
    elif g1 == 0 and g2 == 1:
        return 1 + l1 * rho
    elif g1 == 1 and g2 == 1:
        return 1 - rho
    return 1.0

def simulate_match(
    home: Team,
    away: Team,
    rng: np.random.Generator,
    knockout: bool = False,
) -> tuple[int, int]:
    """
    Simulate a single match.
    Returns (home_goals, away_goals).
    In knockout mode, if tied after 90 min → penalties.
    """
    lam_home = BASE_RATE * home.att * away.defe
    lam_away = BASE_RATE * away.att * home.defe

    # Poisson draw with Dixon-Coles correction via accept-reject
    max_goals = 10
    while True:
        g_h = rng.poisson(lam_home)
        g_a = rng.poisson(lam_away)
        if g_h > max_goals or g_a > max_goals:
            continue
        correction = _dc_correction(g_h, g_a, lam_home, lam_away, RHO)
        # correction is close to 1; accept/reject with small adjustment
        u = rng.uniform(0, 1.05)  # slightly > 1 to handle correction > 1
        if u <= correction:
            break

    if knockout and g_h == g_a:
        # Penalty shootout: slight advantage to stronger attack team
        home_pen_prob = 0.5 + 0.05 * (home.att - away.att)
        home_pen_prob = np.clip(home_pen_prob, 0.35, 0.65)
        if rng.random() < home_pen_prob:
            g_h += 1  # encode winner with +1 (we track pens separately)
        else:
            g_a += 1
        return g_h, g_a   # g_h != g_a guaranteed

    return g_h, g_a

def expected_goals(home: Team, away: Team) -> tuple[float, float]:
    """Analytical expected goals (used for display)."""
    return BASE_RATE * home.att * away.defe, BASE_RATE * away.att * home.defe

# ---------------------------------------------------------------------------
# 4.  GROUP STAGE SIMULATION
# ---------------------------------------------------------------------------

@dataclass
class TeamGroupStats:
    name: str
    pts: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga

def simulate_group(
    group_teams: list[Team],
    rng: np.random.Generator,
    match_trackers: dict = None,
    locked_stats: dict = None,
    locked_pairs: set = None,
) -> tuple[list[str], dict]:
    """
    Simulate a group. Returns (sorted_team_names, stats_dict).
    locked_stats: pre-computed pts/gf/ga from completed matches.
    locked_pairs: set of (home,away) tuples already played — skip simulation for these.
    """
    # Seed stats from locked results if available.
    # IMPORTANT: copy the locked TeamGroupStats so we don't mutate the shared
    # template across simulations (each sim must start from the same locked base).
    if locked_stats:
        stats = {}
        for t in group_teams:
            if t.name in locked_stats:
                src = locked_stats[t.name]
                stats[t.name] = TeamGroupStats(src.name, src.pts, src.gf, src.ga)
            else:
                stats[t.name] = TeamGroupStats(t.name)
    else:
        stats = {t.name: TeamGroupStats(t.name) for t in group_teams}

    fixtures = build_group_fixtures(group_teams)
    for h_name, a_name in fixtures:
        # Skip already-played matches (check both home/away orderings, since
        # build_group_fixtures order may differ from the real fixture order)
        if locked_pairs and (
            (h_name, a_name) in locked_pairs or (a_name, h_name) in locked_pairs
        ):
            continue

        h_team = TEAM_MAP[h_name]
        a_team = TEAM_MAP[a_name]
        gh, ga = simulate_match(h_team, a_team, rng)

        stats[h_name].gf += gh
        stats[h_name].ga += ga
        stats[a_name].gf += ga
        stats[a_name].ga += gh

        if gh > ga:
            stats[h_name].pts += 3
        elif ga > gh:
            stats[a_name].pts += 3
        else:
            stats[h_name].pts += 1
            stats[a_name].pts += 1

        if match_trackers is not None:
            key = tuple(sorted([h_name, a_name]))
            match_trackers["home_wins"][key]  += (1 if gh > ga else 0)
            match_trackers["draws"][key]      += (1 if gh == ga else 0)
            match_trackers["away_wins"][key]  += (1 if ga > gh else 0)
            match_trackers["total_gf"][key]   += gh
            match_trackers["total_ga"][key]   += ga
            match_trackers["count"][key]      += 1

    def sort_key(name: str):
        s = stats[name]
        return (s.pts, s.gd, s.gf, rng.random())

    sorted_names = sorted(stats.keys(), key=sort_key, reverse=True)
    return sorted_names, stats

# ---------------------------------------------------------------------------
# 5.  KNOCKOUT BRACKET  (Round of 32 → Final)
# ---------------------------------------------------------------------------

# Standard WC 2026 R32 bracket seeding pattern
# Groups A-L, top 2 advance; 8 best 3rd place also advance
# For simulation purposes we use a plausible bracket assignment
# (exact draw happens post-group stage in real tournament)

def pick_best_third(third_place_teams: list[tuple[str, float]]) -> list[str]:
    """Pick top-8 third-place teams by average points (simulation-weighted)."""
    return [t[0] for t in sorted(third_place_teams, key=lambda x: -x[1])[:8]]

# ---------------------------------------------------------------------------
# 5b. LOCKED RESULTS LOADER
#
# results.json (optional) lives next to the simulator and locks in completed
# match results so the simulator treats them as facts rather than simulating.
#
# Format:
# {
#   "group_matches": [
#     {"home": "Germany", "away": "Scotland", "home_goals": 5, "away_goals": 1},
#     ...
#   ],
#   "knockout_results": [
#     {"match_id": "M73", "winner": "Canada"},
#     ...
#   ]
# }
# ---------------------------------------------------------------------------

def load_results(results_path: Path) -> dict:
    """Load locked results from results.json if it exists."""
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        group_matches = data.get("group_matches", [])
        knockout      = data.get("knockout_results", [])
        print(f"  [results] Loaded {len(group_matches)} completed group matches, "
              f"{len(knockout)} knockout results from {results_path.name}")
        return data
    return {"group_matches": [], "knockout_results": []}

def build_locked_group_stats(group_matches: list[dict]) -> dict[str, dict[str, "TeamGroupStats"]]:
    """
    Pre-compute pts/gf/ga from completed matches, keyed by group.
    Returns {group_letter: {team_name: TeamGroupStats}}.
    """
    from collections import defaultdict
    locked: dict[str, dict] = defaultdict(dict)
    for m in group_matches:
        home, away = m["home"], m["away"]
        gh, ga     = int(m["home_goals"]), int(m["away_goals"])
        grp = TEAM_MAP[home].group if home in TEAM_MAP else None
        if grp is None:
            continue
        for name in (home, away):
            if name not in locked[grp]:
                locked[grp][name] = TeamGroupStats(name)
        locked[grp][home].gf += gh
        locked[grp][home].ga += ga
        locked[grp][away].gf += ga
        locked[grp][away].ga += gh
        if gh > ga:
            locked[grp][home].pts += 3
        elif ga > gh:
            locked[grp][away].pts += 3
        else:
            locked[grp][home].pts += 1
            locked[grp][away].pts += 1
    return dict(locked)

def get_locked_pairs(group_matches: list[dict]) -> set[tuple[str,str]]:
    """Return set of (home,away) pairs that are already completed."""
    return {(m["home"], m["away"]) for m in group_matches}

# ---------------------------------------------------------------------------
# 6.  MAIN SIMULATION LOOP
# ---------------------------------------------------------------------------

def run_simulation(n_sims: int, seed: Optional[int] = None,
                   results_path: Optional[Path] = None) -> dict:
    rng = np.random.default_rng(seed)

    # Load locked results (completed matches)
    if results_path is None:
        results_path = Path(__file__).parent / "results.json"
    locked_data        = load_results(results_path)
    locked_group_matches = locked_data.get("group_matches", [])
    locked_ko_results  = {r["match_id"]: r["winner"]
                          for r in locked_data.get("knockout_results", [])}
    locked_pairs       = get_locked_pairs(locked_group_matches)
    locked_group_stats = build_locked_group_stats(locked_group_matches)

    # Accumulation structures
    group_names = sorted(GROUPS.keys())
    all_team_names = [t.name for t in TEAMS]

    # --- Group stage trackers ---
    group_pts_acc   = defaultdict(list)   # team → [pts per sim]
    group_gf_acc    = defaultdict(list)
    group_ga_acc    = defaultdict(list)
    group_rank_acc  = defaultdict(list)   # team → [rank in group per sim]
    group_adv_count = defaultdict(int)    # team → sims where they advanced

    # --- Match trackers (group stage) ---
    match_home_wins  = defaultdict(int)
    match_draws      = defaultdict(int)
    match_away_wins  = defaultdict(int)
    match_total_gf   = defaultdict(float)
    match_total_ga   = defaultdict(float)
    match_count      = defaultdict(int)

    # --- Knockout trackers ---
    # stages: r32, r16, qf, sf, final, champion
    ko_stage_reach = defaultdict(lambda: defaultdict(int))

    # --- Survivor trackers (win/advance probability by stage) ---

    print(f"\n{'='*60}")
    print(f"  FIFA World Cup 2026 Monte Carlo Simulator")
    print(f"  Simulations: {n_sims:,}   Seed: {seed}")
    print(f"{'='*60}\n")

    for sim_idx in tqdm(range(n_sims), desc="Simulating", unit="sim"):

        # ── GROUP STAGE ──
        group_standings: dict[str, list[str]] = {}  # group → [1st, 2nd, 3rd, 4th]
        third_place_data: list[tuple[str, int]] = []  # (team, pts)

        sim_pts: dict[str, int] = {}
        sim_gf:  dict[str, int] = {}
        sim_ga:  dict[str, int] = {}

        match_trackers = {
            "home_wins": match_home_wins,
            "draws":     match_draws,
            "away_wins": match_away_wins,
            "total_gf":  match_total_gf,
            "total_ga":  match_total_ga,
            "count":     match_count,
        }

        for grp in group_names:
            teams_in_group = GROUPS[grp]
            sorted_names, stats_lookup = simulate_group(
                teams_in_group, rng,
                match_trackers=match_trackers,
                locked_stats=locked_group_stats.get(grp),
                locked_pairs=locked_pairs,
            )
            group_standings[grp] = sorted_names

            for rank_idx, tname in enumerate(sorted_names):
                s = stats_lookup[tname]
                sim_pts[tname] = s.pts
                sim_gf[tname]  = s.gf
                sim_ga[tname]  = s.ga
                group_rank_acc[tname].append(rank_idx + 1)
                group_pts_acc[tname].append(s.pts)
                group_gf_acc[tname].append(s.gf)
                group_ga_acc[tname].append(s.ga)

                if rank_idx < 2:
                    group_adv_count[tname] += 1  # top-2 guaranteed advance

            third_place_data.append((sorted_names[2], sim_pts[sorted_names[2]]))

        # Select best 8 third-place teams (ranked by pts, then GD, then GF)
        third_ranked = sorted(
            third_place_data,
            key=lambda x: (-x[1], -(sim_gf.get(x[0], 0) - sim_ga.get(x[0], 0)), -sim_gf.get(x[0], 0), rng.random())
        )
        best_thirds = [t for t, _ in third_ranked[:8]]
        # Track which groups the best 8 third-place teams came from
        best_third_groups = set(TEAM_MAP[t].group for t in best_thirds)
        adv_thirds_by_group = {TEAM_MAP[t].group: t for t in best_thirds}
        for tname in best_thirds:
            group_adv_count[tname] += 1

        # All 32 qualifiers
        qualifiers: list[str] = []
        for grp in group_names:
            qualifiers.append(group_standings[grp][0])  # 1st
            qualifiers.append(group_standings[grp][1])  # 2nd
        qualifiers.extend(best_thirds)

        for q in qualifiers:
            ko_stage_reach[q]["r32"] += 1

        # ── KNOCKOUT STAGE — Real FIFA R32 bracket ──
        # R32 matchups per FIFA regulations (Match 73-88):
        # The third-place slot assignments depend on which 8 groups' 3rd-place
        # teams qualified. We look up the correct combination from the FIFA matrix.
        # Reference: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage
        #
        # Slot legend: 1X = winner of group X, 2X = runner-up of group X, 3X = 3rd of group X
        # Fixed slots (not dependent on 3rd-place combos):
        #   M73: 2A vs 2B        M74: 1E vs [3rd slot]   M75: 1F vs 2C
        #   M76: 1C vs 2F        M77: 1I vs [3rd slot]   M78: 2E vs 2I
        #   M79: 1A vs [3rd]     M80: 1L vs [3rd]        M81: 1D vs [3rd]
        #   M82: 1G vs [3rd]     M83: 2K vs 2L           M84: 1H vs 2J
        #   M85: 1B vs [3rd]     M86: 1J vs 2H           M87: 1K vs [3rd]
        #   M88: 2D vs 2G

        def resolve_3rd_slots(adv_groups: set) -> dict:
            """
            Given the set of groups whose 3rd-place teams advanced,
            look up the correct 3rd-place slot assignment from the FIFA matrix.
            Returns dict: slot_label -> team_name
            The 495 combinations are encoded via the group-set key.
            We implement the most common cases; others fall back to best-available.
            """
            # The 8 advancing groups sorted
            g = tuple(sorted(adv_groups))

            # FIFA matrix — maps frozenset of 8 advancing groups to
            # (1A_3rd, 1B_3rd, 1D_3rd, 1E_3rd, 1G_3rd, 1I_3rd, 1K_3rd, 1L_3rd)
            # These are the groups whose 1st-place team faces a 3rd-place team.
            # Source: Annex C of FIFA 2026 regulations (combinations 1-45 cover A-F groups)


            # Define the static slot names immediately at the top of the scope
            slot_names = ['1A', '1B', '1D', '1E', '1G', '1I', '1K', '1L']
            
            # The 8 advancing groups sorted into a standard lookup key
            key = tuple(sorted(adv_groups))
          
            matrix = {
              ('E','F','G','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'F','1G':'H','1I':'G','1K':'L','1L':'K'},
              ('D','F','G','H','I','J','K','L'): {'1A':'H','1B':'G','1D':'I','1E':'D','1G':'J','1I':'F','1K':'L','1L':'K'},
              ('D','E','G','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'D','1G':'H','1I':'G','1K':'L','1L':'K'},
              ('D','E','F','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('D','E','F','G','I','J','K','L'): {'1A':'E','1B':'G','1D':'I','1E':'D','1G':'J','1I':'F','1K':'L','1L':'K'},
              ('D','E','F','G','H','J','K','L'): {'1A':'E','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('D','E','F','G','H','I','K','L'): {'1A':'E','1B':'G','1D':'I','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('D','E','F','G','H','I','J','L'): {'1A':'E','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'L','1L':'I'},
              ('D','E','F','G','H','I','J','K'): {'1A':'E','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'I','1L':'K'},
              ('C','F','G','H','I','J','K','L'): {'1A':'H','1B':'G','1D':'I','1E':'C','1G':'J','1I':'F','1K':'L','1L':'K'},
              ('C','E','G','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'C','1G':'H','1I':'G','1K':'L','1L':'K'},
              ('C','E','F','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'C','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('C','E','F','G','I','J','K','L'): {'1A':'E','1B':'G','1D':'I','1E':'C','1G':'J','1I':'F','1K':'L','1L':'K'},
              ('C','E','F','G','H','J','K','L'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('C','E','F','G','H','I','K','L'): {'1A':'E','1B':'G','1D':'I','1E':'C','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('C','E','F','G','H','I','J','L'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'H','1I':'F','1K':'L','1L':'I'},
              ('C','E','F','G','H','I','J','K'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'H','1I':'F','1K':'I','1L':'K'},
              ('C','D','G','H','I','J','K','L'): {'1A':'H','1B':'G','1D':'I','1E':'C','1G':'J','1I':'D','1K':'L','1L':'K'},
              ('C','D','F','H','I','J','K','L'): {'1A':'C','1B':'J','1D':'I','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('C','D','F','G','I','J','K','L'): {'1A':'C','1B':'G','1D':'I','1E':'D','1G':'J','1I':'F','1K':'L','1L':'K'},
              ('C','D','F','G','H','J','K','L'): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('C','D','F','G','H','I','K','L'): {'1A':'C','1B':'G','1D':'I','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('C','D','F','G','H','I','J','L'): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'L','1L':'I'},
              ('C','D','F','G','H','I','J','K'): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'I','1L':'K'},
              ('C','D','E','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'C','1G':'H','1I':'D','1K':'L','1L':'K'},
              ('C','D','E','G','I','J','K','L'): {'1A':'E','1B':'G','1D':'I','1E':'C','1G':'J','1I':'D','1K':'L','1L':'K'},
              ('C','D','E','G','H','J','K','L'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'H','1I':'D','1K':'L','1L':'K'},
              ('C','D','E','G','H','I','K','L'): {'1A':'E','1B':'G','1D':'I','1E':'C','1G':'H','1I':'D','1K':'L','1L':'K'},
              ('C','D','E','G','H','I','J','L'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'H','1I':'D','1K':'L','1L':'I'},
              ('C','D','E','G','H','I','J','K'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'H','1I':'D','1K':'I','1L':'K'},
              ('C','D','E','F','I','J','K','L'): {'1A':'C','1B':'J','1D':'E','1E':'D','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('C','D','E','F','H','J','K','L'): {'1A':'C','1B':'J','1D':'E','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('C','D','E','F','H','I','K','L'): {'1A':'C','1B':'E','1D':'I','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('C','D','E','F','H','I','J','L'): {'1A':'C','1B':'J','1D':'E','1E':'D','1G':'H','1I':'F','1K':'L','1L':'I'},
              ('C','D','E','F','H','I','J','K'): {'1A':'C','1B':'J','1D':'E','1E':'D','1G':'H','1I':'F','1K':'I','1L':'K'},
              ('C','D','E','F','G','J','K','L'): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'J','1I':'F','1K':'L','1L':'K'},
              ('C','D','E','F','G','I','K','L'): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('C','D','E','F','G','I','J','L'): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'J','1I':'F','1K':'L','1L':'I'},
              ('C','D','E','F','G','I','J','K'): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'J','1I':'F','1K':'I','1L':'K'},
              ('C','D','E','F','G','H','K','L'): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('C','D','E','F','G','H','J','L'): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'L','1L':'E'},
              ('C','D','E','F','G','H','J','K'): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'E','1L':'K'},
              ('C','D','E','F','G','H','I','L'): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'H','1I':'F','1K':'L','1L':'I'},
              ('C','D','E','F','G','H','I','K'): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'H','1I':'F','1K':'I','1L':'K'},
              ('C','D','E','F','G','H','I','J'): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'E','1L':'I'},
              ('B','F','G','H','I','J','K','L'): {'1A':'H','1B':'J','1D':'B','1E':'F','1G':'I','1I':'G','1K':'L','1L':'K'},
              ('B','E','G','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'B','1G':'H','1I':'G','1K':'L','1L':'K'},
              ('B','E','F','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'F','1G':'I','1I':'H','1K':'L','1L':'K'},
              ('B','E','F','G','I','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'F','1G':'I','1I':'G','1K':'L','1L':'K'},
              ('B','E','F','G','H','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'F','1G':'H','1I':'G','1K':'L','1L':'K'},
              ('B','E','F','G','H','I','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'F','1G':'I','1I':'H','1K':'L','1L':'K'},
              ('B','E','F','G','H','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'F','1G':'H','1I':'G','1K':'L','1L':'I'},
              ('B','E','F','G','H','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'F','1G':'H','1I':'G','1K':'I','1L':'K'},
              ('B','D','G','H','I','J','K','L'): {'1A':'H','1B':'J','1D':'B','1E':'D','1G':'I','1I':'G','1K':'L','1L':'K'},
              ('B','D','F','H','I','J','K','L'): {'1A':'H','1B':'J','1D':'B','1E':'D','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('B','D','F','G','I','J','K','L'): {'1A':'I','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'L','1L':'K'},
              ('B','D','F','G','H','J','K','L'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'L','1L':'K'},
              ('B','D','F','G','H','I','K','L'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('B','D','F','G','H','I','J','L'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'L','1L':'I'},
              ('B','D','F','G','H','I','J','K'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'I','1L':'K'},
              ('B','D','E','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'I','1I':'H','1K':'L','1L':'K'},
              ('B','D','E','G','I','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'I','1I':'G','1K':'L','1L':'K'},
              ('B','D','E','G','H','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'H','1I':'G','1K':'L','1L':'K'},
              ('B','D','E','G','H','I','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'I','1I':'H','1K':'L','1L':'K'},
              ('B','D','E','G','H','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'H','1I':'G','1K':'L','1L':'I'},
              ('B','D','E','G','H','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'H','1I':'G','1K':'I','1L':'K'},
              ('B','D','E','F','I','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('B','D','E','F','H','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('B','D','E','F','H','I','K','L'): {'1A':'E','1B':'I','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('B','D','E','F','H','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'I'},
              ('B','D','E','F','H','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'H','1I':'F','1K':'I','1L':'K'},
              ('B','D','E','F','G','J','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'L','1L':'K'},
              ('B','D','E','F','G','I','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('B','D','E','F','G','I','J','L'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'L','1L':'I'},
              ('B','D','E','F','G','I','J','K'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'I','1L':'K'},
              ('B','D','E','F','G','H','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('B','D','E','F','G','H','J','L'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'L','1L':'E'},
              ('B','D','E','F','G','H','J','K'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'E','1L':'K'},
              ('B','D','E','F','G','H','I','L'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'I'},
              ('B','D','E','F','G','H','I','K'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'H','1I':'F','1K':'I','1L':'K'},
              ('B','D','E','F','G','H','I','J'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'E','1L':'I'},
              ('B','C','G','H','I','J','K','L'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'I','1I':'G','1K':'L','1L':'K'},
              ('B','C','F','H','I','J','K','L'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('B','C','F','G','I','J','K','L'): {'1A':'I','1B':'G','1D':'B','1E':'C','1G':'J','1I':'F','1K':'L','1L':'K'},
              ('B','C','F','G','H','J','K','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'F','1K':'L','1L':'K'},
              ('B','C','F','G','H','I','K','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('B','C','F','G','H','I','J','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'F','1K':'L','1L':'I'},
              ('B','C','F','G','H','I','J','K'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'F','1K':'I','1L':'K'},
              ('B','C','E','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'I','1I':'H','1K':'L','1L':'K'},
              ('B','C','E','G','I','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'I','1I':'G','1K':'L','1L':'K'},
              ('B','C','E','G','H','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'H','1I':'G','1K':'L','1L':'K'},
              ('B','C','E','G','H','I','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'I','1I':'H','1K':'L','1L':'K'},
              ('B','C','E','G','H','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'H','1I':'G','1K':'L','1L':'I'},
              ('B','C','E','G','H','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'H','1I':'G','1K':'I','1L':'K'},
              ('B','C','E','F','I','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('B','C','E','F','H','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('B','C','E','F','H','I','K','L'): {'1A':'E','1B':'I','1D':'B','1E':'C','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('B','C','E','F','H','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'H','1I':'F','1K':'L','1L':'I'},
              ('B','C','E','F','H','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'H','1I':'F','1K':'I','1L':'K'},
              ('B','C','E','F','G','J','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'J','1I':'F','1K':'L','1L':'K'},
              ('B','C','E','F','G','I','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('B','C','E','F','G','I','J','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'J','1I':'F','1K':'L','1L':'I'},
              ('B','C','E','F','G','I','J','K'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'J','1I':'F','1K':'I','1L':'K'},
              ('B','C','E','F','G','H','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('B','C','E','F','G','H','J','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'F','1K':'L','1L':'E'},
              ('B','C','E','F','G','H','J','K'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'F','1K':'E','1L':'K'},
              ('B','C','E','F','G','H','I','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'H','1I':'F','1K':'L','1L':'I'},
              ('B','C','E','F','G','H','I','K'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'H','1I':'F','1K':'I','1L':'K'},
              ('B','C','E','F','G','H','I','J'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'F','1K':'E','1L':'I'},
              ('B','C','D','H','I','J','K','L'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'I','1I':'D','1K':'L','1L':'K'},
              ('B','C','D','G','I','J','K','L'): {'1A':'I','1B':'G','1D':'B','1E':'C','1G':'J','1I':'D','1K':'L','1L':'K'},
              ('B','C','D','G','H','J','K','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'D','1K':'L','1L':'K'},
              ('B','C','D','G','H','I','K','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'I','1I':'D','1K':'L','1L':'K'},
              ('B','C','D','G','H','I','J','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'D','1K':'L','1L':'I'},
              ('B','C','D','G','H','I','J','K'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'D','1K':'I','1L':'K'},
              ('B','C','D','F','I','J','K','L'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('B','C','D','F','H','J','K','L'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('B','C','D','F','H','I','K','L'): {'1A':'C','1B':'I','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('B','C','D','F','H','I','J','L'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'I'},
              ('B','C','D','F','H','I','J','K'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'H','1I':'F','1K':'I','1L':'K'},
              ('B','C','D','F','G','J','K','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'L','1L':'K'},
              ('B','C','D','F','G','I','K','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('B','C','D','F','G','I','J','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'L','1L':'I'},
              ('B','C','D','F','G','I','J','K'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'I','1L':'K'},
              ('B','C','D','F','G','H','K','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('B','C','D','F','G','H','J','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'J'},
              ('B','C','D','F','G','H','J','K'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'F','1K':'D','1L':'K'},
              ('B','C','D','F','G','H','I','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'I'},
              ('B','C','D','F','G','H','I','K'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'H','1I':'F','1K':'I','1L':'K'},
              ('B','C','D','F','G','H','I','J'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'F','1K':'D','1L':'I'},
              ('B','C','D','E','I','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'I','1I':'D','1K':'L','1L':'K'},
              ('B','C','D','E','H','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'H','1I':'D','1K':'L','1L':'K'},
              ('B','C','D','E','H','I','K','L'): {'1A':'E','1B':'I','1D':'B','1E':'C','1G':'H','1I':'D','1K':'L','1L':'K'},
              ('B','C','D','E','H','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'H','1I':'D','1K':'L','1L':'I'},
              ('B','C','D','E','H','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'H','1I':'D','1K':'I','1L':'K'},
              ('B','C','D','E','G','J','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'J','1I':'D','1K':'L','1L':'K'},
              ('B','C','D','E','G','I','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'I','1I':'D','1K':'L','1L':'K'},
              ('B','C','D','E','G','I','J','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'J','1I':'D','1K':'L','1L':'I'},
              ('B','C','D','E','G','I','J','K'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'J','1I':'D','1K':'I','1L':'K'},
              ('B','C','D','E','G','H','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'H','1I':'D','1K':'L','1L':'K'},
              ('B','C','D','E','G','H','J','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'D','1K':'L','1L':'E'},
              ('B','C','D','E','G','H','J','K'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'D','1K':'E','1L':'K'},
              ('B','C','D','E','G','H','I','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'H','1I':'D','1K':'L','1L':'I'},
              ('B','C','D','E','G','H','I','K'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'H','1I':'D','1K':'I','1L':'K'},
              ('B','C','D','E','G','H','I','J'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'D','1K':'E','1L':'I'},
              ('B','C','D','E','F','J','K','L'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'E','1I':'F','1K':'L','1L':'K'},
              ('B','C','D','E','F','I','K','L'): {'1A':'C','1B':'E','1D':'B','1E':'D','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('B','C','D','E','F','I','J','L'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'E','1I':'F','1K':'L','1L':'I'},
              ('B','C','D','E','F','I','J','K'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'E','1I':'F','1K':'I','1L':'K'},
              ('B','C','D','E','F','H','K','L'): {'1A':'C','1B':'E','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
              ('B','C','D','E','F','H','J','L'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'E'},
              ('B','C','D','E','F','H','J','K'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'H','1I':'F','1K':'E','1L':'K'},
              ('B','C','D','E','F','H','I','L'): {'1A':'C','1B':'E','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'I'},
              ('B','C','D','E','F','H','I','K'): {'1A':'C','1B':'E','1D':'B','1E':'D','1G':'H','1I':'F','1K':'I','1L':'K'},
              ('B','C','D','E','F','H','I','J'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'H','1I':'F','1K':'E','1L':'I'},
              ('B','C','D','E','F','G','K','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'E','1I':'F','1K':'L','1L':'K'},
              ('B','C','D','E','F','G','J','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'L','1L':'E'},
              ('B','C','D','E','F','G','J','K'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'E','1L':'K'},
              ('B','C','D','E','F','G','I','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'E','1I':'F','1K':'L','1L':'I'},
              ('B','C','D','E','F','G','I','K'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'E','1I':'F','1K':'I','1L':'K'},
              ('B','C','D','E','F','G','I','J'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'J','1I':'F','1K':'E','1L':'I'},
              ('B','C','D','E','F','G','H','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'H','1I':'F','1K':'L','1L':'E'},
              ('B','C','D','E','F','G','H','K'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'H','1I':'F','1K':'E','1L':'K'},
              ('B','C','D','E','F','G','H','J'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'J','1I':'F','1K':'D','1L':'E'},
              ('B','C','D','E','F','G','H','I'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'H','1I':'F','1K':'E','1L':'I'},
              ('A','F','G','H','I','J','K','L'): {'1A':'H','1B':'J','1D':'I','1E':'F','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','E','G','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'A','1G':'H','1I':'G','1K':'L','1L':'K'},
              ('A','E','F','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'F','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','E','F','G','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'F','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','E','F','G','H','J','K','L'): {'1A':'E','1B':'G','1D':'J','1E':'F','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','E','F','G','H','I','K','L'): {'1A':'E','1B':'G','1D':'I','1E':'F','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','E','F','G','H','I','J','L'): {'1A':'E','1B':'G','1D':'J','1E':'F','1G':'A','1I':'H','1K':'L','1L':'I'},
              ('A','E','F','G','H','I','J','K'): {'1A':'E','1B':'G','1D':'J','1E':'F','1G':'A','1I':'H','1K':'I','1L':'K'},
              ('A','D','G','H','I','J','K','L'): {'1A':'H','1B':'J','1D':'I','1E':'D','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','D','F','H','I','J','K','L'): {'1A':'H','1B':'J','1D':'I','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','D','F','G','I','J','K','L'): {'1A':'I','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','D','F','G','H','J','K','L'): {'1A':'H','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','D','F','G','H','I','K','L'): {'1A':'H','1B':'G','1D':'I','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','D','F','G','H','I','J','L'): {'1A':'H','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','D','F','G','H','I','J','K'): {'1A':'H','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','D','E','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'D','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','D','E','G','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'D','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','D','E','G','H','J','K','L'): {'1A':'E','1B':'G','1D':'J','1E':'D','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','D','E','G','H','I','K','L'): {'1A':'E','1B':'G','1D':'I','1E':'D','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','D','E','G','H','I','J','L'): {'1A':'E','1B':'G','1D':'J','1E':'D','1G':'A','1I':'H','1K':'L','1L':'I'},
              ('A','D','E','G','H','I','J','K'): {'1A':'E','1B':'G','1D':'J','1E':'D','1G':'A','1I':'H','1K':'I','1L':'K'},
              ('A','D','E','F','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','D','E','F','H','J','K','L'): {'1A':'H','1B':'J','1D':'E','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','D','E','F','H','I','K','L'): {'1A':'H','1B':'E','1D':'I','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','D','E','F','H','I','J','L'): {'1A':'H','1B':'J','1D':'E','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','D','E','F','H','I','J','K'): {'1A':'H','1B':'J','1D':'E','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','D','E','F','G','J','K','L'): {'1A':'E','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','D','E','F','G','I','K','L'): {'1A':'E','1B':'G','1D':'I','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','D','E','F','G','I','J','L'): {'1A':'E','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','D','E','F','G','I','J','K'): {'1A':'E','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','D','E','F','G','H','K','L'): {'1A':'H','1B':'G','1D':'E','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','D','E','F','G','H','J','L'): {'1A':'H','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'L','1L':'E'},
              ('A','D','E','F','G','H','J','K'): {'1A':'H','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'E','1L':'K'},
              ('A','D','E','F','G','H','I','L'): {'1A':'H','1B':'G','1D':'E','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','D','E','F','G','H','I','K'): {'1A':'H','1B':'G','1D':'E','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','D','E','F','G','H','I','J'): {'1A':'H','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'E','1L':'I'},
              ('A','C','G','H','I','J','K','L'): {'1A':'H','1B':'J','1D':'I','1E':'C','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','C','F','H','I','J','K','L'): {'1A':'H','1B':'J','1D':'I','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','F','G','I','J','K','L'): {'1A':'I','1B':'G','1D':'J','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','F','G','H','J','K','L'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','F','G','H','I','K','L'): {'1A':'H','1B':'G','1D':'I','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','F','G','H','I','J','L'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','C','F','G','H','I','J','K'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','C','E','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'C','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','C','E','G','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'C','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','C','E','G','H','J','K','L'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','C','E','G','H','I','K','L'): {'1A':'E','1B':'G','1D':'I','1E':'C','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','C','E','G','H','I','J','L'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'A','1I':'H','1K':'L','1L':'I'},
              ('A','C','E','G','H','I','J','K'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'A','1I':'H','1K':'I','1L':'K'},
              ('A','C','E','F','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','E','F','H','J','K','L'): {'1A':'H','1B':'J','1D':'E','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','E','F','H','I','K','L'): {'1A':'H','1B':'E','1D':'I','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','E','F','H','I','J','L'): {'1A':'H','1B':'J','1D':'E','1E':'C','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','C','E','F','H','I','J','K'): {'1A':'H','1B':'J','1D':'E','1E':'C','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','C','E','F','G','J','K','L'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','E','F','G','I','K','L'): {'1A':'E','1B':'G','1D':'I','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','E','F','G','I','J','L'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','C','E','F','G','I','J','K'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','C','E','F','G','H','K','L'): {'1A':'H','1B':'G','1D':'E','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','E','F','G','H','J','L'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'F','1K':'L','1L':'E'},
              ('A','C','E','F','G','H','J','K'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'F','1K':'E','1L':'K'},
              ('A','C','E','F','G','H','I','L'): {'1A':'H','1B':'G','1D':'E','1E':'C','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','C','E','F','G','H','I','K'): {'1A':'H','1B':'G','1D':'E','1E':'C','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','C','E','F','G','H','I','J'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'F','1K':'E','1L':'I'},
              ('A','C','D','H','I','J','K','L'): {'1A':'H','1B':'J','1D':'I','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','C','D','G','I','J','K','L'): {'1A':'I','1B':'G','1D':'J','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','C','D','G','H','J','K','L'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','C','D','G','H','I','K','L'): {'1A':'H','1B':'G','1D':'I','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','C','D','G','H','I','J','L'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'D','1K':'L','1L':'I'},
              ('A','C','D','G','H','I','J','K'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'D','1K':'I','1L':'K'},
              ('A','C','D','F','I','J','K','L'): {'1A':'C','1B':'J','1D':'I','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','D','F','H','J','K','L'): {'1A':'H','1B':'J','1D':'F','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','C','D','F','H','I','K','L'): {'1A':'H','1B':'F','1D':'I','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','C','D','F','H','I','J','L'): {'1A':'H','1B':'J','1D':'F','1E':'C','1G':'A','1I':'D','1K':'L','1L':'I'},
              ('A','C','D','F','H','I','J','K'): {'1A':'H','1B':'J','1D':'F','1E':'C','1G':'A','1I':'D','1K':'I','1L':'K'},
              ('A','C','D','F','G','J','K','L'): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','D','F','G','I','K','L'): {'1A':'C','1B':'G','1D':'I','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','D','F','G','I','J','L'): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','C','D','F','G','I','J','K'): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','C','D','F','G','H','K','L'): {'1A':'H','1B':'G','1D':'F','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','C','D','F','G','H','J','L'): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'L','1L':'H'},
              ('A','C','D','F','G','H','J','K'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'F','1K':'D','1L':'K'},
              ('A','C','D','F','G','H','I','L'): {'1A':'H','1B':'G','1D':'F','1E':'C','1G':'A','1I':'D','1K':'L','1L':'I'},
              ('A','C','D','F','G','H','I','K'): {'1A':'H','1B':'G','1D':'F','1E':'C','1G':'A','1I':'D','1K':'I','1L':'K'},
              ('A','C','D','F','G','H','I','J'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'F','1K':'D','1L':'I'},
              ('A','C','D','E','I','J','K','L'): {'1A':'E','1B':'J','1D':'I','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','C','D','E','H','J','K','L'): {'1A':'H','1B':'J','1D':'E','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','C','D','E','H','I','K','L'): {'1A':'H','1B':'E','1D':'I','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','C','D','E','H','I','J','L'): {'1A':'H','1B':'J','1D':'E','1E':'C','1G':'A','1I':'D','1K':'L','1L':'I'},
              ('A','C','D','E','H','I','J','K'): {'1A':'H','1B':'J','1D':'E','1E':'C','1G':'A','1I':'D','1K':'I','1L':'K'},
              ('A','C','D','E','G','J','K','L'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','C','D','E','G','I','K','L'): {'1A':'E','1B':'G','1D':'I','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','C','D','E','G','I','J','L'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'A','1I':'D','1K':'L','1L':'I'},
              ('A','C','D','E','G','I','J','K'): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'A','1I':'D','1K':'I','1L':'K'},
              ('A','C','D','E','G','H','K','L'): {'1A':'H','1B':'G','1D':'E','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','C','D','E','G','H','J','L'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'D','1K':'L','1L':'E'},
              ('A','C','D','E','G','H','J','K'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'D','1K':'E','1L':'K'},
              ('A','C','D','E','G','H','I','L'): {'1A':'H','1B':'G','1D':'E','1E':'C','1G':'A','1I':'D','1K':'L','1L':'I'},
              ('A','C','D','E','G','H','I','K'): {'1A':'H','1B':'G','1D':'E','1E':'C','1G':'A','1I':'D','1K':'I','1L':'K'},
              ('A','C','D','E','G','H','I','J'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'D','1K':'E','1L':'I'},
              ('A','C','D','E','F','J','K','L'): {'1A':'C','1B':'J','1D':'E','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','D','E','F','I','K','L'): {'1A':'C','1B':'E','1D':'I','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','D','E','F','I','J','L'): {'1A':'C','1B':'J','1D':'E','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','C','D','E','F','I','J','K'): {'1A':'C','1B':'J','1D':'E','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','C','D','E','F','H','K','L'): {'1A':'H','1B':'E','1D':'F','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','C','D','E','F','H','J','L'): {'1A':'H','1B':'J','1D':'F','1E':'C','1G':'A','1I':'D','1K':'L','1L':'E'},
              ('A','C','D','E','F','H','J','K'): {'1A':'H','1B':'J','1D':'E','1E':'C','1G':'A','1I':'F','1K':'D','1L':'K'},
              ('A','C','D','E','F','H','I','L'): {'1A':'H','1B':'E','1D':'F','1E':'C','1G':'A','1I':'D','1K':'L','1L':'I'},
              ('A','C','D','E','F','H','I','K'): {'1A':'H','1B':'E','1D':'F','1E':'C','1G':'A','1I':'D','1K':'I','1L':'K'},
              ('A','C','D','E','F','H','I','J'): {'1A':'H','1B':'J','1D':'E','1E':'C','1G':'A','1I':'F','1K':'D','1L':'I'},
              ('A','C','D','E','F','G','K','L'): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','C','D','E','F','G','J','L'): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'L','1L':'E'},
              ('A','C','D','E','F','G','J','K'): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'E','1L':'K'},
              ('A','C','D','E','F','G','I','L'): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','C','D','E','F','G','I','K'): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','C','D','E','F','G','I','J'): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'A','1I':'F','1K':'E','1L':'I'},
              ('A','C','D','E','F','G','H','L'): {'1A':'H','1B':'G','1D':'F','1E':'C','1G':'A','1I':'D','1K':'L','1L':'E'},
              ('A','C','D','E','F','G','H','K'): {'1A':'H','1B':'G','1D':'E','1E':'C','1G':'A','1I':'F','1K':'D','1L':'K'},
              ('A','C','D','E','F','G','H','J'): {'1A':'H','1B':'G','1D':'J','1E':'C','1G':'A','1I':'F','1K':'D','1L':'E'},
              ('A','C','D','E','F','G','H','I'): {'1A':'H','1B':'G','1D':'E','1E':'C','1G':'A','1I':'F','1K':'D','1L':'I'},
              ('A','B','G','H','I','J','K','L'): {'1A':'H','1B':'J','1D':'B','1E':'A','1G':'I','1I':'G','1K':'L','1L':'K'},
              ('A','B','F','H','I','J','K','L'): {'1A':'H','1B':'J','1D':'B','1E':'A','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('A','B','F','G','I','J','K','L'): {'1A':'I','1B':'J','1D':'B','1E':'F','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','B','F','G','H','J','K','L'): {'1A':'H','1B':'J','1D':'B','1E':'F','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','B','F','G','H','I','K','L'): {'1A':'H','1B':'G','1D':'B','1E':'A','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('A','B','F','G','H','I','J','L'): {'1A':'H','1B':'J','1D':'B','1E':'F','1G':'A','1I':'G','1K':'L','1L':'I'},
              ('A','B','F','G','H','I','J','K'): {'1A':'H','1B':'J','1D':'B','1E':'F','1G':'A','1I':'G','1K':'I','1L':'K'},
              ('A','B','E','H','I','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'A','1G':'I','1I':'H','1K':'L','1L':'K'},
              ('A','B','E','G','I','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'A','1G':'I','1I':'G','1K':'L','1L':'K'},
              ('A','B','E','G','H','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'A','1G':'H','1I':'G','1K':'L','1L':'K'},
              ('A','B','E','G','H','I','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'A','1G':'I','1I':'H','1K':'L','1L':'K'},
              ('A','B','E','G','H','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'A','1G':'H','1I':'G','1K':'L','1L':'I'},
              ('A','B','E','G','H','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'A','1G':'H','1I':'G','1K':'I','1L':'K'},
              ('A','B','E','F','I','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'A','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('A','B','E','F','H','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'F','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','B','E','F','H','I','K','L'): {'1A':'E','1B':'I','1D':'B','1E':'F','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','B','E','F','H','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'F','1G':'A','1I':'H','1K':'L','1L':'I'},
              ('A','B','E','F','H','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'F','1G':'A','1I':'H','1K':'I','1L':'K'},
              ('A','B','E','F','G','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'F','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','B','E','F','G','I','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'A','1G':'I','1I':'F','1K':'L','1L':'K'},
              ('A','B','E','F','G','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'F','1G':'A','1I':'G','1K':'L','1L':'I'},
              ('A','B','E','F','G','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'F','1G':'A','1I':'G','1K':'I','1L':'K'},
              ('A','B','E','F','G','H','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'F','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','B','E','F','G','H','J','L'): {'1A':'H','1B':'J','1D':'B','1E':'F','1G':'A','1I':'G','1K':'L','1L':'E'},
              ('A','B','E','F','G','H','J','K'): {'1A':'H','1B':'J','1D':'B','1E':'F','1G':'A','1I':'G','1K':'E','1L':'K'},
              ('A','B','E','F','G','H','I','L'): {'1A':'E','1B':'G','1D':'B','1E':'F','1G':'A','1I':'H','1K':'L','1L':'I'},
              ('A','B','E','F','G','H','I','K'): {'1A':'E','1B':'G','1D':'B','1E':'F','1G':'A','1I':'H','1K':'I','1L':'K'},
              ('A','B','E','F','G','H','I','J'): {'1A':'H','1B':'J','1D':'B','1E':'F','1G':'A','1I':'G','1K':'E','1L':'I'},
              ('A','B','D','H','I','J','K','L'): {'1A':'I','1B':'J','1D':'B','1E':'D','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','B','D','G','I','J','K','L'): {'1A':'I','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','B','D','G','H','J','K','L'): {'1A':'H','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','B','D','G','H','I','K','L'): {'1A':'I','1B':'G','1D':'B','1E':'D','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','B','D','G','H','I','J','L'): {'1A':'H','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'L','1L':'I'},
              ('A','B','D','G','H','I','J','K'): {'1A':'H','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'I','1L':'K'},
              ('A','B','D','F','I','J','K','L'): {'1A':'I','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','D','F','H','J','K','L'): {'1A':'H','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','D','F','H','I','K','L'): {'1A':'H','1B':'I','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','D','F','H','I','J','L'): {'1A':'H','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','B','D','F','H','I','J','K'): {'1A':'H','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','B','D','F','G','J','K','L'): {'1A':'F','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','B','D','F','G','I','K','L'): {'1A':'I','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','D','F','G','I','J','L'): {'1A':'F','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'L','1L':'I'},
              ('A','B','D','F','G','I','J','K'): {'1A':'F','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'I','1L':'K'},
              ('A','B','D','F','G','H','K','L'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','D','F','G','H','J','L'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'J'},
              ('A','B','D','F','G','H','J','K'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'J','1L':'K'},
              ('A','B','D','F','G','H','I','L'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','B','D','F','G','H','I','K'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','B','D','F','G','H','I','J'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'I','1L':'J'},
              ('A','B','D','E','I','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'A','1G':'I','1I':'D','1K':'L','1L':'K'},
              ('A','B','D','E','H','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','B','D','E','H','I','K','L'): {'1A':'E','1B':'I','1D':'B','1E':'D','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','B','D','E','H','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'A','1I':'H','1K':'L','1L':'I'},
              ('A','B','D','E','H','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'A','1I':'H','1K':'I','1L':'K'},
              ('A','B','D','E','G','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','B','D','E','G','I','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'A','1G':'I','1I':'D','1K':'L','1L':'K'},
              ('A','B','D','E','G','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'L','1L':'I'},
              ('A','B','D','E','G','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'I','1L':'K'},
              ('A','B','D','E','G','H','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','B','D','E','G','H','J','L'): {'1A':'H','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'L','1L':'E'},
              ('A','B','D','E','G','H','J','K'): {'1A':'H','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'E','1L':'K'},
              ('A','B','D','E','G','H','I','L'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'A','1I':'H','1K':'L','1L':'I'},
              ('A','B','D','E','G','H','I','K'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'A','1I':'H','1K':'I','1L':'K'},
              ('A','B','D','E','G','H','I','J'): {'1A':'H','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'E','1L':'I'},
              ('A','B','D','E','F','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','D','E','F','I','K','L'): {'1A':'E','1B':'I','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','D','E','F','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','B','D','E','F','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','B','D','E','F','H','K','L'): {'1A':'H','1B':'E','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','D','E','F','H','J','L'): {'1A':'H','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'E'},
              ('A','B','D','E','F','H','J','K'): {'1A':'H','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'E','1L':'K'},
              ('A','B','D','E','F','H','I','L'): {'1A':'H','1B':'E','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','B','D','E','F','H','I','K'): {'1A':'H','1B':'E','1D':'B','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','B','D','E','F','H','I','J'): {'1A':'H','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'E','1L':'I'},
              ('A','B','D','E','F','G','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','D','E','F','G','J','L'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'J'},
              ('A','B','D','E','F','G','J','K'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'J','1L':'K'},
              ('A','B','D','E','F','G','I','L'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','B','D','E','F','G','I','K'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','B','D','E','F','G','I','J'): {'1A':'E','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'I','1L':'J'},
              ('A','B','D','E','F','G','H','L'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'E'},
              ('A','B','D','E','F','G','H','K'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'E','1L':'K'},
              ('A','B','D','E','F','G','H','J'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'E','1L':'J'},
              ('A','B','D','E','F','G','H','I'): {'1A':'H','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'E','1L':'I'},
              ('A','B','C','H','I','J','K','L'): {'1A':'I','1B':'J','1D':'B','1E':'C','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','B','C','G','I','J','K','L'): {'1A':'I','1B':'J','1D':'B','1E':'C','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','B','C','G','H','J','K','L'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','B','C','G','H','I','K','L'): {'1A':'I','1B':'G','1D':'B','1E':'C','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','B','C','G','H','I','J','L'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'G','1K':'L','1L':'I'},
              ('A','B','C','G','H','I','J','K'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'G','1K':'I','1L':'K'},
              ('A','B','C','F','I','J','K','L'): {'1A':'I','1B':'J','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','C','F','H','J','K','L'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','C','F','H','I','K','L'): {'1A':'H','1B':'I','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','C','F','H','I','J','L'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','B','C','F','H','I','J','K'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','B','C','F','G','J','K','L'): {'1A':'C','1B':'J','1D':'B','1E':'F','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','B','C','F','G','I','K','L'): {'1A':'I','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','C','F','G','I','J','L'): {'1A':'C','1B':'J','1D':'B','1E':'F','1G':'A','1I':'G','1K':'L','1L':'I'},
              ('A','B','C','F','G','I','J','K'): {'1A':'C','1B':'J','1D':'B','1E':'F','1G':'A','1I':'G','1K':'I','1L':'K'},
              ('A','B','C','F','G','H','K','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','C','F','G','H','J','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'J'},
              ('A','B','C','F','G','H','J','K'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'J','1L':'K'},
              ('A','B','C','F','G','H','I','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','B','C','F','G','H','I','K'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','B','C','F','G','H','I','J'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'I','1L':'J'},
              ('A','B','C','E','I','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'A','1G':'I','1I':'C','1K':'L','1L':'K'},
              ('A','B','C','E','H','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','B','C','E','H','I','K','L'): {'1A':'E','1B':'I','1D':'B','1E':'C','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','B','C','E','H','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'A','1I':'H','1K':'L','1L':'I'},
              ('A','B','C','E','H','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'A','1I':'H','1K':'I','1L':'K'},
              ('A','B','C','E','G','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','B','C','E','G','I','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'A','1G':'I','1I':'C','1K':'L','1L':'K'},
              ('A','B','C','E','G','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'A','1I':'G','1K':'L','1L':'I'},
              ('A','B','C','E','G','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'A','1I':'G','1K':'I','1L':'K'},
              ('A','B','C','E','G','H','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'H','1K':'L','1L':'K'},
              ('A','B','C','E','G','H','J','L'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'G','1K':'L','1L':'E'},
              ('A','B','C','E','G','H','J','K'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'G','1K':'E','1L':'K'},
              ('A','B','C','E','G','H','I','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'H','1K':'L','1L':'I'},
              ('A','B','C','E','G','H','I','K'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'H','1K':'I','1L':'K'},
              ('A','B','C','E','G','H','I','J'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'G','1K':'E','1L':'I'},
              ('A','B','C','E','F','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','C','E','F','I','K','L'): {'1A':'E','1B':'I','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','C','E','F','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','B','C','E','F','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','B','C','E','F','H','K','L'): {'1A':'H','1B':'E','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','C','E','F','H','J','L'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'E'},
              ('A','B','C','E','F','H','J','K'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'F','1K':'E','1L':'K'},
              ('A','B','C','E','F','H','I','L'): {'1A':'H','1B':'E','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','B','C','E','F','H','I','K'): {'1A':'H','1B':'E','1D':'B','1E':'C','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','B','C','E','F','H','I','J'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'F','1K':'E','1L':'I'},
              ('A','B','C','E','F','G','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','C','E','F','G','J','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'J'},
              ('A','B','C','E','F','G','J','K'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'J','1L':'K'},
              ('A','B','C','E','F','G','I','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','B','C','E','F','G','I','K'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','B','C','E','F','G','I','J'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'I','1L':'J'},
              ('A','B','C','E','F','G','H','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'L','1L':'E'},
              ('A','B','C','E','F','G','H','K'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'E','1L':'K'},
              ('A','B','C','E','F','G','H','J'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'E','1L':'J'},
              ('A','B','C','E','F','G','H','I'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'E','1L':'I'},
              ('A','B','C','D','I','J','K','L'): {'1A':'I','1B':'J','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','B','C','D','H','J','K','L'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','B','C','D','H','I','K','L'): {'1A':'H','1B':'I','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','B','C','D','H','I','J','L'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'I'},
              ('A','B','C','D','H','I','J','K'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'D','1K':'I','1L':'K'},
              ('A','B','C','D','G','J','K','L'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'L','1L':'K'},
              ('A','B','C','D','G','I','K','L'): {'1A':'I','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','B','C','D','G','I','J','L'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'L','1L':'I'},
              ('A','B','C','D','G','I','J','K'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'A','1I':'G','1K':'I','1L':'K'},
              ('A','B','C','D','G','H','K','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','B','C','D','G','H','J','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'J'},
              ('A','B','C','D','G','H','J','K'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'J','1L':'K'},
              ('A','B','C','D','G','H','I','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'I'},
              ('A','B','C','D','G','H','I','K'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'I','1L':'K'},
              ('A','B','C','D','G','H','I','J'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'I','1L':'J'},
              ('A','B','C','D','F','J','K','L'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','C','D','F','I','K','L'): {'1A':'C','1B':'I','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','C','D','F','I','J','L'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','B','C','D','F','I','J','K'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','B','C','D','F','H','K','L'): {'1A':'H','1B':'F','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','B','C','D','F','H','J','L'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'H'},
              ('A','B','C','D','F','H','J','K'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'F','1K':'D','1L':'K'},
              ('A','B','C','D','F','H','I','L'): {'1A':'H','1B':'F','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'I'},
              ('A','B','C','D','F','H','I','K'): {'1A':'H','1B':'F','1D':'B','1E':'C','1G':'A','1I':'D','1K':'I','1L':'K'},
              ('A','B','C','D','F','H','I','J'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'F','1K':'D','1L':'I'},
              ('A','B','C','D','F','G','K','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','C','D','F','G','J','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'J'},
              ('A','B','C','D','F','G','J','K'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'J','1L':'K'},
              ('A','B','C','D','F','G','I','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','B','C','D','F','G','I','K'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','B','C','D','F','G','I','J'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'I','1L':'J'},
              ('A','B','C','D','F','G','H','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'H'},
              ('A','B','C','D','F','G','H','K'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'D','1L':'K'},
              ('A','B','C','D','F','G','H','J'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'D','1L':'J'},
              ('A','B','C','D','F','G','H','I'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'D','1L':'I'},
              ('A','B','C','D','E','J','K','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','B','C','D','E','I','K','L'): {'1A':'E','1B':'I','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','B','C','D','E','I','J','L'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'I'},
              ('A','B','C','D','E','I','J','K'): {'1A':'E','1B':'J','1D':'B','1E':'C','1G':'A','1I':'D','1K':'I','1L':'K'},
              ('A','B','C','D','E','H','K','L'): {'1A':'H','1B':'E','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','B','C','D','E','H','J','L'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'E'},
              ('A','B','C','D','E','H','J','K'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'D','1K':'E','1L':'K'},
              ('A','B','C','D','E','H','I','L'): {'1A':'H','1B':'E','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'I'},
              ('A','B','C','D','E','H','I','K'): {'1A':'H','1B':'E','1D':'B','1E':'C','1G':'A','1I':'D','1K':'I','1L':'K'},
              ('A','B','C','D','E','H','I','J'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'D','1K':'E','1L':'I'},
              ('A','B','C','D','E','G','K','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'K'},
              ('A','B','C','D','E','G','J','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'J'},
              ('A','B','C','D','E','G','J','K'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'J','1L':'K'},
              ('A','B','C','D','E','G','I','L'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'I'},
              ('A','B','C','D','E','G','I','K'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'I','1L':'K'},
              ('A','B','C','D','E','G','I','J'): {'1A':'E','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'I','1L':'J'},
              ('A','B','C','D','E','G','H','L'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'E'},
              ('A','B','C','D','E','G','H','K'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'E','1L':'K'},
              ('A','B','C','D','E','G','H','J'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'E','1L':'J'},
              ('A','B','C','D','E','G','H','I'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'D','1K':'E','1L':'I'},
              ('A','B','C','D','E','F','K','L'): {'1A':'C','1B':'E','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'K'},
              ('A','B','C','D','E','F','J','L'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'E'},
              ('A','B','C','D','E','F','J','K'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'E','1L':'K'},
              ('A','B','C','D','E','F','I','L'): {'1A':'C','1B':'E','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'I'},
              ('A','B','C','D','E','F','I','K'): {'1A':'C','1B':'E','1D':'B','1E':'D','1G':'A','1I':'F','1K':'I','1L':'K'},
              ('A','B','C','D','E','F','I','J'): {'1A':'C','1B':'J','1D':'B','1E':'D','1G':'A','1I':'F','1K':'E','1L':'I'},
              ('A','B','C','D','E','F','H','L'): {'1A':'H','1B':'F','1D':'B','1E':'C','1G':'A','1I':'D','1K':'L','1L':'E'},
              ('A','B','C','D','E','F','H','K'): {'1A':'H','1B':'E','1D':'B','1E':'C','1G':'A','1I':'F','1K':'D','1L':'K'},
              ('A','B','C','D','E','F','H','J'): {'1A':'H','1B':'J','1D':'B','1E':'C','1G':'A','1I':'F','1K':'D','1L':'E'},
              ('A','B','C','D','E','F','H','I'): {'1A':'H','1B':'E','1D':'B','1E':'C','1G':'A','1I':'F','1K':'D','1L':'I'},
              ('A','B','C','D','E','F','G','L'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'L','1L':'E'},
              ('A','B','C','D','E','F','G','K'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'E','1L':'K'},
              ('A','B','C','D','E','F','G','J'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'E','1L':'J'},
              ('A','B','C','D','E','F','G','I'): {'1A':'C','1B':'G','1D':'B','1E':'D','1G':'A','1I':'F','1K':'E','1L':'I'},
              ('A','B','C','D','E','F','G','H'): {'1A':'H','1B':'G','1D':'B','1E':'C','1G':'A','1I':'F','1K':'D','1L':'E'}
            }
            if key in matrix:
                slots = matrix[key]
                # Map the slot names directly to the team name string using get_3rd
                return {slot_name: adv_thirds_by_group[slots[slot_name]] for slot_name in slot_names}
            else:
                # Fallback: assign advancing 3rd-place teams sequentially to slots
                sorted_adv = sorted(list(adv_groups))
                return {slot_names[i]: adv_thirds_by_group[sorted_adv[i]]
                        for i in range(len(slot_names))}

        # Execute the resolution
        thirds = resolve_3rd_slots(best_third_groups)

        def t3(slot: str) -> str:
            """Get third-place team for a given slot, with graceful fallback."""
            if slot in thirds:
                return thirds[slot]
            # fallback: any best third not yet used
            used = set(thirds.values())
            for t in best_thirds:
                if t not in used:
                    return t
            return best_thirds[0]

        # Build the 16 R32 matchups per FIFA schedule
        s = group_standings
        r32_pairs = [
            # --- TOP HALF OF THE BRACKET ---
            # Branch 1 (Feeds into QF Match 97)
            (s['E'][0], t3('1E')),   # M74: 1E vs best-3rd
            (s['I'][0], t3('1I')),   # M77: 1I vs best-3rd -> Winner plays M74 in M89
            
            (s['A'][1], s['B'][1]),  # M73: 2A vs 2B
            (s['F'][0], s['C'][1]),  # M75: 1F vs 2C      -> Winner plays M73 in M90
            
            # Branch 2 (Feeds into QF Match 99)
            (s['C'][0], s['F'][1]),  # M76: 1C vs 2F
            (s['E'][1], s['I'][1]),  # M78: 2E vs 2I      -> Winner plays M76 in M91
            
            (s['A'][0], t3('1A')),   # M79: 1A vs best-3rd
            (s['L'][0], t3('1L')),   # M80: 1L vs best-3rd -> Winner plays M79 in M92
            
            # --- BOTTOM HALF OF THE BRACKET ---
            # Branch 3 (Feeds into QF Match 98)
            (s['K'][1], s['L'][1]),  # M83: 2K vs 2L
            (s['H'][0], s['J'][1]),  # M84: 1H vs 2J      -> Winner plays M83 in M93

            (s['D'][0], t3('1D')),   # M81: 1D vs best-3rd
            (s['G'][0], t3('1G')),   # M82: 1G vs best-3rd -> Winner plays M81 in M94
            
            # Branch 4 (Feeds into QF Match 100)
            (s['J'][0], s['H'][1]),  # M86: 1J vs 2H
            (s['D'][1], s['G'][1]),  # M88: 2D vs 2G      -> Winner plays M86 in M95
          
            (s['B'][0], t3('1B')),   # M85: 1B vs best-3rd
            (s['K'][0], t3('1K')),   # M87: 1K vs best-3rd -> Winner plays M85 in M96
            

        ]

        # Simulate R32 → R16 → QF → SF → Final
        # r32_pairs is a list of 16 (team1, team2) tuples
        # Match IDs in bracket order (not chronological)
        r32_match_ids = ["M74","M77","M73","M75","M83","M84","M81","M82",
                         "M76","M78","M79","M80","M86","M88","M85","M87"]
        r16_match_ids = ["M89","M90","M93","M94","M91","M92","M95","M96"]
        qf_match_ids  = ["M97","M98","M99","M100"]
        sf_match_ids  = ["M101","M102"]
        all_stage_ids = [r32_match_ids, r16_match_ids, qf_match_ids, sf_match_ids]

        current_pairs = r32_pairs
        stage_labels = ["r16", "qf", "sf", "final", "champion"]
        stage_idx = 0

        while current_pairs:
            winners = []
            current_ids = all_stage_ids[stage_idx] if stage_idx < len(all_stage_ids) else []
            for pair_idx, (t1_name, t2_name) in enumerate(current_pairs):
                mid = current_ids[pair_idx] if pair_idx < len(current_ids) else None
                if mid and mid in locked_ko_results:
                    winner = locked_ko_results[mid]
                else:
                    t1 = TEAM_MAP[t1_name]
                    t2 = TEAM_MAP[t2_name]
                    gh, ga = simulate_match(t1, t2, rng, knockout=True)
                    winner = t1_name if gh > ga else t2_name
                winners.append(winner)

            label = stage_labels[stage_idx] if stage_idx < len(stage_labels) else "deep"
            for t in winners:
                ko_stage_reach[t][label] += 1
            stage_idx += 1

            # Pair up winners for next round
            if len(winners) > 1:
                current_pairs = [(winners[i], winners[i+1])
                                 for i in range(0, len(winners), 2)]
            else:
                current_pairs = []  # champion decided

    # ---------------------------------------------------------------------------
    # 7.  AGGREGATE RESULTS
    # ---------------------------------------------------------------------------

    # -- Group results table --
    group_rows = []
    for t in TEAMS:
        avg_pts = np.mean(group_pts_acc[t.name])
        avg_gf  = np.mean(group_gf_acc[t.name])
        avg_ga  = np.mean(group_ga_acc[t.name])
        avg_gd  = avg_gf - avg_ga
        avg_rank = np.mean(group_rank_acc[t.name])
        adv_pct  = group_adv_count[t.name] / n_sims * 100

        group_rows.append({
            "Team":         t.name,
            "Group":        t.group,
            "FIFA_Rank":    t.fifa_rank,
            "Avg_Pts":      round(avg_pts, 2),
            "Avg_GF":       round(avg_gf, 2),
            "Avg_GA":       round(avg_ga, 2),
            "Avg_GD":       round(avg_gd, 2),
            "Avg_GroupRank": round(avg_rank, 2),
            "Advance_Pct":  round(adv_pct, 1),
        })

    group_df = pd.DataFrame(group_rows).sort_values(
        ["Group", "Avg_GroupRank"]
    ).reset_index(drop=True)

    # -- Match results table --
    match_rows = []
    for key in match_count:
        h, a = key
        cnt = match_count[key]
        xgh, xga = expected_goals(TEAM_MAP[h], TEAM_MAP[a])
        match_rows.append({
            "Home":      h,
            "Away":      a,
            "Group":     TEAM_MAP[h].group,
            "Home_Win%": round(match_home_wins[key] / cnt * 100, 1),
            "Draw%":     round(match_draws[key]     / cnt * 100, 1),
            "Away_Win%": round(match_away_wins[key] / cnt * 100, 1),
            "Sim_xGF":   round(match_total_gf[key] / cnt, 2),
            "Sim_xGA":   round(match_total_ga[key] / cnt, 2),
            "Model_xGF": round(xgh, 2),
            "Model_xGA": round(xga, 2),
        })

    match_df = pd.DataFrame(match_rows).sort_values(["Group", "Home"]).reset_index(drop=True)

    # -- Knockout results table --
    ko_stages = ["r32", "r16", "qf", "sf", "final", "champion"]
    ko_rows = []
    for t in TEAMS:
        row = {"Team": t.name, "Group": t.group, "FIFA_Rank": t.fifa_rank}
        for s in ko_stages:
            row[f"{s}_pct"] = round(ko_stage_reach[t.name].get(s, 0) / n_sims * 100, 2)
        ko_rows.append(row)

    ko_df = pd.DataFrame(ko_rows).sort_values("champion_pct", ascending=False).reset_index(drop=True)

    # -- Survivor picks (EV-optimised, game-theory aware) --
    survivor_df, pick_pct_df = compute_survivor_picks(group_df, ko_df)

    return {
        "group_df":    group_df,
        "match_df":    match_df,
        "ko_df":       ko_df,
        "survivor_df": survivor_df,
        "pick_pct_df": pick_pct_df,
        "n_sims":      n_sims,
    }

# ---------------------------------------------------------------------------
# 8.  PUBLIC PICK PERCENTAGE MODEL
#
#  Rationale
#  ---------
#  With 20,000+ entrants the field doesn't pick optimally — they follow a
#  mix of survival probability, brand recognition, and narrative bias.
#  We model pick% as a softmax over a weighted combination of:
#    α × survival_prob      — pure math signal
#    β × fifa_rank_score    — brand / name recognition (chalk following)
#    γ × media_salience     — recency / narrative bias (prior WC winners,
#                             "story" teams, host-nation favourites)
#
#  Calibration reference
#  ---------------------
#  NFL survivor pool research (Brill 2020; Massey-Thaler) found:
#    • Most-picked team in any week captures 25–45% of entries
#    • Correlation between implied win prob and pick% ≈ r=0.70
#    • Brand teams (Cowboys, Patriots) are over-picked by 8–15 pp
#  We use analogous β/γ values calibrated to WC fan poll data.
#
#  The pick% model is intentionally *not* recursive (we don't model
#  other entrants updating their strategy based on yours).  At 20,000
#  entrants your single entry has negligible price impact.
# ---------------------------------------------------------------------------

# Media salience priors — manually tuned, update as tournament approaches.
# 1.0 = maximum narrative weight (defending champion, host nation, iconic brand)
# 0.0 = no extra media pull beyond pure survival odds
MEDIA_SALIENCE: dict[str, float] = {
    "Argentina":   1.00,   # defending WC champion
    "Brazil":      1.00,   # perennial favourite, global brand
    "France":      0.90,   # 2018 champion, stacked squad
    "Germany":     0.85,   # storied brand, casual fans default here
    "England":     0.80,   # massive global fanbase, "it's coming home" narrative
    "Spain":       0.70,
    "Portugal":    0.65,   # Ronaldo halo effect
    "Netherlands": 0.55,
    "Italy":       0.55,
    "USA":         0.60,   # co-host, large domestic pool of pickers
    "Mexico":      0.55,   # co-host
    "Canada":      0.40,   # co-host but lower recognition
    "Uruguay":     0.40,
    "Colombia":    0.35,
    "Japan":       0.35,
    "Senegal":     0.30,
    "Morocco":     0.45,   # 2022 semi-finalist, strong recent narrative
    "Croatia":     0.40,   # 2022 finalist
}

# Softmax temperature: lower = field concentrates more on top teams
# 1.0 is standard; 0.7 means field is MORE chalk-heavy (reasonable for soccer)
PICK_TEMPERATURE = 0.75

def compute_public_pick_pcts(
    teams_list: list[Team],
    survival_probs: dict[str, float],   # team → probability of surviving this round
    stage: str,
    alpha: float = 2.5,    # weight on survival probability
    beta:  float = 0.80,   # weight on FIFA rank score
    gamma: float = 0.55,   # weight on media salience
) -> dict[str, float]:
    """
    Model the public pick% distribution for a given stage.

    Returns dict: team_name → estimated fraction of entries picking that team.
    Picks sum to 1.0 across all eligible teams.
    """
    # FIFA rank score: normalised so rank=1 → 1.0, rank=200 → 0.0
    max_rank = max(t.fifa_rank for t in teams_list)
    rank_scores = {t.name: (max_rank - t.fifa_rank) / max_rank for t in teams_list}

    logits: dict[str, float] = {}
    for t in teams_list:
        surv  = survival_probs.get(t.name, 0.0)
        if surv <= 0:
            logits[t.name] = -999.0   # effectively zero pick%
            continue
        rank_s  = rank_scores[t.name]
        media_s = MEDIA_SALIENCE.get(t.name, 0.25)
        logits[t.name] = (
            alpha * surv
            + beta  * rank_s
            + gamma * media_s
        ) / PICK_TEMPERATURE

    # Softmax
    names  = list(logits.keys())
    vals   = np.array([logits[n] for n in names])
    vals   -= vals.max()   # numerical stability
    exps   = np.exp(vals)
    total  = exps.sum()
    return {n: float(exps[i] / total) for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# 9.  SURVIVOR PICK OPTIMIZER  (EV-maximising, no-repeat, game-theory aware)
#
#  Objective
#  ---------
#  Maximise EXPECTED FIELD ADVANTAGE across ALL picks simultaneously,
#  subject to the "each team used at most once" constraint.
#
#  EV framework
#  ------------
#  In a large survivor pool with N entrants, your expected prize share from
#  a set of picks S is proportional to:
#
#      EV(S) ∝  Π_{t ∈ S}  P(t survives) / P(field picks t AND t survives)
#
#  Simplifying (field pick% ≈ independent of team's actual survival that week):
#
#      EV(S) ∝  Π_{t ∈ S}  P(t survives) / pick_pct(t)
#
#  Taking logs makes this additive (log-EV), which is easier to optimise:
#
#      log_EV(S) = Σ_{t ∈ S}  log(surv_prob(t)) − log(pick_pct(t))
#
#  We maximise log_EV greedily with the no-repeat constraint.
#
#  Three pick strategies are computed and compared:
#    CHALK    — pure survival probability (ignore pick%)
#    EV_OPT   — log-EV maximised (recommended)
#    CONTRARIAN — maximise pick% discount, subject to survival > threshold
# ---------------------------------------------------------------------------

# Minimum survival probability — set per-stage as a fraction of the top team's probability
# This ensures every stage always produces picks even when absolute probabilities are low
MIN_SURVIVAL_FRACTION = 0.20   # must have at least 20% of top team's survival prob

def _log_ev(surv: float, pick_pct: float) -> float:
    """Per-team log-EV contribution. Higher = better value pick."""
    if surv <= 0 or pick_pct <= 0:
        return -999.0
    return math.log(surv) - math.log(pick_pct)

def _greedy_picks(
    candidates: list[str],
    n_picks: int,
    used_teams: set[str],
    score_fn,           # team_name → float (higher = prefer)
    survival_probs: dict[str, float],
    min_surv: float = 0.05,
) -> list[str]:
    """
    Greedy pick selector: choose n_picks teams maximising score_fn,
    subject to: not in used_teams, survival >= min_surv.
    Also diversifies across groups where possible (reduces correlated failure).
    """
    eligible = [
        t for t in candidates
        if t not in used_teams
        and survival_probs.get(t, 0) >= min_surv
    ]
    scored = sorted(eligible, key=score_fn, reverse=True)

    picks: list[str] = []
    used_groups: set[str] = set()
    team_group = {t.name: t.group for t in TEAMS}

    # First pass: group-diverse picks
    for t in scored:
        grp = team_group.get(t, "?")
        if grp not in used_groups:
            picks.append(t)
            used_groups.add(grp)
        if len(picks) == n_picks:
            return picks

    # Second pass: fill remaining slots ignoring group diversity
    for t in scored:
        if t not in picks:
            picks.append(t)
        if len(picks) == n_picks:
            return picks

    return picks


def compute_survivor_picks(
    group_df: pd.DataFrame,
    ko_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute three pick strategies for each survivor stage:
      CHALK    — highest raw survival probability (ignores game theory)
      EV_OPT   — maximises log(surv/pick%) — RECOMMENDED
      CONTRARIAN — heavily down-weights popular teams, accepts lower surv floor

    Returns
    -------
    picks_df : one row per (stage, pick_number, strategy)
    pct_df   : full pick% and EV table for every team at every stage
    """
    all_teams = TEAMS

    # Build survival probability lookup per stage
    # group stage: Advance_Pct
    # knockout stages: r32_pct, r16_pct, qf_pct, sf_pct, final_pct, champion_pct
    surv_by_stage: dict[str, dict[str, float]] = {
        "Group Stage":  dict(zip(group_df["Team"], group_df["Advance_Pct"] / 100)),
        "Round of 32":  dict(zip(ko_df["Team"], ko_df["r16_pct"]      / 100)),
        "Round of 16":  dict(zip(ko_df["Team"], ko_df["qf_pct"]       / 100)),
        "Quarterfinals":dict(zip(ko_df["Team"], ko_df["sf_pct"]       / 100)),
        "Semifinal":    dict(zip(ko_df["Team"], ko_df["final_pct"]    / 100)),
        "Final":        dict(zip(ko_df["Team"], ko_df["champion_pct"] / 100)),
    }

    # Pick counts per stage
    picks_per_stage = {
        "Group Stage":   4,
        "Round of 32":   2,
        "Round of 16":   2,
        "Quarterfinals": 2,
        "Semifinal":     1,
        "Final":         1,
    }

    all_pick_rows: list[dict] = []
    pct_rows: list[dict] = []
    used_chalk:       set[str] = set()
    used_ev:          set[str] = set()
    used_contrarian:  set[str] = set()

    for stage, n_picks in picks_per_stage.items():
        surv = surv_by_stage[stage]
        eligible_teams = [t for t in all_teams if surv.get(t.name, 0) > 0]

        # Compute public pick percentages for this stage
        pick_pcts = compute_public_pick_pcts(eligible_teams, surv, stage)

        # ── Record full per-team pick% table ──
        for t in eligible_teams:
            sp   = surv.get(t.name, 0)
            pp   = pick_pcts.get(t.name, 1e-6)
            lev  = _log_ev(sp, pp)
            pct_rows.append({
                "Stage":         stage,
                "Team":          t.name,
                "Group":         t.group,
                "Survival_Pct":  round(sp * 100, 2),
                "Est_Pick_Pct":  round(pp * 100, 2),
                "Log_EV":        round(lev, 3),
                "EV_Ratio":      round(sp / pp if pp > 0 else 0, 2),
                "Value_Label":   (
                    "★ GREAT VALUE"  if lev > np.percentile([_log_ev(surv.get(x.name,0), pick_pcts.get(x.name,1e-6)) for x in eligible_teams], 75)
                    else "↓ AVOID"   if lev < np.percentile([_log_ev(surv.get(x.name,0), pick_pcts.get(x.name,1e-6)) for x in eligible_teams], 25)
                    else "  OK"
                ),
            })

        candidates = [t.name for t in eligible_teams]

        # Stage-relative survival floor: 20% of the top team's prob for this stage
        top_surv = max(surv.values()) if surv else 1.0
        stage_floor = top_surv * MIN_SURVIVAL_FRACTION

        # ── CHALK: pure survival prob ──
        chalk_picks = _greedy_picks(
            candidates, n_picks, used_chalk,
            score_fn=lambda t: surv.get(t, 0),
            survival_probs=surv,
            min_surv=stage_floor,
        )

        # ── EV_OPT: maximise log-EV ──
        ev_picks = _greedy_picks(
            candidates, n_picks, used_ev,
            score_fn=lambda t: _log_ev(surv.get(t, 0), pick_pcts.get(t, 1e-6)),
            survival_probs=surv,
            min_surv=stage_floor,
        )

        # ── CONTRARIAN: heavily discount popular teams ──
        contrarian_floor = max(top_surv * 0.40, stage_floor)
        contrarian_picks = _greedy_picks(
            candidates, n_picks, used_contrarian,
            score_fn=lambda t: _log_ev(surv.get(t, 0), pick_pcts.get(t, 1e-6)) * 1.5
                               - 0.5 * math.log(max(pick_pcts.get(t, 1e-6), 1e-6)),
            survival_probs=surv,
            min_surv=contrarian_floor,
        )

        # Update used sets
        used_chalk.update(chalk_picks)
        used_ev.update(ev_picks)
        used_contrarian.update(contrarian_picks)

        # ── Store pick rows for all three strategies ──
        for strategy, picks in [
            ("CHALK",       chalk_picks),
            ("EV_OPT",      ev_picks),
            ("CONTRARIAN",  contrarian_picks),
        ]:
            for i, team_name in enumerate(picks):
                sp  = surv.get(team_name, 0)
                pp  = pick_pcts.get(team_name, 1e-6)
                grp = next((t.group for t in TEAMS if t.name == team_name), "?")
                all_pick_rows.append({
                    "Stage":        stage,
                    "Pick_Number":  i + 1,
                    "Strategy":     strategy,
                    "Team":         team_name,
                    "Group":        grp,
                    "Survival_Pct": round(sp * 100, 2),
                    "Est_Pick_Pct": round(pp * 100, 2),
                    "Log_EV":       round(_log_ev(sp, pp), 3),
                    "EV_Ratio":     round(sp / pp if pp > 0 else 0, 2),
                    "Note":         (
                        "Highest survival prob"         if strategy == "CHALK"
                        else "Best log(surv/pick%)"     if strategy == "EV_OPT"
                        else "Contrarian value pick"
                    ),
                })

    picks_df = pd.DataFrame(all_pick_rows)
    pct_df   = pd.DataFrame(pct_rows).sort_values(
        ["Stage", "Log_EV"], ascending=[True, False]
    ).reset_index(drop=True)

    return picks_df, pct_df

# ---------------------------------------------------------------------------
# 10.  OUTPUT & DISPLAY
# ---------------------------------------------------------------------------

def print_summary(results: dict):
    group_df    = results["group_df"]
    match_df    = results["match_df"]
    ko_df       = results["ko_df"]
    survivor_df = results["survivor_df"]
    pick_pct_df = results["pick_pct_df"]
    W = 74

    print(f"\n{'='*W}")
    print("  GROUP STAGE — Top Teams by Advancement Probability")
    print(f"{'='*W}")
    top_adv = group_df.sort_values("Advance_Pct", ascending=False).head(20)
    print(top_adv[["Team","Group","Avg_Pts","Avg_GF","Avg_GA",
                   "Avg_GD","Avg_GroupRank","Advance_Pct"]].to_string(index=False))

    print(f"\n{'='*W}")
    print("  KNOCKOUT — Championship Probabilities (Top 16)")
    print(f"{'='*W}")
    ko_cols = ["Team","Group","r32_pct","r16_pct","qf_pct","sf_pct","final_pct","champion_pct"]
    print(ko_df[ko_cols].head(16).to_string(index=False))

    for strategy in ["EV_OPT", "CHALK", "CONTRARIAN"]:
        label = {
            "EV_OPT":     "★ EV-OPTIMAL  (recommended — beats the field)",
            "CHALK":      "  CHALK       (highest survival, ignores game theory)",
            "CONTRARIAN": "  CONTRARIAN  (discounts popular teams most)",
        }[strategy]
        print(f"\n{'='*W}")
        print(f"  SURVIVOR PICKS — {label}")
        print(f"{'='*W}")
        sub = survivor_df[survivor_df["Strategy"] == strategy]
        print(sub[["Stage","Pick_Number","Team","Survival_Pct",
                   "Est_Pick_Pct","EV_Ratio"]].to_string(index=False))

    print(f"\n{'='*W}")
    print("  PUBLIC PICK % — Group Stage  (sorted by EV — best value at top)")
    print(f"{'='*W}")
    gs = pick_pct_df[pick_pct_df["Stage"] == "Group Stage"].head(20)
    print(gs[["Team","Group","Survival_Pct","Est_Pick_Pct",
              "EV_Ratio","Value_Label"]].to_string(index=False))

    print(f"\n{'='*W}")
    print("  GROUP MATCH PREDICTIONS — Selected High-Profile Fixtures")
    print(f"{'='*W}")
    interesting = match_df[match_df["Home_Win%"] > 40].sort_values(
        "Home_Win%", ascending=False).head(15)
    print(interesting[["Home","Away","Home_Win%","Draw%","Away_Win%",
                       "Sim_xGF","Sim_xGA"]].to_string(index=False))


def save_outputs(results: dict, prefix: str = "wc2026"):
    results["group_df"].to_csv(   f"{prefix}_group_results.csv",    index=False)
    results["match_df"].to_csv(   f"{prefix}_match_results.csv",    index=False)
    results["ko_df"].to_csv(      f"{prefix}_knockout_results.csv", index=False)
    results["survivor_df"].to_csv(f"{prefix}_survivor_picks.csv",   index=False)
    results["pick_pct_df"].to_csv(f"{prefix}_pick_pcts.csv",        index=False)

    with open(f"{prefix}_simulation_summary.txt", "w") as f:
        f.write(f"FIFA World Cup 2026 Monte Carlo Simulation\n")
        f.write(f"Simulations: {results['n_sims']:,}\n\n")
        f.write("=== TOURNAMENT FAVOURITES ===\n")
        for _, row in results["ko_df"].head(8).iterrows():
            f.write(f"  {row['Team']:15s}  champion: {row['champion_pct']:5.1f}%"
                    f"  final: {row['final_pct']:5.1f}%  sf: {row['sf_pct']:5.1f}%\n")
        for strategy in ["EV_OPT", "CHALK", "CONTRARIAN"]:
            f.write(f"\n=== SURVIVOR PICKS — {strategy} ===\n")
            sub = results["survivor_df"][results["survivor_df"]["Strategy"] == strategy]
            for _, row in sub.iterrows():
                f.write(f"  [{row['Stage']}] Pick {int(row['Pick_Number'])}: "
                        f"{row['Team']:15s}  surv={row['Survival_Pct']:.1f}%  "
                        f"pick%={row['Est_Pick_Pct']:.1f}%  EV={row['EV_Ratio']:.2f}x\n")
        f.write("\n=== PICK % INTELLIGENCE (Group Stage) ===\n")
        gs = results["pick_pct_df"][results["pick_pct_df"]["Stage"] == "Group Stage"]
        for _, row in gs.head(20).iterrows():
            f.write(f"  {row['Team']:15s}  surv={row['Survival_Pct']:5.1f}%  "
                    f"pick%={row['Est_Pick_Pct']:5.1f}%  EV={row['EV_Ratio']:5.2f}  {row['Value_Label']}\n")

    for fn in [f"{prefix}_group_results.csv", f"{prefix}_match_results.csv",
               f"{prefix}_knockout_results.csv", f"{prefix}_survivor_picks.csv",
               f"{prefix}_pick_pcts.csv", f"{prefix}_simulation_summary.txt"]:
        print(f"✓ Saved: {fn}")


# ---------------------------------------------------------------------------
# 11.  UTILITY: SINGLE MATCH PROBABILITY TABLE
# ---------------------------------------------------------------------------

def print_match_odds(home_name: str, away_name: str, n: int = 100_000):
    """Print a detailed probability breakdown for any specific match."""
    if home_name not in TEAM_MAP or away_name not in TEAM_MAP:
        print(f"Unknown team(s). Available: {', '.join(TEAM_MAP)}")
        return

    home = TEAM_MAP[home_name]
    away = TEAM_MAP[away_name]
    rng  = np.random.default_rng(0)

    wins = draws = losses = 0
    total_gh = total_ga = 0
    scorelines: dict[tuple, int] = defaultdict(int)

    for _ in range(n):
        gh, ga = simulate_match(home, away, rng)
        scorelines[(gh, ga)] += 1
        total_gh += gh
        total_ga += ga
        if gh > ga:
            wins += 1
        elif gh == ga:
            draws += 1
        else:
            losses += 1

    print(f"\n{'─'*50}")
    print(f"  {home_name} vs {away_name}  ({n:,} sims)")
    print(f"{'─'*50}")
    print(f"  Home win : {wins/n*100:5.1f}%")
    print(f"  Draw     : {draws/n*100:5.1f}%")
    print(f"  Away win : {losses/n*100:5.1f}%")
    print(f"  xG  {home_name}: {total_gh/n:.2f}")
    print(f"  xG  {away_name}: {total_ga/n:.2f}")
    print(f"\n  Top scorelines:")
    for sc, cnt in sorted(scorelines.items(), key=lambda x: -x[1])[:8]:
        print(f"    {sc[0]}-{sc[1]} : {cnt/n*100:4.1f}%")


# ---------------------------------------------------------------------------
# 11.  ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Cup 2026 Monte Carlo Simulator")
    parser.add_argument("--sims",    type=int, default=50000, help="Number of tournament simulations")
    parser.add_argument("--seed",    type=int, default=None,  help="Random seed for reproducibility")
    parser.add_argument("--results", type=Path, default=None, help="Path to results.json with locked match results")
    parser.add_argument("--prefix",  type=str, default="wc2026", help="Output file prefix")
    parser.add_argument("--match",   type=str, nargs=2, metavar=("HOME", "AWAY"),
                        help="Print detailed odds for a specific match, e.g. --match Argentina France")
    args = parser.parse_args()

    if args.match:
        print_match_odds(args.match[0], args.match[1])
    else:
        results = run_simulation(n_sims=args.sims, seed=args.seed,
                             results_path=args.results)
        print_summary(results)
        save_outputs(results, prefix=args.prefix)
        print("\nDone. Good luck in your survivor league! ⚽\n")
      
