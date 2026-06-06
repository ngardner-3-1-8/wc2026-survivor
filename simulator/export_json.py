"""
export_json.py
==============
Runs the WC2026 simulator and writes web-ready JSON to public/data/.
Called by GitHub Actions after each simulation run.

Usage:
    python export_json.py --sims 50000 --out-dir ../public/data
"""

import argparse
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Allow importing simulator from same directory
sys.path.insert(0, str(Path(__file__).parent))
from wc2026_simulator import run_simulation, GROUPS, TEAMS

STAGE_ORDER = [
    "Group Stage", "Round of 32", "Round of 16",
    "Quarterfinals", "Semifinal", "Final"
]

def df_to_records(df):
    """Convert DataFrame to list of dicts with native Python types."""
    return json.loads(df.to_json(orient="records"))


def build_groups_json(group_df, match_df):
    """
    {
      "A": {
        "teams": [ { name, fifa_rank, avg_pts, avg_gf, avg_ga, advance_pct }, ... ],
        "matches": [ { home, away, home_win_pct, draw_pct, away_win_pct, xgf, xga }, ... ]
      }, ...
    }
    """
    groups = {}
    for grp_name, grp_teams in sorted(GROUPS.items()):
        team_rows = group_df[group_df["Group"] == grp_name].sort_values(
            "Avg_GroupRank"
        )
        teams_out = []
        for _, r in team_rows.iterrows():
            teams_out.append({
                "name":        r["Team"],
                "fifa_rank":   int(r["FIFA_Rank"]),
                "avg_pts":     round(float(r["Avg_Pts"]), 2),
                "avg_gf":      round(float(r["Avg_GF"]), 2),
                "avg_ga":      round(float(r["Avg_GA"]), 2),
                "avg_gd":      round(float(r["Avg_GD"]), 2),
                "advance_pct": round(float(r["Advance_Pct"]), 1),
            })

        grp_matches = match_df[match_df["Group"] == grp_name]
        matches_out = []
        for _, r in grp_matches.iterrows():
            matches_out.append({
                "home":          r["Home"],
                "away":          r["Away"],
                "home_win_pct":  round(float(r["Home_Win%"]), 1),
                "draw_pct":      round(float(r["Draw%"]), 1),
                "away_win_pct":  round(float(r["Away_Win%"]), 1),
                "xgf":           round(float(r["Sim_xGF"]), 2),
                "xga":           round(float(r["Sim_xGA"]), 2),
            })

        groups[grp_name] = {"teams": teams_out, "matches": matches_out}
    return groups


def build_knockout_json(ko_df):
    """
    List of teams with per-stage probabilities, sorted by champion_pct desc.
    [ { name, group, r32_pct, r16_pct, qf_pct, sf_pct, final_pct, champion_pct }, ... ]
    """
    out = []
    for _, r in ko_df.iterrows():
        out.append({
            "name":         r["Team"],
            "group":        r["Group"],
            "fifa_rank":    int(r["FIFA_Rank"]),
            "r32_pct":      round(float(r["r32_pct"]), 2),
            "r16_pct":      round(float(r["r16_pct"]), 2),
            "qf_pct":       round(float(r["qf_pct"]), 2),
            "sf_pct":       round(float(r["sf_pct"]), 2),
            "final_pct":    round(float(r["final_pct"]), 2),
            "champion_pct": round(float(r["champion_pct"]), 2),
        })
    return out


def build_survivor_json(survivor_df, pick_pct_df):
    """
    {
      "strategies": {
        "EV_OPT": {
          "Group Stage": [ { pick_number, team, group, survival_pct, pick_pct, ev_ratio }, ... ],
          "Round of 32": [ ... ],
          ...
        },
        "CHALK": { ... },
        "CONTRARIAN": { ... }
      },
      "pick_intelligence": {
        "Group Stage": [ { team, group, survival_pct, pick_pct, ev_ratio, value_label }, ... ],
        ...
      }
    }
    """
    strategies = {}
    for strat in ["EV_OPT", "CHALK", "CONTRARIAN"]:
        strat_picks = {}
        sub = survivor_df[survivor_df["Strategy"] == strat]
        for stage in STAGE_ORDER:
            stage_rows = sub[sub["Stage"] == stage].sort_values("Pick_Number")
            if stage_rows.empty:
                continue
            strat_picks[stage] = [
                {
                    "pick_number":   int(r["Pick_Number"]),
                    "team":          r["Team"],
                    "group":         r["Group"],
                    "survival_pct":  round(float(r["Survival_Pct"]), 2),
                    "pick_pct":      round(float(r["Est_Pick_Pct"]), 2),
                    "ev_ratio":      round(float(r["EV_Ratio"]), 2),
                    "log_ev":        round(float(r["Log_EV"]), 3),
                    "note":          r["Note"],
                }
                for _, r in stage_rows.iterrows()
            ]
        strategies[strat] = strat_picks

    # Pick intelligence per stage
    pick_intel = {}
    for stage in STAGE_ORDER:
        stage_rows = pick_pct_df[pick_pct_df["Stage"] == stage].sort_values(
            "Log_EV", ascending=False
        )
        if stage_rows.empty:
            continue
        pick_intel[stage] = [
            {
                "team":         r["Team"],
                "group":        r["Group"],
                "survival_pct": round(float(r["Survival_Pct"]), 2),
                "pick_pct":     round(float(r["Est_Pick_Pct"]), 2),
                "ev_ratio":     round(float(r["EV_Ratio"]), 2),
                "log_ev":       round(float(r["Log_EV"]), 3),
                "value_label":  r["Value_Label"],
            }
            for _, r in stage_rows.iterrows()
        ]
    return {"strategies": strategies, "pick_intelligence": pick_intel}


def write_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ {path}  ({path.stat().st_size // 1024}KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims",    type=int,  default=50000)
    parser.add_argument("--seed",    type=int,  default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("../public/data"))
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Running {args.sims:,} simulations …")
    print(f"{'='*55}\n")

    results = run_simulation(n_sims=args.sims, seed=args.seed)

    print("\nExporting JSON …")

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_sims":        args.sims,
        "model":         "Dixon-Coles Poisson + MLE ratings",
    }

    write_json(meta,
        args.out_dir / "meta.json")
    write_json(build_groups_json(results["group_df"], results["match_df"]),
        args.out_dir / "groups.json")
    write_json(build_knockout_json(results["ko_df"]),
        args.out_dir / "knockout.json")
    write_json(build_survivor_json(results["survivor_df"], results["pick_pct_df"]),
        args.out_dir / "survivor.json")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
