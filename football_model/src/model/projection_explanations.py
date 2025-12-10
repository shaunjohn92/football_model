"""
projection_explanations.py – Football Model v1

Explanation layer for match projections.

Takes:
    - match_projections_epl_2025_26_sample.csv
    - player_baselines_epl_2025_26_overall.csv
    - team_baselines_epl_2025_26_overall.csv
    - team_style_epl_2025_26.csv
    - player_match_stats_epl_2025_26_sample_enriched.csv   (for role/side)
    - role_vs_role_epl_2025_26.csv                         (matchup stats)

Produces:
    - match_projections_epl_2025_26_sample_with_explanations.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


# ----------------------------------------------------------------------
# Path helpers and loaders
# ----------------------------------------------------------------------


def get_history_dir() -> Path:
    """
    Locate the data/history directory relative to this file.
    """
    project_root = Path(__file__).resolve().parents[2]
    history_dir = project_root / "data" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


def load_explanation_inputs() -> Dict[str, pd.DataFrame]:
    """
    Load projections + baselines + team style + role info from data/history.
    """
    history_dir = get_history_dir()

    proj_path = history_dir / "match_projections_epl_2025_26_sample.csv"
    player_base_path = history_dir / "player_baselines_epl_2025_26_overall.csv"
    team_base_path = history_dir / "team_baselines_epl_2025_26_overall.csv"
    team_style_path = history_dir / "team_style_epl_2025_26.csv"
    players_enriched_path = (
        history_dir / "player_match_stats_epl_2025_26_sample_enriched.csv"
    )
    role_vs_role_path = history_dir / "role_vs_role_epl_2025_26.csv"

    projections_df = pd.read_csv(proj_path)
    player_baselines_df = pd.read_csv(player_base_path)
    team_baselines_df = pd.read_csv(team_base_path)
    team_style_df = pd.read_csv(team_style_path)
    players_enriched_df = pd.read_csv(players_enriched_path)
    role_vs_role_df = pd.read_csv(role_vs_role_path)

    return {
        "projections": projections_df,
        "player_baselines": player_baselines_df,
        "team_baselines": team_baselines_df,
        "team_style": team_style_df,
        "players_enriched": players_enriched_df,
        "role_vs_role": role_vs_role_df,
    }


# ----------------------------------------------------------------------
# Role-vs-role helper
# ----------------------------------------------------------------------


def build_role_matchup_summary(role_vs_role: pd.DataFrame) -> Dict[str, dict]:
    """
    For each attacker_role, compute:

        - best_foul_matchup_def_role
        - best_foul_matchup_side
        - best_fouls_per_pair
        - avg_fouls_per_pair

    Returns a dict keyed by attacker_role.
    """
    df = role_vs_role.copy()

    # Avoid division by zero
    df["fouls_per_pair"] = df.apply(
        lambda row: (row["fouls_drawn"] / row["pair_appearances"])
        if row["pair_appearances"] > 0
        else 0.0,
        axis=1,
    )

    if df.empty:
        return {}

    # Average fouls per pair by attacker role
    avg = (
        df.groupby("attacker_role", dropna=False)["fouls_per_pair"]
        .mean()
        .reset_index()
        .rename(columns={"fouls_per_pair": "avg_fouls_per_pair"})
    )

    # Best (highest fouls_per_pair) matchup per attacker_role
    # Take first after sorting by fouls_per_pair desc
    df_sorted = df.sort_values(
        ["attacker_role", "fouls_per_pair"], ascending=[True, False]
    )
    best = (
        df_sorted.groupby("attacker_role", dropna=False)
        .first()
        .reset_index()[["attacker_role", "defender_role", "side_match", "fouls_per_pair"]]
        .rename(
            columns={
                "defender_role": "best_defender_role",
                "side_match": "best_side_match",
                "fouls_per_pair": "best_fouls_per_pair",
            }
        )
    )

    summary = best.merge(avg, on="attacker_role", how="left")

    lookup: Dict[str, dict] = {}
    for _, row in summary.iterrows():
        attacker_role = row["attacker_role"]
        lookup[attacker_role] = {
            "best_defender_role": row["best_defender_role"],
            "best_side_match": row["best_side_match"],
            "best_fouls_per_pair": float(row["best_fouls_per_pair"]),
            "avg_fouls_per_pair": float(row["avg_fouls_per_pair"]),
        }

    return lookup


def pretty_side_phrase(side_match: str) -> str:
    """
    Convert side_match codes to human-readable phrases.
    """
    if side_match == "same_side":
        return "on the same flank"
    if side_match == "opp_side":
        return "from the opposite flank"
    return "in more central or mixed areas"


# ----------------------------------------------------------------------
# Explanation builder
# ----------------------------------------------------------------------


def build_explanations(
    projections: pd.DataFrame,
    player_baselines: pd.DataFrame,
    team_baselines: pd.DataFrame,
    team_style: pd.DataFrame,
    players_enriched: pd.DataFrame,
    role_vs_role: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join projections with baselines and create explanation strings.
    """
    proj = projections.copy()

    # --- Merge in role + side from enriched players (by fixture/team/player) ---
    role_small = players_enriched[
        ["fixture_id", "team_id", "player_id", "role", "side"]
    ].copy()

    proj = proj.merge(
        role_small,
        on=["fixture_id", "team_id", "player_id"],
        how="left",
    )

    proj["role"] = proj["role"].fillna("UNK")
    proj["side"] = proj["side"].fillna("C")

    # --- Join player baselines ---
    player_cols = [
        "player_id",
        "team_id",
        "player_name",
        "minutes_total",
        "games",
        "avg_minutes",
        "shots_total_per90",
        "shots_on_per90",
        "fouls_committed_per90",
        "fouls_drawn_per90",
        "yellow_cards_per90",
    ]
    player_base_small = player_baselines[player_cols].copy()

    proj = proj.merge(
        player_base_small,
        on=["player_id", "team_id", "player_name"],
        how="left",
        suffixes=("", "_base"),
    )

    # --- Join team baselines for own team ---
    team_cols = [
        "team_id",
        "matches",
        "goals_for_per_match",
        "goals_against_per_match",
        "shots_total_per_match",
        "shots_on_per_match",
        "fouls_committed_per_match",
        "fouls_drawn_per_match",
        "yellow_cards_per_match",
        "red_cards_per_match",
    ]
    team_base_small = team_baselines[team_cols].copy()
    team_base_small = team_base_small.rename(
        columns=lambda c: f"team_{c}" if c != "team_id" else c
    )

    proj = proj.merge(
        team_base_small,
        on="team_id",
        how="left",
    )

    # --- Join team baselines for opponent team ---
    opp_base_small = team_baselines[team_cols].copy()
    opp_base_small = opp_base_small.rename(
        columns=lambda c: f"opp_{c}" if c != "team_id" else "opponent_team_id"
    )

    proj = proj.merge(
        opp_base_small,
        on="opponent_team_id",
        how="left",
    )

    # --- Join team style for own team ---
    style_cols = [
        "team_id",
        "foul_rate",
        "card_rate",
        "shots_allowed_rate",
        "press_intensity",
        "foul_z",
        "card_z",
        "shots_allowed_z",
        "press_z",
        "style_tag",
    ]
    style_small = team_style[style_cols].copy()
    style_small = style_small.rename(
        columns={
            "team_id": "team_id",
            "style_tag": "team_style_tag",
            "foul_rate": "team_foul_rate",
            "card_rate": "team_card_rate",
            "shots_allowed_rate": "team_shots_allowed_rate",
            "press_intensity": "team_press_intensity",
            "foul_z": "team_foul_z",
            "card_z": "team_card_z",
            "shots_allowed_z": "team_shots_allowed_z",
            "press_z": "team_press_z",
        }
    )

    proj = proj.merge(style_small, on="team_id", how="left")

    # --- Join team style for opponent team ---
    opp_style_small = team_style[style_cols].copy()
    opp_style_small = opp_style_small.rename(
        columns={
            "team_id": "opponent_team_id",
            "style_tag": "opp_style_tag",
            "foul_rate": "opp_foul_rate",
            "card_rate": "opp_card_rate",
            "shots_allowed_rate": "opp_shots_allowed_rate",
            "press_intensity": "opp_press_intensity",
            "foul_z": "opp_foul_z",
            "card_z": "opp_card_z",
            "shots_allowed_z": "opp_shots_allowed_z",
            "press_z": "opp_press_z",
        }
    )

    proj = proj.merge(opp_style_small, on="opponent_team_id", how="left")

    # Fill NaN with zeros where numeric
    numeric_cols = [
        "minutes_total",
        "games",
        "avg_minutes",
        "shots_total_per90",
        "shots_on_per90",
        "fouls_committed_per90",
        "fouls_drawn_per90",
        "yellow_cards_per90",
        "team_matches",
        "team_shots_total_per_match",
        "team_fouls_committed_per_match",
        "team_fouls_drawn_per_match",
        "team_yellow_cards_per_match",
        "opp_matches",
        "opp_goals_against_per_match",
        "opp_fouls_committed_per_match",
        "opp_fouls_drawn_per_match",
        "team_foul_rate",
        "team_card_rate",
        "team_shots_allowed_rate",
        "team_press_intensity",
        "team_foul_z",
        "team_card_z",
        "team_shots_allowed_z",
        "team_press_z",
        "opp_foul_rate",
        "opp_card_rate",
        "opp_shots_allowed_rate",
        "opp_press_intensity",
        "opp_foul_z",
        "opp_card_z",
        "opp_shots_allowed_z",
        "opp_press_z",
    ]
    for col in numeric_cols:
        if col in proj.columns:
            proj[col] = proj[col].fillna(0.0)

    proj["team_style_tag"] = proj.get("team_style_tag", "balanced").fillna("balanced")
    proj["opp_style_tag"] = proj.get("opp_style_tag", "balanced").fillna("balanced")

    # Precompute role matchup summary lookup
    role_matchup_lookup = build_role_matchup_summary(role_vs_role)

    # Now build explanation strings row by row
    expl1_list = []
    expl2_list = []
    expl3_list = []
    expl4_list = []

    for _, row in proj.iterrows():
        name = row.get("player_name", "The player")
        role = row.get("role", "UNK")

        games = row.get("games", 0)
        minutes_total = row.get("minutes_total", 0.0)
        avg_minutes = row.get("avg_minutes", 0.0)

        shots_p90 = row.get("shots_total_per90", 0.0)
        shots_on_p90 = row.get("shots_on_per90", 0.0)
        fouls_comm_p90 = row.get("fouls_committed_per90", 0.0)
        fouls_drawn_p90 = row.get("fouls_drawn_per90", 0.0)

        expected_minutes = row.get("expected_minutes", 0.0)
        proj_shots = row.get("proj_shots_total", 0.0)
        proj_shots_on = row.get("proj_shots_on", 0.0)
        proj_fouls_comm = row.get("proj_fouls_committed", 0.0)
        proj_fouls_drawn = row.get("proj_fouls_drawn", 0.0)
        proj_yellows = row.get("proj_yellow_cards", 0.0)

        team_shots_match = row.get("team_shots_total_per_match", 0.0)
        team_matches = row.get("team_matches", 0.0)

        opp_goals_against = row.get("opp_goals_against_per_match", 0.0)
        opp_fouls_comm_match = row.get("opp_fouls_committed_per_match", 0.0)
        opp_fouls_drawn_match = row.get("opp_fouls_drawn_per_match", 0.0)
        opp_matches = row.get("opp_matches", 0.0)

        team_style_tag = row.get("team_style_tag", "balanced")
        opp_style_tag = row.get("opp_style_tag", "balanced")

        # Avoid division by zero in share-of-team calculations
        if team_shots_match and team_shots_match > 0:
            share_shots = proj_shots / team_shots_match
        else:
            share_shots = 0.0

        # --- Explanation 1: baseline & minutes ---
        if games and games > 0:
            expl1 = (
                f"{name} has played {int(games)} match"
                f"{'' if games == 1 else 'es'} ({int(minutes_total)} total minutes), "
                f"averaging about {avg_minutes:.0f} minutes per appearance. "
                f"For this match we expect around {expected_minutes:.0f} minutes."
            )
        else:
            expl1 = (
                f"{name} has limited data so far, so we approximate his minutes "
                f"from this season's appearances and role, giving about {expected_minutes:.0f} minutes here."
            )

        # --- Explanation 2: per-90 shooting baseline ---
        expl2 = (
            f"Per 90 minutes he averages {shots_p90:.1f} shots and "
            f"{shots_on_p90:.1f} shots on target."
        )

        # --- Explanation 3: match projection vs team context ---
        if team_matches and team_matches > 0:
            expl3 = (
                f"His team takes around {team_shots_match:.1f} shots per match, and "
                f"this projection of {proj_shots:.1f} shots ({proj_shots_on:.1f} on target) "
                f"is about {share_shots:.0%} of the team total."
            )
        else:
            expl3 = (
                f"We project about {proj_shots:.1f} shots ({proj_shots_on:.1f} on target) "
                f"for {name} in this match."
            )

        # --- Explanation 4: fouls/cards, opponent context, style + role matchup ---
        base_foul_part = (
            f"He draws {fouls_drawn_p90:.1f} fouls and commits {fouls_comm_p90:.1f} per 90, "
            f"translating to about {proj_fouls_drawn:.1f} drawn and "
            f"{proj_fouls_comm:.1f} committed here. "
        )

        if opp_matches and opp_matches > 0:
            opp_context_part = (
                f"The opposition concedes roughly {opp_goals_against:.1f} goals per match "
                f"and is involved in about {opp_fouls_comm_match + opp_fouls_drawn_match:.1f} fouls per game. "
            )
        else:
            opp_context_part = ""

        style_part = (
            f"Stylistically, his team is classified as {team_style_tag}, "
            f"while the opposition are {opp_style_tag}. "
        )

        # Role-vs-role matchup snippet
        matchup_part = ""
        info = role_matchup_lookup.get(role)
        if info is not None and info["best_fouls_per_pair"] > 0:
            def_role = info["best_defender_role"]
            side_match = info["best_side_match"]
            best_fp = info["best_fouls_per_pair"]
            avg_fp = info["avg_fouls_per_pair"]
            side_phrase = pretty_side_phrase(side_match)

            if avg_fp > 0:
                if best_fp > avg_fp * 1.1:
                    matchup_part = (
                        f"As a {role}, he tends to draw the most fouls when facing {def_role}s "
                        f"{side_phrase}, averaging around {best_fp:.2f} fouls drawn per direct matchup, "
                        f"which is higher than the typical {avg_fp:.2f} for this role."
                    )
                else:
                    matchup_part = (
                        f"As a {role}, his foul-drawing profile against {def_role}s "
                        f"{side_phrase} is fairly typical for this role, at about {best_fp:.2f} "
                        f"fouls per matchup vs a role average of {avg_fp:.2f}."
                    )
            else:
                matchup_part = (
                    f"As a {role}, his foul-drawing matchups against {def_role}s {side_phrase} "
                    f"are in line with the limited data we have so far."
                )

        expl4 = base_foul_part + opp_context_part + style_part + matchup_part

        expl1_list.append(expl1)
        expl2_list.append(expl2)
        expl3_list.append(expl3)
        expl4_list.append(expl4)

    proj["explanation_1"] = expl1_list
    proj["explanation_2"] = expl2_list
    proj["explanation_3"] = expl3_list
    proj["explanation_4"] = expl4_list

    return proj


# ----------------------------------------------------------------------
# Script entrypoint
# ----------------------------------------------------------------------


if __name__ == "__main__":
    data = load_explanation_inputs()
    projections_df = data["projections"]
    player_baselines_df = data["player_baselines"]
    team_baselines_df = data["team_baselines"]
    team_style_df = data["team_style"]
    players_enriched_df = data["players_enriched"]
    role_vs_role_df = data["role_vs_role"]

    print("Loaded inputs for explanations:")
    print("  Projections:", projections_df.shape)
    print("  Player baselines:", player_baselines_df.shape)
    print("  Team baselines:", team_baselines_df.shape)
    print("  Team style:", team_style_df.shape)
    print("  Players enriched:", players_enriched_df.shape)
    print("  Role-vs-role:", role_vs_role_df.shape)

    explained_df = build_explanations(
        projections=projections_df,
        player_baselines=player_baselines_df,
        team_baselines=team_baselines_df,
        team_style=team_style_df,
        players_enriched=players_enriched_df,
        role_vs_role=role_vs_role_df,
    )

    print("\nProjections with explanations shape:", explained_df.shape)
    print("Sample rows:")
    print(
        explained_df[
            [
                "player_name",
                "role",
                "expected_minutes",
                "proj_shots_total",
                "proj_shots_on",
                "proj_fouls_committed",
                "proj_fouls_drawn",
                "proj_yellow_cards",
                "explanation_1",
                "explanation_2",
                "explanation_3",
                "explanation_4",
            ]
        ]
        .head(3)
        .to_string(index=False)
    )

    history_dir = get_history_dir()
    out_path = history_dir / "match_projections_epl_2025_26_sample_with_explanations.csv"
    explained_df.to_csv(out_path, index=False)

    print("\nSaved projections with explanations to:")
    print("  ", out_path)