"""
Streamlit UI for Football Model v1 — now supports:
    • History fixtures with explanations
    • Upcoming fixtures without explanations
    • Single fixture view
    • Slate (date range) view
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
from src.model import team_names

# ----------------------------------------------------------------------
# Paths and loaders
# ----------------------------------------------------------------------


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def get_history_dir() -> Path:
    root = get_project_root()
    hist = root / "data" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    return hist


@st.cache_data
def get_team_name_map() -> dict[int, str]:
    """Return mapping {team_id -> team_name} for EPL 2025–26."""
    teams_df = team_names.load_epl_teams()
    return teams_df.set_index("team_id")["team_name"].to_dict()


@st.cache_data
@st.cache_data
def load_history_data():
    """Load fixtures + projections_with_explanations for HISTORY mode."""
    history_dir = get_history_dir()

    # ✅ Use full history, not the tiny sample
    fixtures_path = history_dir / "fixtures_epl_2025_26_history.csv"
    projections_path = (
        history_dir / "match_projections_epl_2025_26_history_with_explanations.csv"
    )

    try:
        fixtures_df = pd.read_csv(fixtures_path)
    except FileNotFoundError:
        st.error(f"Fixtures file not found: {fixtures_path}")
        return None, None, None

    try:
        proj_df = pd.read_csv(projections_path)
    except FileNotFoundError:
        st.error(f"Projection file not found: {projections_path}")
        return fixtures_df, None, None

    mode = "history"
    return fixtures_df, proj_df, mode


@st.cache_data
def load_upcoming_data():
    """Load fixtures + projections for UPCOMING mode."""
    history_dir = get_history_dir()

    fixtures_path = history_dir / "fixtures_epl_2025_26_upcoming.csv"
    projections_path = history_dir / "match_projections_epl_2025_26_upcoming.csv"

    try:
        fixtures_df = pd.read_csv(fixtures_path)
    except FileNotFoundError:
        st.error(f"Upcoming fixtures file not found: {fixtures_path}")
        return None, None, None

    try:
        proj_df = pd.read_csv(projections_path)
    except FileNotFoundError:
        st.error(f"Upcoming projections file not found: {projections_path}")
        return fixtures_df, None, None

    mode = "upcoming"
    return fixtures_df, proj_df, mode


def build_fixture_labels(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Build labels like: '2025-08-15 19:00 — Arsenal @ Chelsea (fixture 1378969)'."""
    df = fixtures_df.copy()

    # Format date nicely
    df["date_str"] = df["date"].astype(str).str.replace("T", " ").str.slice(0, 16)

    # Map team IDs to names
    id_to_name = get_team_name_map()
    df["home_name"] = df["home_team_id"].map(id_to_name).fillna(
        df["home_team_id"].astype(str)
    )
    df["away_name"] = df["away_team_id"].map(id_to_name).fillna(
        df["away_team_id"].astype(str)
    )

    df["label"] = (
        df["date_str"]
        + " — "
        + df["away_name"]
        + " @ "
        + df["home_name"]
        + "  (fixture "
        + df["fixture_id"].astype(str)
        + ")"
    )

    df = df.sort_values("date_str")
    return df[["fixture_id", "label"]]


# ----------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Football Model v1 – EPL 2025–26",
        layout="wide",
    )

    st.title("⚽ Football Model v1 – EPL 2025–26")
    st.caption("Player projections for history + upcoming fixtures.")

    # Sidebar: data source
    st.sidebar.header("Data source")
    mode_choice = st.sidebar.radio(
        "Choose data source:", ["History", "Upcoming"], index=0
    )

    if mode_choice == "History":
        fixtures_df, proj_df, mode = load_history_data()
    else:
        fixtures_df, proj_df, mode = load_upcoming_data()

    if fixtures_df is None or proj_df is None:
        st.stop()

    # Sidebar: view type
    st.sidebar.header("View")
    view_choice = st.sidebar.radio(
        "What do you want to see?",
        ["Single fixture", "Slate (date range)", "Role diagnostics"],
        index=0,
    )

    id_to_name = get_team_name_map()

    # ------------------------------------------------------------------
    # VIEW 1: Single fixture
    # ------------------------------------------------------------------
    if view_choice == "Single fixture":
        # Select fixture
        st.sidebar.header("Fixture selection")
        fixture_labels = build_fixture_labels(fixtures_df)
        if fixture_labels.empty:
            st.warning("No fixtures available.")
            st.stop()

        label_to_fixture = dict(
            zip(fixture_labels["label"], fixture_labels["fixture_id"])
        )
        fixture_list = list(label_to_fixture.keys())

        selected_label = st.sidebar.selectbox("Choose a fixture:", fixture_list)
        selected_fixture_id = label_to_fixture[selected_label]

        match_proj = proj_df[proj_df["fixture_id"] == selected_fixture_id].copy()

        if match_proj.empty:
            st.warning("No projections for this fixture.")
            st.stop()

        # Sidebar filters
        st.sidebar.header("Filters")

        # Home/away
        side_filters = ["Both teams", "Home only", "Away only"]
        which_side = st.sidebar.radio("Team side", side_filters, index=0)

        if which_side == "Home only":
            match_proj = match_proj[match_proj["is_home"] == True]
        elif which_side == "Away only":
            match_proj = match_proj[match_proj["is_home"] == False]

        # Minutes filter
        min_minutes = st.sidebar.slider(
            "Minimum expected minutes", 0, 90, 30, step=5
        )
        match_proj = match_proj[match_proj["expected_minutes"] >= min_minutes]

        # Role filter
        if "role" in match_proj.columns:
            roles_available = sorted(match_proj["role"].dropna().unique().tolist())
            selected_roles = st.sidebar.multiselect(
                "Filter by role", roles_available, default=roles_available
            )
            if selected_roles:
                match_proj = match_proj[match_proj["role"].isin(selected_roles)]

        # Table
        st.subheader(
            f"{'History' if mode=='history' else 'Upcoming'} projections – fixture_id {selected_fixture_id}"
        )

        base_cols = [
            "team_id",
            "opponent_team_id",
            "player_name",
            "role",
            "expected_minutes",
            "proj_shots_total",
            "proj_shots_on",
            "proj_fouls_committed",
            "proj_fouls_drawn",
            "proj_yellow_cards",
        ]
        base_cols = [c for c in base_cols if c in match_proj.columns]

        table_df = match_proj[base_cols].copy()

        # Map team_id -> team name for display
        if "team_id" in table_df.columns:
            table_df["team"] = table_df["team_id"].map(id_to_name).fillna(
                table_df["team_id"].astype(str)
            )

        # Map opponent_team_id -> opponent team name for display
        if "opponent_team_id" in table_df.columns:
            table_df["opponent"] = table_df["opponent_team_id"].map(
                id_to_name
            ).fillna(table_df["opponent_team_id"].astype(str))

        # Round numeric projections
        for col in [
            "expected_minutes",
            "proj_shots_total",
            "proj_shots_on",
            "proj_fouls_committed",
            "proj_fouls_drawn",
            "proj_yellow_cards",
        ]:
            if col in table_df.columns:
                table_df[col] = table_df[col].astype(float).round(2)

        # Build sort order, preferring team name over ID
        sort_cols = []
        if "team" in table_df.columns:
            sort_cols.append("team")
        elif "team_id" in table_df.columns:
            sort_cols.append("team_id")
        if "expected_minutes" in table_df.columns:
            sort_cols.append("expected_minutes")
        if "player_name" in table_df.columns:
            sort_cols.append("player_name")

        display_df = table_df

        # Hide raw IDs in the UI
        for col_to_drop in ["team_id", "opponent_team_id"]:
            if col_to_drop in display_df.columns:
                display_df = display_df.drop(columns=[col_to_drop])

        st.dataframe(
            display_df.sort_values(
                sort_cols,
                ascending=[True, False, True][: len(sort_cols)],
            ),
            width="stretch",
        )

# --- Download CSV for this fixture ---
        fixture_csv = display_df.sort_values(
            sort_cols,
            ascending=[True, False, True][: len(sort_cols)],
        ).to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download this fixture's projections (CSV)",
            data=fixture_csv,
            file_name=f"fixture_{selected_fixture_id}_projections.csv",
            mime="text/csv",
        )

        # Explanations — only in history mode
        if mode == "history":
            st.subheader("Player explanations")

            players = match_proj["player_name"].tolist()
            selected_player = st.selectbox(
                "Choose a player", options=players, index=0
            )
            row = match_proj[match_proj["player_name"] == selected_player].iloc[0]

            st.markdown(f"### {selected_player}")

            for i in range(1, 5):
                col = f"explanation_{i}"
                if col in row.index:
                    st.markdown(f"**Explanation {i}:**")
                    st.write(str(row[col]))
        else:
            st.info("Explanations are available only for history fixtures.")

    # ------------------------------------------------------------------
    # VIEW 2: Slate (date range)
    # ------------------------------------------------------------------
    elif view_choice == "Slate (date range)":
        st.subheader(
            f"{'History' if mode=='history' else 'Upcoming'} projections – slate (date range)"
        )

        # Sidebar selection for date range
        st.sidebar.header("Slate selection")

        fixtures_copy = fixtures_df.copy()
        fixtures_copy["date_only"] = pd.to_datetime(fixtures_copy["date"]).dt.date
        min_date = fixtures_copy["date_only"].min()
        max_date = fixtures_copy["date_only"].max()

        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        # date_range can be a single date or a tuple of two
        if isinstance(date_range, tuple) or isinstance(date_range, list):
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = end_date = date_range[0]
        else:
            start_date = end_date = date_range

        # Merge fixture date into projections, then filter by date
        proj_all = proj_df.merge(
            fixtures_copy[["fixture_id", "date"]],
            on="fixture_id",
            how="left",
        )

        proj_all["date_only"] = pd.to_datetime(proj_all["date"]).dt.date

        mask = (proj_all["date_only"] >= start_date) & (
            proj_all["date_only"] <= end_date
        )
        slate_proj = proj_all[mask].copy()

        if slate_proj.empty:
            st.warning("No projections found in this date range.")
            st.stop()

        # Sidebar filters (same style as single fixture)
        st.sidebar.header("Filters")

        side_filters = ["Both teams", "Home only", "Away only"]
        which_side = st.sidebar.radio("Team side", side_filters, index=0)

        if which_side == "Home only" and "is_home" in slate_proj.columns:
            slate_proj = slate_proj[slate_proj["is_home"] == True]
        elif which_side == "Away only" and "is_home" in slate_proj.columns:
            slate_proj = slate_proj[slate_proj["is_home"] == False]

        # Minutes filter
        min_minutes = st.sidebar.slider(
            "Minimum expected minutes", 0, 90, 30, step=5
        )
        slate_proj = slate_proj[slate_proj["expected_minutes"] >= min_minutes]

        # Role filter
        if "role" in slate_proj.columns:
            roles_available = sorted(
                slate_proj["role"].dropna().unique().tolist()
            )
            selected_roles = st.sidebar.multiselect(
                "Filter by role", roles_available, default=roles_available
            )
            if selected_roles:
                slate_proj = slate_proj[slate_proj["role"].isin(selected_roles)]

        # Table for the whole slate
        base_cols = [
            "date_only",
            "fixture_id",
            "team_id",
            "opponent_team_id",
            "player_name",
            "role",
            "expected_minutes",
            "proj_shots_total",
            "proj_shots_on",
            "proj_fouls_committed",
            "proj_fouls_drawn",
            "proj_yellow_cards",
        ]
        base_cols = [c for c in base_cols if c in slate_proj.columns]

        table_df = slate_proj[base_cols].copy()

        # Map team + opponent names
        if "team_id" in table_df.columns:
            table_df["team"] = table_df["team_id"].map(id_to_name).fillna(
                table_df["team_id"].astype(str)
            )
        if "opponent_team_id" in table_df.columns:
            table_df["opponent"] = table_df["opponent_team_id"].map(
                id_to_name
            ).fillna(table_df["opponent_team_id"].astype(str))

        # Round numeric projections
        for col in [
            "expected_minutes",
            "proj_shots_total",
            "proj_shots_on",
            "proj_fouls_committed",
            "proj_fouls_drawn",
            "proj_yellow_cards",
        ]:
            if col in table_df.columns:
                table_df[col] = table_df[col].astype(float).round(2)

        # Sort: date → team → minutes → player
        sort_cols = []
        if "date_only" in table_df.columns:
            sort_cols.append("date_only")
        if "team" in table_df.columns:
            sort_cols.append("team")
        elif "team_id" in table_df.columns:
            sort_cols.append("team_id")
        if "expected_minutes" in table_df.columns:
            sort_cols.append("expected_minutes")
        if "player_name" in table_df.columns:
            sort_cols.append("player_name")

        display_df = table_df

        # Hide raw IDs in the UI
        for col_to_drop in ["team_id", "opponent_team_id"]:
            if col_to_drop in display_df.columns:
                display_df = display_df.drop(columns=[col_to_drop])

        st.dataframe(
            display_df.sort_values(
                sort_cols,
                ascending=[True, True, False, True][: len(sort_cols)],
            ),
            width="stretch",
        )

# --- Download CSV for the whole slate ---
        slate_sorted = display_df.sort_values(
            sort_cols,
            ascending=[True, True, False, True][: len(sort_cols)],
        )
        slate_csv = slate_sorted.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download slate projections (CSV)",
            data=slate_csv,
            file_name="slate_projections.csv",
            mime="text/csv",
        )

        # --- Top players by metric within the slate ---
        st.subheader("Top players in slate by metric")

        metric_options = [
            col
            for col in [
                "proj_shots_total",
                "proj_shots_on",
                "proj_fouls_committed",
                "proj_fouls_drawn",
                "proj_yellow_cards",
            ]
            if col in slate_sorted.columns
        ]

        if metric_options:
            selected_metric = st.selectbox(
                "Choose metric", metric_options, index=0
            )
            top_n = st.slider("How many players to show", 5, 50, 20, step=5)

            top_df = slate_sorted.sort_values(
                selected_metric, ascending=False
            ).head(top_n)

            st.dataframe(
                top_df[
                    [
                        col
                        for col in [
                            "date_only",
                            "team",
                            "opponent",
                            "player_name",
                            "role",
                            "expected_minutes",
                            selected_metric,
                        ]
                        if col in top_df.columns
                    ]
                ],
                width='stretch',
            )
        else:
            st.info("No numeric projection columns found for top-player view.")

# ------------------------------------------------------------------
    # VIEW 3: Role diagnostics
    # ------------------------------------------------------------------
    else:
        st.subheader("Role diagnostics – per-role adjustment factors")

        from pathlib import Path

        # Load modifiers CSV
        try:
            hist_dir = get_history_dir()
            mods_path = hist_dir / "role_vs_role_modifiers_epl_2025_26.csv"
            mods_df = pd.read_csv(mods_path)
        except FileNotFoundError:
            st.error(
                "role_vs_role_modifiers_epl_2025_26.csv not found in data/history.\n"
                "Run the role_vs_role_adjustments script first."
            )
            return

        if mods_df.empty:
            st.info("Modifiers file is empty.")
            return

        # Focus on global per-role calibration: defending_role == 'ALL'
        global_df = mods_df[mods_df["defending_role"] == "ALL"].copy()
        if global_df.empty:
            st.info(
                "No rows with defending_role == 'ALL' found in modifiers file.\n"
                "Learning step may not have run yet."
            )
        else:
            # Cleaner column names for display
            display_cols = {
                "attacking_role": "Role",
                "shots_total_factor": "Shots total ×",
                "shots_on_factor": "Shots on target ×",
                "fouls_committed_factor": "Fouls committed ×",
                "fouls_drawn_factor": "Fouls drawn ×",
                "yellow_cards_factor": "Yellow cards ×",
            }

            for old, new in display_cols.items():
                if old in global_df.columns:
                    global_df.rename(columns={old: new}, inplace=True)

            keep_cols = [c for c in display_cols.values() if c in global_df.columns]

            st.markdown("### Global per-role factors (defending_role = ALL)")
            st.dataframe(
                global_df[keep_cols].sort_values("Role"),
                use_container_width=True,
            )

        st.markdown(
            """
            - Values **> 1.0** mean the model was under-projecting that role on that metric (boost applied).  
            - Values **< 1.0** mean the model was over-projecting (nerf applied).  
            These only apply where we had enough history samples for that role.
            """
        )

    st.caption(
        "Data pipelines: history loader → joiner → baselines → team style → role interactions → projections."
    )


if __name__ == "__main__":
    main()