"""Build canonical NBA player identity references."""

from pathlib import Path

import pandas as pd

from src.entity_resolution.normalization import (
    normalize_player_name,
    normalize_matching_name,
    normalize_team_name,
)

from src.validation.historical_audit import (
    resolve_dataset_path,
)


def build_player_reference(
    player_seasons: pd.DataFrame,
) -> pd.DataFrame:
    """Build the authoritative statistical player reference."""

    reference = (
        player_seasons[["player_id", "player_name"]].dropna().drop_duplicates().copy()
    )

    reference["player_id"] = reference["player_id"].astype("Int64")

    reference["normalized_player_name"] = reference["player_name"].map(
        normalize_player_name
    )

    reference["matching_name"] = reference["player_name"].map(normalize_matching_name)

    return reference.sort_values(["player_id", "player_name"]).reset_index(drop=True)


def build_player_team_context(
    start_season: int = 2016,
    end_season: int = 2027,
    raw_dir: str | Path = "data/raw",
    snapshot_season: int = 2027,
) -> pd.DataFrame:
    """Build historical player-team-season identity context.

    Team associations are used for identity resolution,
    not to establish current contractual obligations.
    """

    frames = []

    for season in range(start_season, end_season + 1):

        path = resolve_dataset_path(
            dataset="player_boxscores",
            season=season,
            raw_dir=raw_dir,
            snapshot_season=snapshot_season,
        )

        context = pd.read_parquet(
            path,
            columns=[
                "person_id",
                "first_name",
                "family_name",
                "team_id",
                "team_city",
                "team_name",
            ],
        )

        context = context.rename(columns={"person_id": "player_id"})

        context["season"] = season

        context["player_name"] = (
            context["first_name"].fillna("") + " " + context["family_name"].fillna("")
        ).str.strip()

        context["team"] = (
            context["team_city"].fillna("") + " " + context["team_name"].fillna("")
        ).str.strip()

        context = context[
            [
                "player_id",
                "player_name",
                "team_id",
                "team",
                "season",
            ]
        ].drop_duplicates()

        frames.append(context)

    result = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .dropna(subset=["player_id", "team_id"])
        .copy()
    )

    result["player_id"] = result["player_id"].astype("Int64")

    result["team_id"] = result["team_id"].astype("Int64")

    result = result.loc[result["player_id"].gt(0) & result["team_id"].gt(0)].copy()

    result["normalized_player_name"] = result["player_name"].map(normalize_player_name)

    result["matching_name"] = result["player_name"].map(normalize_matching_name)

    result["normalized_team"] = result["team"].map(normalize_team_name)

    return result.sort_values(
        ["season", "player_id", "team_id", "player_name"]
    ).reset_index(drop=True)


def save_player_references(
    player_reference: pd.DataFrame,
    player_team_context: pd.DataFrame,
    output_dir: str | Path = "data/processed/identities",
) -> dict[str, Path]:
    """Persist reusable player identity reference datasets."""

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "player_reference": (output_dir / "nba_player_reference.parquet"),
        "player_team_context": (output_dir / "nba_player_team_context.parquet"),
    }

    player_reference.to_parquet(
        paths["player_reference"],
        index=False,
    )

    player_team_context.to_parquet(
        paths["player_team_context"],
        index=False,
    )

    return paths
