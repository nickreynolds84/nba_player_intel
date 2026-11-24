from __future__ import annotations

import hashlib
import json
import logging

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

from src.ingestion.season_data import (
    get_release_assets,
    atomic_write,
    read_metadata,
    asset_is_unchanged,
    normalize_label,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

logger = logging.getLogger(__name__)


DATASETS = {
    "player_boxscores": {
        "release": "nba_stats_player_boxscores",
        "filename": "player_boxscores_{season}.parquet",
        "keys": ["game_id", "person_id"],
        "required_columns": {
            "game_id",
            "person_id",
            "team_id",
            "season",
            "season_type_id",
            "side",
            "position",
            "comment",
            "minutes",
            "points",
            "field_goals_made",
            "field_goals_attempted",
            "three_pointers_made",
            "three_pointers_attempted",
            "free_throws_made",
            "free_throws_attempted",
            "rebounds_offensive",
            "rebounds_defensive",
            "rebounds_total",
            "assists",
            "steals",
            "blocks",
            "turnovers",
            "fouls_personal",
            "plus_minus_points",
        },
    },
    "player_game_logs": {
        "release": "nba_stats_player_game_logs",
        "filename": "player_game_logs_{season}.parquet",
        "keys": ["game_id", "player_id"],
        "required_columns": {
            "game_id",
            "player_id",
            "team_id",
            "season",
            "season_type",
            "game_date",
            "matchup",
            "wl",
            "min",
            "pts",
            "fgm",
            "fga",
            "fg3m",
            "fg3a",
            "ftm",
            "fta",
            "oreb",
            "dreb",
            "reb",
            "ast",
            "stl",
            "blk",
            "tov",
            "pf",
            "plus_minus",
        },
    },
    "team_boxscores": {
        "release": "nba_stats_team_boxscores",
        "filename": "team_boxscores_{season}.parquet",
        "keys": ["game_id", "team_id"],
        "required_columns": {
            "game_id",
            "team_id",
            "season",
            "season_type_id",
            "side",
            "minutes",
            "points",
            "field_goals_made",
            "field_goals_attempted",
            "three_pointers_made",
            "three_pointers_attempted",
            "free_throws_made",
            "free_throws_attempted",
            "rebounds_offensive",
            "rebounds_defensive",
            "rebounds_total",
            "assists",
            "steals",
            "blocks",
            "turnovers",
            "fouls_personal",
            "plus_minus_points",
        },
    },
    "schedules": {
        "release": "nba_stats_schedules",
        "filename": "nba_schedule_{season}.parquet",
        "keys": ["game_id"],
        "required_columns": {
            "game_id",
            "season",
            "season_type",
            "game_date",
            "home_team_id",
            "away_team_id",
            "home_pts",
            "away_pts",
            "home_wl",
            "away_wl",
        },
    },
    "lineups": {
        "release": "nba_stats_game_lineups",
        "filename": "nba_lineups_{season}.parquet",
        "keys": ["game_id", "action_number"],
        "required_columns": {
            "game_id",
            "season",
            "action_number",
            "period",
            *[f"home_player_{i}" for i in range(1, 6)],
            *[f"away_player_{i}" for i in range(1, 6)],
        },
    },
}


CORE_DATASETS = (
    "player_boxscores",
    "player_game_logs",
    "team_boxscores",
    "schedules",
    "lineups",
)


def utc_now():

    return datetime.now(timezone.utc)


def normalize_game_ids(series):

    values = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

    valid = values.notna() & values.str.fullmatch(
        r"\d{1,10}",
        na=False,
    )

    if not valid.all():

        raise ValueError(
            f"Invalid game identifiers: " f"{values.loc[~valid].head().tolist()}"
        )

    return values.str.zfill(10)


def normalize_integer_ids(series, label):

    values = pd.to_numeric(
        series,
        errors="raise",
    )

    valid = values.notna() & values.gt(0) & values.mod(1).eq(0)

    if not valid.all():

        raise ValueError(f"Invalid identifiers in {label}")

    return values.astype("int64")


def identify_team_boxscore_placeholders(df):
    """
    Identify placeholder team boxscores for games whose
    participating teams have not yet been determined.

    A placeholder game must contain exactly two records:
    one home and one away, both with zero-valued team IDs,
    missing team metadata, and missing performance statistics.
    """
    team_metadata = [
        "team_city",
        "team_name",
        "team_tricode",
        "team_slug",
    ]

    stat_columns = [
        "minutes",
        "field_goals_made",
        "field_goals_attempted",
        "field_goals_percentage",
        "three_pointers_made",
        "three_pointers_attempted",
        "three_pointers_percentage",
        "free_throws_made",
        "free_throws_attempted",
        "free_throws_percentage",
        "rebounds_offensive",
        "rebounds_defensive",
        "rebounds_total",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "fouls_personal",
        "points",
        "plus_minus_points",
    ]

    required = {
        "game_id",
        "team_id",
        "side",
        *team_metadata,
        *stat_columns,
    }

    mask = pd.Series(False, index=df.index, dtype=bool)

    if not required.issubset(df.columns):
        return mask

    team_ids = pd.to_numeric(
        df["team_id"],
        errors="coerce",
    )

    candidates = (
        team_ids.eq(0)
        & df[team_metadata].isna().all(axis=1)
        & df[stat_columns].isna().all(axis=1)
        & df["side"].isin(["home", "away"])
        & df["game_id"].notna()
    ).fillna(False)

    candidate_games = df.loc[candidates, "game_id"].unique()

    for game_id in candidate_games:
        game_mask = df["game_id"].eq(game_id).fillna(False)
        game_rows = df.loc[game_mask]

        if (
            len(game_rows) == 2
            and set(game_rows["side"]) == {"home", "away"}
            and candidates.loc[game_mask].all()
        ):
            mask.loc[game_mask] = True

    return mask


def validate_game_dataset(df, dataset, season):

    config = DATASETS[dataset]

    df = df.copy()

    df.columns = df.columns.str.lower()

    if df.columns.duplicated().any():

        raise ValueError(f"{dataset}: Duplicate column names")

    missing_columns = config["required_columns"] - set(df.columns)

    if missing_columns:

        raise ValueError(
            f"{dataset} {season}: Missing columns " f"{sorted(missing_columns)}"
        )

    if df.empty:

        raise ValueError(f"{dataset} {season}: Empty dataset")

    season_values = pd.to_numeric(
        df["season"],
        errors="raise",
    )

    if not season_values.eq(season).fillna(False).all():

        raise ValueError(f"{dataset} {season}: Unexpected season values")

    df["game_id"] = normalize_game_ids(df["game_id"])

    source_rows = len(df)
    placeholder_rows = 0
    placeholder_game_ids = []

    if dataset == "team_boxscores":
        placeholder_mask = identify_team_boxscore_placeholders(df)

        placeholder_rows = int(placeholder_mask.sum())

        placeholder_game_ids = df.loc[placeholder_mask, "game_id"].unique().tolist()

        if placeholder_rows:
            logger.warning(
                "%s %s: Identified %s placeholder rows " "across %s games: %s",
                dataset,
                season,
                placeholder_rows,
                len(placeholder_game_ids),
                placeholder_game_ids,
            )

            df = df.loc[~placeholder_mask].copy()

            if df.empty:
                raise ValueError(f"{dataset} {season}: " "No non-placeholder records")

    for column in [
        "player_id",
        "person_id",
        "team_id",
        "home_team_id",
        "away_team_id",
    ]:

        if column in df.columns:

            try:

                df[column] = normalize_integer_ids(
                    df[column],
                    column,
                )

            except ValueError as exc:

                values = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

                valid = (values.notna() & values.gt(0) & values.mod(1).eq(0)).fillna(
                    False
                )

                invalid_values = df.loc[~valid, column].head(10).tolist()

                raise ValueError(
                    f"{dataset} | season={season} | "
                    f"column={column} | "
                    f"invalid_rows={int((~valid).sum())} | "
                    f"sample_values={invalid_values}"
                ) from exc

    keys = config["keys"]

    missing_key_rows = int(df[keys].isna().any(axis=1).sum())

    if missing_key_rows:

        raise ValueError(
            f"{dataset} {season}: " f"{missing_key_rows} rows with missing keys"
        )

    duplicate_rows = int(df.duplicated(keys, keep=False).sum())

    if duplicate_rows:

        # Lineup action identifiers require additional
        # semantic validation before transformation.
        if dataset == "lineups":

            logger.warning(
                "%s %s: %s rows with duplicate action keys",
                dataset,
                season,
                duplicate_rows,
            )

        else:

            raise ValueError(
                f"{dataset} {season}: " f"{duplicate_rows} rows with duplicate keys"
            )

    game_prefix_counts = df["game_id"].str[:3].value_counts().to_dict()

    report = {
        "rows": source_rows,
        "validated_rows": len(df),
        "placeholder_rows": placeholder_rows,
        "placeholder_games": len(placeholder_game_ids),
        "placeholder_game_ids": placeholder_game_ids,
        "columns": len(df.columns),
        "unique_games": int(df["game_id"].nunique()),
        "duplicate_key_rows": duplicate_rows,
        "missing_key_rows": missing_key_rows,
        "game_prefix_counts": {str(k): int(v) for k, v in game_prefix_counts.items()},
        "entirely_null_columns": (df.columns[df.isna().all()].tolist()),
    }

    for column in ["player_id", "person_id"]:

        if column in df.columns:

            report["unique_players"] = int(df[column].nunique())

    if "team_id" in df.columns:

        report["unique_teams"] = int(df["team_id"].nunique())

    if "season_type" in df.columns:

        counts = normalize_label(df["season_type"]).value_counts(dropna=False)

        report["season_type_counts"] = {str(k): int(v) for k, v in counts.items()}

    if "season_type_id" in df.columns:

        counts = df["season_type_id"].value_counts(dropna=False)

        report["season_type_id_counts"] = {str(k): int(v) for k, v in counts.items()}

    if "game_date" in df.columns:

        dates = pd.to_datetime(
            df["game_date"],
            errors="coerce",
        )

        report["invalid_game_dates"] = int(dates.isna().sum())

        if dates.notna().any():

            report["first_game_date"] = dates.min().isoformat()

            report["last_game_date"] = dates.max().isoformat()

    if dataset == "team_boxscores":

        teams_per_game = df.groupby("game_id")["team_id"].nunique()

        report["games_without_two_teams"] = int(teams_per_game.ne(2).sum())

    if dataset == "schedules":

        completed = (
            df["home_pts"].notna()
            & df["away_pts"].notna()
            & df["home_wl"].isin(["W", "L"])
            & df["away_wl"].isin(["W", "L"])
            & df["home_wl"].ne(df["away_wl"])
        )

        report["completed_games"] = int(completed.sum())

        report["other_games"] = int((~completed).sum())

    return report


def download_game_dataset(
    dataset,
    season,
    asset,
    session,
    snapshot=False,
    force=False,
):

    retrieved_at = utc_now()

    if snapshot:

        directory = RAW_DIR / "snapshots" / dataset / str(season)

        timestamp = retrieved_at.strftime("%Y-%m-%dT%H%M%S%fZ")

        path = directory / f"{timestamp}.parquet"

        existing = sorted(
            candidate
            for candidate in directory.glob("*.parquet")
            if candidate.with_suffix(".json").exists()
        )

        if existing and not force:

            latest = existing[-1]

            if asset_is_unchanged(latest, asset):

                logger.info(
                    "Unchanged snapshot: %s",
                    latest,
                )

                return {
                    **read_metadata(latest),
                    "path": str(latest),
                    "status": "unchanged",
                }

    else:

        path = RAW_DIR / dataset / f"{dataset}_{season}.parquet"

        if not force and asset_is_unchanged(path, asset):

            logger.info(
                "Unchanged historical file: %s",
                path,
            )

            return {
                **read_metadata(path),
                "path": str(path),
                "status": "unchanged",
            }

    response = session.get(
        asset["browser_download_url"],
        timeout=120,
    )

    response.raise_for_status()

    content = response.content

    if len(content) != asset["size"]:

        raise ValueError(f"{dataset} {season}: " f"Unexpected download size")

    df = pd.read_parquet(BytesIO(content))

    validation = validate_game_dataset(
        df,
        dataset,
        season,
    )

    metadata = {
        "dataset": dataset,
        "season": season,
        "snapshot": snapshot,
        "asset_id": asset["id"],
        "asset_name": asset["name"],
        "source_release": DATASETS[dataset]["release"],
        "source_url": asset["browser_download_url"],
        "source_created_at": asset["created_at"],
        "source_updated_at": asset["updated_at"],
        "retrieved_at": retrieved_at.isoformat(),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "validation": validation,
    }

    atomic_write(path, content)

    atomic_write(
        path.with_suffix(".json"),
        json.dumps(
            metadata,
            indent=2,
            allow_nan=False,
        ).encode("utf-8"),
    )

    logger.info(
        "Downloaded %s | %s | %s rows | %s games",
        dataset,
        season,
        validation["rows"],
        validation["unique_games"],
    )

    return {
        **metadata,
        "path": str(path),
        "status": "downloaded",
    }


def acquire_game_data(
    seasons,
    datasets=CORE_DATASETS,
    snapshot=False,
    force=False,
):

    seasons = list(seasons)

    results = []

    with requests.Session() as session:

        release_cache = {}

        for dataset in datasets:

            config = DATASETS[dataset]

            release = config["release"]

            if release not in release_cache:

                release_cache[release] = get_release_assets(
                    release,
                    session,
                )

            assets = release_cache[release]

            for season in seasons:

                filename = config["filename"].format(season=season)

                if filename not in assets:

                    raise FileNotFoundError(f"Asset not found: {filename}")

                result = download_game_dataset(
                    dataset=dataset,
                    season=season,
                    asset=assets[filename],
                    session=session,
                    snapshot=snapshot,
                    force=force,
                )

                results.append(result)

    return results


def download_historical_games(
    start_season=2016,
    end_season=2026,
    datasets=CORE_DATASETS,
    force=False,
):

    return acquire_game_data(
        seasons=range(
            start_season,
            end_season + 1,
        ),
        datasets=datasets,
        snapshot=False,
        force=force,
    )


def download_current_games(
    season,
    datasets=CORE_DATASETS,
    force=False,
):

    return acquire_game_data(
        seasons=[season],
        datasets=datasets,
        snapshot=True,
        force=force,
    )
