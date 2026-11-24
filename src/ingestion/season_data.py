from __future__ import annotations

import hashlib
import json
import logging
import os

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

GITHUB_API = "https://api.github.com/repos/" "sportsdataverse/sportsdataverse-data"

logger = logging.getLogger(__name__)


DATASETS = {
    "player_master": {
        "release": "nba_stats_leaguedash",
        "filename": "player_master_{season}.parquet",
        "required_columns": {
            "player_id",
            "player_name",
            "season",
            "season_type",
            "per_mode",
            "gp",
            "min",
            "pts",
            "adv_ts_pct",
            "adv_def_rating",
            "ptdef_def_rim_fga",
            "ptdef_def_rim_fgm",
            "ptdef_def_rim_fg_pct",
        },
    },
    "player_hustle": {
        "release": "nba_stats_hustle",
        "filename": (
            "leaguehustlestatsplayer_" "regular-season_totals_{season}.parquet"
        ),
        "required_columns": {
            "player_id",
            "player_name",
            "season",
            "season_type",
            "per_mode",
            "g",
            "min",
            "contested_shots",
            "deflections",
            "charges_drawn",
            "def_boxouts",
            "def_loose_balls_recovered",
        },
    },
}


def utc_now():
    return datetime.now(timezone.utc)


def normalize_label(series):
    return (
        series.astype("string")
        .str.lower()
        .str.replace("-", " ", regex=False)
        .str.replace("_", " ", regex=False)
        .str.strip()
    )


def get_release_assets(release, session):
    """Retrieve all assets from a GitHub release."""

    response = session.get(
        f"{GITHUB_API}/releases/tags/{release}",
        timeout=30,
    )

    response.raise_for_status()

    release_id = response.json()["id"]

    assets = []
    page = 1

    while True:

        response = session.get(
            f"{GITHUB_API}/releases/{release_id}/assets",
            params={
                "per_page": 100,
                "page": page,
            },
            timeout=30,
        )

        response.raise_for_status()

        batch = response.json()
        assets.extend(batch)

        if len(batch) < 100:
            break

        page += 1

    return {asset["name"]: asset for asset in assets}


def validate_dataset(df, dataset, season):
    """Validate structure without removing players."""

    config = DATASETS[dataset]

    missing_columns = config["required_columns"] - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset} {season}: " f"Missing columns {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(f"{dataset} {season}: Empty dataset")

    if df["season"].isna().any():
        raise ValueError("Missing season values")

    if set(df["season"].unique()) != {season}:
        raise ValueError(f"Unexpected season values: " f"{df['season'].unique()}")

    season_type = normalize_label(df["season_type"])
    per_mode = normalize_label(df["per_mode"])

    if season_type.isna().any():
        raise ValueError("Missing season type")

    if not season_type.isin(["regular season", "playoffs"]).all():
        raise ValueError("Unexpected season type")

    if not per_mode.eq("totals").fillna(False).all():
        raise ValueError("Unexpected normalization mode")

    if df["player_id"].isna().any():
        raise ValueError("Missing player identifiers")

    player_ids = pd.to_numeric(df["player_id"], errors="raise")

    if not ((player_ids > 0) & (player_ids % 1 == 0)).all():
        raise ValueError("Invalid player identifiers")

    keys = pd.DataFrame(
        {
            "player_id": player_ids,
            "season": df["season"],
            "season_type": season_type,
        }
    )

    if keys.duplicated().any():
        raise ValueError(f"{dataset} {season}: Duplicate player keys")

    regular = df.loc[season_type == "regular season"]

    if regular.empty:
        raise ValueError(f"{dataset} {season}: " "No regular-season records")

    if dataset == "player_hustle":
        if not season_type.eq("regular season").all():
            raise ValueError("Unexpected non-regular-season hustle records")

    missingness = regular.isna().mean().mul(100).round(4)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "unique_players": int(df["player_id"].nunique()),
        "regular_season_players": int(regular["player_id"].nunique()),
        "playoff_players": int(
            df.loc[
                season_type == "playoffs",
                "player_id",
            ].nunique()
        ),
        "regular_season_missingness_pct": (missingness.to_dict()),
        "entirely_null_regular_season_columns": (
            missingness[missingness == 100].index.tolist()
        ),
    }


def atomic_write(path, content):
    """Write a file without leaving partial downloads."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:
        with NamedTemporaryFile(
            dir=path.parent,
            delete=False,
        ) as temporary:

            temporary_path = Path(temporary.name)
            temporary.write(content)

        os.replace(temporary_path, path)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def read_metadata(path):

    metadata_path = path.with_suffix(".json")

    if not metadata_path.exists():
        return None

    try:
        return json.loads(metadata_path.read_text())
    except (OSError, ValueError):
        return None


def asset_is_unchanged(path, asset):

    if not path.exists():
        return False

    metadata = read_metadata(path)

    if metadata is None:
        return False

    if not (
        metadata.get("asset_id") == asset["id"]
        and metadata.get("source_updated_at") == asset["updated_at"]
        and metadata.get("size_bytes") == asset["size"]
    ):
        return False

    checksum = hashlib.sha256(path.read_bytes()).hexdigest()

    return checksum == metadata.get("sha256")


def download_dataset(
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

        existing = sorted(directory.glob("*.parquet"))

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
        timeout=60,
    )

    response.raise_for_status()

    content = response.content

    if len(content) != asset["size"]:
        raise ValueError(f"Unexpected download size for {dataset} " f"{season}")

    df = pd.read_parquet(BytesIO(content))

    validation = validate_dataset(
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
        ).encode("utf-8"),
    )

    logger.info(
        "Downloaded %s | %s | %s players",
        dataset,
        season,
        validation["regular_season_players"],
    )

    return {
        **metadata,
        "path": str(path),
        "status": "downloaded",
    }


def acquire_season_data(
    seasons,
    datasets=("player_master", "player_hustle"),
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

                result = download_dataset(
                    dataset=dataset,
                    season=season,
                    asset=assets[filename],
                    session=session,
                    snapshot=snapshot,
                    force=force,
                )

                results.append(result)

    return results


def download_historical_seasons(
    start_season=2016,
    end_season=2026,
    force=False,
):

    return acquire_season_data(
        seasons=range(
            start_season,
            end_season + 1,
        ),
        snapshot=False,
        force=force,
    )


def download_current_snapshot(
    season,
    force=False,
):

    return acquire_season_data(
        seasons=[season],
        snapshot=True,
        force=force,
    )
