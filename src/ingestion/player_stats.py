"""Acquire historical and current NBA player-season statistics."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORICAL_DIR = PROJECT_ROOT / "data/raw/player_stats"

SNAPSHOT_DIR = PROJECT_ROOT / "data/raw/snapshots/player_stats"

RELEASE_API = (
    "https://api.github.com/repos/"
    "sportsdataverse/sportsdataverse-data/releases/tags/"
    "nba_stats_player_season_stats"
)

REQUIRED_COLUMNS = {
    "player_id",
    "player_name",
    "team_id",
    "season",
    "season_type",
    "measure_type",
    "per_mode",
    "age",
    "gp",
    "min",
    "pts",
    "ts_pct",
    "usg_pct",
    "poss",
}

PRIMARY_KEY = [
    "player_id",
    "season",
    "season_type",
    "measure_type",
    "per_mode",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Release discovery
# --------------------------------------------------


def get_release_assets() -> dict:
    """Retrieve available Parquet assets from GitHub."""

    response = requests.get(RELEASE_API, timeout=30)
    response.raise_for_status()

    assets = response.json()["assets"]

    return {
        asset["name"]: asset for asset in assets if asset["name"].endswith(".parquet")
    }


def get_season_asset(
    season: int,
    assets: dict | None = None,
) -> dict:
    """Locate the release asset for a season-ending year."""

    if assets is None:
        assets = get_release_assets()

    filename = f"player_season_stats_{season}.parquet"

    if filename not in assets:
        raise FileNotFoundError(f"No release asset found for season {season}")

    return assets[filename]


# --------------------------------------------------
# Data validation
# --------------------------------------------------


def validate_player_stats(
    df: pd.DataFrame,
    season: int,
) -> dict:
    """Validate raw player-season data."""

    if df.empty:
        raise ValueError("Downloaded dataset is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[PRIMARY_KEY].isna().any().any():
        raise ValueError("Null values found in primary key.")

    seasons = set(df["season"].unique())

    if seasons != {season}:
        raise ValueError(f"Expected season {season}; found {seasons}")

    duplicates = int(df.duplicated(PRIMARY_KEY).sum())

    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate primary keys.")

    regular = df.loc[df["season_type"] == "regular-season"]

    if regular.empty:
        raise ValueError("No regular-season records found.")

    records_per_player = regular.groupby("player_id").size()

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "unique_players": int(df["player_id"].nunique()),
        "regular_season_players": int(regular["player_id"].nunique()),
        "duplicate_keys": duplicates,
        "season_types": sorted(df["season_type"].unique().tolist()),
        "measure_types": sorted(df["measure_type"].dropna().unique().tolist()),
        "per_modes": sorted(df["per_mode"].dropna().unique().tolist()),
        "records_per_player": {
            str(k): int(v)
            for k, v in (records_per_player.value_counts().sort_index().items())
        },
        "max_games_played": int(regular["gp"].max()),
    }


# --------------------------------------------------
# File utilities
# --------------------------------------------------


def calculate_checksum(content: bytes) -> str:
    """Return the SHA-256 checksum."""

    return hashlib.sha256(content).hexdigest()


def metadata_path(path: Path) -> Path:
    """Return the corresponding metadata file path."""

    return path.with_suffix(".json")


def read_metadata(path: Path) -> dict:
    """Read existing ingestion metadata."""

    path = metadata_path(path)

    if not path.exists():
        return {}

    return json.loads(path.read_text())


def atomic_write(path: Path, content: bytes) -> None:
    """Write a file through a temporary path."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def asset_is_unchanged(
    path: Path,
    asset: dict,
) -> bool:
    """Check upstream identity and local file integrity."""

    if not path.exists():
        return False

    metadata = read_metadata(path)

    same_asset = (
        metadata.get("asset_id") == asset["id"]
        and metadata.get("source_updated_at") == asset["updated_at"]
    )

    if not same_asset:
        return False

    local_checksum = calculate_checksum(path.read_bytes())

    return local_checksum == metadata.get("sha256")


# --------------------------------------------------
# Download and persistence
# --------------------------------------------------


def download_asset(
    asset: dict,
    season: int,
    output_path: Path,
) -> dict:
    """Download, validate and persist a release asset."""

    logger.info("Downloading %s", asset["name"])

    response = requests.get(
        asset["browser_download_url"],
        timeout=120,
    )

    response.raise_for_status()

    content = response.content

    if len(content) != asset["size"]:
        raise ValueError("Downloaded file size differs from release metadata.")

    df = pd.read_parquet(BytesIO(content))

    validation = validate_player_stats(df, season)

    metadata = {
        "season": season,
        "asset_id": asset["id"],
        "asset_name": asset["name"],
        "source_url": asset["browser_download_url"],
        "source_created_at": asset["created_at"],
        "source_updated_at": asset["updated_at"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": len(content),
        "sha256": calculate_checksum(content),
        "validation": validation,
    }

    atomic_write(output_path, content)

    atomic_write(
        metadata_path(output_path),
        json.dumps(metadata, indent=2).encode("utf-8"),
    )

    logger.info(
        "Saved %s | %s rows | %s players",
        output_path,
        validation["rows"],
        validation["unique_players"],
    )

    return metadata


# --------------------------------------------------
# Historical ingestion
# --------------------------------------------------


def download_historical_season(
    season: int,
    assets: dict | None = None,
    force: bool = False,
) -> dict:
    """Acquire one completed historical season."""

    asset = get_season_asset(season, assets)

    output_path = HISTORICAL_DIR / f"player_season_stats_{season}.parquet"

    if not force and asset_is_unchanged(output_path, asset):
        logger.info("Season %s unchanged; skipping.", season)

        return {
            "season": season,
            "status": "unchanged",
            "path": str(output_path),
        }

    metadata = download_asset(
        asset,
        season,
        output_path,
    )

    return {
        "season": season,
        "status": "downloaded",
        "path": str(output_path),
        **metadata,
    }


def download_historical_seasons(
    start_season: int = 2016,
    end_season: int = 2026,
    force: bool = False,
) -> list[dict]:
    """Acquire an inclusive range of completed seasons."""

    if start_season > end_season:
        raise ValueError("Invalid season range.")

    assets = get_release_assets()

    results = []

    for season in range(start_season, end_season + 1):
        try:
            result = download_historical_season(
                season=season,
                assets=assets,
                force=force,
            )

        except Exception as exc:
            logger.exception("Failed to acquire season %s", season)

            result = {
                "season": season,
                "status": "failed",
                "error": str(exc),
            }

        results.append(result)

    return results


# --------------------------------------------------
# Current-season ingestion
# --------------------------------------------------


def download_current_snapshot(
    season: int,
    force: bool = False,
) -> dict:
    """Preserve a snapshot of current-season statistics."""

    asset = get_season_asset(season)

    season_dir = SNAPSHOT_DIR / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    existing_snapshots = sorted(
        season_dir.glob("*.parquet"),
        reverse=True,
    )

    if not force:
        for existing in existing_snapshots:
            if asset_is_unchanged(existing, asset):
                logger.info("Current-season asset unchanged.")

                return {
                    "season": season,
                    "status": "unchanged",
                    "path": str(existing),
                }

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S%fZ")

    output_path = season_dir / f"{timestamp}.parquet"

    metadata = download_asset(
        asset,
        season,
        output_path,
    )

    return {
        "season": season,
        "status": "downloaded",
        "path": str(output_path),
        **metadata,
    }
