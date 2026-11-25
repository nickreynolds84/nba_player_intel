"""Orchestrate game-level integrity audits using local raw data."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.validation.game_data import audit_season

logger = logging.getLogger(__name__)


GAME_DATASETS = (
    "player_boxscores",
    "player_game_logs",
    "team_boxscores",
    "schedules",
    "lineups",
)


def resolve_dataset_path(
    dataset,
    season,
    raw_dir="data/raw",
    snapshot_season=2027,
):
    """Resolve a historical file or latest current-season snapshot."""

    if dataset not in GAME_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    raw_dir = Path(raw_dir)

    if season == snapshot_season:

        snapshot_dir = raw_dir / "snapshots" / dataset / str(season)

        candidates = sorted(snapshot_dir.glob("*.parquet"))

        if not candidates:
            raise FileNotFoundError(f"No snapshots found in {snapshot_dir}")

        return candidates[-1]

    path = raw_dir / dataset / f"{dataset}_{season}.parquet"

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def resolve_season_paths(
    season,
    raw_dir="data/raw",
    snapshot_season=2027,
):
    """Resolve all five dataset paths for a season."""

    return {
        dataset: resolve_dataset_path(
            dataset=dataset,
            season=season,
            raw_dir=raw_dir,
            snapshot_season=snapshot_season,
        )
        for dataset in GAME_DATASETS
    }


def load_season_datasets(paths):
    """Load one season of raw game datasets."""

    datasets = {}

    for dataset, path in paths.items():

        logger.info(
            "Loading %s from %s",
            dataset,
            path,
        )

        df = pd.read_parquet(path)

        logger.info(
            "%s: %s rows, %s columns",
            dataset,
            len(df),
            len(df.columns),
        )

        datasets[dataset] = df

    return datasets


def save_season_audit(
    result,
    season_dir,
    source_paths,
):
    """Persist audit summaries, diagnostics and provenance."""

    season_dir = Path(season_dir)
    season_dir.mkdir(parents=True, exist_ok=True)

    with (season_dir / "summary.json").open("w") as f:
        json.dump(
            result["summary"],
            f,
            indent=2,
            default=str,
        )

    manifest = {
        "season": result["summary"]["season"],
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
    }

    with (season_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    details_dir = season_dir / "details"
    details_dir.mkdir(exist_ok=True)

    for name, df in result["details"].items():

        df.to_parquet(
            details_dir / f"{name}.parquet",
            index=None,
        )


def run_historical_audit(
    start_season=2016,
    end_season=2027,
    raw_dir="data/raw",
    output_dir="data/validation/game_data",
    snapshot_season=2027,
    team_seconds_tolerance=5,
    continue_on_error=True,
):
    """Audit seasons sequentially and persist results."""

    if start_season > end_season:
        raise ValueError("start_season must not exceed end_season")

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S%fZ")

    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    summaries = []
    failures = []

    for season in range(
        start_season,
        end_season + 1,
    ):

        logger.info(
            "Starting audit for season %s",
            season,
        )

        datasets = None
        result = None

        try:

            paths = resolve_season_paths(
                season=season,
                raw_dir=raw_dir,
                snapshot_season=snapshot_season,
            )

            datasets = load_season_datasets(paths)

            result = audit_season(
                datasets=datasets,
                season=season,
                team_seconds_tolerance=(team_seconds_tolerance),
            )

            save_season_audit(
                result=result,
                season_dir=run_dir / str(season),
                source_paths=paths,
            )

            summaries.append(
                {
                    **result["summary"],
                    "audit_status": "completed",
                }
            )

            logger.info(
                "Completed audit for season %s",
                season,
            )

        except Exception as exc:

            failure = {
                "season": season,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

            failures.append(failure)

            summaries.append(
                {
                    "season": season,
                    "audit_status": "failed",
                }
            )

            logger.exception(
                "Audit failed for season %s",
                season,
            )

            with (run_dir / "failures.json").open("w") as f:
                json.dump(failures, f, indent=2)

            if not continue_on_error:
                raise

        finally:

            del datasets
            del result

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        run_dir / "historical_summary.csv",
        index=False,
    )

    with (run_dir / "failures.json").open("w") as f:
        json.dump(failures, f, indent=2)

    logger.info(
        "Historical audit results saved to %s",
        run_dir,
    )

    return {
        "run_dir": run_dir,
        "summary": summary_df,
        "failures": failures,
    }
