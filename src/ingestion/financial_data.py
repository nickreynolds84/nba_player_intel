"""Acquisition and snapshot management for NBA financial data."""

from __future__ import annotations

import json
import logging
import shutil

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_DIR = PROJECT_ROOT / "data/raw/financial"

HOOPSHYPE_DATASET = "Mr-Bridge/nba-salary-cap-contracts-2016-2026"

HOOPSHYPE_BASE_URL = (
    "https://huggingface.co/datasets/" f"{HOOPSHYPE_DATASET}/resolve/main"
)

HOOPSHYPE_FILES = {
    "player_salaries": "player_salaries.csv",
    "salary_caps": "cap_tax_aprons.csv",
}

COURTSKETCH_URL = "https://courtsketch.com/transaction_search"

REQUIRED_COLUMNS = {
    "player_salaries": {
        "season_year",
        "season",
        "player_id",
        "player",
        "salary",
        "team_id",
        "team",
    },
    "salary_caps": {
        "season_year",
        "season",
        "salary_cap",
    },
    "contracts": {
        "date",
        "kind",
        "type",
        "teams",
        "player_or_assets",
        "years",
        "total_value_musd",
        "aav_musd",
        "terms_stated",
        "option",
        "flags",
        "notes",
    },
    "trades": {
        "date",
        "kind",
        "type",
        "teams",
        "player_or_assets",
        "notes",
    },
}

SOURCE_DATASETS = {
    "hoopshype": {
        "player_salaries",
        "salary_caps",
    },
    "courtsketch": {
        "contracts",
        "trades",
    },
}


def utc_now() -> datetime:
    """Return the current UTC datetime."""

    return datetime.now(timezone.utc)


def snapshot_timestamp() -> str:
    """Generate a filesystem-safe UTC timestamp."""

    return utc_now().strftime("%Y-%m-%dT%H%M%S%fZ")


def validate_financial_dataset(
    df: pd.DataFrame,
    dataset: str,
) -> None:
    """Validate essential ingestion requirements."""

    if dataset not in REQUIRED_COLUMNS:
        raise ValueError(f"Unsupported financial dataset: {dataset}")

    if df.empty:
        raise ValueError(f"Financial dataset is empty: {dataset}")

    missing = REQUIRED_COLUMNS[dataset] - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset} is missing required columns: " f"{sorted(missing)}"
        )


def save_financial_snapshot(
    source_system: str,
    datasets: dict[str, pd.DataFrame],
    source_locations: dict[str, str],
    acquisition_method: str,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> dict:
    """Persist a complete source-level financial snapshot."""

    if source_system not in SOURCE_DATASETS:
        raise ValueError(f"Unsupported financial source: {source_system}")

    expected = SOURCE_DATASETS[source_system]

    if set(datasets) != expected:
        raise ValueError(f"{source_system} requires datasets: " f"{sorted(expected)}")

    if set(source_locations) != expected:
        raise ValueError("Source locations must be supplied " "for every dataset.")

    for dataset, df in datasets.items():
        validate_financial_dataset(df, dataset)

    raw_dir = Path(raw_dir)

    snapshot_root = raw_dir / source_system / "snapshots"

    snapshot_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = snapshot_timestamp()

    snapshot_dir = snapshot_root / timestamp

    temporary_dir = snapshot_root / f".tmp_{uuid4().hex}"

    temporary_dir.mkdir()

    metadata = {
        "source_system": source_system,
        "acquisition_method": acquisition_method,
        "ingested_at_utc": utc_now().isoformat(),
        "snapshot_id": timestamp,
        "datasets": {},
    }

    try:

        for dataset, df in datasets.items():

            filename = f"{dataset}.parquet"

            output_path = temporary_dir / filename

            df.to_parquet(
                output_path,
                index=False,
            )

            metadata["datasets"][dataset] = {
                "filename": filename,
                "source_location": (source_locations[dataset]),
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": df.columns.tolist(),
            }

        metadata_path = temporary_dir / "metadata.json"

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2,
            )

        temporary_dir.rename(snapshot_dir)

    except Exception:

        shutil.rmtree(
            temporary_dir,
            ignore_errors=True,
        )

        raise

    paths = {dataset: (snapshot_dir / f"{dataset}.parquet") for dataset in datasets}

    logger.info(
        "Saved %s financial snapshot: %s",
        source_system,
        snapshot_dir,
    )

    return {
        "source_system": source_system,
        "snapshot_dir": snapshot_dir,
        "datasets": paths,
        "metadata": snapshot_dir / "metadata.json",
    }


def download_hoopshype_data(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    base_url: str = HOOPSHYPE_BASE_URL,
) -> dict:
    """Download salary and salary cap source datasets."""

    datasets = {}
    source_locations = {}

    for dataset, filename in HOOPSHYPE_FILES.items():

        url = f"{base_url.rstrip('/')}/{filename}"

        logger.info(
            "Downloading financial dataset: %s",
            dataset,
        )

        datasets[dataset] = pd.read_csv(url)

        source_locations[dataset] = url

    return save_financial_snapshot(
        source_system="hoopshype",
        datasets=datasets,
        source_locations=source_locations,
        acquisition_method="remote_csv_download",
        raw_dir=raw_dir,
    )


def import_hoopshype_data(
    salaries_path: str | Path,
    salary_caps_path: str | Path,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> dict:
    """Import previously acquired HoopsHype parquet files."""

    source_paths = {
        "player_salaries": Path(salaries_path),
        "salary_caps": Path(salary_caps_path),
    }

    datasets = {}

    for dataset, path in source_paths.items():

        if not path.is_file():
            raise FileNotFoundError(path)

        datasets[dataset] = pd.read_parquet(path)

    return save_financial_snapshot(
        source_system="hoopshype",
        datasets=datasets,
        source_locations={
            dataset: str(path.resolve()) for dataset, path in source_paths.items()
        },
        acquisition_method="existing_parquet_import",
        raw_dir=raw_dir,
    )


def import_courtsketch_data(
    contracts_path: str | Path,
    trades_path: str | Path,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> dict:
    """Import manually exported CourtSketch CSV files."""

    source_paths = {
        "contracts": Path(contracts_path),
        "trades": Path(trades_path),
    }

    datasets = {}

    for dataset, path in source_paths.items():

        if not path.is_file():
            raise FileNotFoundError(path)

        datasets[dataset] = pd.read_csv(path)

    return save_financial_snapshot(
        source_system="courtsketch",
        datasets=datasets,
        source_locations={
            dataset: str(path.resolve()) for dataset, path in source_paths.items()
        },
        acquisition_method="manual_csv_export",
        raw_dir=raw_dir,
    )


def resolve_latest_financial_snapshot(
    source_system: str,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> Path:
    """Locate the latest completed financial snapshot."""

    if source_system not in SOURCE_DATASETS:
        raise ValueError(f"Unsupported financial source: {source_system}")

    snapshot_root = Path(raw_dir) / source_system / "snapshots"

    if not snapshot_root.is_dir():
        raise FileNotFoundError(snapshot_root)

    snapshots = sorted(
        (
            path
            for path in snapshot_root.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and (path / "metadata.json").is_file()
        ),
        reverse=True,
    )

    if not snapshots:
        raise FileNotFoundError(f"No completed snapshots found for " f"{source_system}")

    return snapshots[0]


def load_financial_snapshot(
    source_system: str,
    snapshot_dir: str | Path | None = None,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> dict[str, pd.DataFrame]:
    """Load all datasets from a completed source snapshot."""

    if source_system not in SOURCE_DATASETS:
        raise ValueError(f"Unsupported financial source: {source_system}")

    if snapshot_dir is None:

        snapshot_dir = resolve_latest_financial_snapshot(
            source_system=source_system,
            raw_dir=raw_dir,
        )

    snapshot_dir = Path(snapshot_dir)

    metadata_path = snapshot_dir / "metadata.json"

    with metadata_path.open(encoding="utf-8") as file:

        metadata = json.load(file)

    if metadata["source_system"] != source_system:
        raise ValueError("Snapshot source does not match " "requested source.")

    datasets = {}

    for dataset in sorted(SOURCE_DATASETS[source_system]):

        dataset_metadata = metadata["datasets"][dataset]

        path = snapshot_dir / dataset_metadata["filename"]

        df = pd.read_parquet(path)

        validate_financial_dataset(
            df,
            dataset,
        )

        if len(df) != dataset_metadata["row_count"]:
            raise ValueError(f"Snapshot row count mismatch: {dataset}")

        datasets[dataset] = df

    return datasets
