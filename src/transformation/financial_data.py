"""Normalization and entity resolution for NBA financial data."""

from __future__ import annotations

import json
import logging
import re

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ingestion import financial_data as ingestion
from src.entity_resolution import player_matcher as pm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/financial"

DEFAULT_IDENTITY_DIR = PROJECT_ROOT / "data/processed/identities"

DEFAULT_TEAM_CONFIG = PROJECT_ROOT / "config/financial/team_names.json"

MARKET_CONTRACT_TYPES = {
    "Signing",
    "Re-signing",
    "Extension",
}

CONTRACT_FLAG_MAPPING = {
    "is_duplicate_report": "duplicate report",
    "is_not_new_contract": "not a new contract",
    "is_non_guaranteed": "non-guaranteed",
    "is_resigning_flag": "re-signed",
    "is_terms_corrected": "terms corrected",
}

BIRTH_DATE_PATTERN = re.compile(
    r"\(\s*b\.",
    flags=re.IGNORECASE,
)


def load_team_names(
    config_path: str | Path = DEFAULT_TEAM_CONFIG,
) -> dict[str, str]:
    """Load canonical NBA team names."""

    with Path(config_path).open(encoding="utf-8") as file:
        return json.load(file)


def normalize_salary_seasons(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert source season years to NBA ending years."""

    result = df.copy()

    result = result.rename(
        columns={
            "season": "season_label",
            "season_year": "source_season_year",
        }
    )

    result["season"] = result["source_season_year"].astype(int) + 1

    return result


def normalize_salaries(
    raw_salaries: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize annual salary source records."""

    ingestion.validate_financial_dataset(
        raw_salaries,
        "player_salaries",
    )

    salaries = normalize_salary_seasons(raw_salaries)

    salaries = salaries.rename(
        columns={
            "player_id": "salary_source_player_id",
            "team_id": "salary_source_team_id",
            "player": "player_name",
        }
    )

    salaries["salary_source_player_id"] = salaries["salary_source_player_id"].astype(
        "Int64"
    )

    return salaries


def normalize_salary_caps(
    raw_salary_caps: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize annual league salary cap records."""

    ingestion.validate_financial_dataset(
        raw_salary_caps,
        "salary_caps",
    )

    return normalize_salary_seasons(raw_salary_caps)


def normalize_contracts(
    raw_contracts: pd.DataFrame,
    team_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize CourtSketch contract reports."""

    ingestion.validate_financial_dataset(
        raw_contracts,
        "contracts",
    )

    if team_names is None:
        team_names = load_team_names()

    contracts = raw_contracts.copy()

    contracts = contracts.rename(
        columns={
            "date": "agreement_date",
            "type": "contract_type",
            "teams": "team_abbreviation",
            "player_or_assets": "player_name",
            "years": "contract_years",
            "option": "option_type",
        }
    )

    contracts["agreement_date"] = pd.to_datetime(
        contracts["agreement_date"],
        errors="raise",
    )

    contracts["agreement_year"] = contracts["agreement_date"].dt.year

    contracts["source_system"] = "courtsketch"

    contracts["reported_contract_value"] = contracts["total_value_musd"] * 1_000_000

    contracts["reported_aav"] = contracts["aav_musd"] * 1_000_000

    contracts["calculated_aav"] = contracts["reported_contract_value"] / contracts[
        "contract_years"
    ].where(contracts["contract_years"].gt(0))

    contract_flags = contracts["flags"].fillna("").str.lower()

    for column, flag in CONTRACT_FLAG_MAPPING.items():

        contracts[column] = contract_flags.str.contains(
            flag,
            regex=False,
        )

    contracts["has_option"] = contracts[
        "option_type"
    ].notna() | contract_flags.str.contains(
        "option",
        regex=False,
    )

    contracts["has_complete_terms"] = contracts["contract_years"].gt(0) & contracts[
        "reported_contract_value"
    ].gt(0)

    contracts["market_valuation_candidate"] = (
        contracts["contract_type"].isin(MARKET_CONTRACT_TYPES)
        & contracts["has_complete_terms"]
        & ~contracts["is_duplicate_report"]
        & ~contracts["is_not_new_contract"]
    )

    contracts["team"] = contracts["team_abbreviation"].map(team_names)

    unknown_teams = (
        contracts.loc[
            contracts["team"].isna(),
            "team_abbreviation",
        ]
        .drop_duplicates()
        .tolist()
    )

    if unknown_teams:
        raise ValueError(f"Unmapped contract teams: {unknown_teams}")

    # Identity matching context only.
    # This is not the contract start season.
    contracts["matching_season"] = contracts["agreement_date"].dt.year + contracts[
        "agreement_date"
    ].dt.month.ge(7).astype(int)

    # CourtSketch does not provide stable player IDs.
    contracts["courtsketch_player_id"] = pd.Series(
        pd.NA,
        index=contracts.index,
        dtype="string",
    )

    return contracts


def extract_courtsketch_names(
    value,
) -> list[str]:
    """Extract source-provided alternative player names."""

    if pd.isna(value):
        return []

    value = str(value).strip()

    # Birth-date qualifiers must not be discarded.
    if BIRTH_DATE_PATTERN.search(value):
        return []

    candidates = []

    for part in value.split("/"):

        part = part.strip()

        if part:
            candidates.append(part)

        cleaned = re.sub(
            r"\([^)]*\)",
            "",
            part,
        ).strip()

        if cleaned:
            candidates.append(cleaned)

    return list(dict.fromkeys(candidates))


def resolve_salary_identities(
    salaries: pd.DataFrame,
    player_reference: pd.DataFrame,
    player_team_context: pd.DataFrame,
    crosswalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resolve HoopsHype players to canonical NBA IDs."""

    return pm.resolve_player_entities(
        records=salaries,
        player_reference=player_reference,
        player_team_context=player_team_context,
        source_system="hoopshype",
        source_id_col="salary_source_player_id",
        name_col="player_name",
        team_col="team",
        season_col="season",
        crosswalk=crosswalk,
    )


def resolve_contract_name_variants(
    resolved_contracts: pd.DataFrame,
    player_reference: pd.DataFrame,
    player_team_context: pd.DataFrame,
    crosswalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resolve alternative names in unmatched CourtSketch records."""

    result = resolved_contracts.copy()

    result["matched_alternative_name"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    alternatives = result.loc[result["player_id"].isna()].copy()

    alternatives["contract_row_id"] = alternatives.index

    alternatives["original_player_name"] = alternatives["player_name"]

    alternatives["alternative_name"] = alternatives["player_name"].map(
        extract_courtsketch_names
    )

    alternatives = alternatives.explode("alternative_name").dropna(
        subset=["alternative_name"]
    )

    alternatives = alternatives.loc[
        alternatives["alternative_name"].ne(alternatives["original_player_name"])
    ].copy()

    if alternatives.empty:
        return result

    resolved_alternatives = pm.resolve_player_entities(
        records=alternatives,
        player_reference=player_reference,
        player_team_context=player_team_context,
        source_system="courtsketch",
        source_id_col="courtsketch_player_id",
        name_col="alternative_name",
        team_col="team",
        season_col="matching_season",
        crosswalk=crosswalk,
    )

    successful = resolved_alternatives.loc[
        resolved_alternatives["player_id"].notna()
    ].copy()

    if successful.empty:
        return result

    context_keys = set(
        player_team_context[
            [
                "player_id",
                "season",
                "normalized_team",
            ]
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    successful["context_confirmed"] = [
        (player_id, season, team) in context_keys
        for player_id, season, team in zip(
            successful["player_id"],
            successful["matching_season"],
            successful["normalized_team"],
        )
    ]

    successful["has_parenthetical"] = successful["original_player_name"].str.contains(
        "(",
        regex=False,
        na=False,
    )

    identity_counts = successful.groupby("contract_row_id")["player_id"].nunique()

    unambiguous_rows = identity_counts[identity_counts.eq(1)].index

    accepted = successful.loc[
        successful["contract_row_id"].isin(unambiguous_rows)
        & (~successful["has_parenthetical"] | successful["context_confirmed"])
    ].copy()

    accepted = accepted.sort_values(
        "match_score",
        ascending=False,
        kind="stable",
    ).drop_duplicates("contract_row_id")

    for _, row in accepted.iterrows():

        idx = row["contract_row_id"]

        result.loc[idx, "player_id"] = row["player_id"]

        result.loc[idx, "match_method"] = "source_name_variant"

        result.loc[idx, "original_match_method"] = row["match_method"]

        result.loc[idx, "match_score"] = row["match_score"]

        result.loc[idx, "matched_alternative_name"] = row["alternative_name"]

        result.loc[idx, "resolution_reason"] = None

    logger.info(
        "Resolved %s additional CourtSketch records " "using source name variants.",
        len(accepted),
    )

    return result


def resolve_contract_identities(
    contracts: pd.DataFrame,
    player_reference: pd.DataFrame,
    player_team_context: pd.DataFrame,
    crosswalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resolve CourtSketch contract player identities."""

    resolved = pm.resolve_player_entities(
        records=contracts,
        player_reference=player_reference,
        player_team_context=player_team_context,
        source_system="courtsketch",
        source_id_col="courtsketch_player_id",
        name_col="player_name",
        team_col="team",
        season_col="matching_season",
        crosswalk=crosswalk,
    )

    # Explicit birth-date qualifiers cannot be verified
    # using the current canonical identity reference.
    # Prevent automatic attribution based on name alone.
    birth_date_mask = (
        resolved["player_name"]
        .fillna("")
        .str.contains(
            BIRTH_DATE_PATTERN,
            regex=True,
        )
    )

    resolved.loc[
        birth_date_mask,
        "player_id",
    ] = pd.NA

    resolved.loc[
        birth_date_mask,
        "match_method",
    ] = "unresolved"

    resolved.loc[
        birth_date_mask,
        "original_match_method",
    ] = None

    resolved.loc[
        birth_date_mask,
        "match_score",
    ] = float("nan")

    resolved.loc[
        birth_date_mask,
        "resolution_reason",
    ] = (
        "Explicit birth-date qualifier requires " "additional identity evidence"
    )

    return resolve_contract_name_variants(
        resolved_contracts=resolved,
        player_reference=player_reference,
        player_team_context=player_team_context,
        crosswalk=crosswalk,
    )


def load_identity_references(
    identity_dir: str | Path = DEFAULT_IDENTITY_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously generated canonical references."""

    identity_dir = Path(identity_dir)

    player_reference = pd.read_parquet(identity_dir / "nba_player_reference.parquet")

    player_team_context = pd.read_parquet(
        identity_dir / "nba_player_team_context.parquet"
    )

    return player_reference, player_team_context


def transform_financial_data(
    hoopshype_data: dict[str, pd.DataFrame],
    courtsketch_data: dict[str, pd.DataFrame],
    player_reference: pd.DataFrame,
    player_team_context: pd.DataFrame,
    crosswalk: pd.DataFrame | None = None,
    team_names: dict[str, str] | None = None,
) -> dict:
    """Transform acquired financial datasets."""

    salaries = normalize_salaries(hoopshype_data["player_salaries"])

    salary_caps = normalize_salary_caps(hoopshype_data["salary_caps"])

    contracts = normalize_contracts(
        courtsketch_data["contracts"],
        team_names=team_names,
    )

    resolved_salaries = resolve_salary_identities(
        salaries=salaries,
        player_reference=player_reference,
        player_team_context=player_team_context,
        crosswalk=crosswalk,
    )

    updated_crosswalk = pm.update_crosswalk(
        resolved_records=resolved_salaries,
        existing_crosswalk=crosswalk,
    )

    resolved_contracts = resolve_contract_identities(
        contracts=contracts,
        player_reference=player_reference,
        player_team_context=player_team_context,
        crosswalk=updated_crosswalk,
    )

    datasets = {
        "player_salaries": resolved_salaries,
        "salary_caps": salary_caps,
        "player_contracts": resolved_contracts,
    }

    expected_counts = {
        "player_salaries": len(hoopshype_data["player_salaries"]),
        "salary_caps": len(hoopshype_data["salary_caps"]),
        "player_contracts": len(courtsketch_data["contracts"]),
    }

    for dataset, expected in expected_counts.items():

        if len(datasets[dataset]) != expected:
            raise ValueError(f"Unexpected population change: {dataset}")

    return {
        "datasets": datasets,
        "crosswalk": updated_crosswalk,
    }


def save_processed_financial_data(
    datasets: dict[str, pd.DataFrame],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    source_snapshots: dict | None = None,
) -> dict:
    """Save processed financial datasets and metadata."""

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {}

    metadata = {
        "transformed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshots": source_snapshots or {},
        "datasets": {},
    }

    for dataset, df in datasets.items():

        path = output_dir / f"{dataset}.parquet"

        temporary_path = output_dir / f".{dataset}.tmp.parquet"

        df.to_parquet(
            temporary_path,
            index=False,
        )

        temporary_path.replace(path)

        paths[dataset] = path

        metadata["datasets"][dataset] = {
            "filename": path.name,
            "row_count": len(df),
            "column_count": len(df.columns),
        }

    metadata_path = output_dir / "metadata.json"

    temporary_metadata = output_dir / ".metadata.tmp.json"

    with temporary_metadata.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    temporary_metadata.replace(metadata_path)

    return {
        "datasets": paths,
        "metadata": metadata_path,
    }


def run_financial_transformation(
    raw_dir: str | Path = ingestion.DEFAULT_RAW_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    identity_dir: str | Path = DEFAULT_IDENTITY_DIR,
    report_dir: str | Path = pm.DEFAULT_REPORT_DIR,
    hoopshype_snapshot: str | Path | None = None,
    courtsketch_snapshot: str | Path | None = None,
    save: bool = True,
) -> dict:
    """Execute the complete financial transformation pipeline."""

    if hoopshype_snapshot is None:
        hoopshype_snapshot = ingestion.resolve_latest_financial_snapshot(
            source_system="hoopshype",
            raw_dir=raw_dir,
        )

    if courtsketch_snapshot is None:
        courtsketch_snapshot = ingestion.resolve_latest_financial_snapshot(
            source_system="courtsketch",
            raw_dir=raw_dir,
        )

    hoopshype_data = ingestion.load_financial_snapshot(
        source_system="hoopshype",
        snapshot_dir=hoopshype_snapshot,
    )

    courtsketch_data = ingestion.load_financial_snapshot(
        source_system="courtsketch",
        snapshot_dir=courtsketch_snapshot,
    )

    player_reference, player_team_context = load_identity_references(
        identity_dir=identity_dir,
    )

    crosswalk = pm.load_crosswalk(Path(identity_dir) / "player_id_crosswalk.parquet")

    result = transform_financial_data(
        hoopshype_data=hoopshype_data,
        courtsketch_data=courtsketch_data,
        player_reference=player_reference,
        player_team_context=player_team_context,
        crosswalk=crosswalk,
    )

    source_snapshots = {
        "hoopshype": str(hoopshype_snapshot),
        "courtsketch": str(courtsketch_snapshot),
    }

    result["source_snapshots"] = source_snapshots

    if save:

        result["output_paths"] = save_processed_financial_data(
            datasets=result["datasets"],
            output_dir=output_dir,
            source_snapshots=source_snapshots,
        )

        result["resolution_paths"] = {}

        for source_system, dataset in {
            "hoopshype": "player_salaries",
            "courtsketch": "player_contracts",
        }.items():

            result["resolution_paths"][source_system] = pm.save_resolution_outputs(
                resolved_records=result["datasets"][dataset],
                crosswalk=result["crosswalk"],
                source_system=source_system,
                output_dir=identity_dir,
                report_dir=report_dir,
            )

    return result
