"""Regression tests for NBA financial data processing."""

import json

import pandas as pd
import pytest

from src.ingestion import financial_data as fi
from src.transformation import financial_data as ft
from src.entity_resolution import player_matcher as pm
from src.entity_resolution.normalization import (
    normalize_player_name,
    normalize_matching_name,
    normalize_team_name,
)


@pytest.fixture
def raw_salaries():

    return pd.DataFrame(
        {
            "season_year": [2024, 2024],
            "season": ["2024-25", "2024-25"],
            "player_id": [1001.0, None],
            "player": ["Test Player", "Unknown Player"],
            "salary": [10_000_000, 2_000_000],
            "team_id": [1, 2],
            "team": ["Boston Celtics", "Miami Heat"],
        }
    )


@pytest.fixture
def raw_salary_caps():

    return pd.DataFrame(
        {
            "season_year": [2024],
            "season": ["2024-25"],
            "salary_cap": [140_588_000],
        }
    )


@pytest.fixture
def raw_contracts():

    return pd.DataFrame(
        {
            "date": [
                "2024-07-06",
                "2024-07-07",
                "2024-07-08",
                "2024-07-09",
                "2024-06-25",
                "2024-07-10",
            ],
            "kind": ["contract"] * 6,
            "type": [
                "Signing",
                "Extension",
                "Two-way",
                "Re-signing",
                "Signing",
                "Signing",
            ],
            "teams": ["BOS"] * 6,
            "player_or_assets": ["Test Player"] * 6,
            "years": [4, 3, 2, 2, 1, 2],
            "total_value_musd": [
                40.0,
                30.0,
                2.0,
                20.0,
                None,
                10.0,
            ],
            "aav_musd": [
                10.0,
                10.0,
                1.0,
                10.0,
                None,
                5.0,
            ],
            "terms_stated": [
                "full",
                "full",
                "full",
                "full",
                "years_only",
                "full",
            ],
            "option": [
                "player",
                None,
                None,
                None,
                None,
                None,
            ],
            "flags": [
                "option",
                "non-guaranteed",
                None,
                "duplicate report",
                None,
                "not a new contract",
            ],
            "notes": [None] * 6,
        }
    )


@pytest.fixture
def team_names():

    return {
        "BOS": "Boston Celtics",
        "MIA": "Miami Heat",
    }


@pytest.fixture
def identity_references():

    reference = pd.DataFrame(
        {
            "player_id": pd.Series(
                [101, 102],
                dtype="Int64",
            ),
            "player_name": [
                "Test Player",
                "Other Player",
            ],
        }
    )

    reference["normalized_player_name"] = reference["player_name"].map(
        normalize_player_name
    )

    reference["matching_name"] = reference["player_name"].map(normalize_matching_name)

    context = reference.copy()

    context["season"] = 2025

    context["team_id"] = [
        1610612738,
        1610612738,
    ]

    context["team_name"] = "Boston Celtics"

    context["normalized_team"] = context["team_name"].map(normalize_team_name)

    return reference, context


def test_salary_normalization(raw_salaries):

    result = ft.normalize_salaries(raw_salaries)

    assert len(result) == len(raw_salaries)

    assert result["season"].tolist() == [
        2025,
        2025,
    ]

    assert result["salary"].tolist() == [
        10_000_000,
        2_000_000,
    ]

    assert result.loc[0, "salary_source_player_id"] == 1001

    assert pd.isna(result.loc[1, "salary_source_player_id"])

    assert str(result["salary_source_player_id"].dtype) == "Int64"

    assert "player_id" not in result.columns


def test_salary_cap_normalization(
    raw_salary_caps,
):

    result = ft.normalize_salary_caps(raw_salary_caps)

    assert result.loc[0, "season"] == 2025

    assert result.loc[0, "salary_cap"] == 140_588_000


def test_contract_normalization(
    raw_contracts,
    team_names,
):

    result = ft.normalize_contracts(
        raw_contracts,
        team_names=team_names,
    )

    assert len(result) == len(raw_contracts)

    assert result.loc[0, "reported_contract_value"] == 40_000_000

    assert result.loc[0, "reported_aav"] == 10_000_000

    assert result.loc[0, "calculated_aav"] == 10_000_000

    assert result.loc[0, "team"] == "Boston Celtics"

    assert result.loc[0, "matching_season"] == 2025

    assert result.loc[4, "matching_season"] == 2024

    assert result["courtsketch_player_id"].isna().all()

    assert "contract_start_season" not in result


def test_market_candidate_classification(
    raw_contracts,
    team_names,
):

    result = ft.normalize_contracts(
        raw_contracts,
        team_names=team_names,
    )

    assert result["market_valuation_candidate"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]

    assert result["has_complete_terms"].sum() == 5

    assert bool(result.loc[0, "has_option"])

    assert bool(result.loc[1, "is_non_guaranteed"])

    assert bool(result.loc[3, "is_duplicate_report"])

    assert bool(result.loc[5, "is_not_new_contract"])


def test_unknown_contract_team_rejected(
    raw_contracts,
    team_names,
):

    raw_contracts.loc[0, "teams"] = "UNKNOWN"

    with pytest.raises(
        ValueError,
        match="Unmapped contract teams",
    ):

        ft.normalize_contracts(
            raw_contracts,
            team_names=team_names,
        )


def test_required_schema_validation(
    raw_salaries,
):

    invalid = raw_salaries.drop(columns=["salary"])

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):

        fi.validate_financial_dataset(
            invalid,
            "player_salaries",
        )


def test_financial_snapshot_roundtrip(
    tmp_path,
    raw_salaries,
    raw_salary_caps,
):

    result = fi.save_financial_snapshot(
        source_system="hoopshype",
        datasets={
            "player_salaries": raw_salaries,
            "salary_caps": raw_salary_caps,
        },
        source_locations={
            "player_salaries": "test_salaries.csv",
            "salary_caps": "test_caps.csv",
        },
        acquisition_method="synthetic_test",
        raw_dir=tmp_path,
    )

    latest = fi.resolve_latest_financial_snapshot(
        source_system="hoopshype",
        raw_dir=tmp_path,
    )

    assert latest == result["snapshot_dir"]

    loaded = fi.load_financial_snapshot(
        source_system="hoopshype",
        raw_dir=tmp_path,
    )

    pd.testing.assert_frame_equal(
        loaded["player_salaries"],
        raw_salaries,
    )

    pd.testing.assert_frame_equal(
        loaded["salary_caps"],
        raw_salary_caps,
    )

    metadata = json.loads(result["metadata"].read_text())

    assert metadata["source_system"] == "hoopshype"

    assert metadata["datasets"]["player_salaries"]["row_count"] == 2


def test_incomplete_snapshot_rejected(
    tmp_path,
    raw_salaries,
):

    with pytest.raises(
        ValueError,
        match="requires datasets",
    ):

        fi.save_financial_snapshot(
            source_system="hoopshype",
            datasets={
                "player_salaries": raw_salaries,
            },
            source_locations={
                "player_salaries": "test.csv",
            },
            acquisition_method="synthetic_test",
            raw_dir=tmp_path,
        )


def test_courtsketch_name_extraction():

    assert ft.extract_courtsketch_names("Legal Name / Preferred Name") == [
        "Legal Name",
        "Preferred Name",
    ]

    assert ft.extract_courtsketch_names("Test Player (Middle)") == [
        "Test Player (Middle)",
        "Test Player",
    ]

    assert ft.extract_courtsketch_names("Test Player (b. 1975-02-27)") == []

    assert ft.extract_courtsketch_names(None) == []


def test_contract_name_variant_resolution(
    raw_contracts,
    team_names,
    identity_references,
):

    reference, context = identity_references

    raw = raw_contracts.iloc[[0]].copy()

    raw["player_or_assets"] = "Unrecognized Alias / Test Player"

    contracts = ft.normalize_contracts(
        raw,
        team_names=team_names,
    )

    result = ft.resolve_contract_identities(
        contracts=contracts,
        player_reference=reference,
        player_team_context=context,
    )

    assert len(result) == 1

    assert result.loc[0, "player_id"] == 101

    assert result.loc[0, "match_method"] == "source_name_variant"

    assert result.loc[0, "matched_alternative_name"] == "Test Player"

    assert result.loc[0, "player_name"] == "Unrecognized Alias / Test Player"


def test_conflicting_name_variants_rejected(
    raw_contracts,
    team_names,
    identity_references,
):

    reference, context = identity_references

    raw = raw_contracts.iloc[[0]].copy()

    raw["player_or_assets"] = "Test Player / Other Player"

    contracts = ft.normalize_contracts(
        raw,
        team_names=team_names,
    )

    result = ft.resolve_contract_identities(
        contracts=contracts,
        player_reference=reference,
        player_team_context=context,
    )

    assert pd.isna(result.loc[0, "player_id"])


def test_birth_date_qualifier_protected(
    raw_contracts,
    team_names,
    identity_references,
):

    reference, context = identity_references

    raw = raw_contracts.iloc[[0]].copy()

    raw["player_or_assets"] = "Test Player (b. 1975-02-27)"

    contracts = ft.normalize_contracts(
        raw,
        team_names=team_names,
    )

    result = ft.resolve_contract_identities(
        contracts=contracts,
        player_reference=reference,
        player_team_context=context,
    )

    assert pd.isna(result.loc[0, "player_id"])

    assert result.loc[0, "match_method"] == "unresolved"

    assert "birth-date" in result.loc[0, "resolution_reason"]
