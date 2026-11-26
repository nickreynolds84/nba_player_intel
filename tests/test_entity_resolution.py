"""Regression tests for player entity resolution."""

import json

import pandas as pd
import pytest

from src.entity_resolution import player_matcher as pm

from src.entity_resolution.normalization import (
    normalize_player_name,
    normalize_matching_name,
    normalize_team_name,
)


@pytest.fixture
def config_dir(tmp_path):

    aliases = {
        "Alexander Smith": "Alex Smith",
    }

    overrides = {
        "test_source": {
            "999": {
                "player_id": 1,
                "source_player_name": "A. Smith",
                "reason": "Confirmed test identity",
            }
        }
    }

    (tmp_path / "player_aliases.json").write_text(
        json.dumps(aliases),
        encoding="utf-8",
    )

    (tmp_path / "player_id_overrides.json").write_text(
        json.dumps(overrides),
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def reference():

    df = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "player_name": [
                "Alex Smith",
                "Jonathan Williams Jr.",
                "José García",
                "Michael Brown",
            ],
        }
    )

    df["normalized_player_name"] = df["player_name"].map(normalize_player_name)

    df["matching_name"] = df["player_name"].map(normalize_matching_name)

    return df


@pytest.fixture
def context(reference):

    df = reference.copy()

    df["season"] = 2027

    df["team"] = [
        "Detroit Pistons",
        "Detroit Pistons",
        "Los Angeles Clippers",
        "Boston Celtics",
    ]

    df["normalized_team"] = df["team"].map(normalize_team_name)

    return df


@pytest.fixture
def resolve(reference, context, config_dir):

    def _resolve(rows, crosswalk=None):

        records = pd.DataFrame(
            rows,
            columns=[
                "external_id",
                "name",
                "team",
                "season",
            ],
        )

        return pm.resolve_player_entities(
            records=records,
            player_reference=reference,
            player_team_context=context,
            source_system="test_source",
            source_id_col="external_id",
            name_col="name",
            team_col="team",
            crosswalk=crosswalk,
            config_dir=config_dir,
        )

    return _resolve


def test_name_normalization():

    assert normalize_player_name("  José García  ") == "jose garcia"

    assert normalize_player_name("J.J. Smith") == "jj smith"

    assert normalize_matching_name("Jonathan Williams Jr.") == "jonathan williams"

    assert normalize_player_name(None) is None


def test_team_normalization():

    assert normalize_team_name("LA Clippers") == "los angeles clippers"


def test_exact_name_match(resolve):

    result = resolve(
        [
            (100, "Alex Smith", "Detroit Pistons", 2027),
        ]
    )

    assert result.iloc[0]["player_id"] == 1

    assert result.iloc[0]["match_method"] == "normalized_name_exact"


def test_contextual_fuzzy_match(resolve):

    result = resolve(
        [
            (
                200,
                "Jonathan Williams",
                "Detroit Pistons",
                2027,
            ),
        ]
    )

    assert result.iloc[0]["player_id"] == 2

    assert result.iloc[0]["match_method"] == "contextual_fuzzy"


def test_fuzzy_wrong_team_rejected(resolve):

    result = resolve(
        [
            (
                200,
                "Jonathan Williams",
                "Boston Celtics",
                2027,
            ),
        ]
    )

    assert pd.isna(result.iloc[0]["player_id"])


def test_fuzzy_wrong_season_rejected(resolve):

    result = resolve(
        [
            (
                200,
                "Jonathan Williams",
                "Detroit Pistons",
                2025,
            ),
        ]
    )

    assert pd.isna(result.iloc[0]["player_id"])


def test_contextual_alias(resolve):

    result = resolve(
        [
            (
                100,
                "Alexander Smith",
                "Detroit Pistons",
                2027,
            ),
        ]
    )

    assert result.iloc[0]["player_id"] == 1

    assert result.iloc[0]["match_method"] == "contextual_alias"


def test_confirmed_identifier_override(resolve):

    result = resolve(
        [
            (999, "A. Smith", "Detroit Pistons", 2027),
        ]
    )

    assert result.iloc[0]["player_id"] == 1

    assert result.iloc[0]["match_method"] == "confirmed_id_mapping"


def test_missing_source_identifier(resolve):

    result = resolve(
        [
            (None, "Alex Smith", "Detroit Pistons", 2027),
        ]
    )

    assert result.iloc[0]["player_id"] == 1

    assert pd.isna(result.iloc[0]["source_player_id"])


def test_unresolved_identity(resolve):

    result = resolve(
        [
            (500, "Unknown Player", "Boston Celtics", 2027),
        ]
    )

    assert pd.isna(result.iloc[0]["player_id"])

    assert result.iloc[0]["match_method"] == "unresolved"


def test_ambiguous_fuzzy_match():

    candidates = pd.DataFrame(
        {
            "player_id": [1, 2],
            "matching_name": [
                "alex smith",
                "alex smith",
            ],
        }
    )

    player_id, score = pm.find_contextual_fuzzy_match(
        "alex smith",
        candidates,
    )

    assert player_id is None
    assert score is None


def test_source_id_propagation(resolve):

    result = resolve(
        [
            (100, "Alex Smith", "Detroit Pistons", 2027),
            (100, "A. Smith", "Detroit Pistons", 2026),
        ]
    )

    assert result["player_id"].tolist() == [1, 1]

    assert result.iloc[1]["match_method"] == "source_id_crosswalk"


def test_crosswalk_reuse(resolve):

    first = resolve(
        [
            (100, "Alex Smith", "Detroit Pistons", 2027),
        ]
    )

    crosswalk = pm.update_crosswalk(first)

    second = resolve(
        [
            (100, "A. Smith", "Boston Celtics", 2027),
        ],
        crosswalk=crosswalk,
    )

    assert second.iloc[0]["player_id"] == 1

    assert second.iloc[0]["match_method"] == "source_id_crosswalk"

    assert second.iloc[0]["original_match_method"] == "normalized_name_exact"


def test_conflicting_source_identifier(resolve):

    result = resolve(
        [
            (100, "Alex Smith", "Detroit Pistons", 2027),
            (100, "Michael Brown", "Boston Celtics", 2027),
        ]
    )

    assert result["player_id"].isna().all()

    assert result["match_method"].eq("source_id_conflict").all()


def test_existing_crosswalk_conflict(resolve):

    first = resolve(
        [
            (100, "Alex Smith", "Detroit Pistons", 2027),
        ]
    )

    crosswalk = pm.update_crosswalk(first)

    second = resolve(
        [
            (100, "Michael Brown", "Boston Celtics", 2027),
        ],
        crosswalk=crosswalk,
    )

    assert pd.isna(second.iloc[0]["player_id"])

    updated = pm.update_crosswalk(
        second,
        existing_crosswalk=crosswalk,
    )

    assert updated.iloc[0]["mapping_status"] == "conflicting"

    third = resolve(
        [
            (100, "Alex Smith", "Detroit Pistons", 2027),
        ],
        crosswalk=updated,
    )

    assert pd.isna(third.iloc[0]["player_id"])


def test_input_population_preserved(resolve):

    rows = [
        (100, "Alex Smith", "Detroit Pistons", 2027),
        (200, "Unknown Player", "Boston Celtics", 2027),
        (100, "Alex Smith", "Detroit Pistons", 2026),
    ]

    result = resolve(rows)

    assert len(result) == len(rows)

    assert result["external_id"].tolist() == [100, 200, 100]


def test_crosswalk_persistence(resolve, tmp_path):

    result = resolve(
        [
            (100, "Alex Smith", "Detroit Pistons", 2027),
            (200, "Unknown Player", "Boston Celtics", 2027),
        ]
    )

    crosswalk = pm.update_crosswalk(result)

    paths = pm.save_resolution_outputs(
        resolved_records=result,
        crosswalk=crosswalk,
        source_system="test_source",
        output_dir=tmp_path / "identities",
        report_dir=tmp_path / "reports",
    )

    loaded = pm.load_crosswalk(paths["crosswalk"])

    pd.testing.assert_frame_equal(
        crosswalk,
        loaded,
    )

    exceptions = pd.read_parquet(paths["exceptions"])

    assert len(exceptions) == 1

    assert exceptions.iloc[0]["source_player_name"] == "Unknown Player"
