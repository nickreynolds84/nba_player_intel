from __future__ import annotations

import json
import logging

from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.season_data import (
    normalize_label,
    validate_dataset,
    atomic_write,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "player_stats"

logger = logging.getLogger(__name__)


PLAYER_KEY = [
    "player_id",
    "season",
    "season_type",
]


# Only metrics empirically validated against
# overlapping player master observations.
DEFENSE_RECOVERY_MAPPING = {
    "def_rating": "def_def_rating",
    "dreb_pct": "def_dreb_pct",
    "pct_dreb": "def_pct_dreb",
    "pct_stl": "def_pct_stl",
    "pct_blk": "def_pct_blk",
    "opp_pts_off_tov": "def_opp_pts_off_tov",
    "opp_pts_2nd_chance": "def_opp_pts_2_nd_chance",
    "opp_pts_fb": "def_opp_pts_fb",
    "opp_pts_paint": "def_opp_pts_paint",
    "def_ws": "def_def_ws",
    "def_ws_raw": "def_def_ws_raw",
}


HUSTLE_METRICS = [
    "g",
    "min",
    "team_id",
    "contested_shots",
    "contested_shots_2_pt",
    "contested_shots_3_pt",
    "deflections",
    "charges_drawn",
    "screen_assists",
    "screen_ast_pts",
    "off_loose_balls_recovered",
    "def_loose_balls_recovered",
    "loose_balls_recovered",
    "pct_loose_balls_recovered_off",
    "pct_loose_balls_recovered_def",
    "off_boxouts",
    "def_boxouts",
    "box_out_player_team_rebs",
    "box_out_player_rebs",
    "box_outs",
    "pct_box_outs_off",
    "pct_box_outs_def",
    "pct_box_outs_team_reb",
    "pct_box_outs_reb",
]


HUSTLE_MAPPING = {col: f"hustle_{col}" for col in HUSTLE_METRICS}

HUSTLE_MAPPING["g"] = "hustle_gp"
HUSTLE_MAPPING["min"] = "hustle_minutes_total"


# Initial data-quality guardrails.
# These are configurable validation thresholds,
# not statistical modeling parameters.
MIN_HUSTLE_COVERAGE = 0.95
MIN_HUSTLE_TEAM_AGREEMENT = 0.95

# Known source defect confirmed empirically.
KNOWN_HUSTLE_QUARANTINE = {2016: "Mislabeled duplicate of 2017 hustle data"}


def utc_now():
    return datetime.now(timezone.utc)


def historical_path(dataset, season):

    filename_templates = {
        "player_master": "player_master_{season}.parquet",
        "player_hustle": "player_hustle_{season}.parquet",
        "player_stats": "player_season_stats_{season}.parquet",
    }

    if dataset not in filename_templates:
        raise ValueError(f"Unknown dataset: {dataset}")

    return RAW_DIR / dataset / filename_templates[dataset].format(season=season)


def read_source_metadata(path):

    metadata_path = path.with_suffix(".json")

    if not metadata_path.exists():
        return {}

    return json.loads(metadata_path.read_text())


def source_reference(path):

    if path is None or not path.exists():
        return None

    metadata = read_source_metadata(path)

    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": metadata.get("sha256"),
        "source_updated_at": metadata.get("source_updated_at"),
        "retrieved_at": metadata.get("retrieved_at"),
    }


def normalize_keys(df):

    df = df.copy()

    player_ids = pd.to_numeric(
        df["player_id"],
        errors="raise",
    )

    if not (player_ids.notna() & player_ids.gt(0) & player_ids.mod(1).eq(0)).all():
        raise ValueError("Invalid player identifiers")

    df["player_id"] = player_ids.astype("int64")

    df["season"] = pd.to_numeric(
        df["season"],
        errors="raise",
    ).astype("int64")

    df["season_type"] = normalize_label(df["season_type"]).str.replace(
        " ", "-", regex=False
    )

    if "per_mode" in df.columns:

        df["per_mode"] = normalize_label(df["per_mode"])

    return df


def regular_season(df):

    df = normalize_keys(df)

    return df.loc[df["season_type"] == "regular-season"].copy()


def validate_unique_keys(df, label):

    if df[PLAYER_KEY].isna().any().any():

        raise ValueError(f"{label}: Missing primary keys")

    if df.duplicated(PLAYER_KEY).any():

        raise ValueError(f"{label}: Duplicate player-season keys")


def load_master(path, season):

    raw = pd.read_parquet(path)

    validate_dataset(
        raw,
        "player_master",
        season,
    )

    master = regular_season(raw)

    validate_unique_keys(
        master,
        "Player master",
    )

    return master


def load_hustle(path, season):

    raw = pd.read_parquet(path)

    validate_dataset(
        raw,
        "player_hustle",
        season,
    )

    hustle = regular_season(raw)

    validate_unique_keys(
        hustle,
        "Player hustle",
    )

    return hustle


def jaccard_similarity(a, b):

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


def assess_hustle_quality(master, hustle):

    master_ids = set(master["player_id"])
    hustle_ids = set(hustle["player_id"])

    comparison = master[PLAYER_KEY + ["gp", "min", "team_id"]].merge(
        hustle[PLAYER_KEY + ["g", "min", "team_id"]],
        on=PLAYER_KEY,
        how="inner",
        suffixes=("_master", "_hustle"),
        validate="one_to_one",
    )

    matched = len(comparison)

    coverage = matched / len(master) if len(master) else 0.0

    if matched:

        gp_diff = (comparison["gp"] - comparison["g"]).abs()

        min_diff = (comparison["min_master"] - comparison["min_hustle"]).abs()

        team_agreement = float(
            comparison["team_id_master"]
            .eq(comparison["team_id_hustle"])
            .fillna(False)
            .mean()
        )

        mean_gp_diff = float(gp_diff.mean())
        median_gp_diff = float(gp_diff.median())

        mean_min_diff = float(min_diff.mean())
        median_min_diff = float(min_diff.median())

    else:

        team_agreement = 0.0
        mean_gp_diff = None
        median_gp_diff = None
        mean_min_diff = None
        median_min_diff = None

    return {
        "master_players": len(master),
        "hustle_players": len(hustle),
        "matched_players": matched,
        "missing_players": len(master_ids - hustle_ids),
        "extra_players": len(hustle_ids - master_ids),
        "coverage": coverage,
        "jaccard_similarity": jaccard_similarity(
            master_ids,
            hustle_ids,
        ),
        "team_agreement": team_agreement,
        "mean_gp_difference": mean_gp_diff,
        "median_gp_difference": median_gp_diff,
        "mean_min_difference": mean_min_diff,
        "median_min_difference": median_min_diff,
    }


def canonical_hustle_comparison(hustle):

    return (
        hustle.drop(columns=["season"])
        .sort_values("player_id")
        .sort_index(axis=1)
        .reset_index(drop=True)
    )


def audit_historical_hustle(masters, hustles):

    quarantine = dict(KNOWN_HUSTLE_QUARANTINE)

    quality = {}

    master_populations = {
        season: set(df["player_id"]) for season, df in masters.items()
    }

    for season, hustle in hustles.items():

        report = assess_hustle_quality(
            masters[season],
            hustle,
        )

        hustle_ids = set(hustle["player_id"])

        similarities = {
            candidate: jaccard_similarity(
                hustle_ids,
                population,
            )
            for candidate, population in master_populations.items()
        }

        best_season = max(
            similarities,
            key=similarities.get,
        )

        report["best_matching_season"] = best_season

        report["best_similarity"] = similarities[best_season]

        quality[season] = report

        if best_season != season:

            quarantine.setdefault(
                season,
                f"Population aligns best with " f"season {best_season}",
            )

        if report["coverage"] < MIN_HUSTLE_COVERAGE:

            quarantine.setdefault(
                season,
                "Insufficient player coverage",
            )

        if report["team_agreement"] < MIN_HUSTLE_TEAM_AGREEMENT:

            quarantine.setdefault(
                season,
                "Insufficient team agreement",
            )

    # Detect identical source data after
    # removing the season label.
    comparable = {
        season: canonical_hustle_comparison(df) for season, df in hustles.items()
    }

    duplicate_pairs = []

    for a, b in combinations(
        sorted(comparable),
        2,
    ):

        if not comparable[a].equals(comparable[b]):
            continue

        duplicate_pairs.append([a, b])

        # Retain a season only when its population
        # clearly aligns better with its own master.
        similarity_a = quality[a]["jaccard_similarity"]

        similarity_b = quality[b]["jaccard_similarity"]

        if similarity_a > similarity_b:

            quarantine.setdefault(
                b,
                f"Duplicate of hustle season {a}",
            )

        elif similarity_b > similarity_a:

            quarantine.setdefault(
                a,
                f"Duplicate of hustle season {b}",
            )

        else:

            quarantine.setdefault(
                a,
                f"Ambiguous duplicate of {b}",
            )

            quarantine.setdefault(
                b,
                f"Ambiguous duplicate of {a}",
            )

    return quarantine, quality, duplicate_pairs


def recover_defensive_metrics(
    master,
    original_path=None,
):

    result = master.copy()

    recovered_count = pd.Series(
        0,
        index=result.index,
        dtype="int64",
    )

    report = {
        "recovered_values": 0,
        "recovered_players": 0,
        "metrics": {},
    }

    missing_metrics = [
        target
        for target in DEFENSE_RECOVERY_MAPPING.values()
        if target not in result.columns or result[target].isna().any()
    ]

    if missing_metrics and original_path is not None:
        if not original_path.exists():
            raise FileNotFoundError(
                f"Defensive recovery source not found: " f"{original_path}"
            )
    if missing_metrics and original_path is not None and original_path.exists():

        original = pd.read_parquet(original_path)

        original = regular_season(original)

        defense = original.loc[
            normalize_label(original["measure_type"]).eq("defense")
            & original["per_mode"].eq("totals")
        ].copy()

        if not defense.empty:

            validate_unique_keys(
                defense,
                "Original defense totals",
            )

            defense = defense.set_index(PLAYER_KEY)

            target_keys = pd.MultiIndex.from_frame(result[PLAYER_KEY])

            for source_col, target_col in DEFENSE_RECOVERY_MAPPING.items():

                if source_col not in defense.columns:
                    continue

                if target_col not in result.columns:
                    result[target_col] = np.nan

                fallback = pd.Series(
                    defense[source_col].reindex(target_keys).to_numpy(),
                    index=result.index,
                )

                primary = result[target_col]

                comparable = primary.notna() & fallback.notna()

                matches = np.isclose(
                    primary.loc[comparable].astype(float),
                    fallback.loc[comparable].astype(float),
                    rtol=1e-6,
                    atol=1e-6,
                )

                mismatches = int((~matches).sum())

                if mismatches:

                    logger.warning(
                        "Recovery skipped for %s: " "%s overlapping mismatches",
                        target_col,
                        mismatches,
                    )

                    report["metrics"][target_col] = {
                        "status": "conflict",
                        "mismatches": mismatches,
                        "recovered_values": 0,
                    }

                    continue

                fill_mask = primary.isna() & fallback.notna()

                result.loc[fill_mask, target_col] = fallback.loc[fill_mask]

                recovered_count += fill_mask.astype("int64")

                report["metrics"][target_col] = {
                    "status": "validated_mapping",
                    "comparable_values": int(comparable.sum()),
                    "mismatches": mismatches,
                    "recovered_values": int(fill_mask.sum()),
                }

    result["defense_recovered_metric_count"] = recovered_count

    result["defense_recovered"] = recovered_count.gt(0)

    result["defense_recovery_source"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    result.loc[
        result["defense_recovered"],
        "defense_recovery_source",
    ] = "original_player_stats"

    report["recovered_values"] = int(recovered_count.sum())

    report["recovered_players"] = int(recovered_count.gt(0).sum())

    return result, report


def integrate_hustle(
    master,
    hustle,
    quarantine_reason=None,
):

    result = master.copy()

    quality = assess_hustle_quality(
        master,
        hustle,
    )

    if quarantine_reason is None:

        if quality["coverage"] < MIN_HUSTLE_COVERAGE:

            quarantine_reason = "Insufficient player coverage"

        elif quality["team_agreement"] < MIN_HUSTLE_TEAM_AGREEMENT:

            quarantine_reason = "Insufficient team agreement"

    if quarantine_reason is not None:

        for target in HUSTLE_MAPPING.values():

            result[target] = np.nan

        result["hustle_available"] = False

        result["hustle_status"] = "quarantined"

        result["hustle_quarantine_reason"] = quarantine_reason

        logger.warning(
            "Hustle data quarantined: %s",
            quarantine_reason,
        )

    else:

        # Reindex preserves the expected output
        # schema if optional metrics are missing.
        subset = hustle.reindex(columns=PLAYER_KEY + HUSTLE_METRICS).rename(
            columns=HUSTLE_MAPPING
        )

        result = result.merge(
            subset,
            on=PLAYER_KEY,
            how="left",
            validate="one_to_one",
            indicator="_hustle_merge",
        )

        result["hustle_available"] = result["_hustle_merge"].eq("both")

        result["hustle_status"] = np.where(
            result["hustle_available"],
            "available",
            "missing_player",
        )

        result["hustle_quarantine_reason"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="string",
        )

        result = result.drop(columns="_hustle_merge")

    quality["quarantined"] = quarantine_reason is not None

    quality["quarantine_reason"] = quarantine_reason

    quality["integrated_players"] = int(result["hustle_available"].sum())

    return result, quality


def transform_player_season(
    master,
    hustle,
    original_path=None,
    quarantine_reason=None,
):

    validate_unique_keys(
        master,
        "Master input",
    )

    validate_unique_keys(
        hustle,
        "Hustle input",
    )

    expected_rows = len(master)

    result, recovery_report = recover_defensive_metrics(
        master,
        original_path,
    )

    result, hustle_report = integrate_hustle(
        result,
        hustle,
        quarantine_reason,
    )

    validate_unique_keys(
        result,
        "Canonical output",
    )

    if len(result) != expected_rows:

        raise ValueError("Transformation changed " "the master player population")

    result = result.sort_values(PLAYER_KEY).reset_index(drop=True)

    report = {
        "rows": len(result),
        "columns": len(result.columns),
        "defense_recovery": recovery_report,
        "hustle_quality": hustle_report,
    }

    return result, report


def save_dataset(df, path, metadata):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_write(
        path,
        df.to_parquet(index=False),
    )

    atomic_write(
        path.with_suffix(".json"),
        json.dumps(
            metadata,
            indent=2,
            allow_nan=False,
        ).encode("utf-8"),
    )

    logger.info(
        "Saved %s | %s rows",
        path,
        len(df),
    )


def build_historical_dataset(
    start_season=2016,
    end_season=2026,
    save=True,
):

    seasons = list(range(start_season, end_season + 1))

    masters = {}
    hustles = {}

    for season in seasons:

        masters[season] = load_master(
            historical_path(
                "player_master",
                season,
            ),
            season,
        )

        hustles[season] = load_hustle(
            historical_path(
                "player_hustle",
                season,
            ),
            season,
        )

    (
        quarantine,
        quality,
        duplicate_pairs,
    ) = audit_historical_hustle(
        masters,
        hustles,
    )

    frames = []
    season_reports = {}

    for season in seasons:

        original_path = historical_path(
            "player_stats",
            season,
        )

        result, report = transform_player_season(
            master=masters[season],
            hustle=hustles[season],
            original_path=original_path,
            quarantine_reason=quarantine.get(season),
        )

        report["sources"] = {
            "player_master": source_reference(
                historical_path(
                    "player_master",
                    season,
                )
            ),
            "player_hustle": source_reference(
                historical_path(
                    "player_hustle",
                    season,
                )
            ),
            "original_player_stats": (source_reference(original_path)),
        }

        frames.append(result)

        season_reports[season] = report

        logger.info(
            "Processed %s | %s players | "
            "%s recovered defensive values | "
            "%s hustle players",
            season,
            len(result),
            report["defense_recovery"]["recovered_values"],
            report["hustle_quality"]["integrated_players"],
        )

    historical = pd.concat(
        frames,
        ignore_index=True,
    )

    validate_unique_keys(
        historical,
        "Historical canonical dataset",
    )

    historical = historical.sort_values(PLAYER_KEY).reset_index(drop=True)

    metadata = {
        "built_at": utc_now().isoformat(),
        "start_season": start_season,
        "end_season": end_season,
        "rows": len(historical),
        "columns": len(historical.columns),
        "quarantined_hustle_seasons": quarantine,
        "duplicate_hustle_pairs": duplicate_pairs,
        "historical_hustle_audit": quality,
        "season_reports": season_reports,
    }

    if save:

        output_path = (
            PROCESSED_DIR / f"player_seasons_" f"{start_season}_{end_season}.parquet"
        )

        save_dataset(
            historical,
            output_path,
            metadata,
        )

    return historical, metadata


def latest_snapshot(dataset, season):

    directory = RAW_DIR / "snapshots" / dataset / str(season)

    candidates = sorted(
        path
        for path in directory.glob("*.parquet")
        if path.with_suffix(".json").exists()
    )

    if not candidates:

        raise FileNotFoundError(f"No complete snapshots in {directory}")

    return candidates[-1]


def build_current_dataset(
    season,
    master_path=None,
    hustle_path=None,
    save=True,
):

    master_path = (
        Path(master_path).resolve()
        if master_path is not None
        else latest_snapshot(
            "player_master",
            season,
        )
    )

    hustle_path = (
        Path(hustle_path).resolve()
        if hustle_path is not None
        else latest_snapshot(
            "player_hustle",
            season,
        )
    )

    master = load_master(
        master_path,
        season,
    )

    hustle = load_hustle(
        hustle_path,
        season,
    )

    # Do not recover current-season measurements
    # from independently timed original snapshots.
    current, report = transform_player_season(
        master=master,
        hustle=hustle,
        original_path=None,
    )

    current["master_snapshot"] = master_path.stem

    current["hustle_snapshot"] = hustle_path.stem

    metadata = {
        "built_at": utc_now().isoformat(),
        "season": season,
        "snapshot": True,
        "games_through_alignment_verified": False,
        "sources": {
            "player_master": source_reference(master_path),
            "player_hustle": source_reference(hustle_path),
        },
        **report,
    }

    if save:

        timestamp = utc_now().strftime("%Y-%m-%dT%H%M%S%fZ")

        output_path = PROCESSED_DIR / "snapshots" / str(season) / f"{timestamp}.parquet"

        save_dataset(
            current,
            output_path,
            metadata,
        )

    return current, metadata
