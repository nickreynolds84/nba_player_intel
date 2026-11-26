"""Reusable player entity resolution engine."""

import json
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from src.entity_resolution.normalization import (
    normalize_player_name,
    normalize_matching_name,
    normalize_team_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config/entity_resolution"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/identities"

DEFAULT_REPORT_DIR = PROJECT_ROOT / "data/validation/entity_resolution"

FUZZY_THRESHOLD = 90.0
MIN_SCORE_MARGIN = 10.0

CROSSWALK_COLUMNS = [
    "source_system",
    "source_player_id",
    "source_player_name",
    "player_id",
    "match_method",
    "match_score",
    "matched_at",
    "last_seen_at",
    "mapping_status",
]


def load_matching_config(
    source_system: str,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> tuple[dict, dict]:
    """Load player aliases and source-specific overrides."""

    config_dir = Path(config_dir)

    with open(
        config_dir / "player_aliases.json",
        encoding="utf-8",
    ) as file:
        raw_aliases = json.load(file)

    with open(
        config_dir / "player_id_overrides.json",
        encoding="utf-8",
    ) as file:
        raw_overrides = json.load(file)

    aliases = {
        normalize_player_name(source_name): normalize_player_name(nba_name)
        for source_name, nba_name in raw_aliases.items()
    }

    overrides = {
        str(source_id): details
        for source_id, details in raw_overrides.get(source_system, {}).items()
    }

    return aliases, overrides


def normalize_source_id(value: object) -> str | None:
    """Represent external identifiers consistently."""

    if pd.isna(value):
        return None

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    value = str(value).strip()

    return value if value else None


def load_crosswalk(
    path: str | Path = (DEFAULT_OUTPUT_DIR / "player_id_crosswalk.parquet"),
) -> pd.DataFrame:
    """Load previously established identity relationships."""

    path = Path(path)

    if not path.exists():
        return pd.DataFrame(columns=CROSSWALK_COLUMNS)

    crosswalk = pd.read_parquet(path)

    if crosswalk.duplicated(["source_system", "source_player_id"]).any():
        raise ValueError("Duplicate crosswalk identifiers")

    return crosswalk


def build_exact_name_lookup(
    player_reference: pd.DataFrame,
) -> dict:
    """Build lookup containing only unambiguous names."""

    lookup = {}

    for name, group in player_reference.groupby("normalized_player_name"):
        ids = group["player_id"].dropna().unique()

        if len(ids) == 1:
            lookup[name] = int(ids[0])

    return lookup


def build_context_lookup(
    player_team_context: pd.DataFrame,
) -> dict:
    """Index historical context by season and team."""

    return {
        key: group.copy()
        for key, group in player_team_context.groupby(["season", "normalized_team"])
    }


def find_contextual_fuzzy_match(
    matching_name: str,
    candidates: pd.DataFrame | None,
    threshold: float = FUZZY_THRESHOLD,
    min_margin: float = MIN_SCORE_MARGIN,
) -> tuple[int | None, float | None]:
    """Return a fuzzy match only when sufficiently distinct."""

    if not matching_name or candidates is None or candidates.empty:
        return None, None

    ranked = candidates[["player_id", "matching_name"]].copy()

    ranked = ranked.loc[ranked["matching_name"].fillna("").ne("")].copy()

    if ranked.empty:
        return None, None

    ranked["score"] = ranked["matching_name"].map(
        lambda name: fuzz.ratio(matching_name, name)
    )

    ranked = ranked.sort_values(
        ["score", "player_id"],
        ascending=[False, True],
    ).drop_duplicates("player_id")

    best = ranked.iloc[0]

    second_score = ranked.iloc[1]["score"] if len(ranked) > 1 else 0.0

    if best["score"] >= threshold and best["score"] - second_score >= min_margin:
        return int(best["player_id"]), float(best["score"])

    return None, None


def find_contextual_alias_match(
    normalized_name: str | None,
    candidates: pd.DataFrame | None,
    aliases: dict,
) -> int | None:
    """Resolve an alternate name using contextual evidence."""

    alias = aliases.get(normalized_name)

    if alias is None or candidates is None or candidates.empty:
        return None

    ids = (
        candidates.loc[
            candidates["normalized_player_name"].eq(alias),
            "player_id",
        ]
        .dropna()
        .unique()
    )

    if len(ids) == 1:
        return int(ids[0])

    return None


def resolve_player_entities(
    records: pd.DataFrame,
    player_reference: pd.DataFrame,
    player_team_context: pd.DataFrame,
    source_system: str,
    source_id_col: str,
    name_col: str,
    team_col: str,
    season_col: str = "season",
    crosswalk: pd.DataFrame | None = None,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    fuzzy_threshold: float = FUZZY_THRESHOLD,
    min_score_margin: float = MIN_SCORE_MARGIN,
) -> pd.DataFrame:
    """Resolve external player records to canonical NBA IDs.

    Input records are preserved. Unresolved and conflicting
    records retain a nullable canonical player ID.
    """

    required = {
        source_id_col,
        name_col,
        team_col,
        season_col,
    }

    missing = required.difference(records.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    aliases, overrides = load_matching_config(
        source_system=source_system,
        config_dir=config_dir,
    )

    if crosswalk is None:
        crosswalk = pd.DataFrame(columns=CROSSWALK_COLUMNS)

    source_crosswalk = crosswalk.loc[
        crosswalk["source_system"].eq(source_system)
    ].copy()

    source_crosswalk["source_player_id"] = source_crosswalk["source_player_id"].map(
        normalize_source_id
    )

    if source_crosswalk.duplicated("source_player_id").any():
        raise ValueError("Duplicate source crosswalk IDs")

    accepted_crosswalk = source_crosswalk.loc[
        source_crosswalk["mapping_status"].eq("accepted")
    ]

    existing_mappings = accepted_crosswalk.set_index("source_player_id").to_dict(
        "index"
    )

    conflict_ids = set(
        source_crosswalk.loc[
            source_crosswalk["mapping_status"].eq("conflicting"),
            "source_player_id",
        ].dropna()
    )

    exact_lookup = build_exact_name_lookup(player_reference)

    context_lookup = build_context_lookup(player_team_context)

    valid_player_ids = set(player_reference["player_id"].dropna()).union(
        player_team_context["player_id"].dropna()
    )

    result = records.copy().reset_index(drop=True)

    result["source_system"] = source_system

    result["source_player_id"] = result[source_id_col].map(normalize_source_id)

    result["source_player_name"] = result[name_col]

    result["normalized_player_name"] = result[name_col].map(normalize_player_name)

    result["matching_name"] = result[name_col].map(normalize_matching_name)

    result["normalized_team"] = result[team_col].map(normalize_team_name)

    result["player_id"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    result["match_method"] = "unresolved"
    result["original_match_method"] = None
    result["match_score"] = float("nan")
    result["resolution_reason"] = None

    def assign_match(
        idx,
        player_id,
        method,
        score=None,
        original_method=None,
    ):
        result.at[idx, "player_id"] = player_id
        result.at[idx, "match_method"] = method

        result.at[idx, "original_match_method"] = original_method or method

        if score is not None and pd.notna(score):
            result.at[idx, "match_score"] = float(score)

    for idx, row in result.iterrows():

        source_id = row["source_player_id"]

        if source_id in conflict_ids:
            continue

        candidates = context_lookup.get(
            (
                row[season_col],
                row["normalized_team"],
            )
        )

        # Establish independent name-based evidence.
        candidate_id = exact_lookup.get(row["normalized_player_name"])

        method = "normalized_name_exact"
        score = 100.0

        if candidate_id is None:

            candidate_id, score = find_contextual_fuzzy_match(
                matching_name=row["matching_name"],
                candidates=candidates,
                threshold=fuzzy_threshold,
                min_margin=min_score_margin,
            )

            method = "contextual_fuzzy"

        if candidate_id is None:

            candidate_id = find_contextual_alias_match(
                normalized_name=(row["normalized_player_name"]),
                candidates=candidates,
                aliases=aliases,
            )

            method = "contextual_alias"
            score = None

        # Retrieve previously established relationships.
        override = overrides.get(source_id)
        existing = existing_mappings.get(source_id)

        evidence_ids = set()

        if candidate_id is not None:
            evidence_ids.add(int(candidate_id))

        if override is not None:
            evidence_ids.add(int(override["player_id"]))

        if existing is not None:
            evidence_ids.add(int(existing["player_id"]))

        # Never silently overwrite contradictory identities.
        if len(evidence_ids) > 1:

            if source_id is not None:
                conflict_ids.add(source_id)

            result.at[idx, "match_method"] = "source_id_conflict"

            result.at[idx, "resolution_reason"] = (
                f"Contradictory NBA IDs: " f"{sorted(evidence_ids)}"
            )

            continue

        if evidence_ids.difference(valid_player_ids):

            if source_id is not None:
                conflict_ids.add(source_id)

            result.at[idx, "match_method"] = "source_id_conflict"

            result.at[idx, "resolution_reason"] = (
                "Mapped NBA ID absent from identity references"
            )

            continue

        # Resolution priority.
        if override is not None:

            assign_match(
                idx,
                int(override["player_id"]),
                "confirmed_id_mapping",
            )

        elif existing is not None:

            assign_match(
                idx,
                int(existing["player_id"]),
                "source_id_crosswalk",
                score=existing["match_score"],
                original_method=existing["match_method"],
            )

        elif candidate_id is not None:

            assign_match(
                idx,
                candidate_id,
                method,
                score=score,
            )

    # Detect conflicting mappings across source records.
    pairs = result[["source_player_id", "player_id"]].dropna().drop_duplicates()

    mapping_counts = pairs.groupby("source_player_id")["player_id"].nunique()

    conflict_ids.update(mapping_counts.loc[mapping_counts.gt(1)].index)

    conflict_mask = result["source_player_id"].isin(conflict_ids)

    result.loc[conflict_mask, "player_id"] = pd.NA

    result.loc[conflict_mask, "match_method"] = "source_id_conflict"

    result.loc[conflict_mask, "original_match_method"] = None

    result.loc[conflict_mask, "match_score"] = float("nan")

    result.loc[
        conflict_mask & result["resolution_reason"].isna(),
        "resolution_reason",
    ] = "Conflicting source identifier mapping"

    # Propagate newly established source ID relationships.
    resolved = result.loc[
        result["player_id"].notna() & result["source_player_id"].notna()
    ].copy()

    propagation_lookup = (
        resolved.drop_duplicates("source_player_id")
        .set_index("source_player_id")
        .to_dict("index")
    )

    unresolved_indices = result.index[result["match_method"].eq("unresolved")]

    for idx in unresolved_indices:

        source_id = result.at[idx, "source_player_id"]

        mapping = propagation_lookup.get(source_id)

        if mapping is not None:

            assign_match(
                idx,
                int(mapping["player_id"]),
                "source_id_crosswalk",
                original_method=(mapping["original_match_method"]),
            )

    result.loc[
        result["match_method"].eq("unresolved"),
        "resolution_reason",
    ] = "No sufficiently supported identity match"

    if len(result) != len(records):
        raise RuntimeError("Entity resolution changed input record count")

    result.index = records.index

    return result


def update_crosswalk(
    resolved_records: pd.DataFrame,
    existing_crosswalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Persist identity relationships and conflict status."""

    if existing_crosswalk is None:
        existing_crosswalk = pd.DataFrame(columns=CROSSWALK_COLUMNS)

    crosswalk = existing_crosswalk.copy()

    crosswalk["source_player_id"] = crosswalk["source_player_id"].map(
        normalize_source_id
    )

    keys = ["source_system", "source_player_id"]

    if crosswalk.duplicated(keys).any():
        raise ValueError("Duplicate crosswalk identifiers")

    entries = {
        (row["source_system"], row["source_player_id"]): row.to_dict()
        for _, row in crosswalk.iterrows()
    }

    timestamp = pd.Timestamp.now(tz="UTC")

    records = resolved_records.loc[resolved_records["source_player_id"].notna()]

    for key, group in records.groupby(keys):

        source_system, source_id = key

        existing = entries.get(key)

        ids = group["player_id"].dropna().unique()

        has_conflict = group["match_method"].eq("source_id_conflict").any()

        if existing is not None:

            existing["last_seen_at"] = timestamp

            if (
                has_conflict
                or len(ids) > 1
                or (
                    len(ids) == 1
                    and pd.notna(existing["player_id"])
                    and int(ids[0]) != int(existing["player_id"])
                )
            ):
                existing["mapping_status"] = "conflicting"

            continue

        if has_conflict or len(ids) > 1:

            entries[key] = {
                "source_system": source_system,
                "source_player_id": source_id,
                "source_player_name": (group.iloc[0]["source_player_name"]),
                "player_id": pd.NA,
                "match_method": "source_id_conflict",
                "match_score": float("nan"),
                "matched_at": pd.NaT,
                "last_seen_at": timestamp,
                "mapping_status": "conflicting",
            }

        elif len(ids) == 1:

            matched = group.loc[group["player_id"].notna()].copy()

            direct_matches = matched.loc[
                matched["match_method"].ne("source_id_crosswalk")
            ]

            row = (
                direct_matches.iloc[0] if not direct_matches.empty else matched.iloc[0]
            )

            entries[key] = {
                "source_system": source_system,
                "source_player_id": source_id,
                "source_player_name": (row["source_player_name"]),
                "player_id": int(ids[0]),
                "match_method": (row["original_match_method"]),
                "match_score": row["match_score"],
                "matched_at": timestamp,
                "last_seen_at": timestamp,
                "mapping_status": "accepted",
            }

    updated = pd.DataFrame(
        list(entries.values()),
        columns=CROSSWALK_COLUMNS,
    )

    updated["player_id"] = updated["player_id"].astype("Int64")

    return updated.sort_values(keys).reset_index(drop=True)


def save_resolution_outputs(
    resolved_records: pd.DataFrame,
    crosswalk: pd.DataFrame,
    source_system: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
) -> dict[str, Path]:
    """Save the crosswalk and run-specific exceptions."""

    output_dir = Path(output_dir)
    report_dir = Path(report_dir) / source_system

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H%M%S%fZ")

    crosswalk_path = output_dir / "player_id_crosswalk.parquet"

    temporary_path = crosswalk_path.with_suffix(".tmp.parquet")

    crosswalk.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(crosswalk_path)

    exceptions = resolved_records.loc[resolved_records["player_id"].isna()].copy()

    exception_path = report_dir / f"exceptions_{timestamp}.parquet"

    exceptions.to_parquet(
        exception_path,
        index=False,
    )

    return {
        "crosswalk": crosswalk_path,
        "exceptions": exception_path,
    }
