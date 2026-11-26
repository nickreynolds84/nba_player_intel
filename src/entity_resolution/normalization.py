"""Normalization utilities for player entity resolution."""

import re
import unicodedata

import pandas as pd

TEAM_ALIASES = {
    "la clippers": "los angeles clippers",
}


def normalize_player_name(value: object) -> str | None:
    """Normalize a player name while preserving generational suffixes."""

    if pd.isna(value):
        return None

    value = unicodedata.normalize("NFKD", str(value))

    value = "".join(char for char in value if not unicodedata.combining(char))

    value = value.casefold().strip()

    value = re.sub(r"[.'’\-]", "", value)
    value = re.sub(r"\s+", " ", value)

    return value if value else None


def normalize_matching_name(value: object) -> str:
    """Normalize a name for contextual fuzzy matching.

    Generational suffixes are removed only for secondary
    matching supported by contextual evidence.
    """

    value = normalize_player_name(value)

    if value is None:
        return ""

    value = re.sub(
        r"\s+(jr|sr|ii|iii|iv|v)$",
        "",
        value,
    )

    return value.strip()


def normalize_team_name(value: object) -> str:
    """Normalize team names across data providers."""

    value = normalize_player_name(value)

    if value is None:
        return ""

    return TEAM_ALIASES.get(value, value)
