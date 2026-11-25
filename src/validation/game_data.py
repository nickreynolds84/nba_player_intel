"""Read-only cross-source game data audit. Pass a mapping of dataset DataFrames."""

import pandas as pd
from src.ingestion import game_data as gd

STATS = dict(
    zip(
        "points field_goals_made field_goals_attempted three_pointers_made three_pointers_attempted free_throws_made free_throws_attempted rebounds_offensive rebounds_defensive rebounds_total assists steals blocks turnovers fouls_personal plus_minus_points".split(),
        "pts fgm fga fg3m fg3a ftm fta oreb dreb reb ast stl blk tov pf plus_minus".split(),
    )
)
KEY = ["game_id", "player_id"]
TEAM = ["game_id", "team_id"]
HOME = [f"home_player_{i}" for i in range(1, 6)]
AWAY = [f"away_player_{i}" for i in range(1, 6)]


def audit_season(datasets, season, team_seconds_tolerance=5):
    """Return summary metrics and diagnostic DataFrames; never mutate inputs."""
    d = {k: v.copy() for k, v in datasets.items()}
    for frame in d.values():
        frame.columns = frame.columns.str.lower()
        frame["game_id"] = gd.normalize_game_ids(frame["game_id"])
    b = d["player_boxscores"].rename(columns={"person_id": "player_id"})
    g = d["player_game_logs"]
    t = d["team_boxscores"]
    s = d["schedules"]
    l = d["lineups"]
    m = {"season": season}
    details = {}

    def record(name, frame):
        details[name] = frame
        m[name] = len(frame)

    tp = gd.identify_team_boxscore_placeholders(t)
    sp = gd.identify_schedule_placeholders(s)
    m.update(team_placeholders=int(tp.sum()), schedule_placeholders=int(sp.sum()))
    t = t.loc[~tp].copy()
    for frame in (b, g, t):
        for col in ("player_id", "team_id"):
            if col in frame:
                frame[col] = gd.normalize_integer_ids(frame[col], col)
    sets = {
        k: set(v.game_id)
        for k, v in [("box", b), ("log", g), ("team", t), ("lineup", l)]
    }
    details["game_coverage"] = pd.DataFrame(
        {r: {q: len(sets[q] - sets[r]) for q in sets} for r in sets}
    )
    completed = (
        s.home_pts.notna()
        & s.away_pts.notna()
        & s.home_wl.isin(["W", "L"])
        & s.away_wl.isin(["W", "L"])
        & s.home_wl.ne(s.away_wl)
    )
    scheduled = (
        s[["home_pts", "away_pts", "home_wl", "away_wl"]].isna().all(axis=1) & ~sp
    )
    m.update(
        completed_games=int(completed.sum()),
        scheduled_games=int(scheduled.sum()),
        other_schedule_rows=int((~(completed | scheduled | sp)).sum()),
    )
    details["schedule_coverage"] = (
        s.loc[completed]
        .assign(covered=lambda x: x.game_id.isin(sets["team"]))
        .groupby(["season_type", "covered"], dropna=False)
        .size()
        .rename("games")
        .reset_index()
    )
    rec = b[KEY].merge(
        g[KEY], on=KEY, how="outer", indicator=True, validate="one_to_one"
    )
    for label in ("both", "left_only", "right_only"):
        m["player_" + label] = int(rec._merge.eq(label).sum())
    x = b.merge(g, on=KEY, suffixes=("_box", "_log"), validate="one_to_one")
    record(
        "player_team_mismatches",
        x.loc[x.team_id_box.ne(x.team_id_log), KEY + ["team_id_box", "team_id_log"]],
    )
    rows = []
    for a, c in STATS.items():
        av = pd.to_numeric(x[a], errors="coerce")
        cv = pd.to_numeric(x[c], errors="coerce")
        diff = (av - cv).abs()
        rows.append(
            dict(
                statistic=a,
                comparable=int(diff.notna().sum()),
                box_missing=int(av.isna().sum()),
                log_missing=int(cv.isna().sum()),
                mismatches=int(diff.gt(0).sum()),
                max_abs_difference=diff.max(),
            )
        )
    details["player_statistics"] = pd.DataFrame(rows)
    # Shooting percentages are checked against each source's own counts.
    rows = []
    for prefix, short in [
        ("field_goals", "fg"),
        ("three_pointers", "fg3"),
        ("free_throws", "ft"),
    ]:
        for frame, source, cols in [
            (
                b,
                "box",
                [prefix + "_made", prefix + "_attempted", prefix + "_percentage"],
            ),
            (g, "log", [short + "m", short + "a", short + "_pct"]),
        ]:
            made, att, pct = [pd.to_numeric(frame[c], errors="coerce") for c in cols]
            diff = (pct - made / att.where(att.gt(0))).abs()
            rows.append(
                dict(
                    source=source,
                    statistic=prefix,
                    zero_attempts=int(att.eq(0).sum()),
                    missing_positive_attempt_pct=int((att.gt(0) & pct.isna()).sum()),
                    invalid_counts=int(
                        (
                            made.lt(0)
                            | att.lt(0)
                            | made.gt(att)
                            | made.isna()
                            | att.isna()
                        ).sum()
                    ),
                    differences_above_tolerance=int(diff.gt(0.001).sum()),
                )
            )
    details["shooting_percentages"] = pd.DataFrame(rows)
    additive = [
        a
        for a in STATS
        if a
        not in [
            "rebounds_offensive",
            "rebounds_defensive",
            "rebounds_total",
            "turnovers",
            "fouls_personal",
            "plus_minus_points",
        ]
    ]
    ta = t.set_index(TEAM)[additive].apply(pd.to_numeric, errors="raise")
    rows = []
    for frame, source in [
        (b, "box"),
        (g.rename(columns={v: k for k, v in STATS.items()}), "log"),
    ]:
        agg = (
            frame.set_index(TEAM)[additive]
            .apply(pd.to_numeric, errors="raise")
            .groupby(level=TEAM)
            .sum(min_count=1)
        )
        m[source + "_team_key_difference"] = len(
            ta.index.symmetric_difference(agg.index)
        )
        diff = (agg - ta).abs()
        for col in additive:
            rows.append(
                dict(
                    source=source,
                    statistic=col,
                    comparable=int(diff[col].notna().sum()),
                    mismatches=int(diff[col].gt(0).sum()),
                    max_abs_difference=diff[col].max(),
                )
            )
    details["team_statistics"] = pd.DataFrame(rows)
    scores = pd.concat(
        [
            s.loc[completed, ["game_id", f"{side}_team_id", f"{side}_pts"]].rename(
                columns={f"{side}_team_id": "team_id", f"{side}_pts": "schedule_points"}
            )
            for side in ("home", "away")
        ],
        ignore_index=True,
    )
    scores = scores.loc[scores.game_id.isin(sets["team"])].copy()
    scores["team_id"] = gd.normalize_integer_ids(scores.team_id, "team_id")
    score = t[TEAM + ["points"]].merge(
        scores, on=TEAM, how="outer", indicator=True, validate="one_to_one"
    )
    record(
        "schedule_score_mismatches",
        score.loc[score._merge.ne("both") | score.points.ne(score.schedule_points)],
    )
    # Minutes: accept exactly 60 seconds as carry, preserve raw strings.
    parts = b.minutes.astype("string").str.strip().str.extract(r"^(\d+):([0-5]\d|60)$")
    seconds = pd.to_numeric(parts[0]) * 60 + pd.to_numeric(parts[1])
    b["seconds_played"] = seconds
    m["minutes_overflow"] = int(parts[1].eq("60").sum())
    logkeys = pd.MultiIndex.from_frame(g[KEY])
    inlog = pd.MultiIndex.from_frame(b[KEY]).isin(logkeys)
    comment = b.comment.fillna("").astype(str).str.strip()
    positive = (
        b[list(STATS)].apply(pd.to_numeric, errors="coerce").ne(0)
        & b[list(STATS)].apply(pd.to_numeric, errors="coerce").notna()
    )
    evidence = seconds.gt(0) | positive.any(axis=1) | inlog
    nonplay = comment.str.match(r"(?i)^(DNP|DND|NWT|Inactive)\b")
    record(
        "participation_conflicts",
        b.loc[evidence & nonplay, KEY + ["minutes", "comment"]],
    )
    record(
        "invalid_minutes",
        b.loc[
            seconds.isna()
            & ~(
                ~evidence
                & nonplay
                & b.minutes.fillna("").astype(str).str.strip().eq("")
            ),
            KEY + ["minutes", "comment"],
        ],
    )
    record(
        "unknown_participation",
        b.loc[~evidence & ~nonplay, KEY + ["minutes", "comment"]],
    )
    mt = b[KEY + ["seconds_played"]].merge(
        g[KEY + ["min"]], on=KEY, validate="one_to_one"
    )
    diff = (mt.seconds_played - pd.to_numeric(mt["min"], errors="coerce") * 60).abs()
    m["minutes_rounding_boundary"] = int(
        (diff.eq(30) & ((mt.seconds_played + 30) // 60).ne(mt["min"])).sum()
    )
    record("minutes_substantive_mismatches", mt.loc[diff.gt(30)])
    u = l.drop_duplicates().copy()
    m["lineup_exact_duplicates"] = len(l) - len(u)
    record(
        "lineup_action_conflicts",
        u.loc[u.duplicated(["game_id", "action_number"], keep=False)],
    )
    ids = u[HOME + AWAY]
    invalid = (
        ids.isna().any(axis=1)
        | ids.le(0).any(axis=1)
        | ids.nunique(axis=1).ne(10)
        | u.period.le(0)
        | u.action_number.lt(0)
    )
    record("invalid_lineups", u.loc[invalid])
    membership = pd.concat(
        [
            u[["game_id"] + cols]
            .melt(id_vars="game_id", value_name="player_id")
            .drop(columns="variable")
            .assign(side=side)
            for cols, side in [(HOME, "home"), (AWAY, "away")]
        ],
        ignore_index=True,
    ).drop_duplicates()
    bm = b[KEY + ["side", "comment"]].copy()
    bm.side = bm.side.astype("string").str.strip().str.lower()
    mc = membership.merge(
        bm,
        on=KEY,
        how="left",
        suffixes=("_lineup", "_box"),
        indicator=True,
        validate="many_to_one",
    )
    record("lineup_missing_players", mc.loc[mc._merge.ne("both")])
    record(
        "lineup_wrong_side",
        mc.loc[mc._merge.eq("both") & mc.side_lineup.ne(mc.side_box)],
    )
    lr = (
        membership[KEY]
        .drop_duplicates()
        .merge(g[KEY], on=KEY, how="outer", indicator=True, validate="one_to_one")
    )
    record("lineup_log_participation_differences", lr.loc[lr._merge.ne("both")])
    periods = u.groupby("game_id").period.max().rename("max_period")
    totals = (
        b.groupby(TEAM)
        .seconds_played.sum(min_count=1)
        .reset_index()
        .merge(periods, on="game_id", how="outer", validate="many_to_one")
    )
    totals["expected_seconds"] = 14400 + 1500 * (totals.max_period - 4).clip(lower=0)
    totals["seconds_difference"] = totals.seconds_played - totals.expected_seconds
    record(
        "team_minutes_warnings",
        totals.loc[
            totals.seconds_difference.abs().gt(team_seconds_tolerance)
            | totals.seconds_difference.isna()
            | totals.max_period.lt(4)
        ],
    )
    details["team_playing_time"] = totals
    return {"summary": m, "details": details}


def audit_all_seasons(seasons, load_season, **kwargs):
    """load_season(year) must return the five raw DataFrames. Failures remain visible."""
    for season in seasons:
        yield audit_season(load_season(season), season, **kwargs)
