#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# src/04_chow.py
#
# Identifies changes of ownership (CHOWs) in two administrative sources and
# writes one facility-level file per source plus their merge.
#
#   NHC Archive  A change is a substantive turnover in a facility's monthly
#                owner list: at least 50 percent of ownership shares differ
#                between one archive month and the facility's previous
#                archive month.
#   HCRIS        A change is each distinct change-of-ownership month reported
#                on a Medicare cost report (worksheet S-2, part II).
#
# Only changes dated inside the study window (config.START_YM to
# config.END_YM) are counted.
#
# -----------------------------------------------------------------------------
# NHC rule, step by step
# -----------------------------------------------------------------------------
#   1. Monthly owner list. For each facility and archive month, collect every
#      owner listed at each ownership level (5%+ indirect, 5%+ direct,
#      partnership interest). Shares are normalized to sum to 100 within a
#      facility, month, and level. Owners with a missing share are dropped
#      when other owners at that level report shares; when none do, owners
#      receive equal shares.
#   2. Comparison level. Consecutive months are compared at the highest level
#      present in both months, in the order indirect, direct, partnership, so
#      that a level appearing in one month and missing in the next is not read
#      as turnover. If the two months share no level, each month's own highest
#      level is used and the comparison is marked in the review file.
#   3. Turnover. Turnover is 1 minus the share overlap between the two owner
#      lists. A month is flagged when turnover is at least TURNOVER_THRESH.
#   4. One sale, several flags. A sale can be listed over several archive
#      months, and an owner list can drop out for a month and return. Flags
#      within MERGE_WINDOW_MONTHS of one another are grouped into one episode,
#      and the episode is judged on its net turnover: the owner list before
#      the first flag against the owner list after the last flag. An episode
#      whose net turnover is below the threshold is a reporting reversal, not
#      a change.
#   5. Family screen (optional). An episode is not counted when at least
#      FAMILY_KEEP_THRESH percent of ownership is held by individuals with the
#      same surname before and after. Only owners CMS lists as individuals
#      enter the screen; organizations never count as family. NHC lists
#      individuals as "LAST, FIRST", so the surname is the text before the
#      comma.
#   6. Dating. An episode is dated by the earliest association date among the
#      owners who are new after it, rounded to the month, provided that date
#      falls no more than DATE_LOOKBACK_MONTHS before the archive month in
#      which the new owners first appear. Otherwise the archive month is used.
#      The archive month lags the sale, often by several months.
#
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
#   data/interim/ownership.csv            (03_clean_ownership.py)
#   data/raw/medicare-cost-reports/mcr_flatfile_20??.*
#
# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
#   data/interim/chow_nh_compare.csv          one row per NHC facility
#   data/interim/chow_mcr.csv                 one row per HCRIS facility
#   data/interim/chow.csv                     facilities in both sources
#   data/interim/chow_nh_compare_review.csv   one row per NHC episode, with
#                                             the owner lists before and after,
#                                             for hand review
#   data/interim/chow_log.txt                 counts at each step and the
#                                             source-by-source crosstab
#
# chow.csv keeps the columns read downstream (n_chow_nh_compare, n_chow_mcr,
# nh_compare_chow_<k>_date, mcr_chow_<k>_date), so 06_panel.py and
# quarterly_panel.py are unchanged.
# =============================================================================

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

# ============================== Settings ======================================
TURNOVER_THRESH = 0.50          # share of ownership that must change
LEVEL_PRIORITY = ("indirect", "direct", "partnership")
MERGE_WINDOW_MONTHS = 6         # flags this close together form one episode
USE_FAMILY_SCREEN = True
FAMILY_KEEP_THRESH = 80.0       # percent retained by one surname
DATE_LOOKBACK_MONTHS = 24       # how far back an association date may date a change

WINDOW_START = pd.Period(cfg.START_YM.replace("/", "-"), "M")
WINDOW_END = pd.Period(cfg.END_YM.replace("/", "-"), "M")

# ============================== Paths =========================================
INTERIM_DIR = cfg.ensure_dir(cfg.INTERIM_DIR)
MCR_DIR = cfg.MCR_DIR

OWNERSHIP_FP = INTERIM_DIR / "ownership.csv"
MCR_PATTERNS = [
    "mcr_flatfile_20??.csv",
    "mcr_flatfile_20??.sas7bdat",
    "mcr_flatfile_20??.xpt",
    "mcr_flatfile_20??.XPT",
]

OWN_OUT = INTERIM_DIR / "chow_nh_compare.csv"
MCR_OUT = INTERIM_DIR / "chow_mcr.csv"
MERGE_OUT = INTERIM_DIR / "chow.csv"
REVIEW_OUT = INTERIM_DIR / "chow_nh_compare_review.csv"
LOG_OUT = INTERIM_DIR / "chow_log.txt"

LOG_LINES: list[str] = []


def log(msg: str) -> None:
    print(msg)
    LOG_LINES.append(msg)


# ============================== Name helpers ==================================
SUFFIXES = (
    r"\b(INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C\.|L\.P\.|LP|LLP|PLC|CO|"
    r"COMPANY|HOLDINGS?|PARTNERS?|PARTNERSHIP|CAPITAL|INVESTMENTS?|TRUST|GROUP)\b"
)


def clean_owner_name(s) -> str:
    """Matching key for an owner: upper case, punctuation and legal suffixes removed.

    Used only to recognize the same owner across months. It is not used to
    decide whether an owner is a person.
    """
    if pd.isna(s) or not str(s).strip():
        return ""
    x = str(s).upper()
    x = re.sub(r"[.,&/()\-']", " ", x)
    x = re.sub(SUFFIXES, "", x)
    return re.sub(r"\s+", " ", x).strip()


def surname_of_individual(raw_name) -> str:
    """Surname of an owner CMS lists as an individual ("LAST, FIRST")."""
    if pd.isna(raw_name):
        return ""
    s = str(raw_name).strip().upper()
    if not s:
        return ""
    last = s.split(",", 1)[0] if "," in s else s.split()[0]
    return re.sub(r"[^A-Z]", "", last)


def month_index(p: pd.Period) -> int:
    return p.year * 12 + p.month - 1


def index_to_period(i: int) -> pd.Period:
    return pd.Period(year=i // 12, month=i % 12 + 1, freq="M")


# ============================== NHC: owner lists ==============================
def load_owner_lists(fp: Path) -> pd.DataFrame:
    """One row per facility, archive month, level, and owner, with a normalized share."""
    usecols = [
        "cms_certification_number", "role", "owner_type", "owner_name",
        "ownership_percentage", "association_date", "year_month",
    ]
    own = pd.read_csv(fp, dtype=str, usecols=usecols, low_memory=False)
    log(f"[nhc] rows read: {len(own):,}")

    own["ccn"] = cfg.normalize_ccn_any(own["cms_certification_number"])
    own["level"] = own["role"].str.strip().str.lower()
    own = own[own["ccn"].notna() & own["level"].isin(LEVEL_PRIORITY)].copy()

    ym = pd.to_datetime(own["year_month"].str.strip() + "/01", format="%Y/%m/%d", errors="coerce")
    own = own[ym.notna()].copy()
    ym = ym[ym.notna()]
    own["ym"] = (ym.dt.year * 12 + ym.dt.month - 1).astype("int32")

    # Clean each distinct name once rather than once per row.
    names = pd.Series(own["owner_name"].unique())
    key_map = dict(zip(names, names.map(clean_owner_name)))
    own["owner"] = own["owner_name"].map(key_map)
    own = own[own["owner"] != ""].copy()

    own["indiv"] = own["owner_type"].str.strip().str.lower().eq("individual")
    own["surname"] = np.where(own["indiv"], own["owner_name"].map(surname_of_individual), "")

    own["pct"] = pd.to_numeric(own["ownership_percentage"], errors="coerce")
    ad = pd.to_datetime(own["association_date"], errors="coerce")
    own["assoc"] = (ad.dt.year * 12 + ad.dt.month - 1).astype("Int32")

    keys = ["ccn", "ym", "level", "owner"]
    g = own.groupby(keys, sort=False)
    agg = g.agg(
        pct=("pct", "sum"),
        n_pct=("pct", "count"),
        assoc=("assoc", "min"),
        indiv=("indiv", "max"),
        surname=("surname", "first"),
        raw_name=("owner_name", "first"),
    ).reset_index()
    agg.loc[agg["n_pct"] == 0, "pct"] = np.nan

    lvl = ["ccn", "ym", "level"]
    has_pct = agg["pct"].notna()
    tot = agg["pct"].where(has_pct, 0).groupby([agg[k] for k in lvl]).transform("sum")
    any_pct = has_pct.groupby([agg[k] for k in lvl]).transform("max")
    n_own = agg.groupby(lvl)["owner"].transform("size")

    agg["w"] = np.where(
        any_pct & (tot > 0),
        agg["pct"].fillna(0) * 100.0 / tot.replace(0, np.nan),
        100.0 / n_own,
    )
    agg = agg[agg["w"] > 0].copy()

    log(f"[nhc] facilities: {agg['ccn'].nunique():,}; facility-months: "
        f"{agg[['ccn', 'ym']].drop_duplicates().shape[0]:,}")
    return agg[["ccn", "ym", "level", "owner", "w", "assoc", "indiv", "surname", "raw_name"]]


def build_rosters(agg: pd.DataFrame) -> dict:
    """ccn -> {month -> {level -> {owner: (share, is_individual, surname, assoc_month, raw_name)}}}."""
    rosters: dict = defaultdict(lambda: defaultdict(dict))
    for r in agg.sort_values(["ccn", "ym"]).itertuples(index=False):
        assoc = None if pd.isna(r.assoc) else int(r.assoc)
        rosters[r.ccn][r.ym].setdefault(r.level, {})[r.owner] = (
            float(r.w), bool(r.indiv), r.surname, assoc, r.raw_name
        )
    return rosters


# ============================== NHC: comparisons ==============================
def comparison_levels(prev: dict, curr: dict) -> tuple[str, str, bool]:
    for lvl in LEVEL_PRIORITY:
        if lvl in prev and lvl in curr:
            return lvl, lvl, False
    top_prev = next(l for l in LEVEL_PRIORITY if l in prev)
    top_curr = next(l for l in LEVEL_PRIORITY if l in curr)
    return top_prev, top_curr, True


def turnover(a: dict, b: dict) -> float:
    overlap = sum(min(a[o][0], b[o][0]) for o in a.keys() & b.keys())
    return max(0.0, 1.0 - overlap / 100.0)


def family_keep(a: dict, b: dict) -> float:
    """Percent of ownership held by the same surname before and after (individuals only)."""
    sa: dict = defaultdict(float)
    sb: dict = defaultdict(float)
    for w, indiv, sn, _, _ in a.values():
        if indiv and sn:
            sa[sn] += w
    for w, indiv, sn, _, _ in b.values():
        if indiv and sn:
            sb[sn] += w
    return sum(min(sa[s], sb.get(s, 0.0)) for s in sa)


def describe(owners: dict, k: int = 12) -> str:
    items = sorted(owners.items(), key=lambda kv: -kv[1][0])[:k]
    return "; ".join(f"{v[4]} ({v[0]:.0f}%)" for _, v in items)


def find_nhc_changes(rosters: dict) -> tuple[pd.DataFrame, dict]:
    stats = defaultdict(int)
    episodes = []

    for ccn, months_dict in rosters.items():
        months = sorted(months_dict)

        # Flag each consecutive pair of archive months with turnover >= threshold.
        flags = []
        for prev_m, curr_m in zip(months, months[1:]):
            stats["pairs_compared"] += 1
            lp, lc, fallback = comparison_levels(months_dict[prev_m], months_dict[curr_m])
            a, b = months_dict[prev_m][lp], months_dict[curr_m][lc]
            if a == b:
                continue
            t = turnover(a, b)
            if t >= TURNOVER_THRESH:
                flags.append((prev_m, curr_m, t, fallback))
                stats["pairs_flagged"] += 1
                stats["pairs_flagged_level_fallback"] += int(fallback)

        # Group nearby flags into episodes.
        groups: list[list] = []
        for f in flags:
            if groups and f[1] - groups[-1][-1][1] <= MERGE_WINDOW_MONTHS:
                groups[-1].append(f)
            else:
                groups.append([f])

        for gi, grp in enumerate(groups, start=1):
            before_m, after_m = grp[0][0], grp[-1][1]
            lp, lc, fallback = comparison_levels(months_dict[before_m], months_dict[after_m])
            a, b = months_dict[before_m][lp], months_dict[after_m][lc]
            net_t = turnover(a, b)
            fam = family_keep(a, b) if USE_FAMILY_SCREEN else 0.0

            # Date: earliest association date among owners new after the episode,
            # bounded below by the lookback; otherwise the first flagged archive month.
            first_seen = grp[0][1]
            new_assoc = [b[o][3] for o in b.keys() - a.keys() if b[o][3] is not None]
            cand = min(new_assoc) if new_assoc else None
            if cand is not None and first_seen - DATE_LOOKBACK_MONTHS <= cand <= first_seen:
                event_m, date_source = cand, "association_date"
            else:
                event_m, date_source = first_seen, "archive_month"
            event_p = index_to_period(event_m)

            if net_t < TURNOVER_THRESH:
                outcome = "reverted"
            elif USE_FAMILY_SCREEN and fam >= FAMILY_KEEP_THRESH:
                outcome = "family_screen"
            elif not (WINDOW_START <= event_p <= WINDOW_END):
                outcome = "outside_window"
            else:
                outcome = "change"
            stats[f"episodes_{outcome}"] += 1
            if outcome == "change":
                stats[f"date_source_{date_source}"] += 1

            episodes.append({
                "cms_certification_number": ccn,
                "episode": gi,
                "n_flagged_months": len(grp),
                "month_before": str(index_to_period(before_m)),
                "month_after": str(index_to_period(after_m)),
                "level_before": lp,
                "level_after": lc,
                "level_fallback": bool(fallback),
                "net_turnover": round(net_t, 4),
                "family_keep_pct": round(fam, 1),
                "event_month": str(event_p),
                "date_source": date_source,
                "outcome": outcome,
                "owners_before": describe(a),
                "owners_after": describe(b),
            })

    cols = [
        "cms_certification_number", "episode", "n_flagged_months", "month_before",
        "month_after", "level_before", "level_after", "level_fallback", "net_turnover",
        "family_keep_pct", "event_month", "date_source", "outcome",
        "owners_before", "owners_after",
    ]
    return pd.DataFrame(episodes, columns=cols), stats


def summarize_changes(events: pd.DataFrame, universe: pd.Series, prefix: str, count_col: str) -> pd.DataFrame:
    """Facility-level counts and wide date columns (<prefix>_chow_<k>_date, YYYY-MM-01)."""
    base = pd.DataFrame({"cms_certification_number": universe.dropna().drop_duplicates()})
    if events.empty:
        base[count_col] = 0
        return base

    ev = events.sort_values(["cms_certification_number", "event_date"]).copy()
    ev["k"] = ev.groupby("cms_certification_number").cumcount() + 1
    wide = ev.pivot(index="cms_certification_number", columns="k", values="event_date")
    wide.columns = [f"{prefix}_chow_{k}_date" for k in wide.columns]
    wide = wide.reset_index()
    n = ev.groupby("cms_certification_number").size().rename(count_col).reset_index()

    out = base.merge(n, on="cms_certification_number", how="left").merge(
        wide, on="cms_certification_number", how="left"
    )
    out[count_col] = out[count_col].fillna(0).astype("Int16")
    return out.sort_values("cms_certification_number").reset_index(drop=True)


def build_chow_from_ownership() -> pd.DataFrame:
    agg = load_owner_lists(OWNERSHIP_FP)
    universe = agg["ccn"]
    rosters = build_rosters(agg)
    del agg

    review, stats = find_nhc_changes(rosters)
    cfg.atomic_overwrite_csv(review, REVIEW_OUT, index=False)
    log(f"[nhc] review file: {REVIEW_OUT} ({len(review):,} episodes)")
    for k in sorted(stats):
        log(f"[nhc] {k}: {stats[k]:,}")

    ch = review[review["outcome"] == "change"].copy()
    ch["event_date"] = pd.PeriodIndex(ch["event_month"], freq="M").to_timestamp().strftime("%Y-%m-%d")
    return summarize_changes(ch, universe, "nh_compare", "n_chow_nh_compare")


# ============================== HCRIS =========================================
def read_mcr_three_cols() -> pd.DataFrame:
    files, seen = [], set()
    for pat in MCR_PATTERNS:
        for f in sorted(MCR_DIR.glob(pat)):
            if f not in seen:
                files.append(f)
                seen.add(f)
    if not files:
        raise FileNotFoundError(f"No MCR files matched in {MCR_DIR}")

    try:
        import pyreadstat
        has_pyreadstat = True
    except Exception:
        has_pyreadstat = False

    targets = {
        "PRVDR_NUM": ["prvdr_num", "provider_number", "provnum", "prvdrnum"],
        "S2_2_CHOW": ["s2_2_chow", "s22_chow", "chow", "s2_2_chow_cd", "s2_2_chow_flag", "s2_2_chow_ind"],
        "S2_2_CHOWDATE": ["s2_2_chowdate", "s2_2_chow_date", "s22_chow_date", "chow_date"],
    }

    def select_three(df: pd.DataFrame) -> pd.DataFrame:
        lower = {c.lower().strip(): c for c in df.columns}
        out = pd.DataFrame(index=df.index)
        for t, cands in targets.items():
            src = next((lower[c] for c in [t.lower()] + cands if c in lower), None)
            out[t] = df[src] if src is not None else pd.NA
        return out

    frames = []
    for fp in files:
        suffix = fp.suffix.lower()
        try:
            if suffix == ".csv":
                df = cfg.read_csv_robust(fp, dtype=str, low_memory=False)
            elif suffix in {".sas7bdat", ".xpt"} and has_pyreadstat:
                reader = pyreadstat.read_sas7bdat if suffix == ".sas7bdat" else pyreadstat.read_xport
                _, meta = reader(fp, metadataonly=True)
                lower = {c.lower().strip(): c for c in meta.column_names}
                use = [lower[c] for t, cands in targets.items()
                       for c in [t.lower()] + cands if c in lower]
                df, _ = reader(fp, usecols=list(dict.fromkeys(use)))
            elif suffix in {".sas7bdat", ".xpt"}:
                fmt = "sas7bdat" if suffix == ".sas7bdat" else "xport"
                df = pd.read_sas(fp, format=fmt, encoding="latin1")
            else:
                continue
            frames.append(select_three(df))
        except Exception as e:
            log(f"[warn] {fp.name}: {e}")

    if not frames:
        return pd.DataFrame(columns=list(targets))
    return pd.concat(frames, ignore_index=True)


def coerce_maybe_sas_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() and num.notna().sum() >= max(1, int(0.5 * s.notna().sum())):
        return (pd.Timestamp("1960-01-01") + pd.to_timedelta(num, unit="D")).astype("datetime64[ns]")
    return pd.to_datetime(s.astype("string").str.strip(), errors="coerce")


def build_chow_from_mcr() -> pd.DataFrame:
    raw = read_mcr_three_cols()
    raw["ccn"] = cfg.normalize_ccn_any(raw["PRVDR_NUM"])
    universe = raw["ccn"]

    raw["date"] = coerce_maybe_sas_date(raw["S2_2_CHOWDATE"])
    ev = raw.loc[raw["ccn"].notna() & raw["date"].notna(), ["ccn", "date"]].copy()
    ev["event_p"] = ev["date"].dt.to_period("M")
    # One change per facility and month: amended or overlapping reports often
    # repeat the same change with slightly different day-level dates.
    ev = ev.drop_duplicates(["ccn", "event_p"])
    n_all = len(ev)
    ev = ev[(ev["event_p"] >= WINDOW_START) & (ev["event_p"] <= WINDOW_END)].copy()
    log(f"[mcr] distinct change months: {n_all:,}; inside study window: {len(ev):,}")

    ev = ev.rename(columns={"ccn": "cms_certification_number"})
    ev["event_date"] = ev["event_p"].dt.to_timestamp().dt.strftime("%Y-%m-%d")
    return summarize_changes(ev[["cms_certification_number", "event_date"]], universe, "mcr", "n_chow_mcr")


# ============================== Merge & save ==================================
def crosstab_lines(merged: pd.DataFrame) -> list[str]:
    def bin_(x):
        return x.clip(upper=2).map({0: "0", 1: "1", 2: "2+"})
    tab = pd.crosstab(bin_(merged["n_chow_mcr"].astype(int)).rename("HCRIS"),
                      bin_(merged["n_chow_nh_compare"].astype(int)).rename("NHC"))
    return ["[crosstab] facilities in both sources, HCRIS rows by NHC columns:"] + \
        tab.to_string().splitlines()


def main():
    log(f"[settings] turnover >= {TURNOVER_THRESH}; levels {LEVEL_PRIORITY}; "
        f"merge window {MERGE_WINDOW_MONTHS} months; family screen {USE_FAMILY_SCREEN} "
        f"(>= {FAMILY_KEEP_THRESH}%); date lookback {DATE_LOOKBACK_MONTHS} months; "
        f"window {WINDOW_START} to {WINDOW_END}")

    own_full = build_chow_from_ownership()
    cfg.atomic_overwrite_csv(own_full, OWN_OUT, index=False)
    log(f"[save] {OWN_OUT} ({len(own_full):,} facilities)")

    mcr_full = build_chow_from_mcr()
    cfg.atomic_overwrite_csv(mcr_full, MCR_OUT, index=False)
    log(f"[save] {MCR_OUT} ({len(mcr_full):,} facilities)")

    merged = own_full.merge(mcr_full, on="cms_certification_number", how="inner")
    for c in ["n_chow_nh_compare", "n_chow_mcr"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype("Int16")
    merged["n_chow_any"] = ((merged["n_chow_nh_compare"] > 0) | (merged["n_chow_mcr"] > 0)).astype("Int8")
    merged = merged.sort_values("cms_certification_number").reset_index(drop=True)
    cfg.atomic_overwrite_csv(merged, MERGE_OUT, index=False)
    log(f"[save] {MERGE_OUT} ({len(merged):,} facilities in both sources)")

    for line in crosstab_lines(merged):
        log(line)

    LOG_OUT.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    print(f"[save] {LOG_OUT}")


if __name__ == "__main__":
    main()
