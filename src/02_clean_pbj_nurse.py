#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# PBJ Nurse Staffing —> Normalize -> Monthly Aggregate -> Combine -> Quarterly
#
# Updated version:
# - preserves the existing monthly PBJ pipeline
# - allows skipping monthly rebuild if pbj_nurse.csv already exists
# - adds a quarterly PBJ output built from the finalized monthly pbj_nurse.csv
# - quarterly staffing/intensity measures are recomputed from quarterly totals
# - applies only light monthly validity cleaning before quarterly aggregation
# - adds quarterly QA / plausibility flags
# =============================================================================

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================== Paths / Config ================================
PBJ_DIR = cfg.PBJ_DIR
PBJ_GLOB = "pbj_nurse_????_Q[1-4].csv"

INTERIM_DIR = cfg.ensure_dir(cfg.INTERIM_DIR)
OUT_FP = INTERIM_DIR / "pbj_nurse.csv"
OUT_FP_QUARTERLY = INTERIM_DIR / "pbj_nurse_quarterly.csv"

KEEP_HOUR_TOTALS = True

# --------------------------- Raw hour columns --------------------------------
# CMS's PBJ nurse header is not stable across the sample period. The 2017
# files use lowercase headers and several variant spellings; 2018 onward use a
# consistent TitleCase schema. 2017 Q1 has no hrs_rndon column at all (only
# hrs_rn_donadmin), and 2017 Q2 contains both. Headers are lowercased on read,
# so all aliases below are lowercase.
#
# Keys are the canonical names used downstream; values list acceptable source
# columns in order of preference, canonical first.
HOUR_COL_ALIASES = {
    "hrs_rn":       ["hrs_rn"],
    "hrs_lpn":      ["hrs_lpn"],
    "hrs_cna":      ["hrs_cna"],
    "hrs_rndon":    ["hrs_rndon", "hrs_rn_donadmin"],
    "hrs_rnadmin":  ["hrs_rnadmin"],
    "hrs_lpnadmin": ["hrs_lpnadmin", "hrs_lpn_admin"],
    "hrs_natrn":    ["hrs_natrn", "hrs_na_trn"],
    "hrs_medaide":  ["hrs_medaide"],
}

# Core direct-care categories. These three define total_hours and every HPRD
# measure used in the paper. Do not extend this list without revisiting all
# downstream staffing results.
CORE_HOUR_COLS = ["hrs_rn", "hrs_lpn", "hrs_cna"]

# Administrative nursing roles, carried through as separate raw-hour columns.
# CMS's Five-Star RN definition is hrs_rn + hrs_rnadmin + hrs_rndon; keeping
# these separate preserves the core definition above while making the
# Five-Star-consistent measure constructible downstream.
ADMIN_HOUR_COLS = ["hrs_rndon", "hrs_rnadmin", "hrs_lpnadmin"]

# Remaining nursing categories that CMS counts toward total nurse staffing but
# that the core three exclude. Carried separately for the same reason.
OTHER_NURSE_HOUR_COLS = ["hrs_natrn", "hrs_medaide"]

ALL_HOUR_COLS = CORE_HOUR_COLS + ADMIN_HOUR_COLS + OTHER_NURSE_HOUR_COLS

# Run flags
RUN_BUILD_MONTHLY = True
RUN_BUILD_QUARTERLY = True

print(f"[paths] PBJ_DIR={PBJ_DIR}")
print(f"[paths] OUT_FP={OUT_FP}")
print(f"[paths] OUT_FP_QUARTERLY={OUT_FP_QUARTERLY}")
print(
    f"[flags] RUN_BUILD_MONTHLY={RUN_BUILD_MONTHLY}, "
    f"RUN_BUILD_QUARTERLY={RUN_BUILD_QUARTERLY}"
)

# ============================== Helpers ======================================
def to_date_from_int_yyyymmdd(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("Int64"), format="%Y%m%d", errors="coerce")


# -------- vectorized CY_QTR parsing ----------
_QRX = re.compile(
    r"(?i)(?:CY)?\s*(20\d{2})?\s*[- ]?Q(?:TR)?\s*([1-4])|^\s*([1-4])\s*$"
)

def normalize_cy_qtr(cy_qtr: pd.Series, workdate: pd.Series) -> pd.Series:
    s = cy_qtr.astype("string")
    m = s.str.extract(_QRX)

    y = pd.to_numeric(m[0], errors="coerce").astype("Int64")
    q = pd.to_numeric(m[1].fillna(m[2]), errors="coerce").astype("Int64")

    y = y.fillna(workdate.dt.year.astype("Int64"))

    out = pd.Series(pd.NA, index=s.index, dtype="string")
    mask = y.notna() & q.notna()
    out.loc[mask] = y[mask].astype(str) + "Q" + q[mask].astype(str)

    still = out.isna()
    if still.any():
        qn = ((workdate.dt.month - 1) // 3 + 1).astype("Int64")
        out.loc[still] = (
            workdate.dt.year.astype("Int64").astype(str) + "Q" + qn.astype(str)
        )

    return out


def read_pbj_csv(fp: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(fp, low_memory=False, sep=",", encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


# ============================== Normalization =================================
def resolve_hour_columns(df: pd.DataFrame, fp_name: str = "") -> pd.DataFrame:
    """Map variant PBJ hour-column spellings onto canonical names.

    Emits a warning rather than failing when a column is absent, so a schema
    change in a single quarter does not abort the build. Warnings are printed
    per file and should be checked after a full rebuild: a column that is
    absent everywhere will otherwise appear as a legitimate column of zeros.
    """
    for canon, aliases in HOUR_COL_ALIASES.items():
        present = [a for a in aliases if a in df.columns]

        if not present:
            print(f"[warn] {fp_name}: no source column for {canon}; filled with 0.0")
            df[canon] = 0.0
            continue

        if len(present) > 1:
            print(
                f"[warn] {fp_name}: multiple source columns for {canon} "
                f"({present}); using {present[0]}"
            )

        if present[0] != canon:
            print(f"[alias] {fp_name}: {present[0]} -> {canon}")
            df[canon] = df[present[0]]

    return df


def normalize_needed_columns(df_raw: pd.DataFrame, fp_name: str = "") -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "provnum" in df.columns and "cms_certification_number" not in df.columns:
        df.rename(columns={"provnum": "cms_certification_number"}, inplace=True)
    if "mdscensus" in df.columns and "mds_census" not in df.columns:
        df.rename(columns={"mdscensus": "mds_census"}, inplace=True)

    df = resolve_hour_columns(df, fp_name)

    if "cms_certification_number" not in df.columns:
        raise ValueError("Missing cms_certification_number/provnum")
    df["cms_certification_number"] = cfg.normalize_ccn_any(df["cms_certification_number"])

    if "workdate" not in df.columns:
        raise ValueError("Missing workdate column")
    if pd.api.types.is_integer_dtype(df["workdate"]) or pd.api.types.is_string_dtype(df["workdate"]):
        df["workdate"] = to_date_from_int_yyyymmdd(df["workdate"])
    else:
        df["workdate"] = pd.to_datetime(df["workdate"], errors="coerce")

    for c in ALL_HOUR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32").fillna(0.0)

    if "mds_census" not in df.columns:
        df["mds_census"] = np.nan
    df["mds_census"] = pd.to_numeric(df["mds_census"], errors="coerce").astype("float32")

    if "cy_qtr" not in df.columns:
        df["cy_qtr"] = pd.NA

    return df[
        [
            "cms_certification_number",
            "workdate",
            *ALL_HOUR_COLS,
            "mds_census",
            "cy_qtr",
        ]
    ]


# ====================== File -> Monthly Aggregation ============================
def process_file_monthly(fp: Path) -> pd.DataFrame:
    df = normalize_needed_columns(read_pbj_csv(fp), fp.name)
    df["quarter_row"] = normalize_cy_qtr(df["cy_qtr"], df["workdate"])

    # Daily
    daily_hours_agg = {c: (c, "sum") for c in ALL_HOUR_COLS}
    daily = (
        df.groupby(["cms_certification_number", "workdate"], as_index=False)
        .agg(
            **daily_hours_agg,
            mds_census=("mds_census", "mean"),
            quarter=("quarter_row", "first"),
        )
    )

    # total_hours is the sum of the three core direct-care categories only.
    # Administrative nursing hours are carried separately and deliberately not
    # added here, so existing staffing estimates are unchanged by their addition.
    daily["total_hours"] = daily[CORE_HOUR_COLS].sum(axis=1).astype("float32")
    daily["year_month_p"] = daily["workdate"].dt.to_period("M")
    daily["days_in_mo"] = daily["workdate"].dt.days_in_month

    # Monthly
    monthly = (
        daily.groupby(["cms_certification_number", "year_month_p"], as_index=False)
        .agg(
            rn_hours_month=("hrs_rn", "sum"),
            lpn_hours_month=("hrs_lpn", "sum"),
            cna_hours_month=("hrs_cna", "sum"),
            rndon_hours_month=("hrs_rndon", "sum"),
            rnadmin_hours_month=("hrs_rnadmin", "sum"),
            lpnadmin_hours_month=("hrs_lpnadmin", "sum"),
            natrn_hours_month=("hrs_natrn", "sum"),
            medaide_hours_month=("hrs_medaide", "sum"),
            total_hours=("total_hours", "sum"),
            resident_days=("mds_census", "sum"),
            avg_daily_census=("mds_census", "mean"),
            days_reported=("workdate", "nunique"),
            days_in_month=("days_in_mo", "max"),
            quarter=("quarter", "first"),
        )
    )

    monthly["coverage_ratio"] = monthly["days_reported"] / monthly["days_in_month"]

    denom = monthly["resident_days"].replace({0: np.nan})
    monthly["rn_hprd"] = monthly["rn_hours_month"] / denom
    monthly["lpn_hprd"] = monthly["lpn_hours_month"] / denom
    monthly["cna_hprd"] = monthly["cna_hours_month"] / denom
    monthly["total_hprd"] = monthly["total_hours"] / denom

    # year_month as 'YYYY/MM'
    ym = monthly["year_month_p"].astype("period[M]")
    monthly["year_month"] = (
        ym.dt.year.astype(int).astype(str)
        + "/"
        + ym.dt.month.astype(int).astype(str).str.zfill(2)
    )

    # Casts
    for c in [
        "rn_hours_month",
        "lpn_hours_month",
        "cna_hours_month",
        "rndon_hours_month",
        "rnadmin_hours_month",
        "lpnadmin_hours_month",
        "natrn_hours_month",
        "medaide_hours_month",
        "total_hours",
        "resident_days",
        "avg_daily_census",
        "rn_hprd",
        "lpn_hprd",
        "cna_hprd",
        "total_hprd",
        "coverage_ratio",
    ]:
        monthly[c] = pd.to_numeric(monthly[c], errors="coerce").astype("float32")

    monthly["days_reported"] = monthly["days_reported"].astype("Int16")
    monthly["days_in_month"] = monthly["days_in_month"].astype("Int16")

    # Final ordering
    monthly = monthly.sort_values(["cms_certification_number", "year_month"], kind="mergesort")

    # Drop temp
    monthly = monthly.drop(columns=["year_month_p"])

    return monthly


# ======================= Monthly builder ======================================
def build_monthly_from_raw():
    files = sorted(PBJ_DIR.glob(PBJ_GLOB))
    print(f"[scan] {len(files)} files found")

    frames = []
    failed = 0

    for fp in files:
        try:
            m = process_file_monthly(fp)
            print(f"[ok] {fp.name}: {len(m):,} rows")
            if not m.empty:
                frames.append(m)
        except Exception as e:
            print(f"[fail] {fp.name}: {e}")
            failed += 1

    monthly = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"[concat] monthly rows = {len(monthly):,}")

    if monthly.empty:
        cfg.atomic_overwrite_csv(monthly, OUT_FP, index=False)
        print(f"[saved] pbj nurse panel → {OUT_FP} (rows=0)")
        return

    cols = [
        "cms_certification_number",
        "quarter",
        "year_month",
        *([
            "rn_hours_month",
            "lpn_hours_month",
            "cna_hours_month",
            "rndon_hours_month",
            "rnadmin_hours_month",
            "lpnadmin_hours_month",
            "natrn_hours_month",
            "medaide_hours_month",
            "total_hours",
        ] if KEEP_HOUR_TOTALS else []),
        "resident_days",
        "avg_daily_census",
        "rn_hprd",
        "lpn_hprd",
        "cna_hprd",
        "total_hprd",
        "days_reported",
        "days_in_month",
        "coverage_ratio",
    ]
    monthly = monthly[cols]

    # ---------- Sort and compute gap_from_prev_months once ----------
    ord_dt = pd.to_datetime(monthly["year_month"] + "/01", format="%Y/%m/%d", errors="coerce")
    monthly = monthly.assign(
        _ord=ord_dt,
        _mi=(ord_dt.dt.year * 12 + ord_dt.dt.month).astype("Int32"),
    )

    monthly = monthly.sort_values(["cms_certification_number", "_ord"], kind="mergesort")

    monthly["gap_from_prev_months"] = (
        monthly.groupby("cms_certification_number")["_mi"]
        .diff()
        .fillna(1)
        .astype("Int16")
        - 1
    ).clip(lower=0)

    monthly = (
        monthly.drop(columns=["_ord", "_mi"])
        .sort_values(["cms_certification_number", "year_month"], kind="mergesort")
        .reset_index(drop=True)
    )

    cfg.atomic_overwrite_csv(monthly, OUT_FP, index=False)

    print(f"[saved] pbj nurse panel → {OUT_FP} (rows={len(monthly):,})")
    print(
        f"[qa] files_read={len(files):,}, "
        f"failed_files={failed:,}, "
        f"unique_ccn={monthly['cms_certification_number'].nunique(dropna=True):,}"
    )


# ======================= Quarterly builder from monthly ========================
def build_quarterly_from_monthly():
    if not OUT_FP.exists():
        raise FileNotFoundError(f"Monthly PBJ panel not found: {OUT_FP}")

    monthly = pd.read_csv(
        OUT_FP,
        dtype={"cms_certification_number": "string"},
        low_memory=False
    )

    if monthly.empty:
        cfg.atomic_overwrite_csv(monthly, OUT_FP_QUARTERLY, index=False)
        print(f"[saved] quarterly pbj nurse panel → {OUT_FP_QUARTERLY} (rows=0)")
        return

    monthly["_ord"] = pd.to_datetime(
        monthly["year_month"] + "/01",
        format="%Y/%m/%d",
        errors="coerce"
    )

    monthly = monthly.dropna(subset=["cms_certification_number", "year_month", "_ord"]).copy()

    # ---------------- Light monthly validity cleaning BEFORE quarterly aggregation
    # Drop only mechanically impossible monthly rows, not the full monthly HPRD filter
    for col in [
        "rn_hours_month", "lpn_hours_month", "cna_hours_month",
        "rndon_hours_month", "rnadmin_hours_month", "lpnadmin_hours_month",
        "natrn_hours_month", "medaide_hours_month",
        "total_hours",
        "resident_days", "avg_daily_census", "days_reported", "days_in_month",
        "coverage_ratio", "gap_from_prev_months"
    ]:
        if col in monthly.columns:
            monthly[col] = pd.to_numeric(monthly[col], errors="coerce")

    before_light = len(monthly)

    valid_mask = pd.Series(True, index=monthly.index)

    # IDs / dates already partly handled, but keep explicit
    valid_mask &= monthly["cms_certification_number"].notna()
    valid_mask &= monthly["year_month"].notna()
    valid_mask &= monthly["_ord"].notna()

    # nonnegative monthly quantities
    for col in ["rn_hours_month", "lpn_hours_month", "cna_hours_month",
                "rndon_hours_month", "rnadmin_hours_month", "lpnadmin_hours_month",
                "natrn_hours_month", "medaide_hours_month",
                "total_hours", "resident_days", "days_reported"]:
        if col in monthly.columns:
            valid_mask &= (monthly[col].isna() | (monthly[col] >= 0))

    # positive days in month
    if "days_in_month" in monthly.columns:
        valid_mask &= monthly["days_in_month"].notna()
        valid_mask &= monthly["days_in_month"] > 0

    # days_reported cannot exceed calendar days
    if {"days_reported", "days_in_month"}.issubset(monthly.columns):
        valid_mask &= (monthly["days_reported"] <= monthly["days_in_month"])

    # coverage ratio should be roughly within [0, 1]
    if "coverage_ratio" in monthly.columns:
        valid_mask &= (monthly["coverage_ratio"].isna() | ((monthly["coverage_ratio"] >= 0) & (monthly["coverage_ratio"] <= 1.01)))

    monthly = monthly.loc[valid_mask].copy()

    print(f"[qa-quarterly] light monthly validity cleaning: {before_light:,} -> {len(monthly):,}")

    monthly["year"] = monthly["_ord"].dt.year.astype("Int64")
    monthly["quarter_num"] = ((monthly["_ord"].dt.month - 1) // 3 + 1).astype("Int64")
    monthly["quarter"] = "Q" + monthly["quarter_num"].astype(str)

    numeric_cols = [
        "rn_hours_month",
        "lpn_hours_month",
        "cna_hours_month",
        "rndon_hours_month",
        "rnadmin_hours_month",
        "lpnadmin_hours_month",
        "natrn_hours_month",
        "medaide_hours_month",
        "total_hours",
        "resident_days",
        "avg_daily_census",
        "days_reported",
        "days_in_month",
        "coverage_ratio",
        "rn_hprd",
        "lpn_hprd",
        "cna_hprd",
        "total_hprd",
        "gap_from_prev_months",
    ]
    for col in numeric_cols:
        if col in monthly.columns:
            monthly[col] = pd.to_numeric(monthly[col], errors="coerce")

    monthly = monthly.sort_values(
        ["cms_certification_number", "year", "quarter_num", "_ord"],
        kind="mergesort"
    )

    grp = ["cms_certification_number", "year", "quarter"]

    qtr = (
        monthly.groupby(grp, sort=False)
        .agg(
            rn_hours_quarter=("rn_hours_month", "sum"),
            lpn_hours_quarter=("lpn_hours_month", "sum"),
            cna_hours_quarter=("cna_hours_month", "sum"),
            rndon_hours_quarter=("rndon_hours_month", "sum"),
            rnadmin_hours_quarter=("rnadmin_hours_month", "sum"),
            lpnadmin_hours_quarter=("lpnadmin_hours_month", "sum"),
            natrn_hours_quarter=("natrn_hours_month", "sum"),
            medaide_hours_quarter=("medaide_hours_month", "sum"),
            total_hours_quarter=("total_hours", "sum"),
            resident_days_quarter=("resident_days", "sum"),
            days_reported_quarter=("days_reported", "sum"),
            days_in_quarter=("days_in_month", "sum"),
            months_observed_in_quarter=("year_month", "nunique"),
            last_year_month_in_quarter=("year_month", "last"),
        )
        .reset_index()
    )

    # Recompute quarterly averages / ratios from quarterly totals
    denom = qtr["resident_days_quarter"].replace({0: np.nan})
    qtr["rn_hprd"] = qtr["rn_hours_quarter"] / denom
    qtr["lpn_hprd"] = qtr["lpn_hours_quarter"] / denom
    qtr["cna_hprd"] = qtr["cna_hours_quarter"] / denom
    qtr["total_hprd"] = qtr["total_hours_quarter"] / denom

    qtr["avg_daily_census"] = qtr["resident_days_quarter"] / qtr["days_in_quarter"].replace({0: np.nan})
    qtr["coverage_ratio"] = qtr["days_reported_quarter"] / qtr["days_in_quarter"].replace({0: np.nan})

    # Order and compute quarter gaps
    q_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    qtr["_qord"] = qtr["quarter"].map(q_order).astype("Int64")
    qtr["_qi"] = (qtr["year"].astype("Int64") * 4 + qtr["_qord"]).astype("Int32")

    qtr = qtr.sort_values(["cms_certification_number", "_qi"], kind="mergesort")

    qtr["gap_from_prev_quarters"] = (
        qtr.groupby("cms_certification_number")["_qi"]
        .diff()
        .fillna(1)
        .astype("Int16")
        - 1
    ).clip(lower=0)

    # ---------------- Quarterly QA / plausibility flags
    qtr["pbj_partial_quarter"] = (qtr["months_observed_in_quarter"] < 3).astype("Int8")
    qtr["pbj_low_coverage"] = (qtr["coverage_ratio"] < 0.80).fillna(False).astype("Int8")
    qtr["pbj_zero_rn_lpn"] = (((qtr["rn_hprd"] == 0) & (qtr["lpn_hprd"] == 0))).fillna(False).astype("Int8")
    qtr["pbj_implausible_hprd"] = (
        (
            (qtr["total_hprd"] < 1.5)
            | (qtr["total_hprd"] > 12)
            | (qtr["cna_hprd"] > 5.25)
        )
    ).fillna(False).astype("Int8")

    # Optional hard impossible-row flags
    qtr["pbj_invalid_quarter"] = (
        (
            qtr["resident_days_quarter"].isna()
            | (qtr["resident_days_quarter"] <= 0)
            | qtr["days_in_quarter"].isna()
            | (qtr["days_in_quarter"] <= 0)
            | qtr["coverage_ratio"].isna()
            | (qtr["coverage_ratio"] < 0)
            | (qtr["coverage_ratio"] > 1.01)
        )
    ).astype("Int8")

    # Final ordering / casts
    float_cols = [
        "rn_hours_quarter",
        "lpn_hours_quarter",
        "cna_hours_quarter",
        "rndon_hours_quarter",
        "rnadmin_hours_quarter",
        "lpnadmin_hours_quarter",
        "natrn_hours_quarter",
        "medaide_hours_quarter",
        "total_hours_quarter",
        "resident_days_quarter",
        "avg_daily_census",
        "rn_hprd",
        "lpn_hprd",
        "cna_hprd",
        "total_hprd",
        "coverage_ratio",
    ]
    for col in float_cols:
        qtr[col] = pd.to_numeric(qtr[col], errors="coerce").astype("float32")

    for col in [
        "days_reported_quarter",
        "days_in_quarter",
        "months_observed_in_quarter",
        "gap_from_prev_quarters",
        "pbj_partial_quarter",
        "pbj_low_coverage",
        "pbj_zero_rn_lpn",
        "pbj_implausible_hprd",
        "pbj_invalid_quarter",
    ]:
        qtr[col] = pd.to_numeric(qtr[col], errors="coerce").astype("Int16")

    qtr = (
        qtr.sort_values(["cms_certification_number", "year", "_qord"], kind="mergesort")
        .drop(columns=["_qord", "_qi"])
        .reset_index(drop=True)
    )

    cfg.atomic_overwrite_csv(qtr, OUT_FP_QUARTERLY, index=False)

    print(f"[saved] quarterly pbj nurse panel → {OUT_FP_QUARTERLY} (rows={len(qtr):,})")
    print(
        f"[qa-quarterly] unique_ccn={qtr['cms_certification_number'].nunique(dropna=True):,}, "
        f"missing_rn_hprd={int(qtr['rn_hprd'].isna().sum()):,}, "
        f"missing_total_hprd={int(qtr['total_hprd'].isna().sum()):,}, "
        f"partial_qtrs={int(qtr['pbj_partial_quarter'].sum()):,}, "
        f"low_coverage_qtrs={int(qtr['pbj_low_coverage'].sum()):,}, "
        f"implausible_hprd_qtrs={int(qtr['pbj_implausible_hprd'].sum()):,}, "
        f"invalid_qtrs={int(qtr['pbj_invalid_quarter'].sum()):,}"
    )


# ============================== Main ==========================================
def main():
    if RUN_BUILD_MONTHLY:
        build_monthly_from_raw()
    else:
        print("[skip] monthly rebuild skipped")

    if RUN_BUILD_QUARTERLY:
        build_quarterly_from_monthly()
    else:
        print("[skip] quarterly build skipped")


if __name__ == "__main__":
    main()