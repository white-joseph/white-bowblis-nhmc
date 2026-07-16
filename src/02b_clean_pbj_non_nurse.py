#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# PBJ Non-Nurse Staffing (PT/OT/Speech) —> Normalize -> Monthly Aggregate ->
# Combine -> Quarterly
#
# Mirrors 02_clean_pbj_nurse.py's structure exactly, applied to the 7
# non-nurse categories requested:
#   - Physical Therapist (PT)
#   - Physical Therapy Assistant (PT assistant)
#   - Physical Therapy Aide (PT aide)
#   - Occupational Therapist (OT)
#   - Occupational Therapy Assistant (OT assistant)
#   - Occupational Therapy Aide (OT aide)
#   - Speech/Language Pathologist (SLP)
#
# NOTE: Per project decision, this uses TOTAL hours only (Hrs_PT, Hrs_OT,
# etc.) -- NOT the employee/contract split (Hrs_PT_emp / Hrs_PT_ctr) that is
# also available in the raw PBJ non-nurse file. Employee-vs-contract and
# wage-schedule analysis is deferred to a later pass per project decision
# (too many moving pieces right now).
#
# NOTE on schema: CMS's PBJ non-nurse data dictionary changes slightly
# between 2022-Q4 and 2023-Q1 (an "Hrs_Admin_fn" footnote column is replaced
# by an "incomplete" flag column). This does not affect the core Hrs_*
# columns used here, so no version branching is needed -- but column
# presence is still checked defensively (fill 0.0 if missing) in case of
# any other quarter-to-quarter naming drift.
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
PBJ_NON_NURSE_DIR = cfg.PBJ_NON_NURSE_DIR
PBJ_GLOB = "pbj_non_nurse_????_Q[1-4].csv"

INTERIM_DIR = cfg.ensure_dir(cfg.INTERIM_DIR)
OUT_FP = INTERIM_DIR / "pbj_non_nurse.csv"
OUT_FP_QUARTERLY = INTERIM_DIR / "pbj_non_nurse_quarterly.csv"

KEEP_HOUR_TOTALS = True

# The 7 requested categories: (raw column suffix, output prefix)
CATEGORIES = [
    ("pt",           "pt"),        # Physical Therapist
    ("ptasst",       "ptasst"),    # Physical Therapy Assistant
    ("ptaide",       "ptaide"),    # Physical Therapy Aide
    ("ot",           "ot"),        # Occupational Therapist
    ("otasst",       "otasst"),    # Occupational Therapy Assistant
    ("otaide",       "otaide"),    # Occupational Therapy Aide
    ("spclangpath",  "slp"),       # Speech/Language Pathologist
]
RAW_HRS_COLS = [f"hrs_{suffix}" for suffix, _ in CATEGORIES]
OUT_PREFIXES = [prefix for _, prefix in CATEGORIES]

# Run flags
RUN_BUILD_MONTHLY = True
RUN_BUILD_QUARTERLY = True

print(f"[paths] PBJ_NON_NURSE_DIR={PBJ_NON_NURSE_DIR}")
print(f"[paths] OUT_FP={OUT_FP}")
print(f"[paths] OUT_FP_QUARTERLY={OUT_FP_QUARTERLY}")
print(
    f"[flags] RUN_BUILD_MONTHLY={RUN_BUILD_MONTHLY}, "
    f"RUN_BUILD_QUARTERLY={RUN_BUILD_QUARTERLY}"
)

# ============================== Helpers ======================================
def to_date_from_int_yyyymmdd(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("Int64"), format="%Y%m%d", errors="coerce")


# -------- vectorized CY_QTR parsing (identical to 02_clean_pbj_nurse.py) ------
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
def normalize_needed_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "provnum" in df.columns and "cms_certification_number" not in df.columns:
        df.rename(columns={"provnum": "cms_certification_number"}, inplace=True)
    if "mdscensus" in df.columns and "mds_census" not in df.columns:
        df.rename(columns={"mdscensus": "mds_census"}, inplace=True)

    # Defensive: fill any missing category column with 0.0 (handles any
    # quarter-to-quarter naming drift in the raw files).
    for col in RAW_HRS_COLS:
        if col not in df.columns:
            df[col] = 0.0

    if "cms_certification_number" not in df.columns:
        raise ValueError("Missing cms_certification_number/provnum")
    df["cms_certification_number"] = cfg.normalize_ccn_any(df["cms_certification_number"])

    if "workdate" not in df.columns:
        raise ValueError("Missing workdate column")
    if pd.api.types.is_integer_dtype(df["workdate"]) or pd.api.types.is_string_dtype(df["workdate"]):
        df["workdate"] = to_date_from_int_yyyymmdd(df["workdate"])
    else:
        df["workdate"] = pd.to_datetime(df["workdate"], errors="coerce")

    for c in RAW_HRS_COLS:
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
            *RAW_HRS_COLS,
            "mds_census",
            "cy_qtr",
        ]
    ]


# ====================== File -> Monthly Aggregation ============================
def process_file_monthly(fp: Path) -> pd.DataFrame:
    df = normalize_needed_columns(read_pbj_csv(fp))
    df["quarter_row"] = normalize_cy_qtr(df["cy_qtr"], df["workdate"])

    # Daily
    daily_agg = {c: (c, "sum") for c in RAW_HRS_COLS}
    daily = (
        df.groupby(["cms_certification_number", "workdate"], as_index=False)
        .agg(
            **daily_agg,
            mds_census=("mds_census", "mean"),
            quarter=("quarter_row", "first"),
        )
    )

    daily["total_hours"] = daily[RAW_HRS_COLS].sum(axis=1).astype("float32")
    daily["year_month_p"] = daily["workdate"].dt.to_period("M")
    daily["days_in_mo"] = daily["workdate"].dt.days_in_month

    # Monthly
    monthly_agg = {f"{prefix}_hours_month": (f"hrs_{suffix}", "sum") for suffix, prefix in CATEGORIES}
    monthly = (
        daily.groupby(["cms_certification_number", "year_month_p"], as_index=False)
        .agg(
            **monthly_agg,
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
    for prefix in OUT_PREFIXES:
        monthly[f"{prefix}_hprd"] = monthly[f"{prefix}_hours_month"] / denom
    monthly["total_hprd"] = monthly["total_hours"] / denom

    # year_month as 'YYYY/MM'
    ym = monthly["year_month_p"].astype("period[M]")
    monthly["year_month"] = (
        ym.dt.year.astype(int).astype(str)
        + "/"
        + ym.dt.month.astype(int).astype(str).str.zfill(2)
    )

    # Casts
    numeric_out_cols = (
        [f"{prefix}_hours_month" for prefix in OUT_PREFIXES]
        + ["total_hours", "resident_days", "avg_daily_census"]
        + [f"{prefix}_hprd" for prefix in OUT_PREFIXES]
        + ["total_hprd", "coverage_ratio"]
    )
    for c in numeric_out_cols:
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
    files = sorted(PBJ_NON_NURSE_DIR.glob(PBJ_GLOB))
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
        print(f"[saved] pbj non-nurse panel → {OUT_FP} (rows=0)")
        return

    hours_cols = [f"{prefix}_hours_month" for prefix in OUT_PREFIXES] + ["total_hours"]
    hprd_cols = [f"{prefix}_hprd" for prefix in OUT_PREFIXES] + ["total_hprd"]

    cols = [
        "cms_certification_number",
        "quarter",
        "year_month",
        *(hours_cols if KEEP_HOUR_TOTALS else []),
        "resident_days",
        "avg_daily_census",
        *hprd_cols,
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

    print(f"[saved] pbj non-nurse panel → {OUT_FP} (rows={len(monthly):,})")
    print(
        f"[qa] files_read={len(files):,}, "
        f"failed_files={failed:,}, "
        f"unique_ccn={monthly['cms_certification_number'].nunique(dropna=True):,}"
    )


# ======================= Quarterly builder from monthly ========================
def build_quarterly_from_monthly():
    if not OUT_FP.exists():
        raise FileNotFoundError(f"Monthly PBJ non-nurse panel not found: {OUT_FP}")

    monthly = pd.read_csv(
        OUT_FP,
        dtype={"cms_certification_number": "string"},
        low_memory=False
    )

    if monthly.empty:
        cfg.atomic_overwrite_csv(monthly, OUT_FP_QUARTERLY, index=False)
        print(f"[saved] quarterly pbj non-nurse panel → {OUT_FP_QUARTERLY} (rows=0)")
        return

    monthly["_ord"] = pd.to_datetime(
        monthly["year_month"] + "/01",
        format="%Y/%m/%d",
        errors="coerce"
    )

    monthly = monthly.dropna(subset=["cms_certification_number", "year_month", "_ord"]).copy()

    hours_cols = [f"{prefix}_hours_month" for prefix in OUT_PREFIXES] + ["total_hours"]
    hprd_cols = [f"{prefix}_hprd" for prefix in OUT_PREFIXES] + ["total_hprd"]

    # ---------------- Light monthly validity cleaning BEFORE quarterly aggregation
    for col in (
        hours_cols
        + ["resident_days", "avg_daily_census", "days_reported", "days_in_month",
           "coverage_ratio", "gap_from_prev_months"]
    ):
        if col in monthly.columns:
            monthly[col] = pd.to_numeric(monthly[col], errors="coerce")

    before_light = len(monthly)

    valid_mask = pd.Series(True, index=monthly.index)

    valid_mask &= monthly["cms_certification_number"].notna()
    valid_mask &= monthly["year_month"].notna()
    valid_mask &= monthly["_ord"].notna()

    # nonnegative monthly quantities
    for col in hours_cols + ["resident_days", "days_reported"]:
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

    numeric_cols = hours_cols + [
        "resident_days",
        "avg_daily_census",
        "days_reported",
        "days_in_month",
        "coverage_ratio",
        "gap_from_prev_months",
    ] + hprd_cols
    for col in numeric_cols:
        if col in monthly.columns:
            monthly[col] = pd.to_numeric(monthly[col], errors="coerce")

    monthly = monthly.sort_values(
        ["cms_certification_number", "year", "quarter_num", "_ord"],
        kind="mergesort"
    )

    grp = ["cms_certification_number", "year", "quarter"]

    quarter_hours_agg = {f"{prefix}_hours_quarter": (f"{prefix}_hours_month", "sum") for prefix in OUT_PREFIXES}
    qtr = (
        monthly.groupby(grp, sort=False)
        .agg(
            **quarter_hours_agg,
            total_hours_quarter=("total_hours", "sum"),
            resident_days_quarter=("resident_days", "sum"),
            days_reported_quarter=("days_reported", "sum"),
            days_in_quarter=("days_in_month", "sum"),
            months_observed_in_quarter=("year_month", "nunique"),
            last_year_month_in_quarter=("year_month", "last"),
        )
        .reset_index()
    )

    # Recompute quarterly HPRD from quarterly totals
    denom = qtr["resident_days_quarter"].replace({0: np.nan})
    for prefix in OUT_PREFIXES:
        qtr[f"{prefix}_hprd"] = qtr[f"{prefix}_hours_quarter"] / denom
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

    # ---------------- Quarterly QA / plausibility flags (structural only --
    # no domain-specific implausible-value thresholds for therapy HPRD yet,
    # since (unlike nursing) there's no established norm to bound against).
    qtr["pbj_partial_quarter"] = (qtr["months_observed_in_quarter"] < 3).astype("Int8")
    qtr["pbj_low_coverage"] = (qtr["coverage_ratio"] < 0.80).fillna(False).astype("Int8")

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
    hours_quarter_cols = [f"{prefix}_hours_quarter" for prefix in OUT_PREFIXES] + ["total_hours_quarter"]
    hprd_quarter_cols = [f"{prefix}_hprd" for prefix in OUT_PREFIXES] + ["total_hprd"]

    float_cols = hours_quarter_cols + [
        "resident_days_quarter",
        "avg_daily_census",
        "coverage_ratio",
    ] + hprd_quarter_cols
    for col in float_cols:
        qtr[col] = pd.to_numeric(qtr[col], errors="coerce").astype("float32")

    for col in [
        "days_reported_quarter",
        "days_in_quarter",
        "months_observed_in_quarter",
        "gap_from_prev_quarters",
        "pbj_partial_quarter",
        "pbj_low_coverage",
        "pbj_invalid_quarter",
    ]:
        qtr[col] = pd.to_numeric(qtr[col], errors="coerce").astype("Int16")

    qtr = (
        qtr.sort_values(["cms_certification_number", "year", "_qord"], kind="mergesort")
        .drop(columns=["_qord", "_qi"])
        .reset_index(drop=True)
    )

    cfg.atomic_overwrite_csv(qtr, OUT_FP_QUARTERLY, index=False)

    print(f"[saved] quarterly pbj non-nurse panel → {OUT_FP_QUARTERLY} (rows={len(qtr):,})")
    print(
        f"[qa-quarterly] unique_ccn={qtr['cms_certification_number'].nunique(dropna=True):,}, "
        f"missing_total_hprd={int(qtr['total_hprd'].isna().sum()):,}, "
        f"partial_qtrs={int(qtr['pbj_partial_quarter'].sum()):,}, "
        f"low_coverage_qtrs={int(qtr['pbj_low_coverage'].sum()):,}, "
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
