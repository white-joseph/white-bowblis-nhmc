#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# PBJ Nurse Staffing —> Normalize -> Monthly Aggregate -> Combine -> Quarterly
#
# Updated version:
# - preserves the existing monthly PBJ pipeline
# - adds a quarterly PBJ output built from the finalized monthly pbj_nurse.csv
# - quarterly staffing/intensity measures are recomputed from quarterly totals
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
RUN_BUILD_QUARTERLY = True

print(f"[paths] PBJ_DIR={PBJ_DIR}")
print(f"[paths] OUT_FP={OUT_FP}")
print(f"[paths] OUT_FP_QUARTERLY={OUT_FP_QUARTERLY}")
print(f"[flags] RUN_BUILD_QUARTERLY={RUN_BUILD_QUARTERLY}")

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
def normalize_needed_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "provnum" in df.columns and "cms_certification_number" not in df.columns:
        df.rename(columns={"provnum": "cms_certification_number"}, inplace=True)
    if "mdscensus" in df.columns and "mds_census" not in df.columns:
        df.rename(columns={"mdscensus": "mds_census"}, inplace=True)

    for col in ["hrs_rn", "hrs_lpn", "hrs_cna"]:
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

    for c in ["hrs_rn", "hrs_lpn", "hrs_cna"]:
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
            "hrs_rn",
            "hrs_lpn",
            "hrs_cna",
            "mds_census",
            "cy_qtr",
        ]
    ]


# ====================== File -> Monthly Aggregation ============================
def process_file_monthly(fp: Path) -> pd.DataFrame:
    df = normalize_needed_columns(read_pbj_csv(fp))
    df["quarter_row"] = normalize_cy_qtr(df["cy_qtr"], df["workdate"])

    # Daily
    daily = (
        df.groupby(["cms_certification_number", "workdate"], as_index=False)
        .agg(
            hrs_rn=("hrs_rn", "sum"),
            hrs_lpn=("hrs_lpn", "sum"),
            hrs_cna=("hrs_cna", "sum"),
            mds_census=("mds_census", "mean"),
            quarter=("quarter_row", "first"),
        )
    )

    daily["total_hours"] = daily[["hrs_rn", "hrs_lpn", "hrs_cna"]].sum(axis=1).astype("float32")
    daily["year_month_p"] = daily["workdate"].dt.to_period("M")
    daily["days_in_mo"] = daily["workdate"].dt.days_in_month

    # Monthly
    monthly = (
        daily.groupby(["cms_certification_number", "year_month_p"], as_index=False)
        .agg(
            rn_hours_month=("hrs_rn", "sum"),
            lpn_hours_month=("hrs_lpn", "sum"),
            cna_hours_month=("hrs_cna", "sum"),
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

    monthly["year"] = monthly["_ord"].dt.year.astype("Int64")
    monthly["quarter_num"] = ((monthly["_ord"].dt.month - 1) // 3 + 1).astype("Int64")
    monthly["quarter"] = "Q" + monthly["quarter_num"].astype(str)

    numeric_cols = [
        "rn_hours_month",
        "lpn_hours_month",
        "cna_hours_month",
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

    # Final ordering / casts
    float_cols = [
        "rn_hours_quarter",
        "lpn_hours_quarter",
        "cna_hours_quarter",
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

    for col in ["days_reported_quarter", "days_in_quarter", "months_observed_in_quarter", "gap_from_prev_quarters"]:
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
        f"missing_total_hprd={int(qtr['total_hprd'].isna().sum()):,}"
    )


# ============================== Main ==========================================
def main():
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
        if RUN_BUILD_QUARTERLY:
            build_quarterly_from_monthly()
        return

    cols = [
        "cms_certification_number",
        "quarter",
        "year_month",
        *(["rn_hours_month", "lpn_hours_month", "cna_hours_month", "total_hours"] if KEEP_HOUR_TOTALS else []),
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

    if RUN_BUILD_QUARTERLY:
        build_quarterly_from_monthly()


if __name__ == "__main__":
    main()