#!/usr/bin/env python
# coding: utf-8
# -----------------------------------------------------------------------------
# Medicare Cost Reports -> Monthly Controls -> Quarterly Controls
#
# Notes:
# - Uses shared paths/helpers from config.py where applicable
# - Preserves existing monthly logic, formulas, corrections, and output names
# - Allows skipping monthly rebuild if mcr.csv already exists
# - Adds a quarterly output built from finalized monthly mcr.csv
# - Quarterly controls use within-quarter mode
# - Ties in the mode are broken by the last observed month in the quarter
# - Adds QA fields describing within-quarter agreement
# -----------------------------------------------------------------------------

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================== Paths =========================================
MCR_DIR = cfg.MCR_DIR
INTERIM_DIR = cfg.ensure_dir(cfg.INTERIM_DIR)
OUT_FP = INTERIM_DIR / "mcr.csv"
OUT_FP_QUARTERLY = INTERIM_DIR / "mcr_quarterly.csv"

RUN_BUILD_MONTHLY = True
RUN_BUILD_QUARTERLY = True

print(f"[paths] MCR_DIR={MCR_DIR}")
print(f"[out]   {OUT_FP}")
print(f"[out]   {OUT_FP_QUARTERLY}")
print(
    f"[flags] RUN_BUILD_MONTHLY={RUN_BUILD_MONTHLY}, "
    f"RUN_BUILD_QUARTERLY={RUN_BUILD_QUARTERLY}"
)

# ============================== Helpers =======================================
def map_ownership_bucket(code_str: str):
    if code_str is None or (isinstance(code_str, float) and pd.isna(code_str)):
        return None
    s = str(code_str).strip().upper().replace(".0", "")
    # 1-2 nonprofit; 3-6 for-profit; 7-13 government
    if s in {"1", "2"}:
        return "Nonprofit"
    if s in {"3", "4", "5", "6"}:
        return "For-profit"
    if s in {"7", "8", "9", "10", "11", "12", "13"}:
        return "Government"
    return None


def _share(n, d):
    return pd.to_numeric(100.0 * (n / d), errors="coerce")


def month_range_df(start, end):
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame({"month": []})
    s = pd.Period(start, "M").to_timestamp("s")
    e = pd.Period(end, "M").to_timestamp("s")
    if e < s:
        s, e = e, s
    months = pd.period_range(s, e, freq="M").to_timestamp("s")
    return pd.DataFrame({"month": months})


def norm_urban(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NA
    s = str(x).strip().upper()
    if s in {"U", "URBAN", "1", "YES", "Y", "TRUE", "T"}:
        return 1
    if s in {"R", "RURAL", "0", "NO", "N", "FALSE", "F", "2"}:
        return 0
    return pd.NA


def chain_from_homeoffice(val):
    """
    EXACT logic based on your data audit:
      'Y','YES','1','T','TRUE'  -> 1
      'N','NO','0','F','FALSE','' (blank) -> 0
      numeric nonzero -> 1, numeric zero/NA -> 0
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0
    s = str(val).strip().upper()
    if s in {"Y", "YES", "1", "T", "TRUE"}:
        return 1
    if s in {"N", "NO", "0", "F", "FALSE", ""}:
        return 0
    try:
        f = float(s)
        return int(f != 0.0)
    except Exception:
        return 0


def last_nonmissing(s: pd.Series):
    s2 = s.dropna()
    if len(s2) == 0:
        return pd.NA
    return s2.iloc[-1]


def mode_with_last_tiebreak(s: pd.Series, round_digits: int | None = None):
    """
    Return within-quarter mode.
    If multiple modes tie, use the last nonmissing observed value among the tied values.
    Optionally round numeric values first to stabilize floating-point behavior.
    """
    s2 = s.dropna()
    if len(s2) == 0:
        return pd.NA

    if round_digits is not None:
        s_work = pd.to_numeric(s2, errors="coerce").round(round_digits)
        s_work.index = s2.index
    else:
        s_work = s2.copy()

    counts = s_work.value_counts(dropna=True)
    if counts.empty:
        return pd.NA

    top_n = counts.iloc[0]
    top_vals = set(counts[counts == top_n].index.tolist())

    for val in reversed(s_work.tolist()):
        if val in top_vals:
            return val

    return s_work.iloc[-1]


def mode_share(s: pd.Series, round_digits: int | None = None):
    """
    Share of nonmissing monthly observations in the quarter equal to the chosen modal value.
    """
    s2 = s.dropna()
    if len(s2) == 0:
        return np.nan

    if round_digits is not None:
        s_work = pd.to_numeric(s2, errors="coerce").round(round_digits)
        s_work.index = s2.index
    else:
        s_work = s2.copy()

    counts = s_work.value_counts(dropna=True)
    if counts.empty:
        return np.nan

    return float(counts.iloc[0] / len(s_work))


def disagreement_flag(s: pd.Series, round_digits: int | None = None):
    """
    1 if more than one distinct nonmissing value appears within quarter, else 0.
    """
    s2 = s.dropna()
    if len(s2) == 0:
        return pd.NA

    if round_digits is not None:
        s2 = pd.to_numeric(s2, errors="coerce").round(round_digits)

    return int(s2.nunique(dropna=True) > 1)


# ============================== Load & Select =================================
# SAS; fallback to CSV
sas_files = sorted(MCR_DIR.glob("mcr_flatfile_20??.sas7bdat"))
xpt_files = sorted(MCR_DIR.glob("mcr_flatfile_20??.xpt"))
csv_files = sorted(MCR_DIR.glob("mcr_flatfile_20??.csv"))

use_sas = len(sas_files + xpt_files) > 0
try:
    import pyreadstat  # noqa: F401
except Exception:
    if use_sas:
        print("[read] pyreadstat not available; falling back to CSV-only")
    use_sas = False

TARGET_SETS = dict(
    PRVDR_NUM=["PRVDR_NUM", "provnum", "prvdr_num", "Provider Number"],
    FY_BGN_DT=["FY_BGN_DT", "fy_bgn_dt", "Cost Report Fiscal Year beginning date"],
    FY_END_DT=["FY_END_DT", "fy_end_dt", "Cost Report Fiscal Year ending date"],
    MRC_OWNERSHIP=["MRC_OWNERSHIP"],

    PAT_DAYS_TOT=["S3_1_patdays_total", "S3_1_PATDAYS_TOTAL"],
    PAT_DAYS_MCR=["S3_1_patdays_medicare", "S3_1_PATDAYS_MEDICARE"],
    PAT_DAYS_MCD=["S3_1_patdays_medicaid", "S3_1_PATDAYS_MEDICAID"],

    AVG_LOS_TOT=["S3_1_avg_los_total", "S3_1_AVG_LOS_TOTAL"],
    AVG_LOS_MCR=["S3_1_avg_los_medicare", "S3_1_AVG_LOS_MEDICARE"],
    AVG_LOS_MCD=["S3_1_avg_los_medicaid", "S3_1_AVG_LOS_MEDICAID"],

    BEDDAYS_AVAIL=["S3_1_beddays_aval", "S3_1_BEDDAYS_AVAL"],
    TOT_BEDS=["S3_1_beds", "S3_1_BEDS"],
    STATE=["MCR_STATE"],
    URBAN=["MCR_URBAN"],
    MCR_homeoffice=["MCR_homeoffice"],
)

def read_one_file(fp: Path) -> pd.DataFrame:
    if fp.suffix.lower() in {".sas7bdat", ".xpt"} and use_sas:
        if fp.suffix.lower() == ".sas7bdat":
            _, meta = pyreadstat.read_sas7bdat(fp, metadataonly=True)
        else:
            _, meta = pyreadstat.read_xport(fp, metadataonly=True)
        cols = list(meta.column_names)
    else:
        if fp.suffix.lower() == ".csv":
            head = pd.read_csv(fp, nrows=0, low_memory=False)
            cols = list(head.columns)
        else:
            fmt = "sas7bdat" if fp.suffix.lower() == ".sas7bdat" else "xport"
            head = pd.read_sas(fp, format=fmt, encoding="latin1", nrows=1)
            cols = list(head.columns)

    actual = {}
    for key, cand in TARGET_SETS.items():
        nm = cfg.find_col_case_insensitive(cols, cand)
        if nm is not None:
            actual[key] = nm

    essentials = ["PRVDR_NUM", "FY_BGN_DT", "FY_END_DT"]
    for k in essentials:
        if k not in actual:
            actual[k] = None

    select_cols_actual = [c for c in actual.values() if c is not None]

    if fp.suffix.lower() in {".sas7bdat", ".xpt"} and use_sas:
        if fp.suffix.lower() == ".sas7bdat":
            df, _ = pyreadstat.read_sas7bdat(
                fp, usecols=select_cols_actual, disable_datetime_conversion=0
            )
        else:
            df, _ = pyreadstat.read_xport(
                fp, usecols=select_cols_actual, disable_datetime_conversion=0
            )
    else:
        if fp.suffix.lower() == ".csv":
            df = pd.read_csv(
                fp,
                usecols=select_cols_actual if select_cols_actual else None,
                low_memory=False,
            )
        else:
            fmt = "sas7bdat" if fp.suffix.lower() == ".sas7bdat" else "xport"
            df = pd.read_sas(fp, format=fmt, encoding="latin1")
            if select_cols_actual:
                df = df[select_cols_actual].copy()

    rename_map = {actual[k]: k for k in actual if actual[k] is not None}
    df = df.rename(columns=rename_map)

    for k in TARGET_SETS.keys():
        if k not in df.columns:
            df[k] = pd.NA

    print(f"[read] {fp.name} rows={len(df):,} cols={len(df.columns)}")
    return df


def _ts_month_start(ym: str) -> pd.Timestamp:
    return pd.Period(ym, "M").to_timestamp("s")


def _ts_month_end(ym: str) -> pd.Timestamp:
    return pd.Period(ym, "M").to_timestamp("s") + pd.offsets.MonthEnd(0)


def apply_value_corrections(df: pd.DataFrame, corrections: list[dict]) -> int:
    """
    Each correction is a dict:
      {"ccn":"015417", "start":"2018/04", "end":"2019/04", "col":"num_beds", "value":75}
    Applied at the FY-row level if [FY_BGN_DT .. FY_END_DT] overlaps [start..end].
    Returns number of rows updated.
    """
    total = 0
    for item in corrections:
        ccn = str(item["ccn"])
        col = str(item["col"])
        start, end = item["start"], item["end"]
        val = item["value"]
        ccn_norm = cfg.normalize_ccn_any(pd.Series([ccn])).iloc[0]
        s_ts, e_ts = _ts_month_start(start), _ts_month_end(end)

        m = (
            df["cms_certification_number"].eq(ccn_norm)
            & df["FY_BGN_DT"].notna()
            & df["FY_END_DT"].notna()
            & (df["FY_BGN_DT"] <= e_ts)
            & (df["FY_END_DT"] >= s_ts)
        )
        if m.any():
            df.loc[m, col] = val
            total += int(m.sum())

    return total


def build_monthly_from_raw():
    files = sorted(sas_files + xpt_files + csv_files)
    if not files:
        raise FileNotFoundError(f"No MCR flatfiles found in {MCR_DIR}")

    frames = [read_one_file(fp) for fp in files]
    raw = pd.concat(frames, ignore_index=True, sort=False).copy()

    # ============================== Normalize & Types =============================
    raw["cms_certification_number"] = cfg.normalize_ccn_any(raw["PRVDR_NUM"])
    raw["FY_BGN_DT"] = pd.to_datetime(raw["FY_BGN_DT"], errors="coerce")
    raw["FY_END_DT"] = pd.to_datetime(raw["FY_END_DT"], errors="coerce")

    for c in [
        "PAT_DAYS_TOT",
        "PAT_DAYS_MCR",
        "PAT_DAYS_MCD",
        "AVG_LOS_TOT",
        "AVG_LOS_MCR",
        "AVG_LOS_MCD",
        "BEDDAYS_AVAIL",
        "TOT_BEDS",
    ]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw["ownership_type"] = raw["MRC_OWNERSHIP"].map(map_ownership_bucket)

    # State / urban
    raw["state"] = raw["STATE"].astype("string").str.strip().str.upper()
    raw.loc[raw["state"].isin(["", "NA", "NAN", "NONE"]), "state"] = pd.NA
    raw["urban"] = raw["URBAN"].apply(norm_urban).astype("Int8")

    # Chain ONLY from MCR_homeoffice
    raw["chain"] = raw["MCR_homeoffice"].apply(chain_from_homeoffice).astype("Int8")

    # Beds (from S3_1_BEDS only)
    raw["num_beds"] = raw["TOT_BEDS"]

    # ============================== OUTLIER / MANUAL CORRECTIONS ==================
    CORR_BEDS = [
        {"ccn": "015417", "start": "2018/04", "end": "2019/04", "col": "num_beds", "value": 75},
        {"ccn": "056337", "start": "2017/01", "end": "2017/12", "col": "num_beds", "value": 142},
        {"ccn": "105782", "start": "2017/01", "end": "2017/12", "col": "num_beds", "value": 120},
        {"ccn": "135080", "start": "2018/01", "end": "2019/12", "col": "num_beds", "value": 60},
        {"ccn": "145816", "start": "2024/01", "end": "2024/06", "col": "num_beds", "value": 203},
        {"ccn": "235022", "start": "2019/01", "end": "2019/12", "col": "num_beds", "value": 92},
        {"ccn": "235157", "start": "2021/01", "end": "2021/12", "col": "num_beds", "value": 104},
        {"ccn": "235638", "start": "2019/01", "end": "2019/12", "col": "num_beds", "value": 77},
        {"ccn": "425129", "start": "2020/04", "end": "2020/12", "col": "num_beds", "value": 108},
    ]

    updated_beds = apply_value_corrections(raw, CORR_BEDS)
    print(f"[beds corrections] updated FY rows: {updated_beds}")

    CORR_INPUTS = []
    updated_inputs = apply_value_corrections(raw, CORR_INPUTS)
    if updated_inputs:
        print(f"[input corrections] updated FY rows: {updated_inputs}")

    # Occupancy rate (primary + fallback)
    fy_days = (raw["FY_END_DT"] - raw["FY_BGN_DT"]).dt.days.add(1).where(lambda s: s > 0)

    primary = np.where(
        raw["PAT_DAYS_TOT"].notna()
        & raw["BEDDAYS_AVAIL"].notna()
        & (raw["BEDDAYS_AVAIL"] > 0),
        (raw["PAT_DAYS_TOT"] / raw["BEDDAYS_AVAIL"]) * 100.0,
        np.nan,
    )

    fallback = np.where(
        raw["PAT_DAYS_TOT"].notna()
        & raw["num_beds"].notna()
        & fy_days.notna()
        & (raw["num_beds"] * fy_days > 0),
        (raw["PAT_DAYS_TOT"] / (raw["num_beds"] * fy_days)) * 100.0,
        np.nan,
    )

    raw["occupancy_rate"] = pd.to_numeric(
        np.where(np.isnan(primary), fallback, primary),
        errors="coerce",
    ).clip(0, 100)

    # ---------------- Spare capacity ----------------
    # (certified beds - residents) / certified beds, using TOTAL CERTIFIED BEDS
    # (S3_1_beds / num_beds) as the denominator -- NOT beddays_available.
    # "Residents" = average daily census implied by the cost report
    # (patient days / calendar days in the fiscal year), since MCR does not
    # report a daily census directly. This differs from occupancy_rate
    # whenever BEDDAYS_AVAIL != num_beds * fy_days (e.g. beds added/removed
    # mid-year), which is why it is NOT simply 1 - occupancy_rate/100.
    avg_daily_census_fy = np.where(
        raw["PAT_DAYS_TOT"].notna() & fy_days.notna() & (fy_days > 0),
        raw["PAT_DAYS_TOT"] / fy_days,
        np.nan,
    )

    raw["spare_capacity"] = pd.to_numeric(
        np.where(
            raw["num_beds"].notna() & (raw["num_beds"] > 0) & ~np.isnan(avg_daily_census_fy),
            (raw["num_beds"] - avg_daily_census_fy) / raw["num_beds"],
            np.nan,
        ),
        errors="coerce",
    ).clip(0, 1)

    # Shares
    raw["pct_medicare"] = _share(raw["PAT_DAYS_MCR"], raw["PAT_DAYS_TOT"]).clip(0, 100)
    raw["pct_medicaid"] = _share(raw["PAT_DAYS_MCD"], raw["PAT_DAYS_TOT"]).clip(0, 100)
    # Average length of stay
    raw["avg_los_total"] = pd.to_numeric(raw["AVG_LOS_TOT"], errors="coerce")
    raw["avg_los_medicare"] = pd.to_numeric(raw["AVG_LOS_MCR"], errors="coerce")
    raw["avg_los_medicaid"] = pd.to_numeric(raw["AVG_LOS_MCD"], errors="coerce")

    # Optional sanity bounds. Keep broad bounds so we do not over-clean.
    for c in ["avg_los_total", "avg_los_medicare", "avg_los_medicaid"]:
        raw[c] = raw[c].where(raw[c].between(0, 5000))
    
    CORR_DERIVED = []
    updated_derived = apply_value_corrections(raw, CORR_DERIVED)
    if updated_derived:
        print(f"[derived corrections] updated FY rows: {updated_derived}")

    # ============================== Expand to Monthly =============================
    eligible = raw.dropna(subset=["cms_certification_number", "FY_BGN_DT", "FY_END_DT"]).copy()

    rows = []
    for r in eligible.itertuples(index=False):
        months = month_range_df(r.FY_BGN_DT, r.FY_END_DT)
        if months.empty:
            continue

        block = months.copy()
        block["cms_certification_number"] = getattr(r, "cms_certification_number")
        block["state"] = getattr(r, "state", pd.NA)
        block["urban"] = getattr(r, "urban", pd.NA)
        block["chain"] = getattr(r, "chain", pd.NA)
        block["num_beds"] = getattr(r, "num_beds", np.nan)
        block["occupancy_rate"] = getattr(r, "occupancy_rate", np.nan)
        block["spare_capacity"] = getattr(r, "spare_capacity", np.nan)
        block["pct_medicare"] = getattr(r, "pct_medicare", np.nan)
        block["pct_medicaid"] = getattr(r, "pct_medicaid", np.nan)
        block["avg_los_total"] = getattr(r, "avg_los_total", np.nan)
        block["avg_los_medicare"] = getattr(r, "avg_los_medicare", np.nan)
        block["avg_los_medicaid"] = getattr(r, "avg_los_medicaid", np.nan)
        block["ownership_type"] = getattr(r, "ownership_type", None)
        rows.append(block)

    monthly = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "cms_certification_number",
                "month",
                "state",
                "urban",
                "chain",
                "num_beds",
                "occupancy_rate",
                "spare_capacity",
                "pct_medicare",
                "pct_medicaid",
                "avg_los_total",
                "avg_los_medicare",
                "avg_los_medicaid",
                "ownership_type",
            ]
        )
    )

    # Deduplicate overlaps within CCN×month
    monthly = (
        monthly.sort_values(["cms_certification_number", "month"])
        .groupby(["cms_certification_number", "month"], as_index=False)
        .agg(
            {
                "state": lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
                "urban": "max",
                "chain": "max",
                "num_beds": "mean",
                "occupancy_rate": "mean",
                "spare_capacity": "mean",
                "pct_medicare": "mean",
                "pct_medicaid": "mean",
                "avg_los_total": "mean",
                "avg_los_medicare": "mean",
                "avg_los_medicaid": "mean",
                "ownership_type": lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
            }
        )
    )

    # Ownership dummies (For-profit is reference)
    ot = monthly["ownership_type"].astype("string").str.strip().str.lower()
    ot = (
        ot.str.replace(r"[\s_]+", "-", regex=True)
        .str.replace(r"^non[- ]?profit$", "nonprofit", regex=True)
        .str.replace(r"^for[- ]?profit$", "for-profit", regex=True)
    )

    monthly["non_profit"] = ot.eq("nonprofit").astype("Int8")
    monthly["government"] = ot.eq("government").astype("Int8")

    # Period labels
    monthly["year_month"] = monthly["month"].dt.strftime("%Y/%m")
    monthly["quarter"] = monthly["month"].dt.to_period("Q").astype(str).str.replace("Q", "Q", regex=False)

    # Coerce numeric ranges/types
    for c in ["pct_medicare", "pct_medicaid", "occupancy_rate"]:
        monthly[c] = pd.to_numeric(monthly[c], errors="coerce").clip(0, 100)

    if "spare_capacity" in monthly.columns:
        monthly["spare_capacity"] = pd.to_numeric(monthly["spare_capacity"], errors="coerce").clip(0, 1)
    
    for c in ["avg_los_total", "avg_los_medicare", "avg_los_medicaid"]:
        monthly[c] = pd.to_numeric(monthly[c], errors="coerce")

        monthly["num_beds"] = pd.to_numeric(monthly["num_beds"], errors="coerce")

    # Reorder
    keep = [
        "cms_certification_number",
        "quarter",
        "year_month",
        "pct_medicare",
        "pct_medicaid",
        "avg_los_total",
        "avg_los_medicare",
        "avg_los_medicaid",
        "num_beds",
        "occupancy_rate",
        "spare_capacity",
        "urban",
        "chain",
        "state",
        "non_profit",
        "government",
    ]
    monthly = monthly[keep].sort_values(["cms_certification_number", "year_month"]).reset_index(drop=True)

    # Save monthly
    cfg.atomic_overwrite_csv(monthly, OUT_FP, index=False)
    print(
        f"[save] controls → {OUT_FP}  rows={len(monthly):,}  "
        f"CCNs={monthly['cms_certification_number'].nunique():,}"
    )
    print(monthly.head(10).to_string(index=False))


def build_quarterly_from_monthly():
    if not OUT_FP.exists():
        raise FileNotFoundError(f"Monthly MCR panel not found: {OUT_FP}")

    monthly = pd.read_csv(
        OUT_FP,
        dtype={"cms_certification_number": "string", "state": "string"},
        low_memory=False
    )

    if monthly.empty:
        cfg.atomic_overwrite_csv(monthly, OUT_FP_QUARTERLY, index=False)
        print(f"[save] quarterly mcr controls → {OUT_FP_QUARTERLY}  (rows=0)")
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

    # enforce numeric types
    for col in [
        "pct_medicare",
        "pct_medicaid",
        "avg_los_total",
        "avg_los_medicare",
        "avg_los_medicaid",
        "num_beds",
        "occupancy_rate",
        "spare_capacity",
    ]:
        if col in monthly.columns:
            monthly[col] = pd.to_numeric(monthly[col], errors="coerce")

    for col in ["urban", "chain", "non_profit", "government"]:
        if col in monthly.columns:
            monthly[col] = pd.to_numeric(monthly[col], errors="coerce").astype("Int8")

    if "state" in monthly.columns:
        monthly["state"] = monthly["state"].astype("string")

    monthly = monthly.sort_values(
        ["cms_certification_number", "year", "quarter_num", "_ord"],
        kind="mergesort"
    )

    grp = ["cms_certification_number", "year", "quarter"]

    qtr = (
        monthly.groupby(grp, sort=False)
        .agg(
            pct_medicare=("pct_medicare", lambda s: mode_with_last_tiebreak(s, round_digits=4)),
            pct_medicaid=("pct_medicaid", lambda s: mode_with_last_tiebreak(s, round_digits=4)),
            avg_los_total=("avg_los_total", lambda s: mode_with_last_tiebreak(s, round_digits=4)),
            avg_los_medicare=("avg_los_medicare", lambda s: mode_with_last_tiebreak(s, round_digits=4)),
            avg_los_medicaid=("avg_los_medicaid", lambda s: mode_with_last_tiebreak(s, round_digits=4)),
            num_beds=("num_beds", lambda s: mode_with_last_tiebreak(s, round_digits=0)),
            occupancy_rate=("occupancy_rate", lambda s: mode_with_last_tiebreak(s, round_digits=4)),
            spare_capacity=("spare_capacity", lambda s: mode_with_last_tiebreak(s, round_digits=4)),
            urban=("urban", mode_with_last_tiebreak),
            chain=("chain", mode_with_last_tiebreak),
            state=("state", mode_with_last_tiebreak),
            non_profit=("non_profit", mode_with_last_tiebreak),
            government=("government", mode_with_last_tiebreak),

            months_observed_in_quarter=("year_month", "nunique"),
            last_year_month_in_quarter=("year_month", "last"),

            pct_medicare_mode_share=("pct_medicare", lambda s: mode_share(s, round_digits=4)),
            pct_medicaid_mode_share=("pct_medicaid", lambda s: mode_share(s, round_digits=4)),
            avg_los_total_mode_share=("avg_los_total", lambda s: mode_share(s, round_digits=4)),
            avg_los_medicare_mode_share=("avg_los_medicare", lambda s: mode_share(s, round_digits=4)),
            avg_los_medicaid_mode_share=("avg_los_medicaid", lambda s: mode_share(s, round_digits=4)),
            num_beds_mode_share=("num_beds", lambda s: mode_share(s, round_digits=0)),
            occupancy_rate_mode_share=("occupancy_rate", lambda s: mode_share(s, round_digits=4)),
            spare_capacity_mode_share=("spare_capacity", lambda s: mode_share(s, round_digits=4)),
            urban_mode_share=("urban", mode_share),
            chain_mode_share=("chain", mode_share),
            state_mode_share=("state", mode_share),
            non_profit_mode_share=("non_profit", mode_share),
            government_mode_share=("government", mode_share),

            pct_medicare_disagreement=("pct_medicare", lambda s: disagreement_flag(s, round_digits=4)),
            pct_medicaid_disagreement=("pct_medicaid", lambda s: disagreement_flag(s, round_digits=4)),
            avg_los_total_disagreement=("avg_los_total", lambda s: disagreement_flag(s, round_digits=4)),
            avg_los_medicare_disagreement=("avg_los_medicare", lambda s: disagreement_flag(s, round_digits=4)),
            avg_los_medicaid_disagreement=("avg_los_medicaid", lambda s: disagreement_flag(s, round_digits=4)),
            num_beds_disagreement=("num_beds", lambda s: disagreement_flag(s, round_digits=0)),
            occupancy_rate_disagreement=("occupancy_rate", lambda s: disagreement_flag(s, round_digits=4)),
            spare_capacity_disagreement=("spare_capacity", lambda s: disagreement_flag(s, round_digits=4)),
            urban_disagreement=("urban", disagreement_flag),
            chain_disagreement=("chain", disagreement_flag),
            state_disagreement=("state", disagreement_flag),
            non_profit_disagreement=("non_profit", disagreement_flag),
            government_disagreement=("government", disagreement_flag),
        )
        .reset_index()
    )

    # cleanup types
    for col in ["pct_medicare", "pct_medicaid", "num_beds", "occupancy_rate"]:
        if col in qtr.columns:
            qtr[col] = pd.to_numeric(qtr[col], errors="coerce")

    if "spare_capacity" in qtr.columns:
        qtr["spare_capacity"] = pd.to_numeric(qtr["spare_capacity"], errors="coerce").clip(0, 1)

    for col in ["urban", "chain", "non_profit", "government"]:
        if col in qtr.columns:
            qtr[col] = pd.to_numeric(qtr[col], errors="coerce").fillna(0).astype("Int8")

    if "state" in qtr.columns:
        qtr["state"] = qtr["state"].astype("string")

    qtr["months_observed_in_quarter"] = pd.to_numeric(
        qtr["months_observed_in_quarter"], errors="coerce"
    ).astype("Int16")

    for col in [
        "pct_medicare_mode_share",
        "pct_medicaid_mode_share",
        "num_beds_mode_share",
        "occupancy_rate_mode_share",
        "spare_capacity_mode_share",
        "urban_mode_share",
        "chain_mode_share",
        "state_mode_share",
        "non_profit_mode_share",
        "government_mode_share",
    ]:
        if col in qtr.columns:
            qtr[col] = pd.to_numeric(qtr[col], errors="coerce").astype("float32")

    for col in [
        "pct_medicare_disagreement",
        "pct_medicaid_disagreement",
        "num_beds_disagreement",
        "occupancy_rate_disagreement",
        "spare_capacity_disagreement",
        "urban_disagreement",
        "chain_disagreement",
        "state_disagreement",
        "non_profit_disagreement",
        "government_disagreement",
    ]:
        if col in qtr.columns:
            qtr[col] = pd.to_numeric(qtr[col], errors="coerce").astype("Int8")

    q_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    qtr["_qord"] = qtr["quarter"].map(q_order)

    qtr = (
        qtr.sort_values(["cms_certification_number", "year", "_qord"], kind="mergesort")
        .drop(columns=["_qord"])
        .reset_index(drop=True)
    )

    cfg.atomic_overwrite_csv(qtr, OUT_FP_QUARTERLY, index=False)

    print(f"[save] quarterly mcr controls → {OUT_FP_QUARTERLY}  ({len(qtr):,} rows)")
    print(
        "[qa-quarterly] unique_ccn="
        f"{qtr['cms_certification_number'].nunique(dropna=True):,}, "
        f"missing_num_beds={int(qtr['num_beds'].isna().sum()) if 'num_beds' in qtr.columns else 0:,}, "
        f"missing_occupancy_rate={int(qtr['occupancy_rate'].isna().sum()) if 'occupancy_rate' in qtr.columns else 0:,}, "
        f"missing_spare_capacity={int(qtr['spare_capacity'].isna().sum()) if 'spare_capacity' in qtr.columns else 0:,}"
    )


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