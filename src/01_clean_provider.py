#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# CMS Provider Info —> Extract -> Standardize -> Combine -> Quarterly
#
# Updated version:
# - preserves the existing monthly provider pipeline
# - adds a quarterly provider output built from the finalized monthly provider.csv
# - quarterly values use the last nonmissing month in the quarter within facility
# =============================================================================

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

# ============================== Config / Paths ================================
NH_ZIP_DIR = cfg.NH_COMPARE_DIR
PROV_DIR = cfg.ensure_dir(cfg.PROVIDER_DIR)
INTERIM_DIR = cfg.ensure_dir(cfg.INTERIM_DIR)

COMBINED_CSV = INTERIM_DIR / "provider.csv"
QUARTERLY_CSV = INTERIM_DIR / "provider_quarterly.csv"

# Run flags
RUN_EXTRACT = False
RUN_COMBINE = True
RUN_BUILD_QUARTERLY = True

print(f"[paths] NH_ZIP_DIR={NH_ZIP_DIR}")
print(f"[paths] PROV_DIR  ={PROV_DIR}")
print(f"[paths] INTERIM   ={INTERIM_DIR}")
print(
    f"[flags] RUN_EXTRACT={RUN_EXTRACT}, "
    f"RUN_COMBINE={RUN_COMBINE}, "
    f"RUN_BUILD_QUARTERLY={RUN_BUILD_QUARTERLY}"
)

# ============================ File selection ==================================
PRIORITY = [
    "providerinfo_download.csv",
    "providerinfo_display.csv",
    "nh_providerinfo",
]

# ============================ Local helper logic ==============================
ALNUM_6_7 = re.compile(r"^[0-9A-Z]{6,7}$")


def clean_primary_ccn_scalar(val):
    """
    Preserve the existing provider-script logic exactly:
    - reject values with + or .
    - zfill purely numeric values to length 6
    - allow 6-7 char alphanumeric IDs
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return np.nan
    s = str(val).strip().upper()
    if "+" in s or "." in s:
        return np.nan
    if s.isdigit():
        return s.zfill(6)
    if ALNUM_6_7.fullmatch(s):
        return s
    return np.nan


def infer_period_from_file(df: pd.DataFrame) -> tuple[int, int]:
    """
    Use FILEDATE or Processing Date inside the CSV
    (after norm_cols -> filedate / processing_date).
    """
    cand_cols = ["filedate", "processing_date"]
    for c in cand_cols:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().any():
                d0 = dt[dt.notna()].iloc[0]
                return int(d0.year), int(d0.month)
    raise ValueError("Could not infer period from FILEDATE / Processing Date in CSV.")


def last_nonmissing(s: pd.Series):
    s2 = s.dropna()
    if len(s2) == 0:
        return pd.NA
    return s2.iloc[-1]


# ============================ Column candidates ===============================
PRIMARY_CCN_ORDER = [
    "cms_certification_number",
    "cms_certification_number_ccn",
    "federal_provider_number",
    "provnum",
    "provider_id",
    "provider_number",
]

HOSP_CANDIDATES = [
    "provider_resides_in_hospital",
    "resides_in_hospital",
    "provider_resides_in_hospital_",
    "inhosp",
]

CASE_MIX_CANDS = [
    "exp_total",
    "cm_total",
    "case_mix_total_nurse_staffing_hours_per_resident_per_day",
    "casemix_total_nurse_staffing_hours_per_resident_per_day",
]

CCRC_CANDS = ["ccrc_facil", "continuing_care_retirement_community"]

BEDS_PROV_CANDS = [
    "bedcert",
    "number_of_certified_beds",
]

SFF_STATUS_TEXT_CANDS = ["special_focus_status"]
SFF_FACILITY_CANDS = ["special_focus_facility"]


def classify_sff_text(text: str | float) -> str | None:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    t = str(text).strip()
    if t == "" or t.lower() == "nan":
        return None
    t = t.replace("\u00A0", " ").replace("—", "-").replace("–", "-")
    tl = t.lower()
    if tl in {"y", "yes"}:
        return "current"
    if tl in {"n", "no"}:
        return "none"
    if "candidate" in tl:
        return "candidate"
    if "former" in tl or "graduated" in tl or "terminated" in tl or "no longer" in tl:
        return "former"
    if "not" in tl and "sff" in tl:
        return "none"
    if tl == "sff" or tl.startswith("sff") or (" sff" in tl):
        return "current"
    return "unknown"


def coalesce_sff_class(text_cls: pd.Series, facility_bool: pd.Series) -> pd.Series:
    out = text_cls.copy()
    mask = out.isna() | (out == "unknown")
    if mask.any():
        tmp = pd.Series(pd.NA, index=out.index, dtype="object")
        tmp.loc[facility_bool == True] = "current"
        tmp.loc[facility_bool == False] = "none"
        out = out.mask(mask & tmp.notna(), tmp)
    return out.fillna("unknown").astype("string")


# ============================ Standardize one month ===========================
def standardize_provider_info(df: pd.DataFrame, yyyy_hint: int, mm_hint: int) -> pd.DataFrame:
    df = cfg.norm_cols(df)

    # Infer period from file contents; if that fails, fall back to inner-zip name
    try:
        yyyy_use, mm_use = infer_period_from_file(df)
    except Exception:
        yyyy_use, mm_use = yyyy_hint, mm_hint

    # Primary CCN
    present_cands = [c for c in PRIMARY_CCN_ORDER if c in df.columns]
    primary = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in present_cands:
        primary = primary.mask(primary.isna() & df[c].notna(), df[c])
    cleaned_ccn = primary.map(clean_primary_ccn_scalar)

    # Hospital flag -> 0/1 Int8
    hosp = pd.Series(pd.NA, index=df.index, dtype="object")
    for cand in HOSP_CANDIDATES:
        if cand in df.columns:
            mapped = df[cand].map(cfg.to_boolish).astype("boolean")
            hosp = hosp.mask(hosp.isna() & mapped.notna(), mapped)
    hosp = hosp.astype("boolean")
    hosp01 = hosp.fillna(False).astype("Int8")

    # CCRC -> 0/1 Int8
    ccrc_bool = pd.Series(pd.NA, index=df.index, dtype="object")
    for cand in CCRC_CANDS:
        if cand in df.columns:
            mapped = df[cand].map(cfg.to_boolish).astype("boolean")
            ccrc_bool = ccrc_bool.mask(ccrc_bool.isna() & mapped.notna(), mapped)
    ccrc_bool = ccrc_bool.astype("boolean")
    ccrc01 = ccrc_bool.fillna(False).astype("Int8")

    # SFF -> 0/1 Int8 based on text/facility
    sff_status_text = None
    for cand in SFF_STATUS_TEXT_CANDS:
        if cand in df.columns:
            sff_status_text = df[cand]
            break

    sff_text_cls = pd.Series(pd.NA, index=df.index, dtype="object")
    if sff_status_text is not None:
        sff_text_cls = sff_status_text.map(classify_sff_text)

    sff_facility_bool = pd.Series(pd.NA, index=df.index, dtype="object")
    for cand in SFF_FACILITY_CANDS:
        if cand in df.columns:
            mapped = df[cand].map(cfg.to_boolish).astype("boolean")
            sff_facility_bool = sff_facility_bool.mask(sff_facility_bool.isna() & mapped.notna(), mapped)
    sff_facility_bool = sff_facility_bool.astype("boolean")

    sff01 = coalesce_sff_class(sff_text_cls, sff_facility_bool).isin(
        ["current", "candidate"]
    ).astype("Int8")

    # Case-mix Raw (kept as-is intentionally)
    case_mix = pd.Series(pd.NA, index=df.index, dtype="object")
    for cand in CASE_MIX_CANDS:
        if cand in df.columns:
            case_mix = case_mix.mask(case_mix.isna() & df[cand].notna(), df[cand])

    # Provider beds -> numeric
    beds_prov = pd.Series(pd.NA, index=df.index, dtype="object")
    for cand in BEDS_PROV_CANDS:
        if cand in df.columns:
            beds_prov = beds_prov.mask(beds_prov.isna() & df[cand].notna(), df[cand])
    beds_prov = pd.to_numeric(beds_prov, errors="coerce")

    # Period fields
    quarter = f"{yyyy_use:04d}Q{(mm_use - 1)//3 + 1}"
    year_month = f"{yyyy_use:04d}/{mm_use:02d}"

    out = pd.DataFrame({
        "cms_certification_number": cleaned_ccn,
        "quarter": quarter,
        "year_month": year_month,
        "provider_resides_in_hospital": hosp01,
        "ccrc_facility": ccrc01,
        "sff_facility": sff01,
        "case_mix_total": case_mix,
        "beds_prov": beds_prov,
    })

    # Drop invalid CCN & duplicates
    out = out.dropna(subset=["cms_certification_number"]).drop_duplicates()
    out = out.sort_values(["cms_certification_number", "year_month"], kind="mergesort").reset_index(drop=True)
    return out


# ============================ Extract -> Standardize -> Write =================
def extract_and_standardize():
    yearly = sorted(p for p in NH_ZIP_DIR.glob("nh_archive_*.zip") if p.is_file())
    if not yearly:
        raise FileNotFoundError(f"No yearly zips found in {NH_ZIP_DIR}")

    written = 0
    skipped_no_inner_date = 0
    skipped_no_provider_csv = 0
    bad_inner_zip = 0

    for yzip in yearly:
        with zipfile.ZipFile(yzip, "r") as yz:
            inner_zips = [n for n in yz.namelist() if n.lower().endswith(".zip")]

            for inner in inner_zips:
                mm, yyyy = cfg.parse_mm_yyyy_from_inner(Path(inner).name)
                if not (mm and yyyy):
                    skipped_no_inner_date += 1
                    continue

                with yz.open(inner) as inner_bytes:
                    try:
                        with zipfile.ZipFile(BytesIO(inner_bytes.read()), "r") as mz:
                            entries = mz.namelist()
                            chosen = cfg.choose_member_by_priority(entries, PRIORITY)
                            if not chosen:
                                skipped_no_provider_csv += 1
                                continue

                            raw = mz.read(chosen)
                            df = cfg.read_csv_bytes_robust(raw, dtype=str, low_memory=False)
                            std = standardize_provider_info(df, yyyy, mm)

                            out_name = f"provider_info_{yyyy:04d}_{mm:02d}.csv"
                            cfg.atomic_overwrite_csv(std, PROV_DIR / out_name, index=False)

                            print(f"[save] {out_name:>22}  rows={len(std):,}")
                            written += 1

                    except zipfile.BadZipFile:
                        bad_inner_zip += 1
                        continue

    print(f"\n[extract+standardize] wrote {written} monthly provider_info CSV(s).")
    print(
        "[extract+standardize] skipped:"
        f" no_inner_date={skipped_no_inner_date:,},"
        f" no_provider_csv={skipped_no_provider_csv:,},"
        f" bad_inner_zip={bad_inner_zip:,}"
    )


# ============================ Combine (de-dupe then 2Q lead) ==================
def combine_monthlies_and_save():
    monthly = sorted(PROV_DIR.glob("provider_info_*.csv"))
    if not monthly:
        raise FileNotFoundError(f"No provider_info_*.csv files found in {PROV_DIR}")

    frames = []
    failed_reads = 0

    for p in monthly:
        try:
            df = pd.read_csv(
                p,
                dtype={"cms_certification_number": "string"},
                low_memory=False
            )

            # Enforce types on dummies after disk read
            for col in ["provider_resides_in_hospital", "ccrc_facility", "sff_facility"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int8")

            # Ensure beds_prov is numeric
            if "beds_prov" in df.columns:
                df["beds_prov"] = pd.to_numeric(df["beds_prov"], errors="coerce").astype("Int64")

            # Case mix: keep numeric where possible
            if "case_mix_total" in df.columns:
                df["case_mix_total"] = pd.to_numeric(df["case_mix_total"], errors="coerce")

            frames.append(df)

        except Exception as e:
            print(f"[warn] failed reading {p.name}: {e}")
            failed_reads += 1

    if not frames:
        raise ValueError("No monthly provider files could be read successfully.")

    prov = pd.concat(frames, ignore_index=True)

    rows_before = len(prov)

    keep_cols = [
        "cms_certification_number",
        "quarter",
        "year_month",
        "provider_resides_in_hospital",
        "ccrc_facility",
        "sff_facility",
        "case_mix_total",
        "beds_prov",
    ]
    prov = prov[[c for c in keep_cols if c in prov.columns]]

    # Drop rows missing essentials
    prov = prov.dropna(subset=["cms_certification_number", "quarter", "year_month"])

    rows_after_required = len(prov)

    # De-duplicate before lead to prevent misalignment
    prov = (
        prov
        .drop_duplicates(["cms_certification_number", "year_month"], keep="first")
        .reset_index(drop=True)
    )

    rows_after_dedup = len(prov)

    # Stable month order
    prov["_ord"] = pd.to_datetime(
        prov["year_month"] + "/01",
        format="%Y/%m/%d",
        errors="coerce"
    )

    prov = prov.sort_values(["cms_certification_number", "_ord"], kind="mergesort")

    # Apply 2-quarter (6-month) lead to case_mix_total by CCN
    if "case_mix_total" in prov.columns:
        prov["case_mix_total"] = (
            prov.groupby("cms_certification_number", sort=False)["case_mix_total"]
            .shift(-6)
        )

    prov = (
        prov
        .drop(columns=["_ord"])
        .sort_values(["cms_certification_number", "year_month"], kind="mergesort")
        .reset_index(drop=True)
    )

    cfg.atomic_overwrite_csv(prov, COMBINED_CSV, index=False)

    # QA summary
    n_ccn = prov["cms_certification_number"].nunique(dropna=True)
    n_missing_beds = int(prov["beds_prov"].isna().sum()) if "beds_prov" in prov.columns else 0
    n_missing_case_mix = int(prov["case_mix_total"].isna().sum()) if "case_mix_total" in prov.columns else 0

    print(f"[save] combined provider panel → {COMBINED_CSV}  ({len(prov):,} rows)")
    print(
        "[qa] monthly_files="
        f"{len(monthly):,}, failed_reads={failed_reads:,}, "
        f"rows_before={rows_before:,}, rows_after_required={rows_after_required:,}, "
        f"rows_after_dedup={rows_after_dedup:,}, unique_ccn={n_ccn:,}, "
        f"missing_beds_prov={n_missing_beds:,}, missing_case_mix_total={n_missing_case_mix:,}"
    )


# ============================ Build quarterly from monthly ====================
def build_quarterly_from_monthly():
    if not COMBINED_CSV.exists():
        raise FileNotFoundError(f"Monthly provider panel not found: {COMBINED_CSV}")

    prov = pd.read_csv(
        COMBINED_CSV,
        dtype={"cms_certification_number": "string"},
        low_memory=False
    )

    if prov.empty:
        cfg.atomic_overwrite_csv(prov, QUARTERLY_CSV, index=False)
        print(f"[save] quarterly provider panel → {QUARTERLY_CSV}  (rows=0)")
        return

    # Parse monthly ordering variable
    prov["_ord"] = pd.to_datetime(
        prov["year_month"] + "/01",
        format="%Y/%m/%d",
        errors="coerce"
    )

    prov = prov.dropna(subset=["cms_certification_number", "year_month", "_ord"]).copy()

    # Rebuild year / quarter from year_month to ensure consistency
    prov["year"] = prov["_ord"].dt.year.astype("Int64")
    prov["quarter_num"] = ((prov["_ord"].dt.month - 1) // 3 + 1).astype("Int64")
    prov["quarter"] = "Q" + prov["quarter_num"].astype(str)

    # Enforce numeric types where appropriate
    for col in ["provider_resides_in_hospital", "ccrc_facility", "sff_facility"]:
        if col in prov.columns:
            prov[col] = pd.to_numeric(prov[col], errors="coerce").astype("Int8")

    if "beds_prov" in prov.columns:
        prov["beds_prov"] = pd.to_numeric(prov["beds_prov"], errors="coerce").astype("Int64")

    if "case_mix_total" in prov.columns:
        prov["case_mix_total"] = pd.to_numeric(prov["case_mix_total"], errors="coerce")

    prov = prov.sort_values(
        ["cms_certification_number", "year", "quarter_num", "_ord"],
        kind="mergesort"
    )

    grp = ["cms_certification_number", "year", "quarter"]

    qtr = (
        prov.groupby(grp, sort=False)
        .agg(
            provider_resides_in_hospital=("provider_resides_in_hospital", last_nonmissing),
            ccrc_facility=("ccrc_facility", last_nonmissing),
            sff_facility=("sff_facility", last_nonmissing),
            case_mix_total=("case_mix_total", last_nonmissing),
            beds_prov=("beds_prov", last_nonmissing),
            months_in_quarter=("year_month", "nunique"),
            last_year_month_in_quarter=("year_month", "last"),
        )
        .reset_index()
    )

    # Type cleanup
    for col in ["provider_resides_in_hospital", "ccrc_facility", "sff_facility"]:
        if col in qtr.columns:
            qtr[col] = pd.to_numeric(qtr[col], errors="coerce").fillna(0).astype("Int8")

    if "beds_prov" in qtr.columns:
        qtr["beds_prov"] = pd.to_numeric(qtr["beds_prov"], errors="coerce").astype("Int64")

    if "case_mix_total" in qtr.columns:
        qtr["case_mix_total"] = pd.to_numeric(qtr["case_mix_total"], errors="coerce")

    q_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    qtr["_qord"] = qtr["quarter"].map(q_order)

    qtr = (
        qtr.sort_values(["cms_certification_number", "year", "_qord"], kind="mergesort")
        .drop(columns=["_qord"])
        .reset_index(drop=True)
    )

    cfg.atomic_overwrite_csv(qtr, QUARTERLY_CSV, index=False)

    print(f"[save] quarterly provider panel → {QUARTERLY_CSV}  ({len(qtr):,} rows)")
    print(
        "[qa-quarterly] unique_ccn="
        f"{qtr['cms_certification_number'].nunique(dropna=True):,}, "
        f"missing_beds_prov={int(qtr['beds_prov'].isna().sum()) if 'beds_prov' in qtr.columns else 0:,}, "
        f"missing_case_mix_total={int(qtr['case_mix_total'].isna().sum()) if 'case_mix_total' in qtr.columns else 0:,}"
    )


# =============================== RUN ==========================================
if __name__ == "__main__":
    if RUN_EXTRACT:
        extract_and_standardize()
    else:
        print("[skip] extraction step skipped")

    if RUN_COMBINE:
        combine_monthlies_and_save()
    else:
        print("[skip] combine step skipped")

    if RUN_BUILD_QUARTERLY:
        build_quarterly_from_monthly()
    else:
        print("[skip] quarterly build skipped")