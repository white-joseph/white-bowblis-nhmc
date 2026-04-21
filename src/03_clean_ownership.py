#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# CMS Ownership —> Extract -> Standardize -> Combine
#
# Notes:
# - Uses shared paths/helpers from config.py where applicable
# - Preserves existing ownership extraction, standardization, and combine logic
# - Does not rename variables or change data content rules
# =============================================================================

from __future__ import annotations

import re
import shutil
import warnings
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

# ============================== Config / Paths ================================
NH_ZIP_DIR = cfg.NH_COMPARE_DIR
OWN_DIR = cfg.ensure_dir(cfg.OWNERSHIP_DIR)
INTERIM_DIR = cfg.ensure_dir(cfg.INTERIM_DIR)

# Provider outputs
PROV_DIR = cfg.PROVIDER_DIR
HOSP_PANEL_CSV = PROV_DIR / "provider_resides_in_hospital_panel.csv"

COMBINED_CSV = INTERIM_DIR / "ownership.csv"

# Flags
DRY_RUN = False
NAME_STYLE = "yyyy_mm"      # "mm_yyyy" or "yyyy_mm"
DO_STANDARDIZE = True
DO_COMBINE = True

print(f"[paths] NH_ZIP_DIR={NH_ZIP_DIR}")
print(f"[paths] OWN_DIR={OWN_DIR}")
print(f"[paths] INTERIM={INTERIM_DIR}")

# ============================ Housekeeping ====================================
for junk in (OWN_DIR / "profiling", OWN_DIR / "qa_reports"):
    if junk.exists() and junk.is_dir():
        shutil.rmtree(junk, ignore_errors=True)

# =========================== Shared/local helpers =============================
def norm_header_ownership(h: str) -> str:
    return re.sub(r"\s+", " ", str(h or "").strip().lower().replace("_", " ")).strip()


def safe_to_datetime_ownership(series: pd.Series) -> pd.Series:
    """
    Robust date parser for association/processing dates.
    - strips leading 'since'
    - explicit formats first; then permissive fallback
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"(?i)^\s*since[:\-]?\s*", "", regex=True)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    m = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
    if m.any():
        out.loc[m] = pd.to_datetime(s[m], format="%Y-%m-%d", errors="coerce")

    m = s.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$")
    if m.any():
        out.loc[m] = pd.to_datetime(s[m], format="%m/%d/%Y", errors="coerce")

    m = s.str.match(r"^\d{1,2}/\d{1,2}/\d{2}$")
    if m.any():
        out.loc[m] = pd.to_datetime(s[m], format="%m/%d/%y", errors="coerce")

    remaining = out.isna() & s.notna()
    looks_like_date = remaining & s.str.contains(r"\d") & s.str.contains(r"[-/]")
    if looks_like_date.any():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`."
            )
            out.loc[looks_like_date] = pd.to_datetime(s[looks_like_date], errors="coerce")

    return out


def clean_ccn_raw(val: object) -> object:
    """
    Preserve alphanumeric CCNs; drop scientific notation/junk.
    Rules:
      - If value contains '.' or '+': drop
      - Keep only [A-Za-z0-9]; uppercase
      - If purely digits -> left-pad to 6
      - Require length between 5 and 7 after cleaning, else drop
    """
    if val is None:
        return pd.NA
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return pd.NA
    if "." in s or "+" in s:
        return pd.NA
    s = re.sub(r"[^0-9A-Za-z]", "", s).upper()
    if s == "":
        return pd.NA
    if s.isdigit():
        s = s.zfill(6)
    if not (5 <= len(s) <= 7):
        return pd.NA
    return s


def is_ownership_basename(name: str) -> bool:
    b = Path(name).name.strip().lower()
    if not re.search(r"(ownership|owner)", b):
        return False
    return bool(re.search(r"\.(csv|txt|tsv)$", b))


def std_name(mm: int, yyyy: int):
    return (
        f"ownership_{yyyy:04d}_{mm:02d}.csv"
        if NAME_STYLE == "yyyy_mm"
        else f"ownership_{mm:02d}_{yyyy:04d}.csv"
    )


RE_MM_YYYY = re.compile(r"ownership_(0[1-9]|1[0-2])_(20\d{2})\.csv$", re.I)
RE_YYYY_MM = re.compile(r"ownership_(20\d{2})_(0[1-9]|1[0-2])\.csv$", re.I)

def parse_ym_from_fname(name: str):
    m = RE_MM_YYYY.search(name)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = RE_YYYY_MM.search(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

# =================== 1) EXTRACT ==============================================
def extract_by_filename_only():
    yearlies = sorted(p for p in NH_ZIP_DIR.glob("nh_archive_*.zip") if p.is_file())
    if not yearlies:
        raise FileNotFoundError(f"No yearly zips found in {NH_ZIP_DIR}")

    extracted, skipped = 0, 0
    notes = []

    for yearly in yearlies:
        with zipfile.ZipFile(yearly, "r") as yz:
            inner_zips = [n for n in yz.namelist() if n.lower().endswith(".zip")]

            for inner in inner_zips:
                mm, yyyy = cfg.parse_mm_yyyy_from_inner(Path(inner).name)
                if not (mm and yyyy):
                    skipped += 1
                    notes.append((yearly.name, inner, "no_mm_yyyy_in_inner_zip_name"))
                    continue

                with yz.open(inner) as inner_bytes:
                    with zipfile.ZipFile(BytesIO(inner_bytes.read()), "r") as mz:
                        names = mz.namelist()
                        candidates = [n for n in names if is_ownership_basename(n)]

                        if not candidates:
                            skipped += 1
                            preview = ", ".join(Path(n).name for n in names[:8])
                            notes.append((yearly.name, inner, f"no_ownership_like_name; sample: {preview}"))
                            continue

                        def sort_key(n):
                            bn = Path(n).name.strip().lower()
                            size = mz.getinfo(n).file_size
                            return (
                                0 if "download" in bn else (1 if "display" in bn else 2),
                                -len(bn),
                                -size,
                                bn,
                            )

                        candidates.sort(key=sort_key)
                        target = candidates[0]

                        out_name = std_name(mm, yyyy)
                        out_path = OWN_DIR / out_name
                        print(f"[{yyyy}-{str(mm).zfill(2)}] {Path(inner).name} → {Path(target).name}  ⇒  {out_path.name}")

                        if not DRY_RUN:
                            data = mz.read(target)
                            out_path.write_bytes(data)

                        extracted += 1

    print(f"\n[extract] extracted={extracted}, skipped={skipped}")
    if notes:
        print("\n[notes] first 25 skip reasons:")
        for yzip, inner, reason in notes[:25]:
            print(f"  - {yzip} :: {inner} → {reason}")
        if len(notes) > 25:
            print(f"  ... and {len(notes)-25} more")

# =================== 2) STANDARDIZE IN-PLACE =================
CANON_MAP = {
    # CCN
    "provnum": "cms_certification_number",
    "federal provider number": "cms_certification_number",
    "cms certification number (ccn)": "cms_certification_number",
    "cms certification number": "cms_certification_number",
    "provider id": "cms_certification_number",
    # Provider
    "provider name": "provider_name",
    "provname": "provider_name",
    # Role
    "role": "role",
    "role desc": "role",
    "role_desc": "role",
    "role played by owner or manager in facility": "role",
    "role played by owner in facility": "role",
    "role of owner or manager": "role",
    # Ownership %
    "ownership percentage": "ownership_percentage",
    "owner percentage": "ownership_percentage",
    "pct ownership": "ownership_percentage",
    "percent ownership": "ownership_percentage",
    # Owner
    "owner name": "owner_name",
    "ownership name": "owner_name",
    "owner": "owner_name",
    "owner type": "owner_type",
    "type of owner": "owner_type",
    "ownership type": "owner_type",
    # Dates
    "processing date": "processing_date",
    "process date": "processing_date",
    "processingdate": "processing_date",
    "processdate": "processing_date",
    "filedate": "processing_date",
    "association date": "association_date",
    "assoc date": "association_date",
}

KEEP_COLS = [
    "cms_certification_number",
    "role",
    "owner_type",
    "owner_name",
    "ownership_percentage",
    "association_date",
    "processing_date",
]

ROLE_KEEP_MAP = {
    "DIRECT": r"5%\s*OR\s*GREATER\s+DIRECT\s+OWNERSHIP\s+INTEREST",
    "INDIRECT": r"5%\s*OR\s*GREATER\s+INDIRECT\s+OWNERSHIP\s+INTEREST",
    "PARTNERSHIP": r"\bPARTNERSHIP\s+INTEREST\b",
}


def normalize_month_df(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    # 1) rename to canonical
    ren = {c: CANON_MAP.get(norm_header_ownership(c), c) for c in df.columns}
    df = df.rename(columns=ren)

    # 2) role filter -> DIRECT / INDIRECT / PARTNERSHIP
    role_raw = df.get("role")
    if role_raw is None:
        role_raw = pd.Series(pd.NA, index=df.index)
    role_up = role_raw.fillna("").astype(str).str.upper()

    role_out = pd.Series(pd.NA, index=df.index, dtype="object")
    for canon, pat in ROLE_KEEP_MAP.items():
        role_out = role_out.mask(role_up.str.contains(pat, regex=True, na=False) == True, canon)

    mask_keep = role_out.isin(list(ROLE_KEEP_MAP.keys()))
    df = df.loc[mask_keep].copy()
    df["role"] = role_out.loc[mask_keep].values

    # 3) ownership % -> numeric 0..100 (auto-scale 0..1 -> 0..100)
    if "ownership_percentage" in df.columns:
        pct = (
            df["ownership_percentage"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        pct = pct.mask(pct.eq("") | pct.str.contains("NO PERCENTAGE", case=False))
        val = pd.to_numeric(pct, errors="coerce")
        if val.dropna().between(0, 1).mean() > 0.85 and val.dropna().between(0, 100).mean() < 0.9:
            val = val * 100.0
        df["ownership_percentage"] = val

    # 4) CCN -> cleaned
    if "cms_certification_number" in df.columns:
        df["cms_certification_number"] = df["cms_certification_number"].map(clean_ccn_raw)

    # 5) dates -> datetime64
    if "processing_date" in df.columns:
        df["processing_date"] = safe_to_datetime_ownership(df["processing_date"])
    if "association_date" in df.columns:
        df["association_date"] = safe_to_datetime_ownership(df["association_date"])

    # 6) processing_date from filename if missing anywhere
    y, m = parse_ym_from_fname(fname)
    if y and m:
        synth = pd.Timestamp(year=y, month=m, day=1)
        if "processing_date" in df.columns:
            df.loc[df["processing_date"].isna(), "processing_date"] = synth
        else:
            df["processing_date"] = synth

    # 6b) enforce association_date never null: fill from processing_date
    if "association_date" in df.columns:
        df["association_date"] = df["association_date"].fillna(df["processing_date"])
    else:
        df["association_date"] = df["processing_date"]

    # 6c) finalize date columns as strings (YYYY-MM-DD)
    for dc in ["association_date", "processing_date"]:
        if dc in df.columns:
            if not np.issubdtype(df[dc].dtype, np.datetime64):
                df[dc] = safe_to_datetime_ownership(df[dc])
            df[dc] = df[dc].dt.strftime("%Y-%m-%d")

    # 7) keep only columns
    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[KEEP_COLS]

    # 8) drop rows with invalid/missing CCN
    before = len(df)
    df = df[df["cms_certification_number"].notna()].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  [ccn] dropped {dropped:,} row(s) with invalid/missing CCN in {fname}")

    # 9) drop exact duplicates
    df = df.drop_duplicates(KEEP_COLS)

    return df


def standardize_in_place_and_combine():
    files = sorted(p for p in OWN_DIR.glob("ownership_*.csv"))
    if not files:
        raise FileNotFoundError(f"No ownership_*.csv files found in {OWN_DIR}")

    # In-place rewrite
    rewritten = 0
    for fp in files:
        try:
            raw = cfg.read_delim_robust(fp, dtype=str)
            clean = normalize_month_df(raw, fp.name)
            cfg.atomic_overwrite_csv(clean, fp, index=False)
            rewritten += 1
            print(f"[rewrite] {fp.name}  rows={len(clean)}")
        except Exception as e:
            print(f"[warn] {fp.name} failed: {e}")

    print(f"\n[standardize] Rewrote {rewritten} file(s).")

    if DO_COMBINE:
        frames = []
        files = sorted(p for p in OWN_DIR.glob("ownership_*.csv"))  # refresh list

        for fp in files:
            try:
                frames.append(cfg.read_delim_robust(fp, dtype=str))
            except Exception as e:
                print(f"[warn] combine read failed {fp.name}: {e}")

        if not frames:
            print("[combine] no frames to combine.")
            return

        combined = pd.concat(frames, ignore_index=True)

        # Build ownership 'date' as month start mirroring provider panel
        proc = pd.to_datetime(combined.get("processing_date"), errors="coerce")
        assoc = pd.to_datetime(combined.get("association_date"), errors="coerce")
        date_used = proc.fillna(assoc)
        combined["date"] = date_used.dt.to_period("M").dt.to_timestamp("s")

        # Month-by-month hospital filter via TRUE-only panel on (CCN, date)
        if HOSP_PANEL_CSV.exists():
            panel = pd.read_csv(
                HOSP_PANEL_CSV,
                dtype={"cms_certification_number": str, "provider_resides_in_hospital": str},
                low_memory=False,
            )

            m = {
                "True": True, "False": False,
                True: True, False: False,
                "true": True, "false": False
            }
            panel["provider_resides_in_hospital"] = panel["provider_resides_in_hospital"].map(m).fillna(False)
            panel = panel[panel["provider_resides_in_hospital"] == True].copy()

            panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
            panel_key = panel[["cms_certification_number", "date"]].dropna().drop_duplicates()

            merged = combined.merge(
                panel_key.assign(_drop=True),
                on=["cms_certification_number", "date"],
                how="left",
                validate="m:1",
            )

            drop_mask = merged["_drop"].fillna(False)
            removed_rows = int(drop_mask.sum())
            removed_ccns = merged.loc[drop_mask, "cms_certification_number"].nunique()

            print(f"[filter-monthly] removed {removed_rows:,} row(s) and {removed_ccns:,} unique CCN(s) (in-hospital months)")

            combined = merged.loc[~drop_mask].drop(columns=["_drop"])

        # Exact dedup across all files
        for c in ["role", "owner_type", "owner_name"]:
            if c in combined.columns:
                combined[c] = combined[c].astype(str).str.strip()

        if "ownership_percentage" in combined.columns:
            combined["ownership_percentage"] = pd.to_numeric(
                combined["ownership_percentage"], errors="coerce"
            ).round(2)

        before = len(combined)
        combined = combined.drop_duplicates(
            subset=[
                "cms_certification_number",
                "role",
                "owner_type",
                "owner_name",
                "ownership_percentage",
                "association_date",
                "processing_date",
            ],
            keep="first",
        ).reset_index(drop=True)
        removed = before - len(combined)
        print(f"[dedup-combined] removed {removed:,} exact duplicate row(s) across all months")

        # Sort by CCN
        key = combined["cms_certification_number"].fillna("~")
        combined = combined.iloc[key.argsort(kind="mergesort")].reset_index(drop=True)

        # Build year_month and quarter from processing_date
        proc_dt = pd.to_datetime(combined["processing_date"], errors="coerce")
        combined["year_month"] = proc_dt.dt.strftime("%Y/%m")
        qnum = ((proc_dt.dt.month - 1) // 3 + 1).astype("Int64")
        combined["quarter"] = proc_dt.dt.year.astype("Int64").astype("string") + "Q" + qnum.astype("string")

        # Drop helper date and replace processing_date with year_month
        combined = combined.drop(columns=["date", "processing_date"])

        desired = [
            "cms_certification_number",
            "role",
            "owner_type",
            "owner_name",
            "ownership_percentage",
            "association_date",
            "year_month",
            "quarter",
        ]
        extras = [c for c in combined.columns if c not in desired]
        combined = combined[desired + extras]

        if "year_month" in combined.columns:
            ord_dt = pd.to_datetime(combined["year_month"] + "/01", format="%Y/%m/%d", errors="coerce")
            combined = (
                combined.assign(_ord=ord_dt)
                .sort_values(["cms_certification_number", "_ord"], kind="mergesort")
                .drop(columns=["_ord"])
                .reset_index(drop=True)
            )

        cfg.atomic_overwrite_csv(combined, COMBINED_CSV, index=False)
        print(f"[save] ownership panel → {COMBINED_CSV}  ({len(combined):,} rows)")


# =============================== RUN PIPELINE =================================
if __name__ == "__main__":
    # 1) Extract monthly ownership CSVs
    extract_by_filename_only()

    # 2) Standardize in place + Combine (with month-by-month hospital filter)
    if DO_STANDARDIZE:
        standardize_in_place_and_combine()