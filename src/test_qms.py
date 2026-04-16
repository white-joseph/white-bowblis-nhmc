#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# CMS Quality Measures (MDS) —> Extract raw monthly CSVs only,
# but keep only CCNs that appear in panel.csv
# =============================================================================

import os
import re
import zipfile
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd

# ============================== Config / Paths ================================
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").is_dir() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

RAW_DIR     = Path(os.getenv("NH_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
CLEAN_DIR   = PROJECT_ROOT / "data" / "clean"
NH_ZIP_DIR  = RAW_DIR / "nh-compare"
QM_DIR      = RAW_DIR / "quality-measures"
PANEL_FP    = CLEAN_DIR / "panel.csv"

QM_DIR.mkdir(parents=True, exist_ok=True)

# Flags
DRY_RUN    = False
NAME_STYLE = "yyyy_mm"   # requested: quality_measures_YYYY_MM.csv

print(f"[paths] NH_ZIP_DIR={NH_ZIP_DIR}")
print(f"[paths] QM_DIR    ={QM_DIR}")
print(f"[paths] PANEL_FP  ={PANEL_FP}")

# ============================ Inner zip date parsing ==========================
MONTH_RE = r"(0[1-9]|1[0-2])"
YEAR_RE  = r"(20\d{2})"
INNER_PATTERNS = [
    re.compile(rf"nh_archive_{MONTH_RE}_{YEAR_RE}\.zip", re.I),
    re.compile(rf"nh_archive_{YEAR_RE}_{MONTH_RE}\.zip", re.I),
    re.compile(rf"nursing_homes_including_rehab_services_archive_{MONTH_RE}_{YEAR_RE}\.zip", re.I),
    re.compile(rf"(?:^|[_-]){MONTH_RE}[_-]{YEAR_RE}\.zip$", re.I),
    re.compile(rf"(?:^|[_-]){YEAR_RE}[_-]{MONTH_RE}\.zip$", re.I),
]

def parse_mm_yyyy_from_inner(name: str):
    for pat in INNER_PATTERNS:
        m = pat.search(name)
        if m:
            nums = [int(x) for x in m.groups() if x and x.isdigit()]
            if len(nums) >= 2:
                a, b = nums[0], nums[1]
                if a <= 12 and b >= 2000:
                    return a, b
                if b <= 12 and a >= 2000:
                    return b, a
    return (None, None)

# ============================ File selection ==================================
def std_name(mm: int, yyyy: int) -> str:
    return (f"quality_measures_{mm:02d}_{yyyy:04d}.csv"
            if NAME_STYLE == "mm_yyyy"
            else f"quality_measures_{yyyy:04d}_{mm:02d}.csv")


def is_pre_aug_2020(mm: int, yyyy: int) -> bool:
    return (yyyy < 2020) or (yyyy == 2020 and mm <= 7)


def is_quality_basename(name: str, mm: int, yyyy: int) -> bool:
    b = Path(name).name.strip().lower()
    if not b.endswith(".csv"):
        return False

    # 2017-01 through 2020-07
    if is_pre_aug_2020(mm, yyyy):
        return b in {
            "qualitymsrmds_download.csv",
            "qualitymsrmds_display.csv",
        }

    # 2020-08 onward
    return b.startswith("nh_qualitymsr_mds")


def sort_key(name: str):
    b = Path(name).name.strip().lower()
    return (
        0 if "download" in b else (1 if "display" in b else 2),
        -len(b),
        b,
    )

# ============================ CCN filtering ===================================
def normalize_ccn(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"\D", "", s)
    if s == "":
        return pd.NA
    return s.zfill(6)


def get_panel_ccns(panel_fp: Path) -> set:
    if not panel_fp.exists():
        raise FileNotFoundError(f"panel.csv not found: {panel_fp}")

    panel = pd.read_csv(panel_fp, usecols=["cms_certification_number"], low_memory=False)
    ccns = set(
        panel["cms_certification_number"]
        .map(normalize_ccn)
        .dropna()
        .unique()
        .tolist()
    )

    if not ccns:
        raise ValueError("No valid CCNs found in panel.csv")

    print(f"[panel] loaded {len(ccns):,} unique CCNs from panel.csv")
    return ccns


def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def read_csv_bytes_with_fallbacks(data: bytes) -> pd.DataFrame:
    attempts = [
        {"encoding": "utf-8", "low_memory": False},
        {"encoding": "utf-8-sig", "low_memory": False},
        {"encoding": "latin1", "low_memory": False},
    ]
    last_err = None

    for kwargs in attempts:
        try:
            return pd.read_csv(BytesIO(data), **kwargs)
        except Exception as e:
            last_err = e

    raise last_err


def filter_to_panel_ccns(df: pd.DataFrame, panel_ccns: set) -> pd.DataFrame:
    ccn_col = first_existing(
        df.columns,
        ["CMS Certification Number (CCN)", "Federal Provider Number", "PROVNUM"]
    )

    if ccn_col is None:
        raise ValueError("Could not find a CCN/provider-number column in quality measures file")

    out = df.copy()
    out[ccn_col] = out[ccn_col].map(normalize_ccn)
    out = out[out[ccn_col].isin(panel_ccns)].copy()

    return out


# =============================== Extraction ===================================
def extract_quality_measure_files():
    panel_ccns = get_panel_ccns(PANEL_FP)

    yearlies = sorted(p for p in NH_ZIP_DIR.glob("nh_archive_*.zip") if p.is_file())
    if not yearlies:
        raise FileNotFoundError(f"No yearly zips found in {NH_ZIP_DIR}")

    extracted, skipped = 0, 0
    total_rows_before, total_rows_after = 0, 0
    notes = []

    for yearly in yearlies:
        with zipfile.ZipFile(yearly, "r") as yz:
            inner_zips = [n for n in yz.namelist() if n.lower().endswith(".zip")]

            for inner in inner_zips:
                mm, yyyy = parse_mm_yyyy_from_inner(Path(inner).name)
                if not (mm and yyyy):
                    skipped += 1
                    notes.append((yearly.name, inner, "no_mm_yyyy_in_inner_zip_name"))
                    continue

                with yz.open(inner) as inner_bytes:
                    try:
                        with zipfile.ZipFile(BytesIO(inner_bytes.read()), "r") as mz:
                            names = mz.namelist()
                            candidates = [n for n in names if is_quality_basename(n, mm, yyyy)]

                            if not candidates:
                                skipped += 1
                                preview = ", ".join(Path(n).name for n in names[:10])
                                notes.append((
                                    yearly.name,
                                    inner,
                                    f"no_quality_measure_match; sample: {preview}"
                                ))
                                continue

                            candidates.sort(key=sort_key)
                            target = candidates[0]

                            out_name = std_name(mm, yyyy)
                            out_path = QM_DIR / out_name

                            raw_data = mz.read(target)
                            df = read_csv_bytes_with_fallbacks(raw_data)

                            n_before = len(df)
                            df = filter_to_panel_ccns(df, panel_ccns)
                            n_after = len(df)

                            total_rows_before += n_before
                            total_rows_after += n_after

                            print(
                                f"[{yyyy}-{mm:02d}] {Path(inner).name} → {Path(target).name}  "
                                f"⇒  {out_path.name}  ({n_before:,} rows -> {n_after:,} kept)"
                            )

                            if not DRY_RUN:
                                df.to_csv(out_path, index=False)

                            extracted += 1

                    except zipfile.BadZipFile:
                        skipped += 1
                        notes.append((yearly.name, inner, "bad_inner_zip"))
                        continue
                    except Exception as e:
                        skipped += 1
                        notes.append((yearly.name, inner, f"processing_error: {e}"))
                        continue

    print(f"\n[extract] extracted={extracted}, skipped={skipped}")
    print(f"[filter] total rows before={total_rows_before:,}, after={total_rows_after:,}")

    if notes:
        print("\n[notes] first 25 skip reasons:")
        for yzip, inner, reason in notes[:25]:
            print(f"  - {yzip} :: {inner} → {reason}")
        if len(notes) > 25:
            print(f"  ... and {len(notes)-25} more")


# =============================== RUN ==========================================
if __name__ == "__main__":
    extract_quality_measure_files()