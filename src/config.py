from __future__ import annotations

import csv
import os
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# PATHS / GLOBAL SETTINGS
# =============================================================================

def find_project_root(start: Optional[Path] = None) -> Path:
    """Walk upward until a directory containing /src is found."""
    root = (start or Path.cwd()).resolve()
    while not (root / "src").is_dir() and root != root.parent:
        root = root.parent
    return root


PROJECT_ROOT = find_project_root()
REPO_ROOT = PROJECT_ROOT
RAW_DIR = Path(os.getenv("NH_DATA_DIR", PROJECT_ROOT / "data" / "raw")).resolve()
INTERIM_DIR = (PROJECT_ROOT / "data" / "interim").resolve()
CLEAN_DIR = (PROJECT_ROOT / "data" / "clean").resolve()
OUTPUTS_DIR = (PROJECT_ROOT / "outputs").resolve()

for _p in (INTERIM_DIR, CLEAN_DIR, OUTPUTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Source folders
NH_COMPARE_DIR = RAW_DIR / "nh-compare"
PROVIDER_DIR = RAW_DIR / "provider-info-files"
OWNERSHIP_DIR = RAW_DIR / "ownership-files"
PBJ_DIR = RAW_DIR / "pbj-nurse"
MCR_DIR = RAW_DIR / "medicare-cost-reports"
QUALITY_DIR = RAW_DIR / "quality-measures"

# Sample windows
START_YM = "2017/01"
END_YM = "2024/06"
START_Q = "2017Q1"
END_Q = "2024Q2"

# Baseline event-date convention
BASELINE_EVENT_SOURCE = "mcr"
BASELINE_EVENT_SHIFT_MONTHS = 0

# Common regex / patterns for NH archive files
MONTH_RE = r"(0[1-9]|1[0-2])"
YEAR_RE = r"(20\d{2})"
INNER_ARCHIVE_PATTERNS = [
    re.compile(rf"nh_archive_{MONTH_RE}_{YEAR_RE}\.zip", re.I),
    re.compile(rf"nh_archive_{YEAR_RE}_{MONTH_RE}\.zip", re.I),
    re.compile(rf"nursing_homes_including_rehab_services_archive_{MONTH_RE}_{YEAR_RE}\.zip", re.I),
    re.compile(rf"(?:^|[_-]){MONTH_RE}[_-]{YEAR_RE}\.zip$", re.I),
    re.compile(rf"(?:^|[_-]){YEAR_RE}[_-]{MONTH_RE}\.zip$", re.I),
]

# =============================================================================
# IO HELPERS
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_robust(
    fp_or_buf,
    *,
    dtype=None,
    low_memory: bool = False,
    usecols=None,
    nrows=None,
    sep=None,
    on_bad_lines="error",
):
    """
    Robust CSV reader for either file paths or byte buffers.
    Tries several encodings before falling back to replacement.
    """
    encs = ("utf-8", "utf-8-sig", "cp1252", "latin1")
    last_err = None
    for enc in encs:
        try:
            return pd.read_csv(
                fp_or_buf,
                dtype=dtype,
                low_memory=low_memory,
                encoding=enc,
                encoding_errors="strict",
                usecols=usecols,
                nrows=nrows,
                sep=sep,
                on_bad_lines=on_bad_lines,
            )
        except Exception as e:
            last_err = e
    return pd.read_csv(
        fp_or_buf,
        dtype=dtype,
        low_memory=low_memory,
        encoding="latin1",
        encoding_errors="replace",
        usecols=usecols,
        nrows=nrows,
        sep=sep,
        on_bad_lines="skip" if on_bad_lines == "error" else on_bad_lines,
    )


def read_csv_bytes_robust(raw: bytes, *, dtype=None, low_memory: bool = False, usecols=None):
    return read_csv_robust(BytesIO(raw), dtype=dtype, low_memory=low_memory, usecols=usecols)


def sniff_delim(fp: Path, nbytes: int = 8192) -> str:
    raw = fp.read_bytes()[:nbytes]
    try:
        dialect = csv.Sniffer().sniff(raw.decode("utf-8", errors="ignore"))
        return dialect.delimiter
    except Exception:
        return "\t" if raw.count(b"\t") > raw.count(b",") else ","


def read_delim_robust(fp: Path, *, dtype=str, nrows=None) -> pd.DataFrame:
    delim = sniff_delim(fp)
    encs = ("utf-8", "utf-8-sig", "cp1252", "latin1")
    for enc in encs:
        try:
            return pd.read_csv(fp, dtype=dtype, sep=delim, encoding=enc, engine="c", low_memory=False, nrows=nrows)
        except Exception:
            try:
                return pd.read_csv(fp, dtype=dtype, sep=delim, encoding=enc, engine="python", on_bad_lines="skip", nrows=nrows)
            except Exception:
                continue
    return pd.read_csv(fp, dtype=dtype, sep=delim, encoding="utf-8", encoding_errors="replace", engine="python", on_bad_lines="skip", nrows=nrows)


def atomic_overwrite_csv(df: pd.DataFrame, out_fp: Path, *, index: bool = False, **kwargs) -> None:
    out_fp = Path(out_fp)
    ensure_dir(out_fp.parent)
    tmp_fp = out_fp.with_suffix(out_fp.suffix + ".tmp")
    df.to_csv(tmp_fp, index=index, **kwargs)
    os.replace(tmp_fp, out_fp)

# =============================================================================
# HEADER / STRING STANDARDIZATION
# =============================================================================

def norm_header(name: str) -> str:
    s = str(name)
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212-]", " ", s)
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-z_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [norm_header(c) for c in out.columns]
    return out


def first_existing(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cols_set = set(cols)
    for c in candidates:
        if c in cols_set:
            return c
    return None


def find_col_case_insensitive(cols: Sequence[str], candidates) -> Optional[str]:
    targets = [candidates] if isinstance(candidates, str) else list(candidates)
    lower_map = {str(c).lower().strip(): c for c in cols}
    for cand in targets:
        key = str(cand).lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None


def first_nonmissing(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return None
    val = s.iloc[0]
    if pd.isna(val):
        return None
    val = str(val).strip()
    return val if val != "" else None

# =============================================================================
# ID / VALUE CLEANING HELPERS
# =============================================================================

def normalize_ccn_any(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    s = s.mask(is_digits, s.str.zfill(6)).replace({"": pd.NA})
    return s


def clean_primary_ccn(series: pd.Series) -> pd.Series:
    return normalize_ccn_any(series)


def safe_to_datetime(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce")


def to_boolish(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().upper()
    if s in {"Y", "YES", "TRUE", "T", "1"}:
        return 1
    if s in {"N", "NO", "FALSE", "F", "0"}:
        return 0
    return pd.NA

# =============================================================================
# DATE / TIME HELPERS
# =============================================================================

def to_monthstart(x) -> pd.Series:
    s = pd.to_datetime(x, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp("s")


def shift_monthstart(x, k: int) -> pd.Series:
    s = pd.to_datetime(x, errors="coerce").dt.to_period("M")
    return (s + k).dt.to_timestamp("s")


def months_diff(later, earlier) -> pd.Series:
    a = pd.to_datetime(later, errors="coerce")
    b = pd.to_datetime(earlier, errors="coerce")
    return (a.dt.year - b.dt.year) * 12 + (a.dt.month - b.dt.month)


def within_k_months(a, b, k: int) -> pd.Series:
    return months_diff(a, b).abs().le(k)


def quarter_to_col(year: int, quarter: int) -> str:
    return f"Q{quarter}_{year}"


def quarter_range(start_y: int, start_q: int, end_y: int, end_q: int):
    out = []
    y, q = start_y, start_q
    while (y < end_y) or (y == end_y and q <= end_q):
        out.append((y, q))
        q += 1
        if q == 5:
            q = 1
            y += 1
    return out


def parse_quarter_label(x):
    if x is None or pd.isna(x):
        return None
    s = str(x).strip().upper()
    m1 = re.search(r"(\d{4})\s*Q([1-4])", s)
    if m1:
        return int(m1.group(1)), int(m1.group(2))
    m2 = re.search(r"Q([1-4])\s*(\d{4})", s)
    if m2:
        return int(m2.group(2)), int(m2.group(1))
    return None


def parse_measure_period(period_str):
    if period_str is None:
        return None
    s = str(period_str).strip().upper()
    matches = re.findall(r"(\d{4})\s*Q([1-4])", s)
    if len(matches) >= 2:
        start_y, start_q = int(matches[0][0]), int(matches[0][1])
        end_y, end_q = int(matches[1][0]), int(matches[1][1])
        qrng = quarter_range(start_y, start_q, end_y, end_q)
        if len(qrng) >= 4:
            return {1: qrng[0], 2: qrng[1], 3: qrng[2], 4: qrng[3]}
    return None


def filter_to_window(df: pd.DataFrame, ym_col: str, start_ym: str = START_YM, end_ym: str = END_YM) -> pd.DataFrame:
    out = df.copy()
    ym = pd.PeriodIndex(out[ym_col].astype("string"), freq="M")
    keep = (ym >= pd.Period(start_ym.replace("/", "-"), freq="M")) & (ym <= pd.Period(end_ym.replace("/", "-"), freq="M"))
    return out.loc[keep].copy()

# =============================================================================
# ARCHIVE HELPERS
# =============================================================================

def parse_mm_yyyy_from_inner(name: str) -> Tuple[Optional[int], Optional[int]]:
    for pat in INNER_ARCHIVE_PATTERNS:
        m = pat.search(name)
        if m:
            nums = [int(x) for x in m.groups() if x and str(x).isdigit()]
            if len(nums) >= 2:
                a, b = nums[0], nums[1]
                if a <= 12 and b >= 2000:
                    return a, b
                if b <= 12 and a >= 2000:
                    return b, a
    return (None, None)


def parse_ym_from_filename(path_or_name) -> Tuple[Optional[int], Optional[int]]:
    name = Path(path_or_name).name
    m = re.search(r"(20\d{2})[_-](0[1-9]|1[0-2])", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(0[1-9]|1[0-2])[_-](20\d{2})", name)
    if m:
        return int(m.group(2)), int(m.group(1))
    return (None, None)


def std_monthly_name(prefix: str, mm: int, yyyy: int, style: str = "yyyy_mm", ext: str = ".csv") -> str:
    if style == "yyyy_mm":
        return f"{prefix}_{yyyy:04d}_{mm:02d}{ext}"
    if style == "mm_yyyy":
        return f"{prefix}_{mm:02d}_{yyyy:04d}{ext}"
    raise ValueError(f"Unknown style: {style}")


def iter_outer_zip_files(base_dir: Path, pattern: str = "*.zip"):
    for fp in sorted(Path(base_dir).glob(pattern)):
        if fp.is_file():
            yield fp


def list_inner_names(zip_fp: Path):
    with zipfile.ZipFile(zip_fp) as zf:
        return zf.namelist()


def choose_member_by_priority(names: Sequence[str], priorities: Sequence[str]) -> Optional[str]:
    lowers = {n.lower(): n for n in names}
    for pref in priorities:
        for low_name, orig_name in lowers.items():
            if Path(low_name).name == pref.lower() or Path(low_name).name.startswith(pref.lower()):
                return orig_name
    return None

# =============================================================================
# PANEL / DATA HELPERS
# =============================================================================

def coalesce_suffix_duplicates(df: pd.DataFrame, suffixes=("_x", "_y")) -> pd.DataFrame:
    out = df.copy()
    sx, sy = suffixes
    for col in list(out.columns):
        if col.endswith(sx):
            base = col[:-len(sx)]
            other = base + sy
            if other in out.columns:
                out[base] = out[col].combine_first(out[other])
                out = out.drop(columns=[col, other])
    return out


def rank_bins_pct(series: pd.Series, q: int = 4, labels=None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    nonmiss = s.notna()
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    if nonmiss.sum() == 0:
        return out
    labels = labels or list(range(1, q + 1))
    try:
        out.loc[nonmiss] = pd.qcut(s.loc[nonmiss], q=q, labels=labels, duplicates="drop")
    except Exception:
        out.loc[nonmiss] = pd.cut(s.loc[nonmiss].rank(method="average", pct=True), bins=np.linspace(0, 1, q + 1), labels=labels, include_lowest=True)
    return out


def bridge_fill_equal(series: pd.Series) -> pd.Series:
    s = series.copy()
    vals = s.to_numpy(copy=True)
    for i in range(1, len(vals) - 1):
        if pd.isna(vals[i]) and not pd.isna(vals[i - 1]) and not pd.isna(vals[i + 1]) and vals[i - 1] == vals[i + 1]:
            vals[i] = vals[i - 1]
    return pd.Series(vals, index=s.index, dtype=s.dtype)

# =============================================================================
# QUALITY-MEASURE HELPERS
# =============================================================================

def quality_schema_columns(cols: Sequence[str]) -> dict:
    return {
        "ccn": first_existing(cols, ["CMS Certification Number (CCN)", "Federal Provider Number", "PROVNUM"]),
        "code": first_existing(cols, ["Measure Code", "MSR_CD"]),
        "description": first_existing(cols, ["Measure Description", "MSR_DESCR"]),
        "period": first_existing(cols, ["Measure Period", "MEASURE_PERIOD"]),
        "q1_score": first_existing(cols, ["Q1 Measure Score", "Q1_MEASURE_SCORE"]),
        "q2_score": first_existing(cols, ["Q2 Measure Score", "Q2_MEASURE_SCORE"]),
        "q3_score": first_existing(cols, ["Q3 Measure Score", "Q3_MEASURE_SCORE"]),
        "q4_score": first_existing(cols, ["Q4 Measure Score", "Q4_MEASURE_SCORE"]),
        "q1_quarter": first_existing(cols, ["Q1 quarter", "Q1_QUARTER"]),
        "q2_quarter": first_existing(cols, ["Q2 quarter", "Q2_QUARTER"]),
        "q3_quarter": first_existing(cols, ["Q3 quarter", "Q3_QUARTER"]),
        "q4_quarter": first_existing(cols, ["Q4 quarter", "Q4_QUARTER"]),
    }