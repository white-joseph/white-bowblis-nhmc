#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# CMS Quality Measures (MDS) -> Extract -> Build selected-code quarterly panel
#
# Final output:
# - one row per cms_certification_number / year / quarter
# - selected measure codes in wide columns
# - quarter labels use Q1, Q2, Q3, Q4
# - quarter assignment comes from reported quarter fields / measure period,
#   NOT from the monthly file name
# - final panel restricted to Q1 2017 through Q2 2024
#
# IMPORTANT:
# - Extraction copies the raw quality CSV out of the archive directly.
# - Standardization happens only when building the final panel.
# =============================================================================

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

import config as cfg

# =============================================================================
# CONFIG / PATHS
# =============================================================================
NH_ZIP_DIR = cfg.NH_COMPARE_DIR
QM_DIR = cfg.ensure_dir(cfg.QUALITY_DIR)
INTERIM_DIR = cfg.ensure_dir(cfg.INTERIM_DIR)

OUT_FILE = INTERIM_DIR / "quality_measures.csv"

DRY_RUN = False
NAME_STYLE = "yyyy_mm"

TARGET_CODES = {
    # Long-stay / existing outcomes
    "401",  # Long-stay ADL decline
    "404",  # Long-stay weight loss
    "405",
    "406",  # Long-stay catheter
    "407",  # Long-stay UTI
    "410",  # Long-stay falls with major injury
    "419",  # Long-stay antipsychotic
    "451",
    "452",  # Long-stay hypnotic
    "453",  # Long-stay pressure injuries

    # Short-stay outcomes / process measures
    "424",  # Short-stay moderate/severe pain
    "425",  # Short-stay new/worsened pressure ulcers
    "430",  # Short-stay pneumococcal vaccine
    "434",  # Short-stay newly receiving antipsychotic
    "471",  # Short-stay improved function
    "472",  # Short-stay influenza vaccine
}

START_YEAR, START_QUARTER = 2017, 1
END_YEAR, END_QUARTER = 2024, 2

print(f"[paths] NH_ZIP_DIR={NH_ZIP_DIR}")
print(f"[paths] QM_DIR    ={QM_DIR}")
print(f"[paths] INTERIM   ={INTERIM_DIR}")

# =============================================================================
# EXTRACT HELPERS
# =============================================================================
def std_name(mm: int, yyyy: int) -> str:
    if NAME_STYLE == "mm_yyyy":
        return f"quality_measures_{mm:02d}_{yyyy:04d}.csv"
    return f"quality_measures_{yyyy:04d}_{mm:02d}.csv"


def is_pre_aug_2020(mm: int, yyyy: int) -> bool:
    return (yyyy < 2020) or (yyyy == 2020 and mm <= 7)


def is_quality_basename(name: str, mm: int, yyyy: int) -> bool:
    b = Path(name).name.strip().lower()
    if not b.endswith(".csv"):
        return False

    if is_pre_aug_2020(mm, yyyy):
        return b in {
            "qualitymsrmds_download.csv",
            "qualitymsrmds_display.csv",
        }

    return b.startswith("nh_qualitymsr_mds")


def sort_key(name: str, zf: zipfile.ZipFile):
    b = Path(name).name.strip().lower()
    size = zf.getinfo(name).file_size
    return (
        0 if "download" in b else (1 if "display" in b else 2),
        -size,
        -len(b),
        b,
    )


def extract_quality_measure_files():
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

                            candidates.sort(key=lambda n: sort_key(n, mz))
                            target = candidates[0]

                            out_name = std_name(mm, yyyy)
                            out_path = QM_DIR / out_name

                            print(f"[{yyyy}-{mm:02d}] {Path(inner).name} -> {Path(target).name} => {out_path.name}")

                            if not DRY_RUN:
                                raw_data = mz.read(target)
                                out_path.write_bytes(raw_data)

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

    if notes:
        print("\n[notes] first 25 skip reasons:")
        for yzip, inner, reason in notes[:25]:
            print(f"  - {yzip} :: {inner} -> {reason}")
        if len(notes) > 25:
            print(f"  ... and {len(notes)-25} more")

# =============================================================================
# PANEL HELPERS
# =============================================================================
def read_csv_with_fallbacks(path: Path, usecols=None, nrows=None) -> pd.DataFrame:
    return cfg.read_csv_robust(
        path,
        dtype=str,
        low_memory=False,
        usecols=usecols,
        nrows=nrows,
        sep=",",
    )


def parse_release_from_filename(path: Path):
    m = re.search(r"quality_measures_(\d{4})_(\d{2})\.csv$", path.name)
    if not m:
        raise ValueError(f"Could not parse year/month from file name: {path.name}")
    yyyy = int(m.group(1))
    mm = int(m.group(2))
    return yyyy, mm


def release_rank(path: Path):
    yyyy, mm = parse_release_from_filename(path)
    return yyyy * 100 + mm


def quarter_in_window(year: int, quarter: int) -> bool:
    return ((year > START_YEAR) or (year == START_YEAR and quarter >= START_QUARTER)) and \
           ((year < END_YEAR) or (year == END_YEAR and quarter <= END_QUARTER))

# =============================================================================
# BUILD FINAL QUARTER PANEL
# =============================================================================
def build_quality_quarter_panel():
    files = sorted(QM_DIR.glob("quality_measures_*.csv"), key=release_rank, reverse=True)

    if not files:
        raise FileNotFoundError(f"No quality_measures_*.csv files found in {QM_DIR}")

    all_chunks = []

    for f in files:
        print(f"Processing {f.name}")

        df = read_csv_with_fallbacks(f)
        df = cfg.norm_cols(df)

        ccn_col = cfg.first_existing(
            df.columns,
            [
                "cms_certification_number_ccn",
                "cms_certification_number",
                "federal_provider_number",
                "provnum",
            ]
        )
        code_col = cfg.first_existing(
            df.columns,
            ["measure_code", "msr_cd"]
        )
        period_col = cfg.first_existing(
            df.columns,
            ["measure_period"]
        )

        q_score_cols = {
            1: cfg.first_existing(df.columns, ["q1_measure_score"]),
            2: cfg.first_existing(df.columns, ["q2_measure_score"]),
            3: cfg.first_existing(df.columns, ["q3_measure_score"]),
            4: cfg.first_existing(df.columns, ["q4_measure_score"]),
        }

        q_label_cols = {
            1: cfg.first_existing(df.columns, ["q1_quarter"]),
            2: cfg.first_existing(df.columns, ["q2_quarter"]),
            3: cfg.first_existing(df.columns, ["q3_quarter"]),
            4: cfg.first_existing(df.columns, ["q4_quarter"]),
        }

        if ccn_col is None or code_col is None:
            print(f"  Skipping {f.name}: missing CCN or code column")
            print(f"    ccn_col={ccn_col}, code_col={code_col}")
            continue

        rename_map = {
            ccn_col: "cms_certification_number",
            code_col: "quality_metric_code",
        }
        if period_col is not None:
            rename_map[period_col] = "measure_period"

        for slot in range(1, 5):
            if q_score_cols[slot] is not None:
                rename_map[q_score_cols[slot]] = f"score_slot_{slot}"
            if q_label_cols[slot] is not None:
                rename_map[q_label_cols[slot]] = f"quarter_label_slot_{slot}"

        df = df.rename(columns=rename_map)

        df["cms_certification_number"] = cfg.normalize_ccn_any(df["cms_certification_number"])
        df["quality_metric_code"] = df["quality_metric_code"].astype(str).str.strip()

        df = df[df["cms_certification_number"].notna()].copy()
        df = df[df["quality_metric_code"].isin(TARGET_CODES)].copy()

        if df.empty:
            print(f"  No target rows found in {f.name}")
            continue

        for slot in range(1, 5):
            score_col = f"score_slot_{slot}"
            if score_col in df.columns:
                df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

        slot_to_quarter = {}

        explicit_ok = True
        for slot in range(1, 5):
            qlab_col = f"quarter_label_slot_{slot}"
            if qlab_col in df.columns:
                qval = cfg.first_nonmissing(df[qlab_col])
                parsed = cfg.parse_quarter_label(qval)
                if parsed is not None:
                    slot_to_quarter[slot] = parsed
                else:
                    explicit_ok = False
                    break
            else:
                explicit_ok = False
                break

        if not explicit_ok:
            slot_to_quarter = {}
            period_val = cfg.first_nonmissing(df["measure_period"]) if "measure_period" in df.columns else None
            parsed_period = cfg.parse_measure_period(period_val)
            if parsed_period is not None:
                slot_to_quarter = parsed_period

        if len(slot_to_quarter) != 4:
            print(f"  Skipping {f.name}: could not map Q1-Q4 to actual calendar quarters")
            continue

        long_parts = []

        for slot, (year, quarter_num) in slot_to_quarter.items():
            if not quarter_in_window(year, quarter_num):
                continue

            score_col = f"score_slot_{slot}"
            if score_col not in df.columns:
                continue

            part = df[
                ["cms_certification_number", "quality_metric_code", score_col]
            ].copy()

            part = part.rename(columns={score_col: "value"})
            part["year"] = year
            part["quarter"] = f"Q{quarter_num}"
            part["source_release_rank"] = release_rank(f)

            long_parts.append(part)

        if not long_parts:
            print(f"  Skipping {f.name}: no usable score columns in target window")
            continue

        chunk = pd.concat(long_parts, ignore_index=True)
        all_chunks.append(chunk)

    if not all_chunks:
        raise ValueError("No data were processed into the final panel.")

    master = pd.concat(all_chunks, ignore_index=True)

    master = master.sort_values(
        by=[
            "cms_certification_number",
            "quality_metric_code",
            "year",
            "quarter",
            "source_release_rank",
        ],
        ascending=[True, True, True, True, False],
        kind="stable",
    )

    master = master.drop_duplicates(
        subset=[
            "cms_certification_number",
            "quality_metric_code",
            "year",
            "quarter",
        ],
        keep="first",
    ).copy()

    wide = master.pivot_table(
        index=["cms_certification_number", "year", "quarter"],
        columns="quality_metric_code",
        values="value",
        aggfunc="first"
    ).reset_index()

    wide.columns.name = None
    wide = wide.rename(columns={code: f"qm_{code}" for code in TARGET_CODES if code in wide.columns})

    for code in sorted(TARGET_CODES):
        col = f"qm_{code}"
        if col not in wide.columns:
            wide[col] = pd.NA

    q_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    wide["_quarter_order"] = wide["quarter"].map(q_order)

    metric_cols = [f"qm_{code}" for code in sorted(TARGET_CODES)]

    wide = wide[
        ["cms_certification_number", "year", "quarter"] + metric_cols + ["_quarter_order"]
    ].sort_values(
        by=["cms_certification_number", "year", "_quarter_order"],
        kind="stable"
    ).drop(columns="_quarter_order").reset_index(drop=True)

    cfg.atomic_overwrite_csv(wide, OUT_FILE, index=False)

    print("\nDone.")
    print(f"Output written to: {OUT_FILE}")
    print(f"Rows: {len(wide):,}")
    print("Columns:")
    print(wide.columns.tolist())

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    extract_quality_measure_files()
    build_quality_quarter_panel()