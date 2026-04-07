from pathlib import Path
import pandas as pd
import re

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(r"C:/Repositories/white-bowblis-nhmc")
RAW_DIR = Path(r"C:\Users\Owner\OneDrive\NursingHomeData\quality-measures")
OUT_DIR = PROJECT_ROOT / "data" / "interim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "quality_measures_quarter_panel.csv"

# ------------------------------------------------------------
# Quarter window wanted in final output
# ------------------------------------------------------------
START_YEAR, START_QUARTER = 2017, 1
END_YEAR, END_QUARTER = 2024, 2

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def read_csv_with_fallbacks(path: Path, usecols=None) -> pd.DataFrame:
    attempts = [
        {"encoding": "utf-8", "low_memory": False, "usecols": usecols},
        {"encoding": "utf-8-sig", "low_memory": False, "usecols": usecols},
        {"encoding": "latin1", "low_memory": False, "usecols": usecols},
    ]
    last_err = None
    for kwargs in attempts:
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err


def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
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


def quarter_to_col(year: int, quarter: int) -> str:
    return f"Q{quarter}_{year}"


def parse_quarter_label(x):
    """
    Accepts things like:
      2019Q3
      2019 Q3
      Q3 2019
      q3 2019
    Returns (year, quarter) or None
    """
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


def quarter_range(start_y, start_q, end_y, end_q):
    out = []
    y, q = start_y, start_q
    while (y < end_y) or (y == end_y and q <= end_q):
        out.append((y, q))
        q += 1
        if q == 5:
            q = 1
            y += 1
    return out


def parse_measure_period(period_str):
    """
    Example:
      2018Q4-2019Q3  -> [(2018,4), (2019,1), (2019,2), (2019,3)]
    """
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


def quarter_sort_key(col_name: str):
    m = re.fullmatch(r"Q([1-4])_(\d{4})", col_name)
    if not m:
        return (9999, 9)
    q = int(m.group(1))
    y = int(m.group(2))
    return (y, q)


def quarter_in_window(year: int, quarter: int) -> bool:
    return ((year > START_YEAR) or (year == START_YEAR and quarter >= START_QUARTER)) and \
           ((year < END_YEAR) or (year == END_YEAR and quarter <= END_QUARTER))


def build_target_quarter_cols(start_year, start_quarter, end_year, end_quarter):
    return [quarter_to_col(y, q) for (y, q) in quarter_range(start_year, start_quarter, end_year, end_quarter)]


# ------------------------------------------------------------
# Candidate column names across schema eras
# ------------------------------------------------------------
CCN_CANDS = ["CMS Certification Number (CCN)", "Federal Provider Number", "PROVNUM"]
CODE_CANDS = ["Measure Code", "MSR_CD"]
DESC_CANDS = ["Measure Description", "MSR_DESCR"]
PERIOD_CANDS = ["Measure Period", "MEASURE_PERIOD"]

Q_SCORE_CANDS = {
    1: ["Q1 Measure Score", "Q1_MEASURE_SCORE"],
    2: ["Q2 Measure Score", "Q2_MEASURE_SCORE"],
    3: ["Q3 Measure Score", "Q3_MEASURE_SCORE"],
    4: ["Q4 Measure Score", "Q4_MEASURE_SCORE"],
}

Q_LABEL_CANDS = {
    1: ["Q1 quarter", "Q1_QUARTER"],
    2: ["Q2 quarter", "Q2_QUARTER"],
    3: ["Q3 quarter", "Q3_QUARTER"],
    4: ["Q4 quarter", "Q4_QUARTER"],
}

# ------------------------------------------------------------
# Process files newest -> oldest
# Newest release wins when the same facility-measure-quarter
# appears multiple times across monthly release files.
# ------------------------------------------------------------
files = sorted(RAW_DIR.glob("quality_measures_*.csv"), key=release_rank, reverse=True)

master = None
target_quarter_cols = build_target_quarter_cols(
    START_YEAR, START_QUARTER, END_YEAR, END_QUARTER
)

for f in files:
    print(f"Processing {f.name}")

    header = pd.read_csv(f, nrows=0)
    cols = header.columns.tolist()

    ccn_col = first_existing(cols, CCN_CANDS)
    code_col = first_existing(cols, CODE_CANDS)
    desc_col = first_existing(cols, DESC_CANDS)
    period_col = first_existing(cols, PERIOD_CANDS)

    q_score_cols = {slot: first_existing(cols, Q_SCORE_CANDS[slot]) for slot in range(1, 5)}
    q_label_cols = {slot: first_existing(cols, Q_LABEL_CANDS[slot]) for slot in range(1, 5)}

    needed = [c for c in [ccn_col, code_col, desc_col, period_col] if c is not None]
    needed += [c for c in q_score_cols.values() if c is not None]
    needed += [c for c in q_label_cols.values() if c is not None]
    needed = list(dict.fromkeys(needed))

    if ccn_col is None or code_col is None or desc_col is None:
        print(f"  Skipping {f.name}: missing one of CCN / code / description")
        continue

    df = read_csv_with_fallbacks(f, usecols=needed)

    rename_map = {
        ccn_col: "cms_certification_number",
        code_col: "quality_metric_code",
        desc_col: "quality_metric_description",
    }
    if period_col is not None:
        rename_map[period_col] = "measure_period"

    for slot in range(1, 5):
        if q_score_cols[slot] is not None:
            rename_map[q_score_cols[slot]] = f"score_slot_{slot}"
        if q_label_cols[slot] is not None:
            rename_map[q_label_cols[slot]] = f"quarter_label_slot_{slot}"

    df = df.rename(columns=rename_map)

    df["cms_certification_number"] = df["cms_certification_number"].astype(str).str.strip()
    df["quality_metric_code"] = df["quality_metric_code"].astype(str).str.strip()
    df["quality_metric_description"] = df["quality_metric_description"].astype(str).str.strip()

    for slot in range(1, 5):
        score_col = f"score_slot_{slot}"
        if score_col in df.columns:
            df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    # Determine actual calendar quarter for each slot
    slot_to_quarter = {}

    explicit_ok = True
    for slot in range(1, 5):
        qlab_col = f"quarter_label_slot_{slot}"
        if qlab_col in df.columns:
            qval = first_nonmissing(df[qlab_col])
            parsed = parse_quarter_label(qval)
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
        period_val = first_nonmissing(df["measure_period"]) if "measure_period" in df.columns else None
        parsed_period = parse_measure_period(period_val)
        if parsed_period is not None:
            slot_to_quarter = parsed_period

    if len(slot_to_quarter) != 4:
        print(f"  Skipping {f.name}: could not map Q1-Q4 to actual quarters")
        continue

    # Only keep slots whose actual quarter is in Q1_2017 to Q2_2024
    quarter_col_map = {}
    for slot, (year, quarter) in slot_to_quarter.items():
        if quarter_in_window(year, quarter):
            quarter_col_map[slot] = quarter_to_col(year, quarter)

    if len(quarter_col_map) == 0:
        print(f"  Skipping {f.name}: no mapped quarters inside target window")
        continue

    id_cols = [
        "cms_certification_number",
        "quality_metric_code",
        "quality_metric_description",
    ]

    rename_scores = {}
    for slot, qcol in quarter_col_map.items():
        score_col = f"score_slot_{slot}"
        if score_col in df.columns:
            rename_scores[score_col] = qcol

    if not rename_scores:
        print(f"  Skipping {f.name}: no score columns found after mapping")
        continue

    chunk = df[id_cols + list(rename_scores.keys())].rename(columns=rename_scores)

    # If duplicate facility-measure rows exist within a file, keep first nonmissing
    chunk = (
        chunk
        .groupby(id_cols, dropna=False, as_index=False)
        .first()
    )

    chunk = chunk.set_index(id_cols)

    if master is None:
        master = chunk
    else:
        master = master.combine_first(chunk)

# ------------------------------------------------------------
# Finalize and save
# ------------------------------------------------------------
if master is None:
    raise ValueError("No data were processed into the panel.")

master = master.reset_index()

# Ensure every target quarter column exists, even if entirely missing
for qcol in target_quarter_cols:
    if qcol not in master.columns:
        master[qcol] = pd.NA

final_cols = [
    "cms_certification_number",
    "quality_metric_code",
    "quality_metric_description",
] + target_quarter_cols

master = master[final_cols].sort_values(
    by=["cms_certification_number", "quality_metric_code", "quality_metric_description"],
    kind="stable"
)

master.to_csv(OUT_FILE, index=False)

print("\nDone.")
print(f"Output written to: {OUT_FILE}")
print(f"Rows: {len(master):,}")
print(f"Quarter columns: {len(target_quarter_cols)}")
print("Quarter range:")
print(f"{target_quarter_cols[0]} to {target_quarter_cols[-1]}")