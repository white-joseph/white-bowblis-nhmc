#!/usr/bin/env python
# coding: utf-8
# -----------------------------------------------------------------------------
# Build Final Quarterly Quality Panel
#
# Notes:
# - Separate from the monthly staffing panel
# - Merges quarterly quality + provider + PBJ + MCR controls
# - Uses the same CHOW agreement logic and same baseline timing source as the
#   monthly panel: MCR change month
# - Converts event month to event quarter
# - Creates quarterly treated / event_time / post / time / time_treated
# - Applies lighter cleanup appropriate for quarterly quality regressions
# - Renames overlapping quarterly metadata columns before merge
# - Carries PBJ QA flags into the final panel
# -----------------------------------------------------------------------------

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

import config as cfg

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================== Paths =========================================
INTERIM = cfg.INTERIM_DIR
CLEAN_DIR = cfg.ensure_dir(cfg.CLEAN_DIR)

QUALITY_FP = INTERIM / "quality_measures.csv"   # change if needed
PROVIDER_Q_FP = INTERIM / "provider_quarterly.csv"
PBJ_Q_FP = INTERIM / "pbj_nurse_quarterly.csv"
MCR_Q_FP = INTERIM / "mcr_quarterly.csv"
CHOW_FP = INTERIM / "chow.csv"

OUT_FINAL_FP = CLEAN_DIR / "quality_panel.csv"

print(
    f"[paths] quality={QUALITY_FP.exists()}  "
    f"provider_q={PROVIDER_Q_FP.exists()}  "
    f"pbj_q={PBJ_Q_FP.exists()}  "
    f"mcr_q={MCR_Q_FP.exists()}  "
    f"chow={CHOW_FP.exists()}"
)
print(f"[out]   quality_panel={OUT_FINAL_FP}")

# ============================== Window Config =================================
START_Q = cfg.START_Q   # e.g. "2017Q1"
END_Q = cfg.END_Q       # e.g. "2024Q2"

# ============================== Helpers =======================================
def normalize_quarter_string(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.upper()
    out = out.str.replace(r"^\s*(\d)\s*$", r"Q\1", regex=True)
    out = out.str.replace(r"^\s*(\d{4})Q([1-4])\s*$", r"Q\2", regex=True)
    out = out.str.replace(r"^\s*Q([1-4])\s*$", r"Q\1", regex=True)
    return out


def quarter_num_from_label(s: pd.Series) -> pd.Series:
    q = normalize_quarter_string(s).str.extract(r"Q([1-4])", expand=False)
    return pd.to_numeric(q, errors="coerce").astype("Int64")


def year_quarter_period(year: pd.Series, quarter: pd.Series) -> pd.PeriodIndex:
    qn = quarter_num_from_label(quarter)
    y = pd.to_numeric(year, errors="coerce").astype("Int64")
    vals = y.astype("string") + "Q" + qn.astype("string")
    return pd.PeriodIndex(vals, freq="Q")


def to_monthstart(x) -> pd.Series:
    s = pd.to_datetime(x, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp("s")


def month_to_quarter_parts(month_series: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(month_series, errors="coerce")
    out = pd.DataFrame(index=month_series.index)
    out["event_year"] = dt.dt.year.astype("Int64")
    out["event_quarter_num"] = ((dt.dt.month - 1) // 3 + 1).astype("Int64")
    out["event_quarter"] = "Q" + out["event_quarter_num"].astype("string")
    return out


def first_chow_month(df: pd.DataFrame, patt: str) -> pd.Series:
    cols = [c for c in df.columns if re.search(patt, c, flags=re.I)]
    if not cols:
        return pd.Series(pd.NaT, index=df.index)
    tmp = df[cols].apply(pd.to_datetime, errors="coerce")
    return to_monthstart(tmp.min(axis=1))


def months_diff(a, b) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.inf
    pa, pb = pd.Period(a, "M"), pd.Period(b, "M")
    return float((pa - pb).n)


def within_k_months(a, b, k=6) -> bool:
    d = months_diff(a, b)
    return (d != np.inf) and (abs(d) <= k)


def rank_bins_pct(s: pd.Series, n_bins: int) -> pd.Series:
    pct = s.rank(method="average", pct=True)
    bins = np.ceil(pct * n_bins)
    bins = pd.to_numeric(bins, errors="coerce").clip(1, n_bins)
    bins = bins.where(s.notna())
    return bins.astype("Int16")


def make_case_mix_bins_and_dummies_quarter(panel: pd.DataFrame, cm_col: str, state_col: str = "state"):
    out = panel.copy()
    out[cm_col] = pd.to_numeric(out[cm_col], errors="coerce")

    out["cm_quart_nat"] = out.groupby(["year", "quarter"], observed=True)[cm_col].transform(
        lambda s: rank_bins_pct(s, 4)
    )
    out["cm_decil_nat"] = out.groupby(["year", "quarter"], observed=True)[cm_col].transform(
        lambda s: rank_bins_pct(s, 10)
    )

    if state_col in out.columns:
        mask = out[state_col].notna()
        out.loc[mask, "cm_quart_state"] = (
            out[mask]
            .groupby(["year", "quarter", state_col], observed=True)[cm_col]
            .transform(lambda s: rank_bins_pct(s, 4))
        ).astype("Int16")
        out.loc[mask, "cm_decil_state"] = (
            out[mask]
            .groupby(["year", "quarter", state_col], observed=True)[cm_col]
            .transform(lambda s: rank_bins_pct(s, 10))
        ).astype("Int16")
    else:
        out["cm_quart_state"] = pd.Series([pd.NA] * len(out), dtype="Int16")
        out["cm_decil_state"] = pd.Series([pd.NA] * len(out), dtype="Int16")

    def dums(df, col, prefix):
        miss = df[col].isna().astype("Int8").rename(f"{prefix}_missing")
        d = pd.get_dummies(df[col], prefix=prefix, dtype="Int8")
        ref = f"{prefix}_1"
        if ref in d.columns:
            d = d.drop(columns=[ref])
        return pd.concat([d, miss], axis=1)

    parts = []
    for col, pre in [
        ("cm_quart_nat", "cm_q_nat"),
        ("cm_decil_nat", "cm_d_nat"),
        ("cm_quart_state", "cm_q_state"),
        ("cm_decil_state", "cm_d_state"),
    ]:
        parts.append(dums(out, col, pre))

    out = pd.concat([out, pd.concat(parts, axis=1)], axis=1)
    return out


def filter_to_window_quarter(df: pd.DataFrame) -> pd.DataFrame:
    qp = year_quarter_period(df["year"], df["quarter"])
    mask = (qp >= pd.Period(START_Q, "Q")) & (qp <= pd.Period(END_Q, "Q"))
    return df.loc[mask].copy()


# ============================== Load ==========================================
quality = pd.read_csv(QUALITY_FP, low_memory=False)
provider_q = pd.read_csv(PROVIDER_Q_FP, low_memory=False)
pbj_q = pd.read_csv(PBJ_Q_FP, low_memory=False)
mcr_q = pd.read_csv(MCR_Q_FP, low_memory=False)
chow = pd.read_csv(CHOW_FP, low_memory=False)

for df in (quality, provider_q, pbj_q, mcr_q, chow):
    if "cms_certification_number" in df.columns:
        df["cms_certification_number"] = cfg.normalize_ccn_any(df["cms_certification_number"])

# normalize year/quarter fields across quarterly inputs
quarterly_dfs = [quality, provider_q, pbj_q, mcr_q]
for df in quarterly_dfs:
    if "year" not in df.columns or "quarter" not in df.columns:
        raise KeyError("Quarterly input missing 'year' or 'quarter'")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["quarter"] = normalize_quarter_string(df["quarter"])

# restrict quarterly inputs to the study window before merge
quality = filter_to_window_quarter(quality)
provider_q = filter_to_window_quarter(provider_q)
pbj_q = filter_to_window_quarter(pbj_q)
mcr_q = filter_to_window_quarter(mcr_q)

# ============================== Rename overlapping metadata ====================
provider_rename = {}
if "months_in_quarter" in provider_q.columns:
    provider_rename["months_in_quarter"] = "provider_months_in_quarter"
if "last_year_month_in_quarter" in provider_q.columns:
    provider_rename["last_year_month_in_quarter"] = "provider_last_year_month_in_quarter"
provider_q = provider_q.rename(columns=provider_rename)

pbj_rename = {}
if "months_observed_in_quarter" in pbj_q.columns:
    pbj_rename["months_observed_in_quarter"] = "pbj_months_observed_in_quarter"
if "last_year_month_in_quarter" in pbj_q.columns:
    pbj_rename["last_year_month_in_quarter"] = "pbj_last_year_month_in_quarter"
pbj_q = pbj_q.rename(columns=pbj_rename)

mcr_rename = {}
if "months_observed_in_quarter" in mcr_q.columns:
    mcr_rename["months_observed_in_quarter"] = "mcr_months_observed_in_quarter"
if "last_year_month_in_quarter" in mcr_q.columns:
    mcr_rename["last_year_month_in_quarter"] = "mcr_last_year_month_in_quarter"
mcr_q = mcr_q.rename(columns=mcr_rename)

# ============================== CHOW agreement filter =========================
chow["n_chow_nh_compare"] = pd.to_numeric(chow.get("n_chow_nh_compare"), errors="coerce").fillna(0).astype(int)
chow["n_chow_mcr"] = pd.to_numeric(chow.get("n_chow_mcr"), errors="coerce").fillna(0).astype(int)
chow["first_nh_month"] = first_chow_month(chow, r"^nh_compare_chow_\d+_date$")
chow["first_mcr_month"] = first_chow_month(chow, r"^mcr_chow_\d+_date$")

def _agree_row(r):
    if r["n_chow_nh_compare"] in (0, 1) and r["n_chow_mcr"] in (0, 1):
        if (r["n_chow_nh_compare"] == 0) and (r["n_chow_mcr"] == 0):
            return True
        if (r["n_chow_nh_compare"] == 1) and (r["n_chow_mcr"] == 1):
            return within_k_months(r["first_nh_month"], r["first_mcr_month"], k=6)
    return False

agree_mask = chow.apply(_agree_row, axis=1)
agree_ccns = set(chow.loc[agree_mask, "cms_certification_number"].dropna().unique())
print(f"[chow] CCNs passing (0/0 or 1/1 within 6m): {len(agree_ccns):,}")

chow_timing = (
    chow.loc[
        chow["cms_certification_number"].isin(agree_ccns),
        ["cms_certification_number", "n_chow_nh_compare", "n_chow_mcr", "first_nh_month", "first_mcr_month"],
    ]
    .drop_duplicates("cms_certification_number")
    .copy()
)

chow_timing["treated_agree"] = (
    (chow_timing["n_chow_nh_compare"] == 1) & (chow_timing["n_chow_mcr"] == 1)
).astype("Int8")

chow_timing["event_month"] = to_monthstart(chow_timing["first_mcr_month"])
event_parts = month_to_quarter_parts(chow_timing["event_month"])
chow_timing = pd.concat([chow_timing, event_parts], axis=1)

# ============================== Merge quarterly base ==========================
keys = ["cms_certification_number", "year", "quarter"]
for name, df in [("quality", quality), ("provider_q", provider_q), ("pbj_q", pbj_q), ("mcr_q", mcr_q)]:
    miss = [k for k in keys if k not in df.columns]
    if miss:
        raise KeyError(f"[{name}] missing key columns: {miss}")

base = (
    quality
    .merge(provider_q, on=keys, how="left")
    .merge(pbj_q, on=keys, how="left")
    .merge(mcr_q, on=keys, how="left")
)

base["cms_certification_number"] = cfg.normalize_ccn_any(base["cms_certification_number"])
base = base[base["cms_certification_number"].isin(agree_ccns)].copy()

base = base.merge(chow_timing, on="cms_certification_number", how="left")

# ============================== Treatment / Post / Event-time =================
qnum = quarter_num_from_label(base["quarter"])
base["time"] = ((base["year"].astype("Int64") - 2017) * 4 + qnum).astype("Int32")

base["treated"] = pd.to_numeric(base["treated_agree"], errors="coerce").fillna(0).astype("Int8")

base["event_time"] = np.nan
base["time_treated"] = pd.Series(pd.NA, index=base.index, dtype="Int32")

mask = base["treated"].eq(1) & base["event_year"].notna() & base["event_quarter_num"].notna()

curr_qi = (base.loc[mask, "year"].astype("Int64") * 4 + quarter_num_from_label(base.loc[mask, "quarter"])).astype("Int32")
event_qi = (base.loc[mask, "event_year"].astype("Int64") * 4 + base.loc[mask, "event_quarter_num"].astype("Int64")).astype("Int32")

base.loc[mask, "event_time"] = (curr_qi - event_qi).astype(int)
base.loc[mask, "time_treated"] = (
    ((base.loc[mask, "event_year"].astype("Int64") - 2017) * 4 + base.loc[mask, "event_quarter_num"].astype("Int64"))
    .astype("Int32")
)

base["post"] = 0
base.loc[mask, "post"] = (base.loc[mask, "event_time"] > 0).astype("Int8")

# ============================== Case-mix dummies ==============================
if "case_mix_total" not in base.columns:
    base["case_mix_total"] = pd.NA
base = make_case_mix_bins_and_dummies_quarter(base, cm_col="case_mix_total", state_col="state")

# ============================== Build initial panel ===========================
metric_cols = sorted([c for c in base.columns if c.startswith("qm_")])

want_cols = [
    "cms_certification_number",
    "year",
    "quarter",
    "time",
    "time_treated",
    "treated",
    "post",
    "event_time",
    "event_month",
    "event_year",
    "event_quarter",
    "first_nh_month",
    "first_mcr_month",

    # provider
    "provider_resides_in_hospital",
    "provider_months_in_quarter",
    "provider_last_year_month_in_quarter",
    "ccrc_facility",
    "sff_facility",
    "beds_prov",

    # pbj
    "rn_hprd",
    "lpn_hprd",
    "cna_hprd",
    "total_hprd",
    "days_reported_quarter",
    "days_in_quarter",
    "coverage_ratio",
    "pbj_months_observed_in_quarter",
    "pbj_last_year_month_in_quarter",
    "gap_from_prev_quarters",
    "pbj_partial_quarter",
    "pbj_low_coverage",
    "pbj_zero_rn_lpn",
    "pbj_implausible_hprd",
    "pbj_invalid_quarter",

    # mcr
    "non_profit",
    "government",
    "chain",
    "num_beds",
    "occupancy_rate",
    "spare_capacity",
    "pct_medicare",
    "pct_medicaid",
    "state",
    "urban",
    "mcr_months_observed_in_quarter",
    "mcr_last_year_month_in_quarter",

    # unified beds if already present
    "beds",
] + metric_cols

cm_dummy_cols_all = [
    c
    for c in base.columns
    if c.startswith(("cm_q_nat_", "cm_d_nat_", "cm_q_state_", "cm_d_state_"))
    or (c.endswith("_missing") and c.startswith(("cm_q_", "cm_d_")))
]
want_cols += [c for c in cm_dummy_cols_all if c not in want_cols]
want_cols = [c for c in want_cols if c in base.columns]

panel = base[want_cols].copy()

# ============================== Light cleanup =================================
if "state" in panel.columns:
    before = len(panel)
    panel = panel.loc[~panel["state"].isin(["AK", "HI"])].copy()
    print(f"[filter] drop AK/HI: {before:,} -> {len(panel):,}")

for c in [
    "rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd",
    "coverage_ratio", "occupancy_rate", "spare_capacity", "pct_medicare", "pct_medicaid",
    "num_beds", "beds_prov", "beds"
]:
    if c in panel.columns:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

for c in metric_cols:
    panel[c] = pd.to_numeric(panel[c], errors="coerce")

for col in [
    "non_profit", "government", "chain", "urban",
    "ccrc_facility", "sff_facility", "provider_resides_in_hospital",
    "post", "treated",
    "pbj_partial_quarter", "pbj_low_coverage", "pbj_zero_rn_lpn",
    "pbj_implausible_hprd", "pbj_invalid_quarter",
]:
    if col in panel.columns:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").astype("Int8")

# construct unified beds if absent
if "beds" not in panel.columns:
    nb = panel["num_beds"] if "num_beds" in panel.columns else pd.Series(np.nan, index=panel.index)
    bp = panel["beds_prov"] if "beds_prov" in panel.columns else pd.Series(np.nan, index=panel.index)

    use_bp = bp.where(bp >= 15)
    use_nb = nb.where(nb >= 15)

    beds_clean = use_bp.fillna(use_nb)
    fallback = nb.where(nb.notna(), bp)
    beds_clean = beds_clean.where(beds_clean.notna(), fallback)

    panel["beds"] = pd.to_numeric(beds_clean, errors="coerce")

if "beds" in panel.columns:
    before = len(panel)
    panel = panel.loc[~(panel["beds"] < 15)].copy()
    print(f"[filter] drop beds < 15: {before:,} -> {len(panel):,}")

if "provider_resides_in_hospital" in panel.columns:
    before = len(panel)
    panel = panel.loc[panel["provider_resides_in_hospital"] != 1].copy()
    print(f"[filter] drop provider_resides_in_hospital==1: {before:,} -> {len(panel):,}")

for c in ["pct_medicare", "pct_medicaid", "occupancy_rate"]:
    if c in panel.columns:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").clip(0, 100)

if "spare_capacity" in panel.columns:
    panel["spare_capacity"] = pd.to_numeric(panel["spare_capacity"], errors="coerce").clip(0, 1)

if "coverage_ratio" in panel.columns:
    panel["coverage_ratio"] = pd.to_numeric(panel["coverage_ratio"], errors="coerce").clip(0, 1)

if {"pct_medicare", "pct_medicaid"}.issubset(panel.columns):
    sums = panel["pct_medicare"] + panel["pct_medicaid"]
    too_high = sums > 100
    if too_high.any():
        scale = 100 / sums[too_high]
        panel.loc[too_high, "pct_medicare"] *= scale
        panel.loc[too_high, "pct_medicaid"] *= scale

# PBJ filtering using upstream QA flags
if "pbj_invalid_quarter" in panel.columns:
    before = len(panel)
    panel = panel.loc[panel["pbj_invalid_quarter"] != 1].copy()
    print(f"[filter] drop pbj_invalid_quarter==1: {before:,} -> {len(panel):,}")

if "pbj_implausible_hprd" in panel.columns:
    before = len(panel)
    panel = panel.loc[panel["pbj_implausible_hprd"] != 1].copy()
    print(f"[filter] drop pbj_implausible_hprd==1: {before:,} -> {len(panel):,}")

# Fallback if flags absent
if (
    "pbj_invalid_quarter" not in panel.columns
    and {"rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd"}.issubset(panel.columns)
):
    before = len(panel)
    mask_bad = (
        ((panel["rn_hprd"] == 0) & (panel["lpn_hprd"] == 0))
        | (panel["total_hprd"] < 1.5)
        | (panel["total_hprd"] > 12)
        | (panel["cna_hprd"] > 5.25)
    )
    panel = panel.loc[~mask_bad].copy()
    print(f"[filter] fallback PBJ plausibility filter: {before:,} -> {len(panel):,}")

# final ordering
qord = quarter_num_from_label(panel["quarter"])
panel = (
    panel.assign(_qord=qord)
    .sort_values(["cms_certification_number", "year", "_qord"], kind="mergesort")
    .drop(columns="_qord")
    .reset_index(drop=True)
)

cfg.atomic_overwrite_csv(panel, OUT_FINAL_FP, index=False)
print(
    f"[done] saved quality panel → {OUT_FINAL_FP} "
    f"rows={len(panel):,} cols={panel.shape[1]}"
)