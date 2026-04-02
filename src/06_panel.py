#!/usr/bin/env python
# coding: utf-8
# -----------------------------------------------------------------------------
# Build Final Panel (outer join) + CHOW filter + treatment/post/event time + case-mix
# -----------------------------------------------------------------------------

import os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================== Paths =========================================
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").is_dir() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

REPO        = PROJECT_ROOT
INTERIM     = REPO / "data" / "interim"
CLEAN_DIR   = REPO / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

PROVIDER_FP = INTERIM / "provider.csv"
PBJ_FP      = INTERIM / "pbj_nurse.csv"
MCR_FP      = INTERIM / "mcr.csv"
CHOW_FP     = INTERIM / "chow.csv"

# ============================== Spec Config ===================================
SPEC_NAME = "date_mcr"          # baseline, date_nhc_m1, date_nhc_p1, date_nhc_m2, date_nhc_p2, date_mcr
EVENT_DATE_SOURCE = "mcr"       # "nhc" or "mcr"
EVENT_SHIFT_MONTHS = 0          # -2, -1, 0, +1, +2

# ============================== Output Paths ==================================
if SPEC_NAME == "baseline":
    OUT_PBJ_FP   = CLEAN_DIR / "pbj_panel.csv"
    OUT_ANL_FP   = CLEAN_DIR / "analytical_panel.csv"
    OUT_FINAL_FP = CLEAN_DIR / "panel.csv"
else:
    OUT_PBJ_FP   = CLEAN_DIR / f"pbj_panel_{SPEC_NAME}.csv"
    OUT_ANL_FP   = CLEAN_DIR / f"analytical_panel_{SPEC_NAME}.csv"
    OUT_FINAL_FP = CLEAN_DIR / f"panel_{SPEC_NAME}.csv"

print(f"[paths] provider={PROVIDER_FP.exists()}  pbj={PBJ_FP.exists()}  mcr={MCR_FP.exists()}  chow={CHOW_FP.exists()}")
print(f"[spec]  SPEC_NAME={SPEC_NAME}  EVENT_DATE_SOURCE={EVENT_DATE_SOURCE}  EVENT_SHIFT_MONTHS={EVENT_SHIFT_MONTHS}")
print(f"[out]   pbj={OUT_PBJ_FP}")
print(f"[out]   analytical={OUT_ANL_FP}")
print(f"[out]   final_panel={OUT_FINAL_FP}")

# ============================== Window Config =================================
START_YM = "2017/01"
END_YM   = "2024/06"
START_Q  = "2017Q1"
END_Q    = "2024Q2"

# ============================== Helpers =======================================
def normalize_ccn_any(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    s = s.mask(is_digits, s.str.zfill(6))
    s = s.replace({"": pd.NA})
    return s

def to_monthstart(x) -> pd.Series:
    s = pd.to_datetime(x, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp("s")

def shift_monthstart(x, k: int) -> pd.Series:
    s = pd.to_datetime(x, errors="coerce").dt.to_period("M")
    return (s + k).dt.to_timestamp("s")

def first_chow_month(df: pd.DataFrame, patt: str) -> pd.Series:
    cols = [c for c in df.columns if re.search(patt, c, flags=re.I)]
    if not cols:
        return pd.Series(pd.NaT, index=df.index)
    tmp = df[cols].apply(pd.to_datetime, errors="coerce")
    return to_monthstart(tmp.min(axis=1))

def months_diff(a, b) -> float:
    if pd.isna(a) or pd.isna(b): return np.inf
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

def make_case_mix_bins_and_dummies(panel: pd.DataFrame, cm_col: str, state_col: str = "state"):
    out = panel.copy()
    out[cm_col] = pd.to_numeric(out[cm_col], errors="coerce")

    # National bins per month
    out["cm_quart_nat"] = out.groupby("year_month", observed=True)[cm_col].transform(lambda s: rank_bins_pct(s, 4))
    out["cm_decil_nat"] = out.groupby("year_month", observed=True)[cm_col].transform(lambda s: rank_bins_pct(s, 10))

    # State×month
    if state_col in out.columns:
        mask = out[state_col].notna()
        out.loc[mask, "cm_quart_state"] = (
            out[mask].groupby(["year_month", state_col], observed=True)[cm_col]
                     .transform(lambda s: rank_bins_pct(s, 4))
        ).astype("Int16")
        out.loc[mask, "cm_decil_state"] = (
            out[mask].groupby(["year_month", state_col], observed=True)[cm_col]
                     .transform(lambda s: rank_bins_pct(s, 10))
        ).astype("Int16")
    else:
        out["cm_quart_state"] = pd.Series([pd.NA]*len(out), dtype="Int16")
        out["cm_decil_state"] = pd.Series([pd.NA]*len(out), dtype="Int16")

    # Dummies
    def dums(df, col, prefix):
        miss = df[col].isna().astype("Int8").rename(f"{prefix}_missing")
        d = pd.get_dummies(df[col], prefix=prefix, dtype="Int8")
        ref = f"{prefix}_1"
        if ref in d.columns:
            d = d.drop(columns=[ref])
        return pd.concat([d, miss], axis=1)

    parts = []
    for col, pre in [("cm_quart_nat","cm_q_nat"), ("cm_decil_nat","cm_d_nat"),
                     ("cm_quart_state","cm_q_state"), ("cm_decil_state","cm_d_state")]:
        parts.append(dums(out, col, pre))
    out = pd.concat([out, pd.concat(parts, axis=1)], axis=1)
    return out

def filter_to_window(df: pd.DataFrame) -> pd.DataFrame:
    if "year_month" in df.columns:
        ym = pd.PeriodIndex(df["year_month"].astype(str), freq="M")
        mask_ym = (ym >= pd.Period(START_YM, "M")) & (ym <= pd.Period(END_YM, "M"))
    else:
        mask_ym = pd.Series(True, index=df.index)
    if "quarter" in df.columns:
        q = pd.PeriodIndex(df["quarter"].astype(str), freq="Q")
        mask_q = (q >= pd.Period(START_Q, "Q")) & (q <= pd.Period(END_Q, "Q"))
    else:
        mask_q = pd.Series(True, index=df.index)
    return df[mask_ym & mask_q].copy()

def coalesce_suffix_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = list(out.columns)
    suffixed = [c for c in cols if re.search(r"\.\d+$", c)]
    by_base = {}
    for c in suffixed:
        base = re.sub(r"\.\d+$", "", c)
        by_base.setdefault(base, []).append(c)
    for base, dups in by_base.items():
        if base in out.columns:
            for dup in dups:
                out[base] = out[base].where(out[base].notna(), out[dup])
            out = out.drop(columns=dups)
        else:
            out[base] = pd.NA
            for dup in dups:
                out[base] = out[base].where(out[base].notna(), out[dup])
            out = out.drop(columns=dups)
    return out

# -------- Fill NA only if surrounded by equal values --
def _fill_between_equal_series(s: pd.Series, numeric: bool = True, tol: float = 1e-9) -> pd.Series:
    arr = s.to_numpy(copy=True)
    n = len(arr)
    is_na = pd.isna(arr)
    i = 0
    while i < n:
        if not is_na[i]:
            i += 1
            continue
        start = i
        while i < n and is_na[i]:
            i += 1
        prev_val = arr[start - 1] if start - 1 >= 0 else np.nan
        next_val = arr[i] if i < n else np.nan
        if not pd.isna(prev_val) and not pd.isna(next_val):
            if numeric:
                equal = np.isclose(prev_val, next_val, atol=tol, rtol=0.0)
            else:
                equal = (prev_val == next_val)
            if equal:
                arr[start:i] = prev_val
    return pd.Series(arr, index=s.index)

def bridge_fill_equal(panel: pd.DataFrame, cols: list[str], group_key: str, numeric: bool) -> pd.DataFrame:
    if not cols:
        return panel
    panel = panel.sort_values([group_key, "year_month"], kind="mergesort")
    def _apply(g):
        for c in cols:
            if c in g.columns:
                g[c] = _fill_between_equal_series(g[c], numeric=numeric)
        return g
    return panel.groupby(group_key, group_keys=False, observed=True).apply(_apply)

def finalize_regression_panel(df: pd.DataFrame) -> pd.DataFrame:
    panel = df.copy()

    # ---------- prep / ordering ----------
    panel["_ord"] = pd.to_datetime(panel["year_month"].astype(str) + "/01", errors="coerce")
    panel = panel.sort_values(["cms_certification_number", "_ord"], kind="mergesort")
    g = panel.groupby("cms_certification_number", sort=False)

    def interp_within_fac(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        return s.interpolate(method="linear", limit_direction="both")

    # ---------- dummies: carry recent values ----------
    for col in ["non_profit", "government", "chain"]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
            panel[col] = g[col].transform(lambda s: s.ffill().bfill())
            panel[col] = panel[col].fillna(0).astype("Int8")

    # ---------- make key continuous vars numeric ----------
    for col in ["occupancy_rate", "pct_medicare", "pct_medicaid"]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    # ---------- occupancy: within-facility interpolation ----------
    if "occupancy_rate" in panel.columns:
        panel["occupancy_rate"] = g["occupancy_rate"].transform(interp_within_fac)

    # ---------- shares: within-facility interpolation ----------
    for col in ["pct_medicare", "pct_medicaid"]:
        if col in panel.columns:
            panel[col] = g[col].transform(interp_within_fac)

    # ---------- state×month medians ----------
    have_state = all(c in panel.columns for c in ["state", "year_month"])
    if have_state:
        sm = (
            panel.groupby(["state", "year_month"], dropna=False)[
                ["occupancy_rate", "pct_medicare", "pct_medicaid"]
            ]
            .median()
            .reset_index()
            .rename(
                columns={
                    "occupancy_rate": "occupancy_rate_sm",
                    "pct_medicare": "pct_medicare_sm",
                    "pct_medicaid": "pct_medicaid_sm",
                }
            )
        )
        panel = panel.merge(sm, on=["state", "year_month"], how="left")
    else:
        panel["occupancy_rate_sm"] = np.nan
        panel["pct_medicare_sm"] = np.nan
        panel["pct_medicaid_sm"] = np.nan

    # ---------- national medians ----------
    nat_occ = pd.to_numeric(panel.get("occupancy_rate"), errors="coerce").median(skipna=True)
    nat_mcr = pd.to_numeric(panel.get("pct_medicare"), errors="coerce").median(skipna=True)
    nat_mcd = pd.to_numeric(panel.get("pct_medicaid"), errors="coerce").median(skipna=True)

    # ---------- apply occupancy fallbacks ----------
    if "occupancy_rate" in panel.columns:
        m = panel["occupancy_rate"].isna()
        if m.any():
            panel.loc[m, "occupancy_rate"] = panel.loc[m, "occupancy_rate_sm"]
        m = panel["occupancy_rate"].isna()
        if m.any():
            panel.loc[m, "occupancy_rate"] = nat_occ
        panel["occupancy_rate"] = pd.to_numeric(panel["occupancy_rate"], errors="coerce").clip(0, 100)

    # ---------- fallback to state×month, then national ----------
    if "pct_medicare" in panel.columns:
        m = panel["pct_medicare"].isna()
        if m.any():
            panel.loc[m, "pct_medicare"] = panel.loc[m, "pct_medicare_sm"]
        m = panel["pct_medicare"].isna()
        if m.any():
            panel.loc[m, "pct_medicare"] = nat_mcr

    if "pct_medicaid" in panel.columns:
        m = panel["pct_medicaid"].isna()
        if m.any():
            panel.loc[m, "pct_medicaid"] = panel.loc[m, "pct_medicaid_sm"]
        m = panel["pct_medicaid"].isna()
        if m.any():
            panel.loc[m, "pct_medicaid"] = nat_mcd

    # ---------- if only one share is missing ----------
    if {"pct_medicare", "pct_medicaid"}.issubset(panel.columns):
        m = panel["pct_medicare"].isna() & panel["pct_medicaid"].notna()
        if m.any():
            candidate = 100 - panel.loc[m, "pct_medicaid"]
            if "pct_medicare_sm" in panel.columns:
                candidate = np.minimum(candidate, panel.loc[m, "pct_medicare_sm"].fillna(candidate))
            panel.loc[m, "pct_medicare"] = candidate

        m = panel["pct_medicaid"].isna() & panel["pct_medicare"].notna()
        if m.any():
            candidate = 100 - panel.loc[m, "pct_medicare"]
            if "pct_medicaid_sm" in panel.columns:
                candidate = np.minimum(candidate, panel.loc[m, "pct_medicaid_sm"].fillna(candidate))
            panel.loc[m, "pct_medicaid"] = candidate

    # ---------- final sanity ----------
    for col in ["pct_medicare", "pct_medicaid"]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce").clip(0, 100)

    if {"pct_medicare", "pct_medicaid"}.issubset(panel.columns):
        sums = panel["pct_medicare"] + panel["pct_medicaid"]
        too_high = sums > 100
        if too_high.any():
            scale = 100 / sums[too_high]
            panel.loc[too_high, "pct_medicare"] *= scale
            panel.loc[too_high, "pct_medicaid"] *= scale

    # ---------- final typing ----------
    for c in ["occupancy_rate", "pct_medicare", "pct_medicaid"]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")

    panel = panel.drop(
        columns=["_ord", "occupancy_rate_sm", "pct_medicare_sm", "pct_medicaid_sm"],
        errors="ignore",
    )
    panel = panel.sort_values(
        ["cms_certification_number", "year_month"], kind="mergesort"
    ).reset_index(drop=True)

    return panel

# ============================== Load ==========================================
provider = pd.read_csv(PROVIDER_FP, low_memory=False)
pbj      = pd.read_csv(PBJ_FP,      low_memory=False)
mcr      = pd.read_csv(MCR_FP,      low_memory=False)
chow     = pd.read_csv(CHOW_FP,     low_memory=False)

for df in (provider, pbj, mcr, chow):
    if "cms_certification_number" in df.columns:
        df["cms_certification_number"] = normalize_ccn_any(df["cms_certification_number"])

# Restrict window BEFORE merge
provider = filter_to_window(provider)
pbj      = filter_to_window(pbj)
mcr      = filter_to_window(mcr)

# ============================== CHOW agreement filter =========================
chow["n_chow_nh_compare"] = pd.to_numeric(chow.get("n_chow_nh_compare"), errors="coerce").fillna(0).astype(int)
chow["n_chow_mcr"]        = pd.to_numeric(chow.get("n_chow_mcr"),        errors="coerce").fillna(0).astype(int)
chow["first_nh_month"]    = first_chow_month(chow, r"^nh_compare_chow_\d+_date$")
chow["first_mcr_month"]   = first_chow_month(chow, r"^mcr_chow_\d+_date$")

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

# Keep BOTH timing sources in the panel so we can choose timing later without rebuilding CHOW logic.
chow_timing = (
    chow.loc[chow["cms_certification_number"].isin(agree_ccns),
             ["cms_certification_number", "n_chow_nh_compare", "n_chow_mcr", "first_nh_month", "first_mcr_month"]]
        .drop_duplicates("cms_certification_number")
        .copy()
)

# In the agreed sample, treated facilities are the validated 1/1 cases.
chow_timing["treatment_agree"] = (
    (chow_timing["n_chow_nh_compare"] == 1) & (chow_timing["n_chow_mcr"] == 1)
).astype("Int8")

# ============================== Outer join base ===============================
keys = ["cms_certification_number","quarter","year_month"]
for name, df in [("provider",provider),("pbj",pbj),("mcr",mcr)]:
    miss = [k for k in keys if k not in df.columns]
    if miss:
        raise KeyError(f"[{name}] missing key columns: {miss}")

base = provider.merge(pbj, on=keys, how="outer") \
               .merge(mcr, on=keys, how="outer")

# Keep only CHOW-agree CCNs
base["cms_certification_number"] = normalize_ccn_any(base["cms_certification_number"])
base = base[base["cms_certification_number"].isin(agree_ccns)].copy()

# Attach NH timing
base = base.merge(chow_timing, on="cms_certification_number", how="left")

# ============================== Treatment / Post / Event-time =================
ym_periods = pd.PeriodIndex(base["year_month"].astype(str), freq="M")

# Global month index starting at 2017/01 = 1
base["time"] = (ym_periods.year * 12 + ym_periods.month) - (2017 * 12 + 1) + 1
base["time"] = base["time"].astype("Int32")

# Standardize both source dates to month start
base["first_nh_month"]  = to_monthstart(base["first_nh_month"])
base["first_mcr_month"] = to_monthstart(base["first_mcr_month"])

# treatment stays fixed to the validated agreed sample
base["treatment"] = pd.to_numeric(base["treatment_agree"], errors="coerce").fillna(0).astype("Int8")

# Choose which raw event month to use
if EVENT_DATE_SOURCE == "nhc":
    base["event_month_unshifted"] = base["first_nh_month"]
elif EVENT_DATE_SOURCE == "mcr":
    base["event_month_unshifted"] = base["first_mcr_month"]
else:
    raise ValueError(f"EVENT_DATE_SOURCE must be 'nhc' or 'mcr', got: {EVENT_DATE_SOURCE}")

# Apply month shift for timing robustness
if EVENT_SHIFT_MONTHS != 0:
    base["event_month"] = shift_monthstart(base["event_month_unshifted"], EVENT_SHIFT_MONTHS)
else:
    base["event_month"] = to_monthstart(base["event_month_unshifted"])

# Store audit fields so later tables/plots always know what timing was used
base["event_date_source"] = EVENT_DATE_SOURCE
base["event_shift_months"] = EVENT_SHIFT_MONTHS

# Convert chosen event month to Period[M]
event_p = pd.to_datetime(base["event_month"], errors="coerce").dt.to_period("M")

# Event time only for validated treated facilities
base["event_time"] = np.nan
mask = base["treatment"].eq(1) & event_p.notna()

ym_y = pd.Series(ym_periods.year,  index=base.index)
ym_m = pd.Series(ym_periods.month, index=base.index)
ep_y = event_p.dt.year
ep_m = event_p.dt.month

et_vals = (ym_y[mask] - ep_y[mask]) * 12 + (ym_m[mask] - ep_m[mask])
base.loc[mask, "event_time"] = et_vals.astype(int)

# time_treated = global month index of chosen event month
base["time_treated"] = pd.Series(pd.NA, index=base.index, dtype="Int32")
tt_vals = (ep_y[mask] * 12 + ep_m[mask]) - (2017 * 12 + 1) + 1
base.loc[mask, "time_treated"] = tt_vals.astype("Int32")

# post = 1 strictly AFTER the chosen event month
base["post"] = 0
base.loc[mask, "post"] = (base.loc[mask, "event_time"] > 0).astype("Int8")

# anticipation1 = 1 for event_time in {-3,-2,-1,0,1,2}
base["anticipation1"] = 0
base.loc[base["event_time"].isin([-3, -2, -1, 0, 1, 2]), "anticipation1"] = 1

# anticipation2 = 1 for event_time in {-3,-2,-1}
base["anticipation2"] = 0
base.loc[base["event_time"].isin([-3, -2, -1]), "anticipation2"] = 1

# ============================== Case-mix dummies ==============================
if "case_mix_total" not in base.columns:
    base["case_mix_total"] = pd.NA
base = make_case_mix_bins_and_dummies(base, cm_col="case_mix_total", state_col="state")

# ============================== Build FINAL panel =============================
want_cols = [
    "cms_certification_number",
    "quarter",
    "year_month",
    "time",
    "time_treated",
    "treatment",
    "post",
    "event_time",
    "event_month",
    "event_month_unshifted",
    "first_nh_month",
    "first_mcr_month",
    "event_date_source",
    "event_shift_months",
    "anticipation1",
    "anticipation2",
    "provider_resides_in_hospital",
    "gap_from_prev_months",
    "coverage_ratio",
    "rn_hppd","lpn_hppd","cna_hppd","total_hppd",
    "non_profit","government",
    "chain",
    "num_beds",
    "beds_prov",
    "ccrc_facility","sff_facility",
    "occupancy_rate",
    "pct_medicare","pct_medicaid",
    "state",
    "urban",
]

cm_dummy_cols_all = [c for c in base.columns if c.startswith(("cm_q_nat_","cm_d_nat_","cm_q_state_","cm_d_state_")) or
                     (c.endswith("_missing") and c.startswith(("cm_q_","cm_d_")))]
want_cols += [c for c in cm_dummy_cols_all if c not in want_cols]

want_cols = [c for c in want_cols if c in base.columns]
panel = base[want_cols].copy()
panel = coalesce_suffix_duplicates(panel)

# ============================== Within-quarter fill (AFTER final panel) =======
binary_quarter_fill = [
    "provider_resides_in_hospital",
    "non_profit", "government",
    "chain",
    "ccrc_facility", "sff_facility",
    "urban",
] + [c for c in panel.columns if c.startswith(("cm_q_nat_","cm_d_nat_","cm_q_state_","cm_d_state_")) or
      (c.endswith("_missing") and c.startswith(("cm_q_","cm_d_")))]

numeric_quarter_fill = [
    "num_beds",
    "beds_prov",
    "occupancy_rate",
    "pct_medicare","pct_medicaid",
]

_fill_cols = [c for c in (binary_quarter_fill + numeric_quarter_fill) if c in panel.columns]
_key_cols  = ["cms_certification_number", "quarter"]

if _fill_cols:
    _orig_dtypes = panel[_fill_cols].dtypes.to_dict()
    filled_block = (
        panel.sort_values(_key_cols + ["year_month"])
             .groupby(_key_cols, observed=True, sort=False)[_fill_cols]
             .transform(lambda df: df.ffill().bfill())
    )
    panel[_fill_cols] = panel[_fill_cols].where(panel[_fill_cols].notna(), filled_block)
    # restore original dtypes where possible (dummies may remain floats until final cast)
    for c, dt in _orig_dtypes.items():
        try:
            panel[c] = panel[c].astype(dt)
        except Exception:
            pass

# ============================== Bridge-fill between equal endpoints ===========
bridge_numeric = [c for c in ["num_beds","beds_prov","occupancy_rate","pct_medicare","pct_medicaid"] if c in panel.columns]
panel = bridge_fill_equal(panel, bridge_numeric, group_key="cms_certification_number", numeric=True)

bridge_binary = [c for c in binary_quarter_fill if c in panel.columns]
panel = bridge_fill_equal(panel, bridge_binary, group_key="cms_certification_number", numeric=False)

# ============================== Unified 'beds' variable (override <15) ========
# Ensure numeric
for c in ["num_beds", "beds_prov"]:
    if c in panel.columns:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

nb  = panel["num_beds"]   if "num_beds"   in panel.columns else pd.Series(np.nan, index=panel.index)
bp  = panel["beds_prov"]  if "beds_prov"  in panel.columns else pd.Series(np.nan, index=panel.index)

# Choose >=15 if available; prefer provider value when it’s valid
use_bp = bp.where(bp >= 15)
use_nb = nb.where(nb >= 15)

# Start with beds_prov (>=15), else num_beds (>=15)
beds_clean = use_bp.fillna(use_nb)

# If neither candidate is >=15, fall back to whichever is non-missing (may be <15). 
fallback = nb.where(nb.notna(), bp)  # prefer num_beds if present, else beds_prov
beds_clean = beds_clean.where(beds_clean.notna(), fallback)

panel["beds"] = pd.to_numeric(beds_clean, errors="coerce")

# Move 'beds' to be immediately after 'beds_prov' for readability
if "beds" in panel.columns:
    cols = list(panel.columns)
    if "beds_prov" in cols:
        cols.remove("beds")
        insert_at = cols.index("beds_prov") + 1
        cols.insert(insert_at, "beds")
        panel = panel[cols]

# ============================== GAP indicator =================================
if "gap_from_prev_months" in panel.columns:
    panel["gap"] = (panel["gap_from_prev_months"] > 0).groupby(panel["cms_certification_number"]).transform("max").astype("Int8")
else:
    panel["gap"] = pd.Series([pd.NA]*len(panel), dtype="Int8")

# ============================== Final types, save PBJ panel ===================
for c in ["num_beds","beds_prov","beds","occupancy_rate","pct_medicare","pct_medicaid"]:
    if c in panel.columns:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

for col in ["non_profit","government","chain","urban","ccrc_facility","sff_facility",
            "provider_resides_in_hospital","gap","post","treatment","anticipation1","anticipation2"]:
    if col in panel.columns:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").astype("Int8")

panel = panel.sort_values(["cms_certification_number","year_month"], kind="mergesort").reset_index(drop=True)

# Always write spec-specific file
panel.to_csv(OUT_PBJ_FP, index=False)
print(f"[save] PBJ panel ({SPEC_NAME}) → {OUT_PBJ_FP} rows={len(panel):,} cols={panel.shape[1]}")

# ============================== Analytical panel ==============================
analytical = panel.copy()
analytical = analytical.replace(r"^\s*$", np.nan, regex=True)

# Drop rows with any NaN in HPPDs
hppd_cols = [c for c in ["rn_hppd","lpn_hppd","cna_hppd","total_hppd"] if c in analytical.columns]
if hppd_cols:
    before = len(analytical)
    analytical = analytical.dropna(subset=hppd_cols)
    print(f"[filter] drop rows with NaN in any HPPD: {before:,} -> {len(analytical):,}")

# ---- HPPD cleaning rules ----
# Drop rows where:
#   (1) rn_hppd == 0 AND lpn_hppd == 0
#   (2) total_hppd < 1.5  OR  total_hppd > 12
#   (3) cna_hppd > 5.25

if {"rn_hppd", "lpn_hppd", "cna_hppd", "total_hppd"}.issubset(analytical.columns):
    # ensure numeric
    for c in ["rn_hppd", "lpn_hppd", "cna_hppd", "total_hppd"]:
        analytical[c] = pd.to_numeric(analytical[c], errors="coerce")

    before = len(analytical)

    # construct masks
    m_joint_zero = (analytical["rn_hppd"] == 0) & (analytical["lpn_hppd"] == 0)
    m_total_low  = analytical["total_hppd"] < 1.5
    m_total_high = analytical["total_hppd"] > 12
    m_cna_high   = analytical["cna_hppd"] > 5.25

    # combine
    zmask = m_joint_zero | m_total_low | m_total_high | m_cna_high

    analytical = analytical.loc[~zmask].copy()

    print(
        f"[filter] drop implausible HPPD: {before:,} -> {len(analytical):,}  "
        f"(joint_zero={m_joint_zero.sum():,}, total<1.5={m_total_low.sum():,}, "
        f"total>12={m_total_high.sum():,}, cna>5.25={m_cna_high.sum():,})"
    )

# ---- Drop implausible bed counts (<15) ----
if "beds" in analytical.columns:
    before = len(analytical)
    analytical = analytical.loc[~(analytical["beds"] < 15)].copy()
    print(f"[filter] drop beds < 15: {before:,} -> {len(analytical):,}")

# Drop hospital-resident rows
if "provider_resides_in_hospital" in analytical.columns:
    before = len(analytical)
    analytical = analytical[analytical["provider_resides_in_hospital"] != 1]
    print(f"[filter] drop 'provider_resides_in_hospital'==1: {before:,} -> {len(analytical):,}")

# Always write spec-specific analytical panel
analytical.to_csv(OUT_ANL_FP, index=False)
print(f"[done] saved analytical panel ({SPEC_NAME}) → {OUT_ANL_FP}")

# ============================== Final regression panel ========================
final_panel = finalize_regression_panel(analytical)

final_panel.to_csv(OUT_FINAL_FP, index=False)
print(f"[done] saved final regression panel ({SPEC_NAME}) → {OUT_FINAL_FP} rows={len(final_panel):,} cols={final_panel.shape[1]}")