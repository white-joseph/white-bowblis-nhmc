#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# surname_screen_diagnostic.py
#
# Quantifies how the family ("surname") screen in 04_chow.py changes the NHC
# ownership-change classification and the resulting treated / control sample.
#
# READ-ONLY with respect to the pipeline: nothing in data/ is modified. Results
# are written to outputs/diagnostics/.
#
# Rules compared (all use the same turnover threshold, layer priority and
# weight normalisation as 04_chow.py, imported from that file):
#
#   baseline : the screen exactly as 04_chow.py implements it today
#   fixed    : (i) person-vs-company is decided on the RAW name, before company
#              suffixes are stripped; (ii) the surname is taken from the correct
#              end of the name (auto-detected from the data); (iii) companies
#              are never matched as "family"
#   none     : no family screen (turnover threshold only)
#
# Outputs (outputs/diagnostics/):
#   surname_screen_summary.txt        full report (also printed)
#   surname_screen_events.csv         every NHC candidate change (turnover >= 50%)
#   surname_screen_review_sample.csv  events where baseline and fixed disagree,
#                                     with owner names, for manual reading
#   surname_screen_facilities.csv     facility-level status under each rule
#
# Usage (from the repo root or from src/):
#   python src/surname_screen_diagnostic.py
# =============================================================================
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------- settings ------------------------------------
NAME_ORDER = "auto"      # "auto" = detect from the data; or force "first" / "last"
MATCH_MONTHS = 6         # window for matching NHC and HCRIS dates (same as 06_panel.py)
REVIEW_SAMPLE_N = 200    # events written to the manual-review file
ORDER_SAMPLE_CCNS = 3000 # facilities sampled when detecting the name order
SEED = 12345

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
OUT_DIR = ROOT / "outputs" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- import the helpers from 04_chow.py ----------------------
# 04_chow.py only defines functions and constants at import time (main() is
# guarded by __name__), so importing it does not rerun the pipeline.
sys.path.insert(0, str(SRC))
_spec = importlib.util.spec_from_file_location("chow04", SRC / "04_chow.py")
chow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chow)
cfg = chow.cfg

OWNERSHIP_FP = Path(chow.OWNERSHIP_FP)
CHOW_FP = Path(chow.MERGE_OUT)
LOOKUPS_FP = Path(chow.INTERIM_DIR) / "facility_lookups.csv"

REPORT: list[str] = []


def say(line: str = "") -> None:
    print(line)
    REPORT.append(line)


# ============================ name handling ==================================
# Words that mark a name as a company, trust or other entity. Adjust if the
# review sample shows misclassified names.
ENTITY_WORDS = {
    "LLC", "INC", "INCORPORATED", "CORP", "CORPORATION", "LP", "LLP", "PLC", "CO",
    "COMPANY", "HOLDING", "HOLDINGS", "HOLDCO", "GROUP", "TRUST", "TR", "FBO",
    "REVOCABLE", "IRREVOCABLE", "LIVING", "ESTATE", "FUND", "CAPITAL", "PARTNERS",
    "PARTNERSHIP", "HEALTH", "HEALTHCARE", "CARE", "AUTHORITY", "HOSPITAL", "CENTER",
    "CENTRE", "NURSING", "HOME", "HOMES", "OPERATING", "OPERATIONS", "MANAGEMENT",
    "ENTERPRISE", "ENTERPRISES", "INVESTOR", "INVESTORS", "INVESTMENT", "INVESTMENTS",
    "PROPERTIES", "REALTY", "VENTURE", "VENTURES", "SERVICE", "SERVICES", "SYSTEM",
    "SYSTEMS", "ASSOCIATES", "FOUNDATION", "MINISTRIES", "SENIOR", "FAMILY", "ENTITIES",
    "GRAT", "UAD",
}
NAME_SUFFIXES = {"JR", "SR", "II", "III", "IV", "MD", "DDS", "DO", "PHD", "ESQ"}


def _tokens(raw: str) -> list[str]:
    s = re.sub(r"['\u2019]", "", str(raw))          # D'Angelo -> DAngelo (one token)
    s = re.sub(r"[^\w\s]", " ", s)
    return [t for t in s.upper().split() if t]


def person_tokens(raw: str) -> list[str] | None:
    """Tokens of a person-like name, else None (entity). Decided on the RAW name."""
    toks = _tokens(raw)
    joined = " ".join(toks)
    if re.search(r"\bL L C\b", joined):
        return None
    if any(t in ENTITY_WORDS or any(ch.isdigit() for ch in t) for t in toks):
        return None
    toks = [t for t in toks if t not in NAME_SUFFIXES]
    return toks if 2 <= len(toks) <= 3 else None


def build_norm_info(own: pd.DataFrame) -> dict:
    """normalised name -> {'person': bool, 'first': str, 'last': str, 'raw': str}."""
    pairs = own[["owner_name_norm", "owner_name"]].dropna().drop_duplicates()
    info: dict = {}
    for norm, raw in zip(pairs["owner_name_norm"], pairs["owner_name"]):
        toks = person_tokens(raw)
        rec = info.get(norm)
        if rec is None:
            info[norm] = {
                "person": toks is not None,
                "first": toks[0] if toks else "",
                "last": toks[-1] if toks else "",
                "raw": str(raw),
            }
        elif toks is None or not rec["person"] or toks[0] != rec["first"] or toks[-1] != rec["last"]:
            rec["person"] = False   # conflicting variants -> treat as entity
    return info


def detect_name_order(own: pd.DataFrame, info: dict) -> tuple[str, dict]:
    """Do co-owners in the same record share the FIRST or the LAST token?
    Family members share a surname, so the end that repeats is the surname."""
    rng = np.random.default_rng(SEED)
    ccns = own["cms_certification_number"].dropna().unique()
    take = rng.choice(ccns, size=min(ORDER_SAMPLE_CCNS, len(ccns)), replace=False)
    sub = own[own["cms_certification_number"].isin(take)]
    n_groups = n_first = n_last = 0
    for _, g in sub.groupby(["cms_certification_number", "association_date", "role"]):
        firsts, lasts = [], []
        for norm in g["owner_name_norm"].dropna().unique():
            rec = info.get(norm)
            if rec and rec["person"]:
                firsts.append(rec["first"])
                lasts.append(rec["last"])
        if len(firsts) < 2:
            continue
        n_groups += 1
        n_first += int(len(set(firsts)) < len(firsts))
        n_last += int(len(set(lasts)) < len(lasts))
    stats = {"groups_with_2plus_persons": n_groups, "share_first_token": n_first, "share_last_token": n_last}
    if n_first > 1.5 * max(n_last, 1):
        return "first", stats
    if n_last > 1.5 * max(n_first, 1):
        return "last", stats
    return "first", {**stats, "warning": "order not clearly identified; defaulting to 'first'"}


def make_fixed_family_fn(info: dict, order: str):
    key = "first" if order == "first" else "last"

    def surname_map(wm: dict) -> dict:
        agg: dict = {}
        for name, p in wm.items():
            rec = info.get(name)
            if rec and rec["person"]:
                agg[rec[key]] = agg.get(rec[key], 0.0) + p
        return agg

    def family_overlap(prev_w: dict, curr_w: dict) -> float:
        ps, cs = surname_map(prev_w), surname_map(curr_w)
        overlap = sum(min(ps.get(k, 0.0), cs.get(k, 0.0)) for k in set(ps) | set(cs))
        denom = max(sum(prev_w.values()), sum(curr_w.values()), 100.0)
        return max(0.0, min(100.0 * overlap / denom, 100.0))

    return family_overlap


# ============================ snapshots / events =============================
def build_snapshots(own: pd.DataFrame) -> pd.DataFrame:
    """Same construction as build_chow_from_ownership() in 04_chow.py."""
    rows = []
    for (ccn, adate), g in own.groupby(["cms_certification_number", "association_date"], sort=True):
        chosen = None
        for lvl in chow.LEVEL_PRIORITY:
            gl = g[g["level"] == lvl]
            if len(gl):
                chosen = (lvl, gl)
                break
        if chosen is None:
            continue
        lvl, gl = chosen
        vec = chow.normalize_weights_allow_missing(gl[["owner_name_norm", "ownership_percentage"]])
        if vec.empty:
            owners = gl["owner_name_norm"].dropna().unique().tolist()
            if not owners:
                continue
            equal = 100.0 / len(owners)
            vec = pd.DataFrame({"owner_name_norm": owners, "ownership_percentage": [equal] * len(owners)})
        wm = dict(zip(vec["owner_name_norm"], vec["ownership_percentage"].astype(float)))
        rows.append({"cms_certification_number": ccn, "association_date": adate,
                     "source_level": lvl, "weights": wm})
    return (pd.DataFrame(rows)
            .sort_values(["cms_certification_number", "association_date"])
            .reset_index(drop=True))


def top_names(wm: dict, info: dict, k: int = 3) -> str:
    items = sorted(wm.items(), key=lambda kv: -kv[1])[:k]
    return "; ".join(f"{info.get(n, {}).get('raw', n)} ({p:.0f}%)" for n, p in items)


def find_candidate_events(snaps: pd.DataFrame, info: dict, fixed_fn) -> pd.DataFrame:
    """All in-window transitions with turnover >= threshold, with the family
    overlap computed under both the baseline and the fixed rule."""
    recs = []
    for ccn, g in snaps.groupby("cms_certification_number", sort=True):
        g = g.sort_values("association_date").reset_index(drop=True)
        if len(g) < 2:
            continue
        prev_w = g.loc[0, "weights"]
        for i in range(1, len(g)):
            curr_w = g.loc[i, "weights"]
            to_start = g.loc[i, "association_date"]
            ov = chow.pct_overlap(prev_w, curr_w)
            turnover = 1.0 - ov if ov is not None else None
            if turnover is None or np.isnan(turnover):
                j = chow.jaccard_names(list(prev_w.keys()), list(curr_w.keys()))
                turnover = None if pd.isna(j) else 1.0 - float(j)
            in_window = pd.notna(to_start) and (to_start >= chow.CUTOFF_DATE)
            if in_window and turnover is not None and turnover >= chow.TURNOVER_THRESH:
                keep_b = chow.surname_family_overlap(prev_w, curr_w)
                keep_f = fixed_fn(prev_w, curr_w)
                base_suppressed = bool(chow.USE_SURNAME_OVERRIDE
                                       and keep_b >= 100.0 * chow.SURNAME_MIN_FRACTION_KEEP)
                fixed_suppressed = bool(keep_f >= 100.0 * chow.SURNAME_MIN_FRACTION_KEEP)
                recs.append({
                    "ccn": ccn,
                    "event_month": pd.Timestamp(to_start).to_period("M").to_timestamp(),
                    "turnover": float(turnover),
                    "keep_baseline": float(keep_b),
                    "keep_fixed": float(keep_f),
                    "counts_baseline": not base_suppressed,
                    "counts_fixed": not fixed_suppressed,
                    "counts_none": True,
                    "prev_owners": top_names(prev_w, info),
                    "curr_owners": top_names(curr_w, info),
                })
            prev_w = curr_w
    return pd.DataFrame(recs)


# ============================ sample rule ====================================
def month_index(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    return d.dt.year * 12 + d.dt.month


def agg_rule(ev: pd.DataFrame, col: str) -> pd.DataFrame:
    d = ev[ev[col]]
    if d.empty:
        return pd.DataFrame(columns=["ccn", "n", "first"]).set_index("ccn")
    return d.groupby("ccn").agg(n=("event_month", "size"), first=("event_month", "min"))


def status_series(n_nh, n_mcr, first_nh, first_mcr, k=MATCH_MONTHS) -> pd.Series:
    """Agreed sample rule from 06_panel.py."""
    gap = (month_index(first_nh) - month_index(first_mcr)).abs()
    control = (n_nh == 0) & (n_mcr == 0)
    treated = (n_nh == 1) & (n_mcr == 1) & (gap <= k)
    out = pd.Series("excluded", index=n_nh.index)
    out[control] = "control"
    out[treated] = "treated"
    return out


def bucket(n: pd.Series) -> pd.Series:
    return n.clip(upper=2).map({0: "0", 1: "1", 2: "2+"})


def crosstab_text(n_mcr: pd.Series, n_nh: pd.Series) -> str:
    ct = pd.crosstab(bucket(n_mcr).rename("HCRIS"), bucket(n_nh).rename("NHC"))
    ct = ct.reindex(index=["0", "1", "2+"], columns=["0", "1", "2+"], fill_value=0)
    return ct.to_string()


def pct(a, b) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


# ================================== main =====================================
def main() -> None:
    say("=" * 78)
    say("SURNAME-SCREEN SENSITIVITY: how the family screen changes the NHC sample")
    say("=" * 78)

    # ---- load -------------------------------------------------------------
    own = pd.read_csv(OWNERSHIP_FP, low_memory=False)
    own["cms_certification_number"] = cfg.normalize_ccn_any(own["cms_certification_number"])
    own["association_date"] = pd.to_datetime(own["association_date"], errors="coerce")
    own = own.dropna(subset=["association_date"]).copy()
    own["owner_name_norm"] = own["owner_name"].map(chow.clean_owner_name)
    own["level"] = own["role"].map(chow.level_bucket)

    cw = pd.read_csv(CHOW_FP, dtype={"cms_certification_number": str}, low_memory=False)
    cw["cms_certification_number"] = cfg.normalize_ccn_any(cw["cms_certification_number"])
    mcr_cols = [c for c in cw.columns if re.match(r"^mcr_chow_\d+_date$", c)]
    nh_cols = [c for c in cw.columns if re.match(r"^nh_compare_chow_\d+_date$", c)]
    for c in mcr_cols + nh_cols:
        cw[c] = pd.to_datetime(cw[c], errors="coerce")
    universe = pd.DataFrame({
        "ccn": cw["cms_certification_number"],
        "n_mcr": pd.to_numeric(cw["n_chow_mcr"], errors="coerce").fillna(0).astype(int),
        "first_mcr": cw[mcr_cols].min(axis=1) if mcr_cols else pd.NaT,
        "n_nh_chowcsv": pd.to_numeric(cw["n_chow_nh_compare"], errors="coerce").fillna(0).astype(int),
    }).set_index("ccn")
    say(f"Facilities in the HCRIS-and-NHC universe (chow.csv): {len(universe):,}")

    # ---- names ------------------------------------------------------------
    info = build_norm_info(own)
    order = NAME_ORDER
    if order == "auto":
        order, st = detect_name_order(own, info)
        say("")
        say("Name format check (do co-owners share the first or the last token?):")
        say(f"  records with 2+ person owners sampled : {st['groups_with_2plus_persons']:,}")
        say(f"  ... sharing the FIRST token           : {st['share_first_token']:,}")
        say(f"  ... sharing the LAST token            : {st['share_last_token']:,}")
        if "warning" in st:
            say(f"  WARNING: {st['warning']}")
        say(f"  -> surname taken from the {order.upper()} token in the 'fixed' rule")
        say("     ('first' means names are stored 'Last First', e.g. 'Smith John')")
    else:
        say(f"Name order forced to: {order}")
    fixed_fn = make_fixed_family_fn(info, order)

    # ---- events -----------------------------------------------------------
    say("")
    say("Building ownership snapshots and transitions (this can take several minutes)...")
    snaps = build_snapshots(own)
    ev = find_candidate_events(snaps, info, fixed_fn)
    if ev.empty:
        say("No candidate events found; nothing to compare.")
        return

    # HCRIS corroboration for each event (same facility, HCRIS date within +/- 6 months)
    mcr_idx = {}
    for ccn, row in cw.set_index("cms_certification_number")[mcr_cols].iterrows():
        vals = pd.to_datetime(row, errors="coerce").dropna()
        if len(vals):
            mcr_idx[ccn] = (vals.dt.year * 12 + vals.dt.month).to_numpy()
    in_universe = ev["ccn"].isin(universe.index)
    ev["in_universe"] = in_universe
    em = (ev["event_month"].dt.year * 12 + ev["event_month"].dt.month).to_numpy()
    hc = np.full(len(ev), np.nan)
    for i, (ccn, ok) in enumerate(zip(ev["ccn"], in_universe)):
        if ok:
            arr = mcr_idx.get(ccn)
            hc[i] = float(arr is not None and np.any(np.abs(arr - em[i]) <= MATCH_MONTHS))
    ev["hcris_within_6m"] = hc

    def cat(r) -> str:
        if r["counts_baseline"] and r["counts_fixed"]:
            return "1 counted under both rules"
        if r["counts_baseline"] and not r["counts_fixed"]:
            return "2 counted by baseline only"
        if (not r["counts_baseline"]) and r["counts_fixed"]:
            return "3 suppressed by baseline only"
        return "4 suppressed under both"
    ev["category"] = ev.apply(cat, axis=1)

    # ---- A. event level ---------------------------------------------------
    say("")
    say("-" * 78)
    say("A. EVENT LEVEL (NHC transitions since 2017 with >= 50% turnover)")
    say("-" * 78)
    n_ev = len(ev)
    say(f"Candidate events: {n_ev:,} across {ev['ccn'].nunique():,} facilities")
    say(f"  suppressed by the baseline family screen : {(~ev['counts_baseline']).sum():,} "
        f"({pct((~ev['counts_baseline']).sum(), n_ev)})")
    say(f"  suppressed by the fixed family screen    : {(~ev['counts_fixed']).sum():,} "
        f"({pct((~ev['counts_fixed']).sum(), n_ev)})")
    say(f"  facilities with at least one baseline-suppressed event: "
        f"{ev.loc[~ev['counts_baseline'], 'ccn'].nunique():,}")
    say(f"  facilities with at least one fixed-suppressed event   : "
        f"{ev.loc[~ev['counts_fixed'], 'ccn'].nunique():,}")
    say("")
    say("Events by how the two rules treat them, and how often HCRIS reports a")
    say("change for the same facility within +/- 6 months (facilities in the universe only):")
    tab = (ev[ev["in_universe"]].groupby("category")
           .agg(events=("ccn", "size"), hcris_corroborated=("hcris_within_6m", "mean")))
    tab["hcris_corroborated"] = (100 * tab["hcris_corroborated"]).round(1).astype(str) + "%"
    say(tab.to_string())
    say("")
    say("How to read it: HCRIS reports legal changes, so corroboration is a diagnostic,")
    say("not proof of a transfer of control. If category 3 (suppressed by baseline only)")
    say("is corroborated about as often as category 1, the baseline screen is hiding")
    say("events that look like substantive transfers. If category 2 (counted by baseline")
    say("only) is corroborated rarely, the baseline screen is counting events that look")
    say("like within-family restructuring.")

    # ---- B. facility level ------------------------------------------------
    say("")
    say("-" * 78)
    say("B. FACILITY LEVEL (universe = facilities in both HCRIS and NHC)")
    say("-" * 78)
    fac = universe.copy()
    for mode in ["baseline", "fixed", "none"]:
        a = agg_rule(ev, f"counts_{mode}")
        fac[f"n_nh_{mode}"] = a["n"].reindex(fac.index).fillna(0).astype(int)
        fac[f"first_nh_{mode}"] = a["first"].reindex(fac.index)
        fac[f"status_{mode}"] = status_series(
            fac[f"n_nh_{mode}"], fac["n_mcr"], fac[f"first_nh_{mode}"], fac["first_mcr"])

    mism = int((fac["n_nh_baseline"] != fac["n_nh_chowcsv"]).sum())
    say(f"Check: replicated baseline vs n_chow_nh_compare in chow.csv: "
        f"{mism:,} facilities differ out of {len(fac):,}")
    if mism:
        say("  WARNING: chow.csv may have been built with an earlier version of 04_chow.py,")
        say("  or the thresholds/cutoff differ. Rerun 04_chow.py before relying on these numbers.")

    for mode, label in [("baseline", "BASELINE (as coded)"),
                        ("fixed", "FIXED (raw-name person test, correct surname end, no company matches)"),
                        ("none", "NO FAMILY SCREEN")]:
        say("")
        say(f"{label}: ownership-change counts, rows = HCRIS, columns = NHC")
        say(crosstab_text(fac["n_mcr"], fac[f"n_nh_{mode}"]))
        s = fac[f"status_{mode}"].value_counts()
        say(f"  treated {s.get('treated', 0):,} | control {s.get('control', 0):,} | "
            f"excluded {s.get('excluded', 0):,}")

    say("")
    say("Facilities whose sample status changes, baseline -> fixed (rows = baseline):")
    tm = pd.crosstab(fac["status_baseline"], fac["status_fixed"])
    tm = tm.reindex(index=["treated", "control", "excluded"],
                    columns=["treated", "control", "excluded"], fill_value=0)
    say(tm.to_string())
    moved = int((fac["status_baseline"] != fac["status_fixed"]).sum())
    say(f"  {moved:,} of {len(fac):,} facilities ({pct(moved, len(fac))}) change status")
    c2e = int(((fac["status_baseline"] == "control") & (fac["status_fixed"] == "excluded")).sum())
    t2e = int(((fac["status_baseline"] == "treated") & (fac["status_fixed"] == "excluded")).sum())
    say(f"  baseline control -> fixed excluded: {c2e:,}  (the baseline screen hid an NHC-recorded")
    say("    change, so these are possibly contaminated controls under the baseline rule)")
    say(f"  baseline treated -> fixed excluded: {t2e:,}  (the baseline screen counted an event")
    say("    the fixed rule reads as within-family, or the fixed rule finds extra changes)")

    # ---- C. final analytic sample ----------------------------------------
    say("")
    say("-" * 78)
    say("C. EFFECT ON THE FINAL ANALYTIC SAMPLE (facility_lookups.csv, non-government)")
    say("-" * 78)
    if LOOKUPS_FP.exists():
        lk = pd.read_csv(LOOKUPS_FP, dtype={"cms_certification_number": str})
        lk["cms_certification_number"] = cfg.normalize_ccn_any(lk["cms_certification_number"])
        if "ever_government" in lk.columns:
            lk = lk[pd.to_numeric(lk["ever_government"], errors="coerce").fillna(0) != 1]
        final = set(lk["cms_certification_number"])
        fac["in_final_sample"] = fac.index.isin(final)
        out_of_sync = int((fac["in_final_sample"] & (fac["status_baseline"] == "excluded")).sum())
        if out_of_sync:
            say(f"WARNING: {out_of_sync:,} facilities in facility_lookups.csv are 'excluded' under the")
            say("  baseline rule, so chow.csv and the panel are out of sync. Rerun 04, 06 and the")
            say("  quarterly panel before relying on this section. They are dropped from the table below.")
        fs = fac[fac["in_final_sample"] & fac["status_baseline"].isin(["treated", "control"])]
        say(f"Facilities in the final analytic sample: {len(fs):,}  "
            f"(treated {(fs['status_baseline'] == 'treated').sum():,}, "
            f"control {(fs['status_baseline'] == 'control').sum():,})")
        say("Their status under the fixed rule (rows = baseline status):")
        tf = pd.crosstab(fs["status_baseline"], fs["status_fixed"])
        tf = tf.reindex(index=["treated", "control"],
                        columns=["treated", "control", "excluded"], fill_value=0)
        say(tf.to_string())
        lost = int(((fs["status_fixed"] == "excluded")).sum())
        swapped = int(((fs["status_baseline"] == "treated") & (fs["status_fixed"] == "control")).sum()
                      + ((fs["status_baseline"] == "control") & (fs["status_fixed"] == "treated")).sum())
        say(f"  would leave the sample (become excluded): {lost:,} ({pct(lost, len(fs))})")
        say(f"  would switch between treated and control : {swapped:,} ({pct(swapped, len(fs))})")
        enter = fac[(~fac["in_final_sample"]) & (fac["status_baseline"] == "excluded")
                    & (fac["status_fixed"].isin(["treated", "control"]))]
        say(f"  facilities that would newly qualify (before the hospital/AK-HI/beds/etc. "
            f"restrictions in 06_panel.py): {len(enter):,}")
        say(f"    of which treated: {(enter['status_fixed'] == 'treated').sum():,}")
    else:
        say(f"{LOOKUPS_FP.name} not found; skipping. Run 06_panel.py first.")

    # ---- save -------------------------------------------------------------
    ev_out = ev.copy()
    ev_out["event_month"] = ev_out["event_month"].dt.strftime("%Y-%m")
    ev_out.to_csv(OUT_DIR / "surname_screen_events.csv", index=False)
    fac.reset_index().to_csv(OUT_DIR / "surname_screen_facilities.csv", index=False)

    dis = ev_out[ev_out["category"].isin(["2 counted by baseline only", "3 suppressed by baseline only"])]
    rng = np.random.default_rng(SEED)
    parts = []
    for c, g in dis.groupby("category"):
        parts.append(g.sample(n=min(len(g), REVIEW_SAMPLE_N // 2), random_state=int(rng.integers(1e6))))
    review = pd.concat(parts) if parts else dis
    review = review[["category", "ccn", "event_month", "turnover", "keep_baseline",
                     "keep_fixed", "hcris_within_6m", "prev_owners", "curr_owners"]]
    review.to_csv(OUT_DIR / "surname_screen_review_sample.csv", index=False)

    say("")
    say("-" * 78)
    say("Caveat: the 'fixed' rule is a proposed correction, not ground truth. Its person-vs-company")
    say("test is a word list (ENTITY_WORDS in this file); read the review sample and extend the")
    say("list if company names are being read as people.")
    say(f"Files written to {OUT_DIR}")
    say("  Read surname_screen_review_sample.csv: for ~20 rows in each category, decide")
    say("  whether the baseline or the fixed rule handled the transfer correctly.")
    (OUT_DIR / "surname_screen_summary.txt").write_text("\n".join(REPORT), encoding="utf-8")


if __name__ == "__main__":
    main()
