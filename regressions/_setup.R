# =============================================================================
# regressions/_setup.R
#
# Shared setup for BOTH analysis panels.
#
# Notes:
# - Canonical panel sources: data/clean/staffing_panel.csv (facility-month)
#   and data/clean/quality_panel.csv (facility-quarter)
# - load_staffing_panel() and load_quality_panel() are the only sanctioned
#   entry points. Both route through apply_facility_lookups(), so the
#   government-ever exclusion, chain_at_start, and CCN normalization are
#   defined once and cannot drift between the two samples.
# - Assumes MCR timing is already the baseline in the panel
# - Assumes staffing variables use *_hprd naming
# - Keeps this file focused on shared setup/helpers, not estimation loops
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(fixest)
  library(stringr)
  library(tibble)
})

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
project_root <- "C:/Repositories/white-bowblis-nhmc"

panel_fp <- file.path(project_root, "data", "clean", "staffing_panel.csv")
quality_panel_fp <- file.path(project_root, "data", "clean", "quality_panel.csv")
out_tables_dir <- file.path(project_root, "outputs", "tables")
out_plots_dir  <- file.path(project_root, "outputs", "plots")

dir.create(out_tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_plots_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Core variable sets
# -----------------------------------------------------------------------------
staffing_outcomes <- c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd")

log_outcome_map <- c(
  rn_hprd    = "ln_rn",
  lpn_hprd   = "ln_lpn",
  cna_hprd   = "ln_cna",
  total_hprd = "ln_total"
)

# Raw PBJ hours (numerator only, not divided by resident-days) -- added
# alongside HPRD per advisor request, to test whether occupancy-driven
# denominator changes are mechanically responsible for the HPRD results.
# Same construction as HPRD's log_outcome_map, kept as a separate map so
# code that loops over log_outcome_map (assuming exactly the 4 HPRD vars)
# is unaffected.
raw_hours_outcomes <- c("rn_hours_month", "lpn_hours_month", "cna_hours_month", "total_hours")

log_raw_hours_map <- c(
  rn_hours_month   = "ln_rn_hours",
  lpn_hours_month  = "ln_lpn_hours",
  cna_hours_month  = "ln_cna_hours",
  total_hours      = "ln_total_hours"
)

# ---------------------------------------------------------------------------
# Quality measure sets (quarterly panel). Previously each quality script
# (quality_event_study.R, quarterly_quality_twfe_tables.R,
# quarterly_summary_stats.R, composition_checks.R) carried its own private
# copy of these code->label maps, which is how they drifted apart. Single
# definition here; scripts should reference these rather than re-declaring.
#
# Vaccination measures (qm_430 pneumococcal, qm_472 influenza) are recorded
# here but deliberately NOT in the main long/short-stay sets, per CM and
# Bowblis's joint decision to omit them from the paper's quality outcomes.
# ---------------------------------------------------------------------------
long_stay_quality_measures <- c(
  qm_406 = "Catheter use",
  qm_419 = "Anti-psychotic medication use",
  qm_452 = "Anti-anxiety/hypnotic medication use",
  qm_453 = "Pressure injuries",
  qm_410 = "Falls with major injury",
  qm_404 = "Weight loss",
  qm_401 = "Decline in physical functioning",
  qm_407 = "Urinary tract infections"
)

# Labor-saving mechanism vs. resident outcome grouping used by the paper's
# two multi-panel quality figures and by the quality tables.
quality_mechanism_measures <- c("qm_406", "qm_419", "qm_452")
quality_outcome_measures   <- c("qm_453", "qm_410", "qm_404", "qm_401", "qm_407")

short_stay_quality_measures <- c(
  qm_434 = "New antipsychotic medication",
  qm_471 = "Improved function"
)

vaccination_quality_measures <- c(
  qm_430 = "Pneumococcal vaccine (short-stay)",
  qm_472 = "Influenza vaccine (short-stay)"
)

# Reporting-window trims established by the measure-code investigation:
# qm_424/qm_425 are effectively discontinued (excluded entirely above);
# qm_471 and qm_472 exist only over the windows below. NA = no trim.
#
# qm_453 (pressure injuries) trim recovered from the retired
# quarterly_quality_twfe_tables.R, which used 2018Q1--2023Q3 specifically.
# trim_quality_measure_window() only filters by YEAR, not quarter, so
# year_max = 2023 here includes all of 2023 (through Q4), three months
# past the original 2023Q3 cutoff. This is a deliberate approximation, not
# a rediscovery of the original evidence at quarter precision -- if the Q4
# reporting gap turns out to be as severe as Q1-Q3, tighten this by adding
# quarter bounds to trim_quality_measure_window() rather than assuming
# year-level trimming is precise enough.
quality_measure_year_windows <- tibble::tribble(
  ~var,     ~year_min,   ~year_max,
  "qm_453", 2018L,       2023L,
  "qm_471", NA_integer_, 2022L,
  "qm_472", 2018L,       2023L
)

base_controls <- c(
  "government",
  "non_profit",
  "chain",
  "beds",
  "occupancy_rate",
  "pct_medicare",
  "pct_medicaid"
)

# Prefer state-based case-mix controls if present
preferred_case_mix_controls <- c(
  "cm_q_state_2",
  "cm_q_state_3",
  "cm_q_state_4"
)

# Fallback national case-mix controls if needed
fallback_case_mix_controls <- c(
  "cm_q_nat_2",
  "cm_q_nat_3",
  "cm_q_nat_4"
)

# Common fixed effects / vcov
fe_unit <- "cms_certification_number"
fe_time <- "year_month"
cluster_var <- "cms_certification_number"

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
mk_log <- function(x) {
  ifelse(is.na(x) | x <= 0, NA_real_, log(x))
}

assert_has_cols <- function(df, cols, df_name = "data") {
  missing_cols <- setdiff(cols, names(df))
  if (length(missing_cols) > 0) {
    stop(
      sprintf(
        "[%s] missing required columns: %s",
        df_name,
        paste(missing_cols, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

intersect_existing <- function(x, df) {
  intersect(x, names(df))
}

# ---------------------------------------------------------------------------
# CCN normalization.
#
# staffing_panel.csv and quality_panel.csv are read separately, and readr
# may type the CCN column differently in each (numeric in one, character in
# the other) depending on what happens to be in the first rows it sniffs.
# A numeric parse silently strips leading zeros, so "015009" and 15009 would
# fail to join even though they are the same facility. Every facility-level
# key in this project passes through here first so both panels agree.
#
# Falls back to a trimmed character value for anything non-numeric rather
# than producing NA, so a genuinely alphanumeric identifier is preserved
# instead of being silently dropped.
# ---------------------------------------------------------------------------
norm_ccn <- function(x) {
  x_chr <- trimws(as.character(x))
  x_num <- suppressWarnings(as.numeric(x_chr))
  ifelse(
    is.na(x_num),
    x_chr,
    formatC(x_num, width = 6, flag = "0", format = "d")
  )
}

# -----------------------------------------------------------------------------
# Facility-level derived attributes -- computed ONCE, from the monthly panel
# -----------------------------------------------------------------------------
# Two facility-level attributes define the sample and a control across BOTH
# panels:
#
#   1. ever_government -- the exclusion set (per C. Moul / J. Bowblis: drop
#      any facility government-owned at any point in the panel).
#   2. chain_at_start  -- baseline chain status (per Bowblis: the
#      time-varying `chain` variable is unreliable, the baseline is not).
#
# Both are derived from the MONTHLY panel and then JOINED onto whichever
# panel is being loaded. They are deliberately NOT recomputed from the
# quarterly panel. Recomputing gives different answers for the same
# facility: a facility can be observed as government in a month that
# survives into staffing_panel.csv but whose quarters are sparse or absent
# in quality_panel.csv, and a quarterly chain baseline anchors on 2017Q1
# rather than 2017/01. Per-panel recomputation is exactly how the staffing
# and quality samples drifted apart, so there is one definition, one source.
#
# Only four columns are read to build these, so the cost is a small fraction
# of a full panel load. Cached in .nhmc_cache so a script that loads both
# panels pays for it once. Pass refresh = TRUE after rebuilding the panels.
# -----------------------------------------------------------------------------
.nhmc_cache <- new.env(parent = emptyenv())

build_facility_lookups <- function(fp = panel_fp, refresh = FALSE) {
  if (!refresh && !is.null(.nhmc_cache$facility_lookups)) {
    return(.nhmc_cache$facility_lookups)
  }

  if (!file.exists(fp)) {
    stop(sprintf("Panel file not found: %s", fp), call. = FALSE)
  }

  raw <- readr::read_csv(
    fp,
    col_select = dplyr::any_of(
      c("cms_certification_number", "year_month", "government", "chain")
    ),
    guess_max = Inf,
    show_col_types = FALSE,
    progress = FALSE
  )

  assert_has_cols(
    raw,
    c("cms_certification_number", "year_month"),
    "facility_lookups"
  )

  raw <- raw %>%
    dplyr::mutate(
      cms_certification_number = norm_ccn(cms_certification_number),
      year_month = as.character(year_month),
      ym_date = as.Date(paste0(year_month, "/01"), format = "%Y/%m/%d")
    )

  ever_government <- if ("government" %in% names(raw)) {
    raw %>%
      dplyr::filter(government == 1) %>%
      dplyr::distinct(cms_certification_number) %>%
      dplyr::pull(cms_certification_number)
  } else {
    character(0)
  }

  chain_lookup <- if ("chain" %in% names(raw)) {
    raw %>%
      dplyr::arrange(cms_certification_number, ym_date) %>%
      dplyr::group_by(cms_certification_number) %>%
      dplyr::summarise(
        chain_jan2017  = chain[year_month == "2017/01"][1],
        chain_earliest = chain[!is.na(chain)][1],
        .groups = "drop"
      ) %>%
      dplyr::mutate(chain_at_start = dplyr::coalesce(chain_jan2017, chain_earliest)) %>%
      dplyr::select(cms_certification_number, chain_at_start)
  } else {
    tibble::tibble(
      cms_certification_number = character(0),
      chain_at_start = integer(0)
    )
  }

  n_fallback <- 0L
  if ("chain" %in% names(raw)) {
    tmp <- raw %>%
      dplyr::group_by(cms_certification_number) %>%
      dplyr::summarise(
        has_jan2017 = any(year_month == "2017/01" & !is.na(chain)),
        has_any     = any(!is.na(chain)),
        .groups = "drop"
      )
    n_fallback <- sum(!tmp$has_jan2017 & tmp$has_any)
  }

  lookups <- list(
    ever_government    = ever_government,
    chain_lookup       = chain_lookup,
    n_chain_fallback   = n_fallback,
    n_facilities_total = dplyr::n_distinct(raw$cms_certification_number),
    source_fp          = fp
  )

  message(sprintf(
    "[setup] facility lookups built from %s: %d facilities, %d ever government-owned, %d chain_at_start fallbacks",
    basename(fp), lookups$n_facilities_total, length(ever_government), n_fallback
  ))

  .nhmc_cache$facility_lookups <- lookups
  lookups
}

# Apply the shared lookups to a panel: normalize the key, drop the
# ever-government facilities, join chain_at_start, factor the key. Used by
# BOTH loaders so the two panels cannot disagree about which facilities are
# in the sample or what their baseline chain status is.
apply_facility_lookups <- function(df, panel_label = "panel") {
  lookups <- build_facility_lookups()

  df <- df %>%
    dplyr::mutate(cms_certification_number = norm_ccn(cms_certification_number))

  n_before <- dplyr::n_distinct(df$cms_certification_number)

  if (length(lookups$ever_government) > 0) {
    df <- df %>%
      dplyr::filter(!(cms_certification_number %in% lookups$ever_government))
  }

  n_after <- dplyr::n_distinct(df$cms_certification_number)

  message(sprintf(
    "[%s] dropped %d facilities ever government-owned (%d -> %d facilities)",
    panel_label, n_before - n_after, n_before, n_after
  ))

  if (nrow(lookups$chain_lookup) > 0) {
    df <- df %>%
      dplyr::left_join(lookups$chain_lookup, by = "cms_certification_number")

    n_missing <- dplyr::n_distinct(
      df$cms_certification_number[is.na(df$chain_at_start)]
    )
    if (n_missing > 0) {
      message(sprintf(
        "[%s] %d facilities have no chain_at_start (present in this panel but not in the monthly panel)",
        panel_label, n_missing
      ))
    }
  }

  df %>%
    dplyr::mutate(cms_certification_number = as.factor(cms_certification_number))
}

# Diagnostic: do the two panels agree on the facility set after the shared
# exclusion is applied? Any disagreement is a data-pipeline issue, not a
# regression issue, and should be visible rather than silently absorbed
# into differing table Ns.
compare_panel_samples <- function() {
  s <- load_staffing_panel()
  q <- load_quality_panel()

  s_ccn <- unique(as.character(s$cms_certification_number))
  q_ccn <- unique(as.character(q$cms_certification_number))

  cat(sprintf("Staffing panel:  %d facilities, %s rows\n",
              length(s_ccn), format(nrow(s), big.mark = ",")))
  cat(sprintf("Quality panel:   %d facilities, %s rows\n",
              length(q_ccn), format(nrow(q), big.mark = ",")))
  cat(sprintf("In both:         %d\n", length(intersect(s_ccn, q_ccn))))
  cat(sprintf("Staffing only:   %d\n", length(setdiff(s_ccn, q_ccn))))
  cat(sprintf("Quality only:    %d\n", length(setdiff(q_ccn, s_ccn))))

  invisible(list(
    staffing_only = setdiff(s_ccn, q_ccn),
    quality_only  = setdiff(q_ccn, s_ccn)
  ))
}

# -----------------------------------------------------------------------------
# Panel loaders
# -----------------------------------------------------------------------------
load_staffing_panel <- function(fp = panel_fp) {
  if (!file.exists(fp)) {
    stop(sprintf("Panel file not found: %s", fp), call. = FALSE)
  }
  
  # guess_max = Inf: readr's default (guess_max = 1000) infers each column's
  # type from only its first 1,000 rows. The panel is sorted by CCN then
  # year_month, so a column that happens to be blank for the first ~1,000
  # rows (e.g. an early-alphabetical facility with a long reporting gap) gets
  # typed as logical instead of double -- every real numeric value later in
  # that column then silently fails to parse and becomes NA. This is exactly
  # what happened to total_hours: rn/lpn/cna_hours_month guessed correctly
  # (type is inferred independently per column) while total_hours alone came
  # back 100% NA, confirmed by total_hprd (built from the same source rows)
  # being fully populated. guess_max = Inf costs an extra pass over the file
  # but is the only fix that doesn't depend on knowing in advance which
  # column will be hit next.
  df <- readr::read_csv(fp, guess_max = Inf, show_col_types = FALSE)
  
  required_cols <- c(
    "cms_certification_number",
    "year_month",
    "quarter",
    "treated",
    "post",
    "event_time"
  )
  assert_has_cols(df, required_cols, "staffing_panel")
  
  # Core types. The CCN is left as a normalized character key here and only
  # converted to a factor at the end of apply_facility_lookups(), after the
  # facility-level joins have happened.
  df <- df %>%
    mutate(
      cms_certification_number = norm_ccn(cms_certification_number),
      year_month = as.character(year_month),
      quarter = as.character(quarter),
      ym_date = as.Date(paste0(year_month, "/01"), format = "%Y/%m/%d")
    )
  
  # Numeric coercion for key variables if present
  numeric_candidates <- c(
    staffing_outcomes,
    raw_hours_outcomes,
    "beds",
    "occupancy_rate",
    "pct_medicare",
    "pct_medicaid",
    "time",
    "time_treated",
    "event_time",
    "coverage_ratio",
    "gap_from_prev_months"
  )
  
  numeric_candidates <- intersect_existing(numeric_candidates, df)
  
  if (length(numeric_candidates) > 0) {
    df <- df %>%
      mutate(across(all_of(numeric_candidates), ~ suppressWarnings(as.numeric(.x))))
  }
  
  # Integer-ish treatment indicators if present
  binary_candidates <- c(
    "treated",
    "post",
    "government",
    "non_profit",
    "chain",
    "urban",
    "gap",
    "provider_resides_in_hospital",
    "ccrc_facility",
    "sff_facility"
  )
  
  binary_candidates <- intersect_existing(binary_candidates, df)
  
  if (length(binary_candidates) > 0) {
    df <- df %>%
      mutate(across(all_of(binary_candidates), ~ suppressWarnings(as.integer(.x))))
  }
  
  # Safe logs for staffing outcomes (HPRD)
  for (nm in names(log_outcome_map)) {
    if (nm %in% names(df)) {
      df[[log_outcome_map[[nm]]]] <- mk_log(df[[nm]])
    }
  }

  # Safe logs for raw hours (numerator only) -- guarded so this doesn't
  # break on an older staffing_panel.csv that predates these columns.
  for (nm in names(log_raw_hours_map)) {
    if (nm %in% names(df)) {
      df[[log_raw_hours_map[[nm]]]] <- mk_log(df[[nm]])
    }
  }

  # Government exclusion and chain_at_start are applied from the shared
  # facility-level lookups (see build_facility_lookups above) rather than
  # recomputed here, so the staffing and quality panels cannot diverge.
  apply_facility_lookups(df, panel_label = "staffing")
}

# -----------------------------------------------------------------------------
# Quarterly quality panel loader.
#
# Mirrors load_staffing_panel(): same government exclusion, same
# chain_at_start, same CCN normalization -- all sourced from the SAME
# facility-level lookups rather than recomputed from quarterly data.
#
# Adds a `year_quarter` key (e.g. "2017Q1") for use as the calendar fixed
# effect and clustering dimension, matching what the existing quality
# scripts construct by hand.
#
# KNOWN GAP (carried over from nested_control_spec_all_outcomes.R):
# quality_panel.csv has no avg_los_total column, so Spec C/D controls for
# quality outcomes silently omit average length of stay via
# intersect_existing()'s tolerance until a quarterly avg_los_total is
# merged in from the monthly panel. Warned about below rather than left
# to be discovered from a coefficient table.
# -----------------------------------------------------------------------------
load_quality_panel <- function(fp = quality_panel_fp) {
  if (!file.exists(fp)) {
    stop(sprintf("Quality panel file not found: %s", fp), call. = FALSE)
  }

  # guess_max = Inf -- see load_staffing_panel() for why. Applied here too
  # since the quarterly panel is subject to the exact same readr sampling
  # behavior even though the total_hours bug itself was only confirmed on
  # the monthly panel.
  df <- readr::read_csv(fp, guess_max = Inf, show_col_types = FALSE)

  required_cols <- c(
    "cms_certification_number",
    "year",
    "quarter",
    "treated",
    "post",
    "event_time"
  )
  assert_has_cols(df, required_cols, "quality_panel")

  df <- df %>%
    mutate(
      cms_certification_number = norm_ccn(cms_certification_number),
      year = suppressWarnings(as.integer(year)),
      quarter = toupper(trimws(as.character(quarter))),
      year_quarter = paste0(year, quarter)
    )

  numeric_candidates <- c(
    staffing_outcomes,
    names(long_stay_quality_measures),
    names(short_stay_quality_measures),
    names(vaccination_quality_measures),
    "beds",
    "num_beds",
    "occupancy_rate",
    "pct_medicare",
    "pct_medicaid",
    "time",
    "time_treated",
    "event_time",
    "coverage_ratio",
    "gap_from_prev_quarters"
  )
  numeric_candidates <- intersect_existing(numeric_candidates, df)

  if (length(numeric_candidates) > 0) {
    df <- df %>%
      mutate(across(all_of(numeric_candidates), ~ suppressWarnings(as.numeric(.x))))
  }

  binary_candidates <- c(
    "treated",
    "post",
    "government",
    "non_profit",
    "chain",
    "urban",
    "provider_resides_in_hospital",
    "ccrc_facility",
    "sff_facility"
  )
  binary_candidates <- intersect_existing(binary_candidates, df)

  if (length(binary_candidates) > 0) {
    df <- df %>%
      mutate(across(all_of(binary_candidates), ~ suppressWarnings(as.integer(.x))))
  }

  if (!("avg_los_total" %in% names(df))) {
    message(
      "[quality] avg_los_total is not present in quality_panel.csv -- Spec C/D ",
      "controls for quality outcomes omit average length of stay until a ",
      "quarterly avg_los_total is merged in from the monthly panel."
    )
  }

  apply_facility_lookups(df, panel_label = "quality")
}

# -----------------------------------------------------------------------------
# Controls helpers
# -----------------------------------------------------------------------------
get_case_mix_controls <- function(df) {
  preferred <- intersect_existing(preferred_case_mix_controls, df)
  if (length(preferred) > 0) {
    return(preferred)
  }
  
  fallback <- intersect_existing(fallback_case_mix_controls, df)
  fallback
}

get_controls <- function(df) {
  c(intersect_existing(base_controls, df), get_case_mix_controls(df))
}

# -----------------------------------------------------------------------------
# Nested A/B/C/D control specifications (per C. Moul, following discussion
# of endogenous regressors). Each spec builds on the previous one. Applied
# by outcome category as follows:
#
#   Spec A = post + FE + beds + chain_at_start
#     -> Case mix, Non-profit status, Strategic/Business-model, Staffing, Quality
#   Spec B = A + case mix (state quartile dummies) + non_profit
#     -> Strategic/Business-model, Staffing, Quality (NOT Case mix/Non-profit
#        themselves -- circular, would be controlling for themselves)
#   Spec C = B + occupancy_rate + pct_medicare + pct_medicaid + avg_los_total
#     -> Staffing, Quality only (NOT Strategic/Business-model -- these ARE
#        the strategic outcomes, so C doesn't apply to them)
#   Spec D = C + staffing (rn_hprd, lpn_hprd, cna_hprd, total_hprd)
#     -> Quality only
#
# Two judgment calls made explicit here (flag/confirm if wrong):
#   - Case mix in spec B is read as the case-mix QUARTILE DUMMIES already
#     used as the project's standard case-mix control elsewhere (state
#     quartiles preferred, national as fallback) -- NOT the raw continuous
#     case_mix_total variable, which has so far only been tested as an
#     OUTCOME (endogeneity check), never as a control.
#   - Staffing in spec D is read as all four individual HPRD measures
#     (RN/LPN/CNA/Total), not just Total, matching how Strategic/Business
#     Model already refers to the whole group of variables in spec C.
#
# `government` does not appear in any spec: after the government-ever
# exclusion in load_staffing_panel(), it is constant (=0) in the remaining
# sample and therefore uninformative as a regressor. Time-varying `chain`
# is never used (per Bowblis) -- only `chain_at_start`.
# -----------------------------------------------------------------------------
controls_A <- function(df) {
  intersect_existing(c("beds", "chain_at_start"), df)
}

controls_B <- function(df) {
  c(controls_A(df), intersect_existing("non_profit", df), get_case_mix_controls(df))
}

controls_C <- function(df) {
  c(controls_B(df), intersect_existing(c("occupancy_rate", "pct_medicare", "pct_medicaid", "avg_los_total"), df))
}

controls_D <- function(df) {
  c(controls_C(df), intersect_existing(staffing_outcomes, df))
}

# Convenience: build the "post + controls" RHS string for a given spec
# letter (A, B, C, or D), excluding any variables in `exclude` (e.g., the
# outcome itself, or its close cousins -- same self-exclusion pattern used
# throughout this project for strategic/circular variables).
make_spec_rhs <- function(df, spec = c("A", "B", "C", "D"), exclude = character(0)) {
  spec <- match.arg(spec)
  ctrls <- switch(spec,
    A = controls_A(df),
    B = controls_B(df),
    C = controls_C(df),
    D = controls_D(df)
  )
  ctrls <- setdiff(ctrls, exclude)
  paste(c("post", ctrls), collapse = " + ")
}

make_controls_rhs <- function(df) {
  ctrls <- get_controls(df)
  if (length(ctrls) == 0) {
    return("1")
  }
  paste(ctrls, collapse = " + ")
}

# -----------------------------------------------------------------------------
# Sample restriction helpers
# -----------------------------------------------------------------------------
sample_full <- function(df) {
  df
}

sample_prepandemic <- function(df) {
  df %>%
    filter(ym_date >= as.Date("2017-01-01"),
           ym_date <= as.Date("2019-12-31"))
}

sample_pandemic <- function(df) {
  df %>%
    filter(ym_date >= as.Date("2020-04-01"),
           ym_date <= as.Date("2024-06-30"))
}

drop_anticipation_window <- function(df) {
  df %>%
    filter(is.na(event_time) | !(event_time %in% c(-3, -2, -1)))
}

drop_event_month <- function(df) {
  df %>%
    filter(is.na(event_time) | event_time != 0)
}

# Same operation as drop_event_month(), named for the quarterly panel where
# event_time is measured in quarters. The transition quarter may contain
# care, assessment, and documentation from both before and after the
# transfer, so it is excluded and tau = -1 is the reference period.
drop_transition_quarter <- function(df) {
  df %>%
    filter(is.na(event_time) | event_time != 0)
}

# Apply the per-measure reporting-window trims in quality_measure_year_windows
# to a single outcome. Returns the data unchanged for measures with no trim.
trim_quality_measure_window <- function(df, measure) {
  w <- quality_measure_year_windows %>%
    dplyr::filter(.data$var == .env$measure)
  if (nrow(w) == 0) return(df)
  if (!is.na(w$year_min[1])) df <- df %>% dplyr::filter(year >= w$year_min[1])
  if (!is.na(w$year_max[1])) df <- df %>% dplyr::filter(year <= w$year_max[1])
  df
}

# -----------------------------------------------------------------------------
# Event-study helpers
# -----------------------------------------------------------------------------
prepare_event_study_data <- function(df, min_et = -24L, max_et = 24L) {
  assert_has_cols(df, c("treated", "event_time"), "event_study_data")
  
  df %>%
    dplyr::group_by(cms_certification_number) %>%
    dplyr::mutate(
      ever_treated = as.integer(any(treated == 1, na.rm = TRUE) | any(!is.na(event_time)))
    ) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      event_time_capped = dplyr::case_when(
        ever_treated == 1L & !is.na(event_time) ~ pmin(pmax(as.integer(event_time), min_et), max_et),
        TRUE ~ 9999L
      )
    )
}

# -----------------------------------------------------------------------------
# Formula builders
# -----------------------------------------------------------------------------
make_twfe_formula <- function(lhs, rhs) {
  if (is.null(rhs) || rhs == "" || rhs == "1") {
    as.formula(sprintf("%s ~ 1 | %s + %s", lhs, fe_unit, fe_time))
  } else {
    as.formula(sprintf("%s ~ %s | %s + %s", lhs, rhs, fe_unit, fe_time))
  }
}

make_post_rhs <- function(df) {
  ctrls <- make_controls_rhs(df)
  if (ctrls == "1") {
    "post"
  } else {
    paste("post +", ctrls)
  }
}

make_event_study_formula <- function(lhs, df, ref = -1L, min_et = -24L, max_et = 24L) {
  ctrls <- make_controls_rhs(df)
  event_part <- sprintf(
    "i(event_time_capped, ever_treated, ref = %s, keep = %s:%s)",
    ref, min_et, max_et
  )
  
  rhs <- if (ctrls == "1") event_part else paste(event_part, ctrls, sep = " + ")
  as.formula(sprintf("%s ~ %s | %s + %s", lhs, rhs, fe_unit, fe_time))
}

# -----------------------------------------------------------------------------
# Model wrappers
# -----------------------------------------------------------------------------
run_feols <- function(formula, data) {
  fixest::feols(
    formula = formula,
    data = data,
    vcov = stats::as.formula(paste0("~", cluster_var))
  )
}

# -----------------------------------------------------------------------------
# Labels / convenience objects
# -----------------------------------------------------------------------------
pretty_outcome_labels <- c(
  rn_hprd    = "RN HPRD",
  lpn_hprd   = "LPN HPRD",
  cna_hprd   = "CNA HPRD",
  total_hprd = "Total HPRD",
  ln_rn      = "log(RN HPRD)",
  ln_lpn     = "log(LPN HPRD)",
  ln_cna     = "log(CNA HPRD)",
  ln_total   = "log(Total HPRD)",
  rn_hours_month  = "RN hours (monthly)",
  lpn_hours_month = "LPN hours (monthly)",
  cna_hours_month = "CNA hours (monthly)",
  total_hours     = "Total hours (monthly)",
  ln_rn_hours     = "log(RN hours)",
  ln_lpn_hours    = "log(LPN hours)",
  ln_cna_hours    = "log(CNA hours)",
  ln_total_hours  = "log(Total hours)"
)

# Quality measure labels fold into the same lookup, so get_pretty_label()
# works for staffing and quality outcomes alike.
pretty_outcome_labels <- c(
  pretty_outcome_labels,
  long_stay_quality_measures,
  short_stay_quality_measures,
  vaccination_quality_measures
)

get_pretty_label <- function(x) {
  if (x %in% names(pretty_outcome_labels)) {
    return(pretty_outcome_labels[[x]])
  }
  x
}

# -----------------------------------------------------------------------------
# Quick startup message
# -----------------------------------------------------------------------------
message("[setup] loaded shared regression setup")
message(sprintf("[setup] panel_fp = %s", panel_fp))