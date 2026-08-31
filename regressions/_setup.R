# =============================================================================
# regressions/_setup.R
#
# Shared setup for the nursing-home ownership-change analysis: panel
# loaders, sample-construction and control-specification helpers, formula
# builders, and shared labels used by every script in regressions/.
#
# This file is sourced by every analysis script in regressions/ and is not
# intended to be run directly.
#
# -----------------------------------------------------------------------------
# Description
# -----------------------------------------------------------------------------
# Defines:
#   (1) load_staffing_panel() / load_quality_panel() -- the two canonical
#       panel loaders (facility-month and facility-quarter, respectively).
#       Both apply a single shared definition of the estimation sample.
#   (2) controls_A() through controls_D() -- four nested control
#       specifications; see "Control specifications" below.
#   (3) Sample-restriction helpers (pre-pandemic / pandemic subsamples,
#       anticipation-window and ownership-transition-period exclusions).
#   (4) Formula builders and a feols() wrapper for two-way fixed-effects
#       estimation with clustered standard errors.
#   (5) Shared quality-measure code-to-label maps and reporting-window
#       trims, so individual scripts do not carry private copies that can
#       drift apart from one another.
#
# -----------------------------------------------------------------------------
# Sample construction
# -----------------------------------------------------------------------------
# Both panels exclude any facility that was government-owned at any point
# during the sample period. This exclusion, along with each facility's
# baseline chain-affiliation status (`chain_at_start`), is computed once,
# from the monthly panel, in build_facility_lookups(), and applied
# identically to both panels by apply_facility_lookups(). See that
# function's docstring for why these are computed once rather than
# separately per panel.
#
# -----------------------------------------------------------------------------
# Control specifications
# -----------------------------------------------------------------------------
# Four nested specifications are defined:
#
#   Spec A = post + beds + chain_at_start
#     Applies to: case mix, non-profit status, business-model outcomes,
#     staffing, and quality. Preferred specification for the project's
#     main tables; see make_spec_rhs().
#   Spec B = A + case-mix quartile dummies + non-profit status
#     Applies to: business-model outcomes, staffing, quality. Not applied
#     to case mix or non-profit status themselves, since a variable cannot
#     serve as its own control.
#   Spec C = B + occupancy rate + Medicare share + Medicaid share + average
#     length of stay
#     Applies to: staffing and quality only. Not applied to business-model
#     outcomes, since these variables constitute the business-model
#     outcome set itself.
#   Spec D = C + RN, LPN, CNA, and Total staffing HPRD
#     Applies to: quality only.
#
# See make_spec_rhs() for the function that builds a specification's
# right-hand side, and controls_A() through controls_D() for the
# specifications themselves, including two design choices documented at
# that section.
#
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
#   data/clean/staffing_panel.csv   Facility-month panel. Read by
#                                    load_staffing_panel() and
#                                    build_facility_lookups().
#   data/clean/quality_panel.csv    Facility-quarter panel. Read by
#                                    load_quality_panel().
#
# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
#   None. This file defines shared objects and functions only; it writes no
#   files and produces no tables or figures on its own.
#
# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
#   R packages: dplyr, readr, fixest, stringr, tibble
#
# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
#   - Assumes Medicare Cost Report (MCR) ownership-change timing is already
#     the baseline event-time definition in both panels.
#   - Assumes staffing variables use *_hprd naming.
#   - This file is limited to shared setup and helper functions; estimation
#     loops and table construction belong in the calling script.
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(fixest)
  library(stringr)
  library(tibble)
})

# =============================================================================
# Paths
# =============================================================================
project_root <- "C:/Repositories/white-bowblis-nhmc"

panel_fp <- file.path(project_root, "data", "clean", "staffing_panel.csv")
quality_panel_fp <- file.path(project_root, "data", "clean", "quality_panel.csv")
out_tables_dir <- file.path(project_root, "outputs", "tables")
out_plots_dir  <- file.path(project_root, "outputs", "plots")

dir.create(out_tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_plots_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# Core variable sets
# =============================================================================
staffing_outcomes <- c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd")

log_outcome_map <- c(
  rn_hprd    = "ln_rn",
  lpn_hprd   = "ln_lpn",
  cna_hprd   = "ln_cna",
  total_hprd = "ln_total"
)

# Raw PBJ hours: the HPRD numerator, not divided by resident-days. Used
# alongside HPRD to test whether occupancy-driven denominator changes
# mechanically drive the HPRD results. Kept as a separate map from
# log_outcome_map so code that loops over log_outcome_map assuming exactly
# the four HPRD variables is unaffected.
raw_hours_outcomes <- c("rn_hours_month", "lpn_hours_month", "cna_hours_month", "total_hours")

log_raw_hours_map <- c(
  rn_hours_month   = "ln_rn_hours",
  lpn_hours_month  = "ln_lpn_hours",
  cna_hours_month  = "ln_cna_hours",
  total_hours      = "ln_total_hours"
)

# -----------------------------------------------------------------------------
# Quality measure sets (quarterly panel)
# -----------------------------------------------------------------------------
# Single definition of the quality-measure code-to-label maps; quality
# scripts should reference these rather than declaring private copies.
#
# Vaccination measures (qm_430, pneumococcal; qm_472, influenza) are
# recorded here but excluded from the main long- and short-stay measure
# sets below -- not part of the paper's quality outcomes.
# -----------------------------------------------------------------------------
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

# Labor-saving mechanism vs. resident-outcome grouping, used by the paper's
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

# Reporting-window trims: qm_424/qm_425 are effectively discontinued
# (excluded entirely above rather than trimmed); qm_471 and qm_472 exist
# only over the windows below. NA indicates no trim.
#
# The qm_453 (pressure injuries) window uses year-level bounds only, not
# quarter-level -- year_max = 2023 includes all of 2023 (through Q4), a few
# months past the narrower window where the measure is actually reported.
# This is a deliberate approximation; tighten by adding quarter bounds to
# trim_quality_measure_window() if the Q4 reporting gap proves material.
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

# Preferred (state-based) case-mix controls, used when present.
preferred_case_mix_controls <- c(
  "cm_q_state_2",
  "cm_q_state_3",
  "cm_q_state_4"
)

# Fallback (national-based) case-mix controls, used when the preferred
# state-based controls are unavailable in a given panel.
fallback_case_mix_controls <- c(
  "cm_q_nat_2",
  "cm_q_nat_3",
  "cm_q_nat_4"
)

# Fixed-effect and clustering dimensions shared across the project's
# monthly-panel specifications.
fe_unit <- "cms_certification_number"
fe_time <- "year_month"
cluster_var <- "cms_certification_number"

# =============================================================================
# Generic helpers
# =============================================================================

# -----------------------------------------------------------------------------
# mk_log()
#
# Computes a "safe" natural log for log-outcome regressions: returns NA
# rather than -Inf or NaN for non-positive or missing input, so affected
# observations are dropped by the regression rather than causing an error.
#
# Arguments:
#   x -- Numeric vector.
#
# Returns:
#   Numeric vector: log(x) where x > 0 and not missing; NA_real_ otherwise.
# -----------------------------------------------------------------------------
mk_log <- function(x) {
  ifelse(is.na(x) | x <= 0, NA_real_, log(x))
}

# -----------------------------------------------------------------------------
# assert_has_cols()
#
# Verifies that a data frame contains a required set of columns, raising an
# informative error naming the missing columns rather than allowing a
# later, less legible failure.
#
# Arguments:
#   df      -- Data frame to check.
#   cols    -- Character vector of column names that must be present.
#   df_name -- Character scalar identifying df in the error message.
#              Defaults to "data".
#
# Returns:
#   TRUE, invisibly, if all required columns are present; otherwise raises
#   an error.
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# intersect_existing()
#
# Filters a candidate vector of column names down to those actually
# present in a data frame, so scripts degrade gracefully when a panel is
# rebuilt without a particular (typically optional) column.
#
# Arguments:
#   x  -- Character vector of candidate column names.
#   df -- Data frame (or any object with a names() method) to check against.
#
# Returns:
#   Character vector: the subset of x present in names(df).
# -----------------------------------------------------------------------------
intersect_existing <- function(x, df) {
  intersect(x, names(df))
}

# -----------------------------------------------------------------------------
# norm_ccn()
#
# Normalizes the CMS Certification Number (CCN) facility key to a common
# representation across both panels. staffing_panel.csv and
# quality_panel.csv are read independently, and readr can infer different
# column types for the CCN in each depending on the values it samples for
# type inference; a numeric parse silently strips leading zeros, which
# would break facility matching across panels ("015009" vs. 15009). Every
# facility-level key in this project passes through this function first.
#
# Arguments:
#   x -- Vector (numeric or character) of raw CCN values.
#
# Returns:
#   Character vector of zero-padded, six-digit CCNs where the input parses
#   as numeric. Falls back to a trimmed character value for non-numeric
#   input, so an alphanumeric identifier is preserved rather than coerced
#   to NA.
# -----------------------------------------------------------------------------
norm_ccn <- function(x) {
  x_chr <- trimws(as.character(x))
  x_num <- suppressWarnings(as.numeric(x_chr))
  ifelse(
    is.na(x_num),
    x_chr,
    formatC(x_num, width = 6, flag = "0", format = "d")
  )
}

# =============================================================================
# Facility-level derived attributes
# =============================================================================
# Two facility-level attributes jointly define the estimation sample and a
# control variable used across both panels:
#
#   (1) ever_government -- the exclusion set: any facility government-owned
#       at any point during the sample period.
#   (2) chain_at_start -- each facility's baseline chain-affiliation status,
#       fixed at the start of the panel rather than time-varying.
#
# Both are derived from the MONTHLY panel and joined onto whichever panel
# is being loaded, rather than recomputed separately from the quarterly
# panel. Recomputing separately can produce different answers for the same
# facility -- for example, a facility observed as government-owned in a
# month that survives into staffing_panel.csv but whose corresponding
# quarters are sparse or absent in quality_panel.csv, or a quarterly chain
# baseline anchored on 2017Q1 rather than the monthly panel's 2017/01. One
# definition, one source, applied identically to both panels.
#
# build_facility_lookups() reads only four columns to construct these
# attributes, so the cost is a small fraction of a full panel load. Results
# are cached in .nhmc_cache so a script that loads both panels pays this
# cost once. Pass refresh = TRUE after rebuilding the underlying panel.
# =============================================================================
.nhmc_cache <- new.env(parent = emptyenv())

# -----------------------------------------------------------------------------
# build_facility_lookups()
#
# Constructs the two facility-level derived attributes described above
# (ever_government, chain_at_start) from the monthly panel, caching the
# result so repeated calls within a session do not re-read the source file.
#
# Arguments:
#   fp      -- File path to the monthly panel CSV. Defaults to panel_fp.
#   refresh -- Logical. If TRUE, ignore any cached result and rebuild from
#              fp. Defaults to FALSE. Pass TRUE after rebuilding the
#              underlying panel.
#
# Returns:
#   A list with elements:
#     ever_government    -- Character vector of normalized CCNs for
#                            facilities ever observed as government-owned.
#     chain_lookup       -- Tibble with columns cms_certification_number
#                            and chain_at_start.
#     n_chain_fallback   -- Integer count of facilities whose
#                            chain_at_start came from the fallback
#                            (earliest observed) rule rather than the
#                            January 2017 value directly.
#     n_facilities_total -- Integer count of distinct facilities in the
#                            source file.
#     source_fp          -- The file path the lookups were built from.
#
# Side effects:
#   Emits a message() summarizing the counts above.
# -----------------------------------------------------------------------------
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

  # chain_at_start: the January 2017 value where available, and otherwise the
  # facility's earliest observed value. The fallback is used because PBJ
  # reporting coverage was still incomplete in January 2017, so a strict
  # January 2017 rule would drop facilities that report reliably from a
  # slightly later date.
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

# -----------------------------------------------------------------------------
# apply_facility_lookups()
#
# Applies the shared facility-level lookups (ever_government,
# chain_at_start) to a panel: normalizes the facility key, drops
# government-ever facilities, joins chain_at_start, and converts the key
# to a factor. Called by both load_staffing_panel() and
# load_quality_panel() so the two panels cannot disagree with one another
# about the estimation sample or baseline chain status.
#
# Arguments:
#   df          -- Data frame to which the lookups should be applied. Must
#                   contain a cms_certification_number column.
#   panel_label -- Character scalar used to label console messages (e.g.,
#                   "staffing" or "quality"). Defaults to "panel".
#
# Returns:
#   The input data frame with:
#     - cms_certification_number normalized and converted to a factor.
#     - Government-ever facilities removed.
#     - chain_at_start joined on.
#
# Side effects:
#   Emits message()s reporting facilities dropped for government
#   ownership, and facilities present in this panel but absent from the
#   monthly panel's chain_at_start lookup.
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# compare_panel_samples()
#
# Diagnostic check of whether the staffing and quality panels agree on the
# facility set after the shared exclusion in apply_facility_lookups() is
# applied. Any disagreement reflects a data-construction difference
# between the two source panels, not a regression-specification issue.
#
# Arguments:
#   None.
#
# Returns:
#   A list, invisibly, with elements:
#     staffing_only -- Character vector of CCNs present in the staffing
#                       panel but not the quality panel.
#     quality_only  -- Character vector of CCNs present in the quality
#                       panel but not the staffing panel.
#
# Side effects:
#   Loads both panels (an expensive operation) and prints a facility-count
#   summary to the console.
# -----------------------------------------------------------------------------
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

# =============================================================================
# Panel loaders
# =============================================================================

# -----------------------------------------------------------------------------
# load_staffing_panel()
#
# Loads and prepares the canonical facility-month staffing panel: reads the
# source CSV, asserts required columns are present, normalizes types,
# constructs log-transformed outcomes, and applies the shared facility-
# level sample restrictions via apply_facility_lookups(). Sanctioned entry
# point for the staffing panel; scripts should not read
# staffing_panel.csv directly.
#
# Arguments:
#   fp -- File path to the staffing panel CSV. Defaults to panel_fp.
#
# Returns:
#   A data frame (tibble) with normalized types, log-transformed HPRD and
#   raw-hours outcomes appended, and the government-ever exclusion and
#   chain_at_start already applied.
#
# Side effects:
#   Emits message()s from assert_has_cols() (on failure) and from
#   apply_facility_lookups(). Raises an error if fp does not exist or is
#   missing a required column.
#
# Notes:
#   guess_max = Inf overrides readr's default column-type inference, which
#   samples only the first 1,000 rows of each column. The panel is sorted by
#   facility and then by month, so a column that is blank for the first
#   thousand rows can be typed as logical rather than numeric, in which case
#   every subsequent value in that column parses as missing. Reading the full
#   file for type inference costs an additional pass but eliminates this
#   failure mode.
# -----------------------------------------------------------------------------
load_staffing_panel <- function(fp = panel_fp) {
  if (!file.exists(fp)) {
    stop(sprintf("Panel file not found: %s", fp), call. = FALSE)
  }

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

  # Core types. The CCN is left as a normalized character key here and
  # converted to a factor only at the end of apply_facility_lookups(),
  # after the facility-level joins have taken place.
  df <- df %>%
    mutate(
      cms_certification_number = norm_ccn(cms_certification_number),
      year_month = as.character(year_month),
      quarter = as.character(quarter),
      ym_date = as.Date(paste0(year_month, "/01"), format = "%Y/%m/%d")
    )

  numeric_candidates <- c(
    staffing_outcomes,
    raw_hours_outcomes,
    "resident_days",
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

  # Log-transformed staffing HPRD outcomes.
  for (nm in names(log_outcome_map)) {
    if (nm %in% names(df)) {
      df[[log_outcome_map[[nm]]]] <- mk_log(df[[nm]])
    }
  }

  # Log-transformed raw hours (numerator only). Guarded so this does not
  # error against an older staffing_panel.csv built before these columns
  # existed.
  for (nm in names(log_raw_hours_map)) {
    if (nm %in% names(df)) {
      df[[log_raw_hours_map[[nm]]]] <- mk_log(df[[nm]])
    }
  }

  # Government-ever exclusion and chain_at_start are applied from the
  # shared facility-level lookups rather than recomputed here, so the
  # staffing and quality panels cannot diverge from one another.
  apply_facility_lookups(df, panel_label = "staffing")
}

# -----------------------------------------------------------------------------
# load_quality_panel()
#
# Loads and prepares the canonical facility-quarter quality panel. Mirrors
# load_staffing_panel(): same government-ever exclusion, same
# chain_at_start, same CCN normalization, all sourced from the same
# facility-level lookups rather than recomputed from quarterly data.
# Sanctioned entry point for the quality panel; scripts should not read
# quality_panel.csv directly.
#
# Arguments:
#   fp -- File path to the quality panel CSV. Defaults to quality_panel_fp.
#
# Returns:
#   A data frame (tibble) with normalized types, a year_quarter key (e.g.,
#   "2017Q1") for use as the calendar fixed effect and clustering
#   dimension, and the government-ever exclusion and chain_at_start
#   already applied.
#
# Side effects:
#   Emits message()s from assert_has_cols() (on failure), from
#   apply_facility_lookups(), and a message if avg_los_total is absent
#   from the source file (see "Notes" below). Raises an error if fp does
#   not exist or is missing a required column.
#
# Notes:
#   guess_max = Inf is applied for the same reason documented in
#   load_staffing_panel() -- the quarterly panel is subject to the same
#   readr type-inference behavior.
#
#   quality_panel.csv does not currently contain an avg_los_total column, so
#   the Spec C and Spec D control sets for quality outcomes omit average
#   length of stay. A message is emitted at load time to make this visible.
# -----------------------------------------------------------------------------
load_quality_panel <- function(fp = quality_panel_fp) {
  if (!file.exists(fp)) {
    stop(sprintf("Quality panel file not found: %s", fp), call. = FALSE)
  }

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

# =============================================================================
# Control-specification helpers
# =============================================================================

# -----------------------------------------------------------------------------
# get_case_mix_controls()
#
# Selects the case-mix control set for a given panel, preferring
# state-based quartile dummies and falling back to national-based quartile
# dummies if the state-based controls are unavailable.
#
# Arguments:
#   df -- Data frame to check for available case-mix columns.
#
# Returns:
#   Character vector of case-mix column names present in df: the state-
#   based set if any of preferred_case_mix_controls are present, otherwise
#   the national-based fallback set.
# -----------------------------------------------------------------------------
get_case_mix_controls <- function(df) {
  preferred <- intersect_existing(preferred_case_mix_controls, df)
  if (length(preferred) > 0) {
    return(preferred)
  }

  fallback <- intersect_existing(fallback_case_mix_controls, df)
  fallback
}

# -----------------------------------------------------------------------------
# get_controls()
#
# Returns the legacy "base controls plus case mix" control set for a given
# panel. Superseded for new work by the nested Spec A-D framework below
# (see make_spec_rhs()); retained for scripts still using the full control
# set.
#
# Arguments:
#   df -- Data frame to check for available control columns.
#
# Returns:
#   Character vector: base_controls intersected with df's columns, plus
#   the result of get_case_mix_controls(df).
# -----------------------------------------------------------------------------
get_controls <- function(df) {
  c(intersect_existing(base_controls, df), get_case_mix_controls(df))
}

# -----------------------------------------------------------------------------
# Nested control specifications (Spec A-D)
# -----------------------------------------------------------------------------
# See the file header's "Control specifications" section for which outcome
# families each specification applies to.
#
# Two design choices are made explicit here:
#
#   (1) "Case mix" in Spec B refers to the project's standard case-mix
#       quartile dummies (get_case_mix_controls(); state quartiles
#       preferred, national quartiles as fallback), not the raw continuous
#       case_mix_total variable. case_mix_total has, to date, only been
#       used as an OUTCOME, never as a control.
#   (2) "Staffing" in Spec D refers to all four individual HPRD measures
#       (RN, LPN, CNA, Total), not Total alone, matching how the
#       business-model outcome family is already referred to as a group in
#       Spec C.
#
# `government` does not appear in any specification: after the
# government-ever exclusion applied in load_staffing_panel(), this
# variable is constant (equal to zero) in the remaining sample and
# therefore uninformative as a regressor. The time-varying `chain`
# variable is never used as a control in any specification; only
# `chain_at_start` is used.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# controls_A()
#
# Builds Spec A's control set: beds and baseline chain affiliation.
#
# Arguments:
#   df -- Data frame to check for available control columns.
#
# Returns:
#   Character vector of control column names present in df.
# -----------------------------------------------------------------------------
controls_A <- function(df) {
  intersect_existing(c("beds", "chain_at_start"), df)
}

# -----------------------------------------------------------------------------
# controls_B()
#
# Builds Spec B's control set: Spec A plus non-profit status and case-mix
# quartile dummies.
#
# Arguments:
#   df -- Data frame to check for available control columns.
#
# Returns:
#   Character vector of control column names present in df.
# -----------------------------------------------------------------------------
controls_B <- function(df) {
  c(controls_A(df), intersect_existing("non_profit", df), get_case_mix_controls(df))
}

# -----------------------------------------------------------------------------
# controls_C()
#
# Builds Spec C's control set: Spec B plus occupancy rate, Medicare share,
# Medicaid share, and average length of stay.
#
# Arguments:
#   df -- Data frame to check for available control columns.
#
# Returns:
#   Character vector of control column names present in df.
# -----------------------------------------------------------------------------
controls_C <- function(df) {
  c(controls_B(df), intersect_existing(c("occupancy_rate", "pct_medicare", "pct_medicaid", "avg_los_total"), df))
}

# -----------------------------------------------------------------------------
# controls_D()
#
# Builds Spec D's control set: Spec C plus RN, LPN, CNA, and Total staffing
# HPRD.
#
# Arguments:
#   df -- Data frame to check for available control columns.
#
# Returns:
#   Character vector of control column names present in df.
# -----------------------------------------------------------------------------
controls_D <- function(df) {
  c(controls_C(df), intersect_existing(staffing_outcomes, df))
}

# -----------------------------------------------------------------------------
# make_spec_rhs()
#
# Builds the right-hand side of a regression formula ("post + controls")
# for a given nested specification (A, B, C, or D), excluding any
# variables named in `exclude` -- typically the outcome itself, or
# variables that would be circular controls for it.
#
# Arguments:
#   df      -- Data frame to check for available control columns.
#   spec    -- Character scalar: one of "A", "B", "C", "D".
#   exclude -- Character vector of variable names to remove from the
#              control set after it is built. Defaults to an empty vector.
#
# Returns:
#   Character scalar: a formula right-hand side of the form
#   "post + control1 + control2 + ...".
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# make_controls_rhs()
#
# Builds a formula right-hand side (controls only, no "post" term) using
# the legacy full control set from get_controls(). Retained for scripts
# still using the pre-Spec-A control set; new work should prefer
# make_spec_rhs().
#
# Arguments:
#   df -- Data frame to check for available control columns.
#
# Returns:
#   Character scalar: either "control1 + control2 + ..." or "1" if no
#   controls are available in df.
# -----------------------------------------------------------------------------
make_controls_rhs <- function(df) {
  ctrls <- get_controls(df)
  if (length(ctrls) == 0) {
    return("1")
  }
  paste(ctrls, collapse = " + ")
}

# =============================================================================
# Sample-restriction helpers
# =============================================================================

# -----------------------------------------------------------------------------
# sample_full()
#
# Identity function representing "no sample restriction," so scripts
# iterating over a list of sample-restriction functions can include the
# unrestricted sample without a special case.
#
# Arguments:
#   df -- Data frame.
#
# Returns:
#   df, unmodified.
# -----------------------------------------------------------------------------
sample_full <- function(df) {
  df
}

# -----------------------------------------------------------------------------
# sample_prepandemic()
#
# Restricts a monthly panel to the pre-pandemic period, January 2017
# through December 2019.
#
# Arguments:
#   df -- Data frame containing a ym_date column (as constructed by
#         load_staffing_panel()).
#
# Returns:
#   df, filtered to ym_date in [2017-01-01, 2019-12-31].
# -----------------------------------------------------------------------------
sample_prepandemic <- function(df) {
  df %>%
    filter(ym_date >= as.Date("2017-01-01"),
           ym_date <= as.Date("2019-12-31"))
}

# -----------------------------------------------------------------------------
# sample_pandemic()
#
# Restricts a monthly panel to the pandemic-era period, April 2020 through
# June 2024.
#
# Arguments:
#   df -- Data frame containing a ym_date column (as constructed by
#         load_staffing_panel()).
#
# Returns:
#   df, filtered to ym_date in [2020-04-01, 2024-06-30].
# -----------------------------------------------------------------------------
sample_pandemic <- function(df) {
  df %>%
    filter(ym_date >= as.Date("2020-04-01"),
           ym_date <= as.Date("2024-06-30"))
}

# -----------------------------------------------------------------------------
# drop_anticipation_window()
#
# Excludes the pre-transition anticipation window (event_time in
# {-3, -2, -1}) from a monthly panel. Standard staffing-sample restriction,
# since staffing may adjust in the months immediately preceding the
# recorded ownership-change date.
#
# Arguments:
#   df -- Data frame containing an event_time column.
#
# Returns:
#   df, with rows where event_time is in {-3, -2, -1} removed. Rows with
#   missing event_time (never-treated facilities) are retained.
# -----------------------------------------------------------------------------
drop_anticipation_window <- function(df) {
  df %>%
    filter(is.na(event_time) | !(event_time %in% c(-3, -2, -1)))
}

# -----------------------------------------------------------------------------
# drop_event_month()
#
# Excludes the ownership-change month itself (event_time == 0) from a
# monthly panel.
#
# Arguments:
#   df -- Data frame containing an event_time column.
#
# Returns:
#   df, with rows where event_time equals 0 removed. Rows with missing
#   event_time are retained.
# -----------------------------------------------------------------------------
drop_event_month <- function(df) {
  df %>%
    filter(is.na(event_time) | event_time != 0)
}

# -----------------------------------------------------------------------------
# drop_transition_quarter()
#
# Excludes the ownership-change quarter itself (event_time == 0) from a
# quarterly panel. Equivalent to drop_event_month(), named separately for
# the quarterly panel, where event_time is measured in quarters. The
# transition quarter can combine care, assessment, and documentation from
# both before and after the ownership transfer, so it is excluded and
# tau = -1 is used as the reference period.
#
# Arguments:
#   df -- Data frame containing an event_time column.
#
# Returns:
#   df, with rows where event_time equals 0 removed. Rows with missing
#   event_time are retained.
# -----------------------------------------------------------------------------
drop_transition_quarter <- function(df) {
  df %>%
    filter(is.na(event_time) | event_time != 0)
}

# -----------------------------------------------------------------------------
# trim_quality_measure_window()
#
# Applies the per-measure reporting-window trim defined in
# quality_measure_year_windows to a single quality outcome, restricting a
# measure with a known reporting gap to the years over which it is
# actually reported.
#
# Arguments:
#   df      -- Data frame containing a year column.
#   measure -- Character scalar: the quality-measure code to look up in
#              quality_measure_year_windows (e.g., "qm_453").
#
# Returns:
#   df, unmodified if measure has no entry in quality_measure_year_windows;
#   otherwise filtered to the [year_min, year_max] window specified for
#   that measure (either bound may be NA, meaning no restriction on that
#   side).
# -----------------------------------------------------------------------------
trim_quality_measure_window <- function(df, measure) {
  w <- quality_measure_year_windows %>%
    dplyr::filter(.data$var == .env$measure)
  if (nrow(w) == 0) return(df)
  if (!is.na(w$year_min[1])) df <- df %>% dplyr::filter(year >= w$year_min[1])
  if (!is.na(w$year_max[1])) df <- df %>% dplyr::filter(year <= w$year_max[1])
  df
}

# =============================================================================
# Event-study helpers
# =============================================================================

# -----------------------------------------------------------------------------
# prepare_event_study_data()
#
# Prepares a panel for event-study estimation by flagging ever-treated
# facilities and constructing a capped event-time variable suitable for
# fixest's i() interaction syntax.
#
# Arguments:
#   df     -- Data frame containing treated and event_time columns.
#   min_et -- Integer scalar: the minimum event time to retain before
#             capping. Defaults to -24L.
#   max_et -- Integer scalar: the maximum event time to retain before
#             capping. Defaults to 24L.
#
# Returns:
#   df, with two columns added:
#     ever_treated       -- Integer (0/1): whether the facility is ever
#                            observed as treated or with a non-missing
#                            event_time.
#     event_time_capped  -- Integer: event_time clipped to [min_et, max_et]
#                            for ever-treated facilities; 9999L (a sentinel
#                            excluded by the estimation window) for
#                            never-treated facilities.
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

# =============================================================================
# Formula builders
# =============================================================================

# -----------------------------------------------------------------------------
# make_twfe_formula()
#
# Builds a two-way fixed-effects feols() formula from an outcome and a
# right-hand-side control string.
#
# Arguments:
#   lhs -- Character scalar: the outcome variable name.
#   rhs -- Character scalar: the control right-hand side (e.g., from
#          make_controls_rhs() or make_spec_rhs()), or NULL / "" / "1" for
#          no controls.
#
# Returns:
#   A formula object of the form "lhs ~ 1 | fe_unit + fe_time" (no
#   controls) or "lhs ~ rhs | fe_unit + fe_time".
# -----------------------------------------------------------------------------
make_twfe_formula <- function(lhs, rhs) {
  if (is.null(rhs) || rhs == "" || rhs == "1") {
    as.formula(sprintf("%s ~ 1 | %s + %s", lhs, fe_unit, fe_time))
  } else {
    as.formula(sprintf("%s ~ %s | %s + %s", lhs, rhs, fe_unit, fe_time))
  }
}

# -----------------------------------------------------------------------------
# make_post_rhs()
#
# Builds a "post + controls" right-hand side using the legacy full control
# set from make_controls_rhs(). Retained for scripts still using the
# pre-Spec-A control set; new work should prefer make_spec_rhs().
#
# Arguments:
#   df -- Data frame to check for available control columns.
#
# Returns:
#   Character scalar: "post" alone if no controls are available, otherwise
#   "post + control1 + control2 + ...".
# -----------------------------------------------------------------------------
make_post_rhs <- function(df) {
  ctrls <- make_controls_rhs(df)
  if (ctrls == "1") {
    "post"
  } else {
    paste("post +", ctrls)
  }
}

# -----------------------------------------------------------------------------
# make_event_study_formula()
#
# Builds an event-study feols() formula using fixest's i() interaction
# syntax, with the legacy full control set from make_controls_rhs().
#
# Arguments:
#   df     -- Data frame prepared by prepare_event_study_data() (must
#             contain event_time_capped and ever_treated).
#   lhs    -- Character scalar: the outcome variable name.
#   ref    -- Integer scalar: the reference (omitted) event-time period.
#             Defaults to -1L.
#   min_et -- Integer scalar: the minimum event time to include. Defaults
#             to -24L.
#   max_et -- Integer scalar: the maximum event time to include. Defaults
#             to 24L.
#
# Returns:
#   A formula object of the form
#   "lhs ~ i(event_time_capped, ever_treated, ref = ref, keep = min_et:max_et)
#          [+ controls] | fe_unit + fe_time".
# -----------------------------------------------------------------------------
make_event_study_formula <- function(lhs, df, ref = -1L, min_et = -24L, max_et = 24L) {
  ctrls <- make_controls_rhs(df)
  event_part <- sprintf(
    "i(event_time_capped, ever_treated, ref = %s, keep = %s:%s)",
    ref, min_et, max_et
  )

  rhs <- if (ctrls == "1") event_part else paste(event_part, ctrls, sep = " + ")
  as.formula(sprintf("%s ~ %s | %s + %s", lhs, rhs, fe_unit, fe_time))
}

# =============================================================================
# Model wrappers
# =============================================================================

# -----------------------------------------------------------------------------
# run_feols()
#
# Thin wrapper around fixest::feols() that applies this project's standard
# clustering variable by default.
#
# Arguments:
#   formula -- A formula object (e.g., from make_twfe_formula() or
#              make_event_study_formula()).
#   data    -- Data frame to estimate on.
#
# Returns:
#   A fixest model object, with standard errors clustered on cluster_var.
# -----------------------------------------------------------------------------
run_feols <- function(formula, data) {
  fixest::feols(
    formula = formula,
    data = data,
    vcov = stats::as.formula(paste0("~", cluster_var))
  )
}

# =============================================================================
# Labels
# =============================================================================
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

# Quality-measure labels are folded into the same lookup, so
# get_pretty_label() serves staffing and quality outcomes alike.
pretty_outcome_labels <- c(
  pretty_outcome_labels,
  long_stay_quality_measures,
  short_stay_quality_measures,
  vaccination_quality_measures
)

# -----------------------------------------------------------------------------
# get_pretty_label()
#
# Looks up a human-readable label for a variable name, for use in table and
# figure construction.
#
# Arguments:
#   x -- Character scalar: a variable name.
#
# Returns:
#   Character scalar: the corresponding entry in pretty_outcome_labels if
#   one exists; otherwise x unchanged.
# -----------------------------------------------------------------------------
get_pretty_label <- function(x) {
  if (x %in% names(pretty_outcome_labels)) {
    return(pretty_outcome_labels[[x]])
  }
  x
}

# =============================================================================
# Startup message
# =============================================================================
message("[setup] loaded shared regression setup")
message(sprintf("[setup] panel_fp = %s", panel_fp))
