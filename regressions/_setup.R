# =============================================================================
# regressions/_setup.R
#
# Shared setup for staffing-panel regressions.
#
# Notes:
# - Canonical panel source: data/clean/staffing_panel.csv
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

# -----------------------------------------------------------------------------
# Panel loader
# -----------------------------------------------------------------------------
load_staffing_panel <- function(fp = panel_fp) {
  if (!file.exists(fp)) {
    stop(sprintf("Panel file not found: %s", fp), call. = FALSE)
  }
  
  df <- readr::read_csv(fp, show_col_types = FALSE)
  
  required_cols <- c(
    "cms_certification_number",
    "year_month",
    "quarter",
    "treated",
    "post",
    "event_time"
  )
  assert_has_cols(df, required_cols, "staffing_panel")
  
  # Core types
  df <- df %>%
    mutate(
      cms_certification_number = as.factor(cms_certification_number),
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
  
  df
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