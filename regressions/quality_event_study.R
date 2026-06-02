# C:/Repositories/white-bowblis-nhmc/regressions/quality_event_study.R
# =============================================================================
# Quality Event Study
#
# Purpose:
#   Estimate quarterly TWFE event-study models for CMS quality outcomes.
#
# Main specification:
#   - Outcome: quarterly quality measure
#   - Event time: quarters relative to ownership change
#   - Drops tau = 0 from the estimation sample
#   - Reference period: tau = -1
#   - Fixed effects: facility and year-quarter
#   - SEs clustered two ways by facility and year-quarter
#
# Quality metric mapping follows quarterly_summary_stats.R:
#   Routine-sensitive process measures:
#     qm_406 = Catheter
#     qm_419 = Antipsychotic
#     qm_452 = Hypnotics / anti-anxiety or hypnotic medication use
#
#   Resident outcome measures:
#     qm_453 = Pressure injuries
#     qm_410 = Falls with major injury
#     qm_404 = Weight Loss
#     qm_401 = ADL Increase
#     qm_407 = Urinary Tract Infections
#
# Outputs:
#   - Individual event-study plots for each quality outcome
#   - 3-panel routine-sensitive process-measure figure
#   - 5-panel resident-outcome figure
#   - Plot index CSV
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(fixest)
  library(stringr)
  library(tibble)
})

options(scipen = 999, digits = 4)

# -----------------------------------------------------------------------------
# 0) Paths
# -----------------------------------------------------------------------------

project_root <- "C:/Repositories/white-bowblis-nhmc"

panel_fp  <- file.path(project_root, "data", "clean", "quality_panel.csv")
plots_dir <- file.path(project_root, "outputs", "plots")

dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 1) Helpers
# -----------------------------------------------------------------------------

assert_has_cols <- function(df, cols, df_name = "data") {
  miss <- setdiff(cols, names(df))
  if (length(miss) > 0) {
    stop(
      sprintf("[%s] missing required columns: %s",
              df_name, paste(miss, collapse = ", ")),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

intersect_existing <- function(x, df) {
  intersect(x, names(df))
}

quarter_num <- function(x) {
  x <- toupper(trimws(as.character(x)))
  suppressWarnings(as.integer(str_extract(x, "[1-4]")))
}

year_quarter_index <- function(year, quarter) {
  yr <- suppressWarnings(as.integer(year))
  qn <- quarter_num(quarter)
  yr * 4L + qn
}

subset_window <- function(df, start_year, start_quarter, end_year, end_quarter) {
  start_idx <- start_year * 4L + start_quarter
  end_idx   <- end_year * 4L + end_quarter
  idx <- year_quarter_index(df$year, df$quarter)
  
  df[idx >= start_idx & idx <= end_idx, , drop = FALSE]
}

drop_tau_zero <- function(df) {
  df %>%
    filter(is.na(event_time) | event_time != 0)
}

prepare_event_study_data_quarterly <- function(df, min_et, max_et) {
  assert_has_cols(
    df,
    c("cms_certification_number", "treated", "event_time"),
    "event_study_data"
  )
  
  df %>%
    group_by(cms_certification_number) %>%
    mutate(
      ever_treated = as.integer(
        any(treated == 1, na.rm = TRUE) | any(!is.na(event_time))
      )
    ) %>%
    ungroup() %>%
    mutate(
      event_time_capped = case_when(
        ever_treated == 1L & !is.na(event_time) ~
          pmin(pmax(as.integer(event_time), min_et), max_et),
        TRUE ~ 9999L
      )
    )
}

get_case_mix_controls <- function(df) {
  preferred <- intersect_existing(
    c("cm_q_state_2", "cm_q_state_3", "cm_q_state_4"),
    df
  )
  
  if (length(preferred) > 0) {
    return(preferred)
  }
  
  fallback <- intersect_existing(
    c("cm_q_nat_2", "cm_q_nat_3", "cm_q_nat_4"),
    df
  )
  
  fallback
}

get_controls <- function(df, include_staffing_controls = FALSE) {
  base_controls <- c(
    "government",
    "non_profit",
    "chain",
    "beds",
    "occupancy_rate",
    "pct_medicare",
    "pct_medicaid"
  )
  
  controls <- c(
    intersect_existing(base_controls, df),
    get_case_mix_controls(df)
  )
  
  # Main quality event-study plots should usually keep this FALSE because
  # staffing is a post-treatment mechanism.
  if (isTRUE(include_staffing_controls)) {
    staffing_controls <- c("rn_hprd", "lpn_hprd", "cna_hprd")
    controls <- c(controls, intersect_existing(staffing_controls, df))
  }
  
  unique(controls)
}

make_controls_rhs <- function(df, include_staffing_controls = FALSE) {
  ctrls <- get_controls(
    df,
    include_staffing_controls = include_staffing_controls
  )
  
  if (length(ctrls) == 0) {
    return("1")
  }
  
  paste(ctrls, collapse = " + ")
}

pick_ref <- function(dat, desired = -1L) {
  ev <- sort(unique(dat$event_time_capped[dat$ever_treated == 1L]))
  ev <- ev[is.finite(ev) & ev != 9999L]
  
  if (!length(ev)) {
    stop("No treated event times found.", call. = FALSE)
  }
  
  if (!is.null(desired) && desired %in% ev) {
    return(as.integer(desired))
  }
  
  if (-1L %in% ev) {
    return(-1L)
  }
  
  negs <- ev[ev < 0L]
  if (length(negs)) {
    return(max(negs))
  }
  
  ev[1]
}

run_es_twfe <- function(lhs, data, controls_rhs, ref_val, window = c(-8L, 8L)) {
  fml <- as.formula(paste0(
    lhs,
    " ~ i(event_time_capped, ever_treated, ref = ", ref_val,
    ", keep = ", window[1], ":", window[2], ") + ",
    controls_rhs,
    " | cms_certification_number + year_quarter"
  ))
  
  feols(
    fml = fml,
    data = data,
    vcov = ~ cms_certification_number + year_quarter,
    lean = TRUE
  )
}

set_plot_font <- function() {
  par(family = "Times New Roman")
}

save_es_plot <- function(model,
                         ref_val,
                         file_stub,
                         ylab_txt,
                         xlab_txt = "Quarters relative to ownership change",
                         xlim_window = c(-8L, 8L),
                         out_dir = plots_dir) {
  if (is.null(model)) return(invisible(NULL))
  
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  out_fp <- file.path(out_dir, paste0(file_stub, ".pdf"))
  
  grDevices::cairo_pdf(
    filename = out_fp,
    width = 9.5,
    height = 6.2
  )
  on.exit(dev.off(), add = TRUE)
  
  set_plot_font()
  
  iplot(
    model,
    ref  = ref_val,
    xlim = xlim_window,
    xlab = xlab_txt,
    ylab = ylab_txt,
    main = "",
    sub  = ""
  )
  
  invisible(out_fp)
}

save_panel_plot <- function(models,
                            refs,
                            labels,
                            file_stub,
                            layout = c(2, 2),
                            width = 11,
                            height = 8.5,
                            xlim_window = c(-8L, 8L),
                            out_dir = plots_dir) {
  stopifnot(length(models) == length(refs), length(models) == length(labels))
  
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  out_fp <- file.path(out_dir, paste0(file_stub, ".pdf"))
  
  grDevices::cairo_pdf(
    filename = out_fp,
    width = width,
    height = height
  )
  on.exit(dev.off(), add = TRUE)
  
  set_plot_font()
  
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  
  par(
    mfrow = layout,
    mar = c(4.2, 4.4, 2.2, 1.2),
    oma = c(0, 0, 0, 0)
  )
  
  for (i in seq_along(models)) {
    iplot(
      models[[i]],
      ref  = refs[[i]],
      xlim = xlim_window,
      xlab = "Quarters relative to ownership change",
      ylab = labels[[i]],
      main = labels[[i]],
      sub  = ""
    )
  }
  
  invisible(out_fp)
}

# -----------------------------------------------------------------------------
# 2) Load quality panel
# -----------------------------------------------------------------------------

df0 <- readr::read_csv(panel_fp, show_col_types = FALSE)

required_cols <- c(
  "cms_certification_number",
  "year",
  "quarter",
  "treated",
  "event_time"
)

assert_has_cols(df0, required_cols, "quality_panel")

df0 <- df0 %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year = suppressWarnings(as.integer(year)),
    quarter = toupper(trimws(as.character(quarter))),
    year_quarter = as.factor(paste0(year, "_", quarter)),
    event_time = suppressWarnings(as.integer(event_time))
  )

numeric_candidates <- c(
  "beds",
  "occupancy_rate",
  "pct_medicare",
  "pct_medicaid",
  "government",
  "non_profit",
  "chain",
  "cm_q_state_2",
  "cm_q_state_3",
  "cm_q_state_4",
  "cm_q_nat_2",
  "cm_q_nat_3",
  "cm_q_nat_4",
  "rn_hprd",
  "lpn_hprd",
  "cna_hprd"
)

numeric_candidates <- intersect_existing(numeric_candidates, df0)

if (length(numeric_candidates) > 0) {
  df0 <- df0 %>%
    mutate(across(all_of(numeric_candidates), ~ suppressWarnings(as.numeric(.x))))
}

# Main figure specification: do not control for staffing.
include_staffing_controls <- FALSE

controls_rhs <- make_controls_rhs(
  df0,
  include_staffing_controls = include_staffing_controls
)

cat("\nControls used:\n", controls_rhs, "\n", sep = "")

# -----------------------------------------------------------------------------
# 3) Outcome definitions
# -----------------------------------------------------------------------------
# These mappings align with quarterly_summary_stats.R.
# Lower values indicate better measured quality.

quality_outcomes <- tibble::tribble(
  ~outcome, ~label, ~group, ~start_year, ~start_quarter, ~end_year, ~end_quarter,
  
  # Routine-sensitive process measures
  "qm_406", "Catheter Use",                              "routine_process", 2017L, 1L, 2024L, 2L,
  "qm_419", "Antipsychotic Medication Use",              "routine_process", 2017L, 1L, 2024L, 2L,
  "qm_452", "Anti-Anxiety or Hypnotic Medication Use",   "routine_process", 2017L, 1L, 2024L, 2L,
  
  # Resident outcome measures
  "qm_453", "Pressure Injuries",                         "resident_outcome", 2018L, 1L, 2023L, 3L,
  "qm_410", "Falls with Major Injury",                   "resident_outcome", 2017L, 1L, 2024L, 2L,
  "qm_404", "Weight Loss",                               "resident_outcome", 2017L, 1L, 2024L, 2L,
  "qm_401", "Decline in Physical Functioning",           "resident_outcome", 2017L, 1L, 2024L, 2L,
  "qm_407", "Urinary Tract Infections",                  "resident_outcome", 2017L, 1L, 2024L, 2L
)

missing_outcomes <- setdiff(quality_outcomes$outcome, names(df0))

if (length(missing_outcomes) > 0) {
  stop(
    sprintf(
      "Missing requested quality outcomes in quality_panel.csv: %s",
      paste(missing_outcomes, collapse = ", ")
    ),
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# 4) Main event-study specification
# -----------------------------------------------------------------------------

event_window <- c(-8L, 8L)

# Preferred quality specification:
#   - Drop tau = 0
#   - Use tau = -1 as reference period
drop_tau0 <- TRUE
desired_ref <- -1L

results <- list()

for (i in seq_len(nrow(quality_outcomes))) {
  
  outcome <- quality_outcomes$outcome[[i]]
  label   <- quality_outcomes$label[[i]]
  
  cat("\n", strrep("=", 80), "\n", sep = "")
  cat("OUTCOME: ", outcome, " — ", label, "\n", sep = "")
  cat(strrep("=", 80), "\n", sep = "")
  
  dat <- subset_window(
    df0,
    start_year    = quality_outcomes$start_year[[i]],
    start_quarter = quality_outcomes$start_quarter[[i]],
    end_year      = quality_outcomes$end_year[[i]],
    end_quarter   = quality_outcomes$end_quarter[[i]]
  )
  
  if (isTRUE(drop_tau0)) {
    dat <- drop_tau_zero(dat)
  }
  
  dat <- prepare_event_study_data_quarterly(
    dat,
    min_et = event_window[1],
    max_et = event_window[2]
  ) %>%
    filter(!is.na(.data[[outcome]]))
  
  ref_val <- pick_ref(dat, desired = desired_ref)
  
  mod <- run_es_twfe(
    lhs = outcome,
    data = dat,
    controls_rhs = controls_rhs,
    ref_val = ref_val,
    window = event_window
  )
  
  cat("Reference period: tau = ", ref_val, "\n", sep = "")
  cat("N = ", format(nrow(dat), big.mark = ","), "\n", sep = "")
  
  print(summary(mod, keep = "^event_time_capped::"))
  
  file_stub <- paste0("twfe_es_quality_", outcome, "_drop_tau0")
  
  save_es_plot(
    model = mod,
    ref_val = ref_val,
    file_stub = file_stub,
    ylab_txt = label,
    xlim_window = event_window,
    out_dir = plots_dir
  )
  
  results[[outcome]] <- list(
    outcome = outcome,
    label = label,
    group = quality_outcomes$group[[i]],
    model = mod,
    ref = ref_val,
    n = nrow(dat),
    plot_file = file.path(plots_dir, paste0(file_stub, ".pdf"))
  )
}

# -----------------------------------------------------------------------------
# 5) Save grouped figures
# -----------------------------------------------------------------------------

routine_process_outcomes <- quality_outcomes %>%
  filter(group == "routine_process") %>%
  pull(outcome)

resident_outcomes <- quality_outcomes %>%
  filter(group == "resident_outcome") %>%
  pull(outcome)

# 3-panel routine-sensitive process figure.
# This uses one row with three plots.
save_panel_plot(
  models = lapply(routine_process_outcomes, function(y) results[[y]]$model),
  refs   = lapply(routine_process_outcomes, function(y) results[[y]]$ref),
  labels = lapply(routine_process_outcomes, function(y) results[[y]]$label),
  file_stub = "twfe_es_quality_routine_process_drop_tau0",
  layout = c(1, 3),
  width = 14,
  height = 4.8,
  xlim_window = event_window,
  out_dir = plots_dir
)

# 5-panel resident-outcome figure.
# This uses a 3-by-2 layout; the sixth panel will remain blank.
save_panel_plot(
  models = lapply(resident_outcomes, function(y) results[[y]]$model),
  refs   = lapply(resident_outcomes, function(y) results[[y]]$ref),
  labels = lapply(resident_outcomes, function(y) results[[y]]$label),
  file_stub = "twfe_es_quality_resident_outcomes_drop_tau0",
  layout = c(3, 2),
  width = 11,
  height = 12,
  xlim_window = event_window,
  out_dir = plots_dir
)

# -----------------------------------------------------------------------------
# 6) Save model index
# -----------------------------------------------------------------------------

model_index <- tibble::tibble(
  outcome = quality_outcomes$outcome,
  label = quality_outcomes$label,
  group = quality_outcomes$group,
  start_year = quality_outcomes$start_year,
  start_quarter = quality_outcomes$start_quarter,
  end_year = quality_outcomes$end_year,
  end_quarter = quality_outcomes$end_quarter,
  reference_tau = vapply(
    quality_outcomes$outcome,
    function(y) results[[y]]$ref,
    integer(1)
  ),
  n = vapply(
    quality_outcomes$outcome,
    function(y) results[[y]]$n,
    integer(1)
  ),
  plot_file = vapply(
    quality_outcomes$outcome,
    function(y) results[[y]]$plot_file,
    character(1)
  )
)

readr::write_csv(
  model_index,
  file.path(plots_dir, "quality_event_study_plot_index_drop_tau0.csv")
)

cat("\nSaved individual quality event-study plots to:\n", plots_dir, "\n", sep = "")

cat("\nIndividual plots:\n")
cat("  - twfe_es_quality_qm_406_drop_tau0.pdf  [Catheter Use]\n")
cat("  - twfe_es_quality_qm_419_drop_tau0.pdf  [Antipsychotic Medication Use]\n")
cat("  - twfe_es_quality_qm_452_drop_tau0.pdf  [Anti-Anxiety or Hypnotic Medication Use]\n")
cat("  - twfe_es_quality_qm_453_drop_tau0.pdf  [Pressure Injuries]\n")
cat("  - twfe_es_quality_qm_410_drop_tau0.pdf  [Falls with Major Injury]\n")
cat("  - twfe_es_quality_qm_404_drop_tau0.pdf  [Weight Loss]\n")
cat("  - twfe_es_quality_qm_401_drop_tau0.pdf  [Decline in Physical Functioning]\n")
cat("  - twfe_es_quality_qm_407_drop_tau0.pdf  [Urinary Tract Infections]\n")

cat("\nGrouped figures:\n")
cat("  - twfe_es_quality_routine_process_drop_tau0.pdf\n")
cat("  - twfe_es_quality_resident_outcomes_drop_tau0.pdf\n")

cat("\nDone.\n")