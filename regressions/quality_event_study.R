# =============================================================================
# regressions/quality_event_study.R
#
# Estimates quarterly event-study models for CMS quality measures, tracing the
# path of each measure in the quarters before and after an ownership change.
#
# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
# Event time is measured in quarters relative to the ownership change. The
# transition quarter (tau = 0) is excluded and tau = -1 is the reference
# period. All models include facility and year-quarter fixed effects, with
# standard errors two-way clustered by facility and year-quarter.
#
# -----------------------------------------------------------------------------
# Quality measures
# -----------------------------------------------------------------------------
# Labor-saving mechanism measures:
#   qm_406  Catheter use
#   qm_419  Anti-psychotic medication use
#   qm_452  Anti-anxiety or hypnotic medication use
#
# Resident outcome measures:
#   qm_453  Pressure injuries
#   qm_410  Falls with major injury
#   qm_404  Weight loss
#   qm_401  Decline in physical functioning
#   qm_407  Urinary tract infections
#
# For every measure, lower values indicate better measured quality.
#
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
#   data/clean/quality_panel.csv
#
# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
#   outputs/plots/  one event-study plot per quality measure, a three-panel
#                   figure of the mechanism measures, a five-panel figure of
#                   the resident outcome measures, and an index CSV
#
# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
#   regressions/_setup.R
#   R packages: dplyr, readr, fixest, stringr, tibble
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

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

plots_dir <- out_plots_dir

dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 1) Helpers
# -----------------------------------------------------------------------------
# assert_has_cols(), intersect_existing(), make_spec_rhs(), and the quality
# measure maps come from _setup.R. Only the quarterly-specific helpers are
# defined here.

quarter_num <- function(x) {
  x <- toupper(trimws(as.character(x)))
  suppressWarnings(as.integer(str_extract(x, "[1-4]")))
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

# Controls follow Spec A's covariates. Treatment is identified by the
# event-time interaction rather than by a post dummy, so post is excluded;
# it would be collinear with the event-time indicators. chain_at_start is
# time-invariant and absorbed by the facility fixed effects.
#
# Occupancy, payer shares, and case mix are not included: they are outcomes
# of ownership change in their own right, so conditioning on them would
# absorb part of the response being estimated. Staffing is likewise a
# post-treatment mechanism and is not controlled for in these figures.

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
  par(family = "sans")
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
# Loaded through load_quality_panel() rather than read directly, so that the
# sample matches every other estimate in the paper. Reading the CSV directly
# bypassed the shared facility lookups.

df0 <- load_quality_panel()

required_cols <- c(
  "cms_certification_number",
  "year",
  "quarter",
  "year_quarter",
  "treated",
  "event_time"
)

assert_has_cols(df0, required_cols, "quality_panel")

df0 <- df0 %>%
  mutate(
    year_quarter = as.factor(year_quarter),
    event_time = suppressWarnings(as.integer(event_time))
  )

controls_rhs <- make_spec_controls_rhs(df0, spec = "A", exclude = "chain_at_start")

cat("\nControls used:\n", controls_rhs, "\n", sep = "")

# -----------------------------------------------------------------------------
# 3) Outcome definitions
# -----------------------------------------------------------------------------
# Measure codes, labels, and groupings are taken from _setup.R so that these
# figures cannot drift from the quality tables. Reporting-window trims are
# applied per measure by trim_quality_measure_window().
# Lower values indicate better measured quality.

quality_outcomes <- tibble::tibble(
  outcome = c(quality_mechanism_measures, quality_outcome_measures),
  group = c(
    rep("routine_process", length(quality_mechanism_measures)),
    rep("resident_outcome", length(quality_outcome_measures))
  )
) %>%
  mutate(label = unname(unlist(long_stay_quality_measures[outcome])))

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
  
  dat <- trim_quality_measure_window(df0, outcome)
  
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
for (i in seq_len(nrow(quality_outcomes))) {
  cat(sprintf(
    "  - twfe_es_quality_%s_drop_tau0.pdf  [%s]\n",
    quality_outcomes$outcome[[i]], quality_outcomes$label[[i]]
  ))
}

cat("\nGrouped figures:\n")
cat("  - twfe_es_quality_routine_process_drop_tau0.pdf\n")
cat("  - twfe_es_quality_resident_outcomes_drop_tau0.pdf\n")

cat("\nDone.\n")