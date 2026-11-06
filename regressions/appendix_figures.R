# =============================================================================
# regressions/appendix_figures.R
#
# Event-study figures reported in the paper's appendix. The main-text figures
# are produced by twfe_event_study.R, quality_event_study.R, and
# stacked_event_study.R; this script produces everything else, so that every
# appendix figure is generated in one place and can be switched on or off
# from the RUN list below.
#
# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
# Staffing (monthly panel; RN, LPN, CNA, and total HPRD in levels):
#   with_anticipation  Full pre-period retained, reference tau = -1.
#   one_year           Event window tau in [-12, 12], donut {-3,-2,-1},
#                      reference tau = -4, endpoints binned at +/-12.
#   narrow_donut       Drops tau in {-2,-1}, reference tau = -4.
#   wide_donut         Drops tau in {-4,-3,-2,-1}, reference tau = -5.
#   pre_pandemic       January 2017 to December 2019, baseline donut.
#   pandemic           April 2020 to June 2024, baseline donut.
#   logs               Log HPRD for all four outcomes, baseline donut.
#
# Operating model (occupancy, Medicare share, Medicaid share, average length
# of stay):
#   These outcomes come from annual cost reports, and each report's value is
#   assigned to every month of its fiscal year, so a monthly event study
#   would trace a staircase rather than a path. They are therefore estimated
#   in annual event-time bins: year k collects months 12k through 12k + 11
#   relative to the ownership change. The reference bin is year -1, which
#   after the donut holds months -12 through -4. Bin -3 holds every month
#   at -25 or earlier and bin 3 every month at 36 or later.
#
# Quality (quarterly panel):
#   The two short-stay measures, which quality_event_study.R does not cover.
#
# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
# Matches twfe_event_study.R and quality_event_study.R: Spec A controls
# (certified beds), facility and calendar-period fixed effects, standard
# errors two-way clustered by facility and calendar period. Staffing figures
# exclude the anticipation window except where noted above.
#
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
#   data/clean/staffing_panel.csv   via load_staffing_panel()
#   data/clean/quality_panel.csv    via load_quality_panel()
#
# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
#   outputs/plots/appendix/*.pdf                      one file per figure panel
#   outputs/plots/appendix/appendix_figures_index.csv sample size, window, and
#                                                     reference for every panel
#   outputs/tables/appendix_opmodel_pretrend.csv      joint test of the
#                                                     pre-period bins
#
# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
#   regressions/_setup.R
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

# -----------------------------------------------------------------------------
# Switches
# -----------------------------------------------------------------------------
RUN <- list(
  staffing_with_anticipation = TRUE,
  staffing_one_year          = TRUE,
  staffing_narrow_donut      = TRUE,
  staffing_wide_donut        = TRUE,
  staffing_pre_pandemic      = TRUE,
  staffing_pandemic          = TRUE,
  staffing_logs              = TRUE,
  opmodel_binned             = TRUE,
  quality_short_stay         = TRUE
)

appendix_dir <- file.path(out_plots_dir, "appendix")
dir.create(appendix_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
figure_index <- list()

record_figure <- function(stub, outcome, spec, window, ref, n) {
  figure_index[[length(figure_index) + 1L]] <<- tibble::tibble(
    file = paste0(stub, ".pdf"), outcome = outcome, spec = spec,
    window = window, reference = ref, n_obs = n
  )
}

save_es_pdf <- function(model, ref, xlim, stub, ylab, xlab) {
  grDevices::cairo_pdf(
    filename = file.path(appendix_dir, paste0(stub, ".pdf")),
    width = 9.5,
    height = 6.2
  )
  on.exit(dev.off(), add = TRUE)
  par(family = "Helvetica")
  iplot(
    model,
    ref  = ref,
    xlim = xlim,
    xlab = xlab,
    ylab = ylab,
    main = "",
    sub  = ""
  )
  invisible(NULL)
}

# -----------------------------------------------------------------------------
# 1) Staffing
# -----------------------------------------------------------------------------
staffing_panel <- load_staffing_panel() %>%
  dplyr::mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.factor(year_month)
  )

staffing_controls <- make_spec_controls_rhs(
  staffing_panel, spec = "A", exclude = "chain_at_start"
)

level_outcomes <- tibble::tribble(
  ~var,         ~tag,    ~ylab,
  "rn_hprd",    "rn",    "RN HPRD",
  "lpn_hprd",   "lpn",   "LPN HPRD",
  "cna_hprd",   "cna",   "CNA HPRD",
  "total_hprd", "total", "Total HPRD"
)

log_outcomes <- tibble::tribble(
  ~var,        ~tag,      ~ylab,
  "ln_rn",     "lnrn",    "Log(RN HPRD)",
  "ln_lpn",    "lnlpn",   "Log(LPN HPRD)",
  "ln_cna",    "lncna",   "Log(CNA HPRD)",
  "ln_total",  "lntotal", "Log(Total HPRD)"
)

# Builds the estimation sample for one specification: restrict the calendar
# period, cap event time at +/-window, then physically drop the excluded
# event months for treated facilities.
make_staffing_sample <- function(sample_fn, window, drop_taus) {
  dat <- staffing_panel %>%
    sample_fn() %>%
    prepare_event_study_data(min_et = -window, max_et = window)
  if (length(drop_taus)) {
    dat <- dat %>%
      dplyr::filter(!(ever_treated == 1L & event_time_capped %in% drop_taus))
  }
  dat
}

fit_staffing_es <- function(y, dat, ref, window) {
  fml <- as.formula(paste0(
    y, " ~ i(event_time_capped, ever_treated, ref = ", ref,
    ", keep = ", -window, ":", window, ") + ",
    staffing_controls,
    " | cms_certification_number + year_month"
  ))
  feols(
    fml,
    data = dat,
    vcov = ~ cms_certification_number + year_month,
    lean = TRUE
  )
}

staffing_specs <- list(
  with_anticipation = list(
    run = RUN$staffing_with_anticipation, sample = sample_full,
    window = 24L, drop = integer(0), ref = -1L
  ),
  one_year = list(
    run = RUN$staffing_one_year, sample = sample_full,
    window = 12L, drop = c(-3L, -2L, -1L), ref = -4L
  ),
  narrow_donut = list(
    run = RUN$staffing_narrow_donut, sample = sample_full,
    window = 24L, drop = c(-2L, -1L), ref = -4L
  ),
  wide_donut = list(
    run = RUN$staffing_wide_donut, sample = sample_full,
    window = 24L, drop = c(-4L, -3L, -2L, -1L), ref = -5L
  ),
  pre_pandemic = list(
    run = RUN$staffing_pre_pandemic, sample = sample_prepandemic,
    window = 24L, drop = c(-3L, -2L, -1L), ref = -4L
  ),
  pandemic = list(
    run = RUN$staffing_pandemic, sample = sample_pandemic,
    window = 24L, drop = c(-3L, -2L, -1L), ref = -4L
  )
)

for (nm in names(staffing_specs)) {
  sp <- staffing_specs[[nm]]
  if (!isTRUE(sp$run)) next

  cat("\n", strrep("=", 80), "\nSTAFFING: ", nm, "\n", strrep("=", 80), "\n", sep = "")
  dat <- make_staffing_sample(sp$sample, sp$window, sp$drop)

  for (i in seq_len(nrow(level_outcomes))) {
    o <- level_outcomes[i, ]
    mod <- fit_staffing_es(o$var, dat, sp$ref, sp$window)
    stub <- sprintf("twfe_es_%s_%s", o$tag, nm)
    save_es_pdf(
      mod, sp$ref, c(-sp$window, sp$window), stub, o$ylab,
      "Months relative to ownership change"
    )
    record_figure(stub, o$var, nm, sp$window, sp$ref, stats::nobs(mod))
    cat("[write] ", stub, ".pdf\n", sep = "")
  }
  rm(dat); gc()
}

# Log HPRD, baseline donut
if (isTRUE(RUN$staffing_logs)) {
  cat("\n", strrep("=", 80), "\nSTAFFING: logs\n", strrep("=", 80), "\n", sep = "")
  dat <- make_staffing_sample(sample_full, 24L, c(-3L, -2L, -1L))
  for (i in seq_len(nrow(log_outcomes))) {
    o <- log_outcomes[i, ]
    mod <- fit_staffing_es(o$var, dat, -4L, 24L)
    stub <- sprintf("twfe_es_%s_logs", o$tag)
    save_es_pdf(
      mod, -4L, c(-24L, 24L), stub, o$ylab,
      "Months relative to ownership change"
    )
    record_figure(stub, o$var, "logs", 24L, -4L, stats::nobs(mod))
    cat("[write] ", stub, ".pdf\n", sep = "")
  }
  rm(dat); gc()
}

# -----------------------------------------------------------------------------
# 2) Operating model, annual event-time bins
# -----------------------------------------------------------------------------
if (isTRUE(RUN$opmodel_binned)) {
  cat("\n", strrep("=", 80), "\nOPERATING MODEL: annual bins\n", strrep("=", 80), "\n", sep = "")

  opmodel_outcomes <- tibble::tribble(
    ~var,              ~tag,        ~ylab,
    "occupancy_rate",  "occupancy", "Occupancy rate (percentage points)",
    "pct_medicare",    "medicare",  "Medicare share of patient days (percentage points)",
    "pct_medicaid",    "medicaid",  "Medicaid share of patient days (percentage points)",
    "avg_los_total",   "los",       "Average length of stay (days)"
  )

  op_ref  <- -1L
  op_bins <- 3L

  op_dat <- staffing_panel %>%
    prepare_event_study_data(min_et = -12L * op_bins, max_et = 12L * op_bins) %>%
    dplyr::filter(!(ever_treated == 1L & event_time_capped %in% c(-3L, -2L, -1L))) %>%
    dplyr::mutate(
      event_year = dplyr::if_else(
        event_time_capped == 9999L,
        9999L,
        as.integer(floor(event_time_capped / 12))
      )
    )

  pretrend_rows <- list()

  for (i in seq_len(nrow(opmodel_outcomes))) {
    o <- opmodel_outcomes[i, ]
    if (!(o$var %in% names(op_dat))) {
      message("[opmodel] ", o$var, " not in staffing panel; skipped.")
      next
    }

    fml <- as.formula(paste0(
      o$var, " ~ i(event_year, ever_treated, ref = ", op_ref,
      ", keep = ", -op_bins, ":", op_bins, ") + ",
      staffing_controls,
      " | cms_certification_number + year_month"
    ))
    mod <- feols(
      fml,
      data = op_dat,
      vcov = ~ cms_certification_number + year_month,
      lean = TRUE
    )

    stub <- sprintf("twfe_es_%s_annual_bins", o$tag)
    save_es_pdf(
      mod, op_ref, c(-op_bins, op_bins), stub, o$ylab,
      "Years relative to ownership change"
    )
    record_figure(stub, o$var, "annual_bins", op_bins, op_ref, stats::nobs(mod))
    cat("[write] ", stub, ".pdf\n", sep = "")

    # Joint test that the pre-period bins (years -3 and -2) are zero.
    w <- tryCatch(
      fixest::wald(mod, keep = "event_year::-[23]:ever_treated", print = FALSE),
      error = function(e) NULL
    )
    pretrend_rows[[length(pretrend_rows) + 1L]] <- tibble::tibble(
      outcome   = o$var,
      statistic = if (is.null(w)) NA_real_ else unname(w$stat),
      df1       = if (is.null(w)) NA_real_ else unname(w$df1),
      df2       = if (is.null(w)) NA_real_ else unname(w$df2),
      p_value   = if (is.null(w)) NA_real_ else unname(w$p)
    )
  }

  pretrend_tbl <- dplyr::bind_rows(pretrend_rows)
  print(pretrend_tbl)
  readr::write_csv(pretrend_tbl, file.path(out_tables_dir, "appendix_opmodel_pretrend.csv"))
  rm(op_dat); gc()
}

# -----------------------------------------------------------------------------
# 3) Quality: short-stay measures
# -----------------------------------------------------------------------------
if (isTRUE(RUN$quality_short_stay)) {
  cat("\n", strrep("=", 80), "\nQUALITY: short-stay measures\n", strrep("=", 80), "\n", sep = "")

  quality_panel <- load_quality_panel() %>%
    dplyr::mutate(
      year_quarter = as.factor(year_quarter),
      event_time = suppressWarnings(as.integer(event_time))
    )
  quality_controls <- make_spec_controls_rhs(
    quality_panel, spec = "A", exclude = "chain_at_start"
  )

  q_window <- 8L
  q_ref    <- -1L

  for (m in names(short_stay_quality_measures)) {
    label <- unname(short_stay_quality_measures[[m]])

    dat <- trim_quality_measure_window(quality_panel, m) %>%
      drop_transition_quarter() %>%
      prepare_event_study_data(min_et = -q_window, max_et = q_window) %>%
      dplyr::filter(!is.na(.data[[m]]))

    fml <- as.formula(paste0(
      m, " ~ i(event_time_capped, ever_treated, ref = ", q_ref,
      ", keep = ", -q_window, ":", q_window, ") + ",
      quality_controls,
      " | cms_certification_number + year_quarter"
    ))
    mod <- feols(
      fml,
      data = dat,
      vcov = ~ cms_certification_number + year_quarter,
      lean = TRUE
    )

    stub <- sprintf("twfe_es_quality_%s_drop_tau0", m)
    save_es_pdf(
      mod, q_ref, c(-q_window, q_window), stub, label,
      "Quarters relative to ownership change"
    )
    record_figure(stub, m, "quality_short_stay", q_window, q_ref, stats::nobs(mod))
    cat("[write] ", stub, ".pdf\n", sep = "")
  }
}

# -----------------------------------------------------------------------------
# 4) Index
# -----------------------------------------------------------------------------
if (length(figure_index)) {
  index_tbl <- dplyr::bind_rows(figure_index)
  readr::write_csv(index_tbl, file.path(appendix_dir, "appendix_figures_index.csv"))
  cat("\nWrote ", nrow(index_tbl), " figure panels to ", appendix_dir, "\n", sep = "")
}
