# =============================================================================
# regressions/callaway_santanna.R
#
# Callaway and Sant'Anna (2021) group-time average treatment effects, as a
# robustness check against the static and stacked two-way fixed-effects
# estimates reported in the paper.
#
# Estimates ATT(g,t) for each treatment cohort g and period t, then aggregates
# into a simple overall effect, an event-study profile, cohort-specific
# effects, and a calendar-time profile.
#
# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
# Covariates follow Spec A (beds), matching the rest of the project.
# Treatment is defined by the first treated period, so the comparison group is
# either never-treated or not-yet-treated facilities depending on
# CONTROL_GROUP below.
#
# Anticipation is handled by the package's own anticipation argument rather
# than by dropping periods: setting ANTICIPATION = 3 moves the base period
# from g-1 to g-4, which is the closest analogue to the donut used elsewhere
# in the project. Note this is not identical to dropping event times -3, -2,
# and -1 from the sample; it changes which period identification is measured
# against, and the excluded periods still contribute as post-treatment.
#
# -----------------------------------------------------------------------------
# A note on frequency
# -----------------------------------------------------------------------------
# The monthly panel has 93 distinct treatment cohorts and roughly 90 periods,
# which implies on the order of several thousand ATT(g,t) cells, many
# estimated on very few facilities. This is slow and produces noisy
# cell-level estimates even when the aggregates are well behaved.
#
# TIME_UNIT below controls this. "quarter" collapses the panel to calendar
# quarters before estimation, which reduces both the number of cohorts and
# the number of periods by roughly a factor of three and is how this
# estimator is usually applied in practice. "month" runs at the native
# frequency and should be expected to take considerably longer.
#
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
#   data/clean/staffing_panel.csv   via load_staffing_panel()
#
# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
#   Console output only, plus one event-study plot per outcome in
#   outputs/plots/ when SAVE_PLOTS is TRUE.
#
# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
#   regressions/_setup.R
#   R packages: did, dplyr, ggplot2
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

if (!requireNamespace("did", quietly = TRUE)) {
  stop(
    "The 'did' package is required. Install it with install.packages(\"did\").",
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(did)
})

options(scipen = 999, digits = 4)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
TIME_UNIT     <- "quarter"   # "quarter" or "month"
OUTCOMES      <- c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd")
CONTROL_GROUP <- "notyettreated"  # or "nevertreated"
EST_METHOD    <- "dr"        # "dr" (doubly robust), "ipw", or "reg"
ANTICIPATION  <- if (TIME_UNIT == "quarter") 1L else 3L
XFORMLA       <- ~ beds
SAVE_PLOTS    <- TRUE
EVENT_WINDOW  <- if (TIME_UNIT == "quarter") c(-8L, 8L) else c(-24L, 24L)

# Set to a positive number to estimate on a random subsample of facilities.
# Useful for checking that the script runs before committing to a full pass.
SUBSAMPLE_FACILITIES <- 0

out_dir <- out_plots_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Build the estimation frame
#
# did::att_gt() requires numeric identifiers throughout: an integer facility
# id, an integer time index, and a group variable equal to the first treated
# period, with 0 for never-treated units.
# -----------------------------------------------------------------------------
keep_cols <- c(
  "cms_certification_number", "year_month", "time", "time_treated",
  "treated", "beds", OUTCOMES
)

df <- load_staffing_panel() %>%
  dplyr::select(dplyr::any_of(keep_cols))

missing_cols <- setdiff(c("time", "time_treated"), names(df))
if (length(missing_cols) > 0) {
  stop(
    sprintf(
      "Panel is missing %s, which this script uses to build the group and time indices.",
      paste(missing_cols, collapse = " and ")
    ),
    call. = FALSE
  )
}

df <- df %>%
  dplyr::mutate(
    ccn = as.character(cms_certification_number),
    time = as.integer(time),
    time_treated = suppressWarnings(as.integer(time_treated))
  ) %>%
  dplyr::filter(!is.na(time))

# Collapse to quarters when requested. Outcomes are averaged within facility
# and quarter; the cohort is the quarter containing the facility's first
# treated month.
if (identical(TIME_UNIT, "quarter")) {
  df <- df %>%
    dplyr::mutate(
      tq = (time - 1L) %/% 3L,
      gq = dplyr::if_else(is.na(time_treated), NA_integer_, (time_treated - 1L) %/% 3L)
    ) %>%
    dplyr::group_by(ccn, tq) %>%
    dplyr::summarise(
      dplyr::across(dplyr::all_of(c(OUTCOMES, "beds")), ~ mean(.x, na.rm = TRUE)),
      gq = dplyr::first(gq),
      .groups = "drop"
    ) %>%
    dplyr::rename(time = tq, time_treated = gq)
}

# Re-index time to consecutive integers starting at 1. did requires the time
# variable to be evenly spaced; re-indexing guards against gaps.
time_levels <- sort(unique(df$time))
df <- df %>%
  dplyr::mutate(
    tt = match(time, time_levels),
    gg = dplyr::if_else(is.na(time_treated), NA_integer_, match(time_treated, time_levels))
  )

# Never-treated units take group 0. Facilities first treated outside the
# observed time range cannot be placed on the index and are dropped.
n_unplaced <- sum(!is.na(df$time_treated) & is.na(df$gg))
if (n_unplaced > 0) {
  message(sprintf(
    "[cs] %s facility-periods have a treatment date outside the observed range and are dropped",
    format(n_unplaced, big.mark = ",")
  ))
  df <- df %>% dplyr::filter(is.na(time_treated) | !is.na(gg))
}

df <- df %>%
  dplyr::mutate(
    gg = dplyr::if_else(is.na(gg), 0L, as.integer(gg)),
    id = as.integer(factor(ccn))
  )

if (SUBSAMPLE_FACILITIES > 0) {
  set.seed(20240101)
  keep_ids <- sample(unique(df$id), min(SUBSAMPLE_FACILITIES, dplyr::n_distinct(df$id)))
  df <- df %>% dplyr::filter(id %in% keep_ids)
  message(sprintf("[cs] estimating on a random subsample of %s facilities",
                  format(length(keep_ids), big.mark = ",")))
}

cat(sprintf(
  "\n[cs] %s: %s rows, %s facilities, %s periods, %s treated cohorts\n",
  TIME_UNIT,
  format(nrow(df), big.mark = ","),
  format(dplyr::n_distinct(df$id), big.mark = ","),
  format(dplyr::n_distinct(df$tt), big.mark = ","),
  format(dplyr::n_distinct(df$gg[df$gg > 0]), big.mark = ",")
))
cat(sprintf("[cs] never-treated facilities: %s\n",
            format(dplyr::n_distinct(df$id[df$gg == 0]), big.mark = ",")))

# -----------------------------------------------------------------------------
# Estimate
# -----------------------------------------------------------------------------
run_cs <- function(yname) {
  cat("\n", strrep("=", 78), "\n", sep = "")
  cat("OUTCOME: ", yname, "\n", sep = "")
  cat(strrep("=", 78), "\n", sep = "")

  dat <- df %>% dplyr::filter(!is.na(.data[[yname]]), !is.na(beds))

  res <- tryCatch(
    did::att_gt(
      yname             = yname,
      tname             = "tt",
      idname            = "id",
      gname             = "gg",
      xformla           = XFORMLA,
      data              = dat,
      control_group     = CONTROL_GROUP,
      anticipation      = ANTICIPATION,
      est_method        = EST_METHOD,
      allow_unbalanced_panel = TRUE,
      base_period       = "universal",
      bstrap            = TRUE,
      cband             = TRUE,
      print_details     = FALSE
    ),
    error = function(e) {
      message(sprintf("[warn] %s failed: %s", yname, conditionMessage(e)))
      NULL
    }
  )

  if (is.null(res)) return(invisible(NULL))

  # Overall effect: a single weighted average of post-treatment ATT(g,t).
  agg_simple <- tryCatch(did::aggte(res, type = "simple", na.rm = TRUE),
                         error = function(e) NULL)
  if (!is.null(agg_simple)) {
    cat("\n--- Overall ATT (simple aggregation) ---\n")
    print(summary(agg_simple))
  }

  # Event-study profile, comparable to the TWFE and stacked event studies.
  agg_dyn <- tryCatch(
    did::aggte(res, type = "dynamic", na.rm = TRUE,
               min_e = EVENT_WINDOW[1], max_e = EVENT_WINDOW[2]),
    error = function(e) NULL
  )
  if (!is.null(agg_dyn)) {
    cat("\n--- Event-study aggregation ---\n")
    print(summary(agg_dyn))

    if (SAVE_PLOTS && requireNamespace("ggplot2", quietly = TRUE)) {
      p <- did::ggdid(agg_dyn) +
        ggplot2::labs(
          title = NULL,
          x = if (identical(TIME_UNIT, "quarter")) "Quarters relative to treatment"
              else "Months relative to treatment",
          y = yname
        )
      fp <- file.path(out_dir, sprintf("cs_es_%s_%s.pdf", yname, TIME_UNIT))
      ggplot2::ggsave(fp, p, width = 9.5, height = 6.2, device = grDevices::cairo_pdf)
      cat("[plot] ", fp, "\n", sep = "")
    }
  }

  # Cohort-specific effects: whether facilities transferred at different
  # points respond differently.
  agg_grp <- tryCatch(did::aggte(res, type = "group", na.rm = TRUE),
                      error = function(e) NULL)
  if (!is.null(agg_grp)) {
    cat("\n--- Cohort aggregation ---\n")
    print(summary(agg_grp))
  }

  # Calendar-time effects: whether the effect differs by period, which is
  # where any pandemic-era difference would appear.
  agg_cal <- tryCatch(did::aggte(res, type = "calendar", na.rm = TRUE),
                      error = function(e) NULL)
  if (!is.null(agg_cal)) {
    cat("\n--- Calendar-time aggregation ---\n")
    print(summary(agg_cal))
  }

  invisible(list(att_gt = res, simple = agg_simple, dynamic = agg_dyn,
                 group = agg_grp, calendar = agg_cal))
}

cs_results <- list()
for (y in OUTCOMES) {
  if (!(y %in% names(df))) {
    message(sprintf("[skip] %s not present in panel", y))
    next
  }
  cs_results[[y]] <- run_cs(y)
  gc(verbose = FALSE)
}

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat("\n", strrep("=", 78), "\n", sep = "")
cat("SUMMARY: overall ATT by outcome\n")
cat(strrep("=", 78), "\n", sep = "")
cat(sprintf("%-14s %14s %14s %12s\n", "Outcome", "ATT", "Std. error", "p-value"))
cat(strrep("-", 78), "\n", sep = "")

for (y in names(cs_results)) {
  s <- cs_results[[y]]$simple
  if (is.null(s)) {
    cat(sprintf("%-14s %14s %14s %12s\n", y, "--", "--", "--"))
    next
  }
  z <- s$overall.att / s$overall.se
  p <- 2 * stats::pnorm(-abs(z))
  cat(sprintf("%-14s %14.4f %14.4f %12.4f\n", y, s$overall.att, s$overall.se, p))
}

cat("\nNotes: Callaway and Sant'Anna (2021) group-time average treatment effects.\n")
cat(sprintf("Control group: %s. Estimation method: %s. Anticipation: %d period(s).\n",
            CONTROL_GROUP, EST_METHOD, ANTICIPATION))
cat("Covariates follow Spec A. Standard errors are clustered by facility via\n")
cat("the multiplier bootstrap; two-way clustering by facility and calendar\n")
cat("period is not available in this estimator.\n\n")
