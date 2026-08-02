# =============================================================================
# regressions/stacked_twfe_post.R
#
# Post-only TWFE difference-in-differences on the stacked (cohort-by-
# cohort) sample. Companion to stacked_event_study.R: same cohort-stacking
# construction, but a single post-period estimate rather than an
# event-time profile.
#
# -----------------------------------------------------------------------------
# Description
# -----------------------------------------------------------------------------
# For each treatment cohort (facilities sharing the same ownership-change
# month), a cohort-specific comparison sample is built consisting of that
# cohort's treated facilities plus never-treated and not-yet-treated
# facilities. The cohort samples are stacked into one dataset and a single
# TWFE regression is estimated with facility-by-cohort, calendar-month, and
# cohort fixed effects. This avoids using already-treated facilities as
# controls for later cohorts, a known source of bias in ordinary two-way
# fixed-effects estimates under staggered treatment timing.
#
# Models are fit one outcome at a time, with coefficients extracted
# immediately and the model object discarded (rm() + gc()), since the
# stacked dataset is large enough that holding several fitted models in
# memory simultaneously is costly.
#
# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
#   outcome ~ post + controls | stack_id + year_month + cohort
#
# stack_id is a facility-by-cohort identifier (a facility appearing in
# multiple cohort samples is treated as a distinct unit within each).
# controls come from make_controls_rhs() (the legacy full control set).
# Standard errors are two-way clustered by facility and calendar month.
# The donut excludes tau in {-3,-2,-1} from the treated observations.
#
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
#   data/clean/staffing_panel.csv (via load_staffing_panel())
#
# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
#   outputs/tables/stacked_twfe_post_full.tex     LaTeX fragment (label
#                                                   tab:stacked-twfe-post)
#   outputs/tables/stacked_twfe_post_full_QA.tex  Standalone compilable doc
#
# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
#   regressions/_setup.R (load_staffing_panel(), make_controls_rhs())
#   R packages: fixest, dplyr
#
# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
#   stacked_event_study.R implements the same cohort-stacking construction
#   (make_stacked_data() here vs. an equivalent function there) as a
#   separate copy. A change to the donut window or cohort definition must
#   currently be made in both files.
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(fixest)
  library(dplyr)
})

options(scipen = 999, digits = 4)

out_dir <- out_tables_dir
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# Load panel
# =============================================================================
keep_cols <- c(
  "cms_certification_number", "year_month", "time", "time_treated",
  "government", "non_profit", "chain", "beds",
  "occupancy_rate", "pct_medicare", "pct_medicaid",
  "cm_q_state_2", "cm_q_state_3", "cm_q_state_4",
  "rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd",
  "ln_rn", "ln_lpn", "ln_cna", "ln_total"
)

df <- load_staffing_panel() %>%
  dplyr::select(any_of(keep_cols)) %>%
  dplyr::mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.factor(year_month)
  )

controls_rhs <- make_controls_rhs(df)

outs_lvl <- c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd")
outs_log <- c(rn_hprd = "ln_rn", lpn_hprd = "ln_lpn", cna_hprd = "ln_cna", total_hprd = "ln_total")
nice_out <- c(rn_hprd = "RN", lpn_hprd = "LPN", cna_hprd = "CNA", total_hprd = "Total")

# =============================================================================
# Cohort assignment
# =============================================================================
# Each facility's cohort g is its treatment month (time_treated), assumed
# constant within a facility. Facilities with no treatment month (never-
# treated) get g = NA and serve as controls for every cohort.
g_df <- df %>%
  group_by(cms_certification_number) %>%
  summarise(
    g = {
      tt <- unique(time_treated[!is.na(time_treated)])
      if (length(tt) == 1) as.integer(tt) else NA_integer_
    },
    .groups = "drop"
  )

df <- df %>% left_join(g_df, by = "cms_certification_number")
cohorts <- sort(unique(df$g[!is.na(df$g)]))
cat("Unique cohorts (treated months):", length(cohorts), "\n")

# =============================================================================
# Stacked dataset construction
# =============================================================================

# -----------------------------------------------------------------------------
# make_stacked_data()
#
# Builds the cohort-stacked estimation sample. For each cohort g0, keeps
# facility-months within [g0 - L, g0 + R] of that cohort's treatment month,
# restricted to facilities that are either never-treated or treated no
# earlier than g0 (so a facility already treated before g0 does not appear
# in g0's comparison group). Within each cohort's sample, treated
# observations in the donut window (relative time in drop_set) are
# dropped, and a post indicator is constructed for relative time >= 0. All
# cohort samples are then row-bound into a single stacked dataset.
#
# Arguments:
#   data        -- Panel data frame with columns time, time_treated (via
#                   g, joined on beforehand), and the outcome/control
#                   columns to retain.
#   cohorts_vec -- Integer vector of cohort (treatment-month) values to
#                   build stacks for.
#   L           -- Integer: number of periods before g0 to include.
#                   Defaults to 24L.
#   R           -- Integer: number of periods after g0 to include.
#                   Defaults to 24L.
#   drop_set    -- Integer vector of relative-time values to exclude for
#                   treated observations (the donut window). Defaults to
#                   -3:-1.
#
# Returns:
#   A single stacked data frame (row-bound across cohorts) with columns:
#     cohort        -- The cohort (g0) this row belongs to.
#     rel           -- Relative time to g0.
#     treated_stack -- 1 if this facility is the treated cohort for this
#                        stack, 0 if serving as a control.
#     stack_id      -- Facility-by-cohort identifier (a facility appearing
#                        in multiple cohort stacks is a distinct unit in
#                        each).
#     post          -- 1 if treated_stack == 1 and rel >= 0, else 0.
#   plus the outcome and control columns selected at the end of the
#   function.
# -----------------------------------------------------------------------------
make_stacked_data <- function(data, cohorts_vec, L = 24L, R = 24L, drop_set = -3:-1) {

  stacked <- lapply(cohorts_vec, function(g0) {

    d <- data %>%
      dplyr::filter(time >= g0 - L, time <= g0 + R) %>%
      dplyr::filter(is.na(g) | g > g0 | g == g0) %>%
      dplyr::mutate(
        cohort = as.integer(g0),
        rel = as.integer(time - g0),
        treated_stack = as.integer(!is.na(g) & g == g0),
        stack_id = interaction(cms_certification_number, cohort, drop = TRUE)
      ) %>%
      # Donut: drop treated observations in the anticipation window.
      dplyr::filter(treated_stack == 0L | !(rel %in% drop_set)) %>%
      dplyr::mutate(post = as.integer(treated_stack == 1L & rel >= 0L)) %>%
      # Trim columns to control memory use across many cohort stacks.
      dplyr::select(
        cms_certification_number, year_month,
        cohort, stack_id, post,
        government, non_profit, chain, beds,
        occupancy_rate, pct_medicare, pct_medicaid,
        cm_q_state_2, cm_q_state_3, cm_q_state_4,
        rn_hprd, lpn_hprd, cna_hprd, total_hprd,
        ln_rn, ln_lpn, ln_cna, ln_total
      )

    d
  })

  dplyr::bind_rows(stacked)
}

L <- 24L
R <- 24L
stack <- make_stacked_data(df, cohorts, L = L, R = R, drop_set = -3:-1)
rm(df); gc()
cat("Stacked rows (baseline donut):", nrow(stack), "\n")

# =============================================================================
# Estimation
# =============================================================================

# -----------------------------------------------------------------------------
# make_fml()
#
# Builds the post-only TWFE formula for a given outcome, on the stacked
# sample's fixed-effect structure.
#
# Arguments:
#   lhs -- Character scalar: the outcome variable name.
#
# Returns:
#   A formula object: "lhs ~ post + controls_rhs | stack_id + year_month + cohort".
# -----------------------------------------------------------------------------
make_fml <- function(lhs) as.formula(paste0(
  lhs, " ~ post + ", controls_rhs, " | stack_id + year_month + cohort"
))

# Two-way clustering (facility + calendar month) is meaningfully more
# expensive to compute than facility-only clustering on this dataset's
# size, but matches the clustering used in stacked_event_study.R.
vc <- ~ cms_certification_number + year_month

# -----------------------------------------------------------------------------
# extract_post()
#
# Extracts the coefficient, standard error, and significance stars for a
# single term from a fitted model, without retaining a reference to the
# model object itself.
#
# Arguments:
#   mod  -- A fixest model object.
#   term -- Character scalar: the coefficient name to extract. Defaults
#           to "post".
#
# Returns:
#   A list with elements b (coefficient), se (standard error), and stars
#   (character: "", "*", "**", or "***" based on the p-value).
# -----------------------------------------------------------------------------
extract_post <- function(mod, term = "post") {
  b  <- unname(coef(mod)[term])
  se <- unname(se(mod)[term])
  p  <- unname(pvalue(mod)[term])
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(b = b, se = se, stars = stars)
}

# Fit levels one outcome at a time.
res_lvl <- list()
for (y in outs_lvl) {
  cat("[fit levels]", y, "\n")
  m <- feols(make_fml(y), data = stack, vcov = vc, lean = TRUE)
  res_lvl[[y]] <- extract_post(m)
  rm(m); gc()
}

# Fit logs one outcome at a time.
res_log <- list()
for (y in outs_lvl) {
  ly <- outs_log[[y]]
  cat("[fit logs]", ly, "\n")
  m <- feols(make_fml(ly), data = stack, vcov = vc, lean = TRUE)
  res_log[[y]] <- extract_post(m)
  rm(m); gc()
}

# =============================================================================
# LaTeX table construction
# =============================================================================

# -----------------------------------------------------------------------------
# fmt_est()
#
# Formats a coefficient, standard error, and significance stars into the
# project's \est{}{}{} LaTeX macro (coefficient over standard error, in one
# cell).
#
# Arguments:
#   b     -- Numeric: the coefficient.
#   se    -- Numeric: the standard error.
#   stars -- Character: significance stars ("", "*", "**", or "***").
#
# Returns:
#   Character scalar: a \est{...}{...}{...} macro call.
# -----------------------------------------------------------------------------
fmt_est <- function(b, se, stars) {
  bstr <- sprintf("%.3f", b)
  if (b > 0) bstr <- paste0("\\phantom{-}", bstr)
  sestr <- sprintf("%.3f", se)
  sprintf("\\est{$%s$}{$%s$}{%s}", bstr, sestr, stars)
}

# -----------------------------------------------------------------------------
# row_from_res()
#
# Builds one LaTeX table row (four outcome columns) from a named list of
# extract_post() results.
#
# Arguments:
#   reslist -- Named list keyed by outcome variable name, with each entry
#              being an extract_post() result (b, se, stars).
#
# Returns:
#   Character scalar: the four formatted cells joined with " & ".
# -----------------------------------------------------------------------------
row_from_res <- function(reslist) {
  paste(vapply(outs_lvl, function(y) {
    fmt_est(reslist[[y]]$b, reslist[[y]]$se, reslist[[y]]$stars)
  }, character(1)), collapse = "  &  ")
}

row_HPRD <- row_from_res(res_lvl)
row_LOG  <- row_from_res(res_log)

N_levels <- format(nrow(stack), big.mark = ",")
N_logs <- format(sum(complete.cases(stack[, c("ln_rn", "ln_lpn", "ln_cna", "ln_total")])), big.mark = ",")

tab <- c(
  "\\begingroup",
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{TWFE DiD Estimates of \\textit{post} on Staffing Outcomes (Stacked Sample, Baseline Donut)}",
  "\\label{tab:stacked-twfe-post}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "",
  "\\begin{tabularx}{\\textwidth}{@{} l YYYY @{} }",
  "\\toprule",
  " & \\multicolumn{4}{c}{\\textbf{Outcomes}} \\\\",
  "\\cmidrule(lr){2-5}",
  sprintf(" & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} \\\\",
          nice_out[["rn_hprd"]], nice_out[["lpn_hprd"]], nice_out[["cna_hprd"]], nice_out[["total_hprd"]]),
  "\\midrule",
  paste0("HPRD & ", row_HPRD, " \\\\"),
  "\\addlinespace[3pt]",
  paste0("Log(HPRD) & ", row_LOG, " \\\\"),
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  sprintf("\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post} with standard errors in parentheses. The stacked sample includes never-treated and not-yet-treated controls for each cohort; the donut excludes $\\tau\\in\\{-3,-2,-1\\}$. Sample sizes: $N_{\\mathrm{HPRD}}=%s$; $N_{\\mathrm{Log}}=%s$.",
          N_levels, N_logs),
  "\\item Specifications include facility-by-cohort fixed effects (stack\\_id), calendar-month fixed effects, cohort fixed effects, and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators.",
  "\\item Standard errors are clustered two ways by facility and calendar month.",
  "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "\\endgroup",
  ""
)

tab_path <- file.path(out_dir, "stacked_twfe_post_full.tex")
writeLines(tab, tab_path, useBytes = TRUE)

qa_doc <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{array}",
  "\\usepackage{makecell}",
  "\\usepackage{newtxtext}",
  "\\usepackage{newtxmath}",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "\\newcommand{\\sym}[1]{\\rlap{$^{#1}$}}",
  "\\newcommand{\\est}[3]{\\makecell[c]{#1\\sym{#3}\\\\ \\footnotesize(#2)}}",
  "\\begin{document}",
  tab,
  "\\end{document}"
)
qa_path <- file.path(out_dir, "stacked_twfe_post_full_QA.tex")
writeLines(qa_doc, qa_path, useBytes = TRUE)

cat("[write] ", normalizePath(tab_path, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(qa_path,  winslash = "\\"), "\n", sep = "")
cat("Done.\n")
