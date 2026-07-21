# =============================================================================
# regressions/fe_only_specification_all_outcomes.R
#
# Purpose:
#   Per C. Moul's request: re-estimate the FULL set of dependent variables
#   (staffing, strategic/business-model, short-stay quality, long-stay
#   quality) using ONLY facility fixed effects, time fixed effects, and the
#   treatment indicator (post) -- i.e., dropping every other control,
#   including the ones (occupancy rate, payer mix, case mix, etc.) that are
#   themselves plausibly endogenous to the ownership change.
#
#   For direct comparison, this ALSO reports the standard fully-controlled
#   specification already used elsewhere in the project, side by side.
#   Moul only explicitly asked for the FE-only numbers, but seeing them next
#   to the standard results is almost certainly the next thing he'd want,
#   given his email says he wants to see how this "runs counter to Bowblis
#   and the literature."
#
# Donut:
#   Monthly (staffing/strategic) outcomes use drop_anticipation_window()
#   (excludes event_time in {-3,-2,-1}), matching the rest of the project.
#   Quarterly (quality) outcomes exclude the transition quarter
#   (event_time == 0), matching composition_checks.R's convention.
#
# Outcome groups:
#   Staffing (monthly):    rn_hprd, lpn_hprd, cna_hprd, total_hprd
#   Strategic (monthly):   occupancy_rate, pct_medicare, pct_medicaid,
#                          avg_los_total, spare_capacity
#   Short-stay quality (quarterly): qm_430, qm_434, qm_471 (2017-2022),
#                          qm_472 (2018-2023) -- same cleaned/trimmed set
#                          used in composition_checks.R
#   Long-stay quality (quarterly):  qm_401, qm_404, qm_406, qm_407, qm_410,
#                          qm_419, qm_452, qm_453 -- matches the measures
#                          already used in the paper's quality figures
#
#   CAVEAT: qm_453 (pressure injuries) may have a coverage transition
#   around Q4 2023 (successor code 479), the same kind of issue we found
#   and corrected for the short-stay measures -- this has NOT been
#   re-verified/trimmed here, since this script's purpose is the FE-only
#   vs. standard-controls comparison, not a further data-cleaning pass.
#   Flagged explicitly rather than silently left alone.
#
# Output:
#   outputs/tables/fe_only_specification_all_outcomes.tex
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(readr)
  library(tibble)
})

options(scipen = 999, digits = 4)

out_dir <- out_tables_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
tex_out_fp <- file.path(out_dir, "fe_only_specification_all_outcomes.tex")

# -----------------------------------------------------------------------------
# Helpers (shared formatting, consistent with the rest of the project)
# -----------------------------------------------------------------------------
coef_se_star <- function(mod, term = "post") {
  ct <- summary(mod)$coeftable
  if (!(term %in% rownames(ct))) return(list(coef = NA, se = NA, stars = ""))
  b  <- unname(ct[term, "Estimate"])
  se <- unname(ct[term, "Std. Error"])
  p  <- unname(ct[term, "Pr(>|t|)"])
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(coef = b, se = se, stars = stars)
}

fmt_est <- function(mod, term = "post") {
  s <- coef_se_star(mod, term)
  if (is.na(s$coef) || is.na(s$se)) return("\\makecell[t]{-- \\\\ (--)}")
  b <- sprintf("%.4f", s$coef); if (s$coef > 0) b <- paste0("\\phantom{-}", b)
  se <- sprintf("%.4f", s$se)
  if (s$stars == "") paste0("\\makecell[t]{$", b, "$ \\\\ $(", se, ")$}")
  else paste0("\\makecell[t]{$", b, "^{", s$stars, "}$ \\\\ $(", se, ")$}")
}

fmt_n <- function(mod) format(nobs(mod), big.mark = ",")

make_row <- function(label, mod_fe_only, mod_controls) {
  paste(label, fmt_est(mod_fe_only), fmt_est(mod_controls), fmt_n(mod_fe_only), sep = " & ")
}

build_table_block <- function(rows, caption, label) {
  c(
    "\\begin{table}[H]",
    "\\centering",
    paste0("\\caption{", caption, "}"),
    paste0("\\label{", label, "}"),
    "\\small",
    "\\setlength{\\tabcolsep}{6pt}",
    "\\begin{tabularx}{\\textwidth}{@{} l Y Y r @{}}",
    "\\toprule",
    "Outcome & FE + Treatment Only & Standard Controls & Observations \\\\",
    "\\midrule",
    paste0(rows, " \\\\"),
    "\\bottomrule",
    "\\end{tabularx}",
    "\\end{table}",
    ""
  )
}

# =============================================================================
# PART A: Staffing (monthly)
# =============================================================================
df <- load_staffing_panel()
df_wo <- drop_anticipation_window(df)

vc_month <- ~ cms_certification_number + year_month
controls_rhs_full <- make_controls_rhs(df_wo)

fit_fe_only <- function(dat, lhs, fe_rhs) {
  feols(as.formula(paste0(lhs, " ~ post | ", fe_rhs)), data = dat, vcov = vc_month, lean = TRUE)
}
fit_with_controls <- function(dat, lhs, ctrls_rhs, fe_rhs) {
  feols(as.formula(paste0(lhs, " ~ post + ", ctrls_rhs, " | ", fe_rhs)), data = dat, vcov = vc_month, lean = TRUE)
}

staffing_outcomes_tbl <- tibble::tribble(
  ~var,          ~label,
  "rn_hprd",     "RN HPRD",
  "lpn_hprd",    "LPN HPRD",
  "cna_hprd",    "CNA HPRD",
  "total_hprd",  "Total HPRD"
)

staffing_rows <- character(nrow(staffing_outcomes_tbl))
for (i in seq_len(nrow(staffing_outcomes_tbl))) {
  v <- staffing_outcomes_tbl$var[i]; lab <- staffing_outcomes_tbl$label[i]
  cat(sprintf("[fit] staffing: %s\n", lab))
  m_fe   <- fit_fe_only(df_wo, v, "cms_certification_number + year_month")
  m_ctrl <- fit_with_controls(df_wo, v, controls_rhs_full, "cms_certification_number + year_month")
  staffing_rows[i] <- make_row(lab, m_fe, m_ctrl)
  rm(m_fe, m_ctrl); gc()
}

# =============================================================================
# PART B: Strategic / business-model outcomes (monthly)
# =============================================================================
strategic_choice_vars <- c("occupancy_rate", "spare_capacity", "pct_medicare", "pct_medicaid", "avg_los_total")

controls_rhs_for_strategic <- function(dat) {
  ctrls <- setdiff(get_controls(dat), strategic_choice_vars)
  paste(ctrls, collapse = " + ")
}
strategic_ctrl_rhs <- controls_rhs_for_strategic(df_wo)

strategic_outcomes_tbl <- tibble::tribble(
  ~var,               ~label,
  "occupancy_rate",   "Occupancy rate",
  "pct_medicare",     "Medicare share",
  "pct_medicaid",     "Medicaid share",
  "avg_los_total",    "Average length of stay",
  "spare_capacity",   "Spare capacity"
)
strategic_outcomes_tbl <- strategic_outcomes_tbl %>% filter(var %in% names(df_wo))

strategic_rows <- character(nrow(strategic_outcomes_tbl))
for (i in seq_len(nrow(strategic_outcomes_tbl))) {
  v <- strategic_outcomes_tbl$var[i]; lab <- strategic_outcomes_tbl$label[i]
  cat(sprintf("[fit] strategic: %s\n", lab))
  m_fe   <- fit_fe_only(df_wo, v, "cms_certification_number + year_month")
  m_ctrl <- fit_with_controls(df_wo, v, strategic_ctrl_rhs, "cms_certification_number + year_month")
  strategic_rows[i] <- make_row(lab, m_fe, m_ctrl)
  rm(m_fe, m_ctrl); gc()
}

# =============================================================================
# PART E: Standard controls -- are THEY affected by ownership change?
# =============================================================================
# Per C. Moul's follow-up: occupancy/payer-mix/LOS/spare-capacity (Part B)
# were already shown to respond to ownership change. This section checks
# the remaining standard controls not yet tested: beds, government,
# non-profit, chain, and case mix. Bivariate (FE + Treatment Only) vs.
# Standard Controls (the OTHER standard controls, self excluded), same
# pattern as Part B.
#
# for_profit is constructed explicitly (1 - government - non_profit) so it
# can be tested directly, with its own standard error, rather than inferred
# algebraically from the government/non_profit coefficients (which would
# require the covariance between two separately-fit models to get a valid
# SE -- not available from two separate regressions).
#
# government/non_profit/for_profit are mutually related by construction, so
# whichever of the three is the outcome, the OTHER TWO ownership-type
# variables are excluded from its control set (not just itself).

df_wo <- df_wo %>% mutate(for_profit = 1 - government - non_profit)

ownership_group <- c("government", "non_profit")

controls_rhs_for_standard <- function(dat, outcome) {
  exclude <- character(0)
  if (outcome %in% c("government", "non_profit", "for_profit")) exclude <- c(exclude, ownership_group)
  if (outcome == "beds") exclude <- c(exclude, "beds")
  if (outcome == "chain") exclude <- c(exclude, "chain")
  if (outcome == "case_mix_total") exclude <- c(exclude, "cm_q_state_2", "cm_q_state_3", "cm_q_state_4")
  ctrls <- setdiff(get_controls(dat), exclude)
  paste(ctrls, collapse = " + ")
}

standard_ctrl_outcomes_tbl <- tibble::tribble(
  ~var,             ~label,
  "beds",           "Beds",
  "government",     "Government ownership (0/1)",
  "non_profit",     "Non-profit ownership (0/1)",
  "for_profit",     "For-profit ownership (0/1)",
  "chain",          "Chain affiliation (0/1)",
  "case_mix_total", "Case mix (total)"
) %>% filter(var %in% names(df_wo))

standard_ctrl_rows <- character(nrow(standard_ctrl_outcomes_tbl))
for (i in seq_len(nrow(standard_ctrl_outcomes_tbl))) {
  v <- standard_ctrl_outcomes_tbl$var[i]; lab <- standard_ctrl_outcomes_tbl$label[i]
  cat(sprintf("[fit] standard control: %s\n", lab))
  ctrls_rhs_v <- controls_rhs_for_standard(df_wo, v)
  m_fe   <- fit_fe_only(df_wo, v, "cms_certification_number + year_month")
  m_ctrl <- fit_with_controls(df_wo, v, ctrls_rhs_v, "cms_certification_number + year_month")
  standard_ctrl_rows[i] <- make_row(lab, m_fe, m_ctrl)
  rm(m_fe, m_ctrl); gc()
}

rm(df, df_wo); gc()

# =============================================================================
# PART C & D: Quality (quarterly) -- short-stay and long-stay
# =============================================================================
quality_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/quality_panel.csv"
df_quality <- read_csv(quality_fp, show_col_types = FALSE) %>%
  mutate(
    cms_certification_number = as.character(cms_certification_number),
    quarter = as.character(quarter),
    year = as.integer(year),
    year_quarter = paste0(year, quarter)
  )

df_quality_post <- df_quality %>% filter(is.na(event_time) | event_time != 0)

vc_quarter <- ~ cms_certification_number + year_quarter

controls_quality <- intersect(
  c("beds", "government", "non_profit", "chain", "occupancy_rate", "pct_medicare", "pct_medicaid"),
  names(df_quality_post)
)
rhs_controls_quality <- paste(c("post", controls_quality), collapse = " + ")
rhs_fe_only_quality <- "post"

fit_q_fe_only <- function(dat, lhs) {
  feols(as.formula(paste0(lhs, " ~ ", rhs_fe_only_quality, " | cms_certification_number + year_quarter")),
        data = dat, vcov = vc_quarter, lean = TRUE)
}
fit_q_controls <- function(dat, lhs) {
  feols(as.formula(paste0(lhs, " ~ ", rhs_controls_quality, " | cms_certification_number + year_quarter")),
        data = dat, vcov = vc_quarter, lean = TRUE)
}

subset_for_outcome <- function(dat, year_min = NA_integer_, year_max = NA_integer_) {
  if (!is.na(year_min)) dat <- dat %>% filter(year >= year_min)
  if (!is.na(year_max)) dat <- dat %>% filter(year <= year_max)
  dat
}

# ---- Short-stay (cleaned/trimmed set) ----
short_stay_specs <- tibble::tribble(
  ~outcome, ~label, ~year_min, ~year_max,
  "qm_430", "Pneumococcal vaccine", NA_integer_, NA_integer_,
  "qm_434", "New antipsychotic medication", NA_integer_, NA_integer_,
  "qm_471", "Improved function", NA_integer_, 2022L,
  "qm_472", "Influenza vaccine", 2018L, 2023L
) %>% filter(outcome %in% names(df_quality_post))

short_stay_rows <- character(nrow(short_stay_specs))
for (i in seq_len(nrow(short_stay_specs))) {
  v <- short_stay_specs$outcome[i]; lab <- short_stay_specs$label[i]
  cat(sprintf("[fit] short-stay quality: %s\n", lab))
  dat_sub <- subset_for_outcome(df_quality_post, short_stay_specs$year_min[i], short_stay_specs$year_max[i])
  m_fe   <- fit_q_fe_only(dat_sub, v)
  m_ctrl <- fit_q_controls(dat_sub, v)
  short_stay_rows[i] <- make_row(lab, m_fe, m_ctrl)
  rm(m_fe, m_ctrl, dat_sub); gc()
}

# ---- Long-stay (matches the measures used in the paper's quality figures) ----
long_stay_specs <- tibble::tribble(
  ~outcome, ~label,
  "qm_401", "Decline in physical functioning",
  "qm_404", "Weight loss",
  "qm_406", "Catheter use",
  "qm_407", "Urinary tract infections",
  "qm_410", "Falls with major injury",
  "qm_419", "Anti-psychotic medication use",
  "qm_452", "Anti-anxiety/hypnotic medication use",
  "qm_453", "Pressure injuries"
) %>% filter(outcome %in% names(df_quality_post))

long_stay_rows <- character(nrow(long_stay_specs))
for (i in seq_len(nrow(long_stay_specs))) {
  v <- long_stay_specs$outcome[i]; lab <- long_stay_specs$label[i]
  cat(sprintf("[fit] long-stay quality: %s\n", lab))
  m_fe   <- fit_q_fe_only(df_quality_post, v)
  m_ctrl <- fit_q_controls(df_quality_post, v)
  long_stay_rows[i] <- make_row(lab, m_fe, m_ctrl)
  rm(m_fe, m_ctrl); gc()
}

# =============================================================================
# Assemble document
# =============================================================================
tex_lines <- c(
  "\\documentclass[12pt]{article}",
  "",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{makecell}",
  "\\usepackage{array}",
  "\\usepackage{caption}",
  "\\usepackage{float}",
  "",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "",
  "\\begin{document}",
  "",
  build_table_block(
    staffing_rows,
    "Staffing Outcomes (Monthly)",
    "tab:fe-only-staffing"
  ),
  build_table_block(
    strategic_rows,
    "Strategic / Business-Model Outcomes (Monthly)",
    "tab:fe-only-strategic"
  ),
  build_table_block(
    short_stay_rows,
    "Short-Stay Quality Measures (Quarterly)",
    "tab:fe-only-short-stay-quality"
  ),
  build_table_block(
    long_stay_rows,
    "Long-Stay Quality Measures (Quarterly)",
    "tab:fe-only-long-stay-quality"
  ),
  build_table_block(
    standard_ctrl_rows,
    "Effects of Ownership Change on Standard Control Variables",
    "tab:fe-only-standard-controls"
  ),
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp, useBytes = TRUE)

cat("\n[write] ", normalizePath(tex_out_fp, winslash = "\\"), "\n", sep = "")
cat("Done.\n")
