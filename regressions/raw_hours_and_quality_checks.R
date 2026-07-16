# =============================================================================
# regressions/raw_hours_and_quality_checks.R
#
# Purpose:
#   Combined document addressing two outstanding advisor requests:
#
#   (A) Staffing on RAW HOURS alongside HPRD (C. Moul, clarified by
#       J. Bowblis: report BOTH, since HPRD is the industry-standard measure
#       reviewers expect, while raw hours (the numerator only) tests whether
#       the occupancy-rate increase is mechanically driving the HPRD decline
#       through the denominator).
#
#   (B) Short-stay quality regressions WITH vs. WITHOUT occupancy rate as a
#       control (C. Moul), to gauge how much of the quality-measure effect
#       is attributable to the occupancy increase itself. Built on the
#       already-cleaned short-stay measure set (qm_424/qm_425 dropped;
#       qm_471/qm_472 trimmed to their actual reporting windows -- see
#       composition_checks.R for that investigation).
#
# Output:
#   outputs/tables/raw_hours_and_quality_checks.tex
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(stringr)
  library(readr)
  library(tibble)
})

options(scipen = 999, digits = 4)

out_dir <- out_tables_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tex_out_fp <- file.path(out_dir, "raw_hours_and_quality_checks.tex")

# -----------------------------------------------------------------------------
# Helper functions (shared style with composition_checks.R)
# -----------------------------------------------------------------------------

coef_se_star <- function(mod, term = "post") {
  sm <- summary(mod)
  ct <- sm$coeftable
  if (!(term %in% rownames(ct))) {
    return(list(coef = NA_real_, se = NA_real_, p = NA_real_, stars = ""))
  }
  b  <- unname(ct[term, "Estimate"])
  se <- unname(ct[term, "Std. Error"])
  p  <- unname(ct[term, "Pr(>|t|)"])
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(coef = b, se = se, p = p, stars = stars)
}

fmt_est <- function(mod, term = "post") {
  s <- coef_se_star(mod, term)
  if (is.na(s$coef) || is.na(s$se)) {
    return("\\makecell[t]{-- \\\\ (--)}")
  }
  b <- sprintf("%.4f", s$coef)
  if (s$coef > 0) b <- paste0("\\phantom{-}", b)
  se <- sprintf("%.4f", s$se)
  if (s$stars == "") {
    paste0("\\makecell[t]{$", b, "$ \\\\ $(", se, ")$}")
  } else {
    paste0("\\makecell[t]{$", b, "^{", s$stars, "}$ \\\\ $(", se, ")$}")
  }
}

fmt_n <- function(mod) format(nobs(mod), big.mark = ",")

# =============================================================================
# PART A: Staffing on raw hours alongside HPRD
# =============================================================================

df <- load_staffing_panel()
stopifnot(all(c("rn_hours_month", "lpn_hours_month", "cna_hours_month", "total_hours",
                "ln_rn_hours", "ln_lpn_hours", "ln_cna_hours", "ln_total_hours") %in% names(df)))

df_wo <- drop_anticipation_window(df)

vc_month <- ~ cms_certification_number + year_month
ctrls_rhs <- make_controls_rhs(df_wo)
rhs_post <- paste("post +", ctrls_rhs)

staffing_pairs <- tibble::tribble(
  ~label,   ~hprd_var,    ~hours_level_var,   ~hours_log_var,
  "RN",     "rn_hprd",    "rn_hours_month",   "ln_rn_hours",
  "LPN",    "lpn_hprd",   "lpn_hours_month",  "ln_lpn_hours",
  "CNA",    "cna_hprd",   "cna_hours_month",  "ln_cna_hours",
  "Total",  "total_hprd", "total_hours",      "ln_total_hours"
)

fit_staffing <- function(lhs) {
  feols(
    as.formula(paste0(lhs, " ~ ", rhs_post, " | cms_certification_number + year_month")),
    data = df_wo, vcov = vc_month, lean = FALSE
  )
}

staffing_models <- staffing_pairs %>%
  rowwise() %>%
  mutate(
    mod_hprd        = list(fit_staffing(hprd_var)),
    mod_hours_level = list(fit_staffing(hours_level_var)),
    mod_hours_log   = list(fit_staffing(hours_log_var)),
    row = paste(
      label,
      fmt_est(mod_hprd),
      fmt_est(mod_hours_level),
      fmt_est(mod_hours_log),
      fmt_n(mod_hprd),
      sep = " & "
    )
  ) %>%
  ungroup()

staffing_table_rows <- paste0(staffing_models$row, " \\\\")

# =============================================================================
# PART B: Short-stay quality with vs. without occupancy rate as a control
# =============================================================================

quality_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/quality_panel.csv"
df_quality <- read_csv(quality_fp, show_col_types = FALSE)

df_quality <- df_quality %>%
  mutate(
    cms_certification_number = as.character(cms_certification_number),
    quarter = as.character(quarter),
    year = as.integer(year),
    year_quarter = paste0(year, quarter)
  )

df_quality_post <- df_quality %>%
  filter(is.na(event_time) | event_time != 0)

vc_quarter <- ~ cms_certification_number + year_quarter

controls_quality_full <- intersect(
  c("beds", "government", "non_profit", "chain", "occupancy_rate", "pct_medicare", "pct_medicaid"),
  names(df_quality_post)
)
controls_quality_no_occ <- setdiff(controls_quality_full, "occupancy_rate")

rhs_no_controls   <- "post"
rhs_excl_occ      <- paste(c("post", controls_quality_no_occ), collapse = " + ")
rhs_incl_occ      <- paste(c("post", controls_quality_full), collapse = " + ")

run_quality <- function(outcome, rhs, dat) {
  feols(
    as.formula(paste0(outcome, " ~ ", rhs, " | cms_certification_number + year_quarter")),
    data = dat, vcov = vc_quarter, lean = FALSE
  )
}

# Same trimming as composition_checks.R: qm_424/qm_425 dropped entirely;
# qm_471 trimmed to 2017-2022; qm_472 trimmed to 2018-2023.
short_stay_specs <- tibble::tribble(
  ~outcome, ~label, ~direction, ~year_min, ~year_max,
  "qm_430", "Pneumococcal vaccine", "Higher is better", NA_integer_, NA_integer_,
  "qm_434", "New antipsychotic medication", "Lower is better", NA_integer_, NA_integer_,
  "qm_471", "Improved function", "Higher is better", NA_integer_, 2022L,
  "qm_472", "Influenza vaccine", "Higher is better", 2018L, 2023L
) %>%
  filter(outcome %in% names(df_quality_post))

subset_for_outcome <- function(dat, year_min, year_max) {
  if (!is.na(year_min)) dat <- dat %>% filter(year >= year_min)
  if (!is.na(year_max)) dat <- dat %>% filter(year <= year_max)
  dat
}

quality_models <- short_stay_specs %>%
  rowwise() %>%
  mutate(
    dat_sub = list(subset_for_outcome(df_quality_post, year_min, year_max)),
    mod_nocontrols = list(run_quality(outcome, rhs_no_controls, dat_sub)),
    mod_excl_occ   = list(run_quality(outcome, rhs_excl_occ, dat_sub)),
    mod_incl_occ   = list(run_quality(outcome, rhs_incl_occ, dat_sub)),
    row = paste(
      label,
      fmt_est(mod_nocontrols),
      fmt_est(mod_excl_occ),
      fmt_est(mod_incl_occ),
      fmt_n(mod_nocontrols),
      sep = " & "
    )
  ) %>%
  ungroup()

quality_table_rows <- paste0(quality_models$row, " \\\\")

# -----------------------------------------------------------------------------
# Build combined LaTeX document
# -----------------------------------------------------------------------------

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
  "% ---------------------------------------------------------------------------",
  "% Table 1: Staffing -- HPRD vs. raw hours",
  "% ---------------------------------------------------------------------------",
  "",
  "\\begin{table}[H]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Effect of Ownership Change on Staffing: HPRD vs. Raw Hours}",
  "\\label{tab:staffing-hprd-vs-hours}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l Y Y Y r @{}}",
  "\\toprule",
  "Staff type & HPRD & Raw hours & log(Raw hours) & Observations \\\\",
  "\\midrule",
  staffing_table_rows,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses.",
  "\\item All models include facility and calendar-month fixed effects and the standard controls. Anticipation window excluded.",
  "\\item Standard errors are clustered two ways by facility and calendar month. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\vspace{1em}",
  "",
  "% ---------------------------------------------------------------------------",
  "% Table 2: Short-stay quality -- with vs. without occupancy rate control",
  "% ---------------------------------------------------------------------------",
  "",
  "\\begin{table}[H]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Effects of Ownership Change on Short-Stay Quality Measures: With vs. Without Occupancy Rate as a Control}",
  "\\label{tab:quality-occupancy-control}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l Y Y Y r @{}}",
  "\\toprule",
  "Outcome & No controls & Controls (excl. occupancy) & Controls (incl. occupancy) & Observations \\\\",
  "\\midrule",
  quality_table_rows,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses. All models include facility and calendar-quarter fixed effects.",
  "\\item Improved function is estimated on 2017--2022 only; influenza vaccine is estimated on 2018--2023 only.",
  "\\item The ownership-change quarter is excluded from all short-stay quality regressions.",
  "\\item Standard errors are clustered two ways by facility and calendar quarter. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp, useBytes = TRUE)

cat("\n[write] ", normalizePath(tex_out_fp, winslash = "\\"), "\n", sep = "")
cat("Done.\n")
