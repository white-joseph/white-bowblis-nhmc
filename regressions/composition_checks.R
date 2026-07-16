# =============================================================================
# composition_checks_monthly_standalone.R
#
# Purpose:
#   Run mechanism/composition checks using the monthly staffing panel and
#   short-stay quality checks using the quarterly quality panel.
#
# Monthly mechanism outcomes:
#   1. Occupancy rate
#   2. Medicare payer mix
#   3. Medicaid payer mix
#   4. Average length of stay
#
#   NOTE: Case mix has been DROPPED from this set of mechanism checks
#   (previously an outcome; removed per project decision).
#   NOTE: Spare capacity has been MOVED OUT of this script -- see
#   spare_capacity_report.R for its distribution, summary stats, and
#   regression table.
#
# Short-stay quality outcomes:
#   1. qm_430: Short-stay pneumococcal vaccine
#   2. qm_434: Short-stay newly receiving antipsychotic medication
#   3. qm_471: Short-stay improved function (trimmed to 2017-2022; missingness
#      degrades sharply in 2023 and the measure is effectively absent in 2024)
#   4. qm_472: Short-stay influenza vaccine (trimmed to 2018-2023; the measure
#      is effectively absent in 2017 and again in 2024)
#
#   NOTE: qm_424 (moderate/severe pain) and qm_425 (new/worsened pressure
#   ulcers) have been DROPPED. Missingness checks showed these measures go
#   to ~100% missing starting in 2019/2020 and stay there for the rest of
#   the panel -- not sparse reporting, effectively discontinued measures.
#   Treating them as full-panel outcomes would be misleading regardless of
#   what controls are added.
#
# Output:
#   outputs/tables/composition_checks_monthly_standalone.tex
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(stringr)
  library(readr)
})

options(scipen = 999, digits = 4)

# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------

out_dir <- out_tables_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tex_out_fp <- file.path(out_dir, "composition_checks_monthly_standalone.tex")

# -----------------------------------------------------------------------------
# Load monthly staffing panel
# -----------------------------------------------------------------------------

df <- load_staffing_panel()

if (!("year" %in% names(df))) {
  df <- df %>%
    mutate(year = as.integer(str_sub(as.character(year_month), 1, 4)))
}

df <- df %>%
  mutate(
    ym_date = as.Date(
      paste0(str_replace(as.character(year_month), "/", "-"), "-01")
    )
  )

# Use same pre-closing adjustment window exclusion as the main staffing models.
df_monthly <- drop_anticipation_window(df)

# -----------------------------------------------------------------------------
# Load quarterly quality panel
# -----------------------------------------------------------------------------

quality_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/quality_panel.csv"

df_quality <- read_csv(quality_fp, show_col_types = FALSE)

df_quality <- df_quality %>%
  mutate(
    cms_certification_number = as.character(cms_certification_number),
    quarter = as.character(quarter),
    year = as.integer(year),
    year_quarter = paste0(year, quarter),
    quarter_index = (year - 2017L) * 4L + as.integer(str_extract(quarter, "[1-4]"))
  )

# Drop event quarter for the post specification.
# This is analogous to excluding the transition quarter in your quality models.
df_quality_post <- df_quality %>%
  filter(is.na(event_time) | event_time != 0)

# -----------------------------------------------------------------------------
# Helper functions
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
  
  stars <- if (is.na(p)) {
    ""
  } else if (p < 0.01) {
    "***"
  } else if (p < 0.05) {
    "**"
  } else if (p < 0.10) {
    "*"
  } else {
    ""
  }
  
  list(coef = b, se = se, p = p, stars = stars)
}

fmt_est <- function(mod, term = "post") {
  
  s <- coef_se_star(mod, term)
  
  if (is.na(s$coef) || is.na(s$se)) {
    return("\\makecell[c]{-- \\\\ (--) }")
  }
  
  b <- sprintf("%.3f", s$coef)
  if (s$coef > 0) b <- paste0("\\phantom{-}", b)
  
  se <- sprintf("%.3f", s$se)
  
  if (s$stars == "") {
    paste0("\\makecell[c]{$", b, "$ \\\\ $(" , se, ")$}")
  } else {
    paste0("\\makecell[c]{$", b, "^{", s$stars, "}$ \\\\ $(" , se, ")$}")
  }
}

fmt_n <- function(mod) {
  format(nobs(mod), big.mark = ",")
}

make_row <- function(label, mod_nocontrols, mod_controls) {
  paste(
    label,
    fmt_est(mod_nocontrols),
    fmt_est(mod_controls),
    fmt_n(mod_nocontrols),
    sep = " & "
  )
}

# -----------------------------------------------------------------------------
# Monthly mechanism regressions
# -----------------------------------------------------------------------------

vc_month <- ~ cms_certification_number + year_month

# No-controls specification:
# Facility fixed effects + calendar-month fixed effects.
m_occ_nocontrols <- feols(
  occupancy_rate ~ post | cms_certification_number + year_month,
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_mcare_nocontrols <- feols(
  pct_medicare ~ post | cms_certification_number + year_month,
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_mcaid_nocontrols <- feols(
  pct_medicaid ~ post | cms_certification_number + year_month,
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_los_nocontrols <- feols(
  avg_los_total ~ post | cms_certification_number + year_month,
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

# Controls specification:
# Do NOT include staffing controls here.
# Staffing is a main outcome of interest and may itself respond to ownership change.
controls_month <- intersect(
  c("beds", "government", "non_profit", "chain"),
  names(df_monthly)
)

rhs_controls_month <- paste(c("post", controls_month), collapse = " + ")

m_occ_controls <- feols(
  as.formula(
    paste0(
      "occupancy_rate ~ ",
      rhs_controls_month,
      " | cms_certification_number + year_month"
    )
  ),
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_mcare_controls <- feols(
  as.formula(
    paste0(
      "pct_medicare ~ ",
      rhs_controls_month,
      " | cms_certification_number + year_month"
    )
  ),
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_mcaid_controls <- feols(
  as.formula(
    paste0(
      "pct_medicaid ~ ",
      rhs_controls_month,
      " | cms_certification_number + year_month"
    )
  ),
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_los_controls <- feols(
  as.formula(
    paste0(
      "avg_los_total ~ ",
      rhs_controls_month,
      " | cms_certification_number + year_month"
    )
  ),
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

# -----------------------------------------------------------------------------
# Short-stay quality regressions
# -----------------------------------------------------------------------------

vc_quarter <- ~ cms_certification_number + year_quarter

controls_quality <- intersect(
  c(
    "beds",
    "government",
    "non_profit",
    "chain",
    "occupancy_rate",
    "pct_medicare",
    "pct_medicaid"
  ),
  names(df_quality_post)
)

rhs_controls_quality <- paste(c("post", controls_quality), collapse = " + ")

run_quality_nocontrols <- function(outcome, dat) {
  feols(
    as.formula(
      paste0(
        outcome,
        " ~ post | cms_certification_number + year_quarter"
      )
    ),
    data = dat,
    vcov = vc_quarter,
    lean = FALSE
  )
}

run_quality_controls <- function(outcome, dat) {
  feols(
    as.formula(
      paste0(
        outcome,
        " ~ ",
        rhs_controls_quality,
        " | cms_certification_number + year_quarter"
      )
    ),
    data = dat,
    vcov = vc_quarter,
    lean = FALSE
  )
}

# year_min/year_max trim each outcome to the window where it's actually
# reported (NA = no trim, use the full sample). See notes above for why
# qm_471 and qm_472 are trimmed; qm_430 and qm_434 are flat across all
# years (consistent with an ordinary minimum-denominator suppression rule,
# not a coverage gap), so they use the full sample.
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

short_stay_models <- short_stay_specs %>%
  rowwise() %>%
  mutate(
    dat_sub = list(subset_for_outcome(df_quality_post, year_min, year_max)),
    mod_nocontrols = list(run_quality_nocontrols(outcome, dat_sub)),
    mod_controls = list(run_quality_controls(outcome, dat_sub)),
    row = make_row(label, mod_nocontrols, mod_controls)
  ) %>%
  ungroup()

# -----------------------------------------------------------------------------
# Build table rows
# -----------------------------------------------------------------------------

row_occ <- make_row(
  "Occupancy rate",
  m_occ_nocontrols,
  m_occ_controls
)

row_mcare <- make_row(
  "Medicare share",
  m_mcare_nocontrols,
  m_mcare_controls
)

row_mcaid <- make_row(
  "Medicaid share",
  m_mcaid_nocontrols,
  m_mcaid_controls
)

row_los <- make_row(
  "Average length of stay",
  m_los_nocontrols,
  m_los_controls
)

short_stay_table_rows <- paste0(short_stay_models$row, " \\\\")

short_stay_direction_lines <- paste0(
  "\\item ",
  short_stay_models$label,
  ": ",
  short_stay_models$direction,
  "."
)

# -----------------------------------------------------------------------------
# Build standalone LaTeX document
# -----------------------------------------------------------------------------
# Both tables are forced to hold their position (float package, [H] specifier)
# with no \clearpage between them, so they land on the same page.

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
  "\\usepackage{setspace}",
  "\\usepackage{float}",
  "",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "",
  "\\begin{document}",
  "",
  "% ---------------------------------------------------------------------------",
  "% Table 1: Monthly mechanism checks",
  "% ---------------------------------------------------------------------------",
  "",
  "\\begin{table}[H]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Effects of Ownership Change on Occupancy, Payer Mix, and Length of Stay}",
  "\\label{tab:composition-checks-monthly}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l Y Y r @{}}",
  "\\toprule",
  "Outcome & No controls & Controls & Observations \\\\",
  "\\midrule",
  paste0(row_occ, " \\\\"),
  paste0(row_mcare, " \\\\"),
  paste0(row_mcaid, " \\\\"),
  paste0(row_los, " \\\\"),
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses. All models include facility and calendar-month fixed effects. The controls column adds beds, government ownership, nonprofit ownership, and chain affiliation.",
  "\\item Standard errors are clustered two ways by facility and calendar month. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\vspace{1em}",
  "",
  "% ---------------------------------------------------------------------------",
  "% Table 2: Short-stay quality checks",
  "% ---------------------------------------------------------------------------",
  "",
  "\\begin{table}[H]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Effects of Ownership Change on Short-Stay Quality Measures}",
  "\\label{tab:short-stay-quality-checks}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l Y Y r @{}}",
  "\\toprule",
  "Outcome & No controls & Controls & Observations \\\\",
  "\\midrule",
  short_stay_table_rows,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses. All models include facility and calendar-quarter fixed effects. The controls column adds beds, government ownership, nonprofit ownership, chain affiliation, occupancy rate, Medicare share, and Medicaid share when available.",
  "\\item Moderate/severe pain (qm\\_424) and new/worsened pressure ulcers (qm\\_425) are dropped from this table: both become effectively unreported (near 100\\% missing) starting in 2019/2020 and remain so for the rest of the sample, rather than showing ordinary sparse reporting.",
  "\\item Improved function is estimated on 2017--2022 only; influenza vaccine is estimated on 2018--2023 only. Both measures show near-complete absence outside these windows (improved function is effectively unreported from 2023 onward; influenza vaccine is effectively unreported in 2017 and again from 2024 onward), so each is trimmed to the years where it is actually reported rather than treated as a full-panel outcome.",
  "\\item The ownership-change quarter is excluded from the short-stay quality regressions.",
  "\\item Standard errors are clustered two ways by facility and calendar quarter. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp)

# -----------------------------------------------------------------------------
# Console summary
# -----------------------------------------------------------------------------

cat("\nSaved table to:\n")
cat(tex_out_fp, "\n")
