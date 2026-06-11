# =============================================================================
# composition_checks_monthly_standalone.R
#
# Purpose:
#   Run mechanism/composition checks using the monthly facility panel.
#
# Outcomes:
#   1. Occupancy rate
#   2. Medicare payer mix
#   3. Medicaid payer mix
#   4. Case mix total, descriptive only from 2018Q2 through 2023Q2
#   5. Average length of stay
#
# Output:
#   outputs/tables/composition_checks_monthly_standalone.tex
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(stringr)
})

options(scipen = 999, digits = 4)

# -----------------------------------------------------------------------------
# Output path
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

# Create a true monthly date for sample restrictions.
# The panel writes year_month as YYYY/MM, so replace "/" with "-".
df <- df %>%
  mutate(
    ym_date = as.Date(
      paste0(str_replace(as.character(year_month), "/", "-"), "-01")
    )
  )

# Use same pre-closing adjustment window exclusion as the main staffing models.
df_monthly <- drop_anticipation_window(df)

# Case-mix descriptive sample:
# Restrict to 2018Q2 through 2023Q2.
df_casemix <- df_monthly %>%
  filter(
    ym_date >= as.Date("2018-04-01"),
    ym_date <= as.Date("2023-06-01")
  )

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

# -----------------------------------------------------------------------------
# Regression specifications
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

m_casemix_nocontrols <- feols(
  case_mix_total ~ post | cms_certification_number + year_month,
  data = df_casemix,
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
controls <- intersect(
  c("beds", "government", "non_profit", "chain"),
  names(df_monthly)
)

rhs_controls <- paste(c("post", controls), collapse = " + ")

m_occ_controls <- feols(
  as.formula(
    paste0(
      "occupancy_rate ~ ",
      rhs_controls,
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
      rhs_controls,
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
      rhs_controls,
      " | cms_certification_number + year_month"
    )
  ),
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_casemix_controls <- feols(
  as.formula(
    paste0(
      "case_mix_total ~ ",
      rhs_controls,
      " | cms_certification_number + year_month"
    )
  ),
  data = df_casemix,
  vcov = vc_month,
  lean = FALSE
)

m_los_controls <- feols(
  as.formula(
    paste0(
      "avg_los_total ~ ",
      rhs_controls,
      " | cms_certification_number + year_month"
    )
  ),
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

# -----------------------------------------------------------------------------
# Economic significance for case mix
# -----------------------------------------------------------------------------
# This is not printed to the table, but is useful to have in the environment.
# You can inspect these after running the script.

case_mix_mean <- mean(df_casemix$case_mix_total, na.rm = TRUE)

case_mix_coef_nocontrols <- coef(m_casemix_nocontrols)["post"]
case_mix_coef_controls   <- coef(m_casemix_controls)["post"]

case_mix_pct_change_nocontrols <- 100 * case_mix_coef_nocontrols / case_mix_mean
case_mix_pct_change_controls   <- 100 * case_mix_coef_controls / case_mix_mean

# -----------------------------------------------------------------------------
# Build standalone LaTeX document
# -----------------------------------------------------------------------------

row_occ <- paste(
  "Occupancy rate",
  fmt_est(m_occ_nocontrols),
  fmt_est(m_occ_controls),
  fmt_n(m_occ_nocontrols),
  sep = " & "
)

row_mcare <- paste(
  "Medicare share",
  fmt_est(m_mcare_nocontrols),
  fmt_est(m_mcare_controls),
  fmt_n(m_mcare_nocontrols),
  sep = " & "
)

row_mcaid <- paste(
  "Medicaid share",
  fmt_est(m_mcaid_nocontrols),
  fmt_est(m_mcaid_controls),
  fmt_n(m_mcaid_nocontrols),
  sep = " & "
)

row_casemix <- paste(
  "Case mix total",
  fmt_est(m_casemix_nocontrols),
  fmt_est(m_casemix_controls),
  fmt_n(m_casemix_nocontrols),
  sep = " & "
)

row_los <- paste(
  "Average length of stay",
  fmt_est(m_los_nocontrols),
  fmt_est(m_los_controls),
  fmt_n(m_los_nocontrols),
  sep = " & "
)

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
  "",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "",
  "\\begin{document}",
  "",
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Effects of Ownership Change on Occupancy, Payer Mix, Case Mix, and Length of Stay}",
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
  paste0(row_casemix, " \\\\"),
  paste0(row_los, " \\\\"),
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses. All models include facility and calendar-month fixed effects. The controls column adds beds, government ownership, nonprofit ownership, and chain affiliation.",
  "\\item Case mix is restricted to observations from 2018Q2 through 2023Q2.",
  "\\item Standard errors are clustered two ways by facility and calendar month. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp)