# =============================================================================
# composition_checks_monthly_standalone.R
#
# Purpose:
#   Run composition checks using the monthly facility panel.
#
# Outcomes:
#   1. Occupancy rate
#   2. Medicare payer mix
#   3. Medicaid payer mix
#   4. Case mix total, descriptive only through 2023Q2
#
# Notes:
#   Occupancy is measured monthly.
#   Payer-mix variables are cost-report-period measures merged to the monthly panel.
#   Case mix is included descriptively and restricted through 2023Q2.
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

# Create a true monthly date for sample restrictions
df <- df %>%
  mutate(
    ym_date = as.Date(
      paste0(str_replace(as.character(year_month), "/", "-"), "-01")
    )
  )

# Use same pre-closing adjustment window exclusion as the main staffing models
df_monthly <- drop_anticipation_window(df)

# Case-mix descriptive sample:
# Restrict through 2023Q2, i.e. through June 2023.
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
  "\\caption{Effects of Ownership Change on Occupancy, Payer Mix, and Case Mix}",
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
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses. All models include facility and calendar-month fixed effects. The controls column adds beds, government ownership, nonprofit ownership, and chain affiliation.",
  "\\item Case-mix is restricted to observations from Q2 of 2018 through Q2 of 2023.",
  "\\item Standard errors are clustered two ways by facility and calendar month. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp)