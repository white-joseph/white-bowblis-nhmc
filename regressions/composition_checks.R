# =============================================================================
# composition_checks.R
#
# Purpose:
#   Run composition checks and create a standalone LaTeX document.
#
# Outcomes:
#   1. Occupancy rate: monthly panel
#   2. Medicare payer mix: annual panel
#   3. Medicaid payer mix: annual panel
#
# Output:
#   outputs/tables/composition_checks_standalone.tex
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

tex_out_fp <- file.path(out_dir, "composition_checks_standalone.tex")

# -----------------------------------------------------------------------------
# Load monthly staffing panel
# -----------------------------------------------------------------------------

df <- load_staffing_panel()

if (!("year" %in% names(df))) {
  df <- df %>%
    mutate(year = as.integer(str_sub(as.character(year_month), 1, 4)))
}

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
# 1. Monthly occupancy model
# -----------------------------------------------------------------------------
# Occupancy is measured monthly. We use the same anticipation-window exclusion
# used in the main staffing specifications.

df_monthly <- drop_anticipation_window(df)

vc_month <- ~ cms_certification_number + year_month

m_occ_rf <- feols(
  occupancy_rate ~ post | cms_certification_number + year_month,
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

monthly_basic_controls <- intersect(
  c("government", "non_profit", "chain", "beds"),
  names(df_monthly)
)

rhs_occ_basic <- paste(c("post", monthly_basic_controls), collapse = " + ")

m_occ_basic <- feols(
  as.formula(
    paste0(
      "occupancy_rate ~ ",
      rhs_occ_basic,
      " | cms_certification_number + year_month"
    )
  ),
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

# -----------------------------------------------------------------------------
# 2. Annual payer-mix models
# -----------------------------------------------------------------------------
# Payer mix should be measured annually. We collapse the monthly panel to a
# facility-year panel and exclude the ownership-change year.

event_years <- df %>%
  filter(!is.na(event_time), event_time == 0) %>%
  group_by(cms_certification_number) %>%
  summarise(
    event_year = min(year, na.rm = TRUE),
    .groups = "drop"
  )

df_annual <- df %>%
  group_by(cms_certification_number, year) %>%
  summarise(
    pct_medicare = mean(pct_medicare, na.rm = TRUE),
    pct_medicaid = mean(pct_medicaid, na.rm = TRUE),
    government = mean(government, na.rm = TRUE),
    non_profit = mean(non_profit, na.rm = TRUE),
    chain = mean(chain, na.rm = TRUE),
    beds = mean(beds, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(event_years, by = "cms_certification_number") %>%
  mutate(
    treated_annual = !is.na(event_year),
    transition_year = treated_annual & year == event_year,
    post = case_when(
      !treated_annual ~ 0,
      year > event_year ~ 1,
      year < event_year ~ 0,
      TRUE ~ NA_real_
    )
  ) %>%
  filter(!transition_year)

vc_year <- ~ cms_certification_number + year

m_mcare_rf <- feols(
  pct_medicare ~ post | cms_certification_number + year,
  data = df_annual,
  vcov = vc_year,
  lean = FALSE
)

m_mcaid_rf <- feols(
  pct_medicaid ~ post | cms_certification_number + year,
  data = df_annual,
  vcov = vc_year,
  lean = FALSE
)

annual_basic_controls <- intersect(
  c("government", "non_profit", "chain", "beds"),
  names(df_annual)
)

rhs_annual_basic <- paste(c("post", annual_basic_controls), collapse = " + ")

m_mcare_basic <- feols(
  as.formula(
    paste0(
      "pct_medicare ~ ",
      rhs_annual_basic,
      " | cms_certification_number + year"
    )
  ),
  data = df_annual,
  vcov = vc_year,
  lean = FALSE
)

m_mcaid_basic <- feols(
  as.formula(
    paste0(
      "pct_medicaid ~ ",
      rhs_annual_basic,
      " | cms_certification_number + year"
    )
  ),
  data = df_annual,
  vcov = vc_year,
  lean = FALSE
)

# -----------------------------------------------------------------------------
# Build standalone LaTeX document
# -----------------------------------------------------------------------------

row_occ <- paste(
  "Occupancy rate",
  "Monthly",
  fmt_est(m_occ_rf),
  fmt_est(m_occ_basic),
  fmt_n(m_occ_rf),
  sep = " & "
)

row_mcare <- paste(
  "Medicare share",
  "Annual",
  fmt_est(m_mcare_rf),
  fmt_est(m_mcare_basic),
  fmt_n(m_mcare_rf),
  sep = " & "
)

row_mcaid <- paste(
  "Medicaid share",
  "Annual",
  fmt_est(m_mcaid_rf),
  fmt_est(m_mcaid_basic),
  fmt_n(m_mcaid_rf),
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
  "",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "",
  "\\begin{document}",
  "",
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Effects of Ownership Change on Occupancy and Payer Mix}",
  "\\label{tab:composition-checks}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l c Y Y r @{}}",
  "\\toprule",
  "Outcome & Frequency & Reduced form & Basic controls & Observations \\\\",
  "\\midrule",
  paste0(row_occ, " \\\\"),
  paste0(row_mcare, " \\\\"),
  paste0(row_mcaid, " \\\\"),
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses. Occupancy rate is measured at the facility-month level using the monthly staffing panel. The monthly occupancy specification excludes the three months immediately preceding the ownership-change month. Medicare and Medicaid payer mix are measured at the facility-year level. The ownership-change year is excluded from the annual payer-mix specifications because annual payer-mix measures may combine pre- and post-change periods.",
  "\\item Reduced-form models include facility and time fixed effects only. Basic-controls models additionally include ownership type, chain affiliation, and beds when available. Standard errors are clustered two ways by facility and calendar period.",
  "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp)