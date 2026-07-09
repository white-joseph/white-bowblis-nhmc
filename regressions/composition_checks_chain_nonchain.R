# =============================================================================
# regressions/composition_checks_chain_nonchain.R
#
# Purpose:
#   Split the monthly composition/mechanism checks (occupancy rate, Medicare
#   share, Medicaid share, average length of stay) by CHAIN vs. NON-CHAIN
#   status, using the SAME baseline classification convention already used
#   for staffing (twfe_post.R): chain status as of January 2017, fixed, with
#   two fully SEPARATE regressions run on each subsample.
#
# NOTE: "chain" is dropped from the control set here (unlike the main
# composition_checks.R), since it is constant within each subsample by
# construction and would be collinear / uninformative as a control.
#
# Output:
#   outputs/tables/composition_checks_chain_nonchain.tex
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(stringr)
})

options(scipen = 999, digits = 4)

out_dir <- out_tables_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tex_out_fp <- file.path(out_dir, "composition_checks_chain_nonchain.tex")

# -----------------------------------------------------------------------------
# Load monthly staffing panel
# -----------------------------------------------------------------------------
df <- load_staffing_panel()

if (!("year" %in% names(df))) {
  df <- df %>%
    mutate(year = as.integer(str_sub(as.character(year_month), 1, 4)))
}

df_wo <- drop_anticipation_window(df)

# -----------------------------------------------------------------------------
# Baseline chain classification: January 2017 (matches twfe_post.R)
# -----------------------------------------------------------------------------
jan2017_chain <- df %>%
  filter(year_month == "2017/01") %>%
  distinct(cms_certification_number, chain)

chain_ccns <- jan2017_chain %>%
  filter(chain == 1) %>%
  pull(cms_certification_number)

nonchain_ccns <- jan2017_chain %>%
  filter(chain == 0) %>%
  pull(cms_certification_number)

df_chain    <- df_wo %>% filter(cms_certification_number %in% chain_ccns)
df_nonchain <- df_wo %>% filter(cms_certification_number %in% nonchain_ccns)

cat(sprintf(
  "[classification] chain (Jan 2017): %d facilities; non-chain (Jan 2017): %d facilities\n",
  length(chain_ccns), length(nonchain_ccns)
))

# -----------------------------------------------------------------------------
# Helper functions (same style as composition_checks.R)
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
    return("\\makecell[c]{-- \\\\ (--) }")
  }
  b <- sprintf("%.3f", s$coef)
  if (s$coef > 0) b <- paste0("\\phantom{-}", b)
  se <- sprintf("%.3f", s$se)
  if (s$stars == "") {
    paste0("\\makecell[c]{$", b, "$ \\\\ $(", se, ")$}")
  } else {
    paste0("\\makecell[c]{$", b, "^{", s$stars, "}$ \\\\ $(", se, ")$}")
  }
}

fmt_n <- function(mod) format(nobs(mod), big.mark = ",")

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
# Regression setup
# -----------------------------------------------------------------------------
vc_month <- ~ cms_certification_number + year_month

# Controls WITHOUT chain (constant within each subsample by construction).
controls_month <- c("beds", "government", "non_profit")

fit_outcome <- function(dat, lhs, controls = TRUE) {
  controls_avail <- intersect(controls_month, names(dat))
  if (controls) {
    rhs <- paste(c("post", controls_avail), collapse = " + ")
  } else {
    rhs <- "post"
  }
  feols(
    as.formula(paste0(lhs, " ~ ", rhs, " | cms_certification_number + year_month")),
    data = dat, vcov = vc_month, lean = FALSE
  )
}

outcomes <- tibble::tribble(
  ~var,             ~label,
  "occupancy_rate",  "Occupancy rate",
  "pct_medicare",    "Medicare share",
  "pct_medicaid",    "Medicaid share",
  "avg_los_total",   "Average length of stay"
)

fit_group <- function(dat) {
  purrr::map(outcomes$var, function(v) {
    list(
      nocontrols = fit_outcome(dat, v, controls = FALSE),
      controls   = fit_outcome(dat, v, controls = TRUE)
    )
  }) %>% setNames(outcomes$var)
}

fits_chain    <- fit_group(df_chain)
fits_nonchain <- fit_group(df_nonchain)

rows_chain <- purrr::map2_chr(
  outcomes$var, outcomes$label,
  ~ make_row(.y, fits_chain[[.x]]$nocontrols, fits_chain[[.x]]$controls)
)
rows_nonchain <- purrr::map2_chr(
  outcomes$var, outcomes$label,
  ~ make_row(.y, fits_nonchain[[.x]]$nocontrols, fits_nonchain[[.x]]$controls)
)

# -----------------------------------------------------------------------------
# Build LaTeX table
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
  "\\begin{table}[H]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Effects of Ownership Change: Chain vs. Non-chain Facilities}",
  "\\label{tab:composition-checks-chain-nonchain}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l Y Y r @{}}",
  "\\toprule",
  "Outcome & No controls & Controls & Observations \\\\",
  "\\midrule",
  paste0("\\multicolumn{4}{@{}l}{\\textbf{Panel A: Chain, N = ", length(chain_ccns), " facilities}} \\\\[2pt]"),
  paste0(rows_chain, " \\\\"),
  "\\addlinespace[6pt]",
  paste0("\\multicolumn{4}{@{}l}{\\textbf{Panel B: Non-chain, N = ", length(nonchain_ccns), " facilities}} \\\\[2pt]"),
  paste0(rows_nonchain, " \\\\"),
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses. The controls column adds beds, government ownership, and nonprofit ownership.",
  "\\item Standard errors are clustered two ways by facility and calendar month. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp, useBytes = TRUE)

cat("\n[write] ", normalizePath(tex_out_fp, winslash = "\\"), "\n", sep = "")
cat("Done.\n")
