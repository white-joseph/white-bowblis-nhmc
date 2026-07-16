# =============================================================================
# regressions/twfe_post_non_nurse.R
#
# Purpose:
#   Effect of ownership change on non-nurse (therapy) staffing HPRD --
#   same core specification as the main nurse staffing regressions
#   (facility + calendar-month FE, standard controls, anticipation window
#   excluded, two-way clustering). Levels only (no raw hours / no logs,
#   per project decision -- several of these categories have a large share
#   of exact zeros, e.g. PT aide and OT aide, which makes logs and the
#   raw-hours mechanism check less meaningful here than for nursing).
#
# Outcomes:
#   PT, PT assistant, PT aide, OT, OT assistant, OT aide, SLP, and the
#   combined non-nurse total (nonnurse_total_hprd).
#
# Output:
#   outputs/tables/twfe_post_non_nurse.tex
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(tibble)
})

options(scipen = 999, digits = 4)

out_dir <- out_tables_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tex_out_fp <- file.path(out_dir, "twfe_post_non_nurse.tex")

# -----------------------------------------------------------------------------
# Load panel
# -----------------------------------------------------------------------------
df <- load_staffing_panel()

non_nurse_outcomes <- tibble::tribble(
  ~var,                 ~label,
  "pt_hprd",            "Physical Therapist (PT)",
  "ptasst_hprd",        "PT Assistant",
  "ptaide_hprd",        "PT Aide",
  "ot_hprd",            "Occupational Therapist (OT)",
  "otasst_hprd",        "OT Assistant",
  "otaide_hprd",        "OT Aide",
  "slp_hprd",           "Speech-Language Pathologist",
  "nonnurse_total_hprd","Total Non-Nurse"
)

stopifnot(all(non_nurse_outcomes$var %in% names(df)))

df_wo <- drop_anticipation_window(df)

# -----------------------------------------------------------------------------
# Regression setup -- same core spec as the main staffing regressions
# -----------------------------------------------------------------------------
vc_month <- ~ cms_certification_number + year_month
ctrls_rhs <- make_controls_rhs(df_wo)
rhs_post <- paste("post +", ctrls_rhs)

fit_outcome <- function(lhs) {
  feols(
    as.formula(paste0(lhs, " ~ ", rhs_post, " | cms_certification_number + year_month")),
    data = df_wo, vcov = vc_month, lean = FALSE
  )
}

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
  if (s$stars == "") {
    paste0("\\makecell[t]{$", b, "$ \\\\ $(", se, ")$}")
  } else {
    paste0("\\makecell[t]{$", b, "^{", s$stars, "}$ \\\\ $(", se, ")$}")
  }
}

fmt_n <- function(mod) format(nobs(mod), big.mark = ",")

non_nurse_models <- non_nurse_outcomes %>%
  rowwise() %>%
  mutate(
    mod = list(fit_outcome(var)),
    row = paste(label, fmt_est(mod), fmt_n(mod), sep = " & ")
  ) %>%
  ungroup()

table_rows <- paste0(non_nurse_models$row, " \\\\")

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
  "\\caption{Effect of Ownership Change on Non-Nurse (Therapy) Staffing}",
  "\\label{tab:twfe-post-non-nurse}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l Y r @{}}",
  "\\toprule",
  "Staff type & HPRD & Observations \\\\",
  "\\midrule",
  table_rows,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses. Facility and calendar-month fixed effects, standard controls , anticipation window excluded.",
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
