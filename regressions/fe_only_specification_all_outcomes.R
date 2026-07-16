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

build_table_block <- function(rows, caption, label, extra_notes = character(0)) {
  c(
    "\\begin{table}[H]",
    "\\centering",
    "\\begin{threeparttable}",
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
    "",
    "\\begin{tablenotes}[flushleft]",
    "\\footnotesize",
    "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses. \\textbf{FE + Treatment Only} includes ONLY facility and time fixed effects plus \\textit{post} -- no other covariates. \\textbf{Standard Controls} is the fully-controlled specification used elsewhere in this project.",
    extra_notes,
    "\\item Standard errors are clustered two ways by facility and time period. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
    "\\end{tablenotes}",
    "\\end{threeparttable}",
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

staffing_rows <- sapply(seq_len(nrow(staffing_outcomes_tbl)), function(i) {
  v <- staffing_outcomes_tbl$var[i]; lab <- staffing_outcomes_tbl$label[i]
  m_fe   <- fit_fe_only(df_wo, v, "cms_certification_number + year_month")
  m_ctrl <- fit_with_controls(df_wo, v, controls_rhs_full, "cms_certification_number + year_month")
  make_row(lab, m_fe, m_ctrl)
})

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

strategic_rows <- sapply(seq_len(nrow(strategic_outcomes_tbl)), function(i) {
  v <- strategic_outcomes_tbl$var[i]; lab <- strategic_outcomes_tbl$label[i]
  m_fe   <- fit_fe_only(df_wo, v, "cms_certification_number + year_month")
  m_ctrl <- fit_with_controls(df_wo, v, strategic_ctrl_rhs, "cms_certification_number + year_month")
  make_row(lab, m_fe, m_ctrl)
})

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

short_stay_rows <- sapply(seq_len(nrow(short_stay_specs)), function(i) {
  v <- short_stay_specs$outcome[i]; lab <- short_stay_specs$label[i]
  dat_sub <- subset_for_outcome(df_quality_post, short_stay_specs$year_min[i], short_stay_specs$year_max[i])
  m_fe   <- fit_q_fe_only(dat_sub, v)
  m_ctrl <- fit_q_controls(dat_sub, v)
  make_row(lab, m_fe, m_ctrl)
})

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

long_stay_rows <- sapply(seq_len(nrow(long_stay_specs)), function(i) {
  v <- long_stay_specs$outcome[i]; lab <- long_stay_specs$label[i]
  m_fe   <- fit_q_fe_only(df_quality_post, v)
  m_ctrl <- fit_q_controls(df_quality_post, v)
  make_row(lab, m_fe, m_ctrl)
})

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
  "\\section*{FE-Only vs. Standard-Controls Specification, All Outcomes}",
  "Per C. Moul's request: each outcome estimated with (1) ONLY facility and time fixed effects plus \\textit{post} (no other covariates), and (2) the standard fully-controlled specification used elsewhere in this project, for direct comparison. All specifications use the donut design (anticipation window / transition period excluded).",
  "",
  build_table_block(
    staffing_rows,
    "Staffing Outcomes (Monthly)",
    "tab:fe-only-staffing"
  ),
  "\\clearpage",
  build_table_block(
    strategic_rows,
    "Strategic / Business-Model Outcomes (Monthly)",
    "tab:fe-only-strategic",
    extra_notes = "\\item Occupancy rate, spare capacity, Medicare share, Medicaid share, and average length of stay are excluded from each other's control set in the Standard Controls column (each pair is mechanically related)."
  ),
  build_table_block(
    short_stay_rows,
    "Short-Stay Quality Measures (Quarterly)",
    "tab:fe-only-short-stay-quality",
    extra_notes = c(
      "\\item Moderate/severe pain (qm\\_424) and new/worsened pressure ulcers (qm\\_425) are excluded (effectively unreported from 2019/2020 onward). Improved function is estimated on 2017--2022 only; influenza vaccine is estimated on 2018--2023 only, reflecting each measure's actual reporting window.",
      "\\item The ownership-change quarter is excluded from all quality regressions."
    )
  ),
  "\\clearpage",
  build_table_block(
    long_stay_rows,
    "Long-Stay Quality Measures (Quarterly)",
    "tab:fe-only-long-stay-quality",
    extra_notes = "\\item CAVEAT: pressure injuries (qm\\_453) may have a coverage transition around Q4 2023 (successor code 479 identified in a separate investigation of the short-stay measures) that has NOT been re-verified or trimmed here."
  ),
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp, useBytes = TRUE)

cat("\n[write] ", normalizePath(tex_out_fp, winslash = "\\"), "\n", sep = "")
cat("Done.\n")
