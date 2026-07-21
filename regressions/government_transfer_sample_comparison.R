# =============================================================================
# regressions/government_transfer_sample_comparison.R
#
# Purpose:
#   Investigative table (not a paper table): re-estimate ALL outcomes
#   (staffing, strategic, short-stay quality, long-stay quality) across
#   three sample variants, to see whether the ~74 Texas/Indiana
#   government-transfer facilities (identified in
#   government_transition_concentration_check.R) are diluting or distorting
#   the main results:
#
#     (1) Full sample (current, as used everywhere else in the project)
#     (2) Full sample EXCLUDING the government-transfer facilities entirely
#     (3) ONLY the government-transfer facilities as the treated group
#         (vs. never-treated)
#
#   A 4th variant is still TBD (Joe to clarify) -- easy to add as another
#   column later.
#
#   Standard fully-controlled specification throughout (not the FE-only
#   stripped-down version built for C. Moul -- this is a separate,
#   investigative question).
#
# Output:
#   outputs/tables/government_transfer_sample_comparison.tex
#   (simple table: plain header, no notes -- this is not a paper table)
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
tex_out_fp <- file.path(out_dir, "government_transfer_sample_comparison.tex")

# -----------------------------------------------------------------------------
# Helpers
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

make_row3 <- function(label, mod_full, mod_excl, mod_only) {
  paste(label, fmt_est(mod_full), fmt_est(mod_excl), fmt_est(mod_only), sep = " & ")
}

build_table_block <- function(rows, caption, label) {
  c(
    "\\begin{table}[H]",
    "\\centering",
    paste0("\\caption{", caption, "}"),
    paste0("\\label{", label, "}"),
    "\\small",
    "\\setlength{\\tabcolsep}{6pt}",
    "\\begin{tabularx}{\\textwidth}{@{} l Y Y Y @{}}",
    "\\toprule",
    "Outcome & Full Sample & Excl. Gov-Transfer & Gov-Transfer Only \\\\",
    "\\midrule",
    paste0(rows, " \\\\"),
    "\\bottomrule",
    "\\end{tabularx}",
    "\\end{table}",
    ""
  )
}

get_mode <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_real_)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# -----------------------------------------------------------------------------
# Identify the government-transfer facilities
# -----------------------------------------------------------------------------
df <- load_staffing_panel()

pre_gov <- df %>%
  filter(treated == 1, event_time >= -12, event_time <= -4) %>%
  group_by(cms_certification_number) %>%
  summarise(gov_pre = get_mode(government), .groups = "drop")

post_gov <- df %>%
  filter(treated == 1, event_time >= 4, event_time <= 12) %>%
  group_by(cms_certification_number) %>%
  summarise(gov_post = get_mode(government), .groups = "drop")

gov_transfer_ccns <- pre_gov %>%
  inner_join(post_gov, by = "cms_certification_number") %>%
  filter(gov_pre == 0, gov_post == 1) %>%
  pull(cms_certification_number)

cat(sprintf("Government-transfer facilities identified: %d\n\n", length(gov_transfer_ccns)))

df <- df %>% mutate(is_gov_transfer = cms_certification_number %in% gov_transfer_ccns)
df_wo <- drop_anticipation_window(df)

# Three monthly samples
dat_full <- df_wo
dat_excl <- df_wo %>% filter(!is_gov_transfer)
dat_only <- df_wo %>% filter(treated == 0 | is_gov_transfer)

vc_month <- ~ cms_certification_number + year_month

fit_month <- function(dat, lhs, ctrls_rhs) {
  feols(as.formula(paste0(lhs, " ~ post + ", ctrls_rhs, " | cms_certification_number + year_month")),
        data = dat, vcov = vc_month, lean = TRUE)
}

# =============================================================================
# Staffing (monthly)
# =============================================================================
staffing_ctrl_rhs <- make_controls_rhs(dat_full)

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
  m_full <- fit_month(dat_full, v, staffing_ctrl_rhs)
  m_excl <- fit_month(dat_excl, v, staffing_ctrl_rhs)
  m_only <- fit_month(dat_only, v, staffing_ctrl_rhs)
  staffing_rows[i] <- make_row3(lab, m_full, m_excl, m_only)
  rm(m_full, m_excl, m_only); gc()
}

# =============================================================================
# Strategic / business-model outcomes (monthly)
# =============================================================================
strategic_choice_vars <- c("occupancy_rate", "spare_capacity", "pct_medicare", "pct_medicaid", "avg_los_total")
strategic_ctrl_rhs <- paste(setdiff(get_controls(dat_full), strategic_choice_vars), collapse = " + ")

strategic_outcomes_tbl <- tibble::tribble(
  ~var,               ~label,
  "occupancy_rate",   "Occupancy rate",
  "pct_medicare",     "Medicare share",
  "pct_medicaid",     "Medicaid share",
  "avg_los_total",    "Average length of stay",
  "spare_capacity",   "Spare capacity"
) %>% filter(var %in% names(dat_full))

strategic_rows <- character(nrow(strategic_outcomes_tbl))
for (i in seq_len(nrow(strategic_outcomes_tbl))) {
  v <- strategic_outcomes_tbl$var[i]; lab <- strategic_outcomes_tbl$label[i]
  cat(sprintf("[fit] strategic: %s\n", lab))
  m_full <- fit_month(dat_full, v, strategic_ctrl_rhs)
  m_excl <- fit_month(dat_excl, v, strategic_ctrl_rhs)
  m_only <- fit_month(dat_only, v, strategic_ctrl_rhs)
  strategic_rows[i] <- make_row3(lab, m_full, m_excl, m_only)
  rm(m_full, m_excl, m_only); gc()
}

rm(df, df_wo, dat_full, dat_excl, dat_only); gc()

# =============================================================================
# Quality (quarterly) -- short-stay and long-stay
# =============================================================================
quality_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/quality_panel.csv"
df_quality <- read_csv(quality_fp, show_col_types = FALSE) %>%
  mutate(
    cms_certification_number = as.character(cms_certification_number),
    quarter = as.character(quarter),
    year = as.integer(year),
    year_quarter = paste0(year, quarter),
    is_gov_transfer = cms_certification_number %in% gov_transfer_ccns
  )

df_quality_post <- df_quality %>% filter(is.na(event_time) | event_time != 0)

q_full <- df_quality_post
q_excl <- df_quality_post %>% filter(!is_gov_transfer)
q_only <- df_quality_post %>% filter(treated == 0 | is_gov_transfer)

vc_quarter <- ~ cms_certification_number + year_quarter

controls_quality <- intersect(
  c("beds", "government", "non_profit", "chain", "occupancy_rate", "pct_medicare", "pct_medicaid"),
  names(df_quality_post)
)
rhs_controls_quality <- paste(c("post", controls_quality), collapse = " + ")

fit_quarter <- function(dat, lhs) {
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
  ymin <- short_stay_specs$year_min[i]; ymax <- short_stay_specs$year_max[i]
  m_full <- fit_quarter(subset_for_outcome(q_full, ymin, ymax), v)
  m_excl <- fit_quarter(subset_for_outcome(q_excl, ymin, ymax), v)
  m_only <- fit_quarter(subset_for_outcome(q_only, ymin, ymax), v)
  short_stay_rows[i] <- make_row3(lab, m_full, m_excl, m_only)
  rm(m_full, m_excl, m_only); gc()
}

# ---- Long-stay ----
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
  m_full <- fit_quarter(q_full, v)
  m_excl <- fit_quarter(q_excl, v)
  m_only <- fit_quarter(q_only, v)
  long_stay_rows[i] <- make_row3(lab, m_full, m_excl, m_only)
  rm(m_full, m_excl, m_only); gc()
}

# =============================================================================
# Assemble document (simple header, no notes -- investigative only)
# =============================================================================
tex_lines <- c(
  "\\documentclass[12pt]{article}",
  "",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{makecell}",
  "\\usepackage{array}",
  "\\usepackage{float}",
  "",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "",
  "\\begin{document}",
  "",
  build_table_block(staffing_rows, "Staffing Outcomes (Monthly)", "tab:govtransfer-staffing"),
  build_table_block(strategic_rows, "Strategic / Business-Model Outcomes (Monthly)", "tab:govtransfer-strategic"),
  build_table_block(short_stay_rows, "Short-Stay Quality Measures (Quarterly)", "tab:govtransfer-short-stay"),
  build_table_block(long_stay_rows, "Long-Stay Quality Measures (Quarterly)", "tab:govtransfer-long-stay"),
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp, useBytes = TRUE)

cat("\n[write] ", normalizePath(tex_out_fp, winslash = "\\"), "\n", sep = "")
cat("Done.\n")
