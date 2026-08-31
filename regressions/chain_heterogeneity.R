# =============================================================================
# regressions/chain_heterogeneity.R
#
# Estimates the effect of ownership change separately for chain-affiliated and
# independent facilities, across staffing, business-model, and quality
# outcomes. Heterogeneity is estimated by sample split rather than by
# interacting treatment with chain status, so that fixed effects and control
# coefficients are free to differ across the two groups.
#
# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
#   outcome ~ post + beds | facility + calendar period
#
# Spec A as defined in _setup.R, matching the main tables in post_tables.R.
# chain_at_start is excluded from the right-hand side in every table here
# because it is the sample-split variable and is constant within each
# subsample by construction.
#
# Monthly outcomes exclude the anticipation window (event_time in -3, -2, -1).
# Quarterly outcomes exclude the transition quarter (event_time == 0).
#
# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------
#   Table 1  Staffing: RN, LPN, CNA, and total HPRD in levels and logs.
#            Reported as two stacked panels, one per chain status, since each
#            panel already occupies four outcome columns.
#   Table 2  Business-model outcomes: occupancy rate, Medicare and Medicaid
#            shares, and average length of stay. Reported with chain and
#            non-chain estimates side by side.
#   Table 3  Quality measures, long-stay and short-stay, chain and non-chain
#            side by side. Reported for the baseline specification only;
#            the staffing-control variant used in the main quality table is
#            not reproduced here. Vaccination measures are excluded.
#
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
#   data/clean/staffing_panel.csv   via load_staffing_panel()
#   data/clean/quality_panel.csv    via load_quality_panel()
#
# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
#   outputs/tables/post_heterogeneity_chain_table.tex          (tab:het-chain)
#   outputs/tables/post_heterogeneity_chain_business_table.tex (tab:het-chain-business)
#   outputs/tables/post_heterogeneity_chain_quality_table.tex  (tab:het-chain-quality)
#   outputs/tables/chain_heterogeneity_preview.tex             (standalone preview document)
#
# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
#   regressions/_setup.R
#   R packages: dplyr, fixest, tibble
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

SPEC <- "A"
ALWAYS_EXCLUDE <- "chain_at_start"

# -----------------------------------------------------------------------------
# Estimation and formatting helpers
#
# Follows the same conventions as post_tables.R so that estimates are
# formatted identically across the main and heterogeneity tables.
# -----------------------------------------------------------------------------
fit_post <- function(dat, lhs, vc, fe_rhs) {
  rhs <- make_spec_rhs(dat, spec = SPEC, exclude = union(ALWAYS_EXCLUDE, lhs))
  feols(
    as.formula(paste0(lhs, " ~ ", rhs, " | ", fe_rhs)),
    data = dat, vcov = vc, lean = TRUE
  )
}

safe_fit <- function(dat, lhs, vc, fe_rhs, label = lhs) {
  if (!(lhs %in% names(dat)) || nrow(dat) == 0) {
    message(sprintf("[skip] %s not available for this subsample", label))
    return(NULL)
  }
  cat(sprintf("[fit] %s (N = %s)\n", label, format(nrow(dat), big.mark = ",")))
  tryCatch(
    fit_post(dat, lhs, vc, fe_rhs),
    error = function(e) {
      message(sprintf("[warn] %s failed: %s", label, e$message))
      NULL
    }
  )
}

coef_se_star <- function(mod, term = "post") {
  if (is.null(mod)) return(list(coef = NA, se = NA, stars = ""))
  ct <- summary(mod)$coeftable
  if (!(term %in% rownames(ct))) return(list(coef = NA, se = NA, stars = ""))
  p <- unname(ct[term, "Pr(>|t|)"])
  list(
    coef  = unname(ct[term, "Estimate"]),
    se    = unname(ct[term, "Std. Error"]),
    stars = if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  )
}

fmt_est <- function(mod, digits = 4) {
  s <- coef_se_star(mod)
  if (is.na(s$coef) || is.na(s$se)) return("\\makecell[t]{-- \\\\ (--)}")
  b <- formatC(s$coef, format = "f", digits = digits)
  if (s$coef > 0) b <- paste0("\\phantom{-}", b)
  se <- formatC(s$se, format = "f", digits = digits)
  if (s$stars == "") {
    paste0("\\makecell[t]{$", b, "$ \\\\ $(", se, ")$}")
  } else {
    paste0("\\makecell[t]{$", b, "^{", s$stars, "}$ \\\\ $(", se, ")$}")
  }
}

fmt_n <- function(mod) if (is.null(mod)) "--" else format(nobs(mod), big.mark = ",")

panel_header <- function(label, ncols) {
  sprintf("\\multicolumn{%d}{@{}l}{\\textbf{%s}} \\\\[2pt]", ncols, label)
}

sig_note <- "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."

spec_note_monthly <- paste0(
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, ",
  "with standard errors in parentheses. All specifications include facility and ",
  "calendar-month fixed effects and control for the number of certified beds. ",
  "Standard errors are two-way clustered by facility and calendar month."
)

spec_note_quarterly <- paste0(
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, ",
  "with standard errors in parentheses. All specifications include facility and ",
  "calendar-quarter fixed effects and control for the number of certified beds. ",
  "Standard errors are two-way clustered by facility and calendar quarter."
)

chain_note <- paste0(
  "\\item Chain status is each facility's classification at their first ",
  "observation in the panel. Facilities with no available chain classification ",
  "are excluded from every split."
)

wrap_table <- function(body, caption, label, colspec, header_row, notes, size = "\\small") {
  c(
    "\\begin{table}[!ht]",
    "\\centering",
    "\\begin{threeparttable}",
    paste0("\\caption{", caption, "}"),
    paste0("\\label{", label, "}"),
    size,
    "\\setlength{\\tabcolsep}{6pt}",
    "",
    paste0("\\begin{tabularx}{\\textwidth}{", colspec, "}"),
    "\\toprule",
    header_row,
    "\\midrule",
    body,
    "\\bottomrule",
    "\\end{tabularx}",
    "",
    "\\begin{tablenotes}[flushleft]",
    "\\footnotesize",
    notes,
    "\\end{tablenotes}",
    "",
    "\\end{threeparttable}",
    "\\end{table}"
  )
}

write_fragment <- function(lines, fname) {
  fp <- file.path(out_dir, fname)
  writeLines(lines, fp, useBytes = TRUE)
  cat("[write] ", normalizePath(fp, winslash = "\\"), "\n", sep = "")
}

# =============================================================================
# TABLE 1: Chain vs. non-chain -- Staffing
# =============================================================================
staff_labels <- c("RN", "LPN", "CNA", "Total")
staffing_rows <- list(
  list(label = "HPRD",      vars = c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd"), digits = 4),
  list(label = "Log(HPRD)", vars = c("ln_rn", "ln_lpn", "ln_cna", "ln_total"),          digits = 4)
)

build_split_table <- function(panel_a_label, panel_a_data, panel_b_label, panel_b_data,
                               vc, fe_rhs) {
  make_panel_rows <- function(label, dat) {
    rows <- character(0)
    for (i in seq_along(staffing_rows)) {
      r <- staffing_rows[[i]]
      cells <- character(4)
      for (j in seq_along(r$vars)) {
        mod <- safe_fit(dat, r$vars[j], vc, fe_rhs,
                         label = paste(label, r$label, r$vars[j]))
        cells[j] <- fmt_est(mod, digits = r$digits)
        rm(mod); gc(verbose = FALSE)
      }
      rows <- c(rows, paste0(paste(c(r$label, cells), collapse = " & "), " \\\\"))
      if (i < length(staffing_rows)) rows <- c(rows, "\\addlinespace[0.4em]")
    }
    rows
  }

  c(
    panel_header(sprintf("%s (N = %s facility-months)", panel_a_label,
                          format(nrow(panel_a_data), big.mark = ",")), 5),
    make_panel_rows(panel_a_label, panel_a_data),
    "\\addlinespace[0.7em]",
    panel_header(sprintf("%s (N = %s facility-months)", panel_b_label,
                          format(nrow(panel_b_data), big.mark = ",")), 5),
    make_panel_rows(panel_b_label, panel_b_data)
  )
}

keep_monthly <- c(
  "cms_certification_number", "year_month", "ym_date", "event_time", "post", "treated",
  "beds", "chain_at_start",
  "rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd",
  "ln_rn", "ln_lpn", "ln_cna", "ln_total",
  "occupancy_rate", "pct_medicare", "pct_medicaid", "avg_los_total"
)

df_full <- load_staffing_panel()
df <- df_full %>% dplyr::select(dplyr::any_of(keep_monthly))
rm(df_full); gc(verbose = FALSE)

df_wo <- drop_anticipation_window(df)
rm(df); gc(verbose = FALSE)

vc_month <- ~ cms_certification_number + year_month
fe_month <- "cms_certification_number + year_month"

df_chain    <- df_wo %>% dplyr::filter(chain_at_start == 1)
df_nonchain <- df_wo %>% dplyr::filter(chain_at_start == 0)

n_missing_chain <- dplyr::n_distinct(
  df_wo$cms_certification_number[is.na(df_wo$chain_at_start)]
)
if (n_missing_chain > 0) {
  message(sprintf(
    "[chain-het] %d facilities have no chain_at_start and are excluded from every split table",
    n_missing_chain
  ))
}

chain_body <- build_split_table(
  "Chain (baseline)", df_chain,
  "Non-chain (baseline)", df_nonchain,
  vc_month, fe_month
)

chain_tex <- wrap_table(
  chain_body,
  caption = "Effect of Ownership Change on Nursing Staffing: Chain vs. Non-Chain Facilities",
  label = "tab:het-chain",
  colspec = "@{} l Y Y Y Y @{}",
  header_row = paste0("Outcome & ", paste(staff_labels, collapse = " & "), " \\\\"),
  notes = c(spec_note_monthly, chain_note, sig_note)
)

write_fragment(chain_tex, "post_heterogeneity_chain_table.tex")

# =============================================================================
# TABLE 2: Chain vs. non-chain -- Business model
#
# Outcome set matches the business-model table in post_tables.R.
# =============================================================================
business_spec <- tibble::tribble(
  ~var,              ~label,
  "occupancy_rate",  "Occupancy rate",
  "pct_medicare",    "Medicare share",
  "pct_medicaid",    "Medicaid share",
  "avg_los_total",   "Average length of stay"
) %>% dplyr::filter(var %in% names(df_wo))

business_body <- character(0)
for (i in seq_len(nrow(business_spec))) {
  v <- business_spec$var[i]
  mod_chain    <- safe_fit(df_chain,    v, vc_month, fe_month, label = paste("Chain", business_spec$label[i]))
  mod_nonchain <- safe_fit(df_nonchain, v, vc_month, fe_month, label = paste("Non-chain", business_spec$label[i]))
  business_body <- c(
    business_body,
    paste0(business_spec$label[i], " & ", fmt_est(mod_chain, 4), " & ", fmt_est(mod_nonchain, 4), " \\\\")
  )
  rm(mod_chain, mod_nonchain); gc(verbose = FALSE)
}

business_tex <- wrap_table(
  business_body,
  caption = "Effect of Ownership Change on Business-Model Outcomes: Chain vs. Non-Chain Facilities",
  label = "tab:het-chain-business",
  colspec = "@{} l Y Y @{}",
  header_row = "Outcome & Chain & Non-chain \\\\",
  notes = c(
    spec_note_monthly,
    paste0(
      "\\item Occupancy rate is residents as a share of available bed-days. Payer ",
      "shares are shares of patient days. Chain sample: N = ", format(nrow(df_chain), big.mark = ","),
      " facility-months. Non-chain sample: N = ", format(nrow(df_nonchain), big.mark = ","), " facility-months."
    ),
    chain_note,
    sig_note
  )
)

write_fragment(business_tex, "post_heterogeneity_chain_business_table.tex")

rm(df_wo, df_chain, df_nonchain); gc(verbose = FALSE)

# =============================================================================
# TABLE 3: Chain vs. non-chain -- Quality
# =============================================================================
keep_quarterly <- c(
  "cms_certification_number", "year", "quarter", "year_quarter",
  "event_time", "post", "treated",
  "beds", "chain_at_start",
  names(long_stay_quality_measures),
  names(short_stay_quality_measures)
)

df_q_full <- load_quality_panel()
df_q <- df_q_full %>% dplyr::select(dplyr::any_of(keep_quarterly))
rm(df_q_full); gc(verbose = FALSE)

df_q_post <- drop_transition_quarter(df_q)
rm(df_q); gc(verbose = FALSE)

vc_quarter <- ~ cms_certification_number + year_quarter
fe_quarter <- "cms_certification_number + year_quarter"

df_q_chain    <- df_q_post %>% dplyr::filter(chain_at_start == 1)
df_q_nonchain <- df_q_post %>% dplyr::filter(chain_at_start == 0)

n_missing_chain_q <- dplyr::n_distinct(
  df_q_post$cms_certification_number[is.na(df_q_post$chain_at_start)]
)
if (n_missing_chain_q > 0) {
  message(sprintf(
    "[chain-het-quality] %d facilities have no chain_at_start and are excluded from the quality chain split",
    n_missing_chain_q
  ))
}

build_quality_chain_block <- function(codes, label_map) {
  rows <- character(0)
  for (v in codes) {
    lab <- unname(label_map[[v]])
    dat_chain    <- trim_quality_measure_window(df_q_chain, v)
    dat_nonchain <- trim_quality_measure_window(df_q_nonchain, v)
    m_chain    <- safe_fit(dat_chain,    v, vc_quarter, fe_quarter, label = paste("Chain", lab))
    m_nonchain <- safe_fit(dat_nonchain, v, vc_quarter, fe_quarter, label = paste("Non-chain", lab))
    rows <- c(rows, paste0(
      lab, " & ", fmt_est(m_chain, 4), " & ", fmt_est(m_nonchain, 4), " \\\\"
    ))
    rm(dat_chain, dat_nonchain, m_chain, m_nonchain); gc(verbose = FALSE)
  }
  rows
}

mech_rows  <- build_quality_chain_block(quality_mechanism_measures, long_stay_quality_measures)
outc_rows  <- build_quality_chain_block(quality_outcome_measures, long_stay_quality_measures)
short_rows <- build_quality_chain_block(names(short_stay_quality_measures), short_stay_quality_measures)

quality_body <- c(
  panel_header("Panel A: Long-stay labor-saving mechanism measures", 3),
  mech_rows,
  "\\addlinespace[0.6em]",
  panel_header("Panel B: Long-stay resident outcome measures", 3),
  outc_rows,
  "\\addlinespace[0.6em]",
  panel_header("Panel C: Short-stay measures", 3),
  short_rows
)

quality_tex <- wrap_table(
  quality_body,
  caption = "Effect of Ownership Change on Quality Measures: Chain vs. Non-Chain Facilities",
  label = "tab:het-chain-quality",
  colspec = "@{} l Y Y @{}",
  header_row = "Outcome & Chain & Non-chain \\\\",
  notes = c(
    spec_note_quarterly,
    paste0(
      "\\item Long-stay measures (Panels A-B) and short-stay measures (Panel C) are ",
      "constructed from different resident populations and are not directly ",
      "comparable to one another. For every measure, lower values indicate better ",
      "measured quality."
    ),
    paste0(
      "\\item Pressure injuries is estimated on 2018--2023 only; improved function ",
      "is estimated on 2017--2022 only.",
      "Chain sample: N = ", format(nrow(df_q_chain), big.mark = ","),
      " facility-quarters. Non-chain sample: N = ", format(nrow(df_q_nonchain), big.mark = ","),
      " facility-quarters."
    ),
    chain_note,
    sig_note
  )
)

write_fragment(quality_tex, "post_heterogeneity_chain_quality_table.tex")

rm(df_q_post, df_q_chain, df_q_nonchain); gc(verbose = FALSE)

# =============================================================================
# Preview document
# =============================================================================
preview <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{makecell}",
  "\\usepackage{array}",
  "\\usepackage{amsmath}",
  "\\usepackage{caption}",
  "\\captionsetup{labelfont=bf, font=small}",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "",
  "\\begin{document}",
  chain_tex,
  "\\clearpage",
  business_tex,
  "\\clearpage",
  quality_tex,
  "\\end{document}"
)

write_fragment(preview, "chain_heterogeneity_preview.tex")

cat("\nDone. Three chain-split heterogeneity tables written.\n")
