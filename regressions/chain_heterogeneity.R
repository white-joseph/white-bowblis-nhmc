# =============================================================================
# regressions/chain_heterogeneity.R
#
# Estimates the effect of ownership change on quality measures separately for
# chain-affiliated and independent facilities. Heterogeneity is estimated by
# sample split rather than by interacting treatment with chain status, so that
# fixed effects and control coefficients are free to differ across the two
# groups.
#
# The staffing and business-model chain splits are produced by post_tables.R,
# where they are reported as additional panels and columns of the main
# staffing and business-model tables rather than as separate exhibits. Only
# the quality split is produced here, because the main quality table already
# uses its columns for the with- and without-staffing-controls pair and
# cannot absorb a third and fourth column.
#
# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
#   outcome ~ post + beds | facility + calendar quarter
#
# Spec A as defined in _setup.R, matching the main tables in post_tables.R.
# chain_at_start is excluded from the right-hand side because it is the
# sample-split variable and is constant within each subsample by construction.
# The transition quarter (event_time == 0) is excluded.
#
# Reported for the baseline specification only; the staffing-control variant
# used in the main quality table is not reproduced here. Vaccination measures
# are excluded.
#
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
#   data/clean/quality_panel.csv    via load_quality_panel()
#
# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
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
# Chain vs. non-chain -- Quality
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
  quality_tex,
  "\\end{document}"
)

write_fragment(preview, "chain_heterogeneity_preview.tex")

cat("\nDone. Chain-split quality table written.\n")
