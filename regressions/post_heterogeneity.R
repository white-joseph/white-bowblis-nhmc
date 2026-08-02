# =============================================================================
# regressions/post_heterogeneity.R
#
# Sample-split heterogeneity for staffing outcomes -- one script for both
# splits CM asked about, replacing twfe_post.R's two remaining live tables.
#
# Replaces:
#   twfe_post.R's Table 2 (tab:twfe-prepost)        -- pre- vs. post-pandemic
#   twfe_post.R's Table 3 (tab:twfe-chain-nonchain)  -- chain vs. non-chain
#
# twfe_post.R's Table 1 (tab:twfe-post-full) was already superseded by
# post_staffing_table.tex in post_tables.R. Once this script exists,
# twfe_post.R has no output left that isn't reproduced elsewhere and can be
# fully retired.
#
# CHANGE FROM twfe_post.R: chain classification.
#   twfe_post.R recomputed a baseline chain status from year_month == "2017/01"
#   locally, inline, every run. This script uses chain_at_start instead --
#   the project's shared variable (built once in _setup.R's
#   build_facility_lookups(), Jan-2017 baseline with a fallback to each
#   facility's own earliest observed value for facilities absent from the
#   panel that month). Same idea, but a single definition instead of a
#   second inline copy that could drift from it.
#
# SCOPE: staffing outcomes only, matching twfe_post.R's actual scope. Does
# NOT yet cover business-model or quality heterogeneity -- extend this
# script rather than starting a new one if those are wanted later.
#
# SPECIFICATION: Spec A (post + beds), same as post_tables.R. chain_at_start
# is excluded from the controls in the chain-split table specifically
# because it is now the sample-split variable itself and is constant within
# each subsample by construction (same reasoning already used in
# composition_checks_chain_nonchain_preevent.R). It was already excluded by
# default in the pandemic split too, for consistency across both tables.
#
# Output:
#   outputs/tables/post_heterogeneity_prepandemic_table.tex  (tab:het-prepandemic)
#   outputs/tables/post_heterogeneity_chain_table.tex          (tab:het-chain)
#   outputs/tables/post_heterogeneity_preview.tex              (standalone preview doc)
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
# Estimation + formatting helpers (same conventions as post_tables.R)
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

spec_note <- paste0(
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, ",
  "with standard errors in parentheses. All specifications include facility and ",
  "calendar-month fixed effects and control for the number of certified beds. ",
  "Standard errors are two-way clustered by facility and calendar month. The ",
  "sample excludes facilities government-owned at any point during the study period. ",
  "The anticipation window ($\\tau = -3, -2, -1$) is excluded."
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

# Build a 2-panel HPRD + Log(HPRD) staffing table for any two named
# subsample data.frames. Shared by both splits below so the table layout
# can't drift between them.
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

# =============================================================================
# Load panel once, build both splits off the same base
# =============================================================================
keep_monthly <- c(
  "cms_certification_number", "year_month", "ym_date", "event_time", "post", "treated",
  "beds", "chain_at_start",
  "rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd",
  "ln_rn", "ln_lpn", "ln_cna", "ln_total"
)

df_full <- load_staffing_panel()
df <- df_full %>% dplyr::select(dplyr::any_of(keep_monthly))
rm(df_full); gc(verbose = FALSE)

df_wo <- drop_anticipation_window(df)
rm(df); gc(verbose = FALSE)

vc_month <- ~ cms_certification_number + year_month
fe_month <- "cms_certification_number + year_month"

# =============================================================================
# TABLE 1: Pre-pandemic vs. pandemic
# =============================================================================
df_pre  <- sample_prepandemic(df_wo)
df_pand <- sample_pandemic(df_wo)

prepandemic_body <- build_split_table(
  "Pre-pandemic (2017/01\u20132019/12)", df_pre,
  "Pandemic (2020/04\u20132024/06)", df_pand,
  vc_month, fe_month
)

prepandemic_tex <- wrap_table(
  prepandemic_body,
  caption = "Effect of Ownership Change on Nursing Staffing: Pre-Pandemic vs. Pandemic",
  label = "tab:het-prepandemic",
  colspec = "@{} l Y Y Y Y @{}",
  header_row = paste0("Outcome & ", paste(staff_labels, collapse = " & "), " \\\\"),
  notes = c(spec_note, sig_note)
)

write_fragment(prepandemic_tex, "post_heterogeneity_prepandemic_table.tex")

# =============================================================================
# TABLE 2: Chain vs. non-chain (baseline chain_at_start)
# =============================================================================
df_chain    <- df_wo %>% dplyr::filter(chain_at_start == 1)
df_nonchain <- df_wo %>% dplyr::filter(chain_at_start == 0)

n_missing_chain <- dplyr::n_distinct(
  df_wo$cms_certification_number[is.na(df_wo$chain_at_start)]
)
if (n_missing_chain > 0) {
  message(sprintf(
    "[het-chain] %d facilities have no chain_at_start and are excluded from both panels of the chain split",
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
  notes = c(
    spec_note,
    paste0(
      "\\item Chain status is each facility's baseline classification ",
      "(\\textit{chain\\_at\\_start}): chain status in January 2017, falling back ",
      "to the facility's own earliest observed value if absent from the panel ",
      "that month. Facilities with no available chain classification are excluded ",
      "from both panels."
    ),
    sig_note
  )
)

write_fragment(chain_tex, "post_heterogeneity_chain_table.tex")

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
  prepandemic_tex,
  "\\clearpage",
  chain_tex,
  "\\end{document}"
)

write_fragment(preview, "post_heterogeneity_preview.tex")

cat("\nDone. Two heterogeneity tables written.\n")
