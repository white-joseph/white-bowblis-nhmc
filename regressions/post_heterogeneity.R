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
# TABLE 3 ADDED (2026-08-02): occupancy-bin heterogeneity, folded in from
# occupancy_bin_heterogeneity.R / occupancy_bin_heterogeneity_with_nevertreated.R.
# Per Joe: use the FULL-PANEL (with-never-treated) design specifically --
# see the Moul email thread on why never-treated facilities belong in the
# estimation sample as controls for the calendar fixed effects, even though
# their own bin assignment is a mathematically inert placeholder (post = 0
# for them always). The ever-treated-only version is dropped, not kept as
# an alternative.
#
# The two old scripts wrote to the SAME output filename
# (occupancy_bin_heterogeneity.tex) -- whichever ran last silently
# overwrote the other's table. That bug is moot once there is only one
# version of this table.
#
# SPEC CHANGE: the old scripts used the full control set with a
# "strategic-choice variables excluded from each other's controls" rule
# (occupancy/payer-mix/LOS could not control for one another). Under Spec A
# (post + beds), none of those variables are controls to begin with, so
# that exclusion list is unnecessary here -- not just dropped for
# consistency, but genuinely moot under this specification.
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
#   outputs/tables/post_heterogeneity_occupancy_bin_table.tex   (tab:het-occupancy-bin)
#   outputs/tables/occupancy_bin_classification_summary.csv    (baseline bin assignments)
#   outputs/tables/post_heterogeneity_preview.tex              (standalone preview doc)
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(tibble)
  library(readr)
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

rm(df_wo, df_pre, df_pand, df_chain, df_nonchain); gc(verbose = FALSE)

# =============================================================================
# TABLE 3: Occupancy-bin heterogeneity (full panel, per Moul)
#
# Bins are defined by TREATED facilities' baseline occupancy_rate, averaged
# over event_time in [-12,-4] (before the anticipation window). Never-
# treated facilities are assigned a placeholder bin ("<70%", the reference
# level) -- this is mathematically inert since post = 0 for them always, so
# it never enters post:occ_bin. Including them changes what the calendar
# fixed effects (and therefore post) are estimated relative to, which is
# the whole point of using the full-panel design per Moul.
#
# Model: outcome ~ post + post:occ_bin + beds | facility + calendar-month.
# Reference bin ("<70%") reports its own total effect via the bare `post`
# coefficient; every other bin reports its raw DIFFERENCE from the
# reference (the post:occ_binX interaction), which is what supports formal
# cross-bin comparison rather than reading each bin's total effect in
# isolation. A pooled (non-binned) column is included alongside for direct
# comparison to the bin-specific estimates.
# =============================================================================
keep_occ <- c(
  "cms_certification_number", "year_month", "event_time", "post", "treated",
  "beds", "chain_at_start",
  "occupancy_rate", "pct_medicare", "pct_medicaid", "avg_los_total"
)

df_occ_full <- load_staffing_panel()
df_occ_raw <- df_occ_full %>% dplyr::select(dplyr::any_of(keep_occ))
rm(df_occ_full); gc(verbose = FALSE)

bin_level_order <- c("<70%", "70-80%", "80-90%", "90-95%", ">95%")
REF_BIN <- "<70%"

assign_occ_bin <- function(x) {
  dplyr::case_when(
    x < 70 ~ "<70%",
    x >= 70 & x < 80 ~ "70-80%",
    x >= 80 & x < 90 ~ "80-90%",
    x >= 90 & x <= 95 ~ "90-95%",
    x > 95 ~ ">95%",
    TRUE ~ NA_character_
  )
}

baseline_treated <- df_occ_raw %>%
  dplyr::filter(treated == 1, event_time >= -12, event_time <= -4) %>%
  dplyr::group_by(cms_certification_number) %>%
  dplyr::summarise(
    baseline_occupancy = mean(occupancy_rate, na.rm = TRUE),
    n_baseline_months = sum(!is.na(occupancy_rate)),
    .groups = "drop"
  ) %>%
  dplyr::filter(n_baseline_months > 0, is.finite(baseline_occupancy)) %>%
  dplyr::mutate(occ_bin_treated = assign_occ_bin(baseline_occupancy))

write_csv(baseline_treated, file.path(out_dir, "occupancy_bin_classification_summary.csv"))

n_nevertreated_total <- dplyr::n_distinct(df_occ_raw$cms_certification_number[df_occ_raw$treated == 0])

cat("\n=== Occupancy-bin classification ===\n")
cat(sprintf("Treated facilities with usable baseline (event_time -12 to -4): %d\n", nrow(baseline_treated)))
cat(sprintf("Never-treated facilities (ALL included, placeholder bin -- inert): %d\n", n_nevertreated_total))
print(table(baseline_treated$occ_bin_treated))

df_occ <- df_occ_raw %>%
  drop_anticipation_window() %>%
  dplyr::left_join(
    baseline_treated %>% dplyr::select(cms_certification_number, baseline_occupancy, occ_bin_treated),
    by = "cms_certification_number"
  ) %>%
  dplyr::filter(treated == 0 | !is.na(occ_bin_treated)) %>%
  dplyr::mutate(
    occ_bin = dplyr::if_else(treated == 1, occ_bin_treated, REF_BIN),
    occ_bin = factor(occ_bin, levels = bin_level_order),
    occ_bin = relevel(occ_bin, ref = REF_BIN)
  ) %>%
  dplyr::select(-occ_bin_treated)
rm(df_occ_raw); gc(verbose = FALSE)

cat(sprintf(
  "[het-occupancy-bin] facility-months: %s (%d facilities)\n\n",
  format(nrow(df_occ), big.mark = ","), dplyr::n_distinct(df_occ$cms_certification_number)
))

bin_display_labels <- c(
  "<70%"   = "$<$70\\%",
  "70-80%" = "70--80\\%",
  "80-90%" = "80--90\\%",
  "90-95%" = "90--95\\%",
  ">95%"   = "$>$95\\%"
)
bin_n <- baseline_treated %>% dplyr::count(occ_bin_treated, name = "n_facilities")

get_bin_effect <- function(mod, bin_label) {
  if (bin_label == REF_BIN) return(coef_se_star(mod, "post"))
  coef_se_star(mod, paste0("post:occ_bin", bin_label))
}

occ_outcomes <- tibble::tribble(
  ~var,             ~label,
  "occupancy_rate", "Occupancy rate",
  "pct_medicare",   "Medicare share",
  "pct_medicaid",   "Medicaid share",
  "avg_los_total",  "Average length of stay"
) %>% dplyr::filter(var %in% names(df_occ))

fit_occ_outcome <- function(v) {
  rhs <- make_spec_rhs(df_occ, spec = SPEC, exclude = union(ALWAYS_EXCLUDE, v))
  fml_binned <- as.formula(sprintf("%s ~ post + post:occ_bin + %s | %s", v, rhs, fe_month))
  fml_pooled <- as.formula(sprintf("%s ~ %s | %s", v, rhs, fe_month))
  list(
    binned = tryCatch(feols(fml_binned, data = df_occ, vcov = vc_month, lean = TRUE),
                       error = function(e) { message(sprintf("[warn] %s (binned) failed: %s", v, e$message)); NULL }),
    pooled = tryCatch(feols(fml_pooled, data = df_occ, vcov = vc_month, lean = TRUE),
                       error = function(e) { message(sprintf("[warn] %s (pooled) failed: %s", v, e$message)); NULL })
  )
}

cat("[het-occupancy-bin] fitting", nrow(occ_outcomes), "outcomes (binned + pooled)\n")
occ_fits <- setNames(lapply(occ_outcomes$var, fit_occ_outcome), occ_outcomes$var)

fmt_bin_cell <- function(s) {
  # Single-line cell (coefficient and SE side by side), not stacked. The
  # 2-line \makecell version produced a vertical misalignment on the
  # "Average length of stay" row specifically -- the row where every bin
  # cell lacks significance stars but Pooled has them, so Pooled's taller
  # first line (from the "^{***}" superscript) sits at a different height
  # than its unstarred neighbors. This is the same failure mode already
  # fixed in robustness_checks.R by switching to single-line cells, applied
  # here for the same reason rather than guessing at a third theory.
  if (is.na(s$coef) || is.na(s$se)) return("--")
  b <- formatC(s$coef, format = "f", digits = 4)
  if (s$coef > 0) b <- paste0("\\phantom{-}", b)
  se <- formatC(s$se, format = "f", digits = 4)
  if (s$stars == "") {
    paste0("$", b, "$ $(", se, ")$")
  } else {
    paste0("$", b, "^{", s$stars, "}$ $(", se, ")$")
  }
}

bin_facility_counts <- sapply(bin_level_order, function(bl) {
  n_fac <- bin_n$n_facilities[bin_n$occ_bin_treated == bl]
  if (length(n_fac) == 0) n_fac <- 0
  format(n_fac, big.mark = ",")
})

occ_body <- character(0)
occ_body <- c(occ_body, paste0(
  "Facilities (treated) & ", paste(bin_facility_counts, collapse = " & "),
  " & ", format(nrow(baseline_treated), big.mark = ","), " \\\\"
))
occ_body <- c(occ_body, "\\addlinespace[0.4em]")

for (i in seq_len(nrow(occ_outcomes))) {
  v <- occ_outcomes$var[i]
  lab <- occ_outcomes$label[i]
  bin_cells <- sapply(bin_level_order, function(bl) fmt_bin_cell(get_bin_effect(occ_fits[[v]]$binned, bl)))
  pooled_cell <- fmt_bin_cell(coef_se_star(occ_fits[[v]]$pooled, "post"))
  occ_body <- c(occ_body, paste0(lab, " & ", paste(bin_cells, collapse = " & "), " & ", pooled_cell, " \\\\"))
  if (i < nrow(occ_outcomes)) occ_body <- c(occ_body, "\\addlinespace[0.4em]")
}

occ_col_headers <- paste(bin_display_labels[bin_level_order], collapse = " & ")

occ_tex <- wrap_table(
  occ_body,
  caption = "Effect of Ownership Change on Business-Model Outcomes by Baseline Occupancy Bin",
  label = "tab:het-occupancy-bin",
  colspec = "@{} l Y Y Y Y Y Y @{}",
  header_row = paste0("Outcome & ", occ_col_headers, " & Pooled \\\\"),
  notes = c(
    spec_note,
    paste0(
      "\\item Sample includes ALL facilities (treated and never-treated). Treated ",
      "facilities are classified by baseline occupancy rate, averaged over ",
      "$\\tau \\in [-12,-4]$. Never-treated facilities ($N = ", format(n_nevertreated_total, big.mark = ","),
      "$) are assigned a placeholder bin ($<$70\\%), which is mathematically inert ",
      "since \\textit{post} $= 0$ for them always -- they enter the sample only to ",
      "help identify the calendar fixed effects."
    ),
    paste0(
      "\\item The reference bin ($<$70\\%) reports its own total effect (the bare ",
      "\\textit{post} coefficient). Every other bin reports its raw difference from ",
      "the reference bin (the \\textit{post} $\\times$ bin interaction), which supports ",
      "formal cross-bin comparison. \"Pooled\" is the non-binned \\textit{post} effect ",
      "on the same sample, for direct comparison to the bin-specific estimates."
    ),
    sig_note
  )
)

write_fragment(occ_tex, "post_heterogeneity_occupancy_bin_table.tex")

rm(df_occ); gc(verbose = FALSE)

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
  "\\clearpage",
  occ_tex,
  "\\end{document}"
)

write_fragment(preview, "post_heterogeneity_preview.tex")

cat("\nDone. Three heterogeneity tables written.\n")
