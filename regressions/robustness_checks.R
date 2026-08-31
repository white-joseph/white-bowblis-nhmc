# =============================================================================
# regressions/robustness_checks.R
#
# Reports whether the estimated effect of ownership change on nursing staffing
# is sensitive to the sample restriction used or to the choice of case-mix
# control. Produces a single summary table of eleven specifications.
#
# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
#   outcome ~ post + beds | facility + calendar month
#
# Spec A as defined in _setup.R, matching the main tables in post_tables.R.
# Standard errors are two-way clustered by facility and calendar month.
#
# Rows (1)-(7) hold the control set fixed and vary the sample: the baseline
# anticipation-window exclusion, four alternative restrictions on gaps in
# facility reporting, and two alternative anticipation-window widths.
#
# Rows (8)-(11) hold the sample fixed at row (1) and add one case-mix control
# set per row, varying the reference group (state or national) and the bin
# granularity (quartiles or deciles). Spec A includes no case-mix control, so
# these rows are not directly comparable to rows (1)-(7) and are labeled
# accordingly.
#
# Outcomes are reported in levels only. This table asks whether the sign and
# significance of the post coefficient survive alternative sample and control
# choices; the corresponding log specifications are reported in the staffing
# table in post_tables.R.
#
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
#   data/clean/staffing_panel.csv   via load_staffing_panel()
#
# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
#   outputs/tables/twfe_robustness_summary_code.tex     (label tab:twfe-robustness-summary)
#   outputs/tables/twfe_robustness_summary_preview.tex  (standalone preview document)
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

outs_order   <- c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd")
outs_labels  <- c("RN", "LPN", "CNA", "Total")

vc_month <- ~ cms_certification_number + year_month
fe_month <- "cms_certification_number + year_month"

# -----------------------------------------------------------------------------
# Estimation and formatting helpers
# -----------------------------------------------------------------------------
fit_one <- function(dat, lhs, extra_controls = character(0)) {
  rhs <- make_spec_rhs(dat, spec = SPEC, exclude = union(ALWAYS_EXCLUDE, lhs))
  if (length(extra_controls) > 0) {
    extra_controls <- setdiff(intersect_existing(extra_controls, dat), lhs)
    if (length(extra_controls) > 0) {
      rhs <- paste(rhs, paste(extra_controls, collapse = " + "), sep = " + ")
    }
  }
  tryCatch(
    feols(as.formula(paste0(lhs, " ~ ", rhs, " | ", fe_month)),
          data = dat, vcov = vc_month, lean = TRUE),
    error = function(e) {
      message(sprintf("[warn] %s failed: %s", lhs, e$message))
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

# Formats a coefficient and its standard error side by side on a single line.
# This table has many rows, so single-line cells are used here rather than the
# stacked two-line cells used in the main tables.
fmt_est <- function(mod, digits = 4) {
  s <- coef_se_star(mod)
  if (is.na(s$coef) || is.na(s$se)) return("--")
  b <- formatC(s$coef, format = "f", digits = digits)
  if (s$coef > 0) b <- paste0("\\phantom{-}", b)
  se <- formatC(s$se, format = "f", digits = digits)
  if (s$stars == "") {
    paste0("$", b, "$ $(", se, ")$")
  } else {
    paste0("$", b, "^{", s$stars, "}$ $(", se, ")$")
  }
}

sig_note <- "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."

# -----------------------------------------------------------------------------
# Load panel
# -----------------------------------------------------------------------------
keep_monthly <- c(
  "cms_certification_number", "year_month", "event_time", "post", "treated",
  "beds", "chain_at_start", "gap_from_prev_months",
  "rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd",
  "cm_q_state_2", "cm_q_state_3", "cm_q_state_4",
  paste0("cm_d_state_", 2:10),
  "cm_q_nat_2", "cm_q_nat_3", "cm_q_nat_4",
  paste0("cm_d_nat_", 2:10)
)

df_full <- load_staffing_panel()
df <- df_full %>% dplyr::select(dplyr::any_of(keep_monthly))
rm(df_full); gc(verbose = FALSE)

# -----------------------------------------------------------------------------
# Rows 1-7: sample-restriction variants, no case-mix controls
#
# Row (1) is the baseline sample used throughout the paper: the anticipation
# window (event_time in -3, -2, -1) excluded, with no restriction on gaps in
# facility reporting.
# -----------------------------------------------------------------------------
df_baseline <- drop_anticipation_window(df)

restriction_specs <- list(
  list(label = "(1) Baseline (no anticipation)", data = df_baseline),
  list(label = "(2) Sample excludes \\textit{gap} $> 6$",
       data = df_baseline %>% dplyr::filter(is.na(gap_from_prev_months) | gap_from_prev_months <= 6)),
  list(label = "(3) Sample excludes \\textit{gap} $> 3$",
       data = df_baseline %>% dplyr::filter(is.na(gap_from_prev_months) | gap_from_prev_months <= 3)),
  list(label = "(4) Sample excludes \\textit{gap} $> 1$",
       data = df_baseline %>% dplyr::filter(is.na(gap_from_prev_months) | gap_from_prev_months <= 1)),
  list(label = "(5) Sample excludes \\textit{gap} $> 0$",
       data = df_baseline %>% dplyr::filter(is.na(gap_from_prev_months) | gap_from_prev_months == 0)),
  list(label = "(6) Drop $t \\in \\{-4,-3,-2,-1\\}$",
       data = df %>% dplyr::filter(is.na(event_time) | !(event_time %in% -4:-1))),
  list(label = "(7) Drop $t \\in \\{-2,-1\\}$ only",
       data = df %>% dplyr::filter(is.na(event_time) | !(event_time %in% c(-2, -1))))
)

# -----------------------------------------------------------------------------
# Rows 8-11: case-mix control granularity, sample held at the row (1) baseline
#
# Spec A includes no case-mix control, so each row here is Spec A with one
# case-mix control set added, and is distinct from row (1) rather than a
# repeat of it.
# -----------------------------------------------------------------------------
case_mix_variants <- list(
  "State Quartile" = c("cm_q_state_2", "cm_q_state_3", "cm_q_state_4"),
  "State Decile"   = paste0("cm_d_state_", 2:10),
  "National Quartile" = c("cm_q_nat_2", "cm_q_nat_3", "cm_q_nat_4"),
  "National Decile"   = paste0("cm_d_nat_", 2:10)
)

for (nm in names(case_mix_variants)) {
  present <- intersect(case_mix_variants[[nm]], names(df_baseline))
  missing <- setdiff(case_mix_variants[[nm]], names(df_baseline))
  cat(sprintf("[check] %-20s present=%d, missing=%s\n", nm, length(present),
              if (length(missing) == 0) "none" else paste(missing, collapse = ", ")))
  case_mix_variants[[nm]] <- present
}

case_mix_specs <- lapply(names(case_mix_variants), function(nm) {
  list(
    label = paste0("(", 8 + match(nm, names(case_mix_variants)) - 1, ") + Case-mix: ", nm),
    data = df_baseline,
    extra_controls = case_mix_variants[[nm]]
  )
})

all_specs <- c(restriction_specs, case_mix_specs)

# -----------------------------------------------------------------------------
# Estimation
# -----------------------------------------------------------------------------
fit_spec_row <- function(spec) {
  extra <- if (!is.null(spec$extra_controls)) spec$extra_controls else character(0)
  level_mods <- lapply(outs_order, function(y) fit_one(spec$data, y, extra_controls = extra))
  list(level = level_mods, n = nrow(spec$data))
}

cat("\n=== Fitting", length(all_specs), "robustness rows x", length(outs_order), "outcomes (levels only) ===\n")
all_fits <- lapply(all_specs, function(spec) {
  cat("[row]", spec$label, "\n")
  fit_spec_row(spec)
})

# -----------------------------------------------------------------------------
# Table construction
# -----------------------------------------------------------------------------
build_row_cells <- function(mods, digits = 4) {
  paste(sapply(mods, fmt_est, digits = digits), collapse = " & ")
}

panel_rows <- function() {
  rows <- character(0)
  for (i in seq_along(all_specs)) {
    spec <- all_specs[[i]]
    fit  <- all_fits[[i]]
    n_fmt <- format(fit$n, big.mark = ",")
    cells <- build_row_cells(fit$level)
    rows <- c(rows, paste0(spec$label, " & ", n_fmt, " & ", cells, " \\\\"))
    if (i < length(all_specs)) rows <- c(rows, "\\addlinespace[0.35em]")
    if (i == length(restriction_specs)) rows <- c(rows, "\\addlinespace[3pt]")
  }
  rows
}

body <- panel_rows()

rob_frag <- c(
  "\\begingroup",
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{TWFE estimates of \\textit{post} on staffing levels (HPRD): sample-restriction and case-mix robustness.}",
  "\\label{tab:twfe-robustness-summary}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "",
  "\\begin{tabularx}{\\textwidth}{@{} l c YYYY @{} }",
  "\\toprule",
  " &  & \\multicolumn{4}{c}{\\textbf{Outcomes}} \\\\",
  "\\cmidrule(lr){3-6}",
  paste0(" & \\textbf{N} & ", paste0("\\textbf{", outs_labels, "}", collapse = " & "), " \\\\"),
  "\\midrule",
  body,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the \\textit{post} coefficient on staffing levels (HPRD), with two-way clustered standard errors (facility and calendar month) in parentheses.",
  paste0(
    "\\item Rows (1)--(7) hold the control set fixed at Spec A (post + number of ",
    "certified beds, plus facility and calendar-month fixed effects) and vary the ",
    "sample restriction or anticipation-window definition, as described in each row ",
    "label. Row (1) is the project's standard sample: the anticipation window ",
    "($\\tau = -3, -2, -1$) excluded, no additional gap restriction."
  ),
  paste0(
    "\\item Rows (8)--(11) hold the sample fixed at row (1) and add one case-mix ",
    "control set on top of Spec A per row -- Spec A itself includes no case-mix ",
    "control, so these rows are not directly comparable to rows (1)--(7). ",
    "``State Quartile'' is the default case-mix control used elsewhere in the ",
    "paper. Quartile bins ",
    "use dummies for bins 2--4; decile bins use dummies for bins 2--10 (bin 1 omitted ",
    "as reference in both cases)."
  ),
  sig_note,
  "\\end{tablenotes}",
  "",
  "\\end{threeparttable}",
  "\\end{table}",
  "\\endgroup",
  ""
)

frag_path <- file.path(out_dir, "twfe_robustness_summary_code.tex")
writeLines(rob_frag, frag_path, useBytes = TRUE)
cat("[write] ", normalizePath(frag_path, winslash = "\\"), "\n", sep = "")

preview <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{makecell}",
  "\\usepackage{array}",
  "\\usepackage{caption}",
  "\\captionsetup{labelfont=bf, font=small}",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "\\begin{document}",
  rob_frag,
  "\\end{document}"
)

preview_path <- file.path(out_dir, "twfe_robustness_summary_preview.tex")
writeLines(preview, preview_path, useBytes = TRUE)
cat("[write] ", normalizePath(preview_path, winslash = "\\"), "\n", sep = "")

cat("\nDone. Robustness summary table written (", length(all_specs), " rows).\n", sep = "")
