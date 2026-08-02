# =============================================================================
# regressions/post_tables.R
#
# The paper's three main post-only TWFE tables, from one script.
#
# Replaces:
#   twfe_post.R                    staffing HPRD + logs
#   post_staffing.R                staffing HPRD + raw hours (4x4 grid)
#   quarterly_quality_twfe_tables.R  quality, +/- staffing controls
#   twfe_interactions.R            chain/pandemic interactions -- DROPPED, see below
#   composition_checks.R           monthly mechanism block (business-model table);
#                                   short-stay quality block folded in below
#
#   twfe_post_non_nurse.R         non-nurse (therapy) staffing HPRD
#
# Quality table (Table 3) now covers long-stay AND short-stay measures in one
# table, labeled by panel (per Joe -- specification isn't the priority right
# now; get all quality results in one place, labeled clearly). Vaccination
# measures (qm_430, qm_472) remain excluded per the earlier CM/Bowblis
# decision -- that decision is untouched by this change.
#
# NOTE: an earlier version of this script derived a resident-days row
# (total_hours / total_hprd) to make the HPRD/hours decomposition identity
# estimable on the page. Dropped per Joe -- not something advisors have asked
# for, and staffing_panel.csv has no native resident-days column, so it would
# have been a derived quantity rather than a directly measured one.
#
# Partially replaces composition_checks.R (its business-model block; short-stay
# quality is now folded directly into Table 3 above, so composition_checks.R's
# short-stay block is superseded too -- see conversation history for the
# raw_hours_and_quality_checks.R overlap this also touches).
#
# -----------------------------------------------------------------------------
# SPECIFICATION -- Spec A, per CM's email
# -----------------------------------------------------------------------------
#   outcome ~ post + beds | facility + calendar period
#
# Spec A is defined in _setup.R as post + beds + chain_at_start. chain_at_start
# is time-invariant by construction, so facility fixed effects absorb it and
# fixest drops it on every fit. Harmless -- the `post` coefficient is identical
# either way -- but it fires on ~40 fits and clutters the console, so it is
# excluded explicitly here (same approach as nested_control_spec_all_outcomes.R).
# This is local to this script, not a change to controls_A() in _setup.R.
#
# Standard errors are two-way clustered by facility and calendar period.
# Monthly outcomes drop the anticipation window (event_time in -3,-2,-1);
# quarterly outcomes drop the transition quarter (event_time == 0).
#
# The nested B/C/D tiers are NOT reproduced here. They live in
# nested_control_spec_all_outcomes.R as the decomposition exercise CM
# described, and duplicating them would create two sources for the same number.
#
# -----------------------------------------------------------------------------
# INTERACTIONS DROPPED
# -----------------------------------------------------------------------------
# twfe_interactions.R produced twfe_post_chain_diff.tex and
# twfe_post_pandemic_diff.tex, both currently \input{} in ma_thesis.tex. Not
# carried forward:
#   - CM asked for pre/post-COVID and Independent-acquired-by-Chain as SAMPLE
#     SPLITS, not interactions.
#   - The chain interaction used the fixed January-2017 baseline, which the
#     pre-event classification supersedes.
#   - The script read the deleted panel.csv and could not run.
#
# CONSEQUENCE TO VERIFY: dropping twfe_post_pandemic_diff.tex removes one half
# of the Table 6 / Table 10 pandemic directional contradiction. The other half
# (the pre/post split in twfe_post.R) is not reproduced here either, since the
# split is a heterogeneity cut rather than a main table. Before telling CM the
# contradiction is resolved, re-run the split on the CURRENT panel and confirm
# the surviving estimate is stable -- the contradiction most likely came from
# the two tables sitting on different panels, but that should be demonstrated
# rather than assumed.
#
# Both \input{} lines must be removed from ma_thesis.tex or it will render
# stale tables from the old panel.
#
# -----------------------------------------------------------------------------
# Output:
#   outputs/tables/post_staffing_table.tex        (label tab:post-staffing)
#   outputs/tables/post_business_model_table.tex  (label tab:post-business)
#   outputs/tables/post_quality_table.tex         (label tab:post-quality)
#   outputs/tables/post_non_nurse_table.tex       (label tab:post-non-nurse)
#   outputs/tables/post_tables_preview.tex        (standalone preview doc)
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

# Case mix as an OUTCOME is still undecided (flagged in the outline). Included
# here with its measurement-break trim applied. Set to FALSE to drop the row.
INCLUDE_CASE_MIX <- TRUE
CASE_MIX_MIN <- "2017/10"
CASE_MIX_MAX <- "2023/12"

# -----------------------------------------------------------------------------
# Estimation + formatting helpers
# -----------------------------------------------------------------------------
fit_post <- function(dat, lhs, vc, fe_rhs, extra_controls = character(0)) {
  rhs <- make_spec_rhs(
    dat,
    spec = SPEC,
    exclude = union(ALWAYS_EXCLUDE, lhs)
  )
  if (length(extra_controls) > 0) {
    extra_controls <- setdiff(intersect_existing(extra_controls, dat), lhs)
    if (length(extra_controls) > 0) {
      rhs <- paste(rhs, paste(extra_controls, collapse = " + "), sep = " + ")
    }
  }
  feols(
    as.formula(paste0(lhs, " ~ ", rhs, " | ", fe_rhs)),
    data = dat, vcov = vc, lean = TRUE
  )
}

safe_fit <- function(dat, lhs, vc, fe_rhs, extra_controls = character(0), label = lhs) {
  if (!(lhs %in% names(dat))) {
    message(sprintf("[skip] %s not present in panel", lhs))
    return(NULL)
  }
  cat(sprintf("[fit] %s (N = %s)\n", label, format(nrow(dat), big.mark = ",")))
  tryCatch(
    fit_post(dat, lhs, vc, fe_rhs, extra_controls),
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

# Coefficient over standard error, stacked in one cell. digits is fixed at 4
# throughout this script (staffing HPRD, hours, business-model shares, and
# quality percentages all report at the same precision) rather than varying
# by outcome, so columns are visually comparable.
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
  "calendar-period fixed effects and control for the number of certified beds. ",
  "Standard errors are two-way clustered by facility and calendar period. The ",
  "sample excludes facilities government-owned at any point during the study period."
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
# Monthly panel
# =============================================================================
keep_monthly <- c(
  "cms_certification_number", "year_month", "event_time", "post", "treated",
  "beds", "chain_at_start",
  "rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd",
  "rn_hours_month", "lpn_hours_month", "cna_hours_month", "total_hours",
  "ln_rn", "ln_lpn", "ln_cna", "ln_total",
  "ln_rn_hours", "ln_lpn_hours", "ln_cna_hours", "ln_total_hours",
  "occupancy_rate", "spare_capacity", "pct_medicare", "pct_medicaid",
  "avg_los_total", "case_mix_total"
)

df_m_full <- load_staffing_panel()
df_m <- df_m_full %>% dplyr::select(dplyr::any_of(keep_monthly))
rm(df_m_full); gc(verbose = FALSE)

df_m_wo <- drop_anticipation_window(df_m)
rm(df_m); gc(verbose = FALSE)

vc_month <- ~ cms_certification_number + year_month
fe_month <- "cms_certification_number + year_month"

# =============================================================================
# TABLE 1: Staffing (4x4 grid)
# =============================================================================
staff_cols <- c("rn", "lpn", "cna", "total")
staff_labels <- c("RN", "LPN", "CNA", "Total")

# All four row-types report at 4 digits for consistency.
staffing_rows <- list(
  list(label = "HPRD",       vars = c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd"),                       digits = 4),
  list(label = "Log(HPRD)",  vars = c("ln_rn", "ln_lpn", "ln_cna", "ln_total"),                               digits = 4),
  list(label = "Hours",      vars = c("rn_hours_month", "lpn_hours_month", "cna_hours_month", "total_hours"),  digits = 4),
  list(label = "Log(hours)", vars = c("ln_rn_hours", "ln_lpn_hours", "ln_cna_hours", "ln_total_hours"),        digits = 4)
)

# NOTE: an observations row (per outcome row) was tried and pulled per Joe --
# not something advisors have asked for yet. Revisit if CM/Bowblis want it.
# The N discrepancy between HPRD and raw hours flagged during that pass is
# still open -- see the console message below.
staffing_body <- character(0)

for (i in seq_along(staffing_rows)) {
  r <- staffing_rows[[i]]
  cells <- character(4)
  for (j in seq_along(r$vars)) {
    v <- r$vars[j]
    mod <- safe_fit(df_m_wo, v, vc_month, fe_month, label = paste(r$label, v))
    cells[j] <- fmt_est(mod, digits = r$digits)
    rm(mod); gc(verbose = FALSE)
  }
  staffing_body <- c(staffing_body, paste0(paste(c(r$label, cells), collapse = " & "), " \\\\"))
  if (i < length(staffing_rows)) {
    staffing_body <- c(staffing_body, "\\addlinespace[0.4em]")
  }
}

# Diagnostic only, not printed in the table: HPRD and raw hours are built
# from the same source row (HPRD = hours / resident-days), so HPRD should
# never have MORE non-missing observations than its own numerator. Surfacing
# this to console so it isn't silently lost now that the table has no N row.
n_hprd  <- sum(!is.na(df_m_wo$total_hprd))
n_hours <- sum(!is.na(df_m_wo$total_hours))
if (n_hprd > n_hours) {
  message(sprintf(
    "[check] total_hprd has %s non-missing obs vs %s for total_hours (diff = %s) -- investigate, see post_tables.R header",
    format(n_hprd, big.mark = ","), format(n_hours, big.mark = ","), format(n_hprd - n_hours, big.mark = ",")
  ))
}

staffing_tex <- wrap_table(
  staffing_body,
  caption = "Effect of Ownership Change on Nursing Staffing",
  label = "tab:post-staffing",
  colspec = "@{} l Y Y Y Y @{}",
  header_row = paste0("Outcome & ", paste(staff_labels, collapse = " & "), " \\\\"),
  notes = c(
    spec_note,
    paste0(
      "\\item HPRD is hours per resident day; hours are the HPRD numerator, ",
      "reported to test whether the HPRD decline is mechanically driven by the ",
      "occupancy-rate increase in the denominator."
    ),
    "\\item The anticipation window ($\\tau = -3, -2, -1$) is excluded.",
    sig_note
  )
)

write_fragment(staffing_tex, "post_staffing_table.tex")

# =============================================================================
# TABLE 2: Business model (single coefficient column)
# =============================================================================
business_spec <- tibble::tribble(
  ~var,              ~label,                       ~digits,
  "occupancy_rate",  "Occupancy rate",             4,
  "spare_capacity",  "Spare capacity",             4,
  "pct_medicare",    "Medicare share",             4,
  "pct_medicaid",    "Medicaid share",             4,
  "avg_los_total",   "Average length of stay",     4
) %>% dplyr::filter(var %in% names(df_m_wo))

business_body <- character(0)
for (i in seq_len(nrow(business_spec))) {
  v <- business_spec$var[i]
  mod <- safe_fit(df_m_wo, v, vc_month, fe_month, label = business_spec$label[i])
  business_body <- c(
    business_body,
    paste0(business_spec$label[i], " & ", fmt_est(mod, business_spec$digits[i]),
           " & ", fmt_n(mod), " \\\\")
  )
  rm(mod); gc(verbose = FALSE)
}

# Case mix carries a measurement-break trim (see nested_control_spec_all_outcomes.R
# for the evidence: CMS methodology breaks at 2017/10 and 2024/01 shift both the
# level and the cross-sectional spread).
case_mix_note <- character(0)
if (INCLUDE_CASE_MIX && "case_mix_total" %in% names(df_m_wo)) {
  df_cm <- df_m_wo %>%
    dplyr::filter(year_month >= CASE_MIX_MIN, year_month <= CASE_MIX_MAX)
  mod_cm <- safe_fit(df_cm, "case_mix_total", vc_month, fe_month, label = "Case mix")
  business_body <- c(
    business_body,
    paste0("Case mix & ", fmt_est(mod_cm, 4), " & ", fmt_n(mod_cm), " \\\\")
  )
  case_mix_note <- paste0(
    "\\item Case mix is restricted to ", CASE_MIX_MIN, "--", CASE_MIX_MAX, ". ",
    "CMS changed its case-mix methodology in October 2017 and January 2024; both ",
    "breaks shift the level and the cross-sectional spread simultaneously across ",
    "the whole sample, which facility and period fixed effects do not absorb."
  )
  rm(df_cm, mod_cm); gc(verbose = FALSE)
}

business_tex <- wrap_table(
  business_body,
  caption = "Effect of Ownership Change on Business-Model Outcomes",
  label = "tab:post-business",
  colspec = "@{} l Y r @{}",
  header_row = "Outcome & Coefficient (SE) & Observations \\\\",
  notes = c(
    spec_note,
    paste0(
      "\\item Occupancy rate is residents as a share of available bed-days; spare ",
      "capacity is unused certified beds as a share of certified beds. Payer shares ",
      "are shares of patient days. The anticipation window ($\\tau = -3, -2, -1$) is excluded."
    ),
    case_mix_note,
    sig_note
  )
)

write_fragment(business_tex, "post_business_model_table.tex")

rm(df_m_wo); gc(verbose = FALSE)

# =============================================================================
# TABLE 3: Quality (with and without staffing controls)
# =============================================================================
keep_quarterly <- c(
  "cms_certification_number", "year", "quarter", "year_quarter",
  "event_time", "post", "treated",
  "beds", "chain_at_start",
  "rn_hprd", "lpn_hprd", "cna_hprd",
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

STAFFING_CONTROLS <- c("rn_hprd", "lpn_hprd", "cna_hprd")

# label_map lets one function serve both the long-stay measure set (whose
# labels live in long_stay_quality_measures) and the short-stay set (whose
# labels live in short_stay_quality_measures), without hardcoding which map
# to use inside the loop.
#
# trim_quality_measure_window() is applied per measure before fitting --
# previously this table fit qm_453 on the full sample with no trim at all,
# missing the 2018-2023 reporting-window restriction recovered from the
# retired quarterly_quality_twfe_tables.R (see _setup.R). qm_471 goes
# through the same explicit call now too, rather than relying on it having
# been trimmed anywhere upstream.
build_quality_block <- function(codes, label_map) {
  rows <- character(0)
  n_with <- integer(0)
  for (v in codes) {
    lab <- unname(label_map[[v]])
    dat_v <- trim_quality_measure_window(df_q_post, v)
    m1 <- safe_fit(dat_v, v, vc_quarter, fe_quarter, label = paste(lab, "(1)"))
    m2 <- safe_fit(dat_v, v, vc_quarter, fe_quarter,
                   extra_controls = STAFFING_CONTROLS, label = paste(lab, "(2)"))
    rows <- c(rows, paste0(
      lab, " & ", fmt_est(m1, 4), " & ", fmt_est(m2, 4), " & ", fmt_n(m1), " \\\\"
    ))
    if (!is.null(m2)) n_with <- c(n_with, nobs(m2))
    rm(dat_v, m1, m2); gc(verbose = FALSE)
  }
  list(rows = rows, n_with = n_with)
}

mech <- build_quality_block(quality_mechanism_measures, long_stay_quality_measures)
outc <- build_quality_block(quality_outcome_measures, long_stay_quality_measures)
# Short-stay measures, per Joe: include alongside long-stay rather than in a
# separate table, labeled by stay-type panel header so there's no ambiguity
# about which population each row describes. Vaccination measures (qm_430,
# qm_472) remain excluded per CM/Bowblis's decision -- unaffected by this change.
short <- build_quality_block(names(short_stay_quality_measures), short_stay_quality_measures)

quality_body <- c(
  panel_header("Panel A: Long-stay labor-saving mechanism measures", 4),
  mech$rows,
  "\\addlinespace[0.6em]",
  panel_header("Panel B: Long-stay resident outcome measures", 4),
  outc$rows,
  "\\addlinespace[0.6em]",
  panel_header("Panel C: Short-stay measures", 4),
  short$rows
)

quality_tex <- wrap_table(
  quality_body,
  caption = "Effect of Ownership Change on Quality Measures",
  label = "tab:post-quality",
  colspec = "@{} l Y Y r @{}",
  header_row = "Outcome & (1) & (2) & Observations \\\\",
  notes = c(
    spec_note,
    paste0(
      "\\item Column (1) is the baseline specification. Column (2) adds RN, LPN, ",
      "and CNA hours per resident day as controls. Because staffing is itself ",
      "affected by ownership change, column (2) is not a preferred estimate of the ",
      "total effect; it is reported to show whether the quality response is ",
      "attenuated after conditioning on measured staffing inputs."
    ),
    paste0(
      "\\item Long-stay measures (Panels A-B) and short-stay measures (Panel C) are ",
      "constructed from different resident populations and are not directly ",
      "comparable to one another. For every measure, lower values indicate better ",
      "measured quality."
    ),
    paste0(
      "\\item Pressure injuries is estimated on 2018--2023 only; improved function ",
      "is estimated on 2017--2022 only. Both measures show near-complete absence ",
      "outside these windows and are trimmed to the years where they are actually ",
      "reported rather than treated as full-panel outcomes. Vaccination measures ",
      "are excluded from this table."
    ),
    paste0(
      "\\item The transition quarter ($\\tau = 0$) is excluded. Observations are ",
      "reported for column (1); column (2) drops facility-quarters with missing staffing."
    ),
    sig_note
  )
)

write_fragment(quality_tex, "post_quality_table.tex")

# Column (2) sample sizes are not printed in the table -- report them to the
# console so any material divergence from column (1) is visible.
cat("\n=== Column (2) observation counts (staffing controls added) ===\n")
print(c(mech$n_with, outc$n_with, short$n_with))

rm(df_q_post); gc(verbose = FALSE)

# =============================================================================
# TABLE 4: Non-nurse (therapy) staffing (single coefficient column)
#
# Superseded from twfe_post_non_nurse.R. Levels only, per that script's own
# rationale (several categories -- PT aide, OT aide -- have a large share of
# exact zeros, making logs and a raw-hours mechanism check less meaningful
# here than for nursing). SPEC CHANGE: the original script used the full
# control set (make_controls_rhs); this uses Spec A, matching every other
# table in this script -- so these numbers will differ from what
# twfe_post_non_nurse.R currently produces, the same shift that already
# happened when robustness_checks.R moved to Spec A.
# =============================================================================
keep_non_nurse <- c(
  "cms_certification_number", "year_month", "event_time", "post", "treated",
  "beds", "chain_at_start",
  "pt_hprd", "ptasst_hprd", "ptaide_hprd",
  "ot_hprd", "otasst_hprd", "otaide_hprd",
  "slp_hprd", "nonnurse_total_hprd"
)

df_nn_full <- load_staffing_panel()
df_nn <- df_nn_full %>% dplyr::select(dplyr::any_of(keep_non_nurse))
rm(df_nn_full); gc(verbose = FALSE)

df_nn_wo <- drop_anticipation_window(df_nn)
rm(df_nn); gc(verbose = FALSE)

non_nurse_spec <- tibble::tribble(
  ~var,                  ~label,
  "pt_hprd",             "Physical Therapist (PT)",
  "ptasst_hprd",         "PT Assistant",
  "ptaide_hprd",         "PT Aide",
  "ot_hprd",             "Occupational Therapist (OT)",
  "otasst_hprd",         "OT Assistant",
  "otaide_hprd",         "OT Aide",
  "slp_hprd",            "Speech-Language Pathologist",
  "nonnurse_total_hprd", "Total Non-Nurse"
) %>% dplyr::filter(var %in% names(df_nn_wo))

non_nurse_body <- character(0)
for (i in seq_len(nrow(non_nurse_spec))) {
  v <- non_nurse_spec$var[i]
  mod <- safe_fit(df_nn_wo, v, vc_month, fe_month, label = non_nurse_spec$label[i])
  non_nurse_body <- c(
    non_nurse_body,
    paste0(non_nurse_spec$label[i], " & ", fmt_est(mod, 4), " & ", fmt_n(mod), " \\\\")
  )
  rm(mod); gc(verbose = FALSE)
}

non_nurse_tex <- wrap_table(
  non_nurse_body,
  caption = "Effect of Ownership Change on Non-Nurse (Therapy) Staffing",
  label = "tab:post-non-nurse",
  colspec = "@{} l Y r @{}",
  header_row = "Staff type & Coefficient (SE) & Observations \\\\",
  notes = c(
    spec_note,
    paste0(
      "\\item Outcomes are hours per resident day (HPRD) for each non-nurse staff ",
      "type. Levels only are reported: several categories (PT Aide, OT Aide) have a ",
      "large share of exact zeros, which makes log specifications and a raw-hours ",
      "decomposition less informative here than for nursing staff."
    ),
    "\\item The anticipation window ($\\tau = -3, -2, -1$) is excluded.",
    sig_note
  )
)

write_fragment(non_nurse_tex, "post_non_nurse_table.tex")

rm(df_nn_wo); gc(verbose = FALSE)

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
  staffing_tex,
  "\\clearpage",
  business_tex,
  "\\clearpage",
  quality_tex,
  "\\clearpage",
  non_nurse_tex,
  "\\end{document}"
)

write_fragment(preview, "post_tables_preview.tex")

cat("\nDone. Four post-only TWFE tables written.\n")
