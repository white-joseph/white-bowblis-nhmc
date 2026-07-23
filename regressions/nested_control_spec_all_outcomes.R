# =============================================================================
# regressions/nested_control_spec_all_outcomes.R
#
# Purpose:
#   Implements C. Moul's nested control-spec framework (email, per his
#   conversation with Bowblis) across every outcome category in the paper,
#   using the project's STANDARD design (post-indicator TWFE, facility +
#   calendar-period fixed effects, donut window, two-way clustering) -- NOT
#   the Callaway-Sant'Anna estimator in callaway_santanna_event_study.R,
#   which is a separate exploratory check, not the main design.
#
#   TABLE FORMAT (per Joe, matches the rest of the repo): one table PER SPEC
#   LETTER, not one table per outcome category. Each table has a single
#   coefficient-on-`post` column and stacks every outcome category to which
#   that spec letter applies, grouped by panel-header rows. Tables are large
#   by design -- that's intentional, not a bug.
#     Table 1 (Spec A): Case mix, Non-profit status, Strategic/Business
#       Model, Staffing, Quality  -- every category (A applies to all of them)
#     Table 2 (Spec B): Strategic/Business Model, Staffing, Quality
#       (NOT Case mix/Non-profit -- B's added controls ARE those outcomes)
#     Table 3 (Spec C): Staffing, Quality
#       (NOT Strategic/Business Model -- C's added controls ARE those outcomes)
#     Table 4 (Spec D): Quality only
#       (D's added control set IS staffing)
#
#   Two decisions confirmed in CM's email are applied exactly as specified:
#     1. Every facility EVER government-owned is excluded (Bowblis agrees;
#        results are slightly stronger without them). load_staffing_panel()
#        already does this for the monthly panel; this script applies the
#        same rule to the quarterly quality panel (see
#        load_quality_panel_for_specs() below), which composition_checks.R
#        and fe_only_specification_all_outcomes.R did NOT previously do --
#        flag this if quality numbers here look different from anything
#        previously circulated.
#     2. Chain is never used as a control in its raw (time-varying) form --
#        only chain_at_start. Built for the quality panel the same way
#        _setup.R builds it for the monthly panel (baseline period, with
#        fallback to each facility's earliest observed chain value).
#
#   CASE-MIX MEASUREMENT BREAK (per Joe's question -- checked directly
#   against the panel, not assumed):
#     Monthly mean/sd/min/max of case_mix_total, staffing_panel.csv:
#       2017/09: mean 4.180, sd 0.374, range [2.40, 6.90]
#       2017/10: mean 3.232, sd 0.290, range [2.13, 5.53]   <- sharp break down
#       2023/12: mean 3.156, sd 0.296, range [2.10, 5.67]
#       2024/01: mean 3.776, sd 0.553, range [2.06, 10.37]  <- sharp break up,
#         AND the spread roughly doubles, AND the ceiling nearly doubles from
#         a value that had been stable at ~5.5-6.9 for the entire panel.
#     Between those two dates the series just drifts normally (no comparable
#     jump anywhere in between). Both breaks show up in the SAME calendar
#     month for the whole sample simultaneously, which is the signature of a
#     CMS measurement/methodology change, not a real shift in resident acuity.
#     Facility and month fixed effects absorb a pure common level shift, but
#     they do NOT absorb the accompanying change in cross-sectional spread,
#     and any facility whose ownership-change event window straddles one of
#     these two dates would have a pre/post comparison contaminated by the
#     measurement change rather than by the treatment. case_mix_total as an
#     OUTCOME is therefore restricted to 2017/10 through 2023/12 (see
#     CASE_MIX_STABLE_MIN/MAX below).
#     NOTE: this only concerns case_mix_total as an OUTCOME. Whether the
#     case-mix quartile dummies (cm_q_state_2/3/4) used as a CONTROL
#     elsewhere are affected depends on whether those quartiles are assigned
#     within-period or off a single pooled ranking -- worth checking
#     separately, not addressed here.
#
#   KNOWN GAP (carried over): quality_panel.csv has no avg_los_total column.
#   Spec C/D controls for quality outcomes silently omit average length of
#   stay via _setup.R's intersect_existing() tolerance until a quarterly
#   avg_los_total is merged in from the monthly panel.
#
#   Donut: monthly outcomes use drop_anticipation_window() (event_time in
#   {-3,-2,-1} dropped); quarterly outcomes drop the transition quarter
#   (event_time == 0) -- both match the convention already used throughout
#   the project.
#
# Output:
#   outputs/tables/nested_control_spec_tableA.tex
#   outputs/tables/nested_control_spec_tableB.tex
#   outputs/tables/nested_control_spec_tableC.tex
#   outputs/tables/nested_control_spec_tableD.tex
#   outputs/tables/nested_control_spec_all_outcomes.tex  (combined preview doc)
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

# -----------------------------------------------------------------------------
# Shared formatting helpers (same convention as fe_only_specification_all_outcomes.R)
# -----------------------------------------------------------------------------
coef_se_star <- function(mod, term = "post") {
  if (is.null(mod)) return(list(coef = NA, se = NA, stars = ""))
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

fmt_n <- function(mod) if (is.null(mod)) "--" else format(nobs(mod), big.mark = ",")

# -----------------------------------------------------------------------------
# Spec-tier fitting: thin wrapper around _setup.R's make_spec_rhs(), which
# already builds "post + <tier's controls>" for a given spec letter.
#
# chain_at_start is a per-facility BASELINE value -- it is constant within a
# facility for the entire panel by construction. Every spec here already
# includes facility fixed effects, so chain_at_start is perfectly collinear
# with those FE and fixest drops it automatically on every single fit
# (harmless -- the `post` coefficient is identical either way -- but it fires
# ~50+ times across this script and clutters the console, and repeated
# collinearity handling on a wide data frame is part of what was making runs
# slow/unstable). Excluded explicitly here instead of relying on the
# auto-drop. It still does real work in callaway_santanna_event_study.R,
# which has no facility FE to absorb it -- this exclusion is local to this
# script only, not a change to controls_A/B/C/D in _setup.R.
ALWAYS_EXCLUDE <- "chain_at_start"

fit_spec <- function(dat, lhs, spec, vc, fe_rhs, exclude = character(0)) {
  rhs <- make_spec_rhs(dat, spec = spec, exclude = union(exclude, union(ALWAYS_EXCLUDE, lhs)))
  feols(as.formula(paste0(lhs, " ~ ", rhs, " | ", fe_rhs)), data = dat, vcov = vc, lean = TRUE)
}

# Runs ONE spec letter across a tibble of outcomes, one row per outcome.
# Optional per-outcome sample trims: year_min/year_max (integer year, for the
# quarterly quality panel) or ym_min/ym_max (character "YYYY/MM", for the
# monthly staffing panel -- used for the case-mix measurement-break trim).
run_outcomes_single_spec <- function(dat, outcomes_tbl, spec, vc, fe_rhs) {
  if (!nrow(outcomes_tbl)) return(character(0))
  rows <- character(nrow(outcomes_tbl))
  for (i in seq_len(nrow(outcomes_tbl))) {
    v   <- outcomes_tbl$var[i]
    lab <- outcomes_tbl$label[i]

    dat_i <- dat
    if ("year_min" %in% names(outcomes_tbl) && !is.na(outcomes_tbl$year_min[i])) {
      dat_i <- dat_i %>% dplyr::filter(year >= outcomes_tbl$year_min[i])
    }
    if ("year_max" %in% names(outcomes_tbl) && !is.na(outcomes_tbl$year_max[i])) {
      dat_i <- dat_i %>% dplyr::filter(year <= outcomes_tbl$year_max[i])
    }
    if ("ym_min" %in% names(outcomes_tbl) && !is.na(outcomes_tbl$ym_min[i])) {
      dat_i <- dat_i %>% dplyr::filter(year_month >= outcomes_tbl$ym_min[i])
    }
    if ("ym_max" %in% names(outcomes_tbl) && !is.na(outcomes_tbl$ym_max[i])) {
      dat_i <- dat_i %>% dplyr::filter(year_month <= outcomes_tbl$ym_max[i])
    }

    cat(sprintf("[fit] %s: spec %s (N before fit = %s)\n", lab, spec, format(nrow(dat_i), big.mark = ",")))
    mod <- tryCatch(
      fit_spec(dat_i, v, spec, vc, fe_rhs),
      error = function(e) {
        message(sprintf("[warn] %s / spec %s failed: %s", lab, spec, e$message))
        NULL
      }
    )
    rows[i] <- paste(c(lab, fmt_est(mod), fmt_n(mod)), collapse = " & ")
    rm(mod, dat_i); gc(verbose = FALSE)
  }
  rows
}

# -----------------------------------------------------------------------------
# Table assembly: one table per spec letter, category panel-header rows,
# single "Coefficient (SE)" column.
# -----------------------------------------------------------------------------
panel_header <- function(label) {
  sprintf("\\multicolumn{3}{@{}l}{\\textbf{%s}} \\\\[2pt]", label)
}

assemble_spec_table <- function(blocks, caption, label, notes) {
  body <- character(0)
  first <- TRUE
  for (cat_label in names(blocks)) {
    rows <- blocks[[cat_label]]
    if (!length(rows)) next
    if (!first) body <- c(body, "\\addlinespace[4pt]")
    body <- c(body, panel_header(cat_label), paste0(rows, " \\\\"))
    first <- FALSE
  }
  c(
    "\\begin{table}[!ht]",
    "\\centering",
    "\\begin{threeparttable}",
    paste0("\\caption{", caption, "}"),
    paste0("\\label{", label, "}"),
    "\\footnotesize",
    "\\setlength{\\tabcolsep}{6pt}",
    "\\begin{tabularx}{\\textwidth}{@{} l Y r @{}}",
    "\\toprule",
    "Outcome & Coefficient (SE) & Observations \\\\",
    "\\midrule",
    body,
    "\\bottomrule",
    "\\end{tabularx}",
    "",
    "\\begin{tablenotes}[flushleft]",
    "\\footnotesize",
    notes,
    "\\end{tablenotes}",
    "\\end{threeparttable}",
    "\\end{table}",
    ""
  )
}

# Note style matched to the rest of the repo (twfe_post.R, composition_checks.R):
# a short "Notes:" line, a covariates/FE line, a significance line -- not the
# longer explanatory asides that belong in code comments, not table footnotes.
common_notes <- c(
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses. All specifications include facility and calendar-period fixed effects; sample excludes facilities ever government-owned.",
  "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."
)

# =============================================================================
# PART 1: Monthly staffing panel -- Case mix, Non-profit status, Strategic/
#         Business Model, and Staffing outcomes
# =============================================================================
# load_staffing_panel() returns the FULL panel (~100 columns, including a lot
# of cm_d_nat_*/cm_d_state_* dummy variables this script never touches) for
# ~630K rows. Keeping all of that in memory across ~70+ sequential feols()
# fits (this script refits staffing/strategic/quality at multiple spec
# tiers) is almost certainly what was crashing R. Slimmed to only the
# columns actually used below, immediately after loading, then the wide
# copy is dropped and gc()'d before any models are fit.
keep_cols_monthly <- c(
  "cms_certification_number", "year_month", "event_time", "post",
  "beds", "chain_at_start",
  "case_mix_total", "non_profit",
  "occupancy_rate", "pct_medicare", "pct_medicaid", "avg_los_total", "spare_capacity",
  "rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd",
  "cm_q_state_2", "cm_q_state_3", "cm_q_state_4"
)

df_full <- load_staffing_panel()
df <- df_full %>% dplyr::select(any_of(keep_cols_monthly))
rm(df_full); gc()

df_wo <- drop_anticipation_window(df)
rm(df); gc()

vc_month <- ~ cms_certification_number + year_month
fe_month <- "cms_certification_number + year_month"

# Case-mix stable window (see header note above for the empirical evidence).
CASE_MIX_STABLE_MIN <- "2017/10"
CASE_MIX_STABLE_MAX <- "2023/12"

case_mix_tbl <- tibble::tribble(
  ~var,             ~label,             ~ym_min,               ~ym_max,
  "case_mix_total", "Case mix (total)", CASE_MIX_STABLE_MIN,  CASE_MIX_STABLE_MAX
) %>% dplyr::filter(var %in% names(df_wo))

non_profit_tbl <- tibble::tribble(
  ~var,         ~label,
  "non_profit", "Non-profit status (0/1)"
) %>% dplyr::filter(var %in% names(df_wo))

strategic_tbl <- tibble::tribble(
  ~var,              ~label,
  "occupancy_rate",  "Occupancy rate",
  "pct_medicare",    "Medicare share",
  "pct_medicaid",    "Medicaid share",
  "avg_los_total",   "Average length of stay",
  "spare_capacity",  "Spare capacity"
) %>% dplyr::filter(var %in% names(df_wo))

staffing_tbl <- tibble::tribble(
  ~var,          ~label,
  "rn_hprd",     "RN HPRD",
  "lpn_hprd",    "LPN HPRD",
  "cna_hprd",    "CNA HPRD",
  "total_hprd",  "Total HPRD"
) %>% dplyr::filter(var %in% names(df_wo))

# =============================================================================
# PART 2: Quarterly quality panel
# =============================================================================
load_quality_panel_for_specs <- function() {
  fp <- file.path(project_root, "data", "clean", "quality_panel.csv")
  qdf <- readr::read_csv(fp, show_col_types = FALSE) %>%
    dplyr::mutate(
      cms_certification_number = as.character(cms_certification_number),
      quarter = as.character(quarter),
      year = suppressWarnings(as.integer(year)),
      year_quarter = paste0(year, quarter)
    )

  if ("government" %in% names(qdf)) {
    ever_gov <- qdf %>%
      dplyr::filter(government == 1) %>%
      dplyr::distinct(cms_certification_number) %>%
      dplyr::pull(cms_certification_number)
    n0 <- dplyr::n_distinct(qdf$cms_certification_number)
    qdf <- qdf %>% dplyr::filter(!(cms_certification_number %in% ever_gov))
    n1 <- dplyr::n_distinct(qdf$cms_certification_number)
    message(sprintf(
      "[quality] dropped %d facilities ever government-owned (%d -> %d facilities)",
      length(ever_gov), n0, n1
    ))
  }

  if ("chain" %in% names(qdf)) {
    lookup <- qdf %>%
      dplyr::arrange(cms_certification_number, year, quarter) %>%
      dplyr::group_by(cms_certification_number) %>%
      dplyr::summarise(
        chain_2017q1   = chain[year == 2017 & quarter == "Q1"][1],
        chain_earliest = chain[!is.na(chain)][1],
        .groups = "drop"
      ) %>%
      dplyr::mutate(chain_at_start = dplyr::coalesce(chain_2017q1, chain_earliest)) %>%
      dplyr::select(cms_certification_number, chain_at_start)

    n_fallback <- sum(is.na(lookup$chain_2017q1) & !is.na(lookup$chain_at_start))
    qdf <- qdf %>% dplyr::left_join(lookup, by = "cms_certification_number")
    message(sprintf("[quality] chain_at_start: %d facilities used fallback (earliest observed chain)", n_fallback))
  }

  if (!("avg_los_total" %in% names(qdf))) {
    warning(
      "[quality] avg_los_total is NOT present in quality_panel.csv. Spec C/D ",
      "controls for quality outcomes will silently omit average length of ",
      "stay until this is merged in from the monthly staffing panel.",
      call. = FALSE
    )
  }

  # Same memory discipline as the monthly panel: drop the wide set of
  # cm_d_nat_*/cm_d_state_* dummy columns this script never uses before
  # handing the data frame back.
  keep_cols_quality <- c(
    "cms_certification_number", "year", "quarter", "year_quarter", "event_time", "post",
    "beds", "chain_at_start", "non_profit",
    "occupancy_rate", "pct_medicare", "pct_medicaid",
    "rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd",
    "cm_q_state_2", "cm_q_state_3", "cm_q_state_4",
    "qm_401", "qm_404", "qm_406", "qm_407", "qm_410", "qm_419", "qm_452", "qm_453",
    "qm_430", "qm_434", "qm_471", "qm_472"
  )
  qdf %>% dplyr::select(any_of(keep_cols_quality))
}

df_quality <- load_quality_panel_for_specs()
df_quality_post <- df_quality %>% dplyr::filter(is.na(event_time) | event_time != 0)
rm(df_quality); gc()

vc_quarter <- ~ cms_certification_number + year_quarter
fe_quarter <- "cms_certification_number + year_quarter"

short_stay_tbl <- tibble::tribble(
  ~var,     ~label,                          ~year_min,      ~year_max,
  "qm_430", "Pneumococcal vaccine",          NA_integer_,    NA_integer_,
  "qm_434", "New antipsychotic medication",  NA_integer_,    NA_integer_,
  "qm_471", "Improved function",             NA_integer_,    2022L,
  "qm_472", "Influenza vaccine",             2018L,          2023L
) %>% dplyr::filter(var %in% names(df_quality_post))

long_stay_tbl <- tibble::tribble(
  ~var,     ~label,
  "qm_401", "Decline in physical functioning",
  "qm_404", "Weight loss",
  "qm_406", "Catheter use",
  "qm_407", "Urinary tract infections",
  "qm_410", "Falls with major injury",
  "qm_419", "Anti-psychotic medication use",
  "qm_452", "Anti-anxiety/hypnotic medication use",
  "qm_453", "Pressure injuries"
) %>% dplyr::filter(var %in% names(df_quality_post))

# =============================================================================
# Fit + assemble: one table per spec letter
# =============================================================================

# ---- Table 1: Spec A (all categories) ----
tab1_blocks <- list(
  "Case Mix"                   = run_outcomes_single_spec(df_wo, case_mix_tbl,   "A", vc_month, fe_month),
  "Non-Profit Status"          = run_outcomes_single_spec(df_wo, non_profit_tbl, "A", vc_month, fe_month),
  "Strategic / Business Model" = run_outcomes_single_spec(df_wo, strategic_tbl,  "A", vc_month, fe_month),
  "Staffing"                   = run_outcomes_single_spec(df_wo, staffing_tbl,   "A", vc_month, fe_month),
  "Quality (Short-Stay)"       = run_outcomes_single_spec(df_quality_post, short_stay_tbl, "A", vc_quarter, fe_quarter),
  "Quality (Long-Stay)"        = run_outcomes_single_spec(df_quality_post, long_stay_tbl,  "A", vc_quarter, fe_quarter)
)

tab1_tex <- assemble_spec_table(
  tab1_blocks,
  caption = "Spec A (Treatment + FE + Beds + Chain-at-Start): All Outcomes",
  label = "tab:nested-spec-A",
  notes = c(
    common_notes,
    "\\item Transition quarter excluded from quality specifications."
  )
)

# ---- Table 2: Spec B (Strategic, Staffing, Quality) ----
tab2_blocks <- list(
  "Strategic / Business Model" = run_outcomes_single_spec(df_wo, strategic_tbl, "B", vc_month, fe_month),
  "Staffing"                   = run_outcomes_single_spec(df_wo, staffing_tbl,  "B", vc_month, fe_month),
  "Quality (Short-Stay)"       = run_outcomes_single_spec(df_quality_post, short_stay_tbl, "B", vc_quarter, fe_quarter),
  "Quality (Long-Stay)"        = run_outcomes_single_spec(df_quality_post, long_stay_tbl,  "B", vc_quarter, fe_quarter)
)

tab2_tex <- assemble_spec_table(
  tab2_blocks,
  caption = "Spec B (A + Case-mix + Non-profit): Strategic, Staffing, and Quality Outcomes",
  label = "tab:nested-spec-B",
  notes = c(
    common_notes,
    "\\item Transition quarter excluded from quality specifications."
  )
)

# ---- Table 3: Spec C (Staffing, Quality) ----
tab3_blocks <- list(
  "Staffing"             = run_outcomes_single_spec(df_wo, staffing_tbl, "C", vc_month, fe_month),
  "Quality (Short-Stay)" = run_outcomes_single_spec(df_quality_post, short_stay_tbl, "C", vc_quarter, fe_quarter),
  "Quality (Long-Stay)"  = run_outcomes_single_spec(df_quality_post, long_stay_tbl,  "C", vc_quarter, fe_quarter)
)

tab3_tex <- assemble_spec_table(
  tab3_blocks,
  caption = "Spec C (B + Occupancy + Payer Mix + LOS): Staffing and Quality Outcomes",
  label = "tab:nested-spec-C",
  notes = c(
    common_notes,
    "\\item Transition quarter excluded from quality specifications."
  )
)

# df_wo's last use was Table 3 (Staffing at Spec C); everything left is
# quality-only, so free it now rather than carrying it through the rest of
# the script.
rm(df_wo); gc()

# ---- Table 4: Spec D (Quality only) ----
tab4_blocks <- list(
  "Quality (Short-Stay)" = run_outcomes_single_spec(df_quality_post, short_stay_tbl, "D", vc_quarter, fe_quarter),
  "Quality (Long-Stay)"  = run_outcomes_single_spec(df_quality_post, long_stay_tbl,  "D", vc_quarter, fe_quarter)
)

tab4_tex <- assemble_spec_table(
  tab4_blocks,
  caption = "Spec D (C + Staffing): Quality Outcomes",
  label = "tab:nested-spec-D",
  notes = c(
    common_notes,
    "\\item Transition quarter excluded from quality specifications."
  )
)

# -----------------------------------------------------------------------------
# Write outputs
# -----------------------------------------------------------------------------
write_fragment <- function(lines, fname) {
  fp <- file.path(out_dir, fname)
  writeLines(lines, fp, useBytes = TRUE)
  cat("[write] ", normalizePath(fp, winslash = "\\"), "\n", sep = "")
}

write_fragment(tab1_tex, "nested_control_spec_tableA.tex")
write_fragment(tab2_tex, "nested_control_spec_tableB.tex")
write_fragment(tab3_tex, "nested_control_spec_tableC.tex")
write_fragment(tab4_tex, "nested_control_spec_tableD.tex")

# NOTE: no \usepackage{float} and no [H] placement here on purpose. These
# tables are big enough (up to ~23 rows) that forcing exact ("H") placement
# on a float taller than the remaining page can push LaTeX to skip a page
# entirely, producing the blank first/third-page effect. Standard [!ht]
# placement (matching wald.R / twfe_post.R) lets LaTeX place each table on
# whatever page it actually fits on instead.
preview_lines <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{makecell}",
  "\\usepackage{array}",
  "\\usepackage{caption}",
  "",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "",
  "\\begin{document}",
  tab1_tex,
  "\\clearpage",
  tab2_tex,
  "\\clearpage",
  tab3_tex,
  "\\clearpage",
  tab4_tex,
  "\\end{document}"
)

write_fragment(preview_lines, "nested_control_spec_all_outcomes.tex")

cat("\nDone. Nested control-spec tables (Spec A, B, C, D) written.\n")
