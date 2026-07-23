# =============================================================================
# regressions/callaway_santanna_event_study.R
#
# Purpose:
#   Callaway & Sant'Anna (2021) group-time ATT / doubly-robust event-study
#   estimator, run alongside the existing TWFE event study
#   (twfe_event_study.R) and stacked event study (stacked_event_study.R).
#
#   Why this file exists (see thesis discussion with C. Moul):
#     - The TWFE event study estimates a single set of event-time
#       coefficients pooling all cohorts together. When treatment timing is
#       staggered and effects are dynamic/heterogeneous across cohorts,
#       already-treated facilities act as "controls" for later-treated
#       facilities in some of the underlying 2x2 comparisons TWFE averages
#       over, and those comparisons can enter with NEGATIVE weight
#       (Goodman-Bacon 2021; de Chaisemartin & D'Haultfoeuille 2020). The
#       coefficient can then move in a direction that has nothing to do with
#       the true dynamic treatment effect.
#     - stacked_event_study.R already fixes the comparison-group problem by
#       construction (each cohort is only ever compared to not-yet-treated /
#       never-treated facilities within its own window), which is the right
#       instinct. But it still pools cohorts into ONE regression with
#       implicit weights set by relative cell sizes, not by an efficiency- or
#       variance-based rule, and the joint pre-trend test built on top of it
#       (wald.R's ginv-based chi-square) needs a Moore-Penrose pseudoinverse
#       because the pre-period coefficient covariance matrix is often close
#       to singular -- a sign the test is being pushed past where the
#       asymptotics are comfortable.
#     - This script instead estimates group-time ATTs, ATT(g,t), one clean
#       2x2 DiD at a time, each using only untreated-at-that-time
#       comparison units, then aggregates them into an event-study with
#       NON-negative, package-derived weights (Callaway & Sant'Anna 2021).
#       Two things Moul flagged explicitly are now first-class arguments
#       instead of hand-rolled logic:
#         - "never-treated facility inclusion logic" -> control_group =
#           "nevertreated" vs "notyettreated" (both run below)
#         - the donut window -> anticipation = k (periods), instead of
#           physically dropping tau in {-3,-2,-1} from the sample
#       And the pre-trend check becomes: does the simultaneous
#       (sup-t / Roth 2022-style) confidence band for the pre-period
#       dynamic effects cover zero -- the same logic already cited in the
#       paper for the donut-window defense, now applied directly to a valid
#       estimator instead of to TWFE coefficients that may be contaminated.
#
#   Covariates: uses controls_A() (beds, chain_at_start) from _setup.R, NOT
#   the project's standard TWFE control set (which also includes
#   occupancy_rate, pct_medicare, pct_medicaid). Those three are plausibly
#   affected by the ownership change itself (they are examined as OUTCOMES
#   elsewhere in the paper -- see fe_only_specification_all_outcomes.R's own
#   caveat about this). Conditioning a doubly-robust estimator's
#   outcome-regression step on a "bad control" reintroduces exactly the
#   endogeneity problem Moul's FE-only request was trying to get away from,
#   so the leaner spec is used here on purpose. If this choice looks wrong,
#   flag it -- easy to swap for controls_B()/controls_C() in one line below.
#
# Requires: install.packages("did")   (Callaway & Sant'Anna's reference
#   implementation; CRAN package "did", not to be confused with base R.)
#
# NOTE ON PACKAGE INTERNALS:
#   Field names on the returned MP/AGGTEobj objects ($egt, $att.egt,
#   $se.egt, $crit.val.egt, $overall.att, $overall.se) are correct as of the
#   package versions this was written against, but this was NOT executed
#   against your local R installation (no R available in the sandbox this
#   was written in). If any extraction step below errors, run
#   `str(es_obj)` on the offending aggte() output and the field names will
#   be obvious from there -- the estimation calls (att_gt/aggte themselves)
#   are the part that matters and are stable across recent package versions.
#
# Output:
#   outputs/plots/cs_es_{rn,lpn,cna,total}_baseline.pdf
#     (preferred spec: control_group = "notyettreated", anticipation = 3)
#   outputs/tables/cs_es_{rn,lpn,cna,total}_coefs.csv
#   outputs/tables/pretrend_and_att_callaway_santanna_fragment.tex
#     (2x2 control_group x anticipation robustness grid, all 4 outcomes)
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

if (!requireNamespace("did", quietly = TRUE)) {
  stop("Package 'did' is required for this script. Install it with: install.packages(\"did\")", call. = FALSE)
}

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tibble)
  library(readr)
})

options(scipen = 999, digits = 4)

out_dir   <- out_tables_dir
plots_dir <- out_plots_dir
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)

frag_path <- file.path(out_dir, "pretrend_and_att_callaway_santanna_fragment.tex")

# -----------------------------------------------------------------------------
# Load panel, build id / g / t for the `did` package
# -----------------------------------------------------------------------------
keep_cols <- c(
  "cms_certification_number", "year_month", "time", "time_treated",
  "chain_at_start", "beds",
  "rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd"
)

df0 <- load_staffing_panel() %>%
  dplyr::mutate(cms_certification_number = as.character(cms_certification_number)) %>%
  dplyr::select(any_of(keep_cols)) %>%
  dplyr::mutate(
    id = as.integer(factor(cms_certification_number)),
    g  = dplyr::if_else(is.na(time_treated), 0L, as.integer(time_treated)),
    t  = as.integer(time)
  )

cat(sprintf("[cs] facilities = %s, cohorts (g>0) = %d, never-treated = %d\n",
            format(dplyr::n_distinct(df0$id), big.mark = ","),
            dplyr::n_distinct(df0$g[df0$g > 0]),
            dplyr::n_distinct(df0$id[!(df0$id %in% df0$id[df0$g > 0])])))

# -----------------------------------------------------------------------------
# Covariates: lean, pre-determined set only (see header note)
# -----------------------------------------------------------------------------
cs_covariates <- controls_A(df0)
xformla <- if (length(cs_covariates) == 0) {
  ~1
} else {
  stats::as.formula(paste("~", paste(cs_covariates, collapse = " + ")))
}
cat("[cs] xformla:", deparse(xformla), "\n")

outs_lvl <- c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd")
nice_out <- c(rn_hprd = "RN", lpn_hprd = "LPN", cna_hprd = "CNA", total_hprd = "Total")

# -----------------------------------------------------------------------------
# Fitting wrappers
# -----------------------------------------------------------------------------
run_cs <- function(yname, data, control_group, anticipation) {
  did::att_gt(
    yname                 = yname,
    tname                 = "t",
    idname                = "id",
    gname                 = "g",
    xformla               = xformla,
    data                  = data,
    panel                 = TRUE,
    allow_unbalanced_panel = TRUE,
    control_group         = control_group,   # "nevertreated" or "notyettreated"
    anticipation          = anticipation,    # donut, in periods (months)
    base_period           = "varying",
    est_method            = "dr",            # doubly-robust (outcome regression + IPW)
    bstrap                = TRUE,
    biters                = 1000,
    cband                 = TRUE
  )
}

agg_es <- function(att_obj, min_e = -24L, max_e = 24L) {
  did::aggte(att_obj, type = "dynamic", min_e = min_e, max_e = max_e,
             na.rm = TRUE, bstrap = TRUE, cband = TRUE)
}

agg_simple <- function(att_obj) {
  did::aggte(att_obj, type = "simple", na.rm = TRUE, bstrap = TRUE, cband = TRUE)
}

extract_es <- function(es_obj) {
  tibble::tibble(
    event_time = as.integer(es_obj$egt),
    estimate   = as.numeric(es_obj$att.egt),
    se         = as.numeric(es_obj$se.egt)
  ) %>%
    dplyr::mutate(
      crit  = es_obj$crit.val.egt,   # single sup-t critical value, applied uniformly
      ci_lo = estimate - crit * se,
      ci_hi = estimate + crit * se
    ) %>%
    dplyr::arrange(event_time)
}

# Roth (2022)-style joint pre-trend check: does the SIMULTANEOUS confidence
# band for the pre-period dynamic effects cover zero everywhere? This is the
# same logic already used in the paper's donut-window defense, applied
# directly to the CS estimator's own valid simultaneous band, in place of
# the hand-rolled chi-square/ginv Wald test used elsewhere in the project.
pretrend_check <- function(coefs_df) {
  pre <- dplyr::filter(coefs_df, event_time < 0)
  if (!nrow(pre)) return(list(note = "no pre-period estimates"))
  covers_zero <- with(pre, ci_lo <= 0 & ci_hi >= 0)
  list(
    n_pre               = nrow(pre),
    all_band_covers_zero = all(covers_zero),
    max_abs_t           = max(abs(pre$estimate / pre$se), na.rm = TRUE)
  )
}

fmt_pretrend_cell <- function(chk) {
  if (!is.null(chk$note)) return("$\\,$")
  verdict <- if (isTRUE(chk$all_band_covers_zero)) "pass" else "FAIL"
  sprintf("%s ($\\max|t|=%.2f$, %d pre-periods)", verdict, chk$max_abs_t, chk$n_pre)
}

fmt_att_cell <- function(att, se) {
  if (is.na(att) || is.na(se)) return("$\\,$")
  bstr <- sprintf("%.4f", att)
  if (att > 0) bstr <- paste0("\\phantom{-}", bstr)
  sprintf("$%s$ $(%.4f)$", bstr, se)
}

# -----------------------------------------------------------------------------
# Preferred spec: not-yet-treated comparison group, 3-month donut
# (matches the anticipation window already dropped elsewhere in the project)
# -----------------------------------------------------------------------------
PREFERRED_CG  <- "notyettreated"
PREFERRED_ANT <- 3L

main_fits <- list()
for (y in outs_lvl) {
  cat(sprintf("\n[cs] fitting preferred spec: %s (control_group=%s, anticipation=%d)\n",
              y, PREFERRED_CG, PREFERRED_ANT))
  att <- tryCatch(
    run_cs(y, df0, PREFERRED_CG, PREFERRED_ANT),
    error = function(e) { message(sprintf("[cs] att_gt failed for %s: %s", y, e$message)); NULL }
  )
  if (is.null(att)) { main_fits[[y]] <- NULL; next }
  main_fits[[y]] <- list(att = att, es = agg_es(att), simple = agg_simple(att))
}

# -----------------------------------------------------------------------------
# Plots + coefficient CSVs (preferred spec)
# -----------------------------------------------------------------------------
for (y in outs_lvl) {
  fit <- main_fits[[y]]
  if (is.null(fit)) next

  coefs <- extract_es(fit$es)

  coefs_csv_fp <- file.path(out_dir, sprintf("cs_es_%s_coefs.csv", sub("_hprd$", "", y)))
  readr::write_csv(coefs, coefs_csv_fp)
  cat("[write] ", normalizePath(coefs_csv_fp, winslash = "\\"), "\n", sep = "")

  p <- ggplot(coefs, aes(x = event_time, y = estimate)) +
    geom_hline(yintercept = 0, linetype = "dotted", color = "grey40") +
    geom_vline(xintercept = -0.5, linetype = "dashed", color = "grey40") +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.4, color = "steelblue") +
    geom_point(color = "steelblue", size = 1.6) +
    labs(
      x = "Months relative to ownership change (anticipation-adjusted)",
      y = paste0(nice_out[[y]], " HPRD")
    ) +
    theme_minimal(base_size = 12, base_family = "sans") +
    theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      panel.grid.minor = element_blank()
    )

  plot_fp <- file.path(plots_dir, sprintf("cs_es_%s_baseline.pdf", sub("_hprd$", "", y)))
  ggsave(plot_fp, plot = p, width = 7, height = 5, device = "pdf")
  cat("[write] ", normalizePath(plot_fp, winslash = "\\"), "\n", sep = "")
}

# -----------------------------------------------------------------------------
# Robustness grid: control_group x anticipation (2x2), all 4 outcomes
#   - directly answers Moul's "never-treated inclusion logic" question by
#     running both nevertreated and notyettreated explicitly
#   - directly answers "is the donut doing the work" by running
#     anticipation = 0 alongside anticipation = 3
# -----------------------------------------------------------------------------
robust_grid <- expand.grid(
  control_group = c("notyettreated", "nevertreated"),
  anticipation  = c(0L, 3L),
  stringsAsFactors = FALSE
)

robust_results <- list()
for (i in seq_len(nrow(robust_grid))) {
  cg  <- robust_grid$control_group[i]
  ant <- robust_grid$anticipation[i]
  key <- paste0(cg, "_ant", ant)
  cat(sprintf("\n[cs robustness] control_group=%s, anticipation=%d\n", cg, ant))

  row <- list()
  for (y in outs_lvl) {
    att <- tryCatch(run_cs(y, df0, cg, ant),
                     error = function(e) { message(sprintf("[cs robustness] failed for %s/%s: %s", key, y, e$message)); NULL })
    if (is.null(att)) { row[[y]] <- list(overall = NA, overall_se = NA, pretrend = list(note = "model failed")); next }

    es   <- agg_es(att)
    simp <- agg_simple(att)
    coefs <- extract_es(es)

    row[[y]] <- list(
      overall    = simp$overall.att,
      overall_se = simp$overall.se,
      pretrend   = pretrend_check(coefs)
    )
  }
  robust_results[[key]] <- list(control_group = cg, anticipation = ant, row = row)
}

# -----------------------------------------------------------------------------
# Build LaTeX fragment: Panel A (overall dynamic ATT), Panel B (pretrend check)
# -----------------------------------------------------------------------------
row_label <- function(cg, ant) {
  cg_lab <- if (cg == "notyettreated") "Not-yet-treated" else "Never-treated"
  ant_lab <- if (ant == 0) "no donut" else sprintf("%d-month donut", ant)
  sprintf("%s, %s", cg_lab, ant_lab)
}

mk_att_row <- function(res) {
  cells <- vapply(outs_lvl, function(y) fmt_att_cell(res$row[[y]]$overall, res$row[[y]]$overall_se), character(1))
  paste0(row_label(res$control_group, res$anticipation), " & ", paste(cells, collapse = " & "), " \\\\")
}

mk_pretrend_row <- function(res) {
  cells <- vapply(outs_lvl, function(y) fmt_pretrend_cell(res$row[[y]]$pretrend), character(1))
  paste0(row_label(res$control_group, res$anticipation), " & ", paste(cells, collapse = " & "), " \\\\")
}

att_rows      <- vapply(robust_results, mk_att_row, character(1))
pretrend_rows <- vapply(robust_results, mk_pretrend_row, character(1))

frag <- c(
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Callaway--Sant'Anna (2021) Group-Time ATT: Overall Dynamic Effect and Pre-Trend Check}",
  "\\label{tab:callaway-santanna-robustness}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "",
  "\\begin{tabularx}{\\textwidth}{@{} l YYYY @{} }",
  "\\toprule",
  " & \\multicolumn{4}{c}{\\textbf{Outcomes}} \\\\",
  "\\cmidrule(lr){2-5}",
  sprintf(" & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} \\\\",
          nice_out[["rn_hprd"]], nice_out[["lpn_hprd"]], nice_out[["cna_hprd"]], nice_out[["total_hprd"]]),
  "\\midrule",
  "\\multicolumn{5}{@{}l}{\\textbf{Panel A: Overall dynamic ATT (post-period average), standard error below}} \\\\[2pt]",
  att_rows,
  "\\addlinespace[6pt]",
  "\\multicolumn{5}{@{}l}{\\textbf{Panel B: Joint pre-trend check (simultaneous band covers zero?)}} \\\\[2pt]",
  pretrend_rows,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Doubly-robust group-time ATT estimator (Callaway \\& Sant'Anna 2021), estimated separately for each combination of comparison group (never-treated vs.\\ not-yet-treated) and anticipation window (0 vs.\\ 3 months, i.e.\\ no donut vs.\\ the project's standard donut). Panel A reports the overall post-treatment ATT from the dynamic (event-study) aggregation. Panel B reports whether the simultaneous (sup-t) confidence band for the pre-period event-study coefficients covers zero at every pre-period, in the spirit of Roth (2022); \\texttt{FAIL} indicates at least one pre-period estimate is excluded from the band.",
  sprintf("\\item Covariates: %s (deliberately excludes occupancy rate, payer mix, and case mix, which are examined as outcomes elsewhere in the paper).",
          if (length(cs_covariates) == 0) "none" else paste(cs_covariates, collapse = ", ")),
  "\\item All specifications drop facilities ever government-owned (see \\texttt{load\\_staffing\\_panel()}) and use \\texttt{chain\\_at\\_start} rather than time-varying chain status.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  ""
)

writeLines(frag, frag_path, useBytes = TRUE)
cat("\n[write] ", normalizePath(frag_path, winslash = "\\"), "\n", sep = "")

cat("\nDone. Callaway-Sant'Anna preferred-spec plots/CSVs and robustness fragment written.\n")
