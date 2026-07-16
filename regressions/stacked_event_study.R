# =============================================================================
# regressions/stacked_event_study.R
#
# Purpose:
#   Stacked difference-in-differences event study for staffing HPRD outcomes,
#   used as a robustness check against staggered-adoption TWFE concerns
#   (Goodman-Bacon). Combines:
#     - the cohort-stacking approach already used in stacked_twfe_post.R
#       (each treated cohort g compared only to never-treated / not-yet-
#       treated facilities within its own event window)
#     - an actual EVENT-TIME specification (not just post/pre), so dynamics
#       can be plotted and pre-trends can be tested
#     - the joint Wald pretrend-test machinery already used in wald.R
#
#   Built on the CURRENT staffing_panel.csv (not the old panel.csv used by
#   stacked_twfe_post.R), so column names match the rest of the project
#   (rn_hprd, not rn_hppd; etc.).
#
#   TWO windows are estimated, matching the old (panel.csv-based) Table 8:
#     - 2 Year Window with Donut: event window +/-24 months, tests
#       tau = -24..-5, reference tau = -4
#     - 1 Year Window with Donut: event window +/-12 months, tests
#       tau = -12..-5, reference tau = -4
#   Both physically drop tau = -3,-2,-1 for treated units before fitting.
#
# Plots:
#   Built directly from extracted model coefficients (NOT fixest's iplot()),
#   so only event-times that actually have a fitted coefficient are ever
#   drawn -- donut months are explicitly filtered out a second time at the
#   plotting stage as a safeguard, guaranteeing no stray points there.
#   Plain sans-serif font throughout (ggplot2's default is already
#   sans-serif; base_family is set explicitly for portability).
#
# Output:
#   outputs/tables/pretrend_wald_tests_stacked_levels_fragment.tex
#     (inputtable LaTeX fragment -- matches the \input{} already present,
#     commented out, in ma_thesis.tex)
#   outputs/plots/stacked_es_rn_baseline.pdf
#   outputs/plots/stacked_es_lpn_baseline.pdf
#   outputs/plots/stacked_es_cna_baseline.pdf
#   outputs/plots/stacked_es_total_baseline.pdf
#     (from the 2-year window model, matching the paper's existing figure)
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(ggplot2)
  library(tibble)
  library(MASS)  # ginv
})

options(scipen = 999, digits = 4)

out_dir <- out_tables_dir
plots_dir <- out_plots_dir
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)

wald_frag_path <- file.path(out_dir, "pretrend_wald_tests_stacked_levels_fragment.tex")

# -----------------------------------------------------------------------------
# Load current panel
# -----------------------------------------------------------------------------
keep_cols <- c(
  "cms_certification_number", "year_month", "time", "time_treated",
  "government", "non_profit", "chain", "beds",
  "occupancy_rate", "pct_medicare", "pct_medicaid",
  "cm_q_state_2", "cm_q_state_3", "cm_q_state_4",
  "rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd"
)

df0 <- load_staffing_panel() %>%
  dplyr::select(any_of(keep_cols)) %>%
  dplyr::mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.factor(year_month)
  )

controls_rhs <- make_controls_rhs(df0)

outs_lvl <- c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd")
nice_out <- c(rn_hprd = "RN", lpn_hprd = "LPN", cna_hprd = "CNA", total_hprd = "Total")

# -----------------------------------------------------------------------------
# Cohort g_i (treatment month), same convention as stacked_twfe_post.R
# -----------------------------------------------------------------------------
g_df <- df0 %>%
  group_by(cms_certification_number) %>%
  summarise(
    g = {
      tt <- unique(time_treated[!is.na(time_treated)])
      if (length(tt) == 1) as.integer(tt) else NA_integer_
    },
    .groups = "drop"
  )

df0 <- df0 %>% left_join(g_df, by = "cms_certification_number")
cohorts <- sort(unique(df0$g[!is.na(df0$g)]))
cat("[stacked] unique cohorts (treated months):", length(cohorts), "\n")

# -----------------------------------------------------------------------------
# Build stacked data WITH event time (not just post), donut applied by
# physically dropping rel in {-3,-2,-1} for treated units
# -----------------------------------------------------------------------------
REF <- -4L
DONUT_SET <- c(-3L, -2L, -1L)

make_stacked_event_data <- function(data, cohorts_vec, L, R, donut_set) {
  stacked <- lapply(cohorts_vec, function(g0) {
    d <- data %>%
      dplyr::filter(time >= g0 - L, time <= g0 + R) %>%
      dplyr::filter(is.na(g) | g > g0 | g == g0) %>%
      dplyr::mutate(
        cohort = as.integer(g0),
        rel = as.integer(time - g0),
        treated_stack = as.integer(!is.na(g) & g == g0),
        stack_id = interaction(cms_certification_number, cohort, drop = TRUE)
      ) %>%
      # Donut: physically drop treated rows at the excluded event times.
      # Never-treated/not-yet-treated comparison rows are untouched.
      dplyr::filter(!(treated_stack == 1L & rel %in% donut_set)) %>%
      dplyr::select(
        cms_certification_number, year_month,
        cohort, stack_id, rel, treated_stack,
        government, non_profit, chain, beds,
        occupancy_rate, pct_medicare, pct_medicaid,
        cm_q_state_2, cm_q_state_3, cm_q_state_4,
        rn_hprd, lpn_hprd, cna_hprd, total_hprd
      )
    d
  })
  dplyr::bind_rows(stacked)
}

vc_stack <- ~ cms_certification_number + year_month  # two-way clustering (facility + calendar month)

make_es_fml <- function(lhs, keep_levels) {
  as.formula(paste0(
    lhs,
    " ~ i(rel, treated_stack, ref = ", REF, ", keep = c(",
    paste(keep_levels, collapse = ","), ")) + ",
    controls_rhs,
    " | stack_id + year_month + cohort"
  ))
}

fit_window <- function(L, R) {
  keep_levels <- setdiff(-L:R, DONUT_SET)
  stack <- make_stacked_event_data(df0, cohorts, L = L, R = R, donut_set = DONUT_SET)
  cat(sprintf("[stacked] window +/-%d: rows = %s\n", L, format(nrow(stack), big.mark = ",")))

  mods <- list()
  for (y in outs_lvl) {
    cat(sprintf("  [fit] window +/-%d, %s\n", L, y))
    mods[[y]] <- feols(make_es_fml(y, keep_levels), data = stack, vcov = vc_stack, lean = TRUE)
    gc()
  }
  list(mods = mods, n = nrow(stack))
}

cat("\n=== Fitting 2-year window (+/-24 months) ===\n")
win24 <- fit_window(L = 24L, R = 24L)

cat("\n=== Fitting 1-year window (+/-12 months) ===\n")
win12 <- fit_window(L = 12L, R = 12L)

# -----------------------------------------------------------------------------
# Joint Wald pretrend test (reused pattern from wald.R)
# -----------------------------------------------------------------------------
.es_pick <- function(mod, var = "rel", trt = "treated_stack") {
  cn <- names(coef(mod))
  if (is.null(cn) || !length(cn)) return(list(names = character(0), taus = integer(0)))
  pat <- sprintf("^%s::[-]?[0-9]+:%s$", var, trt)
  es_names <- grep(pat, cn, value = TRUE)
  get_tau <- function(s) as.integer(regmatches(s, regexpr("-?[0-9]+", s)))
  taus <- vapply(es_names, get_tau, integer(1))
  names(taus) <- es_names
  list(names = es_names, taus = taus)
}

pretrend_wald <- function(mod, ref_tau, from, to, var = "rel", trt = "treated_stack") {
  if (is.null(mod)) return(list(note = "Model is NULL"))
  es <- .es_pick(mod, var, trt)
  if (!length(es$names)) return(list(note = "No ES coefficients found"))

  pre_idx <- es$taus < 0L & es$taus != ref_tau & es$taus >= from & es$taus <= to
  pre_names <- names(es$taus)[pre_idx]
  if (!length(pre_names)) return(list(note = "No preperiod coefficients in window"))

  b <- coef(mod)[pre_names]
  V <- vcov(mod)[pre_names, pre_names, drop = FALSE]

  W <- as.numeric(t(b) %*% MASS::ginv(V) %*% b)
  df_w <- qr(V)$rank
  pval <- pchisq(W, df = df_w, lower.tail = FALSE)

  list(statistic = W, df = df_w, p.value = pval, window = c(from, to))
}

fmt_wald_cell <- function(res) {
  if (!is.null(res$note)) return("$\\,$")
  sprintf("$%.2f$ (%d) [%.4f]", res$statistic, res$df, res$p.value)
}

wald_24 <- lapply(outs_lvl, function(y) pretrend_wald(win24$mods[[y]], ref_tau = REF, from = -24L, to = -5L))
names(wald_24) <- outs_lvl

wald_12 <- lapply(outs_lvl, function(y) pretrend_wald(win12$mods[[y]], ref_tau = REF, from = -12L, to = -5L))
names(wald_12) <- outs_lvl

cat("\n[wald] 2 Year Window with Donut:\n")
for (y in outs_lvl) cat(sprintf("  %-6s -> %s\n", nice_out[[y]], fmt_wald_cell(wald_24[[y]])))
cat("[wald] 1 Year Window with Donut:\n")
for (y in outs_lvl) cat(sprintf("  %-6s -> %s\n", nice_out[[y]], fmt_wald_cell(wald_12[[y]])))

# -----------------------------------------------------------------------------
# Build inputtable LaTeX fragment (two rows, matches old Table 8's structure)
# -----------------------------------------------------------------------------
mk_row <- function(rowlabel, reslist) {
  cells <- vapply(outs_lvl, function(y) fmt_wald_cell(reslist[[y]]), character(1))
  paste0(rowlabel, " & ", paste(cells, collapse = " & "), " \\\\")
}

row_24 <- mk_row("2 Year Window with Donut", wald_24)
row_12 <- mk_row("1 Year Window with Donut", wald_12)

N_24 <- format(win24$n, big.mark = ",")
N_12 <- format(win12$n, big.mark = ",")

wald_tab <- c(
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Joint Wald Tests of Pre-Trends: Stacked Event Study (Levels)}",
  "\\label{tab:pretrend-wald-tests-stacked-levels}",
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
  row_24,
  row_12,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the Wald $\\chi^2$ statistic for the joint null that all pre-treatment event-time coefficients equal zero, followed by degrees of freedom in parentheses and the p-value in brackets.",
  sprintf("\\item Tested windows and reference periods: 2 Year Window with Donut tests $\\tau=-24$ to $\\tau=-5$ with reference $\\tau=%d$ (dropping $\\tau=-3,-2,-1$); 1 Year Window with Donut tests $\\tau=-12$ to $\\tau=-5$ with reference $\\tau=%d$ (dropping $\\tau=-3,-2,-1$).", REF, REF),
  sprintf("\\item Sample sizes (stacked rows): 2 Year Window with Donut ($N=%s$); 1 Year Window with Donut ($N=%s$).", N_24, N_12),
  "\\item Stacked design: each treated cohort is compared only to never-treated and not-yet-treated facilities within its own event window; the donut excludes $\\tau=-3,-2,-1$ (physically dropped from the estimation sample for treated units).",
  "\\item All specifications include facility-by-cohort fixed effects (stack\\_id), calendar-month fixed effects, cohort fixed effects, and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators. Standard errors are clustered by facility.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  ""
)

writeLines(wald_tab, wald_frag_path, useBytes = TRUE)
cat("\n[write] ", normalizePath(wald_frag_path, winslash = "\\"), "\n", sep = "")

# -----------------------------------------------------------------------------
# Plots: custom ggplot2 build from extracted coefficients (2-year window)
#   - Only event-times with an actual fitted coefficient are ever included.
#   - Donut taus filtered out again here as an explicit safeguard, even
#     though the underlying rows/keep-list already exclude them.
#   - Plain sans-serif font.
# -----------------------------------------------------------------------------
extract_es_coefs <- function(mod, ref_tau) {
  es <- .es_pick(mod)
  if (!length(es$names)) return(tibble(event_time = integer(0), estimate = numeric(0), se = numeric(0)))

  keep_idx <- !(es$taus %in% DONUT_SET)  # safeguard: never plot donut taus
  nm <- es$names[keep_idx]
  taus <- es$taus[keep_idx]

  b <- coef(mod)[nm]
  s <- se(mod)[nm]

  out <- tibble(event_time = taus, estimate = unname(b), se = unname(s))
  # Add the reference point explicitly (estimate = 0, se = 0 by construction)
  out <- bind_rows(out, tibble(event_time = ref_tau, estimate = 0, se = 0))
  out %>% arrange(event_time)
}

plot_fp <- c(
  rn_hprd    = file.path(plots_dir, "stacked_es_rn_baseline.pdf"),
  lpn_hprd   = file.path(plots_dir, "stacked_es_lpn_baseline.pdf"),
  cna_hprd   = file.path(plots_dir, "stacked_es_cna_baseline.pdf"),
  total_hprd = file.path(plots_dir, "stacked_es_total_baseline.pdf")
)

plot_title <- c(
  rn_hprd    = "Stacked Event Study: RN",
  lpn_hprd   = "Stacked Event Study: LPN",
  cna_hprd   = "Stacked Event Study: CNA",
  total_hprd = "Stacked Event Study: Total"
)

plot_ylab <- c(
  rn_hprd    = "RN HPRD",
  lpn_hprd   = "LPN HPRD",
  cna_hprd   = "CNA HPRD",
  total_hprd = "Total HPRD"
)

for (y in outs_lvl) {
  coefs <- extract_es_coefs(win24$mods[[y]], ref_tau = REF)
  coefs <- coefs %>%
    mutate(ci_lo = estimate - 1.96 * se, ci_hi = estimate + 1.96 * se)

  # Export the exact coefficients/SEs alongside the plot, so pre-trend
  # patterns can be inspected numerically rather than only visually.
  coefs_csv_fp <- file.path(out_dir, sprintf("stacked_es_%s_coefs.csv", sub("_hprd$", "", y)))
  readr::write_csv(coefs, coefs_csv_fp)
  cat("[write] ", normalizePath(coefs_csv_fp, winslash = "\\"), "\n", sep = "")

  p <- ggplot(coefs, aes(x = event_time, y = estimate)) +
    geom_hline(yintercept = 0, linetype = "dotted", color = "grey40") +
    geom_vline(xintercept = -0.5, linetype = "dashed", color = "grey40") +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.4, color = "steelblue") +
    geom_point(color = "steelblue", size = 1.6) +
    labs(
      x = "Months relative to ownership change",
      y = plot_ylab[[y]]
    ) +
    theme_minimal(base_size = 12, base_family = "sans") +
    theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      panel.grid.minor = element_blank()
    )

  ggsave(plot_fp[[y]], plot = p, width = 7, height = 5, device = "pdf")
  cat("[write] ", normalizePath(plot_fp[[y]], winslash = "\\"), "\n", sep = "")
}

cat("\nDone. Stacked event study (2-year and 1-year windows), Wald fragment, and plots all regenerated.\n")
