# =============================================================================
# regressions/wald.R
#
# Joint Wald tests of the pre-treatment event-time coefficients, used to assess
# whether treated and control facilities were on parallel paths before the
# ownership change. Reported separately for monthly staffing outcomes and
# quarterly quality outcomes.
#
# -----------------------------------------------------------------------------
# Staffing specifications
# -----------------------------------------------------------------------------
# Outcomes are RN, LPN, CNA, and total hours per resident day, in levels and
# logs. Three specifications are reported:
#
#   (1) Two-year window, full pre-period.
#       Event window tau in [-24, 24], reference period tau = -1, pre-trend
#       tested over tau = -24 to -2.
#
#   (2) Two-year window, excluding the immediate pre-transfer months.
#       Event window tau in [-24, 24], excluding tau = -3, -2, -1, reference
#       period tau = -4, pre-trend tested over tau = -24 to -5.
#
#   (3) One-year window, excluding the immediate pre-transfer months.
#       Event window tau in [-12, 12], excluding tau = -3, -2, -1, reference
#       period tau = -4, pre-trend tested over tau = -12 to -5.
#
# -----------------------------------------------------------------------------
# Quality specifications
# -----------------------------------------------------------------------------
# Estimated on the quarterly panel for the quality measures reported in the
# paper. The transition quarter (tau = 0) is excluded and tau = -1 is the
# reference period.
#
# Standard errors are two-way clustered by facility and calendar period
# throughout.
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
#   outputs/tables/wald-test-staffing.tex
#   outputs/tables/wald-test-quality.tex
#
# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
#   regressions/_setup.R
#   R packages: MASS
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(MASS)  # ginv(), for the generalized inverse in the Wald statistic
})

options(scipen = 999, digits = 4)

# ------------------------------ Paths ------------------------------
out_dir <- out_tables_dir
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------ Load ------------------------------
keep_cols <- c(
  "cms_certification_number", "year_month", "event_time", "treated",
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
  ) %>%
  dplyr::group_by(cms_certification_number) %>%
  dplyr::mutate(
    ever_treated = as.integer(any(treated == 1, na.rm = TRUE) | any(!is.na(event_time)))
  ) %>%
  dplyr::ungroup()

# ------------------------------ Helpers ------------------------------
# cap event_time to [-WIN, WIN] for treated; set 9999 for never-treated
prep_df <- function(dat, WIN) {
  dat %>%
    mutate(
      event_time_capped = dplyr::case_when(
        ever_treated == 1L & !is.na(event_time) ~ pmin(pmax(as.integer(event_time), -as.integer(WIN)), as.integer(WIN)),
        TRUE ~ 9999L
      ),
      ln_rn    = mk_log(rn_hprd),
      ln_lpn   = mk_log(lpn_hprd),
      ln_cna   = mk_log(cna_hprd),
      ln_total = mk_log(total_hprd)
    )
}

controls_rhs <- make_controls_rhs(df0)

pick_ref <- function(dat, desired = NULL) {
  ev <- sort(unique(dat$event_time_capped[dat$ever_treated == 1L]))
  ev <- ev[is.finite(ev) & ev != 9999L]
  if (!length(ev)) stop("No treated event times found.")
  if (!is.null(desired) && desired %in% ev) return(as.integer(desired))
  if (-1L %in% ev) return(-1L)
  if (-4L %in% ev) return(-4L)
  negs <- ev[ev < 0L]
  if (length(negs)) return(max(negs))
  return(ev[1])
}

run_es_twfe <- function(lhs, data, ref_val, WIN) {
  fml <- as.formula(paste0(
    lhs,
    " ~ i(event_time_capped, ever_treated, ref = ", ref_val, ", keep = -", as.integer(WIN), ":", as.integer(WIN), ") + ",
    controls_rhs,
    " | cms_certification_number + year_month"
  ))
  feols(
    fml,
    data = data,
    vcov = ~ cms_certification_number + year_month,
    lean = TRUE
  )
}

.es_pick <- function(mod, var = "event_time_capped", trt = "ever_treated") {
  cn <- names(coef(mod))
  if (is.null(cn) || !length(cn)) return(list(names = character(0), taus = integer(0)))
  pat <- sprintf("^%s::[-]?[0-9]+:%s$", var, trt)
  es_names <- grep(pat, cn, value = TRUE)
  get_tau <- function(s) as.integer(regmatches(s, regexpr("-?[0-9]+", s)))
  taus <- vapply(es_names, get_tau, integer(1))
  names(taus) <- es_names
  list(names = es_names, taus = taus)
}

pretrend_wald <- function(mod, ref_tau, from, to,
                          var = "event_time_capped", trt = "ever_treated") {
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

outs_lvl <- c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd")
nice_out <- c(rn_hprd = "RN", lpn_hprd = "LPN", cna_hprd = "CNA", total_hprd = "Total")
outs_log <- c(rn_hprd = "ln_rn", lpn_hprd = "ln_lpn", cna_hprd = "ln_cna", total_hprd = "ln_total")

# ------------------------------ Build three specs ------------------------------
skip2 <- c(-3L, -2L, -1L)

# Spec 1: WITH anticipation, WIN=24
WIN_A <- 24L
dat_with_24 <- prep_df(df0, WIN_A)
ref_with_24 <- pick_ref(dat_with_24, desired = -1L)
test_with_24 <- c(-WIN_A, -2L)  # test -24..-2 (exclude ref -1)

# Spec 2: WITHOUT anticipation, WIN=24 (drop -3,-2,-1), ref -4, test -24..-5
dat_wo_24 <- dat_with_24 %>%
  filter(!(ever_treated == 1L & event_time_capped %in% skip2))
ref_wo_24 <- pick_ref(dat_wo_24, desired = -4L)
test_wo_24 <- c(-WIN_A, -5L)

# Spec 3: WITHOUT anticipation, WIN=12 (drop -3,-2,-1), ref -4, test -12..-5
WIN_C <- 12L
dat_wo_12 <- prep_df(df0, WIN_C) %>%
  filter(!(ever_treated == 1L & event_time_capped %in% skip2))
ref_wo_12 <- pick_ref(dat_wo_12, desired = -4L)
test_wo_12 <- c(-WIN_C, -5L)

specs <- list(
  with_anticip = list(
    row_label = "2 Year Full Pre-Window",
    dat = dat_with_24,
    WIN = WIN_A,
    ref = ref_with_24,
    test_from = test_with_24[1],
    test_to   = test_with_24[2]
  ),
  wo_anticip_24 = list(
    row_label = "2 Year Window with Donut",
    dat = dat_wo_24,
    WIN = WIN_A,
    ref = ref_wo_24,
    test_from = test_wo_24[1],
    test_to   = test_wo_24[2]
  ),
  wo_anticip_12 = list(
    row_label = "1 Year Window with Donut",
    dat = dat_wo_12,
    WIN = WIN_C,
    ref = ref_wo_12,
    test_from = test_wo_12[1],
    test_to   = test_wo_12[2]
  )
)

# ------------------------------ Fit + Wald tests ------------------------------
fit_models_for_spec <- function(sp, is_log = FALSE) {
  mods <- list()
  for (y in outs_lvl) {
    lhs <- if (!is_log) y else outs_log[[y]]
    if (is_log && (lhs %in% names(sp$dat)) && all(is.na(sp$dat[[lhs]]))) {
      mods[[y]] <- NULL
    } else {
      mods[[y]] <- tryCatch(run_es_twfe(lhs, sp$dat, sp$ref, sp$WIN), error = function(e) NULL)
    }
  }
  mods
}

wald_for_spec <- function(mods, sp) {
  res <- lapply(outs_lvl, function(y) {
    pretrend_wald(mods[[y]], ref_tau = sp$ref, from = sp$test_from, to = sp$test_to)
  })
  names(res) <- outs_lvl
  res
}

wald_lvl <- list()
wald_log <- list()
N_rows   <- list()

for (nm in names(specs)) {
  sp <- specs[[nm]]
  
  mods_lvl <- fit_models_for_spec(sp, is_log = FALSE)
  mods_log <- fit_models_for_spec(sp, is_log = TRUE)
  
  wald_lvl[[nm]] <- wald_for_spec(mods_lvl, sp)
  wald_log[[nm]] <- wald_for_spec(mods_log, sp)
  
  N_rows[[nm]] <- nrow(sp$dat)
  cat("[spec]", nm, "|", sp$row_label, "| N =", format(N_rows[[nm]], big.mark = ","), "\n")
}

mk_row <- function(rowlabel, reslist) {
  cells <- vapply(outs_lvl, function(y) fmt_wald_cell(reslist[[y]]), character(1))
  paste0(rowlabel, " & ", paste(cells, collapse = " & "), " \\\\")
}

# ------------------------------ LaTeX table ------------------------------
wald_caption <- "Joint Wald Tests of Pre-trends for Monthly Staffing"
wald_label   <- "tab:wald-test-staffing"

notes_windows <- paste0(
  "\\item Tested windows and reference periods: ",
  "2 Year Full Pre-Window tests $\\tau=", specs$with_anticip$test_from, "$ to $\\tau=", specs$with_anticip$test_to,
  "$ with reference $\\tau=", specs$with_anticip$ref, "$; ",
  "2 Year Window with Donut tests $\\tau=", specs$wo_anticip_24$test_from, "$ to $\\tau=", specs$wo_anticip_24$test_to,
  "$ with reference $\\tau=", specs$wo_anticip_24$ref, "$ (dropping $\\tau=-3,-2,-1$); ",
  "1 Year Window with Donut tests $\\tau=", specs$wo_anticip_12$test_from, "$ to $\\tau=", specs$wo_anticip_12$test_to,
  "$ with reference $\\tau=", specs$wo_anticip_12$ref, "$ (dropping $\\tau=-3,-2,-1$)."
)

notes_N <- paste0(
  "\\item Sample sizes (rows): ",
  "2 Year Full Pre-Window ($N=", format(N_rows$with_anticip, big.mark = ","), "$); ",
  "2 Year Window with Donut ($N=", format(N_rows$wo_anticip_24, big.mark = ","), "$); ",
  "1 Year Window with Donut ($N=", format(N_rows$wo_anticip_12, big.mark = ","), "$)."
)

wald_tab <- c(
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  sprintf("\\caption{%s}", wald_caption),
  sprintf("\\label{%s}", wald_label),
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
  
  "\\multicolumn{5}{@{}l}{\\textbf{Panel A: HPRD}} \\\\[2pt]",
  mk_row(specs$with_anticip$row_label,      wald_lvl$with_anticip),
  mk_row(specs$wo_anticip_24$row_label,     wald_lvl$wo_anticip_24),
  mk_row(specs$wo_anticip_12$row_label,     wald_lvl$wo_anticip_12),
  
  "\\addlinespace[6pt]",
  "\\multicolumn{5}{@{}l}{\\textbf{Panel B: Log(HPRD)}} \\\\[2pt]",
  mk_row(specs$with_anticip$row_label,      wald_log$with_anticip),
  mk_row(specs$wo_anticip_24$row_label,     wald_log$wo_anticip_24),
  mk_row(specs$wo_anticip_12$row_label,     wald_log$wo_anticip_12),
  
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the Wald $\\chi^2$ statistic for the joint null that all pre-treatment event-time coefficients equal zero, followed by degrees of freedom in parentheses and the p-value in brackets.",
  notes_windows,
  notes_N,
  "\\item All specifications include facility and month fixed effects and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  ""
)

# ------------------------------ Write staffing Wald table ------------------------------
wald_staffing_path <- file.path(out_dir, "wald-test-staffing.tex")
writeLines(wald_tab, wald_staffing_path, useBytes = TRUE)

cat("\n[write] ", normalizePath(wald_staffing_path, winslash = "\\"), "\n", sep = "")
cat("Done with staffing Wald tests.\n")

# ================================================================
# Quality Joint Wald pretrend tests
#
# Outputs:
#   - outputs/tables/wald-test-quality.tex
#
# Notes:
#   - Uses only the quality metrics included in the paper.
#   - Produces one inputtable LaTeX table, not a standalone QA document.
#   - Excludes the ownership-change quarter (tau = 0) in preferred
#     quality specifications and uses tau = -1 as the reference period.
# ================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(fixest)
  library(MASS)
})

# ------------------------------ Quality paths ------------------------------
if (!exists("project_root")) {
  project_root <- "C:/Repositories/white-bowblis-nhmc"
}

quality_panel_fp <- file.path(project_root, "data", "clean", "quality_panel.csv")
quality_out_dir  <- out_tables_dir
dir.create(quality_out_dir, showWarnings = FALSE, recursive = TRUE)

quality_wald_path <- file.path(quality_out_dir, "wald-test-quality.tex")

# ------------------------------ Quality helpers ------------------------------
q_assert_has_cols <- function(df, cols, df_name = "data") {
  miss <- setdiff(cols, names(df))
  if (length(miss) > 0) {
    stop(
      sprintf("[%s] missing required columns: %s", df_name, paste(miss, collapse = ", ")),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

q_intersect_existing <- function(x, df) {
  intersect(x, names(df))
}

q_quarter_num <- function(x) {
  x <- toupper(trimws(as.character(x)))
  suppressWarnings(as.integer(stringr::str_extract(x, "[1-4]")))
}

q_year_quarter_index <- function(year, quarter) {
  qn <- q_quarter_num(quarter)
  yr <- suppressWarnings(as.integer(year))
  yr * 4L + qn
}

q_subset_window <- function(df, start_year, start_quarter, end_year, end_quarter) {
  start_idx <- start_year * 4L + start_quarter
  end_idx   <- end_year * 4L + end_quarter
  idx <- q_year_quarter_index(df$year, df$quarter)
  df[idx >= start_idx & idx <= end_idx, , drop = FALSE]
}

q_drop_event_quarter <- function(df) {
  df %>% filter(is.na(event_time) | event_time != 0)
}

q_prepare_event_study_data <- function(df, min_et, max_et) {
  q_assert_has_cols(df, c("treated", "event_time"), "quality_event_study_data")

  df %>%
    group_by(cms_certification_number) %>%
    mutate(
      ever_treated = as.integer(any(treated == 1, na.rm = TRUE) | any(!is.na(event_time)))
    ) %>%
    ungroup() %>%
    mutate(
      event_time_capped = case_when(
        ever_treated == 1L & !is.na(event_time) ~ pmin(pmax(as.integer(event_time), min_et), max_et),
        TRUE ~ 9999L
      )
    )
}

q_get_case_mix_controls <- function(df) {
  preferred <- q_intersect_existing(c("cm_q_state_2", "cm_q_state_3", "cm_q_state_4"), df)
  if (length(preferred) > 0) return(preferred)

  fallback <- q_intersect_existing(c("cm_q_nat_2", "cm_q_nat_3", "cm_q_nat_4"), df)
  fallback
}

q_get_controls <- function(df) {
  base_controls <- c(
    "government",
    "non_profit",
    "chain",
    "beds",
    "occupancy_rate",
    "pct_medicare",
    "pct_medicaid"
  )
  c(q_intersect_existing(base_controls, df), q_get_case_mix_controls(df))
}

q_make_controls_rhs <- function(df) {
  ctrls <- q_get_controls(df)
  if (length(ctrls) == 0) return("1")
  paste(ctrls, collapse = " + ")
}

q_pick_ref <- function(dat, desired = NULL) {
  ev <- sort(unique(dat$event_time_capped[dat$ever_treated == 1L]))
  ev <- ev[is.finite(ev) & ev != 9999L]
  if (!length(ev)) stop("No treated event times found.")
  if (!is.null(desired) && desired %in% ev) return(as.integer(desired))
  if (-1L %in% ev) return(-1L)
  negs <- ev[ev < 0L]
  if (length(negs)) return(max(negs))
  return(ev[1])
}

q_run_es_twfe <- function(lhs, data, controls_rhs, ref_val, window) {
  fml <- as.formula(paste0(
    lhs,
    " ~ i(event_time_capped, ever_treated, ref = ", ref_val,
    ", keep = ", window[1], ":", window[2], ") + ",
    controls_rhs,
    " | cms_certification_number + year_quarter"
  ))

  feols(
    fml = fml,
    data = data,
    vcov = ~ cms_certification_number + year_quarter,
    lean = TRUE
  )
}

q_es_pick <- function(mod, var = "event_time_capped", trt = "ever_treated") {
  cn <- names(coef(mod))
  if (is.null(cn) || !length(cn)) return(list(names = character(0), taus = integer(0)))
  pat <- sprintf("^%s::[-]?[0-9]+:%s$", var, trt)
  es_names <- grep(pat, cn, value = TRUE)
  get_tau <- function(s) as.integer(regmatches(s, regexpr("-?[0-9]+", s)))
  taus <- vapply(es_names, get_tau, integer(1))
  names(taus) <- es_names
  list(names = es_names, taus = taus)
}

q_pretrend_wald <- function(mod, ref_tau, from, to,
                            var = "event_time_capped", trt = "ever_treated") {
  if (is.null(mod)) return(list(note = "Model is NULL"))
  es <- q_es_pick(mod, var, trt)
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

q_fmt_wald_cell <- function(res) {
  if (!is.null(res$note)) return("$\\,$")
  sprintf("$%.2f$ (%d) [%.4f]", res$statistic, res$df, res$p.value)
}

q_escape_latex <- function(x) {
  x <- gsub("\\\\", "\\\\textbackslash{}", x)
  x <- gsub("([#$%&_{}])", "\\\\\\1", x, perl = TRUE)
  x <- gsub("~", "\\\\textasciitilde{}", x, fixed = TRUE)
  x <- gsub("\\^", "\\\\textasciicircum{}", x)
  x
}

# ------------------------------ Load quality panel ------------------------------
q_df0 <- readr::read_csv(quality_panel_fp, show_col_types = FALSE)

q_required_cols <- c(
  "cms_certification_number",
  "year",
  "quarter",
  "treated",
  "event_time"
)
q_assert_has_cols(q_df0, q_required_cols, "quality_panel")

q_df0 <- q_df0 %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year = suppressWarnings(as.integer(year)),
    quarter = toupper(trimws(as.character(quarter))),
    year_quarter = paste0(year, "_", quarter)
  )

q_numeric_candidates <- c(
  "beds", "occupancy_rate", "pct_medicare", "pct_medicaid",
  "event_time", "government", "non_profit", "chain"
)
q_numeric_candidates <- q_intersect_existing(q_numeric_candidates, q_df0)

if (length(q_numeric_candidates) > 0) {
  q_df0 <- q_df0 %>%
    mutate(across(all_of(q_numeric_candidates), ~ suppressWarnings(as.numeric(.x))))
}

q_controls_rhs <- q_make_controls_rhs(q_df0)

# ------------------------------ Quality outcomes used in the paper ------------------------------
# Confirm these QM mappings against the final data dictionary before publication.
quality_outcome_windows <- list(
  qm_401 = c(2017L, 1L, 2024L, 2L), # Catheter
  qm_410 = c(2017L, 1L, 2024L, 2L), # Antipsychotic
  qm_434 = c(2017L, 1L, 2024L, 2L), # Hypnotics / anti-anxiety or hypnotic medication
  qm_453 = c(2018L, 1L, 2023L, 3L), # Pressure injuries
  qm_419 = c(2017L, 1L, 2024L, 2L), # Falls with major injury
  qm_406 = c(2017L, 1L, 2024L, 2L), # Weight loss
  qm_407 = c(2017L, 1L, 2024L, 2L), # ADL increase / physical functioning decline
  qm_404 = c(2017L, 1L, 2024L, 2L)  # Urinary tract infections
)

quality_nice_out <- c(
  qm_401 = "Catheter",
  qm_410 = "Antipsychotic",
  qm_434 = "Hypnotics",
  qm_453 = "Pressure injuries",
  qm_419 = "Falls",
  qm_406 = "Weight loss",
  qm_407 = "ADL increase",
  qm_404 = "UTI"
)

quality_outcomes <- names(quality_outcome_windows)

q_missing_outcomes <- setdiff(quality_outcomes, names(q_df0))
if (length(q_missing_outcomes) > 0) {
  stop(sprintf("Missing requested quality outcomes: %s",
               paste(q_missing_outcomes, collapse = ", ")),
       call. = FALSE)
}

# ------------------------------ Quality specs ------------------------------
quality_specs <- list(
  full_8 = list(
    row_label = "2 Year Full Pre-Window",
    window = c(-8L, 8L),
    drop_event_quarter = FALSE,
    ref_desired = -1L,
    test_from = -8L,
    test_to = -1L
  ),
  drop_q0_8 = list(
    row_label = "2 Year Window, Excluding Ownership-Change Quarter",
    window = c(-8L, 8L),
    drop_event_quarter = TRUE,
    ref_desired = -1L,
    test_from = -8L,
    test_to = -1L
  ),
  drop_q0_4 = list(
    row_label = "1 Year Window, Excluding Ownership-Change Quarter",
    window = c(-4L, 4L),
    drop_event_quarter = TRUE,
    ref_desired = -1L,
    test_from = -4L,
    test_to = -1L
  )
)

# ------------------------------ Run quality models and Wald tests ------------------------------
quality_results <- list()
quality_sample_sizes <- list()

for (outcome in quality_outcomes) {
  cat("\n", strrep("=", 80), "\n", sep = "")
  cat("QUALITY OUTCOME: ", outcome, "\n", sep = "")
  cat(strrep("=", 80), "\n", sep = "")

  win <- quality_outcome_windows[[outcome]]
  dat_base <- q_subset_window(q_df0, win[1], win[2], win[3], win[4])

  quality_results[[outcome]] <- list()
  quality_sample_sizes[[outcome]] <- list()

  for (sp_nm in names(quality_specs)) {
    sp <- quality_specs[[sp_nm]]

    dat_sp <- dat_base
    if (isTRUE(sp$drop_event_quarter)) {
      dat_sp <- q_drop_event_quarter(dat_sp)
    }

    dat_sp <- q_prepare_event_study_data(
      dat_sp,
      min_et = sp$window[1],
      max_et = sp$window[2]
    ) %>%
      filter(!is.na(.data[[outcome]]))

    ref_val <- q_pick_ref(dat_sp, desired = sp$ref_desired)

    mod <- tryCatch(
      q_run_es_twfe(
        lhs = outcome,
        data = dat_sp,
        controls_rhs = q_controls_rhs,
        ref_val = ref_val,
        window = sp$window
      ),
      error = function(e) {
        warning(sprintf("Quality model failed for %s / %s: %s", outcome, sp_nm, e$message))
        NULL
      }
    )

    wald <- q_pretrend_wald(
      mod,
      ref_tau = ref_val,
      from = sp$test_from,
      to = sp$test_to
    )

    quality_results[[outcome]][[sp_nm]] <- list(
      model = mod,
      ref = ref_val,
      wald = wald
    )

    quality_sample_sizes[[outcome]][[sp_nm]] <- nrow(dat_sp)

    cat("[quality spec] ", sp_nm,
        " | ref=", ref_val,
        " | N=", format(nrow(dat_sp), big.mark = ","),
        "\n", sep = "")
  }
}

q_mk_row <- function(rowlabel, outcomes, spec_name) {
  cells <- vapply(
    outcomes,
    function(y) q_fmt_wald_cell(quality_results[[y]][[spec_name]]$wald),
    FUN.VALUE = character(1)
  )
  paste0(rowlabel, " & ", paste(cells, collapse = " & "), " \\\\")
}

# ------------------------------ One inputtable LaTeX table ------------------------------
# Transposed layout: outcomes are rows and specification windows are columns.
# This keeps the table portrait/vertical and avoids very wide outcome headers.

q_spec_names <- c("full_8", "drop_q0_8", "drop_q0_4")

q_spec_labels <- c(
  full_8    = "2 Year Full Pre-Window",
  drop_q0_8 = "2 Year Window, Excluding Ownership-Change Quarter",
  drop_q0_4 = "1 Year Window, Excluding Ownership-Change Quarter"
)

q_mk_transposed_row <- function(outcome) {
  cells <- vapply(
    q_spec_names,
    function(sp) q_fmt_wald_cell(quality_results[[outcome]][[sp]]$wald),
    FUN.VALUE = character(1)
  )

  paste0(
    quality_nice_out[[outcome]],
    " & ",
    paste(cells, collapse = " & "),
    " \\\\"
  )
}

q_table_rows <- vapply(
  quality_outcomes,
  q_mk_transposed_row,
  FUN.VALUE = character(1)
)

q_Ns_full <- paste(
  vapply(quality_outcomes, function(y) paste0(quality_nice_out[[y]], "=", format(quality_sample_sizes[[y]][["full_8"]], big.mark = ",")), character(1)),
  collapse = "; "
)

q_Ns_drop8 <- paste(
  vapply(quality_outcomes, function(y) paste0(quality_nice_out[[y]], "=", format(quality_sample_sizes[[y]][["drop_q0_8"]], big.mark = ",")), character(1)),
  collapse = "; "
)

q_Ns_drop4 <- paste(
  vapply(quality_outcomes, function(y) paste0(quality_nice_out[[y]], "=", format(quality_sample_sizes[[y]][["drop_q0_4"]], big.mark = ",")), character(1)),
  collapse = "; "
)

quality_wald_tab <- c(
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Joint Wald Tests of Pre-Trends for Quarterly Quality Measures}",
  "\\label{tab:wald-test-quality}",
  "\\small",
  "\\setlength{\\tabcolsep}{5pt}",
  "",
  "\\begin{tabular}{@{}lccc@{}}",
  "\\toprule",
  "Outcome & \\textbf{2 Year Full Pre-Window} & \\textbf{2 Year, with Donut} & \\textbf{1 Year, with Donut} \\\\",
  "\\midrule",
  q_table_rows,
  "\\bottomrule",
  "\\end{tabular}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the Wald $\\chi^2$ statistic for the joint null that all pre-treatment event-time coefficients equal zero, followed by degrees of freedom in parentheses and the p-value in brackets.",
  "\\item The 2 Year Full Pre-Window tests $\\tau=-8$ through $\\tau=-1$, with $\\tau=-1$ as the omitted reference period.",
  "\\item The ownership-change-quarter-excluded specifications omit $\\tau=0$ because the quarter of ownership change may combine pre- and post-transfer care, assessment, and documentation.",
  "\\item The 1 Year Window tests $\\tau=-4$ through $\\tau=-1$, with $\\tau=-1$ as the omitted reference period.",
  paste0("\\item Sample sizes by outcome: 2 Year Full Pre-Window [", q_Ns_full, "]."),
  paste0("\\item Sample sizes by outcome: 2 Year, with Donut [", q_Ns_drop8, "]."),
  paste0("\\item Sample sizes by outcome: 1 Year, with Donut [", q_Ns_drop4, "]."),
  paste0("\\item All specifications include facility and quarter fixed effects and covariates: ", q_escape_latex(q_controls_rhs), "."),
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  ""
)

writeLines(quality_wald_tab, quality_wald_path, useBytes = TRUE)

cat("\n[write] ", normalizePath(quality_wald_path, winslash = "\\"), "\n", sep = "")
cat("Done with quality Wald tests.\n")
