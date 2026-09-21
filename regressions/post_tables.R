# =============================================================================
# regressions/post_tables.R
#
# Produces the paper's post-only difference-in-differences tables: staffing,
# business-model outcomes, and quality, each reported pooled and by baseline
# chain affiliation, plus an appendix table testing whether the chain and
# non-chain estimates differ.
#
# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
#   outcome ~ post + beds | facility + calendar period
#
# Estimated by two-way fixed effects, with standard errors two-way clustered
# by facility and calendar period. This is Spec A as defined in _setup.R
# (post + beds + chain_at_start); chain_at_start is time-invariant by
# construction and therefore absorbed by the facility fixed effects, so it is
# excluded from the right-hand side here. The estimate on post is unaffected
# by that exclusion.
#
# Monthly outcomes exclude the anticipation window (event_time in -3, -2, -1).
# Quarterly outcomes exclude the transition quarter (event_time == 0).
#
# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------
#   Table 1  Staffing. Panel A reports hours per resident day (HPRD) in levels
#            and logs. Panel B reports the two components of HPRD separately:
#            raw monthly hours (the numerator) and resident days (the
#            denominator, i.e. facility census). Reporting the components
#            separately distinguishes a change in labor purchased from a change
#            in the census over which those hours are spread. Panels C and D
#            repeat Panel A separately for facilities chain-affiliated and
#            independent at baseline.
#   Table 2  Business-model outcomes: occupancy rate, Medicare and Medicaid
#            shares of patient days, and average length of stay, reported
#            pooled and separately by baseline chain affiliation.
#   Table 3  Quality measures, long-stay and short-stay. Columns (1) and (2)
#            report all facilities without and with staffing controls;
#            columns (3) and (4) split the baseline specification by chain
#            affiliation. Vaccination measures are excluded.
#   Table 4  Appendix: saturated interaction tests of whether the chain and
#            non-chain estimates in Tables 1-3 differ.
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
#   outputs/tables/post_staffing_table.tex        (tab:post-staffing)
#   outputs/tables/post_business_model_table.tex  (tab:post-business)
#   outputs/tables/post_quality_table.tex         (tab:post-quality)
#   outputs/tables/chain_interaction_tests.tex    (tab:chain-interaction)
#   outputs/tables/post_tables_preview.tex        (standalone preview document)
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

# Formats a coefficient over its standard error in a single table cell. All
# outcomes in this script report at four decimal places so that columns remain
# visually comparable.
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

# Variant used by the staffing and business-model tables, which omits the
# government-ownership exclusion sentence; that restriction is stated once in
# the data section of the paper.
spec_note_trimmed <- paste0(
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, ",
  "with standard errors in parentheses. All specifications include facility and ",
  "calendar-period fixed effects and control for the number of certified beds. ",
  "Standard errors are two-way clustered by facility and calendar period."
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
  "resident_days",
  "ln_rn", "ln_lpn", "ln_cna", "ln_total",
  "occupancy_rate", "pct_medicare", "pct_medicaid", "avg_los_total"
)

df_m_full <- load_staffing_panel()
df_m <- df_m_full %>% dplyr::select(dplyr::any_of(keep_monthly))
rm(df_m_full); gc(verbose = FALSE)

df_m_wo <- drop_anticipation_window(df_m)
rm(df_m); gc(verbose = FALSE)

vc_month <- ~ cms_certification_number + year_month
fe_month <- "cms_certification_number + year_month"

# Baseline chain subsamples. Heterogeneity is estimated by sample split
# rather than by interacting treatment with chain status, so that fixed
# effects and control coefficients are free to differ across the two groups.
# chain_at_start is constant within each subsample by construction and is
# excluded from the right-hand side throughout this script.
df_chain    <- df_m_wo %>% dplyr::filter(chain_at_start == 1)
df_nonchain <- df_m_wo %>% dplyr::filter(chain_at_start == 0)

n_missing_chain <- dplyr::n_distinct(
  df_m_wo$cms_certification_number[is.na(df_m_wo$chain_at_start)]
)
if (n_missing_chain > 0) {
  message(sprintf(
    "[chain] %d facilities have no chain_at_start and are excluded from the split panels",
    n_missing_chain
  ))
}

# =============================================================================
# TABLE 1: Staffing -- Panel A (HPRD), Panel B (decomposition)
# =============================================================================
staff_labels <- c("RN", "LPN", "CNA", "Total")

# Panel A: the preferred HPRD specification, levels and logs.
panel_a_rows <- list(
  list(label = "HPRD",      vars = c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd"), digits = 4),
  list(label = "Log(HPRD)", vars = c("ln_rn", "ln_lpn", "ln_cna", "ln_total"),          digits = 4)
)

# Panel B: the components of HPRD. Hours is the numerator and is reported by
# staff type, as in Panel A. Resident days is the denominator and is a single
# facility-level census measure, identical across staff types by construction,
# so it is estimated once and reported in a cell spanning all four columns.
panel_b_hours_row <- list(label = "Hours", vars = c("rn_hours_month", "lpn_hours_month", "cna_hours_month", "total_hours"), digits = 4)

staffing_body <- character(0)

staffing_body <- c(staffing_body, panel_header("Panel A: Hours per resident day (HPRD)", 5))
for (i in seq_along(panel_a_rows)) {
  r <- panel_a_rows[[i]]
  cells <- character(4)
  for (j in seq_along(r$vars)) {
    v <- r$vars[j]
    mod <- safe_fit(df_m_wo, v, vc_month, fe_month, label = paste(r$label, v))
    cells[j] <- fmt_est(mod, digits = r$digits)
    rm(mod); gc(verbose = FALSE)
  }
  staffing_body <- c(staffing_body, paste0(paste(c(r$label, cells), collapse = " & "), " \\\\"))
  if (i < length(panel_a_rows)) {
    staffing_body <- c(staffing_body, "\\addlinespace[0.4em]")
  }
}

staffing_body <- c(staffing_body, "\\addlinespace[0.7em]")
staffing_body <- c(staffing_body, panel_header("Panel B: Hours and resident days", 5))

hours_cells <- character(4)
for (j in seq_along(panel_b_hours_row$vars)) {
  v <- panel_b_hours_row$vars[j]
  mod <- safe_fit(df_m_wo, v, vc_month, fe_month, label = paste(panel_b_hours_row$label, v))
  hours_cells[j] <- fmt_est(mod, digits = panel_b_hours_row$digits)
  rm(mod); gc(verbose = FALSE)
}
staffing_body <- c(staffing_body, paste0(paste(c(panel_b_hours_row$label, hours_cells), collapse = " & "), " \\\\"))
staffing_body <- c(staffing_body, "\\addlinespace[0.4em]")

mod_rd <- safe_fit(df_m_wo, "resident_days", vc_month, fe_month, label = "Resident days")
rd_cell <- fmt_est(mod_rd, digits = 4)
staffing_body <- c(staffing_body, paste0("Resident days & \\multicolumn{4}{c}{", rd_cell, "} \\\\"))
rm(mod_rd); gc(verbose = FALSE)

# ---------------------------------------------------------------------------
# Panels C and D: the same HPRD specification estimated separately by baseline
# chain affiliation. Reported as stacked panels rather than as a separate
# table so that the pooled and split estimates share one set of outcome
# columns and can be read against one another directly.
# ---------------------------------------------------------------------------
add_hprd_panel <- function(body, dat, panel_label) {
  body <- c(body, "\\addlinespace[0.7em]")
  body <- c(body, panel_header(
    sprintf("%s (N = %s facility-months)", panel_label, format(nrow(dat), big.mark = ",")),
    5
  ))
  for (i in seq_along(panel_a_rows)) {
    r <- panel_a_rows[[i]]
    cells <- character(4)
    for (j in seq_along(r$vars)) {
      mod <- safe_fit(dat, r$vars[j], vc_month, fe_month,
                      label = paste(panel_label, r$label, r$vars[j]))
      cells[j] <- fmt_est(mod, digits = r$digits)
      rm(mod); gc(verbose = FALSE)
    }
    body <- c(body, paste0(paste(c(r$label, cells), collapse = " & "), " \\\\"))
    if (i < length(panel_a_rows)) body <- c(body, "\\addlinespace[0.4em]")
  }
  body
}

staffing_body <- add_hprd_panel(staffing_body, df_chain,    "Panel C: Chain-affiliated at baseline")
staffing_body <- add_hprd_panel(staffing_body, df_nonchain, "Panel D: Independent at baseline")

# Consistency check, reported to the console only. HPRD is constructed as
# hours divided by resident days, so it cannot have more non-missing
# observations than its own numerator; a warning here indicates a problem in
# panel construction upstream.
n_hprd  <- sum(!is.na(df_m_wo$total_hprd))
n_hours <- sum(!is.na(df_m_wo$total_hours))
if (n_hprd > n_hours) {
  message(sprintf(
    "[check] total_hprd has %s non-missing observations vs %s for total_hours (difference = %s)",
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
    spec_note_trimmed,
    paste0(
      "\\item Panel A reports staffing HPRD. Panel B decomposes ",
      "each staffing measure into its numerator (raw monthly hours, the total ",
      "quantity of nursing labor purchased) and denominator (resident days, i.e. ",
      "facility census)."
    ),
    paste0(
      "\\item Panels C and D re-estimate Panel A separately by baseline chain ",
      "affiliation. Chain status is each facility's classification at its first ",
      "observation in the panel. Facilities with no available chain ",
      "classification are excluded from both split panels."
    ),
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
  "pct_medicare",    "Medicare share",             4,
  "pct_medicaid",    "Medicaid share",             4,
  "avg_los_total",   "Average length of stay",     4
) %>% dplyr::filter(var %in% names(df_m_wo))

business_body <- character(0)
for (i in seq_len(nrow(business_spec))) {
  v <- business_spec$var[i]
  d <- business_spec$digits[i]
  m_pool <- safe_fit(df_m_wo,     v, vc_month, fe_month, label = paste("Pooled", business_spec$label[i]))
  m_ch   <- safe_fit(df_chain,    v, vc_month, fe_month, label = paste("Chain", business_spec$label[i]))
  m_nc   <- safe_fit(df_nonchain, v, vc_month, fe_month, label = paste("Non-chain", business_spec$label[i]))
  business_body <- c(
    business_body,
    paste0(business_spec$label[i], " & ", fmt_est(m_pool, d),
           " & ", fmt_est(m_ch, d), " & ", fmt_est(m_nc, d), " \\\\")
  )
  rm(m_pool, m_ch, m_nc); gc(verbose = FALSE)
}

business_tex <- wrap_table(
  business_body,
  caption = "Effect of Ownership Change on Business-Model Outcomes",
  label = "tab:post-business",
  colspec = "@{} l Y Y Y @{}",
  header_row = "Outcome & Pooled & Chain & Non-chain \\\\",
  notes = c(
    spec_note_trimmed,
    paste0(
      "\\item Occupancy rate is residents as a share of available bed-days. Payer shares ",
      "are shares of patient days."
    ),
    paste0(
      "\\item The chain and non-chain columns re-estimate the pooled specification ",
      "separately by baseline chain affiliation, defined as each facility's ",
      "classification at its first observation in the panel. Pooled sample: N = ",
      format(nrow(df_m_wo), big.mark = ","), " facility-months. Chain: N = ",
      format(nrow(df_chain), big.mark = ","), ". Non-chain: N = ",
      format(nrow(df_nonchain), big.mark = ","), "."
    ),
    sig_note
  )
)

write_fragment(business_tex, "post_business_model_table.tex")

# =============================================================================
# Interaction tests: are the chain and non-chain estimates different?
#
# The split panels and columns above report each subsample separately but do
# not test whether the two differ. Comparing significance across separately
# estimated models is not such a test.
#
# The specification below is the fully saturated interaction:
#
#   outcome ~ post + post:chain + beds + beds:chain
#             | facility + period^chain
#
# Interacting every right-hand-side term and the calendar-period fixed effects
# with the split variable reproduces the two separate regressions exactly, so
# the point estimates implied here match the split tables. What it adds is the
# coefficient on post:chain, which is the difference between the two groups
# together with a standard error. chain_at_start itself is time-invariant and
# absorbed by the facility fixed effects. Facilities with no chain
# classification are dropped, matching the split samples.
# =============================================================================
fit_interaction <- function(dat, lhs, vc, time_var) {
  if (!(lhs %in% names(dat))) return(NULL)
  d <- dat %>% dplyr::filter(!is.na(chain_at_start))
  if (nrow(d) == 0) return(NULL)

  fml <- as.formula(paste0(
    lhs, " ~ post + post:chain_at_start + beds + beds:chain_at_start",
    " | cms_certification_number + ", time_var, "^chain_at_start"
  ))

  cat(sprintf("[interaction] %s (N = %s)\n", lhs, format(nrow(d), big.mark = ",")))
  tryCatch(
    feols(fml, data = d, vcov = vc, lean = TRUE),
    error = function(e) {
      message(sprintf("[warn] interaction %s failed: %s", lhs, e$message))
      NULL
    }
  )
}

interaction_row <- function(dat, v, label, vc, time_var, digits = 4) {
  mod <- fit_interaction(dat, v, vc, time_var)
  cell <- "\\makecell[t]{-- \\\\ (--)}"
  # fmt_est() reports the coefficient on post; the quantity of interest here
  # is the interaction term, so the cell is formatted directly.
  if (!is.null(mod)) {
    s <- coef_se_star(mod, term = "post:chain_at_start")
    if (!is.na(s$coef)) {
      b <- formatC(s$coef, format = "f", digits = digits)
      if (s$coef > 0) b <- paste0("\\phantom{-}", b)
      se <- formatC(s$se, format = "f", digits = digits)
      cell <- if (s$stars == "") {
        paste0("\\makecell[t]{$", b, "$ \\\\ $(", se, ")$}")
      } else {
        paste0("\\makecell[t]{$", b, "^{", s$stars, "}$ \\\\ $(", se, ")$}")
      }
    }
  }
  rm(mod); gc(verbose = FALSE)
  paste0(label, " & ", cell, " \\\\")
}

interaction_staffing_rows <- c(
  interaction_row(df_m_wo, "rn_hprd",    "RN HPRD",    vc_month, "year_month"),
  interaction_row(df_m_wo, "lpn_hprd",   "LPN HPRD",   vc_month, "year_month"),
  interaction_row(df_m_wo, "cna_hprd",   "CNA HPRD",   vc_month, "year_month"),
  interaction_row(df_m_wo, "total_hprd", "Total HPRD", vc_month, "year_month")
)

interaction_business_rows <- character(0)
for (i in seq_len(nrow(business_spec))) {
  interaction_business_rows <- c(
    interaction_business_rows,
    interaction_row(df_m_wo, business_spec$var[i], business_spec$label[i],
                    vc_month, "year_month", business_spec$digits[i])
  )
}

rm(df_m_wo, df_chain, df_nonchain); gc(verbose = FALSE)

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

# Baseline chain subsamples for columns (3) and (4).
df_q_chain    <- df_q_post %>% dplyr::filter(chain_at_start == 1)
df_q_nonchain <- df_q_post %>% dplyr::filter(chain_at_start == 0)

n_missing_chain_q <- dplyr::n_distinct(
  df_q_post$cms_certification_number[is.na(df_q_post$chain_at_start)]
)
if (n_missing_chain_q > 0) {
  message(sprintf(
    "[chain] %d facilities have no chain_at_start and are excluded from the quality split",
    n_missing_chain_q
  ))
}

# Estimates one block of quality measures and returns formatted table rows,
# four columns per measure: (1) baseline, (2) with staffing controls, (3) the
# baseline specification on chain-affiliated facilities, and (4) on
# independent facilities. The label_map argument allows the same function to
# serve the long-stay and short-stay measure sets, whose labels are defined
# separately in _setup.R.
#
# Each measure is passed through trim_quality_measure_window() before
# estimation, restricting measures with known reporting gaps to the years over
# which they are actually reported.
build_quality_block <- function(codes, label_map) {
  rows <- character(0)
  n_with <- integer(0)
  for (v in codes) {
    lab <- unname(label_map[[v]])
    dat_v  <- trim_quality_measure_window(df_q_post, v)
    dat_ch <- trim_quality_measure_window(df_q_chain, v)
    dat_nc <- trim_quality_measure_window(df_q_nonchain, v)

    m1 <- safe_fit(dat_v, v, vc_quarter, fe_quarter, label = paste(lab, "(1)"))
    m2 <- safe_fit(dat_v, v, vc_quarter, fe_quarter,
                   extra_controls = STAFFING_CONTROLS, label = paste(lab, "(2)"))
    m3 <- safe_fit(dat_ch, v, vc_quarter, fe_quarter, label = paste(lab, "(3) chain"))
    m4 <- safe_fit(dat_nc, v, vc_quarter, fe_quarter, label = paste(lab, "(4) non-chain"))

    rows <- c(rows, paste0(
      lab, " & ", fmt_est(m1, 4), " & ", fmt_est(m2, 4),
      " & ", fmt_est(m3, 4), " & ", fmt_est(m4, 4), " \\\\"
    ))
    if (!is.null(m2)) n_with <- c(n_with, nobs(m2))
    rm(dat_v, dat_ch, dat_nc, m1, m2, m3, m4); gc(verbose = FALSE)
  }
  list(rows = rows, n_with = n_with)
}

mech <- build_quality_block(quality_mechanism_measures, long_stay_quality_measures)
outc <- build_quality_block(quality_outcome_measures, long_stay_quality_measures)
# Short-stay measures are reported in the same table as the long-stay measures
# but under a separate panel header, since the two are constructed from
# different resident populations.
short <- build_quality_block(names(short_stay_quality_measures), short_stay_quality_measures)

quality_body <- c(
  panel_header("Panel A: Long-stay labor-saving mechanism measures", 5),
  mech$rows,
  "\\addlinespace[0.6em]",
  panel_header("Panel B: Long-stay resident outcome measures", 5),
  outc$rows,
  "\\addlinespace[0.6em]",
  panel_header("Panel C: Short-stay measures", 5),
  short$rows
)

quality_tex <- wrap_table(
  quality_body,
  caption = "Effect of Ownership Change on Quality Measures",
  label = "tab:post-quality",
  colspec = "@{} l Y Y Y Y @{}",
  header_row = paste0(
    " & \\multicolumn{2}{c}{All facilities} & \\multicolumn{2}{c}{By chain affiliation} \\\\\n",
    "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n",
    "Outcome & (1) & (2) & (3) Chain & (4) Non-chain \\\\"
  ),
  notes = c(
    spec_note,
    paste0(
      "\\item Column (1) is the baseline specification. Column (2) adds RN, LPN, ",
      "and nurse aide hours per resident day as controls. Because staffing is itself ",
      "affected by ownership change, column (2) is not a preferred estimate of the ",
      "total effect; it is reported to show whether the quality response is ",
      "attenuated after conditioning on measured staffing inputs. Columns (3) and ",
      "(4) re-estimate column (1) separately by baseline chain affiliation, defined ",
      "as each facility's classification at its first observation in the panel."
    ),
    paste0(
      "\\item Long-stay measures (Panels A-B) and short-stay measures (Panel C) are ",
      "constructed from different resident populations and are not directly ",
      "comparable to one another. For every measure, lower values indicate better ",
      "measured quality."
    ),
    paste0(
      "\\item Pressure injuries is estimated on 2018--2023 only; improved function ",
      "is estimated on 2017--2022 only, the years in which each is reported. ",
      "Vaccination measures are excluded. The transition quarter ($\\tau = 0$) is excluded."
    ),
    sig_note
  )
)

write_fragment(quality_tex, "post_quality_table.tex")

# Sample sizes are not printed in the table; they are reported to the console
# so that any material divergence across columns is visible.
cat("\n=== Column (2) observation counts (staffing controls added) ===\n")
print(c(mech$n_with, outc$n_with, short$n_with))
cat(sprintf(
  "\n=== Quality split samples: chain %s, non-chain %s facility-quarters ===\n",
  format(nrow(df_q_chain), big.mark = ","), format(nrow(df_q_nonchain), big.mark = ",")
))

rm(df_q_chain, df_q_nonchain); gc(verbose = FALSE)

# ---- interaction tests for quality outcomes ----
build_interaction_quality_rows <- function(codes, label_map) {
  rows <- character(0)
  for (v in codes) {
    lab <- unname(label_map[[v]])
    dat <- trim_quality_measure_window(df_q_post, v)
    rows <- c(rows, interaction_row(dat, v, lab, vc_quarter, "year_quarter"))
    rm(dat); gc(verbose = FALSE)
  }
  rows
}

interaction_quality_rows <- c(
  build_interaction_quality_rows(quality_mechanism_measures, long_stay_quality_measures),
  build_interaction_quality_rows(quality_outcome_measures, long_stay_quality_measures),
  build_interaction_quality_rows(names(short_stay_quality_measures), short_stay_quality_measures)
)

rm(df_q_post); gc(verbose = FALSE)

# =============================================================================
# TABLE 5: Chain vs. non-chain differences (appendix)
# =============================================================================
interaction_body <- c(
  panel_header("Panel A: Staffing", 2),
  interaction_staffing_rows,
  "\\addlinespace[0.6em]",
  panel_header("Panel B: Business model", 2),
  interaction_business_rows,
  "\\addlinespace[0.6em]",
  panel_header("Panel C: Quality", 2),
  interaction_quality_rows
)

interaction_tex <- wrap_table(
  interaction_body,
  caption = "Tests of Differences Between Chain-Affiliated and Independent Facilities",
  label = "tab:chain-interaction",
  colspec = "@{} l Y @{}",
  header_row = "Outcome & Chain $-$ Non-chain \\\\",
  notes = c(
    paste0(
      "\\item \\textit{Notes:} Each cell reports the coefficient on the ",
      "interaction between \\textit{post} and baseline chain affiliation, with ",
      "standard errors in parentheses. This coefficient is the difference ",
      "between the chain and non-chain estimates reported elsewhere in the ",
      "paper; a negative value indicates a smaller effect among chain-affiliated ",
      "facilities."
    ),
    paste0(
      "\\item Every right-hand-side term and the calendar-period fixed effects ",
      "are interacted with chain affiliation, so the specification reproduces ",
      "the separate subsample regressions exactly while yielding a standard ",
      "error for their difference. All specifications include facility fixed ",
      "effects and control for the number of certified beds. Standard errors ",
      "are two-way clustered by facility and calendar period."
    ),
    paste0(
      "\\item Panels A and B are estimated on the facility-month panel ",
      "excluding the anticipation window ($\\tau = -3, -2, -1$); Panel C on the ",
      "facility-quarter panel excluding the transition quarter ($\\tau = 0$). ",
      "Facilities with no available chain classification are excluded."
    ),
    sig_note
  )
)

write_fragment(interaction_tex, "chain_interaction_tests.tex")

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
  interaction_tex,
  "\\end{document}"
)

write_fragment(preview, "post_tables_preview.tex")

cat("\nDone. Four tables written.\n")
