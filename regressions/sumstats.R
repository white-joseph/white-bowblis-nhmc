# =============================================================================
# regressions/sumstats.R
#
# Summary statistics for both analysis panels, reported as a single table with
# one panel per sample: Panel A for the facility-month staffing panel and
# Panel B for the facility-quarter quality panel.
#
# Both panels are loaded through the shared loaders in _setup.R, so the
# government-ownership exclusion and the definition of baseline chain status
# are identical across the two samples by construction. Any remaining
# difference in facility counts between them reflects a difference in data
# coverage rather than in sample definition; compare_panel_samples() reports
# that difference directly.
#
# -----------------------------------------------------------------------------
# Table design
# -----------------------------------------------------------------------------
# Reported N is the per-variable count of non-missing observations and
# therefore varies down the table, which makes per-measure reporting windows
# and coverage gaps visible rather than obscuring them behind a single
# sample-wide count. Facility counts are reported in the table notes.
#
# Two variables are deliberately omitted. The government-ownership indicator
# is identically zero in both samples, since government-owned facilities are
# excluded upstream. The time-varying chain indicator is omitted in favor of
# baseline chain status, which is the chain measure used throughout the paper.
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
#   outputs/tables/sumstats_combined.tex  (label tab:sumstats)
#   outputs/tables/sumstats_preview.tex   (standalone preview document)
#
# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
#   regressions/_setup.R
#   R packages: dplyr, purrr, tibble
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(purrr)
  library(tibble)
})

options(scipen = 999, digits = 3)

out_dir <- out_tables_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

N_COLS <- 4  # Variable, Mean, SD, N

# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------
# The `kind` field attached to each variable determines its number format, so
# that formatting rules are stored alongside the variable definition:
#   hprd   -> 3 decimals            (0.427)
#   pct    -> 1 decimal             (82.4)
#   count  -> 0 decimals, big.mark  (12,340)
#   beds   -> 1 decimal             (107.3)
#   binary -> 3 decimals            (0.712)
#   qm     -> 3 decimals
to_num <- function(x) suppressWarnings(as.numeric(x))

fmt_val <- function(x, kind) {
  if (is.na(x)) return("--")
  switch(
    kind,
    hprd   = formatC(x, format = "f", digits = 3),
    pct    = formatC(x, format = "f", digits = 1),
    count  = formatC(x, format = "f", digits = 0, big.mark = ","),
    beds   = formatC(x, format = "f", digits = 1),
    binary = formatC(x, format = "f", digits = 3),
    qm     = formatC(x, format = "f", digits = 3),
    formatC(x, format = "f", digits = 3)
  )
}

fmt_int <- function(x) format(x, big.mark = ",", trim = TRUE, scientific = FALSE)

# Mean, SD, and non-missing count in one pass.
mean_sd_n <- function(x) {
  x <- to_num(x)
  x <- x[is.finite(x)]
  if (length(x) == 0) return(list(mean = NA_real_, sd = NA_real_, n = 0L))
  list(mean = mean(x), sd = stats::sd(x), n = length(x))
}

# -----------------------------------------------------------------------------
# Row builders
# -----------------------------------------------------------------------------
# `spec` is a tibble of (var, label, kind). Variables not present in the panel
# are skipped, so the script tolerates a panel rebuilt without an optional
# column.
build_rows <- function(df, spec) {
  spec <- spec %>% dplyr::filter(var %in% names(df))
  if (!nrow(spec)) return(character(0))

  purrr::pmap_chr(
    list(spec$var, spec$label, spec$kind),
    function(v, lab, kind) {
      s <- mean_sd_n(df[[v]])
      paste0(
        lab, " & ",
        fmt_val(s$mean, kind), " & ",
        fmt_val(s$sd, kind), " & ",
        fmt_int(s$n), " \\\\"
      )
    }
  )
}

# Full-width row spanning all columns.
span_row <- function(inner) {
  paste0("\\multicolumn{", N_COLS, "}{@{}l}{", inner, "} \\\\[2pt]")
}

subheader <- function(label) {
  span_row(paste0("\\textit{", label, "}"))
}

panelheader <- function(label) {
  span_row(paste0("\\textbf{", label, "}"))
}

# -----------------------------------------------------------------------------
# Variable specifications
# -----------------------------------------------------------------------------
staffing_hprd_spec <- tibble::tribble(
  ~var,          ~label,        ~kind,
  "rn_hprd",     "RN HPRD",     "hprd",
  "lpn_hprd",    "LPN HPRD",    "hprd",
  "cna_hprd",    "CNA HPRD",    "hprd",
  "total_hprd",  "Total HPRD",  "hprd"
)

# Raw monthly hours are the HPRD numerator, and are summarized alongside HPRD
# so that both baselines are available when reading the staffing table.
staffing_hours_spec <- tibble::tribble(
  ~var,               ~label,                    ~kind,
  "rn_hours_month",   "RN hours (monthly)",      "count",
  "lpn_hours_month",  "LPN hours (monthly)",     "count",
  "cna_hours_month",  "CNA hours (monthly)",     "count",
  "total_hours",      "Total hours (monthly)",   "count"
)

facility_spec_monthly <- tibble::tribble(
  ~var,              ~label,                                ~kind,
  "non_profit",      "Non-profit",                          "binary",
  "chain_at_start",  "Chain affiliation (baseline)",        "binary",
  "beds",            "Beds",                                "beds",
  "occupancy_rate",  "Occupancy rate (\\%)",                "pct",
  "pct_medicare",    "\\% Medicare",                        "pct",
  "pct_medicaid",    "\\% Medicaid",                        "pct",
  "avg_los_total",   "Average length of stay (days)",       "beds",
  "cm_q_state_2",    "Acuity quartile 2",                   "binary",
  "cm_q_state_3",    "Acuity quartile 3",                   "binary",
  "cm_q_state_4",    "Acuity quartile 4",                   "binary"
)

facility_spec_quarterly <- tibble::tribble(
  ~var,              ~label,                                ~kind,
  "non_profit",      "Non-profit",                          "binary",
  "chain_at_start",  "Chain affiliation (baseline)",        "binary",
  "beds",            "Beds",                                "beds",
  "occupancy_rate",  "Occupancy rate (\\%)",                "pct",
  "pct_medicare",    "\\% Medicare",                        "pct",
  "pct_medicaid",    "\\% Medicaid",                        "pct",
  "cm_q_state_2",    "Acuity quartile 2",                   "binary",
  "cm_q_state_3",    "Acuity quartile 3",                   "binary",
  "cm_q_state_4",    "Acuity quartile 4",                   "binary"
)

# Quality-measure labels and groupings are taken from _setup.R so that they
# remain consistent with the quality tables and figures.
qm_spec <- function(codes) {
  tibble::tibble(
    var   = codes,
    label = unname(long_stay_quality_measures[codes]),
    kind  = "qm"
  )
}

mechanism_spec <- qm_spec(quality_mechanism_measures)
outcome_spec   <- qm_spec(quality_outcome_measures)

# -----------------------------------------------------------------------------
# Load panels, retaining only the columns summarized below
# -----------------------------------------------------------------------------
keep_monthly <- unique(c(
  "cms_certification_number", "year_month", "treated",
  staffing_hprd_spec$var, staffing_hours_spec$var, facility_spec_monthly$var
))

df_m_full <- load_staffing_panel()
df_m <- df_m_full %>% dplyr::select(dplyr::any_of(keep_monthly))
rm(df_m_full); gc(verbose = FALSE)

keep_quarterly <- unique(c(
  "cms_certification_number", "year", "quarter", "year_quarter", "treated",
  mechanism_spec$var, outcome_spec$var,
  staffing_hprd_spec$var, facility_spec_quarterly$var
))

df_q_full <- load_quality_panel()
df_q <- df_q_full %>% dplyr::select(dplyr::any_of(keep_quarterly))
rm(df_q_full); gc(verbose = FALSE)

# -----------------------------------------------------------------------------
# Sample overviews (for the panel header rows)
# -----------------------------------------------------------------------------
make_overview <- function(df, period_col) {
  list(
    rows       = nrow(df),
    facilities = dplyr::n_distinct(df$cms_certification_number),
    treated    = dplyr::n_distinct(
      df$cms_certification_number[df$treated %in% c(1L, 1, "1")]
    ),
    period_min = suppressWarnings(min(df[[period_col]], na.rm = TRUE)),
    period_max = suppressWarnings(max(df[[period_col]], na.rm = TRUE))
  )
}

ov_m <- make_overview(df_m, "year_month")
ov_q <- make_overview(df_q, "year_quarter")

cat("\n=== Sample overview ===\n")
cat(sprintf("Staffing (facility-month):   %s rows, %d facilities, %d treated\n",
            fmt_int(ov_m$rows), ov_m$facilities, ov_m$treated))
cat(sprintf("Quality  (facility-quarter): %s rows, %d facilities, %d treated\n",
            fmt_int(ov_q$rows), ov_q$facilities, ov_q$treated))
cat(sprintf("Facility-count difference:   %d\n\n",
            ov_m$facilities - ov_q$facilities))

# -----------------------------------------------------------------------------
# Table body
# -----------------------------------------------------------------------------
body <- c(
  panelheader("Panel A: Staffing sample (facility--month)"),
  subheader("Staffing outcomes"),
  build_rows(df_m, staffing_hprd_spec),
  "\\addlinespace[0.4em]",
  build_rows(df_m, staffing_hours_spec),
  "\\addlinespace[0.6em]",
  subheader("Facility characteristics"),
  build_rows(df_m, facility_spec_monthly),

  "\\addlinespace[0.9em]",

  panelheader("Panel B: Quality sample (facility--quarter)"),
  subheader("Quality measures"),
  build_rows(df_q, mechanism_spec),
  "\\addlinespace[0.4em]",
  build_rows(df_q, outcome_spec),
  "\\addlinespace[0.6em]",
  subheader("Staffing and facility characteristics"),
  build_rows(df_q, staffing_hprd_spec),
  "\\addlinespace[0.4em]",
  build_rows(df_q, facility_spec_quarterly)
)

# -----------------------------------------------------------------------------
# Table notes
# -----------------------------------------------------------------------------
# Facility and treated-facility counts are reported here, since the
# per-variable N column cannot convey them.
sample_note <- paste0(
  "\\item Panel A: ", fmt_int(ov_m$facilities), " facilities (",
  fmt_int(ov_m$treated), " with an ownership change), ",
  ov_m$period_min, "--", ov_m$period_max, ". ",
  "Panel B: ", fmt_int(ov_q$facilities), " facilities (",
  fmt_int(ov_q$treated), " with an ownership change), ",
  ov_q$period_min, "--", ov_q$period_max, "."
)

notes <- c(
  paste0(
    "\\item \\textit{Notes:} $N$ is the number of non-missing observations ",
    "for each variable. HPRD denotes hours per resident day; monthly hours ",
    "are the HPRD numerator. Chain affiliation is measured at the start of ",
    "each facility's panel and held fixed. Acuity quartiles are within-state, ",
    "within-period case-mix quartiles, with quartile 1 omitted. For every ",
    "quality measure, lower values indicate better measured quality."
  ),
  sample_note,
  paste0(
    "\\item Both samples exclude hospital-based facilities, facilities outside ",
    "the 48 contiguous states, and any facility government-owned at any point ",
    "during the study period."
  )
)

fragment <- c(
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Summary Statistics}",
  "\\label{tab:sumstats}",
  "\\footnotesize",
  "\\setlength{\\tabcolsep}{8pt}",
  "",
  "\\begin{tabularx}{\\textwidth}{@{} X r r r @{} }",
  "\\toprule",
  "\\textbf{Variable} & \\textbf{Mean} & \\textbf{SD} & \\textbf{N} \\\\",
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

# -----------------------------------------------------------------------------
# Write output
# -----------------------------------------------------------------------------
write_fragment <- function(lines, fname) {
  fp <- file.path(out_dir, fname)
  writeLines(lines, fp, useBytes = TRUE)
  cat("[write] ", normalizePath(fp, winslash = "\\"), "\n", sep = "")
}

write_fragment(fragment, "sumstats_combined.tex")

preview_doc <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{array}",
  "\\usepackage{caption}",
  "\\captionsetup{labelfont=bf, font=small}",
  "",
  "\\begin{document}",
  fragment,
  "\\end{document}"
)

write_fragment(preview_doc, "sumstats_preview.tex")

cat("\nDone. Combined summary statistics table written.\n")
