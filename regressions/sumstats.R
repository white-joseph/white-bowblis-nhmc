# =============================================================================
# regressions/sumstats.R
#
# Summary statistics for BOTH analysis panels, as a single combined table.
#
# Replaces:
#   summary_statistics.R       -- read data/clean/panel.csv, which no longer
#                                 exists, and used the old *_hppd column
#                                 names. Could not run.
#   quarterly_summary_stats.R  -- read quality_panel.csv directly, applying
#                                 neither the government-ever exclusion nor
#                                 chain_at_start, so the quality sample did
#                                 not match any staffing table in the paper.
#
# Both panels are loaded through _setup.R's loaders, which route through the
# same facility-level lookups. The government exclusion set and
# chain_at_start are therefore identical across the two samples by
# construction. Any remaining difference in facility counts between the
# panels is a real data difference rather than a definitional one --
# compare_panel_samples() reports it directly.
#
# Output:
#   outputs/tables/sumstats_combined.tex   fragment for \input (label tab:sumstats)
#   outputs/tables/sumstats_preview.tex    standalone compilable doc
#
# PAPER WIRING: this script no longer writes summary_statistics_code.tex or
# quality_summary_statistics_code.tex. Those files still exist on disk from
# the old scripts, so ma_thesis.tex will still COMPILE if its \input{} lines
# are left alone -- it will just silently render stale numbers. Repoint the
# staffing \input{} at sumstats_combined.tex, delete the quality \input{}
# line and its surrounding \ref{tab:quality_sumstats} sentence, then delete
# the two stale .tex files.
#
# TABLE DESIGN:
#   - One exhibit, two panels (Health Economics allows 8 figures + tables
#     total; two exhibits on descriptives is expensive).
#   - N is per-variable non-missing count, so it varies down the table. This
#     is deliberate: it surfaces the qm_471/qm_472 reporting-window trims and
#     any coverage gaps in the raw-hours columns rather than hiding them
#     behind a single sample-wide figure.
#   - Facility counts stay in the notes, on a single compact line. A
#     per-variable N cannot convey them, but hanging them off the
#     \textbf{Panel A/B} header rows reads as clutter.
#
# NOT reported, deliberately:
#   government  -- every ever-government facility is excluded upstream, so
#                  the dummy is identically zero in both samples.
#   chain       -- the time-varying version is unreliable per Bowblis;
#                  chain_at_start is the project's chain variable.
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
# `kind` drives the number format so digit rules live with the variable
# definition rather than in a lookup function that has to be kept in sync:
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
# `spec` is a tibble of (var, label, kind). Variables absent from the panel
# are silently skipped -- the same tolerance used elsewhere in the project,
# so this does not break on a panel rebuilt without an optional column.
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

# Raw hours are the HPRD numerator. They belong here now that the paper
# reports both -- a reader comparing the HPRD and raw-hours coefficient
# columns needs both baselines to benchmark against.
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

# Quality measures pull labels and grouping from _setup.R, so this script
# cannot drift from quality_event_study.R or the quality tables.
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
# Load panels (slimmed immediately -- only the columns summarized below)
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
# Notes
# -----------------------------------------------------------------------------
# Facility and treated-facility counts live here rather than in the panel
# header rows: a per-variable N column cannot convey them, but appending
# them to the \textbf{Panel A/B} rows reads as clutter. Kept to a single
# compact line covering both panels.
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
# Write
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
