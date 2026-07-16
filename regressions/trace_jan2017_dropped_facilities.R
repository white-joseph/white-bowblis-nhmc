# =============================================================================
# regressions/trace_jan2017_dropped_facilities.R
#
# Purpose:
#   805 facilities exist in the raw NH Compare archive for January 2017 but
#   are absent from the final staffing_panel.csv for that month. This script
#   checks each intermediate pipeline output (provider.csv, pbj_nurse.csv,
#   ownership.csv, mcr.csv) to find exactly which stage(s) drop them, so we
#   can tell whether this is a genuine data problem or a pipeline artifact
#   (e.g., an inner join requiring simultaneous presence across all sources).
#
# Output: printed to console (no file output needed for this diagnostic).
# =============================================================================

library(dplyr)
library(readr)

dropped_ccns <- read_csv(
  "C:/Repositories/white-bowblis-nhmc/data/interim/dropped_facilities_jan2017_check.csv",
  col_types = cols(cms_certification_number = col_character())
)$cms_certification_number

cat(sprintf("Checking %d facilities (present in raw Jan 2017 NHC archive, absent from final panel Jan 2017)\n\n", length(dropped_ccns)))

check_stage <- function(fp, label, ccn_col = "cms_certification_number", ym_col = "year_month", ym_val = "2017/01") {
  df <- read_csv(fp, col_types = cols(.default = "c"), progress = FALSE)
  names(df) <- tolower(names(df))
  ccn_col <- tolower(ccn_col)
  ym_col <- tolower(ym_col)
  if (!(ccn_col %in% names(df))) {
    cat(sprintf("[%s] Could not find column '%s' -- skipping\n", label, ccn_col))
    return(invisible(NULL))
  }
  df[[ccn_col]] <- formatC(as.numeric(df[[ccn_col]]), width = 6, flag = "0", format = "d")
  if (ym_col %in% names(df)) {
    df_month <- df %>% filter(.data[[ym_col]] == ym_val)
  } else {
    df_month <- df
  }
  present_ccns <- unique(df_month[[ccn_col]])
  n_present <- sum(dropped_ccns %in% present_ccns)
  cat(sprintf(
    "[%s] Of the %d dropped facilities: %d ARE present in this file (%s%s), %d are NOT\n",
    label, length(dropped_ccns), n_present,
    if (ym_col %in% names(df)) paste0("filtered to ", ym_val) else "no year_month column, checked whole file",
    "",
    length(dropped_ccns) - n_present
  ))
}

cat("=== Stage 1: provider.csv ===\n")
check_stage("C:/Repositories/white-bowblis-nhmc/data/interim/provider.csv", "provider.csv")

cat("\n=== Stage 2: pbj_nurse.csv ===\n")
check_stage("C:/Repositories/white-bowblis-nhmc/data/interim/pbj_nurse.csv", "pbj_nurse.csv")

cat("\n=== Stage 3: ownership.csv ===\n")
check_stage("C:/Repositories/white-bowblis-nhmc/data/interim/ownership.csv", "ownership.csv")

cat("\n=== Stage 4: mcr.csv ===\n")
check_stage("C:/Repositories/white-bowblis-nhmc/data/interim/mcr.csv", "mcr.csv")

cat("\n=== Stage 5: quality_measures.csv ===\n")
check_stage("C:/Repositories/white-bowblis-nhmc/data/interim/quality_measures.csv", "quality_measures.csv")

cat("\nDone. Whichever stage(s) show a LOW presence count is where these facilities are being dropped.\n")
cat("If ALL stages show them present but the FINAL panel doesn't, the drop happens in 06_panel.py's merge/join logic itself.\n")
