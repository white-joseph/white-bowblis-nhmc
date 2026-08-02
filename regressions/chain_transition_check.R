# =============================================================================
# regressions/chain_transition_check.R
#
# Measures how often chain affiliation changes around a facility's own
# ownership-change event, among treated (ever-CHOW) facilities.
#
# -----------------------------------------------------------------------------
# Description
# -----------------------------------------------------------------------------
# For each treated facility, chain status is summarized separately in a
# pre-event window and a post-event window, and the two are compared to
# classify the facility into one of four transition categories: stayed
# non-chain, became chain, stayed chain, or left chain.
#
# Pre and post windows mirror the spare_capacity baseline convention:
#   pre  = event_time in [-12, -4]  (before the anticipation window)
#   post = event_time in [+4, +12]  (after the anticipation window)
# Within each window, a facility's chain status is taken as the mode
# (most common value) across months, so a single noisy month does not
# determine the classification.
#
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
#   data/clean/staffing_panel.csv (via load_staffing_panel())
#
# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
#   outputs/tables/chain_transition_summary.csv
#     Counts and percentages of facilities in each transition category.
#
# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
#   regressions/_setup.R (load_staffing_panel(), out_tables_dir)
#   R packages: dplyr, readr
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

options(scipen = 999, digits = 4)

df <- load_staffing_panel()
stopifnot(all(c("treated", "event_time", "chain") %in% names(df)))

# -----------------------------------------------------------------------------
# get_mode()
#
# Returns the most frequently occurring non-missing value in a vector.
# Used to summarize a facility's chain status across several months into a
# single value per window.
#
# Arguments:
#   x -- Numeric vector (chain status, coded 0/1).
#
# Returns:
#   Numeric scalar: the modal value of x, ignoring NAs. NA_real_ if x has
#   no non-missing values.
# -----------------------------------------------------------------------------
get_mode <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_real_)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# -----------------------------------------------------------------------------
# Pre- and post-event chain classification
# -----------------------------------------------------------------------------
pre_chain <- df %>%
  filter(treated == 1, event_time >= -12, event_time <= -4) %>%
  group_by(cms_certification_number) %>%
  summarise(pre_chain = get_mode(chain), .groups = "drop")

post_chain <- df %>%
  filter(treated == 1, event_time >= 4, event_time <= 12) %>%
  group_by(cms_certification_number) %>%
  summarise(post_chain = get_mode(chain), .groups = "drop")

transitions <- pre_chain %>%
  inner_join(post_chain, by = "cms_certification_number") %>%
  filter(!is.na(pre_chain), !is.na(post_chain)) %>%
  mutate(
    transition = case_when(
      pre_chain == 0 & post_chain == 0 ~ "Stayed non-chain",
      pre_chain == 0 & post_chain == 1 ~ "Became chain",
      pre_chain == 1 & post_chain == 1 ~ "Stayed chain",
      pre_chain == 1 & post_chain == 0 ~ "Left chain",
      TRUE ~ NA_character_
    )
  )

# -----------------------------------------------------------------------------
# Summary and output
# -----------------------------------------------------------------------------
summary_tab <- transitions %>%
  count(transition, name = "n_facilities") %>%
  mutate(pct = 100 * n_facilities / sum(n_facilities)) %>%
  arrange(desc(n_facilities))

cat("=== Chain status transitions among treated facilities ===\n")
cat(sprintf("Treated facilities with usable pre AND post chain classification: %d\n\n", nrow(transitions)))
print(summary_tab)

out_fp <- file.path(out_tables_dir, "chain_transition_summary.csv")
write_csv(summary_tab, out_fp)
cat("\n[write] ", normalizePath(out_fp, winslash = "\\"), "\n", sep = "")
