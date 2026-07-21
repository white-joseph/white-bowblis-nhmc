# =============================================================================
# regressions/coverage_check.R
#
# Purpose:
#   Check how common short/incomplete panel coverage is, in two ways:
#
#   (1) TREATED facilities: for each, how many months of data exist BEFORE
#       and AFTER their own ownership-change event? A facility could pass
#       CHOW verification but only be observed for a few months pre- or
#       post-event (e.g., entered the panel shortly before being sold, or
#       exited/closed shortly after) -- which would make the "before vs.
#       after" comparison rest on very little actual data for that
#       facility.
#
#   (2) NEVER-TREATED facilities: since there's no event to be pre/post of,
#       this instead checks each facility's TOTAL observed duration in the
#       panel (first to last month, and total months actually observed) --
#       i.e., are there control facilities only present for a short window
#       (newly certified, closed early, or just sparse reporting)?
#
#   No restriction currently exists in the pipeline requiring a minimum
#   pre/post coverage window -- this investigates whether that matters in
#   practice.
#
# Output: console only.
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
})

options(scipen = 999, digits = 4)

hr <- function(title) {
  cat("\n", strrep("=", 78), "\n", sep = "")
  cat(title, "\n")
  cat(strrep("=", 78), "\n", sep = "")
}
subhr <- function(title) cat("\n---", title, "---\n")

df <- load_staffing_panel()
stopifnot(all(c("treated", "event_time", "year_month") %in% names(df)))

# =============================================================================
# PART 1: TREATED facilities -- pre/post coverage relative to their own event
# =============================================================================
hr("1. TREATED FACILITIES: COVERAGE BEFORE/AFTER THEIR OWN OWNERSHIP CHANGE")

treated_coverage <- df %>%
  filter(treated == 1, !is.na(event_time)) %>%
  group_by(cms_certification_number) %>%
  summarise(
    min_event_time = min(event_time, na.rm = TRUE),
    max_event_time = max(event_time, na.rm = TRUE),
    n_months_total = n_distinct(year_month),
    .groups = "drop"
  ) %>%
  mutate(
    months_pre  = pmax(0, -min_event_time),   # how many months BEFORE the event exist
    months_post = pmax(0, max_event_time)     # how many months AFTER the event exist
  )

cat(sprintf("Total treated facilities: %d\n\n", nrow(treated_coverage)))

subhr("Distribution of months of PRE-event coverage available")
print(summary(treated_coverage$months_pre))

subhr("Distribution of months of POST-event coverage available")
print(summary(treated_coverage$months_post))

subhr("How many treated facilities have AT LEAST 12 / 24 months on each side?")
cat(sprintf("At least 12 months PRE-event:  %d (%.1f%%)\n",
            sum(treated_coverage$months_pre >= 12), 100 * mean(treated_coverage$months_pre >= 12)))
cat(sprintf("At least 24 months PRE-event:  %d (%.1f%%)\n",
            sum(treated_coverage$months_pre >= 24), 100 * mean(treated_coverage$months_pre >= 24)))
cat(sprintf("At least 12 months POST-event: %d (%.1f%%)\n",
            sum(treated_coverage$months_post >= 12), 100 * mean(treated_coverage$months_post >= 12)))
cat(sprintf("At least 24 months POST-event: %d (%.1f%%)\n",
            sum(treated_coverage$months_post >= 24), 100 * mean(treated_coverage$months_post >= 24)))

subhr("Facilities with BOTH sides covered (symmetric window)")
cat(sprintf("At least 12mo PRE *and* 12mo POST: %d (%.1f%%)\n",
            sum(treated_coverage$months_pre >= 12 & treated_coverage$months_post >= 12),
            100 * mean(treated_coverage$months_pre >= 12 & treated_coverage$months_post >= 12)))
cat(sprintf("At least 24mo PRE *and* 24mo POST: %d (%.1f%%)\n",
            sum(treated_coverage$months_pre >= 24 & treated_coverage$months_post >= 24),
            100 * mean(treated_coverage$months_pre >= 24 & treated_coverage$months_post >= 24)))

subhr("The most concerning cases: barely any pre- or post-period at all")
cat(sprintf("Less than 4 months PRE-event (i.e., basically only the anticipation window or less): %d (%.1f%%)\n",
            sum(treated_coverage$months_pre < 4), 100 * mean(treated_coverage$months_pre < 4)))
cat(sprintf("Less than 4 months POST-event: %d (%.1f%%)\n",
            sum(treated_coverage$months_post < 4), 100 * mean(treated_coverage$months_post < 4)))

cat("\nSample of facilities with the THINNEST pre-event coverage:\n")
print(treated_coverage %>% arrange(months_pre) %>% select(cms_certification_number, months_pre, months_post) %>% head(10))

cat("\nSample of facilities with the THINNEST post-event coverage:\n")
print(treated_coverage %>% arrange(months_post) %>% select(cms_certification_number, months_pre, months_post) %>% head(10))

# =============================================================================
# PART 2: NEVER-TREATED facilities -- total observed duration
# =============================================================================
hr("2. NEVER-TREATED FACILITIES: TOTAL OBSERVED DURATION IN THE PANEL")

df_dates <- df %>%
  mutate(ym_date = as.Date(paste0(gsub("/", "-", year_month), "-01")))

nevertreated_coverage <- df_dates %>%
  filter(treated == 0) %>%
  group_by(cms_certification_number) %>%
  summarise(
    first_month = min(ym_date, na.rm = TRUE),
    last_month  = max(ym_date, na.rm = TRUE),
    n_months_observed = n_distinct(year_month),
    .groups = "drop"
  ) %>%
  mutate(
    span_months = as.integer(round(as.numeric(difftime(last_month, first_month, units = "days")) / 30.44)) + 1
  )

cat(sprintf("Total never-treated facilities: %d\n\n", nrow(nevertreated_coverage)))

subhr("Distribution of TOTAL MONTHS OBSERVED (not necessarily contiguous)")
print(summary(nevertreated_coverage$n_months_observed))

subhr("Distribution of CALENDAR SPAN (first to last observed month)")
print(summary(nevertreated_coverage$span_months))

subhr("How many never-treated facilities have short total coverage?")
cat(sprintf("Fewer than 12 months observed total: %d (%.1f%%)\n",
            sum(nevertreated_coverage$n_months_observed < 12), 100 * mean(nevertreated_coverage$n_months_observed < 12)))
cat(sprintf("Fewer than 24 months observed total: %d (%.1f%%)\n",
            sum(nevertreated_coverage$n_months_observed < 24), 100 * mean(nevertreated_coverage$n_months_observed < 24)))
cat(sprintf("At least 12 months observed total:   %d (%.1f%%)\n",
            sum(nevertreated_coverage$n_months_observed >= 12), 100 * mean(nevertreated_coverage$n_months_observed >= 12)))
cat(sprintf("At least 24 months observed total:   %d (%.1f%%)\n",
            sum(nevertreated_coverage$n_months_observed >= 24), 100 * mean(nevertreated_coverage$n_months_observed >= 24)))

cat("\nSample of never-treated facilities with the SHORTEST total coverage:\n")
print(nevertreated_coverage %>% arrange(n_months_observed) %>%
        select(cms_certification_number, first_month, last_month, n_months_observed) %>% head(10))

cat("\nDone.\n")
