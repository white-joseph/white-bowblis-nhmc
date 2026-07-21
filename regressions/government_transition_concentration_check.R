# =============================================================================
# regressions/government_transition_concentration_check.R
#
# Purpose:
#   Follow-up to government_transition_check.R: the sample of "became
#   government" facilities looked heavily concentrated in a single buyer
#   (Hendricks County Hospital, acquiring facilities previously owned by
#   Forrest Preston / Life Care Affiliates II) around August-December 2018.
#   This tallies the POST-acquisition owner name across ALL "became
#   government" facilities (not just a sample of 8), to quantify exactly
#   how concentrated this pattern is.
#
# Output: console only.
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

options(scipen = 999, digits = 4)

get_mode <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_real_)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

df <- load_staffing_panel()

pre_gov <- df %>%
  filter(treated == 1, event_time >= -12, event_time <= -4) %>%
  group_by(cms_certification_number) %>%
  summarise(gov_pre = get_mode(government), .groups = "drop")

post_gov <- df %>%
  filter(treated == 1, event_time >= 4, event_time <= 12) %>%
  group_by(cms_certification_number) %>%
  summarise(gov_post = get_mode(government), .groups = "drop")

event_month <- df %>%
  filter(treated == 1, event_time == 0) %>%
  distinct(cms_certification_number, year_month) %>%
  rename(event_year_month = year_month)

became_gov <- pre_gov %>%
  inner_join(post_gov, by = "cms_certification_number") %>%
  left_join(event_month, by = "cms_certification_number") %>%
  filter(gov_pre == 0, gov_post == 1)

cat(sprintf("Total facilities that BECAME government-owned: %d\n\n", nrow(became_gov)))

# -----------------------------------------------------------------------------
# Pull the POST-acquisition owner name for each, from ownership.csv
# -----------------------------------------------------------------------------
ownership_fp <- "C:/Repositories/white-bowblis-nhmc/data/interim/ownership.csv"
ownership <- read_csv(ownership_fp, show_col_types = FALSE) %>%
  mutate(cms_certification_number = as.character(cms_certification_number))

became_gov <- became_gov %>%
  mutate(cms_certification_number = as.character(cms_certification_number))

# For each facility, grab the LAST reported owner_name/owner_type in the panel
# (i.e., their most recent, presumably post-acquisition, ownership record).
post_owner <- ownership %>%
  filter(cms_certification_number %in% became_gov$cms_certification_number) %>%
  filter(role %in% c("DIRECT")) %>%  # avoid double-counting INDIRECT/PARTNERSHIP rows
  group_by(cms_certification_number) %>%
  filter(year_month == max(year_month, na.rm = TRUE)) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  select(cms_certification_number, owner_name, owner_type, association_date)

cat("=== Distribution of POST-acquisition owner name, across ALL 'became government' facilities ===\n")
owner_tally <- post_owner %>%
  count(owner_name, sort = TRUE) %>%
  mutate(pct = round(100 * n / sum(n), 1))
print(as.data.frame(owner_tally))

cat(sprintf(
  "\nTop owner accounts for %.1f%% of all 'became government' facilities (%d of %d)\n",
  owner_tally$pct[1], owner_tally$n[1], sum(owner_tally$n)
))

# Also check concentration of the PRE-acquisition owner (was it also one
# dominant seller, e.g. Forrest Preston / Life Care Affiliates II?)
pre_owner <- ownership %>%
  filter(cms_certification_number %in% became_gov$cms_certification_number) %>%
  filter(role %in% c("DIRECT")) %>%
  group_by(cms_certification_number) %>%
  filter(year_month == min(year_month, na.rm = TRUE)) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  select(cms_certification_number, owner_name)

cat("\n=== Distribution of PRE-acquisition (earliest observed) owner name ===\n")
seller_tally <- pre_owner %>%
  count(owner_name, sort = TRUE) %>%
  mutate(pct = round(100 * n / sum(n), 1))
print(as.data.frame(seller_tally) %>% head(10))

cat("\nDone.\n")

# -----------------------------------------------------------------------------
# State distribution of "became government" facilities
# -----------------------------------------------------------------------------
cat("\n=== State distribution of 'became government' facilities ===\n")

state_lookup <- df %>%
  filter(cms_certification_number %in% became_gov$cms_certification_number) %>%
  distinct(cms_certification_number, state) %>%
  group_by(cms_certification_number) %>%
  slice_head(n = 1) %>%
  ungroup()

state_tally <- state_lookup %>%
  count(state, sort = TRUE) %>%
  mutate(pct = round(100 * n / sum(n), 1))

print(as.data.frame(state_tally))

cat(sprintf(
  "\nTop state accounts for %.1f%% of all 'became government' facilities (%d of %d)\n",
  state_tally$pct[1], state_tally$n[1], sum(state_tally$n)
))

cat("\nDone.\n")
