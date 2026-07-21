# =============================================================================
# regressions/government_transition_check.R
#
# Purpose:
#   Per C. Moul's question: confirm concrete instances of facilities
#   transitioning INTO government ownership around their ownership-change
#   event, and pull their raw ownership records so the actual mechanism is
#   visible (not just the aggregate regression coefficient).
#
# Method (same pre/post window convention as chain_transition_check.R):
#   PRE  = mode of `government` over event_time in [-12,-4]
#   POST = mode of `government` over event_time in [+4,+12]
#   "Became government" = gov_pre == 0 & gov_post == 1
#
# For a sample of these facilities, pulls their raw ownership.csv records
# (owner name/type, association dates) around the transition date, so the
# actual mechanism (real government takeover vs. a data/coding quirk) can
# be inspected directly.
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

# -----------------------------------------------------------------------------
# Identify facilities transitioning INTO government ownership
# -----------------------------------------------------------------------------
df <- load_staffing_panel()
stopifnot(all(c("treated", "event_time", "government") %in% names(df)))

pre_gov <- df %>%
  filter(treated == 1, event_time >= -12, event_time <= -4) %>%
  group_by(cms_certification_number) %>%
  summarise(gov_pre = get_mode(government), .groups = "drop")

post_gov <- df %>%
  filter(treated == 1, event_time >= 4, event_time <= 12) %>%
  group_by(cms_certification_number) %>%
  summarise(gov_post = get_mode(government), .groups = "drop")

# Grab the actual event date (year_month at event_time == 0, or nearest) for
# each facility, to know when to look at ownership.csv around.
event_month <- df %>%
  filter(treated == 1, event_time == 0) %>%
  distinct(cms_certification_number, year_month) %>%
  rename(event_year_month = year_month)

transitions <- pre_gov %>%
  inner_join(post_gov, by = "cms_certification_number") %>%
  left_join(event_month, by = "cms_certification_number") %>%
  filter(!is.na(gov_pre), !is.na(gov_post))

cat(sprintf("Treated facilities with usable pre AND post government classification: %d\n\n", nrow(transitions)))

became_gov <- transitions %>% filter(gov_pre == 0, gov_post == 1)
left_gov   <- transitions %>% filter(gov_pre == 1, gov_post == 0)
stayed_gov <- transitions %>% filter(gov_pre == 1, gov_post == 1)
stayed_priv<- transitions %>% filter(gov_pre == 0, gov_post == 0)

cat("=== Government ownership transition summary ===\n")
cat(sprintf("Became government (0 -> 1):     %d\n", nrow(became_gov)))
cat(sprintf("Left government   (1 -> 0):     %d\n", nrow(left_gov)))
cat(sprintf("Stayed government (1 -> 1):     %d\n", nrow(stayed_gov)))
cat(sprintf("Stayed private    (0 -> 0):     %d\n", nrow(stayed_priv)))

cat("\n=== Sample of facilities that BECAME government-owned ===\n")
print(became_gov %>% select(cms_certification_number, event_year_month) %>% head(15))

# -----------------------------------------------------------------------------
# Pull raw ownership.csv records for a sample of these facilities, to see
# the actual owner name/type around the transition.
# -----------------------------------------------------------------------------
ownership_fp <- "C:/Repositories/white-bowblis-nhmc/data/interim/ownership.csv"
ownership <- tryCatch(read_csv(ownership_fp, show_col_types = FALSE), error = function(e) NULL)

if (is.null(ownership)) {
  cat("\nCould not load ownership.csv -- skipping raw ownership record inspection.\n")
} else {
  ownership <- ownership %>%
    mutate(cms_certification_number = as.character(cms_certification_number))

  sample_ccns <- became_gov %>% slice_head(n = 8) %>% pull(cms_certification_number)

  cat("\n=== Raw ownership.csv records for a sample of 'became government' facilities ===\n")
  for (ccn in sample_ccns) {
    ev_month <- became_gov$event_year_month[became_gov$cms_certification_number == ccn]
    cat(sprintf("\n--- CCN %s (event month: %s) ---\n", ccn, ifelse(length(ev_month) && !is.na(ev_month[1]), ev_month[1], "unknown")))
    recs <- ownership %>%
      filter(cms_certification_number == ccn) %>%
      arrange(year_month) %>%
      select(any_of(c("year_month", "role", "owner_type", "owner_name", "ownership_percentage", "association_date")))
    print(as.data.frame(recs))
  }
}

cat("\nDone.\n")
