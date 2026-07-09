# =============================================================================
# regressions/chain_transition_check.R
#
# Purpose:
#   Among TREATED (ever-CHOW) facilities, check how often chain status
#   changes around the ownership-change event:
#     - non-chain (pre) -> chain (post)   "became chain"
#     - chain (pre)     -> non-chain (post) "left chain"
#   vs. facilities whose chain status is unchanged across the event.
#
#   Pre/post windows mirror the spare_capacity baseline convention:
#     pre  = event_time in [-12, -4]  (before the anticipation window)
#     post = event_time in [+4, +12] (after the anticipation window)
#   Within each window, a facility's chain status is taken as the MODE
#   (most common value) to avoid being thrown off by a single noisy month.
#
# Output:
#   outputs/tables/chain_transition_summary.csv
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

options(scipen = 999, digits = 4)

df <- load_staffing_panel()
stopifnot(all(c("treated", "event_time", "chain") %in% names(df)))

get_mode <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_real_)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

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
