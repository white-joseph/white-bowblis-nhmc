# =============================================================================
# regressions/chow_facility_profile.R
#
# Purpose:
#   Descriptive profile of facilities that undergo ownership change (CHOW),
#   answering a set of exploratory questions rather than producing a
#   regression table:
#     1. What kind of facilities change ownership? (for-profit/non-profit/
#        government, chain/non-chain)
#     2. What do they change INTO? (ownership-type transition matrix)
#     3. What do they look like just before the sale? (occupancy, spare
#        capacity, beds, case mix, payer mix) compared to never-treated
#        facilities
#     4. Are they close to staffing minimums pre-sale?
#     5. Is quality already worse pre-sale?
#
#   Console output only, organized into clearly labeled sections.
#
# Baseline window convention (consistent with the rest of this project):
#   PRE  = event_time in [-12, -4]  (before the anticipation window)
#   POST = event_time in [+4, +12]  (after the anticipation window)
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(tibble)
})

options(scipen = 999, digits = 4)

hr <- function(title) {
  cat("\n", strrep("=", 78), "\n", sep = "")
  cat(title, "\n")
  cat(strrep("=", 78), "\n", sep = "")
}
subhr <- function(title) cat("\n---", title, "---\n")

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
df <- load_staffing_panel()

quality_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/quality_panel.csv"
df_quality <- tryCatch(read_csv(quality_fp, show_col_types = FALSE), error = function(e) NULL)

stopifnot(all(c("treated", "event_time", "government", "non_profit", "chain") %in% names(df)))

# Simple 3-category ownership type from the available binary flags
own_type <- function(gov, np) {
  case_when(
    gov == 1 ~ "Government",
    np == 1  ~ "Non-profit",
    TRUE     ~ "For-profit"
  )
}

# -----------------------------------------------------------------------------
# 1. WHO CHANGES OWNERSHIP: pre-acquisition ownership type / chain status
# -----------------------------------------------------------------------------
hr("1. WHAT KIND OF FACILITIES CHANGE OWNERSHIP?")

pre <- df %>%
  filter(treated == 1, event_time >= -12, event_time <= -4) %>%
  group_by(cms_certification_number) %>%
  summarise(
    gov_pre   = round(mean(government, na.rm = TRUE)),
    np_pre    = round(mean(non_profit, na.rm = TRUE)),
    chain_pre = round(mean(chain, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(own_type_pre = own_type(gov_pre, np_pre))

cat(sprintf("Treated facilities with usable pre-period ownership data: %d\n\n", nrow(pre)))

subhr("Ownership type BEFORE acquisition")
print(pre %>% count(own_type_pre, name = "n") %>% mutate(pct = round(100 * n / sum(n), 1)))

subhr("Chain status BEFORE acquisition")
print(pre %>% count(chain_pre, name = "n") %>% mutate(pct = round(100 * n / sum(n), 1)))

subhr("Ownership type x Chain status BEFORE acquisition")
print(pre %>% count(own_type_pre, chain_pre, name = "n") %>% arrange(own_type_pre, chain_pre))

# -----------------------------------------------------------------------------
# 2. WHAT DO THEY CHANGE INTO?
# -----------------------------------------------------------------------------
hr("2. WHAT DO THEY CHANGE INTO?")

post <- df %>%
  filter(treated == 1, event_time >= 4, event_time <= 12) %>%
  group_by(cms_certification_number) %>%
  summarise(
    gov_post   = round(mean(government, na.rm = TRUE)),
    np_post    = round(mean(non_profit, na.rm = TRUE)),
    chain_post = round(mean(chain, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(own_type_post = own_type(gov_post, np_post))

transitions <- pre %>%
  inner_join(post, by = "cms_certification_number") %>%
  select(cms_certification_number, own_type_pre, chain_pre, own_type_post, chain_post)

cat(sprintf("Treated facilities with usable pre AND post ownership data: %d\n\n", nrow(transitions)))

subhr("Ownership-type transition matrix (rows = before, cols = after)")
print(table(transitions$own_type_pre, transitions$own_type_post, dnn = c("Before", "After")))

subhr("Chain-status transition matrix (rows = before, cols = after)")
print(table(transitions$chain_pre, transitions$chain_post, dnn = c("Before", "After")))

subhr("Share of facilities whose ownership TYPE changed (e.g. non-profit -> for-profit)")
type_change_pct <- 100 * mean(transitions$own_type_pre != transitions$own_type_post)
cat(sprintf("%.1f%% of treated facilities changed ownership TYPE across their sale\n", type_change_pct))

# -----------------------------------------------------------------------------
# 3. FACILITY CHARACTERISTICS JUST BEFORE THE SALE
# -----------------------------------------------------------------------------
hr("3. FACILITY CHARACTERISTICS JUST BEFORE THE SALE (vs. never-treated facilities)")

char_vars <- c("beds", "occupancy_rate", "spare_capacity", "case_mix_total", "pct_medicare", "pct_medicaid")
char_vars <- char_vars[char_vars %in% names(df)]

pre_chars <- df %>%
  filter(treated == 1, event_time >= -12, event_time <= -4) %>%
  summarise(across(all_of(char_vars), ~ mean(.x, na.rm = TRUE)))

never_chars <- df %>%
  filter(treated == 0) %>%
  summarise(across(all_of(char_vars), ~ mean(.x, na.rm = TRUE)))

char_compare <- bind_rows(
  pre_chars %>% mutate(group = "Treated (pre-sale)", .before = 1),
  never_chars %>% mutate(group = "Never-treated (overall)", .before = 1)
)
print(as.data.frame(char_compare))

# -----------------------------------------------------------------------------
# 4. ARE THEY CLOSE TO STAFFING MINIMUMS PRE-SALE?
# -----------------------------------------------------------------------------
hr("4. STAFFING LEVELS PRE-SALE: HOW CLOSE TO THE LOW END?")

staff_vars <- c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd")
staff_vars <- staff_vars[staff_vars %in% names(df)]

pre_staff <- df %>%
  filter(treated == 1, event_time >= -12, event_time <= -4) %>%
  summarise(across(all_of(staff_vars), ~ mean(.x, na.rm = TRUE)))

never_staff <- df %>%
  filter(treated == 0) %>%
  summarise(across(all_of(staff_vars), ~ mean(.x, na.rm = TRUE)))

staff_compare <- bind_rows(
  pre_staff %>% mutate(group = "Treated (pre-sale)", .before = 1),
  never_staff %>% mutate(group = "Never-treated (overall)", .before = 1)
)
subhr("Average HPRD: treated pre-sale vs. never-treated")
print(as.data.frame(staff_compare))

# Percentile rank of the treated pre-sale average within the FULL national
# distribution of facility-months (a rough "how close to the low end" check).
subhr("Percentile rank of treated facilities' pre-sale average HPRD, within the full national distribution of facility-months")
for (v in staff_vars) {
  full_vals <- df[[v]]
  full_vals <- full_vals[is.finite(full_vals)]
  pre_val <- pre_staff[[v]]
  pctile <- mean(full_vals <= pre_val, na.rm = TRUE) * 100
  cat(sprintf("  %-12s pre-sale avg = %.3f  ->  %.1f percentile of national distribution\n", v, pre_val, pctile))
}

cat("\nNote: CMS's federal minimum staffing rule (2024) sets a total nurse staffing\n")
cat("floor of 3.48 HPRD; comparing the pre-sale total_hprd average above to this\n")
cat("threshold gives a sense of how close to a binding regulatory floor these\n")
cat("facilities are operating before being sold.\n")

# -----------------------------------------------------------------------------
# 5. IS QUALITY ALREADY WORSE PRE-SALE?
# -----------------------------------------------------------------------------
hr("5. QUALITY MEASURES PRE-SALE (vs. never-treated facilities)")

if (is.null(df_quality)) {
  cat("quality_panel.csv could not be loaded -- skipping this section.\n")
} else {
  qm_vars <- names(df_quality)[str_detect(names(df_quality), "^qm_")]
  cat(sprintf("Quality measures found in quality_panel.csv: %s\n\n", paste(qm_vars, collapse = ", ")))

  if (!("event_time" %in% names(df_quality)) || !("treated" %in% names(df_quality))) {
    cat("quality_panel.csv is missing treated/event_time -- skipping this section.\n")
  } else {
    pre_q <- df_quality %>%
      filter(treated == 1, event_time < 0, event_time >= -4) %>%
      summarise(across(all_of(qm_vars), ~ mean(.x, na.rm = TRUE)))

    never_q <- df_quality %>%
      filter(treated == 0) %>%
      summarise(across(all_of(qm_vars), ~ mean(.x, na.rm = TRUE)))

    q_compare <- bind_rows(
      pre_q %>% mutate(group = "Treated (pre-sale, last year)", .before = 1),
      never_q %>% mutate(group = "Never-treated (overall)", .before = 1)
    )
    print(as.data.frame(q_compare))
  }
}

cat("\n", strrep("=", 78), "\n", sep = "")
cat("Done. Review each section above -- this is descriptive only, no causal claims.\n")
cat(strrep("=", 78), "\n", sep = "")
