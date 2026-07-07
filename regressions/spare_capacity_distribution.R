# =============================================================================
# regressions/spare_capacity_distribution.R
#
# Purpose:
#   Inspect the distribution of spare_capacity in the monthly staffing panel:
#     - overall summary stats
#     - by year
#     - by treated vs. never-treated facilities
#     - histogram / density / boxplot
#
# Output:
#   outputs/tables/spare_capacity_summary.csv
#   outputs/plots/spare_capacity_hist.png
#   outputs/plots/spare_capacity_density_by_treated.png
#   outputs/plots/spare_capacity_box_by_year.png
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(ggplot2)
})

options(scipen = 999, digits = 4)

# -----------------------------------------------------------------------------
# Load panel
# -----------------------------------------------------------------------------
df <- load_staffing_panel()

if (!("year" %in% names(df))) {
  df <- df %>%
    mutate(year = as.integer(str_sub(as.character(year_month), 1, 4)))
}

stopifnot("spare_capacity" %in% names(df))

df <- df %>%
  mutate(spare_capacity = suppressWarnings(as.numeric(spare_capacity)))

# -----------------------------------------------------------------------------
# Summary stats: overall
# -----------------------------------------------------------------------------
summarize_sc <- function(x) {
  x <- x[is.finite(x)]
  tibble::tibble(
    n        = length(x),
    n_na     = sum(!is.finite(x)),
    mean     = mean(x, na.rm = TRUE),
    sd       = sd(x, na.rm = TRUE),
    min      = min(x, na.rm = TRUE),
    p10      = quantile(x, 0.10, na.rm = TRUE),
    p25      = quantile(x, 0.25, na.rm = TRUE),
    median   = median(x, na.rm = TRUE),
    p75      = quantile(x, 0.75, na.rm = TRUE),
    p90      = quantile(x, 0.90, na.rm = TRUE),
    max      = max(x, na.rm = TRUE)
  )
}

overall_summary <- summarize_sc(df$spare_capacity) %>%
  mutate(group = "overall", .before = 1)

# -----------------------------------------------------------------------------
# Summary stats: by year
# -----------------------------------------------------------------------------
by_year_summary <- df %>%
  group_by(year) %>%
  summarize(
    n      = sum(is.finite(spare_capacity)),
    mean   = mean(spare_capacity, na.rm = TRUE),
    sd     = sd(spare_capacity, na.rm = TRUE),
    median = median(spare_capacity, na.rm = TRUE),
    p10    = quantile(spare_capacity, 0.10, na.rm = TRUE),
    p90    = quantile(spare_capacity, 0.90, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  rename(group = year) %>%
  mutate(group = as.character(group))

# -----------------------------------------------------------------------------
# Summary stats: by treated vs. never-treated
# (ever_treated = 1 if the facility is in the validated CHOW-treated sample)
# -----------------------------------------------------------------------------
df <- df %>%
  group_by(cms_certification_number) %>%
  mutate(ever_treated = as.integer(any(treated == 1, na.rm = TRUE))) %>%
  ungroup()

by_treated_summary <- df %>%
  group_by(ever_treated) %>%
  summarize(
    n      = sum(is.finite(spare_capacity)),
    mean   = mean(spare_capacity, na.rm = TRUE),
    sd     = sd(spare_capacity, na.rm = TRUE),
    median = median(spare_capacity, na.rm = TRUE),
    p10    = quantile(spare_capacity, 0.10, na.rm = TRUE),
    p90    = quantile(spare_capacity, 0.90, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    group = ifelse(ever_treated == 1, "ever_treated", "never_treated"),
    .before = 1
  ) %>%
  select(-ever_treated)

# -----------------------------------------------------------------------------
# Combine and write summary table
# -----------------------------------------------------------------------------
summary_out <- bind_rows(
  overall_summary,
  by_year_summary,
  by_treated_summary
)

summary_fp <- file.path(out_tables_dir, "spare_capacity_summary.csv")
readr::write_csv(summary_out, summary_fp)

cat("Wrote summary table to:\n", summary_fp, "\n\n")
print(overall_summary)
print(by_treated_summary)

# -----------------------------------------------------------------------------
# Plot 1: Histogram (overall)
# -----------------------------------------------------------------------------
p_hist <- ggplot(df %>% filter(is.finite(spare_capacity)), aes(x = spare_capacity)) +
  geom_histogram(bins = 60, fill = "steelblue", color = "white", boundary = 0) +
  labs(
    title = "Distribution of Spare Capacity",
    subtitle = "(Certified beds \u2212 average census) / certified beds",
    x = "Spare capacity",
    y = "Count (facility-months)"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  filename = file.path(out_plots_dir, "spare_capacity_hist.png"),
  plot = p_hist, width = 7, height = 5, dpi = 300
)

# -----------------------------------------------------------------------------
# Plot 2: Density, ever-treated vs. never-treated
# -----------------------------------------------------------------------------
p_density <- df %>%
  filter(is.finite(spare_capacity)) %>%
  mutate(group_lbl = ifelse(ever_treated == 1, "Ever treated (CHOW)", "Never treated")) %>%
  ggplot(aes(x = spare_capacity, fill = group_lbl, color = group_lbl)) +
  geom_density(alpha = 0.35) +
  labs(
    title = "Spare Capacity: Ever-Treated vs. Never-Treated Facilities",
    x = "Spare capacity",
    y = "Density",
    fill = NULL, color = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(
  filename = file.path(out_plots_dir, "spare_capacity_density_by_treated.png"),
  plot = p_density, width = 7, height = 5, dpi = 300
)

# -----------------------------------------------------------------------------
# Plot 3: Boxplot by year
# -----------------------------------------------------------------------------
p_box <- df %>%
  filter(is.finite(spare_capacity), !is.na(year)) %>%
  ggplot(aes(x = factor(year), y = spare_capacity)) +
  geom_boxplot(fill = "steelblue", alpha = 0.5, outlier.size = 0.5) +
  labs(
    title = "Spare Capacity by Year",
    x = "Year",
    y = "Spare capacity"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(
  filename = file.path(out_plots_dir, "spare_capacity_box_by_year.png"),
  plot = p_box, width = 8, height = 5, dpi = 300
)

cat("\nSaved plots to:\n")
cat(file.path(out_plots_dir, "spare_capacity_hist.png"), "\n")
cat(file.path(out_plots_dir, "spare_capacity_density_by_treated.png"), "\n")
cat(file.path(out_plots_dir, "spare_capacity_box_by_year.png"), "\n")
