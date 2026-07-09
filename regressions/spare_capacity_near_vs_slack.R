# =============================================================================
# regressions/spare_capacity_near_vs_slack.R
#
# Purpose:
#   Test whether SPARE CAPACITY ITSELF evolves differently after ownership
#   change (CHOW) for facilities that were acquired NEAR CAPACITY (little
#   slack) versus those acquired with more SLACK CAPACITY.
#
#   Motivation: if new owners re-optimize occupancy toward some target/
#   production-frontier level, slack-capacity facilities have room to fill
#   beds post-acquisition (spare capacity should fall), while near-capacity
#   facilities are already close to the frontier and have little room left
#   to move.
#
# Classification (DEFAULT ASSUMPTION -- flag to advisor, easy to change):
#   - Uses each treated facility's average spare_capacity over
#     event_time in [-12, -4], i.e., roughly 4-12 months before the
#     ownership change, deliberately BEFORE the anticipation window
#     (event_time in {-3,-2,-1}) that is excluded elsewhere.
#   - Facilities are split at the MEDIAN of this pre-acquisition average,
#     among treated facilities with a usable (non-missing) baseline:
#       near_capacity = 1 if baseline spare_capacity <= median (less slack)
#       near_capacity = 0 if baseline spare_capacity >  median (more slack)
#
# Sample:
#   Restricted to EVER-TREATED (CHOW) facilities only -- this compares
#   treated-near-capacity vs. treated-slack-capacity to EACH OTHER.
#
# NOTE: The static post-effect regression table for this split now lives in
# spare_capacity_report.R (combined with the distribution and summary
# stats). This script covers the EVENT-STUDY / dynamic piece only -- the
# before/after trajectory of spare_capacity for the two groups.
#
# Output:
#   outputs/tables/spare_capacity_near_vs_slack_group_summary.csv
#   outputs/plots/spare_capacity_near_vs_slack_density.png
#   outputs/plots/spare_capacity_near_vs_slack_event_study.png
#   outputs/tables/spare_capacity_near_vs_slack_event_study_coefs.csv
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(tibble)
  library(readr)
  library(ggplot2)
})

options(scipen = 999, digits = 4)

# -----------------------------------------------------------------------------
# Load panel
# -----------------------------------------------------------------------------
df <- load_staffing_panel()
stopifnot(all(c("spare_capacity", "occupancy_rate") %in% names(df)))

# -----------------------------------------------------------------------------
# Baseline classification: near-capacity vs. slack-capacity at acquisition
# -----------------------------------------------------------------------------
baseline_window <- df %>%
  filter(treated == 1, event_time >= -12, event_time <= -4) %>%
  group_by(cms_certification_number) %>%
  summarise(
    baseline_spare_capacity = mean(spare_capacity, na.rm = TRUE),
    n_baseline_months = sum(!is.na(spare_capacity)),
    .groups = "drop"
  ) %>%
  filter(n_baseline_months > 0, is.finite(baseline_spare_capacity))

baseline_median <- median(baseline_window$baseline_spare_capacity, na.rm = TRUE)

baseline_window <- baseline_window %>%
  mutate(
    near_capacity = as.integer(baseline_spare_capacity <= baseline_median)
  )

cat(sprintf(
  "[classification] treated facilities with usable baseline (event_time -12 to -4): %d\n",
  nrow(baseline_window)
))
cat(sprintf("[classification] median baseline spare_capacity: %.3f\n", baseline_median))
cat(sprintf(
  "[classification] near-capacity (<= median): %d facilities; slack-capacity (> median): %d facilities\n",
  sum(baseline_window$near_capacity == 1),
  sum(baseline_window$near_capacity == 0)
))

# -----------------------------------------------------------------------------
# Build analysis sample: treated facilities only, with classification attached
# -----------------------------------------------------------------------------
df_treated <- df %>%
  filter(treated == 1) %>%
  inner_join(
    baseline_window %>%
      select(cms_certification_number, baseline_spare_capacity, near_capacity),
    by = "cms_certification_number"
  )

df_treated_static <- drop_anticipation_window(df_treated)

cat(sprintf(
  "[sample] facility-months (anticipation excluded): %s (%d facilities)\n",
  format(nrow(df_treated_static), big.mark = ","),
  n_distinct(df_treated_static$cms_certification_number)
))

# -----------------------------------------------------------------------------
# Descriptive: distribution of spare_capacity by group (sanity check + report)
# -----------------------------------------------------------------------------
group_summary <- df_treated_static %>%
  group_by(near_capacity) %>%
  summarise(
    n      = sum(is.finite(spare_capacity)),
    mean   = mean(spare_capacity, na.rm = TRUE),
    sd     = sd(spare_capacity, na.rm = TRUE),
    median = median(spare_capacity, na.rm = TRUE),
    p10    = quantile(spare_capacity, 0.10, na.rm = TRUE),
    p90    = quantile(spare_capacity, 0.90, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(group = ifelse(near_capacity == 1, "Near capacity", "Slack capacity"), .before = 1) %>%
  select(-near_capacity)

group_summary_fp <- file.path(out_tables_dir, "spare_capacity_near_vs_slack_group_summary.csv")
readr::write_csv(group_summary, group_summary_fp)
print(group_summary)

p_density <- df_treated_static %>%
  filter(is.finite(spare_capacity)) %>%
  mutate(group_lbl = ifelse(near_capacity == 1, "Near capacity", "Slack capacity")) %>%
  ggplot(aes(x = spare_capacity, fill = group_lbl, color = group_lbl)) +
  geom_density(alpha = 0.35) +
  geom_vline(xintercept = baseline_median, linetype = "dashed") +
  labs(
    title = "Spare Capacity: Near-Capacity vs. Slack-Capacity Acquisitions",
    subtitle = sprintf("Split at median pre-acquisition spare capacity (%.3f)", baseline_median),
    x = "Spare capacity (full observed window, treated facilities)",
    y = "Density",
    fill = NULL, color = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

density_plot_fp <- file.path(out_plots_dir, "spare_capacity_near_vs_slack_density.png")
ggsave(density_plot_fp, plot = p_density, width = 7, height = 5, dpi = 300)

# -----------------------------------------------------------------------------
# Shared regression setup (used by the event study below)
# -----------------------------------------------------------------------------

vc <- as.formula(paste0("~ ", fe_unit, " + ", fe_time))
fe_part <- paste0("| ", fe_unit, " + ", fe_time)

# Controls, excluding the outcome itself (and its close cousin) so we don't
# control away the variation we're trying to explain.
outcome_related_excludes <- list(
  spare_capacity = c("occupancy_rate", "spare_capacity"),
  occupancy_rate = c("occupancy_rate", "spare_capacity")
)

controls_rhs_for <- function(dat, outcome) {
  ctrls <- get_controls(dat)
  ctrls <- setdiff(ctrls, outcome_related_excludes[[outcome]])
  if (length(ctrls) == 0) return("1")
  paste(ctrls, collapse = " + ")
}

# =============================================================================
# Event-study: dynamic path of spare_capacity by group
# =============================================================================
# Consistent with the rest of the project's event studies: the anticipation
# window (event_time in {-3,-2,-1}) is DROPPED, and event_time = -4 is used
# as the reference period (rather than -1).

min_et <- -18L
max_et <- 18L
ref_et <- -4L

prep_et <- function(dat) {
  dat <- drop_anticipation_window(dat)
  dat %>%
    mutate(event_time_capped = pmin(pmax(as.integer(event_time), min_et), max_et))
}

run_event_study_group <- function(dat) {
  dat <- prep_et(dat)
  ctrls <- controls_rhs_for(dat, "spare_capacity")
  rhs_ctrl_part <- if (ctrls == "1") "" else paste0(" + ", ctrls)
  fml <- as.formula(sprintf(
    "spare_capacity ~ i(event_time_capped, ref = %d, keep = %d:%d)%s %s",
    ref_et, min_et, max_et, rhs_ctrl_part, fe_part
  ))
  feols(fml, data = dat, vcov = vc, lean = TRUE)
}

df_near  <- df_treated %>% filter(near_capacity == 1)
df_slack <- df_treated %>% filter(near_capacity == 0)

mod_near  <- run_event_study_group(df_near)
mod_slack <- run_event_study_group(df_slack)

extract_event_coefs <- function(mod, group_label) {
  ct <- summary(mod)$coeftable
  terms <- rownames(ct)
  keep <- grepl("^event_time_capped::", terms)
  et <- as.integer(sub("^event_time_capped::(-?[0-9]+).*$", "\\1", terms[keep]))
  tibble(
    event_time = et,
    estimate = unname(ct[keep, "Estimate"]),
    se = unname(ct[keep, "Std. Error"]),
    group = group_label
  )
}

event_coefs <- bind_rows(
  extract_event_coefs(mod_near, "Near capacity"),
  extract_event_coefs(mod_slack, "Slack capacity"),
  tibble(event_time = ref_et, estimate = 0, se = 0, group = "Near capacity"),
  tibble(event_time = ref_et, estimate = 0, se = 0, group = "Slack capacity")
) %>%
  mutate(
    ci_lo = estimate - 1.96 * se,
    ci_hi = estimate + 1.96 * se
  ) %>%
  arrange(group, event_time)

event_coefs_fp <- file.path(out_tables_dir, "spare_capacity_near_vs_slack_event_study_coefs.csv")
readr::write_csv(event_coefs, event_coefs_fp)

p_event <- ggplot(event_coefs, aes(x = event_time, y = estimate, color = group, fill = group)) +
  geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  geom_hline(yintercept = 0, linetype = "dotted") +
  geom_vline(xintercept = -0.5, linetype = "dashed") +
  labs(
    title = "Spare Capacity Around Ownership Change: Near-Capacity vs. Slack-Capacity",
    subtitle = sprintf("Reference period: event_time = %d (anticipation window excluded)", ref_et),
    x = "Months relative to ownership change (event_time)",
    y = "Change in spare capacity (relative to reference period)",
    color = NULL, fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

event_plot_fp <- file.path(out_plots_dir, "spare_capacity_near_vs_slack_event_study.png")
ggsave(event_plot_fp, plot = p_event, width = 8, height = 5.5, dpi = 300)

# -----------------------------------------------------------------------------
# Console summary
# -----------------------------------------------------------------------------
cat("\n[write] ", normalizePath(group_summary_fp, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(density_plot_fp, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(event_coefs_fp, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(event_plot_fp, winslash = "\\"), "\n", sep = "")
cat("Done.\n")
