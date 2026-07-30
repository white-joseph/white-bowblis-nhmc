# =============================================================================
# regressions/goodman_bacon_decomposition.R
#
# Goodman-Bacon decomposition of the standard TWFE post-only coefficient.
#
# Uses a SHIFTED TREATMENT DATE rather than donut-hole row exclusion, because
# bacon() requires a strictly balanced panel -- donut exclusion creates
# facility-specific gaps (each treated facility missing a different set of
# months relative to ITS OWN acquisition date), which is structurally
# incompatible with that requirement. Shifting when `post` switches on
# absorbs the anticipation window into treatment instead of excluding it, so
# no rows get dropped and the panel stays balanced (checked explicitly below,
# since donut-created gaps aren't necessarily the ONLY source of imbalance --
# facility entry/exit is a separate issue, also checked).
#
# ESTIMAND CAVEAT: this does not decompose the exact coefficient in
# twfe_post_full.tex. That table's `post` compares genuinely-untreated
# pre-period to genuinely-post-transition period, with the anticipation
# window dropped entirely. This version pools the anticipation window INTO
# "post" instead. Related estimand, not identical -- say so explicitly if
# this goes near the paper.
#
# Runs TWO versions per outcome: no covariates (the version Goodman-Bacon's
# theorem is proven for exactly) and with the standard control set (matches
# the actual spec, but the clean decomposition guarantee is only exact
# without covariates -- treat the covariate version as an approximation).
#
# Output:
#   outputs/tables/bacon_shifted_raw_{outcome}_{nocov,cov}.csv
#   outputs/tables/bacon_shifted_summary_by_type.csv
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(bacondecomp)
  library(fixest)
  library(dplyr)
  library(tibble)
})

options(scipen = 999, digits = 4)

out_dir <- out_tables_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

SHIFT <- 3L   # matches the donut width (tau in -3,-2,-1)

# -----------------------------------------------------------------------------
# 0) Load, build shifted treatment indicator
# -----------------------------------------------------------------------------
df <- load_staffing_panel() %>%
  mutate(
    post_shifted = case_when(
      is.na(time_treated) ~ 0L,
      time >= (time_treated - SHIFT) ~ 1L,
      TRUE ~ 0L
    )
  )

cat(sprintf("\n[shift] post_shifted switches on at event_time = -%d instead of 0.\n", SHIFT))

controls_rhs <- make_controls_rhs(df)
outs_order <- staffing_outcomes

# -----------------------------------------------------------------------------
# 1) Balance check -- donut-shift fixes ONE source of imbalance; facility
#    entry/exit (openings, closures, reporting gaps) is a SEPARATE issue,
#    checked explicitly rather than assumed away.
# -----------------------------------------------------------------------------
cat("\n", strrep("=", 70), "\nBALANCE CHECK\n", strrep("=", 70), "\n", sep = "")

time_range <- range(df$time, na.rm = TRUE)
all_times <- seq(time_range[1], time_range[2])

facility_coverage <- df %>%
  distinct(cms_certification_number, time) %>%
  count(cms_certification_number, name = "n_periods") %>%
  mutate(complete = n_periods == length(all_times))

n_complete <- sum(facility_coverage$complete)
n_incomplete <- sum(!facility_coverage$complete)

cat(sprintf("Facilities with COMPLETE coverage (%d periods): %s\n", length(all_times), format(n_complete, big.mark = ",")))
cat(sprintf("Facilities with INCOMPLETE coverage: %s\n", format(n_incomplete, big.mark = ",")))

if (n_incomplete > 0) {
  cat(sprintf(
    "\n*** RESTRICTING to the %s facilities with complete coverage. This is a\n*** SEPARATE sample restriction from the donut/shift fix -- likely excludes\n*** facilities that opened, closed, or had reporting gaps mid-panel.\n",
    format(n_complete, big.mark = ",")
  ))
  complete_ccns <- facility_coverage %>% filter(complete) %>% pull(cms_certification_number)
  df_balanced <- df %>% filter(cms_certification_number %in% complete_ccns)
} else {
  cat("\nPanel is already fully balanced.\n")
  df_balanced <- df
}

df_balanced <- df_balanced %>% mutate(cms_certification_number = as.character(cms_certification_number))

# -----------------------------------------------------------------------------
# 2) Sanity check: bacon()'s own weighted average vs. plain feols() coefficient
# -----------------------------------------------------------------------------
cat("\n\n", strrep("=", 70), "\nSANITY CHECK: bacon() vs. feols(), balanced sample\n", strrep("=", 70), "\n", sep = "")

results_nocov <- list()
results_cov   <- list()

for (y in outs_order) {
  if (!(y %in% names(df_balanced)) || all(is.na(df_balanced[[y]]))) next
  dat_y <- df_balanced %>% filter(!is.na(.data[[y]]))

  ref_fml <- as.formula(sprintf("%s ~ post_shifted | cms_certification_number + year_month", y))
  ref_m <- feols(ref_fml, data = dat_y, vcov = ~cms_certification_number + year_month, lean = TRUE)
  ref_b <- unname(coef(ref_m)["post_shifted"])

  bd_nocov <- tryCatch(
    bacon(as.formula(sprintf("%s ~ post_shifted", y)), data = dat_y,
          id_var = "cms_certification_number", time_var = "time", quietly = TRUE),
    error = function(e) { cat(sprintf("  [%s, no covariates] bacon() failed: %s\n", y, conditionMessage(e))); NULL }
  )
  if (!is.null(bd_nocov)) {
    bacon_avg <- weighted.mean(bd_nocov$estimate, bd_nocov$weight)
    cat(sprintf("\n%s: feols() = %.4f | bacon() weighted avg = %.4f  (diff = %.4f)\n", y, ref_b, bacon_avg, ref_b - bacon_avg))
    results_nocov[[y]] <- bd_nocov
  }

  bd_cov <- tryCatch(
    bacon(as.formula(sprintf("%s ~ post_shifted + %s", y, controls_rhs)), data = dat_y,
          id_var = "cms_certification_number", time_var = "time", quietly = TRUE),
    error = function(e) { cat(sprintf("  [%s, with covariates] bacon() failed: %s\n", y, conditionMessage(e))); NULL }
  )
  if (!is.null(bd_cov)) results_cov[[y]] <- bd_cov
}

# -----------------------------------------------------------------------------
# 3) Weight + average estimate by comparison type
# -----------------------------------------------------------------------------
summarize_bacon <- function(bd, label) {
  bd %>%
    group_by(type) %>%
    summarise(total_weight = sum(weight), avg_estimate = weighted.mean(estimate, weight), n_comparisons = n(), .groups = "drop") %>%
    mutate(version = label)
}

cat("\n\n", strrep("=", 70), "\nWEIGHT + AVERAGE ESTIMATE BY COMPARISON TYPE\n", strrep("=", 70), "\n", sep = "")

all_summaries <- list()
for (y in names(results_nocov)) {
  cat("\n---", y, "(no covariates) ---\n")
  s_nocov <- summarize_bacon(results_nocov[[y]], "no_covariates")
  print(s_nocov %>% select(-version))
  all_summaries[[paste0(y, "_nocov")]] <- s_nocov %>% mutate(outcome = y)

  if (!is.null(results_cov[[y]])) {
    cat("\n---", y, "(with covariates -- approximate) ---\n")
    s_cov <- summarize_bacon(results_cov[[y]], "with_covariates")
    print(s_cov %>% select(-version))
    all_summaries[[paste0(y, "_cov")]] <- s_cov %>% mutate(outcome = y)
  }
}

for (y in names(results_nocov)) {
  readr::write_csv(results_nocov[[y]], file.path(out_dir, sprintf("bacon_shifted_raw_%s_nocov.csv", y)))
  if (!is.null(results_cov[[y]])) readr::write_csv(results_cov[[y]], file.path(out_dir, sprintf("bacon_shifted_raw_%s_cov.csv", y)))
}

summary_all <- bind_rows(all_summaries)
readr::write_csv(summary_all, file.path(out_dir, "bacon_shifted_summary_by_type.csv"))

cat("\n\nDone. Read bacon_shifted_summary_by_type.csv: compare avg_estimate for\n")
cat("'Earlier vs Later Treated'/'Later vs Earlier Treated' (contaminated) against\n")
cat("'Treated vs Untreated' (clean). Similar averages and/or low contaminated\n")
cat("weight = naive TWFE wasn't meaningfully biased by staggered timing.\n")

# Save for cross-reference in the CS reconciliation script
saveRDS(list(results_nocov = results_nocov, results_cov = results_cov, summary_all = summary_all),
        file.path(out_dir, "bacon_decomposition_objects.rds"))
