# regressions/diagnose_cs_full.R
#
# Comprehensive diagnostic: inspects the ACTUAL DATA (not just the fitted
# att_gt() output) to find why pre-test Wald statistics are absurdly large
# at both monthly and quarterly granularity. Checks, in order:
#   1) Control-group size available to each cohort at each relevant period
#   2) Covariate variation WITHIN each cohort's estimation sample (near-
#      constant covariates in a subsample cause near-singular/degenerate
#      variance even if not fully collinear)
#   3) Raw att_gt() cell-level ATT/SE, sorted by most extreme t-stats
#   4) Cross-references (1)+(2) against (3) to see if the worst cells line
#      up with thin control pools or near-constant covariates

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")
suppressPackageStartupMessages({ library(did); library(dplyr); library(tibble) })

out_dir <- out_tables_dir
outs_lvl <- c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd")
nice_out <- c(rn_hprd = "RN", lpn_hprd = "LPN", cna_hprd = "CNA", total_hprd = "Total")

GRANULARITY <- "quarterly"   # run once as "quarterly", then again as "monthly"

# -----------------------------------------------------------------------------
# 0) Rebuild the exact estimation sample used (same construction as before)
# -----------------------------------------------------------------------------
df0 <- load_staffing_panel() %>%
  mutate(id_num = as.integer(factor(cms_certification_number)))

make_baseline <- function(data, var, new_name) {
  lookup <- data %>%
    dplyr::arrange(cms_certification_number, ym_date) %>%
    dplyr::group_by(cms_certification_number) %>%
    dplyr::summarise(!!new_name := .data[[var]][!is.na(.data[[var]])][1], .groups = "drop")
  data %>% dplyr::left_join(lookup, by = "cms_certification_number")
}

df0 <- df0 %>%
  make_baseline("non_profit", "non_profit_at_start") %>%
  make_baseline("beds", "beds_at_start")

if (GRANULARITY == "quarterly") {
  df0 <- df0 %>%
    mutate(
      time_use  = ceiling(time / 3),
      g_use_raw = ifelse(is.na(time_treated), NA_integer_, ceiling(time_treated / 3))
    )
} else {
  df0 <- df0 %>% mutate(time_use = time, g_use_raw = time_treated)
}

df0 <- df0 %>% mutate(g_use = ifelse(is.na(g_use_raw), 0L, as.integer(g_use_raw)))

MIN_COHORT_SIZE <- 10L
cohort_sizes <- df0 %>%
  filter(g_use != 0L) %>%
  distinct(cms_certification_number, g_use) %>%
  count(g_use, name = "n_facilities")
small_cohorts <- cohort_sizes %>% filter(n_facilities < MIN_COHORT_SIZE) %>% pull(g_use)
drop_ccns <- df0 %>% filter(g_use %in% small_cohorts) %>% distinct(cms_certification_number) %>% pull(cms_certification_number)
df_use <- df0 %>% filter(!(cms_certification_number %in% drop_ccns))

cat(sprintf("\n[%s] Facilities in estimation sample: %d. Cohorts: %d.\n",
            GRANULARITY, dplyr::n_distinct(df_use$cms_certification_number),
            dplyr::n_distinct(df_use$g_use[df_use$g_use != 0])))

# -----------------------------------------------------------------------------
# 1) Control-group size available to EACH cohort at EACH period.
#    For a given cohort g, the "control pool" at period t is: never-treated
#    facilities (g_use==0) PLUS not-yet-treated facilities (g_use > t).
#    If this is small for some (g,t), that cell's variance estimate will be
#    unstable regardless of overall sample size.
# -----------------------------------------------------------------------------
cat("\n", strrep("=", 70), "\n1) CONTROL POOL SIZE BY (g,t) -- smallest 20 cells\n", strrep("=", 70), "\n", sep = "")

all_g <- sort(unique(df_use$g_use[df_use$g_use != 0]))
all_t <- sort(unique(df_use$time_use))

control_pool_check <- expand.grid(g = all_g, t = all_t) %>%
  as_tibble() %>%
  rowwise() %>%
  mutate(
    n_treated_g_at_t = sum(df_use$g_use == g & df_use$time_use == t),
    n_control_at_t   = sum((df_use$g_use == 0 | df_use$g_use > t) & df_use$time_use == t)
  ) %>%
  ungroup() %>%
  filter(n_treated_g_at_t > 0)   # only cells that actually get evaluated

cat("Smallest treated-group sizes at their own evaluated periods:\n")
print(control_pool_check %>% arrange(n_treated_g_at_t) %>% slice_head(n = 20))

cat("\nSmallest control-pool sizes across all evaluated cells:\n")
print(control_pool_check %>% arrange(n_control_at_t) %>% slice_head(n = 20))

readr::write_csv(control_pool_check, file.path(out_dir, sprintf("diag_control_pool_%s.csv", GRANULARITY)))

# -----------------------------------------------------------------------------
# 2) Covariate variation WITHIN each cohort's OWN treated group.
#    A cohort where ALL treated facilities share the same chain_at_start /
#    non_profit_at_start value (e.g., all chain=0) makes that covariate
#    contribute nothing but noise to that cohort's regression -- not fully
#    collinear (so it won't be silently dropped), but can still destabilize
#    the variance estimate for that cell.
# -----------------------------------------------------------------------------
cat("\n\n", strrep("=", 70), "\n2) COVARIATE VARIATION WITHIN EACH TREATED COHORT\n", strrep("=", 70), "\n", sep = "")

covar_variation <- df_use %>%
  filter(g_use != 0L) %>%
  distinct(cms_certification_number, g_use, chain_at_start, non_profit_at_start, beds_at_start) %>%
  group_by(g_use) %>%
  summarise(
    n = n(),
    chain_var       = var(chain_at_start, na.rm = TRUE),
    non_profit_var  = var(non_profit_at_start, na.rm = TRUE),
    beds_sd         = sd(beds_at_start, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(chain_var)   # near-zero variance = potential problem cohort

cat("Cohorts with LOWEST chain_at_start variance (near-constant within cohort):\n")
print(covar_variation %>% slice_head(n = 10))

cat("\nCohorts with LOWEST non_profit_at_start variance:\n")
print(covar_variation %>% arrange(non_profit_var) %>% slice_head(n = 10))

readr::write_csv(covar_variation, file.path(out_dir, sprintf("diag_covar_variation_%s.csv", GRANULARITY)))

# -----------------------------------------------------------------------------
# 3) Raw att_gt() cell-level results -- requires the saved .rds from the
#    corresponding script (monthly or quarterly version)
# -----------------------------------------------------------------------------
cat("\n\n", strrep("=", 70), "\n3) RAW att_gt() CELL RESULTS -- most extreme t-stats\n", strrep("=", 70), "\n", sep = "")

cell_results <- list()
for (y in outs_lvl) {
  fname <- if (GRANULARITY == "monthly") sprintf("cs_attgt_monthly_%s.rds", sub("_hprd$", "", y))
           else sprintf("cs_attgt_%s.rds", sub("_hprd$", "", y))
  fp <- file.path(out_dir, fname)
  if (!file.exists(fp)) { cat(sprintf("\n%s: %s not found, skipping.\n", nice_out[[y]], fname)); next }

  res <- readRDS(fp)
  cell_tbl <- tibble(g = res$group, t = res$t, att = res$att, se = res$se) %>%
    mutate(t_stat = att / se, abs_t = abs(t_stat)) %>%
    filter(!is.na(att), !is.na(se))

  cell_results[[y]] <- cell_tbl

  cat("\n---", nice_out[[y]], "---\n")
  print(cell_tbl %>% arrange(desc(abs_t)) %>% slice_head(n = 8))
  cat(sprintf("SE range: [%.6f, %.4f]. Cells with SE < 0.0001: %d of %d.\n",
              min(cell_tbl$se, na.rm = TRUE), max(cell_tbl$se, na.rm = TRUE),
              sum(cell_tbl$se < 1e-4, na.rm = TRUE), nrow(cell_tbl)))
}

# -----------------------------------------------------------------------------
# 4) Cross-reference: for RN specifically, join the worst cells against
#    control-pool size and covariate variation for that cohort
# -----------------------------------------------------------------------------
if (!is.null(cell_results[["rn_hprd"]])) {
  cat("\n\n", strrep("=", 70), "\n4) CROSS-REFERENCE (RN): worst cells vs. control pool size + covariate variation\n", strrep("=", 70), "\n", sep = "")

  worst_cells <- cell_results[["rn_hprd"]] %>% arrange(desc(abs_t)) %>% slice_head(n = 10)

  cross_check <- worst_cells %>%
    left_join(control_pool_check, by = c("g", "t")) %>%
    left_join(covar_variation, by = c("g" = "g_use"))

  print(cross_check %>% select(g, t, att, se, t_stat, n_treated_g_at_t, n_control_at_t, chain_var, non_profit_var, beds_sd))
}

cat("\n\nDone. Run this once with GRANULARITY <- \"quarterly\", then again with \"monthly\", and paste both outputs.\n")
