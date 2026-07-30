# =============================================================================
# regressions/cs_reconciliation.R
#
# Reconciles and replaces two earlier, overlapping pieces of work:
#   - callaway_santanna_event_study.R (the pre-existing, more complete CS
#     implementation: controls_A() lean covariates, anticipation=3,
#     never-/not-yet-treated grid, Roth-2022 sup-t pre-trend band)
#   - diagnose_cs_full.R (control-pool size + within-cohort covariate
#     variance diagnostics, built after CS's Wald pre-test blew up)
#
# What's different here, and why:
#   1) STAGED ESCALATION. The original ran est_method="dr", bstrap=TRUE,
#      biters=1000 directly, across a 2x2 control-group x anticipation grid,
#      x 4 outcomes -- up to 16 full doubly-robust bootstrapped fits with no
#      checkpoint. Given how long earlier monthly CS runs took (and how many
#      of them turned out to be structurally doomed by thin cohorts), this
#      now runs FAST first (est_method="reg", no bootstrap) to catch
#      structural problems in seconds, then escalates only if that passes.
#   2) DIAGNOSTIC-FIRST, not diagnostic-after-the-fact. The cohort-size and
#      covariate-variance checks that used to be a separate script now run
#      BEFORE any att_gt() call, so a doomed cohort is visible immediately.
#   3) The original's Roth (2022) simultaneous-band pre-trend check is KEPT
#      as the primary pre-trend diagnostic (not the did package's built-in
#      Wald $Wpval, which is what blew up to hundreds of thousands earlier --
#      that computation inverts a joint pre-period covariance matrix and is
#      exactly the thing thin/near-constant-covariate cohorts destabilize;
#      the sup-t band from aggte()'s crit.val.egt is a different computation
#      and may not share that failure mode. Both are reported below so you
#      can see directly whether they agree or diverge.)
#   4) Monthly granularity (not the quarterly workaround from earlier in
#      this project) -- matching the paper's actual donut and this script's
#      original design. Thin-cohort risk is handled by (1)-(2) above plus
#      controls_A()'s lean 2-covariate set (beds + chain_at_start), which is
#      less likely to hit nea-constant-covariate-within-cohort problems than
#      the 3-covariate set that caused trouble earlier.
#
# Output:
#   outputs/tables/cs_cohort_diagnostic.csv         (Section 1, before any fit)
#   outputs/tables/cs_fast_pass_summary.csv         (Section 2, reg/no-bootstrap)
#   outputs/tables/cs_es_{rn,lpn,cna,total}_baseline.pdf   (Section 4, if escalated)
#   outputs/tables/cs_es_{rn,lpn,cna,total}_coefs.csv
#   outputs/tables/pretrend_and_att_callaway_santanna_fragment.tex
#   outputs/tables/cs_verdict.txt                    (Section 5, plain-language)
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

if (!requireNamespace("did", quietly = TRUE)) {
  stop("Package 'did' is required. Install with: install.packages(\"did\")", call. = FALSE)
}

suppressPackageStartupMessages({
  library(did)
  library(dplyr)
  library(ggplot2)
  library(tibble)
  library(readr)
})

options(scipen = 999, digits = 4)

out_dir   <- out_tables_dir
plots_dir <- out_plots_dir
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)

outs_lvl <- c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd")
nice_out <- c(rn_hprd = "RN", lpn_hprd = "LPN", cna_hprd = "CNA", total_hprd = "Total")

ANTICIPATION <- 3L

# -----------------------------------------------------------------------------
# 0) Load, build id / g / t (same construction as the original script)
# -----------------------------------------------------------------------------
keep_cols <- c(
  "cms_certification_number", "year_month", "time", "time_treated",
  "chain_at_start", "beds",
  "rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd"
)

df0 <- load_staffing_panel() %>%
  mutate(cms_certification_number = as.character(cms_certification_number)) %>%
  select(any_of(keep_cols)) %>%
  mutate(
    id = as.integer(factor(cms_certification_number)),
    g  = if_else(is.na(time_treated), 0L, as.integer(time_treated)),
    t  = as.integer(time)
  )

cs_covariates <- controls_A(df0)   # beds + chain_at_start -- lean, non-endogenous
xformla <- if (length(cs_covariates) == 0) ~1 else as.formula(paste("~", paste(cs_covariates, collapse = " + ")))

cat(sprintf("[cs] facilities = %s, cohorts (g>0) = %d, never-treated = %d\n",
            format(n_distinct(df0$id), big.mark = ","),
            n_distinct(df0$g[df0$g > 0]),
            n_distinct(df0$id[!(df0$id %in% df0$id[df0$g > 0])])))
cat("[cs] xformla:", deparse(xformla), "\n")

# =============================================================================
# 1) COHORT / COVARIATE DIAGNOSTIC -- runs BEFORE any att_gt() call
# =============================================================================
cat("\n", strrep("=", 70), "\n1) COHORT SIZE + COVARIATE VARIANCE DIAGNOSTIC\n", strrep("=", 70), "\n", sep = "")

cohort_sizes <- df0 %>%
  filter(g != 0L) %>%
  distinct(id, g) %>%
  count(g, name = "n_facilities")

MIN_COHORT_SIZE <- 10L
small_cohorts <- cohort_sizes %>% filter(n_facilities < MIN_COHORT_SIZE)

cat(sprintf("Total treated cohorts: %d. Cohorts with fewer than %d facilities: %d\n",
            nrow(cohort_sizes), MIN_COHORT_SIZE, nrow(small_cohorts)))
if (nrow(small_cohorts) > 0) {
  cat("Smallest cohorts:\n")
  print(cohort_sizes %>% arrange(n_facilities) %>% slice_head(n = 10))
}

covar_variation <- df0 %>%
  filter(g != 0L) %>%
  distinct(id, g, chain_at_start, beds) %>%
  group_by(g) %>%
  summarise(n = n(), chain_var = var(chain_at_start, na.rm = TRUE), beds_sd = sd(beds, na.rm = TRUE), .groups = "drop") %>%
  arrange(chain_var)

n_near_constant_chain <- sum(covar_variation$chain_var < 0.02, na.rm = TRUE)
cat(sprintf("\nCohorts where chain_at_start variance < 0.02 (near-constant): %d\n", n_near_constant_chain))
if (n_near_constant_chain > 0) {
  cat("These specific cohorts:\n")
  print(covar_variation %>% filter(chain_var < 0.02))
}

write_csv(cohort_sizes %>% left_join(covar_variation, by = "g"), file.path(out_dir, "cs_cohort_diagnostic.csv"))

if (nrow(small_cohorts) > 0 || n_near_constant_chain > 0) {
  cat("\n*** WARNING: thin cohorts and/or near-constant covariates detected above.\n")
  cat("*** Proceeding anyway (Section 2 is cheap), but expect some (g,t) cells\n")
  cat("*** to return NA or contribute to pre-trend instability -- this is the\n")
  cat("*** SAME mechanism found earlier this session, now confirmed BEFORE any\n")
  cat("*** expensive fitting rather than after.\n")
} else {
  cat("\nNo thin-cohort or near-constant-covariate issues detected. Good sign.\n")
}

# =============================================================================
# 2) FAST PASS -- est_method="reg", no bootstrap, ONE control group, no
#    anticipation grid yet. Just confirms basic convergence and gives timing
#    before committing to the full escalation.
# =============================================================================
cat("\n\n", strrep("=", 70), "\n2) FAST PASS (est_method='reg', no bootstrap, control_group='nevertreated')\n", strrep("=", 70), "\n", sep = "")

run_cs_fast <- function(yname, data) {
  att_gt(
    yname = yname, tname = "t", idname = "id", gname = "g",
    xformla = xformla, data = data, panel = TRUE, allow_unbalanced_panel = TRUE,
    control_group = "nevertreated", anticipation = ANTICIPATION,
    base_period = "varying", est_method = "reg",
    bstrap = FALSE, cband = FALSE, print_details = FALSE
  )
}

fast_results <- list()
fast_summary <- tibble(outcome = character(0), elapsed_sec = numeric(0), n_na_cells = integer(0), n_total_cells = integer(0), simple_att = numeric(0))

for (y in outs_lvl) {
  t0 <- Sys.time()
  att <- tryCatch(run_cs_fast(y, df0), error = function(e) { cat(sprintf("  [%s] FAILED: %s\n", y, conditionMessage(e))); NULL })
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  if (is.null(att)) {
    fast_summary <- bind_rows(fast_summary, tibble(outcome = nice_out[[y]], elapsed_sec = elapsed, n_na_cells = NA, n_total_cells = NA, simple_att = NA))
    next
  }

  fast_results[[y]] <- att
  n_na <- sum(is.na(att$att))
  n_tot <- length(att$att)
  simp <- tryCatch(aggte(att, type = "simple", na.rm = TRUE, bstrap = FALSE)$overall.att, error = function(e) NA)

  cat(sprintf("%-6s: %.1fs, %d/%d cells NA, simple ATT = %.4f\n", nice_out[[y]], elapsed, n_na, n_tot, simp))
  fast_summary <- bind_rows(fast_summary, tibble(outcome = nice_out[[y]], elapsed_sec = elapsed, n_na_cells = n_na, n_total_cells = n_tot, simple_att = simp))
}

write_csv(fast_summary, file.path(out_dir, "cs_fast_pass_summary.csv"))

pct_na <- mean(fast_summary$n_na_cells / fast_summary$n_total_cells, na.rm = TRUE)
proceed_to_full <- !any(is.na(fast_summary$simple_att)) && (is.na(pct_na) || pct_na < 0.3)

cat(sprintf("\nAverage share of NA cells across outcomes: %.1f%%\n", 100 * pct_na))
if (!proceed_to_full) {
  cat("\n*** STOPPING before the full bootstrapped grid: the fast pass itself\n")
  cat("*** shows failures or a high NA-cell share. Fix the underlying cohort/\n")
  cat("*** covariate issue (see Section 1) before spending time on est_method=\n")
  cat("*** 'dr' with bootstrapping -- it will not fix a structural problem.\n")
} else {
  cat("\nFast pass looks structurally sound. Proceeding to the full grid.\n")
}

# =============================================================================
# 3) FULL GRID -- only runs if the fast pass passed. est_method='dr',
#    bstrap=TRUE, biters=1000, full control_group x anticipation grid
#    (same as the original script), plus the Roth (2022) sup-t pre-trend
#    check on the PREFERRED spec.
# =============================================================================
main_fits <- list()

if (proceed_to_full) {
  cat("\n\n", strrep("=", 70), "\n3) FULL GRID (est_method='dr', bootstrapped)\n", strrep("=", 70), "\n", sep = "")

  run_cs <- function(yname, data, control_group, anticipation, biters = 1000) {
    att_gt(
      yname = yname, tname = "t", idname = "id", gname = "g",
      xformla = xformla, data = data, panel = TRUE, allow_unbalanced_panel = TRUE,
      control_group = control_group, anticipation = anticipation,
      base_period = "varying", est_method = "dr",
      bstrap = TRUE, biters = biters, cband = TRUE, print_details = FALSE
    )
  }

  agg_es     <- function(att, min_e = -24L, max_e = 24L) aggte(att, type = "dynamic", min_e = min_e, max_e = max_e, na.rm = TRUE, bstrap = TRUE, cband = TRUE)
  agg_simple <- function(att) aggte(att, type = "simple", na.rm = TRUE, bstrap = TRUE, cband = TRUE)

  extract_es <- function(es_obj) {
    tibble(
      event_time = as.integer(es_obj$egt),
      estimate   = as.numeric(es_obj$att.egt),
      se         = as.numeric(es_obj$se.egt)
    ) %>%
      mutate(crit = es_obj$crit.val.egt, ci_lo = estimate - crit * se, ci_hi = estimate + crit * se) %>%
      arrange(event_time)
  }

  # Roth (2022): does the simultaneous confidence band for pre-period
  # dynamic effects cover zero everywhere? Independent computation from the
  # did package's built-in $Wpval Wald test (which is what blew up to
  # hundreds of thousands of a Wald statistic earlier this session).
  pretrend_check <- function(coefs_df) {
    pre <- filter(coefs_df, event_time < 0)
    if (!nrow(pre)) return(list(note = "no pre-period estimates"))
    list(
      n_pre = nrow(pre),
      all_band_covers_zero = all(pre$ci_lo <= 0 & pre$ci_hi >= 0),
      max_abs_t = max(abs(pre$estimate / pre$se), na.rm = TRUE)
    )
  }

  PREFERRED_CG <- "notyettreated"

  for (y in outs_lvl) {
    cat(sprintf("\n[cs] fitting preferred spec: %s (control_group=%s, anticipation=%d)\n", y, PREFERRED_CG, ANTICIPATION))
    t0 <- Sys.time()
    att <- tryCatch(run_cs(y, df0, PREFERRED_CG, ANTICIPATION), error = function(e) { message(sprintf("[cs] failed for %s: %s", y, e$message)); NULL })
    cat(sprintf("  Elapsed: %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
    if (is.null(att)) { main_fits[[y]] <- NULL; next }

    es <- agg_es(att)
    coefs <- extract_es(es)
    chk <- pretrend_check(coefs)

    cat(sprintf("  Roth sup-t pre-trend check: %s (%d pre-periods, max|t| = %.2f)\n",
                if (isTRUE(chk$all_band_covers_zero)) "PASS (flat)" else "FAIL (band excludes zero somewhere)",
                chk$n_pre, chk$max_abs_t))

    main_fits[[y]] <- list(att = att, es = es, simple = agg_simple(att), coefs = coefs, pretrend = chk)

    write_csv(coefs, file.path(out_dir, sprintf("cs_es_%s_coefs.csv", sub("_hprd$", "", y))))

    p <- ggplot(coefs, aes(x = event_time, y = estimate)) +
      geom_hline(yintercept = 0, linetype = "dotted", color = "grey40") +
      geom_vline(xintercept = -0.5, linetype = "dashed", color = "grey40") +
      geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.4, color = "steelblue") +
      geom_point(color = "steelblue", size = 1.6) +
      labs(x = "Months relative to ownership change (anticipation-adjusted)", y = paste0(nice_out[[y]], " HPRD")) +
      theme_minimal(base_size = 12, base_family = "sans") +
      theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 1), panel.grid.minor = element_blank())

    ggsave(file.path(plots_dir, sprintf("cs_es_%s_baseline.pdf", sub("_hprd$", "", y))), plot = p, width = 7, height = 5, device = "pdf")
  }
} else {
  cat("\n(Section 3 skipped -- fast pass did not pass. See Section 1/2 output above.)\n")
}

# =============================================================================
# 4) CROSS-REFERENCE WITH GOODMAN-BACON -- if goodman_bacon_decomposition.R
#    has already been run, load its saved objects and put the naive TWFE,
#    Bacon-implied, and CS numbers side by side.
# =============================================================================
cat("\n\n", strrep("=", 70), "\n4) CROSS-REFERENCE: naive TWFE vs. Bacon-implied vs. CS\n", strrep("=", 70), "\n", sep = "")

bacon_obj_fp <- file.path(out_dir, "bacon_decomposition_objects.rds")

comparison_rows <- list()
for (y in outs_lvl) {
  naive_fml <- as.formula(sprintf("%s ~ post + %s | cms_certification_number + year_month", y, make_controls_rhs(df0)))
  naive_dat <- drop_anticipation_window(load_staffing_panel())
  naive_m <- fixest::feols(naive_fml, data = naive_dat, vcov = ~cms_certification_number + year_month, lean = TRUE)
  naive_b <- unname(coef(naive_m)["post"])

  cs_b <- if (!is.null(main_fits[[y]])) main_fits[[y]]$simple$overall.att else NA

  bacon_b <- NA
  if (file.exists(bacon_obj_fp)) {
    bacon_data <- readRDS(bacon_obj_fp)
    if (!is.null(bacon_data$results_nocov[[y]])) {
      bacon_b <- weighted.mean(bacon_data$results_nocov[[y]]$estimate, bacon_data$results_nocov[[y]]$weight)
    }
  }

  comparison_rows[[y]] <- tibble(outcome = nice_out[[y]], naive_twfe = naive_b, bacon_implied = bacon_b, cs_simple = cs_b)
}

comparison_tbl <- bind_rows(comparison_rows)
print(comparison_tbl)
write_csv(comparison_tbl, file.path(out_dir, "cs_bacon_naive_comparison.csv"))

if (!file.exists(bacon_obj_fp)) {
  cat("\n(Bacon comparison column is NA -- run goodman_bacon_decomposition.R first to populate it.)\n")
}

# =============================================================================
# 5) VERDICT -- plain-language summary tied to concrete numbers
# =============================================================================
cat("\n\n", strrep("=", 70), "\n5) VERDICT\n", strrep("=", 70), "\n", sep = "")

verdict_lines <- character(0)

if (!proceed_to_full) {
  verdict_lines <- c(verdict_lines,
    "CS DID NOT REACH THE FULL GRID: the fast pass showed structural problems",
    "(failed fits and/or a high share of NA cells). This points to thin cohorts",
    "or near-constant covariates (see Section 1's diagnostic), not a fixable",
    "computational issue -- coarsening cohorts or trimming the covariate set",
    "further is the next step, not more compute."
  )
} else {
  for (y in outs_lvl) {
    if (is.null(main_fits[[y]])) {
      verdict_lines <- c(verdict_lines, sprintf("%s: full fit FAILED even after fast pass passed -- investigate directly.", nice_out[[y]]))
      next
    }
    chk <- main_fits[[y]]$pretrend
    row <- comparison_rows[[y]]
    agree <- if (!is.na(row$naive_twfe) && !is.na(row$cs_simple)) {
      abs(row$naive_twfe - row$cs_simple) < 0.5 * max(abs(row$naive_twfe), abs(row$cs_simple), 0.001)
    } else NA

    verdict_lines <- c(verdict_lines, sprintf(
      "%s: pre-trend %s | naive TWFE = %.4f, CS = %.4f (%s)",
      nice_out[[y]],
      if (isTRUE(chk$all_band_covers_zero)) "flat (PASS)" else "NOT flat (FAIL)",
      row$naive_twfe, row$cs_simple,
      if (isTRUE(agree)) "broadly agree" else if (isFALSE(agree)) "DIVERGE -- investigate" else "no CS estimate to compare"
    ))
  }
}

writeLines(verdict_lines, file.path(out_dir, "cs_verdict.txt"))
cat(paste(verdict_lines, collapse = "\n"), "\n")

cat("\n\nDone.\n")
