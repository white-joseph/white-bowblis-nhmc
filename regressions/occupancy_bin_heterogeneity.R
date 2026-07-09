# =============================================================================
# regressions/occupancy_bin_heterogeneity.R
#
# Purpose:
#   Replace the earlier median-split near/slack-capacity classification with
#   a non-parametric bin design, per advisor feedback (C. Miller / J. Bowblis):
#
#     Baseline occupancy = average occupancy_rate over event_time in [-12,-4]
#     (same pre-acquisition window used for the spare_capacity baseline).
#
#     Bins:  <70%,  [70,80),  [80,90),  [90,95],  >95
#
#   For each bin, estimate the treatment effect of ownership change on
#   occupancy_rate. ALSO estimate the identical (pooled, non-binned)
#   treatment effect on the SAME analytic sample, for direct comparison
#   against the bin-specific estimates -- exactly as requested.
#
# Design:
#   Single pooled regression with post interacted with bin dummies
#   (bin "<70%" as the omitted/reference category), so each bin's TOTAL
#   effect is recovered via a linear combination (post + bin interaction).
#   This mirrors the near/slack-capacity interaction design used elsewhere
#   in this project, generalized from 2 groups to 5 bins.
#
# Sample:
#   Restricted to ever-treated (CHOW) facilities with a usable baseline
#   occupancy value (event_time -12 to -4), anticipation window excluded --
#   same restriction for BOTH the binned model and the pooled comparison
#   model, so the comparison is apples-to-apples.
#
# Output:
#   outputs/tables/occupancy_bin_heterogeneity.tex
#   outputs/tables/occupancy_bin_classification_summary.csv
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(tibble)
  library(readr)
})

options(scipen = 999, digits = 4)

out_dir <- out_tables_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tex_out_fp   <- file.path(out_dir, "occupancy_bin_heterogeneity.tex")
class_out_fp <- file.path(out_dir, "occupancy_bin_classification_summary.csv")

# -----------------------------------------------------------------------------
# Load panel
# -----------------------------------------------------------------------------
df <- load_staffing_panel()
stopifnot(all(c("occupancy_rate", "treated", "event_time") %in% names(df)))

# -----------------------------------------------------------------------------
# Baseline occupancy classification (event_time -12 to -4, same window as
# the spare_capacity baseline)
# -----------------------------------------------------------------------------
baseline_occ <- df %>%
  filter(treated == 1, event_time >= -12, event_time <= -4) %>%
  group_by(cms_certification_number) %>%
  summarise(
    baseline_occupancy = mean(occupancy_rate, na.rm = TRUE),
    n_baseline_months = sum(!is.na(occupancy_rate)),
    .groups = "drop"
  ) %>%
  filter(n_baseline_months > 0, is.finite(baseline_occupancy))

# Bins: <70, [70,80), [80,90), [90,95], >95
# (occupancy_rate is stored on a 0-100 scale, matching the rest of the panel)
baseline_occ <- baseline_occ %>%
  mutate(
    occ_bin = case_when(
      baseline_occupancy < 70 ~ "<70%",
      baseline_occupancy >= 70 & baseline_occupancy < 80 ~ "70-80%",
      baseline_occupancy >= 80 & baseline_occupancy < 90 ~ "80-90%",
      baseline_occupancy >= 90 & baseline_occupancy <= 95 ~ "90-95%",
      baseline_occupancy > 95 ~ ">95%",
      TRUE ~ NA_character_
    ),
    occ_bin = factor(occ_bin, levels = c("<70%", "70-80%", "80-90%", "90-95%", ">95%"))
  )

write_csv(baseline_occ, class_out_fp)

cat("=== Baseline occupancy bin classification ===\n")
cat(sprintf("Treated facilities with usable baseline (event_time -12 to -4): %d\n", nrow(baseline_occ)))
print(table(baseline_occ$occ_bin))
cat("\n")

# -----------------------------------------------------------------------------
# Build analysis sample
# -----------------------------------------------------------------------------
df_treated <- df %>%
  filter(treated == 1) %>%
  inner_join(
    baseline_occ %>% select(cms_certification_number, baseline_occupancy, occ_bin),
    by = "cms_certification_number"
  ) %>%
  drop_anticipation_window()

cat(sprintf(
  "[sample] facility-months (anticipation excluded): %s (%d facilities)\n\n",
  format(nrow(df_treated), big.mark = ","), n_distinct(df_treated$cms_certification_number)
))

# -----------------------------------------------------------------------------
# Regression setup
# -----------------------------------------------------------------------------
vc <- as.formula(paste0("~ ", fe_unit, " + ", fe_time))
fe_part <- paste0("| ", fe_unit, " + ", fe_time)

# Controls excluding occupancy_rate / spare_capacity (occupancy_rate is the
# outcome here; spare_capacity is near-collinear with it).
controls_rhs <- {
  ctrls <- setdiff(get_controls(df_treated), c("occupancy_rate", "spare_capacity"))
  paste(ctrls, collapse = " + ")
}

# ---- Binned model: post interacted with bin dummies, "<70%" as reference ----
df_treated <- df_treated %>% mutate(occ_bin = relevel(occ_bin, ref = "<70%"))

fml_binned <- as.formula(sprintf(
  "occupancy_rate ~ post + post:occ_bin + %s %s",
  controls_rhs, fe_part
))

m_binned <- feols(fml_binned, data = df_treated, vcov = vc, lean = TRUE)

# ---- Pooled/"identical" model: no bins, SAME sample ----
fml_pooled <- as.formula(sprintf(
  "occupancy_rate ~ post + %s %s",
  controls_rhs, fe_part
))

m_pooled <- feols(fml_pooled, data = df_treated, vcov = vc, lean = TRUE)

# -----------------------------------------------------------------------------
# Helpers: coefficients, linear combinations, formatting
# -----------------------------------------------------------------------------
coef_se_star <- function(mod, term) {
  ct <- summary(mod)$coeftable
  if (!(term %in% rownames(ct))) return(list(coef = NA, se = NA, stars = ""))
  b  <- unname(ct[term, "Estimate"])
  se <- unname(ct[term, "Std. Error"])
  p  <- unname(ct[term, "Pr(>|t|)"])
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(coef = b, se = se, stars = stars)
}

lincom <- function(mod, weights_named) {
  b <- coef(mod); V <- vcov(mod)
  terms <- names(weights_named)
  if (!all(terms %in% names(b))) return(list(est = NA, se = NA, stars = ""))
  w <- weights_named[terms]
  est <- sum(w * b[terms])
  VV <- V[terms, terms, drop = FALSE]
  se <- sqrt(as.numeric(t(w) %*% VV %*% w))
  p <- 2 * (1 - pnorm(abs(est / se)))
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(est = est, se = se, stars = stars)
}

fmt_cell <- function(b, se, stars) {
  if (is.na(b) || is.na(se)) return("\\makecell[c]{-- \\\\ (--)}")
  bstr <- sprintf("%.4f", b); if (b > 0) bstr <- paste0("\\phantom{-}", bstr)
  sestr <- sprintf("%.4f", se)
  paste0("\\makecell[c]{$", bstr, "^{", stars, "}$ \\\\ $(", sestr, ")$}")
}

# ---- Bin-specific total effects ----
bin_levels <- levels(df_treated$occ_bin)  # "<70%","70-80%","80-90%","90-95%",">95%"

get_bin_effect <- function(bin_label) {
  if (bin_label == "<70%") {
    s <- coef_se_star(m_binned, "post")
    return(list(b = s$coef, se = s$se, stars = s$stars))
  }
  interaction_term <- paste0("post:occ_bin", bin_label)
  lc <- lincom(m_binned, c(post = 1, setNames(1, interaction_term)))
  list(b = lc$est, se = lc$se, stars = lc$stars)
}

bin_n <- baseline_occ %>% count(occ_bin, name = "n_facilities")

rows <- sapply(bin_levels, function(bl) {
  eff <- get_bin_effect(bl)
  n_fac <- bin_n$n_facilities[bin_n$occ_bin == bl]
  if (length(n_fac) == 0) n_fac <- 0
  paste(
    bl,
    fmt_cell(eff$b, eff$se, eff$stars),
    format(n_fac, big.mark = ","),
    sep = " & "
  )
})

pooled_s <- coef_se_star(m_pooled, "post")
pooled_row <- paste(
  "Pooled (identical spec, same sample, no bins)",
  fmt_cell(pooled_s$coef, pooled_s$se, pooled_s$stars),
  format(n_distinct(df_treated$cms_certification_number), big.mark = ","),
  sep = " & "
)

# -----------------------------------------------------------------------------
# Build LaTeX table
# -----------------------------------------------------------------------------

tex_lines <- c(
  "\\documentclass[12pt]{article}",
  "",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{makecell}",
  "\\usepackage{array}",
  "\\usepackage{caption}",
  "\\usepackage{float}",
  "",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "",
  "\\begin{document}",
  "",
  "\\begin{table}[H]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Occupancy Response to Ownership Change by Baseline Occupancy Bin}",
  "\\label{tab:occupancy-bin-heterogeneity}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l Y r @{}}",
  "\\toprule",
  "Baseline occupancy bin & Effect of \\textit{post} on occupancy rate & Facilities \\\\",
  "\\midrule",
  paste0(rows, " \\\\"),
  "\\midrule",
  paste0(pooled_row, " \\\\"),
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Sample restricted to ever-treated (CHOW) facilities with a usable baseline occupancy rate, defined as the average occupancy rate over event\\_time $\\in [-12,-4]$ (4--12 months before ownership change, prior to the anticipation window). Bins: $<$70\\%, [70,80), [80,90), [90,95], $>$95\\%.",
  "\\item Bin-specific effects come from a single pooled regression with \\textit{post} interacted with bin dummies (``$<$70\\%'' as the reference category); each bin's total effect is \\textit{post} plus its interaction term where applicable. The pooled row re-estimates the identical specification WITHOUT bin interactions, on the SAME analytic sample, for direct comparison.",
  "\\item All specifications include facility and calendar-month fixed effects and the standard control set (excluding occupancy rate and spare capacity, which are excluded as controls since occupancy rate is the outcome here). Anticipation window excluded.",
  "\\item Standard errors are clustered two ways by facility and calendar month. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. Note some bins (especially $>$95\\% and possibly $<$70\\%) may have limited facility counts, reducing power for those estimates.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp, useBytes = TRUE)

cat("\n[write] ", normalizePath(tex_out_fp, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(class_out_fp, winslash = "\\"), "\n", sep = "")
cat("Done.\n")
