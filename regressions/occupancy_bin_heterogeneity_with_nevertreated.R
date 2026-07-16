# =============================================================================
# regressions/occupancy_bin_heterogeneity_with_nevertreated.R
#
# Purpose:
#   Companion to occupancy_bin_heterogeneity.R, answering C. Moul's question
#   directly: what happens to the occupancy-bin results if never-treated
#   facilities are included in the estimation sample, rather than restricting
#   to ever-treated (CHOW) facilities only?
#
# Classification (SIMPLIFIED per C. Moul's observation):
#   - Treated facilities: baseline occupancy = average occupancy_rate over
#     event_time in [-12,-4] (i.e., relative to EACH FACILITY'S OWN
#     acquisition timing). Same as occupancy_bin_heterogeneity.R. This is
#     the only classification that matters for the fitted coefficients.
#   - Never-treated facilities: occ_bin only ever enters the model through
#     the interaction post:occ_bin. Since post = 0 for every never-treated
#     observation, their bin label is multiplied by zero regardless of what
#     it is -- it is mathematically INERT. There is therefore no need to
#     construct a real baseline occupancy classification for them at all.
#     They are assigned a PLACEHOLDER value (the reference bin, "<70%")
#     purely so they are not dropped from the sample via NA handling.
#     Their actual contribution to the model is helping pin down the
#     calendar (year_month) fixed effects and other controls, not the
#     treatment coefficients themselves.
#   - Because the placeholder does not depend on any real baseline data,
#     never-treated facilities are no longer required to have valid
#     occupancy_rate data in any specific calendar window (e.g. 2017 Q1) to
#     be included -- this actually INCREASES the never-treated sample size
#     relative to the earlier fixed-calendar-baseline version, since it
#     avoids the same kind of early-2017 PBJ-coverage-gap exclusion we
#     found elsewhere in this project.
#
# Sample:
#   FULL panel (treated + never-treated), anticipation window excluded.
#   Treated facilities WITHOUT a usable baseline occupancy (event_time
#   -12 to -4) are still excluded, since their bin assignment is NOT inert
#   and a placeholder would misrepresent them.
#
# Output:
#   outputs/tables/occupancy_bin_heterogeneity_with_nevertreated.tex
#   outputs/tables/occupancy_bin_classification_summary_with_nevertreated.csv
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
class_out_fp <- file.path(out_dir, "occupancy_bin_classification_summary_with_nevertreated.csv")

# -----------------------------------------------------------------------------
# Load panel
# -----------------------------------------------------------------------------
df <- load_staffing_panel()
stopifnot(all(c(
  "occupancy_rate", "pct_medicare", "pct_medicaid", "avg_los_total",
  "treated", "event_time"
) %in% names(df)))

# Shared bin-assignment function (fixed absolute thresholds)
assign_bin <- function(x) {
  case_when(
    x < 70 ~ "<70%",
    x >= 70 & x < 80 ~ "70-80%",
    x >= 80 & x < 90 ~ "80-90%",
    x >= 90 & x <= 95 ~ "90-95%",
    x > 95 ~ ">95%",
    TRUE ~ NA_character_
  )
}
bin_level_order <- c("<70%", "70-80%", "80-90%", "90-95%", ">95%")
REF_BIN <- "<70%"

# -----------------------------------------------------------------------------
# Treated classification: event-relative baseline (only real classification
# needed -- this is the only one that affects the fitted coefficients)
# -----------------------------------------------------------------------------
baseline_treated <- df %>%
  filter(treated == 1, event_time >= -12, event_time <= -4) %>%
  group_by(cms_certification_number) %>%
  summarise(
    baseline_occupancy = mean(occupancy_rate, na.rm = TRUE),
    n_baseline_months = sum(!is.na(occupancy_rate)),
    .groups = "drop"
  ) %>%
  filter(n_baseline_months > 0, is.finite(baseline_occupancy)) %>%
  mutate(occ_bin_treated = assign_bin(baseline_occupancy))

write_csv(baseline_treated, class_out_fp)

n_nevertreated_total <- n_distinct(df$cms_certification_number[df$treated == 0])

cat("=== Baseline occupancy bin classification ===\n")
cat(sprintf("Treated facilities with usable baseline (event_time -12 to -4): %d\n", nrow(baseline_treated)))
cat(sprintf("Never-treated facilities (ALL included, placeholder bin -- inert): %d\n", n_nevertreated_total))
print(table(baseline_treated$occ_bin_treated))
cat("\n")

# -----------------------------------------------------------------------------
# Build analysis sample: FULL panel, anticipation window excluded.
#   - Treated facilities without a usable baseline are dropped (their bin
#     assignment is NOT inert, so a placeholder would misrepresent them).
#   - Never-treated facilities are ALL kept, with a placeholder bin.
# -----------------------------------------------------------------------------
df_full <- df %>%
  drop_anticipation_window() %>%
  left_join(
    baseline_treated %>% select(cms_certification_number, baseline_occupancy, occ_bin_treated),
    by = "cms_certification_number"
  ) %>%
  filter(treated == 0 | !is.na(occ_bin_treated)) %>%
  mutate(
    occ_bin = if_else(treated == 1, occ_bin_treated, REF_BIN),
    occ_bin = factor(occ_bin, levels = bin_level_order),
    occ_bin = relevel(occ_bin, ref = REF_BIN)
  ) %>%
  select(-occ_bin_treated)

cat(sprintf(
  "[sample] facility-months (anticipation excluded): %s (%d facilities)\n\n",
  format(nrow(df_full), big.mark = ","), n_distinct(df_full$cms_certification_number)
))

bin_levels <- levels(df_full$occ_bin)

bin_display_labels <- c(
  "<70%"    = "$<$70\\%",
  "70-80%"  = "70--80\\%",
  "80-90%"  = "80--90\\%",
  "90-95%"  = "90--95\\%",
  ">95%"    = "$>$95\\%"
)

# Facility counts per bin: TREATED facilities only (their real
# classification). Never-treated facilities don't have a real bin identity,
# so they're reported separately rather than folded into "<70%".
bin_n <- baseline_treated %>% count(occ_bin_treated, name = "n_facilities") %>%
  rename(occ_bin = occ_bin_treated)

# -----------------------------------------------------------------------------
# Regression setup (identical to main script)
# -----------------------------------------------------------------------------
vc <- as.formula(paste0("~ ", fe_unit, " + ", fe_time))
fe_part <- paste0("| ", fe_unit, " + ", fe_time)

strategic_choice_vars <- c("occupancy_rate", "spare_capacity", "pct_medicare", "pct_medicaid", "avg_los_total")

controls_rhs_for <- function(dat, outcome) {
  ctrls <- setdiff(get_controls(dat), strategic_choice_vars)
  if (length(ctrls) == 0) return("1")
  paste(ctrls, collapse = " + ")
}

coef_se_star <- function(mod, term) {
  ct <- summary(mod)$coeftable
  if (!(term %in% rownames(ct))) return(list(coef = NA, se = NA, stars = ""))
  b  <- unname(ct[term, "Estimate"])
  se <- unname(ct[term, "Std. Error"])
  p  <- unname(ct[term, "Pr(>|t|)"])
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(coef = b, se = se, stars = stars)
}

fmt_cell <- function(b, se, stars) {
  if (is.na(b) || is.na(se)) return("\\makecell[t]{-- \\\\ (--)}")
  bstr <- sprintf("%.4f", b); if (b > 0) bstr <- paste0("\\phantom{-}", bstr)
  sestr <- sprintf("%.4f", se)
  paste0("\\makecell[t]{$", bstr, "^{", stars, "}$ \\\\ $(", sestr, ")$}")
}

get_bin_effect <- function(mod, bin_label) {
  if (bin_label == REF_BIN) {
    s <- coef_se_star(mod, "post")
    return(list(b = s$coef, se = s$se, stars = s$stars))
  }
  interaction_term <- paste0("post:occ_bin", bin_label)
  s <- coef_se_star(mod, interaction_term)
  list(b = s$coef, se = s$se, stars = s$stars)
}

# -----------------------------------------------------------------------------
# Fit binned + pooled models for each outcome, on the FULL sample
# -----------------------------------------------------------------------------
outcomes <- tibble::tribble(
  ~var,             ~label,
  "occupancy_rate",  "Occupancy rate",
  "pct_medicare",    "Medicare share",
  "pct_medicaid",    "Medicaid share",
  "avg_los_total",   "Average length of stay"
)

fit_outcome <- function(v) {
  ctrls <- controls_rhs_for(df_full, v)
  fml_binned <- as.formula(sprintf("%s ~ post + post:occ_bin + %s %s", v, ctrls, fe_part))
  fml_pooled <- as.formula(sprintf("%s ~ post + %s %s", v, ctrls, fe_part))
  list(
    binned = feols(fml_binned, data = df_full, vcov = vc, lean = TRUE),
    pooled = feols(fml_pooled, data = df_full, vcov = vc, lean = TRUE)
  )
}

fits <- setNames(lapply(outcomes$var, fit_outcome), outcomes$var)

# -----------------------------------------------------------------------------
# Build transposed table: rows = outcomes, columns = bins (+ pooled)
# -----------------------------------------------------------------------------
build_cell <- function(v, bin_label) {
  eff <- get_bin_effect(fits[[v]]$binned, bin_label)
  fmt_cell(eff$b, eff$se, eff$stars)
}

bin_facility_counts <- sapply(bin_levels, function(bl) {
  n_fac <- bin_n$n_facilities[bin_n$occ_bin == bl]
  if (length(n_fac) == 0) n_fac <- 0
  paste0("\\makecell[t]{", format(n_fac, big.mark = ","), " \\\\ \\mbox{}}")
})

facilities_row <- paste(
  "Facilities (treated)",
  paste(bin_facility_counts, collapse = " & "),
  paste0("\\makecell[t]{", format(nrow(baseline_treated), big.mark = ","), " \\\\ \\mbox{}}"),
  sep = " & "
)

outcome_rows <- sapply(seq_len(nrow(outcomes)), function(i) {
  v <- outcomes$var[i]
  lab <- outcomes$label[i]
  bin_cells <- sapply(bin_levels, build_cell, v = v)
  pooled_s <- coef_se_star(fits[[v]]$pooled, "post")
  pooled_cell <- fmt_cell(pooled_s$coef, pooled_s$se, pooled_s$stars)
  paste(lab, paste(bin_cells, collapse = " & "), pooled_cell, sep = " & ")
})

# -----------------------------------------------------------------------------
# Build LaTeX table
# -----------------------------------------------------------------------------
bin_col_headers <- paste(bin_display_labels[bin_levels], collapse = " & ")

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
  "\\caption{Response to Ownership Change by Baseline Occupancy Bin, Full Sample (Including Never-Treated Facilities)}",
  "\\label{tab:occupancy-bin-heterogeneity-with-nevertreated}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l Y Y Y Y Y Y @{}}",
  "\\toprule",
  paste0("Outcome & ", bin_col_headers, " & Pooled \\\\"),
  "\\midrule",
  paste0(facilities_row, " \\\\"),
  "\\midrule",
  paste0(outcome_rows, " \\\\"),
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  sprintf(
    "\\item \\textit{Notes:} Sample includes ALL facilities (treated and never-treated), anticipation window excluded. Treated facilities are classified by baseline occupancy over event\\_time $\\in [-12,-4]$. Never-treated facilities ($N=%s$) are assigned a placeholder bin ($<$70\\%%)",
    format(n_nevertreated_total, big.mark = ",")
  ),
  "\\item All specifications include facility and calendar-month fixed effects and the standard controls. Occupancy rate, spare capacity, Medicare share, Medicaid share, and average length of stay are excluded as controls from all four of these strategic-choice regressions. Anticipation window excluded.",
  "\\item Standard errors are clustered two ways by facility and calendar month. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
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
