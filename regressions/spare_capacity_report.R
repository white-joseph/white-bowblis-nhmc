# =============================================================================
# regressions/spare_capacity_report.R
#
# Purpose:
#   Single combined report for spare_capacity:
#     1. Distribution (histogram)
#     2. Summary statistics table (styled to match the project's other
#        summary-stats tables, e.g. quarterly_summary_stats.R -- just with
#        more columns, since this is reporting the full five-number summary
#        for a single variable rather than Mean/SD across many variables)
#     3. The near-capacity vs. slack-capacity static regression table
#        (spare_capacity ~ post + post:near_capacity + controls), TRANSPOSED
#        so outcomes (Spare capacity, Occupancy rate) run across the top and
#        the estimates (post, difference, near-capacity effect) run down the
#        side.
#
#   Classification of near-capacity vs. slack-capacity and the dynamic
#   (event-study) version of this analysis live in
#   spare_capacity_near_vs_slack.R -- this script covers the static /
#   descriptive side only.
#
# Output:
#   outputs/tables/spare_capacity_report.tex
#   outputs/plots/spare_capacity_report_hist.png
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(tibble)
  library(ggplot2)
})

options(scipen = 999, digits = 4)

out_dir <- out_tables_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_plots_dir, recursive = TRUE, showWarnings = FALSE)

tex_out_fp <- file.path(out_dir, "spare_capacity_report.tex")
hist_fp     <- file.path(out_plots_dir, "spare_capacity_report_hist.png")

# -----------------------------------------------------------------------------
# Load monthly staffing panel
# -----------------------------------------------------------------------------
df <- load_staffing_panel()
stopifnot(all(c("spare_capacity", "occupancy_rate") %in% names(df)))

# Same sample convention as the rest of the monthly mechanism/staffing models.
df_monthly <- drop_anticipation_window(df)

# =============================================================================
# 1 & 2. Distribution + summary statistics
# =============================================================================

sc_vals <- df_monthly$spare_capacity[is.finite(df_monthly$spare_capacity)]

fmt_int <- function(x) format(x, big.mark = ",", trim = TRUE, scientific = FALSE)
fmt_dec <- function(x, k = 3) ifelse(is.na(x), "NA", formatC(x, format = "f", digits = k))

sc_stats <- tibble(
  n      = length(sc_vals),
  mean   = mean(sc_vals),
  sd     = sd(sc_vals),
  min    = min(sc_vals),
  p25    = quantile(sc_vals, 0.25),
  median = median(sc_vals),
  p75    = quantile(sc_vals, 0.75),
  max    = max(sc_vals)
)

summary_stat_rows <- c(
  paste0("N & ", fmt_int(sc_stats$n), " \\\\"),
  paste0("Mean & ", fmt_dec(sc_stats$mean), " \\\\"),
  paste0("SD & ", fmt_dec(sc_stats$sd), " \\\\"),
  paste0("Min & ", fmt_dec(sc_stats$min), " \\\\"),
  paste0("P25 & ", fmt_dec(sc_stats$p25), " \\\\"),
  paste0("Median & ", fmt_dec(sc_stats$median), " \\\\"),
  paste0("P75 & ", fmt_dec(sc_stats$p75), " \\\\"),
  paste0("Max & ", fmt_dec(sc_stats$max), " \\\\")
)

p_hist <- ggplot(
  df_monthly %>% filter(is.finite(spare_capacity)),
  aes(x = spare_capacity)
) +
  geom_histogram(bins = 60, fill = "steelblue", color = "white", boundary = 0) +
  labs(
    title = NULL,
    x = "Spare capacity",
    y = "Count (facility-months)"
  ) +
  theme_minimal(base_size = 12)

ggsave(hist_fp, plot = p_hist, width = 6.5, height = 4, dpi = 300)

# Also colocate a copy of the histogram right next to the .tex file, and
# reference it by BARE FILENAME (not an absolute path) in \includegraphics.
# Absolute Windows paths with a drive letter can be blocked or mishandled
# by some LaTeX distributions' file-access security settings, which can
# derail rendering of everything after the failed \includegraphics call.
# A bare filename resolved relative to the .tex file's own directory avoids
# that entirely.
hist_fp_local <- file.path(out_dir, basename(hist_fp))
file.copy(hist_fp, hist_fp_local, overwrite = TRUE)
hist_fp_tex <- basename(hist_fp)

# =============================================================================
# 3. Near-capacity vs. slack-capacity static regression (transposed table)
# =============================================================================

# ---- Baseline classification (same rule as spare_capacity_near_vs_slack.R) ----
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
  mutate(near_capacity = as.integer(baseline_spare_capacity <= baseline_median))

df_treated <- df %>%
  filter(treated == 1) %>%
  inner_join(
    baseline_window %>% select(cms_certification_number, near_capacity),
    by = "cms_certification_number"
  ) %>%
  drop_anticipation_window()

# ---- Regressions ----
vc <- as.formula(paste0("~ ", fe_unit, " + ", fe_time))
fe_part <- paste0("| ", fe_unit, " + ", fe_time)

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

make_static_fml <- function(dat, lhs) {
  ctrls <- controls_rhs_for(dat, lhs)
  rhs_ctrl_part <- if (ctrls == "1") "" else paste0(" + ", ctrls)
  as.formula(sprintf("%s ~ post + post:near_capacity%s %s", lhs, rhs_ctrl_part, fe_part))
}

m_sc <- feols(
  make_static_fml(df_treated, "spare_capacity"),
  data = df_treated, vcov = vc, lean = TRUE
)

m_occ <- feols(
  make_static_fml(df_treated, "occupancy_rate"),
  data = df_treated, vcov = vc, lean = TRUE
)

# ---- Helpers: coefficients, linear combinations, formatting ----
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

# ---- Extract the three estimates for each outcome ----
sc_slack <- coef_se_star(m_sc, "post")
sc_diff  <- coef_se_star(m_sc, "post:near_capacity")
sc_near  <- lincom(m_sc, c(post = 1, "post:near_capacity" = 1))

occ_slack <- coef_se_star(m_occ, "post")
occ_diff  <- coef_se_star(m_occ, "post:near_capacity")
occ_near  <- lincom(m_occ, c(post = 1, "post:near_capacity" = 1))

# ---- Build TRANSPOSED table: outcomes across the top, estimates down the side ----
row_slack <- paste(
  "Slack-capacity: \\textit{post}",
  fmt_cell(sc_slack$coef, sc_slack$se, sc_slack$stars),
  fmt_cell(occ_slack$coef, occ_slack$se, occ_slack$stars),
  sep = " & "
)

row_diff <- paste(
  "Difference (Near $-$ Slack): \\textit{post} $\\times$ Near-capacity",
  fmt_cell(sc_diff$coef, sc_diff$se, sc_diff$stars),
  fmt_cell(occ_diff$coef, occ_diff$se, occ_diff$stars),
  sep = " & "
)

row_near <- paste(
  "Near-capacity effect: \\textit{post} $+$ interaction",
  fmt_cell(sc_near$est, sc_near$se, sc_near$stars),
  fmt_cell(occ_near$est, occ_near$se, occ_near$stars),
  sep = " & "
)

row_n <- paste(
  "Observations",
  format(nobs(m_sc), big.mark = ","),
  format(nobs(m_occ), big.mark = ","),
  sep = " & "
)

# =============================================================================
# Build combined LaTeX document
# =============================================================================

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
  "\\usepackage{graphicx}",
  "\\usepackage{float}",
  "\\captionsetup{labelfont=bf, font=small}",
  "",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "",
  "\\begin{document}",
  "",
  "% ---------------------------------------------------------------------------",
  "% Distribution of spare capacity",
  "% ---------------------------------------------------------------------------",
  "",
  "\\begin{figure}[H]",
  "\\centering",
  paste0("\\includegraphics[width=0.8\\textwidth]{", hist_fp_tex, "}"),
  "\\caption{Distribution of Spare Capacity (Facility-Months)}",
  "\\label{fig:spare-capacity-hist}",
  "\\end{figure}",
  "",
  "\\begin{table}[H]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Summary Statistics: Spare Capacity}",
  "\\label{tab:spare-capacity-summary}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabular}{l r}",
  "\\toprule",
  "\\textbf{Statistic} & \\textbf{Spare Capacity} \\\\",
  "\\midrule",
  summary_stat_rows,
  "\\bottomrule",
  "\\end{tabular}",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Sample excludes the anticipation window.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\vspace{1em}",
  "",
  "% ---------------------------------------------------------------------------",
  "% Near-capacity vs. slack-capacity static regression (transposed)",
  "% ---------------------------------------------------------------------------",
  "",
  "\\begin{table}[H]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Spare Capacity and Occupancy Response to Ownership Change: Near-Capacity vs. Slack-Capacity Acquisitions}",
  "\\label{tab:spare-capacity-near-vs-slack-transposed}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l Y Y @{}}",
  "\\toprule",
  " & \\textbf{Spare capacity} & \\textbf{Occupancy rate} \\\\",
  "\\midrule",
  paste0(row_slack, " \\\\"),
  paste0(row_diff, " \\\\"),
  paste0(row_near, " \\\\"),
  "\\midrule",
  paste0(row_n, " \\\\"),
  "\\bottomrule",
  "\\end{tabularx}",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Sample restricted to ever-treated facilities, split at the median pre-acquisition spare capacity (average spare capacity over event\\_time $\\in [-12,-4]$, prior to the anticipation window).",
  sprintf(
    "\\item Near-capacity $=$ baseline spare capacity $\\leq$ %.3f (median); Slack-capacity $=$ above. $N = %d$ treated facilities with a usable baseline (%d near-capacity, %d slack-capacity).",
    baseline_median, nrow(baseline_window),
    sum(baseline_window$near_capacity == 1), sum(baseline_window$near_capacity == 0)
  ),
  "\\item Controls exclude occupancy rate and spare capacity from each other's controls. All models include facility and calendar-month fixed effects; anticipation window excluded.",
  "\\item Standard errors are clustered two ways by facility and calendar month. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp, useBytes = TRUE)

# -----------------------------------------------------------------------------
# Console summary
# -----------------------------------------------------------------------------
cat("\n[write] ", normalizePath(tex_out_fp, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(hist_fp, winslash = "\\"), "\n", sep = "")
cat("Done.\n")
