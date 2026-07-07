# =============================================================================
# composition_checks_monthly_standalone.R
#
# Purpose:
#   Run mechanism/composition checks using the monthly staffing panel and
#   short-stay quality checks using the quarterly quality panel.
#
# Monthly mechanism outcomes:
#   1. Occupancy rate
#   2. Medicare payer mix
#   3. Medicaid payer mix
#   4. Spare capacity ((certified beds - avg. daily census) / certified beds)
#   5. Average length of stay
#
#   NOTE: Case mix has been DROPPED from this set of mechanism checks
#   (previously outcome #4; removed per project decision).
#
# Short-stay quality outcomes:
#   1. qm_424: Short-stay moderate/severe pain
#   2. qm_425: Short-stay new/worsened pressure ulcers
#   3. qm_430: Short-stay pneumococcal vaccine
#   4. qm_434: Short-stay newly receiving antipsychotic medication
#   5. qm_471: Short-stay improved function
#   6. qm_472: Short-stay influenza vaccine
#
# Output:
#   outputs/tables/composition_checks_monthly_standalone.tex
#   outputs/plots/spare_capacity_hist_mechanism.png
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(stringr)
  library(readr)
  library(ggplot2)
})

options(scipen = 999, digits = 4)

# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------

out_dir <- out_tables_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_plots_dir, recursive = TRUE, showWarnings = FALSE)

tex_out_fp <- file.path(out_dir, "composition_checks_monthly_standalone.tex")
sc_hist_fp <- file.path(out_plots_dir, "spare_capacity_hist_mechanism.png")

# -----------------------------------------------------------------------------
# Load monthly staffing panel
# -----------------------------------------------------------------------------

df <- load_staffing_panel()

if (!("year" %in% names(df))) {
  df <- df %>%
    mutate(year = as.integer(str_sub(as.character(year_month), 1, 4)))
}

df <- df %>%
  mutate(
    ym_date = as.Date(
      paste0(str_replace(as.character(year_month), "/", "-"), "-01")
    )
  )

stopifnot("spare_capacity" %in% names(df))

# Use same pre-closing adjustment window exclusion as the main staffing models.
df_monthly <- drop_anticipation_window(df)

# -----------------------------------------------------------------------------
# Load quarterly quality panel
# -----------------------------------------------------------------------------

quality_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/quality_panel.csv"

df_quality <- read_csv(quality_fp, show_col_types = FALSE)

df_quality <- df_quality %>%
  mutate(
    cms_certification_number = as.character(cms_certification_number),
    quarter = as.character(quarter),
    year = as.integer(year),
    year_quarter = paste0(year, quarter),
    quarter_index = (year - 2017L) * 4L + as.integer(str_extract(quarter, "[1-4]"))
  )

# Drop event quarter for the post specification.
# This is analogous to excluding the transition quarter in your quality models.
df_quality_post <- df_quality %>%
  filter(is.na(event_time) | event_time != 0)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

coef_se_star <- function(mod, term = "post") {
  
  sm <- summary(mod)
  ct <- sm$coeftable
  
  if (!(term %in% rownames(ct))) {
    return(list(coef = NA_real_, se = NA_real_, p = NA_real_, stars = ""))
  }
  
  b  <- unname(ct[term, "Estimate"])
  se <- unname(ct[term, "Std. Error"])
  p  <- unname(ct[term, "Pr(>|t|)"])
  
  stars <- if (is.na(p)) {
    ""
  } else if (p < 0.01) {
    "***"
  } else if (p < 0.05) {
    "**"
  } else if (p < 0.10) {
    "*"
  } else {
    ""
  }
  
  list(coef = b, se = se, p = p, stars = stars)
}

fmt_est <- function(mod, term = "post") {
  
  s <- coef_se_star(mod, term)
  
  if (is.na(s$coef) || is.na(s$se)) {
    return("\\makecell[c]{-- \\\\ (--) }")
  }
  
  b <- sprintf("%.3f", s$coef)
  if (s$coef > 0) b <- paste0("\\phantom{-}", b)
  
  se <- sprintf("%.3f", s$se)
  
  if (s$stars == "") {
    paste0("\\makecell[c]{$", b, "$ \\\\ $(" , se, ")$}")
  } else {
    paste0("\\makecell[c]{$", b, "^{", s$stars, "}$ \\\\ $(" , se, ")$}")
  }
}

fmt_n <- function(mod) {
  format(nobs(mod), big.mark = ",")
}

make_row <- function(label, mod_nocontrols, mod_controls) {
  paste(
    label,
    fmt_est(mod_nocontrols),
    fmt_est(mod_controls),
    fmt_n(mod_nocontrols),
    sep = " & "
  )
}

# -----------------------------------------------------------------------------
# Monthly mechanism regressions
# -----------------------------------------------------------------------------

vc_month <- ~ cms_certification_number + year_month

# No-controls specification:
# Facility fixed effects + calendar-month fixed effects.
m_occ_nocontrols <- feols(
  occupancy_rate ~ post | cms_certification_number + year_month,
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_mcare_nocontrols <- feols(
  pct_medicare ~ post | cms_certification_number + year_month,
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_mcaid_nocontrols <- feols(
  pct_medicaid ~ post | cms_certification_number + year_month,
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_sc_nocontrols <- feols(
  spare_capacity ~ post | cms_certification_number + year_month,
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_los_nocontrols <- feols(
  avg_los_total ~ post | cms_certification_number + year_month,
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

# Controls specification:
# Do NOT include staffing controls here.
# Staffing is a main outcome of interest and may itself respond to ownership change.
controls_month <- intersect(
  c("beds", "government", "non_profit", "chain"),
  names(df_monthly)
)

rhs_controls_month <- paste(c("post", controls_month), collapse = " + ")

m_occ_controls <- feols(
  as.formula(
    paste0(
      "occupancy_rate ~ ",
      rhs_controls_month,
      " | cms_certification_number + year_month"
    )
  ),
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_mcare_controls <- feols(
  as.formula(
    paste0(
      "pct_medicare ~ ",
      rhs_controls_month,
      " | cms_certification_number + year_month"
    )
  ),
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_mcaid_controls <- feols(
  as.formula(
    paste0(
      "pct_medicaid ~ ",
      rhs_controls_month,
      " | cms_certification_number + year_month"
    )
  ),
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_sc_controls <- feols(
  as.formula(
    paste0(
      "spare_capacity ~ ",
      rhs_controls_month,
      " | cms_certification_number + year_month"
    )
  ),
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

m_los_controls <- feols(
  as.formula(
    paste0(
      "avg_los_total ~ ",
      rhs_controls_month,
      " | cms_certification_number + year_month"
    )
  ),
  data = df_monthly,
  vcov = vc_month,
  lean = FALSE
)

# -----------------------------------------------------------------------------
# Economic significance for spare capacity
# -----------------------------------------------------------------------------

spare_capacity_mean <- mean(df_monthly$spare_capacity, na.rm = TRUE)

spare_capacity_coef_nocontrols <- coef(m_sc_nocontrols)["post"]
spare_capacity_coef_controls   <- coef(m_sc_controls)["post"]

spare_capacity_pct_change_nocontrols <- 100 * spare_capacity_coef_nocontrols / spare_capacity_mean
spare_capacity_pct_change_controls   <- 100 * spare_capacity_coef_controls / spare_capacity_mean

# -----------------------------------------------------------------------------
# Distribution of spare capacity (reported alongside the mechanism table)
# -----------------------------------------------------------------------------

sc_vals <- df_monthly$spare_capacity[is.finite(df_monthly$spare_capacity)]

sc_dist <- tibble::tibble(
  n      = length(sc_vals),
  mean   = mean(sc_vals),
  sd     = sd(sc_vals),
  p10    = quantile(sc_vals, 0.10),
  p25    = quantile(sc_vals, 0.25),
  median = median(sc_vals),
  p75    = quantile(sc_vals, 0.75),
  p90    = quantile(sc_vals, 0.90),
  max    = max(sc_vals)
)

fmt3 <- function(x) sprintf("%.3f", x)

sc_dist_table_rows <- c(
  paste0("N (facility-months) & ", format(sc_dist$n, big.mark = ","), " \\\\"),
  paste0("Mean & ", fmt3(sc_dist$mean), " \\\\"),
  paste0("SD & ", fmt3(sc_dist$sd), " \\\\"),
  paste0("P10 & ", fmt3(sc_dist$p10), " \\\\"),
  paste0("P25 & ", fmt3(sc_dist$p25), " \\\\"),
  paste0("Median & ", fmt3(sc_dist$median), " \\\\"),
  paste0("P75 & ", fmt3(sc_dist$p75), " \\\\"),
  paste0("P90 & ", fmt3(sc_dist$p90), " \\\\"),
  paste0("Max & ", fmt3(sc_dist$max), " \\\\")
)

# Histogram for the report
p_sc_hist <- ggplot(
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

ggsave(sc_hist_fp, plot = p_sc_hist, width = 6.5, height = 4, dpi = 300)

sc_hist_fp_tex <- gsub("\\\\", "/", sc_hist_fp)

# -----------------------------------------------------------------------------
# Short-stay quality regressions
# -----------------------------------------------------------------------------

vc_quarter <- ~ cms_certification_number + year_quarter

controls_quality <- intersect(
  c(
    "beds",
    "government",
    "non_profit",
    "chain",
    "occupancy_rate",
    "pct_medicare",
    "pct_medicaid"
  ),
  names(df_quality_post)
)

rhs_controls_quality <- paste(c("post", controls_quality), collapse = " + ")

run_quality_nocontrols <- function(outcome) {
  feols(
    as.formula(
      paste0(
        outcome,
        " ~ post | cms_certification_number + year_quarter"
      )
    ),
    data = df_quality_post,
    vcov = vc_quarter,
    lean = FALSE
  )
}

run_quality_controls <- function(outcome) {
  feols(
    as.formula(
      paste0(
        outcome,
        " ~ ",
        rhs_controls_quality,
        " | cms_certification_number + year_quarter"
      )
    ),
    data = df_quality_post,
    vcov = vc_quarter,
    lean = FALSE
  )
}

short_stay_specs <- tibble::tribble(
  ~outcome, ~label, ~direction,
  "qm_424", "Moderate/severe pain", "Lower is better",
  "qm_425", "New/worsened pressure ulcers", "Lower is better",
  "qm_430", "Pneumococcal vaccine", "Higher is better",
  "qm_434", "New antipsychotic medication", "Lower is better",
  "qm_471", "Improved function", "Higher is better",
  "qm_472", "Influenza vaccine", "Higher is better"
) %>%
  filter(outcome %in% names(df_quality_post))

short_stay_models <- short_stay_specs %>%
  rowwise() %>%
  mutate(
    mod_nocontrols = list(run_quality_nocontrols(outcome)),
    mod_controls = list(run_quality_controls(outcome)),
    row = make_row(label, mod_nocontrols, mod_controls)
  ) %>%
  ungroup()

# -----------------------------------------------------------------------------
# Build table rows
# -----------------------------------------------------------------------------

row_occ <- make_row(
  "Occupancy rate",
  m_occ_nocontrols,
  m_occ_controls
)

row_mcare <- make_row(
  "Medicare share",
  m_mcare_nocontrols,
  m_mcare_controls
)

row_mcaid <- make_row(
  "Medicaid share",
  m_mcaid_nocontrols,
  m_mcaid_controls
)

row_sc <- make_row(
  "Spare capacity",
  m_sc_nocontrols,
  m_sc_controls
)

row_los <- make_row(
  "Average length of stay",
  m_los_nocontrols,
  m_los_controls
)

short_stay_table_rows <- paste0(short_stay_models$row, " \\\\")

short_stay_direction_lines <- paste0(
  "\\item ",
  short_stay_models$label,
  ": ",
  short_stay_models$direction,
  "."
)

# -----------------------------------------------------------------------------
# Build standalone LaTeX document
# -----------------------------------------------------------------------------

sc_notes_line <- paste0(
  "\\item Spare capacity mean is ", sprintf("%.3f", spare_capacity_mean),
  "; the no-controls coefficient implies a ", sprintf("%.1f", spare_capacity_pct_change_nocontrols),
  "\\% change relative to the mean, and the controls specification implies a ",
  sprintf("%.1f", spare_capacity_pct_change_controls), "\\% change."
)

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
  "\\usepackage{setspace}",
  "\\usepackage{graphicx}",
  "",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "",
  "\\begin{document}",
  "",
  "% ---------------------------------------------------------------------------",
  "% Distribution of spare capacity",
  "% ---------------------------------------------------------------------------",
  "",
  "\\begin{figure}[!ht]",
  "\\centering",
  paste0("\\includegraphics[width=0.8\\textwidth]{", sc_hist_fp_tex, "}"),
  "\\caption{Distribution of Spare Capacity (Facility-Months)}",
  "\\label{fig:spare-capacity-hist}",
  "\\end{figure}",
  "",
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Summary Statistics: Spare Capacity}",
  "\\label{tab:spare-capacity-summary}",
  "\\small",
  "\\begin{tabular}{l r}",
  "\\toprule",
  "Statistic & Value \\\\",
  "\\midrule",
  sc_dist_table_rows,
  "\\bottomrule",
  "\\end{tabular}",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Spare capacity is defined as (certified beds $-$ average daily census) / certified beds, computed from Medicare Cost Report data. Sample excludes the anticipation window used in the main staffing models.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\clearpage",
  "",
  "% ---------------------------------------------------------------------------",
  "% Table 1: Monthly mechanism checks",
  "% ---------------------------------------------------------------------------",
  "",
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Effects of Ownership Change on Occupancy, Payer Mix, Spare Capacity, and Length of Stay}",
  "\\label{tab:composition-checks-monthly}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l Y Y r @{}}",
  "\\toprule",
  "Outcome & No controls & Controls & Observations \\\\",
  "\\midrule",
  paste0(row_occ, " \\\\"),
  paste0(row_mcare, " \\\\"),
  paste0(row_mcaid, " \\\\"),
  paste0(row_sc, " \\\\"),
  paste0(row_los, " \\\\"),
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses. All models include facility and calendar-month fixed effects. The controls column adds beds, government ownership, nonprofit ownership, and chain affiliation.",
  sc_notes_line,
  "\\item Standard errors are clustered two ways by facility and calendar month. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\clearpage",
  "",
  "% ---------------------------------------------------------------------------",
  "% Table 2: Short-stay quality checks",
  "% ---------------------------------------------------------------------------",
  "",
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Effects of Ownership Change on Short-Stay Quality Measures}",
  "\\label{tab:short-stay-quality-checks}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l Y Y r @{}}",
  "\\toprule",
  "Outcome & No controls & Controls & Observations \\\\",
  "\\midrule",
  short_stay_table_rows,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses. All models include facility and calendar-quarter fixed effects. The controls column adds beds, government ownership, nonprofit ownership, chain affiliation, occupancy rate, Medicare share, and Medicaid share when available.",
  "\\item The ownership-change quarter is excluded from the short-stay quality regressions.",
  "\\item Standard errors are clustered two ways by facility and calendar quarter. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp)

# -----------------------------------------------------------------------------
# Console summary
# -----------------------------------------------------------------------------

cat("\nSaved table to:\n")
cat(tex_out_fp, "\n\n")
cat("Saved histogram to:\n")
cat(sc_hist_fp, "\n\n")

cat("Spare capacity mean:", round(spare_capacity_mean, 3), "\n")
cat("Spare capacity percent change, no controls:", round(spare_capacity_pct_change_nocontrols, 3), "\n")
cat("Spare capacity percent change, controls:", round(spare_capacity_pct_change_controls, 3), "\n")
