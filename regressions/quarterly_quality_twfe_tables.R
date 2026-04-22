# =============================================================================
# quarterly_quality_twfe_tables.R
#
# Preliminary quarterly quality TWFE post regressions with standalone LaTeX output
#
# Output:
#   - one standalone .tex document containing every table produced by this script
#
# Windows:
#   - Full sample (2017 Q1 -- 2024 Q2): 401, 404, 406, 407, 410, 419, 434, 452
#   - Restricted (2017 Q1 -- 2023 Q3): 405, 451, 471
#   - Restricted (2018 Q1 -- 2023 Q3): 453
#
# Specifications:
#   - With anticipation
#   - Without anticipation (drop event_time == -1)
#   - No staffing controls
#   - No covid split
#   - No chain split
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(fixest)
  library(stringr)
})

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
project_root <- "C:/Repositories/white-bowblis-nhmc"
panel_fp <- file.path(project_root, "data", "clean", "quality_panel.csv")
out_dir  <- file.path(project_root, "outputs/tables")

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

tex_out_fp <- file.path(out_dir, "quarterly_quality_twfe_preliminary.tex")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
assert_has_cols <- function(df, cols, df_name = "data") {
  miss <- setdiff(cols, names(df))
  if (length(miss) > 0) {
    stop(
      sprintf("[%s] missing required columns: %s", df_name, paste(miss, collapse = ", ")),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

intersect_existing <- function(x, df) {
  intersect(x, names(df))
}

quarter_num <- function(x) {
  x <- toupper(trimws(as.character(x)))
  suppressWarnings(as.integer(str_extract(x, "[1-4]")))
}

year_quarter_index <- function(year, quarter) {
  qn <- quarter_num(quarter)
  yr <- suppressWarnings(as.integer(year))
  yr * 4L + qn
}

subset_window <- function(df, start_year, start_quarter, end_year, end_quarter) {
  start_idx <- start_year * 4L + start_quarter
  end_idx   <- end_year * 4L + end_quarter

  idx <- year_quarter_index(df$year, df$quarter)
  df[idx >= start_idx & idx <= end_idx, , drop = FALSE]
}

drop_anticipation_quarter <- function(df) {
  df %>% filter(is.na(event_time) | event_time != 0)
}

get_case_mix_controls <- function(df) {
  preferred <- intersect_existing(c("cm_q_state_2", "cm_q_state_3", "cm_q_state_4"), df)
  if (length(preferred) > 0) return(preferred)

  fallback <- intersect_existing(c("cm_q_nat_2", "cm_q_nat_3", "cm_q_nat_4"), df)
  fallback
}

get_controls <- function(df) {
  base_controls <- c(
    "government",
    "non_profit",
    "chain",
    "beds",
    "occupancy_rate",
    "pct_medicare",
    "pct_medicaid",
    "rn_hprd",
    "lpn_hprd",
    "cna_hprd"
  )
  c(intersect_existing(base_controls, df), get_case_mix_controls(df))
}

make_controls_rhs <- function(df) {
  ctrls <- get_controls(df)
  if (length(ctrls) == 0) return("1")
  paste(ctrls, collapse = " + ")
}

make_fml <- function(lhs, rhs) {
  as.formula(sprintf("%s ~ %s | cms_certification_number + year_quarter", lhs, rhs))
}

coef_se_star_n <- function(mod, term = "post") {
  if (is.null(mod)) return(list(coef = NA_real_, se = NA_real_, p = NA_real_, stars = "", n = NA_integer_))

  sm <- summary(mod)
  ct <- sm$coeftable

  if (!(term %in% rownames(ct))) {
    return(list(coef = NA_real_, se = NA_real_, p = NA_real_, stars = "", n = nobs(mod)))
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

  list(coef = b, se = se, p = p, stars = stars, n = nobs(mod))
}

fmt_est_n <- function(b, se, stars, n) {
  if (is.na(b) || is.na(se)) return("\\estn{$\\,$}{$\\,$}{}{\\,}")
  bstr <- sprintf("%.3f", b)
  if (b > 0) bstr <- paste0("\\phantom{-}", bstr)
  sestr <- sprintf("%.3f", se)
  nstr <- format(n, big.mark = ",")
  sprintf("\\estn{%s}{%s}{%s}{%s}", bstr, sestr, stars, nstr)
}

pretty_outcome_label <- function(x) {
  code <- str_replace(x, "^qm_", "")
  paste0("QM ", code)
}

escape_latex <- function(x) {
  x <- gsub("\\\\", "\\\\textbackslash{}", x)
  x <- gsub("([#$%&_{}])", "\\\\\\1", x, perl = TRUE)
  x <- gsub("~", "\\\\textasciitilde{}", x, fixed = TRUE)
  x <- gsub("\\^", "\\\\textasciicircum{}", x)
  x
}

# -----------------------------------------------------------------------------
# Load panel
# -----------------------------------------------------------------------------
if (!file.exists(panel_fp)) {
  stop(sprintf("Panel file not found: %s", panel_fp), call. = FALSE)
}

df <- readr::read_csv(panel_fp, show_col_types = FALSE)

required_cols <- c(
  "cms_certification_number",
  "year",
  "quarter",
  "treated",
  "post",
  "event_time"
)
assert_has_cols(df, required_cols, "quality_panel")

df <- df %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year = suppressWarnings(as.integer(year)),
    quarter = toupper(trimws(as.character(quarter))),
    year_quarter = paste0(year, "_", quarter)
  )

numeric_candidates <- c(
  "beds",
  "occupancy_rate",
  "pct_medicare",
  "pct_medicaid",
  "time",
  "time_treated",
  "event_time",
  "coverage_ratio",
  "government",
  "non_profit",
  "chain"
)
numeric_candidates <- intersect_existing(numeric_candidates, df)

if (length(numeric_candidates) > 0) {
  df <- df %>%
    mutate(across(all_of(numeric_candidates), ~ suppressWarnings(as.numeric(.x))))
}

controls_rhs <- make_controls_rhs(df)
rhs <- if (controls_rhs == "1") "post" else paste("post +", controls_rhs)
vc <- ~ cms_certification_number + year_quarter

# -----------------------------------------------------------------------------
# Outcome sets / windows
# -----------------------------------------------------------------------------
outcomes_full <- c("qm_401", "qm_404", "qm_406", "qm_407", "qm_410", "qm_419", "qm_434", "qm_452")
outcomes_2017_2023q3 <- c("qm_405", "qm_451", "qm_471")
outcomes_2018_2023q3 <- c("qm_453")

all_requested <- c(outcomes_full, outcomes_2017_2023q3, outcomes_2018_2023q3)
missing_outcomes <- setdiff(all_requested, names(df))
if (length(missing_outcomes) > 0) {
  stop(
    sprintf("These requested outcomes are missing from quality_panel.csv: %s",
            paste(missing_outcomes, collapse = ", ")),
    call. = FALSE
  )
}

datasets <- list(
  full_with = subset_window(df, 2017, 1, 2024, 2),
  full_without = drop_anticipation_quarter(subset_window(df, 2017, 1, 2024, 2)),

  restr17_with = subset_window(df, 2017, 1, 2023, 3),
  restr17_without = drop_anticipation_quarter(subset_window(df, 2017, 1, 2023, 3)),

  restr18_with = subset_window(df, 2018, 1, 2023, 3),
  restr18_without = drop_anticipation_quarter(subset_window(df, 2018, 1, 2023, 3))
)

# -----------------------------------------------------------------------------
# Model fitters
# -----------------------------------------------------------------------------
fit_one <- function(dsub, y) {
  if (!(y %in% names(dsub))) return(NULL)

  dsub <- dsub %>% filter(!is.na(.data[[y]]))
  if (nrow(dsub) == 0) return(NULL)
  if (length(unique(dsub[[y]])) <= 1) return(NULL)

  feols(
    fml = make_fml(y, rhs),
    data = dsub,
    vcov = vc,
    lean = TRUE
  )
}

fit_block <- function(dsub, outcomes) {
  out <- list()
  for (y in outcomes) {
    out[[y]] <- fit_one(dsub, y)
  }
  out
}

# -----------------------------------------------------------------------------
# Run models
# -----------------------------------------------------------------------------
fits <- list(
  full_with = fit_block(datasets$full_with, outcomes_full),
  full_without = fit_block(datasets$full_without, outcomes_full),

  restr17_with = fit_block(datasets$restr17_with, outcomes_2017_2023q3),
  restr17_without = fit_block(datasets$restr17_without, outcomes_2017_2023q3),

  restr18_with = fit_block(datasets$restr18_with, outcomes_2018_2023q3),
  restr18_without = fit_block(datasets$restr18_without, outcomes_2018_2023q3)
)

# -----------------------------------------------------------------------------
# Table builders
# -----------------------------------------------------------------------------
build_row <- function(mset, outcomes) {
  cells <- lapply(outcomes, function(y) {
    s <- coef_se_star_n(mset[[y]], "post")
    fmt_est_n(s$coef, s$se, s$stars, s$n)
  })
  paste(cells, collapse = "  &  ")
}

one_row_table_fragment <- function(mset, outcomes, caption, label, notes_extra = NULL) {
  n_out <- length(outcomes)
  header_labels <- paste(vapply(outcomes, pretty_outcome_label, character(1)), collapse = " & ")
  colspec <- paste0("@{} l ", paste(rep("Y", n_out), collapse = " "), " @{}")
  row_post <- build_row(mset, outcomes)

  c(
    "\\begingroup",
    "\\begin{table}[!ht]",
    "\\centering",
    "\\begin{threeparttable}",
    sprintf("\\caption{%s}", caption),
    sprintf("\\label{%s}", label),
    "\\small",
    "\\setlength{\\tabcolsep}{5pt}",
    "",
    sprintf("\\begin{tabularx}{\\textwidth}{%s}", colspec),
    "\\toprule",
    paste0(" & ", header_labels, " \\\\"),
    "\\midrule",
    paste0("Post & ", row_post, " \\\\"),
    "\\bottomrule",
    "\\end{tabularx}",
    "",
    "\\begin{tablenotes}[flushleft]",
    "\\footnotesize",
    "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post} with two-way clustered standard errors (by facility and quarter) in parentheses. The third line in each cell reports significance stars, and the fourth line reports the estimation sample size used for that outcome.",
    sprintf("\\item All specifications include facility and quarter fixed effects and covariates: %s.", escape_latex(controls_rhs)),
    "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
    if (!is.null(notes_extra)) paste0("\\item ", notes_extra) else NULL,
    "\\end{tablenotes}",
    "\\end{threeparttable}",
    "\\end{table}",
    "\\endgroup",
    ""
  )
}

# -----------------------------------------------------------------------------
# Build all tables
# -----------------------------------------------------------------------------
tab1 <- one_row_table_fragment(
  mset = fits$full_with,
  outcomes = outcomes_full,
  caption = "Preliminary TWFE Estimates of \\textit{post} on Quality Measures (2017~Q1--2024~Q2, With anticipation)",
  label = "tab:qtwfe-full-with",
  notes_extra = "Outcomes included: QM 401, 404, 406, 407, 410, 419, 434, and 452. Sample window: 2017~Q1--2024~Q2."
)

tab2 <- one_row_table_fragment(
  mset = fits$full_without,
  outcomes = outcomes_full,
  caption = "Preliminary TWFE Estimates of \\textit{post} on Quality Measures (2017~Q1--2024~Q2, Without anticipation)",
  label = "tab:qtwfe-full-without",
  notes_extra = "Without anticipation drops the quarter immediately preceding treatment (event\\_time = -1). Outcomes included: QM 401, 404, 406, 407, 410, 419, 434, and 452. Sample window: 2017~Q1--2024~Q2."
)

tab3 <- one_row_table_fragment(
  mset = fits$restr17_with,
  outcomes = outcomes_2017_2023q3,
  caption = "Preliminary TWFE Estimates of \\textit{post} on Quality Measures (2017~Q1--2023~Q3, With anticipation)",
  label = "tab:qtwfe-restr17-with",
  notes_extra = "Outcomes included: QM 405, 451, and 471. Sample window: 2017~Q1--2023~Q3."
)

tab4 <- one_row_table_fragment(
  mset = fits$restr17_without,
  outcomes = outcomes_2017_2023q3,
  caption = "Preliminary TWFE Estimates of \\textit{post} on Quality Measures (2017~Q1--2023~Q3, Without anticipation)",
  label = "tab:qtwfe-restr17-without",
  notes_extra = "Without anticipation drops the quarter immediately preceding treatment (event\\_time = -1). Outcomes included: QM 405, 451, and 471. Sample window: 2017~Q1--2023~Q3."
)

tab5 <- one_row_table_fragment(
  mset = fits$restr18_with,
  outcomes = outcomes_2018_2023q3,
  caption = "Preliminary TWFE Estimates of \\textit{post} on Quality Measure 453 (2018~Q1--2023~Q3, With anticipation)",
  label = "tab:qtwfe-restr18-with",
  notes_extra = "Outcome included: QM 453. Sample window: 2018~Q1--2023~Q3."
)

tab6 <- one_row_table_fragment(
  mset = fits$restr18_without,
  outcomes = outcomes_2018_2023q3,
  caption = "Preliminary TWFE Estimates of \\textit{post} on Quality Measure 453 (2018~Q1--2023~Q3, Without anticipation)",
  label = "tab:qtwfe-restr18-without",
  notes_extra = "Without anticipation drops the quarter immediately preceding treatment (event\\_time = -1). Outcome included: QM 453. Sample window: 2018~Q1--2023~Q3."
)

# -----------------------------------------------------------------------------
# Write standalone LaTeX document
# -----------------------------------------------------------------------------
full_doc <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{array}",
  "\\usepackage{makecell}",
  "\\usepackage{caption}",
  "\\usepackage{longtable}",
  "",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "\\newcommand{\\estn}[4]{\\makecell[c]{$#1$ \\\\ $(#2)$ \\\\ #3 \\\\ {\\scriptsize N=#4}}}",
  "",
  "\\begin{document}",
  "",
  "\\section*{Preliminary Quarterly Quality TWFE Tables}",
  "This document collects every LaTeX table produced by \\texttt{quarterly\\_quality\\_twfe\\_tables.R}.",
  "",
  tab1,
  "\\clearpage",
  tab2,
  "\\clearpage",
  tab3,
  "\\clearpage",
  tab4,
  "\\clearpage",
  tab5,
  "\\clearpage",
  tab6,
  "",
  "\\end{document}",
  ""
)

writeLines(full_doc, tex_out_fp, useBytes = TRUE)

cat("\nSaved standalone LaTeX file:\n")
cat(" - ", tex_out_fp, "\n", sep = "")
cat("\nDone.\n")
