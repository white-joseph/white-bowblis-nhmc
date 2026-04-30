# =============================================================================
# quarterly_quality_twfe_tables_panel_style.R
#
# Quarterly quality TWFE post regressions
# Styled to mirror staffing TWFE tables:
#   - columns = outcomes
#   - Panel A = with tau = -1
#   - Panel B = without tau = -1
#   - rows inside each panel = Metric, Log(Metric)
#
# Output:
#   C:/Repositories/white-bowblis-nhmc/tables/quarterly_quality_twfe_panel_style.tex
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
panel_fp     <- file.path(project_root, "data", "clean", "quality_panel.csv")
tables_dir   <- file.path(project_root, "tables")

if (!dir.exists(tables_dir)) dir.create(tables_dir, recursive = TRUE)

tex_out_fp <- file.path(tables_dir, "quarterly_quality_twfe_panel_style.tex")

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

drop_tau_minus1 <- function(df) {
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

escape_latex <- function(x) {
  x <- gsub("\\\\", "\\\\textbackslash{}", x)
  x <- gsub("([#$%&_{}])", "\\\\\\1", x, perl = TRUE)
  x <- gsub("~", "\\\\textasciitilde{}", x, fixed = TRUE)
  x <- gsub("\\^", "\\\\textasciicircum{}", x)
  x
}

pretty_outcome_label <- function(x) {
  code <- str_replace(x, "^qm_", "")
  paste0("QM ", code)
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
# Outcome windows
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

# -----------------------------------------------------------------------------
# Logged variables
# -----------------------------------------------------------------------------
make_log_quality_vars <- function(df, outcomes) {
  out <- df
  for (y in outcomes) {
    ln_name <- paste0("ln_", y)
    out[[ln_name]] <- ifelse(is.na(out[[y]]), NA_real_, log(out[[y]] + 1))
  }
  out
}

# -----------------------------------------------------------------------------
# Fit helpers
# -----------------------------------------------------------------------------
fit_one <- function(dsub, y, rhs, vc) {
  if (!(y %in% names(dsub))) return(NULL)
  
  dsub_y <- dsub %>% filter(!is.na(.data[[y]]))
  if (nrow(dsub_y) == 0) return(NULL)
  if (length(unique(dsub_y[[y]])) <= 1) return(NULL)
  
  feols(
    fml = make_fml(y, rhs),
    data = dsub_y,
    vcov = vc,
    lean = TRUE
  )
}

fit_block_levels_logs <- function(dsub, outcomes, rhs, vc) {
  dsub <- make_log_quality_vars(dsub, outcomes)
  
  res <- list(level = list(), log = list(), n_level = list(), n_log = list())
  
  for (y in outcomes) {
    d_level <- dsub %>% filter(!is.na(.data[[y]]))
    res$n_level[[y]] <- nrow(d_level)
    if (nrow(d_level) > 0 && length(unique(d_level[[y]])) > 1) {
      res$level[[y]] <- feols(
        fml = make_fml(y, rhs),
        data = d_level,
        vcov = vc,
        lean = TRUE
      )
    } else {
      res$level[[y]] <- NULL
    }
    
    lncol <- paste0("ln_", y)
    d_log <- dsub %>% filter(!is.na(.data[[lncol]]))
    res$n_log[[y]] <- nrow(d_log)
    if (nrow(d_log) > 0 && length(unique(d_log[[lncol]])) > 1) {
      res$log[[y]] <- feols(
        fml = make_fml(lncol, rhs),
        data = d_log,
        vcov = vc,
        lean = TRUE
      )
    } else {
      res$log[[y]] <- NULL
    }
  }
  
  res
}

coef_se_star <- function(mod, term = "post") {
  if (is.null(mod)) return(list(coef = NA_real_, se = NA_real_, p = NA_real_, stars = ""))
  
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

fmt_est <- function(b, se, stars) {
  if (is.na(b) || is.na(se)) return("\\est{$\\,$}{$\\,$}")
  
  bstr <- sprintf("%.3f", b)
  if (b > 0) bstr <- paste0("\\phantom{-}", bstr)
  sestr <- sprintf("%.3f", se)
  
  coef_part <- if (stars == "") {
    paste0("$", bstr, "$")
  } else {
    paste0("$", bstr, "^{", stars, "}$")
  }
  
  sprintf("\\est{%s}{$(%s)$}", coef_part, sestr)
}

build_row <- function(mset, outcomes) {
  cells <- lapply(outcomes, function(y) {
    s <- coef_se_star(mset[[y]])
    fmt_est(s$coef, s$se, s$stars)
  })
  paste(cells, collapse = " & ")
}

# -----------------------------------------------------------------------------
# Table builder
# -----------------------------------------------------------------------------
two_panel_quality_table <- function(res_with, res_without, outcomes,
                                    caption, label, notes_extra = NULL,
                                    landscape = FALSE) {
  
  header_labels <- paste(vapply(outcomes, pretty_outcome_label, character(1)), collapse = " & ")
  colspec <- paste0("@{} l ", paste(rep("Y", length(outcomes)), collapse = " "), " @{}")
  
  rowA1 <- build_row(res_with$level, outcomes)
  rowA2 <- build_row(res_with$log, outcomes)
  rowB1 <- build_row(res_without$level, outcomes)
  rowB2 <- build_row(res_without$log, outcomes)
  
  Ns_with_level <- paste(vapply(outcomes, function(y) format(res_with$n_level[[y]], big.mark = ","), character(1)), collapse = ", ")
  Ns_with_log   <- paste(vapply(outcomes, function(y) format(res_with$n_log[[y]], big.mark = ","), character(1)), collapse = ", ")
  Ns_wo_level   <- paste(vapply(outcomes, function(y) format(res_without$n_level[[y]], big.mark = ","), character(1)), collapse = ", ")
  Ns_wo_log     <- paste(vapply(outcomes, function(y) format(res_without$n_log[[y]], big.mark = ","), character(1)), collapse = ", ")
  
  open_landscape  <- if (landscape) "\\begin{landscape}" else NULL
  close_landscape <- if (landscape) "\\end{landscape}" else NULL
  
  c(
    "\\begingroup",
    open_landscape,
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
    paste0(" & \\multicolumn{", length(outcomes), "}{c}{\\textbf{Outcomes}} \\\\"),
    sprintf("\\cmidrule(lr){2-%d}", length(outcomes) + 1),
    paste0(" & ", header_labels, " \\\\"),
    "\\midrule",
    sprintf("\\multicolumn{%d}{@{}l}{\\textbf{Panel A: With $\\tau=-1$}} \\\\[2pt]", length(outcomes) + 1),
    paste0("Metric & ", rowA1, " \\\\"),
    paste0("Log(Metric) & ", rowA2, " \\\\"),
    "\\addlinespace[4pt]",
    sprintf("\\multicolumn{%d}{@{}l}{\\textbf{Panel B: Without $\\tau=-1$}} \\\\[2pt]", length(outcomes) + 1),
    paste0("Metric & ", rowB1, " \\\\"),
    paste0("Log(Metric) & ", rowB2, " \\\\"),
    "\\bottomrule",
    "\\end{tabularx}",
    "",
    "\\begin{tablenotes}[flushleft]",
    "\\footnotesize",
    "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post} with two-way clustered standard errors (by facility and quarter) in parentheses. Panel~A uses the sample with $\\tau=-1$ retained. Panel~B excludes observations with $\\tau=-1$. Rows report levels and log-transformed quality metrics, where logs are computed as $\\log(1+y)$.",
    paste0("\\item Sample sizes by outcome. Panel~A levels: [", Ns_with_level, "]. Panel~A logs: [", Ns_with_log, "]. Panel~B levels: [", Ns_wo_level, "]. Panel~B logs: [", Ns_wo_log, "]."),
    sprintf("\\item All specifications include facility and quarter fixed effects and covariates: %s.", escape_latex(controls_rhs)),
    "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
    if (!is.null(notes_extra)) paste0("\\item ", notes_extra) else NULL,
    "\\end{tablenotes}",
    "\\end{threeparttable}",
    "\\end{table}",
    close_landscape,
    "\\endgroup",
    ""
  )
}

# -----------------------------------------------------------------------------
# Build datasets
# -----------------------------------------------------------------------------
df_full_with    <- subset_window(df, 2017, 1, 2024, 2)
df_full_without <- drop_tau_minus1(df_full_with)

df_2017q3_with    <- subset_window(df, 2017, 1, 2023, 3)
df_2017q3_without <- drop_tau_minus1(df_2017q3_with)

df_2018q3_with    <- subset_window(df, 2018, 1, 2023, 3)
df_2018q3_without <- drop_tau_minus1(df_2018q3_with)

# -----------------------------------------------------------------------------
# Fit models
# -----------------------------------------------------------------------------
fits_full_with    <- fit_block_levels_logs(df_full_with, outcomes_full, rhs, vc)
fits_full_without <- fit_block_levels_logs(df_full_without, outcomes_full, rhs, vc)

fits_2017q3_with    <- fit_block_levels_logs(df_2017q3_with, outcomes_2017_2023q3, rhs, vc)
fits_2017q3_without <- fit_block_levels_logs(df_2017q3_without, outcomes_2017_2023q3, rhs, vc)

fits_2018q3_with    <- fit_block_levels_logs(df_2018q3_with, outcomes_2018_2023q3, rhs, vc)
fits_2018q3_without <- fit_block_levels_logs(df_2018q3_without, outcomes_2018_2023q3, rhs, vc)

# -----------------------------------------------------------------------------
# Build LaTeX tables
# -----------------------------------------------------------------------------
tab1 <- two_panel_quality_table(
  res_with    = fits_full_with,
  res_without = fits_full_without,
  outcomes    = outcomes_full,
  caption     = "Two-Way Fixed Effects Estimates of \\textit{post} on Quality Outcomes (2017~Q1--2024~Q2)",
  label       = "tab:qtwfe_full",
  notes_extra = "Outcomes included: QM 401, 404, 406, 407, 410, 419, 434, and 452.",
  landscape   = FALSE
)

tab2 <- two_panel_quality_table(
  res_with    = fits_2017q3_with,
  res_without = fits_2017q3_without,
  outcomes    = outcomes_2017_2023q3,
  caption     = "Two-Way Fixed Effects Estimates of \\textit{post} on Quality Outcomes (2017~Q1--2023~Q3)",
  label       = "tab:qtwfe_2017_2023q3",
  notes_extra = "Outcomes included: QM 405, 451, and 471.",
  landscape   = FALSE
)

tab3 <- two_panel_quality_table(
  res_with    = fits_2018q3_with,
  res_without = fits_2018q3_without,
  outcomes    = outcomes_2018_2023q3,
  caption     = "Two-Way Fixed Effects Estimates of \\textit{post} on Quality Outcome QM 453 (2018~Q1--2023~Q3)",
  label       = "tab:qtwfe_453",
  notes_extra = "Outcome included: QM 453.",
  landscape   = FALSE
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
  "\\usepackage{pdflscape}",
  "\\usepackage{newtxtext}",
  "\\usepackage{newtxmath}",
  "\\captionsetup{labelfont=bf, font=small}",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "\\newcommand{\\est}[2]{\\makecell[c]{#1 \\\\ #2}}",
  "",
  "\\begin{document}",
  "",
  tab1,
  tab2,
  "\\clearpage",
  tab3,
  "",
  "\\end{document}",
  ""
)

writeLines(full_doc, tex_out_fp, useBytes = TRUE)

cat("\nSaved standalone LaTeX file:\n")
cat(" - ", tex_out_fp, "\n", sep = "")
cat("\nDone.\n")