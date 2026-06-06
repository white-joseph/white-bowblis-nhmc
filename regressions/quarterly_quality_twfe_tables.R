# C:/Repositories/white-bowblis-nhmc/regressions/quarterly_quality_twfe_tables.R
# =============================================================================
# Quarterly Quality TWFE Post Table
#
# Purpose:
#   Estimate average post-ownership-change TWFE models for selected quarterly
#   CMS quality measures.
#
# Main specification:
#   - Outcome: quarterly CMS quality metric
#   - Treatment variable: post
#   - Drops tau = 0 from the estimation sample
#   - Fixed effects: facility and year-quarter
#   - SEs clustered two ways by facility and year-quarter
#
# Table structure:
#   - Column 1: post estimate without staffing controls
#   - Column 2: post estimate with staffing controls
#   - Grouped into:
#       (1) Routine-sensitive process measures
#       (2) Resident outcome measures
#
# Output:
#   - Directly inputtable LaTeX table:
#       outputs/tables/quarterly_quality_twfe_post.tex
#
# Notes:
#   - This script does NOT create a standalone LaTeX document.
#   - The output is meant to be called directly in the paper using:
#       \input{C:/Repositories/white-bowblis-nhmc/outputs/tables/quarterly_quality_twfe_post.tex}
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(fixest)
  library(stringr)
  library(tibble)
})

options(scipen = 999, digits = 4)

# -----------------------------------------------------------------------------
# 0) Paths
# -----------------------------------------------------------------------------

project_root <- "C:/Repositories/white-bowblis-nhmc"

panel_fp   <- file.path(project_root, "data", "clean", "quality_panel.csv")
tables_dir <- file.path(project_root, "outputs", "tables")

dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)

tex_out_fp <- file.path(tables_dir, "quarterly_quality_twfe_post.tex")

# -----------------------------------------------------------------------------
# 1) Helpers
# -----------------------------------------------------------------------------

assert_has_cols <- function(df, cols, df_name = "data") {
  miss <- setdiff(cols, names(df))
  
  if (length(miss) > 0) {
    stop(
      sprintf("[%s] missing required columns: %s",
              df_name, paste(miss, collapse = ", ")),
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
  yr <- suppressWarnings(as.integer(year))
  qn <- quarter_num(quarter)
  yr * 4L + qn
}

subset_window <- function(df, start_year, start_quarter, end_year, end_quarter) {
  start_idx <- start_year * 4L + start_quarter
  end_idx   <- end_year * 4L + end_quarter
  idx <- year_quarter_index(df$year, df$quarter)
  
  df[idx >= start_idx & idx <= end_idx, , drop = FALSE]
}

drop_tau_zero <- function(df) {
  df %>%
    filter(is.na(event_time) | event_time != 0)
}

get_case_mix_controls <- function(df) {
  preferred <- intersect_existing(
    c("cm_q_state_2", "cm_q_state_3", "cm_q_state_4"),
    df
  )
  
  if (length(preferred) > 0) {
    return(preferred)
  }
  
  fallback <- intersect_existing(
    c("cm_q_nat_2", "cm_q_nat_3", "cm_q_nat_4"),
    df
  )
  
  fallback
}

get_base_controls <- function(df) {
  base_controls <- c(
    "government",
    "non_profit",
    "chain",
    "beds",
    "occupancy_rate",
    "pct_medicare",
    "pct_medicaid"
  )
  
  unique(c(
    intersect_existing(base_controls, df),
    get_case_mix_controls(df)
  ))
}

get_staffing_controls <- function(df) {
  intersect_existing(c("rn_hprd", "lpn_hprd", "cna_hprd"), df)
}

make_controls_rhs <- function(df, include_staffing = FALSE) {
  ctrls <- get_base_controls(df)
  
  if (isTRUE(include_staffing)) {
    ctrls <- unique(c(ctrls, get_staffing_controls(df)))
  }
  
  if (length(ctrls) == 0) {
    return("1")
  }
  
  paste(ctrls, collapse = " + ")
}

make_fml <- function(lhs, rhs) {
  as.formula(sprintf(
    "%s ~ %s | cms_certification_number + year_quarter",
    lhs,
    rhs
  ))
}

fit_one <- function(dsub, y, rhs, vc) {
  if (!(y %in% names(dsub))) {
    return(NULL)
  }
  
  dsub_y <- dsub %>%
    filter(!is.na(.data[[y]]))
  
  if (nrow(dsub_y) == 0) {
    return(NULL)
  }
  
  if (length(unique(dsub_y[[y]])) <= 1) {
    return(NULL)
  }
  
  feols(
    fml = make_fml(y, rhs),
    data = dsub_y,
    vcov = vc,
    lean = TRUE
  )
}

coef_se_star <- function(mod, term = "post") {
  if (is.null(mod)) {
    return(list(coef = NA_real_, se = NA_real_, p = NA_real_, stars = ""))
  }
  
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
  if (is.na(b) || is.na(se)) {
    return("\\makecell[c]{$\\,$ \\\\ $\\,$}")
  }
  
  bstr <- sprintf("%.3f", b)
  if (b > 0) {
    bstr <- paste0("\\phantom{-}", bstr)
  }
  
  sestr <- sprintf("%.3f", se)
  
  coef_part <- if (stars == "") {
    paste0("$", bstr, "$")
  } else {
    paste0("$", bstr, "^{", stars, "}$")
  }
  
  paste0("\\makecell[c]{", coef_part, " \\\\ $(",
         sestr, ")$}")
}

escape_latex <- function(x) {
  x <- gsub("\\\\", "\\\\textbackslash{}", x)
  x <- gsub("([#$%&_{}])", "\\\\\\1", x, perl = TRUE)
  x <- gsub("~", "\\\\textasciitilde{}", x, fixed = TRUE)
  x <- gsub("\\^", "\\\\textasciicircum{}", x)
  x
}

# -----------------------------------------------------------------------------
# 2) Load panel
# -----------------------------------------------------------------------------

if (!file.exists(panel_fp)) {
  stop(sprintf("Panel file not found: %s", panel_fp), call. = FALSE)
}

df <- readr::read_csv(panel_fp, show_col_types = FALSE)

required_cols <- c(
  "cms_certification_number",
  "year",
  "quarter",
  "post",
  "event_time"
)

assert_has_cols(df, required_cols, "quality_panel")

df <- df %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year = suppressWarnings(as.integer(year)),
    quarter = toupper(trimws(as.character(quarter))),
    year_quarter = as.factor(paste0(year, "_", quarter)),
    event_time = suppressWarnings(as.integer(event_time)),
    post = suppressWarnings(as.integer(post))
  )

numeric_candidates <- c(
  "beds",
  "occupancy_rate",
  "pct_medicare",
  "pct_medicaid",
  "event_time",
  "post",
  "government",
  "non_profit",
  "chain",
  "rn_hprd",
  "lpn_hprd",
  "cna_hprd",
  "cm_q_state_2",
  "cm_q_state_3",
  "cm_q_state_4",
  "cm_q_nat_2",
  "cm_q_nat_3",
  "cm_q_nat_4"
)

numeric_candidates <- intersect_existing(numeric_candidates, df)

if (length(numeric_candidates) > 0) {
  df <- df %>%
    mutate(across(all_of(numeric_candidates), ~ suppressWarnings(as.numeric(.x))))
}

vc <- ~ cms_certification_number + year_quarter

# -----------------------------------------------------------------------------
# 3) Outcomes and windows
# -----------------------------------------------------------------------------
# These mappings match the quarterly summary-statistics script.
# Lower values indicate better measured quality for all listed outcomes.

outcome_map <- tibble::tribble(
  ~section,                                  ~label,                                      ~var,      ~start_y, ~start_q, ~end_y, ~end_q,
  
  "Labor Saving Mechanisms",      "Catheter Use",                              "qm_406",  2017L,   1L,       2024L,  2L,
  "Labor Saving Mechanisms",      "Antipsychotic Medication Use",              "qm_419",  2017L,   1L,       2024L,  2L,
  "Labor Saving Mechanisms",      "Anti-Anxiety or Hypnotic Medication Use",   "qm_452",  2017L,   1L,       2024L,  2L,
  
  "Resident Outcome Measures",               "Pressure Injuries",                         "qm_453",  2018L,   1L,       2023L,  3L,
  "Resident Outcome Measures",               "Falls with Major Injury",                   "qm_410",  2017L,   1L,       2024L,  2L,
  "Resident Outcome Measures",               "Weight Loss",                               "qm_404",  2017L,   1L,       2024L,  2L,
  "Resident Outcome Measures",               "Decline in Physical Functioning",           "qm_401",  2017L,   1L,       2024L,  2L,
  "Resident Outcome Measures",               "Urinary Tract Infections",                  "qm_407",  2017L,   1L,       2024L,  2L
)

missing_outcomes <- setdiff(outcome_map$var, names(df))

if (length(missing_outcomes) > 0) {
  stop(
    sprintf(
      "These requested outcomes are missing from quality_panel.csv: %s",
      paste(missing_outcomes, collapse = ", ")
    ),
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# 4) Run TWFE post models
# -----------------------------------------------------------------------------

results <- vector("list", nrow(outcome_map))

for (i in seq_len(nrow(outcome_map))) {
  row_i <- outcome_map[i, ]
  
  cat("\n", strrep("=", 80), "\n", sep = "")
  cat("OUTCOME: ", row_i$var, " — ", row_i$label, "\n", sep = "")
  cat(strrep("=", 80), "\n", sep = "")
  
  dsub <- df %>%
    subset_window(
      start_year    = row_i$start_y,
      start_quarter = row_i$start_q,
      end_year      = row_i$end_y,
      end_quarter   = row_i$end_q
    ) %>%
    drop_tau_zero()
  
  rhs_no_staff <- make_controls_rhs(dsub, include_staffing = FALSE)
  rhs_staff    <- make_controls_rhs(dsub, include_staffing = TRUE)
  
  fit_no_staff <- fit_one(
    dsub = dsub,
    y = row_i$var,
    rhs = paste("post +", rhs_no_staff),
    vc = vc
  )
  
  fit_staff <- fit_one(
    dsub = dsub,
    y = row_i$var,
    rhs = paste("post +", rhs_staff),
    vc = vc
  )
  
  n_no_staff <- dsub %>%
    filter(!is.na(.data[[row_i$var]])) %>%
    nrow()
  
  n_staff <- n_no_staff
  
  c1 <- coef_se_star(fit_no_staff, term = "post")
  c2 <- coef_se_star(fit_staff, term = "post")
  
  results[[i]] <- tibble(
    section = row_i$section,
    label = row_i$label,
    var = row_i$var,
    est_no_staff = fmt_est(c1$coef, c1$se, c1$stars),
    est_staff = fmt_est(c2$coef, c2$se, c2$stars),
    n_no_staff = n_no_staff,
    n_staff = n_staff,
    window = paste0(row_i$start_y, "Q", row_i$start_q, "--",
                    row_i$end_y, "Q", row_i$end_q)
  )
  
  cat("N = ", format(n_no_staff, big.mark = ","), "\n", sep = "")
}

res_tbl <- bind_rows(results)

# -----------------------------------------------------------------------------
# 5) Build directly inputtable LaTeX table
# -----------------------------------------------------------------------------

build_rows <- function(tbl) {
  out <- c()
  sections <- unique(tbl$section)
  
  for (sec in sections) {
    sec_tbl <- tbl %>%
      filter(section == sec)
    
    out <- c(
      out,
      paste0("\\multicolumn{3}{@{}l}{\\textbf{", sec, "}} \\\\[2pt]")
    )
    
    sec_lines <- sec_tbl %>%
      transmute(
        line = paste0(
          escape_latex(label),
          " & ",
          est_no_staff,
          " & ",
          est_staff,
          " \\\\"
        )
      ) %>%
      pull(line)
    
    out <- c(out, sec_lines, "\\addlinespace[4pt]")
  }
  
  out
}

body_lines <- build_rows(res_tbl)

sample_note <- paste(
  res_tbl %>%
    transmute(
      txt = paste0(
        label,
        " [",
        window,
        "; N=",
        format(n_no_staff, big.mark = ","),
        "]"
      )
    ) %>%
    pull(txt),
  collapse = "; "
)

controls_no_staff <- make_controls_rhs(df, include_staffing = FALSE)
controls_staff    <- make_controls_rhs(df, include_staffing = TRUE)

tex_lines <- c(
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Two-Way Fixed Effects Estimates of Ownership Change on Quality Measures}",
  "\\label{tab:quality-twfe-post}",
  "\\small",
  "\\setlength{\\tabcolsep}{8pt}",
  "",
  "\\begin{tabularx}{\\textwidth}{@{} l >{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X @{}}",
  "\\toprule",
  "\\textbf{Quality Measure} & \\textbf{Without Staffing Controls} & \\textbf{With Staffing Controls} \\\\",
  "\\midrule",
  body_lines,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{Post}, with two-way clustered standard errors by facility and calendar quarter in parentheses.",
  "\\item All specifications include facility fixed effects, year-quarter fixed effects, and time-varying facility controls. The ownership-change quarter, $\\tau=0$, is excluded from the estimation sample.",
  "\\item The first column excludes staffing controls. The second column additionally controls for RN, LPN, and CNA hours per resident day. Because staffing may be affected by ownership change, staffing-controlled estimates are interpreted descriptively rather than as preferred total effects.",
  "\\item Lower values indicate better measured quality for all listed quality measures.",
  paste0("\\item Non-staffing controls: ", escape_latex(controls_no_staff), "."),
  paste0("\\item Staffing-controlled specification: ", escape_latex(controls_staff), "."),
  paste0("\\item Outcome-specific windows and estimation sample sizes: ", escape_latex(sample_note), "."),
  "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  ""
)

writeLines(tex_lines, tex_out_fp, useBytes = TRUE)

cat("\nSaved directly inputtable LaTeX table:\n")
cat(" - ", tex_out_fp, "\n", sep = "")
cat("\nUse in paper with:\n")
cat("\\input{C:/Repositories/white-bowblis-nhmc/outputs/tables/quarterly_quality_twfe_post.tex}\n")
cat("\nDone.\n")