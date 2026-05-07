# =============================================================================
# quarterly_quality_twfe_mechanism_table.R
#
# Builds a standalone LaTeX table with selected quarterly quality outcomes:
#   - Column 1: post estimate without staffing controls
#   - Column 2: post estimate with staffing controls
#   - Grouped into "Labor Saving Mechanisms" and "Outcomes"
#
# Sample restriction:
#   - exclude tau = 0  (drop event_time == 0)
#
# Output:
#   C:/Repositories/white-bowblis-nhmc/tables/quarterly_quality_mechanism_table.tex
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(fixest)
  library(stringr)
  library(tibble)
})

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
project_root <- "C:/Repositories/white-bowblis-nhmc"
panel_fp     <- file.path(project_root, "data", "clean", "quality_panel.csv")
tables_dir   <- file.path(project_root, "outputs/tables")

if (!dir.exists(tables_dir)) dir.create(tables_dir, recursive = TRUE)

tex_out_fp <- file.path(tables_dir, "quarterly_quality_mechanism_table.tex")

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

drop_tau_zero <- function(df) {
  df %>% filter(is.na(event_time) | event_time != 0)
}

get_case_mix_controls <- function(df) {
  preferred <- intersect_existing(c("cm_q_state_2", "cm_q_state_3", "cm_q_state_4"), df)
  if (length(preferred) > 0) return(preferred)
  
  fallback <- intersect_existing(c("cm_q_nat_2", "cm_q_nat_3", "cm_q_nat_4"), df)
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
  c(intersect_existing(base_controls, df), get_case_mix_controls(df))
}

get_staffing_controls <- function(df) {
  intersect_existing(c("rn_hprd", "lpn_hprd", "cna_hprd"), df)
}

make_controls_rhs <- function(df, include_staffing = FALSE) {
  ctrls <- get_base_controls(df)
  if (include_staffing) {
    ctrls <- c(ctrls, get_staffing_controls(df))
  }
  if (length(ctrls) == 0) return("1")
  paste(ctrls, collapse = " + ")
}

make_fml <- function(lhs, rhs) {
  as.formula(sprintf("%s ~ %s | cms_certification_number + year_quarter", lhs, rhs))
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

escape_latex <- function(x) {
  x <- gsub("\\\\", "\\\\textbackslash{}", x)
  x <- gsub("([#$%&_{}])", "\\\\\\1", x, perl = TRUE)
  x <- gsub("~", "\\\\textasciitilde{}", x, fixed = TRUE)
  x <- gsub("\\^", "\\\\textasciicircum{}", x)
  x
}

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
  "event_time",
  "post",
  "government",
  "non_profit",
  "chain",
  "rn_hprd",
  "lpn_hprd",
  "cna_hprd"
)
numeric_candidates <- intersect_existing(numeric_candidates, df)

if (length(numeric_candidates) > 0) {
  df <- df %>%
    mutate(across(all_of(numeric_candidates), ~ suppressWarnings(as.numeric(.x))))
}

vc <- ~ cms_certification_number + year_quarter

# -----------------------------------------------------------------------------
# Outcomes and windows
# -----------------------------------------------------------------------------
outcome_map <- tribble(
  ~section,                    ~label,                         ~var,      ~start_y, ~start_q, ~end_y, ~end_q,
  "Labor Saving Mechanisms",   "Catheter",                     "qm_406",  2017L,    1L,       2024L,  2L,
  "Labor Saving Mechanisms",   "Antipsychotic",                "qm_419",  2017L,    1L,       2024L,  2L,
  "Labor Saving Mechanisms",   "Hypnotics",                    "qm_452",  2017L,    1L,       2024L,  2L,
  
  "Outcomes",                  "Pressure injuries",            "qm_453",  2018L,    1L,       2023L,  3L,
  "Outcomes",                  "Falls with major injury",      "qm_410",  2017L,    1L,       2024L,  2L,
  "Outcomes",                  "Weight Loss",                  "qm_404",  2017L,    1L,       2024L,  2L,
  "Outcomes",                  "ADL Increase",                 "qm_401",  2017L,    1L,       2024L,  2L,
  "Outcomes",                  "Urinary Tract Infections",     "qm_407",  2017L,    1L,       2024L,  2L
)

missing_outcomes <- setdiff(outcome_map$var, names(df))
if (length(missing_outcomes) > 0) {
  stop(
    sprintf("These requested outcomes are missing from quality_panel.csv: %s",
            paste(missing_outcomes, collapse = ", ")),
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# Run models
# -----------------------------------------------------------------------------
results <- vector("list", nrow(outcome_map))

for (i in seq_len(nrow(outcome_map))) {
  row_i <- outcome_map[i, ]
  
  dsub <- df %>%
    subset_window(row_i$start_y, row_i$start_q, row_i$end_y, row_i$end_q) %>%
    drop_tau_zero()
  
  rhs_no_staff <- make_controls_rhs(dsub, include_staffing = FALSE)
  rhs_staff    <- make_controls_rhs(dsub, include_staffing = TRUE)
  
  fit_no_staff <- fit_one(dsub, row_i$var, paste("post +", rhs_no_staff), vc)
  fit_staff    <- fit_one(dsub, row_i$var, paste("post +", rhs_staff), vc)
  
  n_no_staff <- dsub %>% filter(!is.na(.data[[row_i$var]])) %>% nrow()
  n_staff    <- dsub %>% filter(!is.na(.data[[row_i$var]])) %>% nrow()
  
  results[[i]] <- tibble(
    section = row_i$section,
    label   = row_i$label,
    var     = row_i$var,
    est_no_staff = fmt_est(
      coef_se_star(fit_no_staff)$coef,
      coef_se_star(fit_no_staff)$se,
      coef_se_star(fit_no_staff)$stars
    ),
    est_staff = fmt_est(
      coef_se_star(fit_staff)$coef,
      coef_se_star(fit_staff)$se,
      coef_se_star(fit_staff)$stars
    ),
    n_no_staff = n_no_staff,
    n_staff = n_staff,
    window = paste0(row_i$start_y, "Q", row_i$start_q, "--", row_i$end_y, "Q", row_i$end_q)
  )
}

res_tbl <- bind_rows(results)

# -----------------------------------------------------------------------------
# Build LaTeX table
# -----------------------------------------------------------------------------
build_rows <- function(tbl) {
  out <- c()
  sections <- unique(tbl$section)
  
  for (sec in sections) {
    sec_tbl <- tbl %>% filter(section == sec)
    
    out <- c(
      out,
      paste0("\\multicolumn{3}{@{}l}{\\textbf{", sec, "}} \\\\[2pt]")
    )
    
    sec_lines <- sec_tbl %>%
      transmute(
        line = paste0(label, " & ", est_no_staff, " & ", est_staff, " \\\\")
      ) %>%
      pull(line)
    
    out <- c(out, sec_lines, "\\addlinespace[4pt]")
  }
  
  out
}

body_lines <- build_rows(res_tbl)

ns_note <- paste(
  res_tbl %>%
    transmute(txt = paste0(label, " [", window, "; N=", format(n_no_staff, big.mark = ","), "]")) %>%
    pull(txt),
  collapse = "; "
)

controls_no_staff <- make_controls_rhs(df, include_staffing = FALSE)
controls_staff    <- make_controls_rhs(df, include_staffing = TRUE)

full_doc <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{array}",
  "\\usepackage{makecell}",
  "\\usepackage{caption}",
  "\\usepackage{newtxtext}",
  "\\usepackage{newtxmath}",
  "\\captionsetup{labelfont=bf, font=small}",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "\\newcommand{\\est}[2]{\\makecell[c]{#1 \\\\ #2}}",
  "",
  "\\begin{document}",
  "",
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Two-Way Fixed Effects Estimates of \\textit{post} on Selected Quality Metrics}",
  "\\label{tab:qtwfe_mechanisms}",
  "\\small",
  "\\setlength{\\tabcolsep}{8pt}",
  "",
  "\\begin{tabularx}{\\textwidth}{@{} l Y Y @{} }",
  "\\toprule",
  "\\textbf{Metric} & \\textbf{w/o staffing as a control} & \\textbf{w/ staffing as a control} \\\\",
  "\\midrule",
  body_lines,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post} with two-way clustered standard errors (by facility and quarter) in parentheses.",
  "\\item All specifications exclude observations with $\\tau = 0$.",
  "\\item The first column of estimates includes standard controls only. The second column additionally includes RN, LPN, and CNA HPRD as staffing controls.",
  paste0("\\item Standard controls: ", escape_latex(controls_no_staff), "."),
  paste0("\\item Staffing-controlled specification: ", escape_latex(controls_staff), "."),
  paste0("\\item Outcome-specific windows and estimation sample sizes: ", escape_latex(ns_note), "."),
  "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\end{document}",
  ""
)

writeLines(full_doc, tex_out_fp, useBytes = TRUE)

cat("\nSaved standalone LaTeX file:\n")
cat(" - ", tex_out_fp, "\n", sep = "")
cat("\nDone.\n")