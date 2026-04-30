# summary_stats_quality_write_tex.R
# Writes LaTeX summary statistics table for quarterly quality panel:
#   - outputs/tables/quality_summary_statistics.tex        (full doc)
#   - outputs/tables/quality_summary_statistics_code.tex   (table fragment only)

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(purrr)
  library(stringr)
  library(tibble)
})

options(scipen = 999, digits = 3)

# ---- Paths ----
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/quality_panel.csv"
out_dir  <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Load panel ----
df <- read_csv(panel_fp, show_col_types = FALSE) %>%
  mutate(
    cms_certification_number = as.character(cms_certification_number),
    year = suppressWarnings(as.integer(year)),
    quarter = toupper(trimws(as.character(quarter))),
    year_quarter = paste0(year, quarter)
  )

# ---- Overview for notes ----
treated_var <- if ("treated" %in% names(df)) "treated" else if ("treatment" %in% names(df)) "treatment" else NA_character_

treated_ccns <- if (!is.na(treated_var)) {
  n_distinct(df$cms_certification_number[df[[treated_var]] %in% c(1, "1")])
} else {
  NA_integer_
}

overview <- tibble(
  rows = nrow(df),
  ccns = n_distinct(df$cms_certification_number),
  treated_ccns = treated_ccns,
  min_year_quarter = suppressWarnings(min(df$year_quarter, na.rm = TRUE)),
  max_year_quarter = suppressWarnings(max(df$year_quarter, na.rm = TRUE))
)

avg_quarters_per_ccn <- df %>%
  distinct(cms_certification_number, year, quarter) %>%
  count(cms_certification_number, name = "quarters") %>%
  summarize(avg_quarters = mean(quarters, na.rm = TRUE)) %>%
  pull(avg_quarters)

overview$avg_quarters_per_ccn <- avg_quarters_per_ccn

# ---- Helpers ----
to_num <- function(x) suppressWarnings(as.numeric(x))

summarize_mean_sd <- function(x) {
  x <- to_num(x)
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(tibble(Mean = NA_real_, SD = NA_real_))
  }
  tibble(
    Mean = mean(x),
    SD   = sd(x)
  )
}

fmt_int <- function(x) format(x, big.mark = ",", trim = TRUE, scientific = FALSE)
fmt_dec <- function(x, k = 3) ifelse(is.na(x), "NA", formatC(x, format = "f", digits = k))
fmt_pct1 <- function(x) ifelse(is.na(x), "NA", formatC(x, format = "f", digits = 1))

digits_for <- function(var) {
  if (str_detect(var, "^qm_")) return(3)
  if (var %in% c("rn_hprd","lpn_hprd","cna_hprd","total_hprd")) return(3)
  if (var %in% c("occupancy_rate","pct_medicare","pct_medicaid")) return(1)
  if (var %in% c("beds")) return(1)
  if (var %in% c("government","non_profit","chain","cm_q_state_2","cm_q_state_3","cm_q_state_4")) return(3)
  3
}

pretty_qm <- function(x) {
  code <- str_replace(x, "^qm_", "")
  paste0("QM ", code)
}

pretty_name <- c(
  government     = "Government (dummy)",
  non_profit     = "Non-profit (dummy)",
  chain          = "Chain affiliation (dummy)",
  beds           = "Beds",
  occupancy_rate = "Occupancy rate (\\%)",
  pct_medicare   = "\\% Medicare",
  pct_medicaid   = "\\% Medicaid",
  cm_q_state_2   = "Acuity quartile 2 (state-quarter)",
  cm_q_state_3   = "Acuity quartile 3 (state-quarter)",
  cm_q_state_4   = "Acuity quartile 4 (state-quarter)",
  rn_hprd        = "RN HPRD",
  lpn_hprd       = "LPN HPRD",
  cna_hprd       = "CNA HPRD",
  total_hprd     = "Total HPRD"
)

# ---- Define variables ----
# Main quality outcomes
preferred_qm <- c(
  "qm_401","qm_404","qm_406","qm_407",
  "qm_410","qm_419","qm_434","qm_452",
  "qm_405","qm_451","qm_471","qm_453"
)

panelA_vars <- intersect(preferred_qm, names(df))

# Optional staffing rows if you want them in Panel B
staffing_vars <- c("rn_hprd","lpn_hprd","cna_hprd","total_hprd")
staffing_vars <- intersect(staffing_vars, names(df))

panelB_vars <- c(
  staffing_vars,
  intersect(c(
    "government","non_profit","chain",
    "beds","occupancy_rate","pct_medicare","pct_medicaid",
    "cm_q_state_2","cm_q_state_3","cm_q_state_4"
  ), names(df))
)

# ---- Build summary rows ----
make_panel_rows <- function(vars, panel_title) {
  if (length(vars) == 0) return(character(0))
  
  tbl <- purrr::map_dfr(vars, function(v) {
    s <- summarize_mean_sd(df[[v]])
    tibble(
      variable = v,
      Mean = s$Mean,
      SD = s$SD
    )
  }) %>%
    rowwise() %>%
    mutate(
      VarLabel = if (str_detect(variable, "^qm_")) {
        pretty_qm(variable)
      } else if (variable %in% names(pretty_name)) {
        unname(pretty_name[variable])
      } else {
        variable
      },
      MeanStr = if (variable %in% c("occupancy_rate","pct_medicare","pct_medicaid")) {
        fmt_pct1(Mean)
      } else {
        fmt_dec(Mean, digits_for(variable))
      },
      SDStr = if (variable %in% c("occupancy_rate","pct_medicare","pct_medicaid")) {
        fmt_pct1(SD)
      } else {
        fmt_dec(SD, digits_for(variable))
      }
    ) %>%
    ungroup()
  
  c(
    paste0("\\multicolumn{3}{@{}l}{\\textbf{", panel_title, "}} \\\\[2pt]"),
    tbl %>%
      transmute(line = paste0(VarLabel, " & ", MeanStr, " & ", SDStr, " \\\\")) %>%
      pull(line)
  )
}

panelA_lines <- make_panel_rows(panelA_vars, "Panel A: Quality outcome variables")
panelB_lines <- make_panel_rows(panelB_vars, "Panel B: Staffing and control variables")

# ---- Strings for notes ----
rows_str   <- fmt_int(overview$rows)
ccns_str   <- fmt_int(overview$ccns)
trt_str    <- ifelse(is.na(overview$treated_ccns), "NA", fmt_int(overview$treated_ccns))
period_str <- paste0(overview$min_year_quarter, "--", overview$max_year_quarter)
avgq_str   <- fmt_dec(overview$avg_quarters_per_ccn, 1)

# ---- Notes ----
notes_line <- paste0(
  "\\item \\textit{Notes:} The unit of observation is facility--quarter. ",
  "QM denotes a CMS quality metric. ",
  "When included, RN, LPN, and CNA denote registered nurses, licensed practical nurses, and certified nursing assistants, respectively. ",
  "HPRD denotes hours per resident day. ",
  "Occupancy rate is the ratio of residents to certified beds (percent). ",
  "\\% Medicare and \\% Medicaid are the shares of residents covered by Medicare and Medicaid, respectively. ",
  "Government and Non-profit are ownership-type indicator variables (for-profit omitted category). ",
  "Chain affiliation is an indicator for membership in a multi-facility chain. ",
  "Acuity quartiles are state--quarter quartiles of resident acuity (case-mix) with quartile 1 omitted. ",
  "Rows $=$ ", rows_str,
  "; Facilities $=$ ", ccns_str,
  "; Treated facilities $=$ ", trt_str,
  "; Period $=$ ", period_str,
  "; Average quarters per facility $=$ ", avgq_str, "."
)

# ---- Table fragment ----
fragment <- c(
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Summary Statistics}",
  "\\label{tab:quality_sumstats}",
  "\\small",
  "\\setlength{\\tabcolsep}{8pt}",
  "",
  "\\begin{tabularx}{\\textwidth}{@{} l r r @{} }",
  "\\toprule",
  "\\textbf{Variable} & \\textbf{Mean} & \\textbf{SD} \\\\",
  "\\midrule",
  panelA_lines,
  "\\addlinespace[0.8em]",
  panelB_lines,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  notes_line,
  "\\end{tablenotes}",
  "",
  "\\end{threeparttable}",
  "\\end{table}"
)

# ---- Full document wrapper ----
full_doc <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{array}",
  "\\usepackage{caption}",
  "\\captionsetup{labelfont=bf, font=small}",
  "",
  "\\begin{document}",
  fragment,
  "\\end{document}"
)

# ---- Write outputs ----
full_path <- file.path(out_dir, "quality_summary_statistics.tex")
frag_path <- file.path(out_dir, "quality_summary_statistics_code.tex")

writeLines(full_doc, full_path, useBytes = TRUE)
writeLines(fragment, frag_path, useBytes = TRUE)

cat("Wrote:\n - ", normalizePath(full_path, winslash = "\\"),
    "\n - ", normalizePath(frag_path, winslash = "\\"), "\n", sep = "")