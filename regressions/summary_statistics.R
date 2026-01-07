# summary_stats_write_tex.R
# Writes LaTeX summary statistics table:
#   - outputs/tables/summary_statistics.tex        (full doc)
#   - outputs/tables/summary_statistics_code.tex   (table fragment only)
#
# NEW FORMAT:
#   - One table, two panels inside same tabular:
#       Panel A: Outcome variables (RN/LPN/CNA/Total HPPD)
#       Panel B: Control variables (ownership type dummies, chain, beds, occupancy, payer mix, acuity quartiles)
#   - Only Mean and SD columns
#   - No gap/coverage ratio
#   - Ownership/chain summarized at observation level (not CCN level)
#   - Sample sizes in table notes

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
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_dir  <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Load panel ----
df <- read_csv(panel_fp, show_col_types = FALSE) %>%
  mutate(
    cms_certification_number = as.character(cms_certification_number),
    year_month = as.character(year_month)
  )

# ---- Overview for notes ----
overview <- tibble(
  rows = nrow(df),
  ccns = n_distinct(df$cms_certification_number),
  treated_ccns = n_distinct(df$cms_certification_number[df$treatment %in% c(1, "1")]),
  min_year_month = suppressWarnings(min(df$year_month, na.rm = TRUE)),
  max_year_month = suppressWarnings(max(df$year_month, na.rm = TRUE))
)

avg_months_per_ccn <- df %>%
  distinct(cms_certification_number, year_month) %>%
  count(cms_certification_number, name = "months") %>%
  summarize(avg_months = mean(months, na.rm = TRUE)) %>%
  pull(avg_months)

overview$avg_months_per_ccn <- avg_months_per_ccn

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

# digits by variable
digits_for <- function(var) {
  if (var %in% c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")) return(3)
  if (var %in% c("occupancy_rate","pct_medicare","pct_medicaid")) return(1)
  if (var %in% c("beds")) return(1)
  # binaries + quartiles
  if (var %in% c("government","non_profit","chain","cm_q_state_2","cm_q_state_3","cm_q_state_4")) return(3)
  3
}

pretty_name <- c(
  rn_hppd        = "RN HPPD",
  lpn_hppd       = "LPN HPPD",
  cna_hppd       = "CNA HPPD",
  total_hppd     = "Total HPPD",
  
  government     = "Government (dummy)",
  non_profit     = "Non-profit (dummy)",
  chain          = "Chain affiliation (dummy)",
  
  beds           = "Beds",
  occupancy_rate = "Occupancy rate (\\%)",
  pct_medicare   = "\\% Medicare",
  pct_medicaid   = "\\% Medicaid",
  
  cm_q_state_2   = "Acuity quartile 2 (state-month)",
  cm_q_state_3   = "Acuity quartile 3 (state-month)",
  cm_q_state_4   = "Acuity quartile 4 (state-month)"
)

# ---- Define variables (only keep ones that exist) ----
panelA_vars <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")
panelB_vars <- c(
  "government","non_profit","chain",
  "beds","occupancy_rate","pct_medicare","pct_medicaid",
  "cm_q_state_2","cm_q_state_3","cm_q_state_4"
)

panelA_vars <- intersect(panelA_vars, names(df))
panelB_vars <- intersect(panelB_vars, names(df))

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
      VarLabel = dplyr::coalesce(pretty_name[[variable]], variable),
      MeanStr  = if (variable %in% c("occupancy_rate","pct_medicare","pct_medicaid"))
        fmt_pct1(Mean) else fmt_dec(Mean, digits_for(variable)),
      SDStr    = if (variable %in% c("occupancy_rate","pct_medicare","pct_medicaid"))
        fmt_pct1(SD) else fmt_dec(SD, digits_for(variable))
    ) %>%
    ungroup()
  
  c(
    paste0("\\multicolumn{3}{@{}l}{\\textbf{", panel_title, "}} \\\\[2pt]"),
    tbl %>% transmute(line = paste0(VarLabel, " & ", MeanStr, " & ", SDStr, " \\\\")) %>% pull(line)
  )
}

panelA_lines <- make_panel_rows(panelA_vars, "Panel A: Outcome variables")
panelB_lines <- make_panel_rows(panelB_vars, "Panel B: Control variables")

# ---- Strings for notes ----
rows_str   <- fmt_int(overview$rows)
ccns_str   <- fmt_int(overview$ccns)
trt_str    <- fmt_int(overview$treated_ccns)
period_str <- paste0(overview$min_year_month, "–", overview$max_year_month)
avgm_str   <- fmt_dec(overview$avg_months_per_ccn, 1)

# ---- Table fragment ----
fragment <- c(
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Summary Statistics}",
  "\\label{tab:sumstats}",
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
  paste0("\\item \\textit{Notes:} Rows $=$ ", rows_str,
         "; Facilities $=$ ", ccns_str,
         "; Treated facilities $=$ ", trt_str,
         "; Period $=$ ", period_str,
         "; Average months per facility $=$ ", avgm_str, "."),
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
full_path <- file.path(out_dir, "summary_statistics.tex")
frag_path <- file.path(out_dir, "summary_statistics_code.tex")

writeLines(full_doc, full_path, useBytes = TRUE)
writeLines(fragment, frag_path, useBytes = TRUE)

cat("Wrote:\n - ", normalizePath(full_path, winslash = "\\"),
    "\n - ", normalizePath(frag_path, winslash = "\\"), "\n", sep = "")