# C:/Repositories/white-bowblis-nhmc/regressions/stacked_twfe_post.R
# Basic TWFE DiD (post-only) on the STACKED dataset
# Fix: avoid "*** recursive gc invocation" by:
#   - trimming stacked dataset columns
#   - fitting ONE model at a time (no lapply over 24M rows)
#   - extracting post coef/SE/p immediately, then rm()+gc()
# Outputs:
#   - outputs/tables/stacked_twfe_post_full.tex
#   - outputs/tables/stacked_twfe_post_full_QA.tex

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
})

options(scipen = 999, digits = 4)

# ------------------------------ Paths ------------------------------
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_dir  <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------ Load ------------------------------
keep_cols <- c(
  "cms_certification_number","year_month",
  "time","time_treated",
  "government","non_profit","chain","beds",
  "occupancy_rate","pct_medicare","pct_medicaid",
  "cm_q_state_2","cm_q_state_3","cm_q_state_4",
  "rn_hppd","lpn_hppd","cna_hppd","total_hppd"
)

raw <- read_csv(panel_fp, show_col_types = FALSE)
keep_cols_present <- intersect(keep_cols, names(raw))

df <- raw[, keep_cols_present] %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.factor(year_month)
  )
rm(raw)

# ------------------------------ Logs ------------------------------
mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)
df <- df %>%
  mutate(
    ln_rn    = mk_log(rn_hppd),
    ln_lpn   = mk_log(lpn_hppd),
    ln_cna   = mk_log(cna_hppd),
    ln_total = mk_log(total_hppd)
  )

outs_lvl <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")
outs_log <- c(rn_hppd="ln_rn", lpn_hppd="ln_lpn", cna_hppd="ln_cna", total_hppd="ln_total")

# ------------------------------ Controls ------------------------------
controls_rhs <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)

# ------------------------------ Cohort g_i ------------------------------
g_df <- df %>%
  group_by(cms_certification_number) %>%
  summarise(
    g = {
      tt <- unique(time_treated[!is.na(time_treated)])
      if (length(tt) == 1) as.integer(tt) else NA_integer_
    },
    .groups = "drop"
  )

df <- df %>% left_join(g_df, by = "cms_certification_number")
cohorts <- sort(unique(df$g[!is.na(df$g)]))
cat("Unique cohorts (treated months):", length(cohorts), "\n")

# ------------------------------ Build stacked data (baseline donut) ------------------------------
make_stacked_data <- function(data, cohorts_vec, L = 24L, R = 24L, drop_set = -3:-1) {
  
  stacked <- lapply(cohorts_vec, function(g0) {
    
    d <- data %>%
      dplyr::filter(time >= g0 - L, time <= g0 + R) %>%
      dplyr::filter(is.na(g) | g > g0 | g == g0) %>%
      dplyr::mutate(
        cohort = as.integer(g0),
        rel = as.integer(time - g0),
        treated_stack = as.integer(!is.na(g) & g == g0),
        stack_id = interaction(cms_certification_number, cohort, drop = TRUE)
      ) %>%
      # baseline donut: drop treated obs at -3,-2,-1
      dplyr::filter(treated_stack == 0L | !(rel %in% drop_set)) %>%
      # post for TWFE DiD on stacked
      dplyr::mutate(post = as.integer(treated_stack == 1L & rel >= 0L)) %>%
      # trim columns HARD to keep RAM down
      dplyr::select(
        cms_certification_number, year_month,
        cohort, stack_id, post,
        government, non_profit, chain, beds,
        occupancy_rate, pct_medicare, pct_medicaid,
        cm_q_state_2, cm_q_state_3, cm_q_state_4,
        rn_hppd, lpn_hppd, cna_hppd, total_hppd,
        ln_rn, ln_lpn, ln_cna, ln_total
      )
    
    d
  })
  
  dplyr::bind_rows(stacked)
}

L <- 24L
R <- 24L
stack <- make_stacked_data(df, cohorts, L = L, R = R, drop_set = -3:-1)
rm(df); gc()
cat("Stacked rows (baseline donut):", nrow(stack), "\n")

# ------------------------------ Fit TWFE DiD on stacked ------------------------------
make_fml <- function(lhs) as.formula(paste0(
  lhs, " ~ post + ", controls_rhs, " | stack_id + year_month + cohort"
))
vc <- ~ cms_certification_number  # facility-only clustering (stable)

# Extract post info without keeping giant model objects around
extract_post <- function(mod, term = "post") {
  b  <- unname(coef(mod)[term])
  se <- unname(se(mod)[term])
  p  <- unname(pvalue(mod)[term])
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(b = b, se = se, stars = stars)
}

# Fit LEVELS one-by-one
res_lvl <- list()
for (y in outs_lvl) {
  cat("[fit levels]", y, "\n")
  m <- feols(make_fml(y), data = stack, vcov = vc, lean = TRUE)
  res_lvl[[y]] <- extract_post(m)
  rm(m); gc()
}

# Fit LOGS one-by-one
res_log <- list()
for (y in outs_lvl) {
  ly <- outs_log[[y]]
  cat("[fit logs]", ly, "\n")
  m <- feols(make_fml(ly), data = stack, vcov = vc, lean = TRUE)
  res_log[[y]] <- extract_post(m)
  rm(m); gc()
}

# ------------------------------ LaTeX helpers ------------------------------
fmt_est <- function(b, se, stars) {
  bstr <- sprintf("%.3f", b)
  if (b > 0) bstr <- paste0("\\phantom{-}", bstr)
  sestr <- sprintf("%.3f", se)
  sprintf("\\est{$%s$}{$%s$}{%s}", bstr, sestr, stars)
}

row_from_res <- function(reslist) {
  paste(vapply(outs_lvl, function(y) {
    fmt_est(reslist[[y]]$b, reslist[[y]]$se, reslist[[y]]$stars)
  }, character(1)), collapse = "  &  ")
}

row_HPPD <- row_from_res(res_lvl)
row_LOG  <- row_from_res(res_log)

N_levels <- format(nrow(stack), big.mark = ",")
# Logs N: rows where all logs are non-missing (simple, conservative)
N_logs <- format(sum(complete.cases(stack[, c("ln_rn","ln_lpn","ln_cna","ln_total")])), big.mark = ",")

# ------------------------------ Write LaTeX table ------------------------------
tab <- c(
  "\\begingroup",
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{TWFE DiD Estimates of \\textit{post} on Staffing Outcomes (Stacked Sample, Baseline Donut)}",
  "\\label{tab:stacked-twfe-post}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "",
  "\\begin{tabularx}{\\textwidth}{@{} l YYYY @{} }",
  "\\toprule",
  " & \\multicolumn{4}{c}{\\textbf{Outcomes}} \\\\",
  "\\cmidrule(lr){2-5}",
  " & \\textbf{RN} & \\textbf{LPN} & \\textbf{CNA} & \\textbf{Total} \\\\",
  "\\midrule",
  paste0("HPPD & ", row_HPPD, " \\\\"),
  "\\addlinespace[3pt]",
  paste0("Log(HPPD) & ", row_LOG, " \\\\"),
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  sprintf("\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post} with facility-clustered standard errors in parentheses. The stacked sample includes never-treated and not-yet-treated controls for each cohort; the donut excludes $\\tau\\in\\{-3,-2,-1\\}$. Sample sizes: $N_{\\mathrm{HPPD}}=%s$; $N_{\\mathrm{Log}}=%s$.",
          N_levels, N_logs),
  "\\item Specifications include facility-by-cohort fixed effects (stack\\_id), calendar-month fixed effects, cohort fixed effects, and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators.",
  "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "\\endgroup",
  ""
)

tab_path <- file.path(out_dir, "stacked_twfe_post_full.tex")
writeLines(tab, tab_path, useBytes = TRUE)

qa_doc <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{array}",
  "\\usepackage{makecell}",
  "\\usepackage{newtxtext}",
  "\\usepackage{newtxmath}",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "\\newcommand{\\sym}[1]{\\rlap{$^{#1}$}}",
  "\\newcommand{\\est}[3]{\\makecell[c]{#1\\sym{#3}\\\\ \\footnotesize(#2)}}",
  "\\begin{document}",
  tab,
  "\\end{document}"
)
qa_path <- file.path(out_dir, "stacked_twfe_post_full_QA.tex")
writeLines(qa_doc, qa_path, useBytes = TRUE)

cat("[write] ", normalizePath(tab_path, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(qa_path,  winslash = "\\"), "\n", sep = "")
cat("Done.\n")