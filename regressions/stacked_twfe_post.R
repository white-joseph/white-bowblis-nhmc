# C:/Repositories/white-bowblis-nhmc/regressions/stacked_twfe_post.R
# Basic TWFE DiD (post-only) on the STACKED dataset
# - Baseline stacked sample with donut (-3,-2,-1 dropped for treated cohort)
# - Outcomes: RN/LPN/CNA/Total in levels and logs
# - FE: stack_id + year_month + cohort
# - SE: cluster at facility (cms_certification_number)
# - Outputs LaTeX table: stacked_twfe_post_full.tex (+ QA)

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
})

options(scipen = 999, digits = 4)

panel_fp  <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_dir   <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
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

# logs
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

# controls
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

df <- df %>%
  left_join(g_df, by = "cms_certification_number")

cohorts <- sort(unique(df$g[!is.na(df$g)]))
cat("Unique cohorts (treated months):", length(cohorts), "\n")

# ------------------------------ Build stacked data (baseline donut) ------------------------------
make_stacked_data <- function(data, cohorts_vec, L = 24L, R = 24L, drop_set = -3:-1) {
  
  stacked <- lapply(cohorts_vec, function(g0) {
    
    d <- data %>%
      filter(time >= g0 - L, time <= g0 + R) %>%
      filter(is.na(g) | g > g0 | g == g0) %>%
      mutate(
        cohort = as.integer(g0),
        rel = as.integer(time - g0),
        treated_stack = as.integer(!is.na(g) & g == g0),
        stack_id = interaction(cms_certification_number, cohort, drop = TRUE)
      ) %>%
      # baseline donut: drop treated obs at -3,-2,-1
      filter(treated_stack == 0L | !(rel %in% drop_set)) %>%
      # define post for TWFE DiD on stacked sample
      mutate(post = as.integer(treated_stack == 1L & rel >= 0L)) %>%
      select(
        cms_certification_number, year_month,
        cohort, rel, treated_stack, stack_id, post,
        government, non_profit, chain, beds,
        occupancy_rate, pct_medicare, pct_medicaid,
        cm_q_state_2, cm_q_state_3, cm_q_state_4,
        rn_hppd, lpn_hppd, cna_hppd, total_hppd,
        ln_rn, ln_lpn, ln_cna, ln_total
      )
    
    d
  })
  
  bind_rows(stacked)
}

stack <- make_stacked_data(df, cohorts, L = 24L, R = 24L, drop_set = -3:-1)
cat("Stacked rows (baseline donut):", nrow(stack), "\n")

# ------------------------------ Fit TWFE DiD on stacked ------------------------------
make_fml <- function(lhs) as.formula(paste0(
  lhs, " ~ post + ", controls_rhs, " | stack_id + year_month + cohort"
))

vc <- ~ cms_certification_number  # facility-only clustering (stable)

mods_lvl <- lapply(outs_lvl, function(y) feols(make_fml(y), data = stack, vcov = vc, lean = TRUE))
names(mods_lvl) <- outs_lvl

mods_log <- lapply(outs_lvl, function(y) {
  ly <- outs_log[[y]]
  feols(make_fml(ly), data = stack, vcov = vc, lean = TRUE)
})
names(mods_log) <- outs_lvl

# ------------------------------ Helpers to build LaTeX table ------------------------------
coef_se_star <- function(mod, term = "post") {
  sm <- summary(mod)
  b  <- unname(coef(mod)[term])
  se <- unname(sm$coeftable[term, "Std. Error"])
  p  <- unname(sm$coeftable[term, "Pr(>|t|)"])
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(coef = b, se = se, stars = stars)
}

fmt_est <- function(b, se, stars) {
  bstr  <- sprintf("%.3f", b); if (b > 0) bstr <- paste0("\\phantom{-}", bstr)
  sestr <- sprintf("%.3f", se)
  sprintf("\\est{$%s$}{$%s$}{%s}", bstr, sestr, stars)
}

build_row <- function(modset) {
  paste(vapply(outs_lvl, function(y) {
    s <- coef_se_star(modset[[y]])
    fmt_est(s$coef, s$se, s$stars)
  }, character(1)), collapse = "  &  ")
}

row_HPPD <- build_row(mods_lvl)
row_LOG  <- build_row(mods_log)

N_levels <- format(nrow(stack), big.mark = ",")
# For logs: count rows where all needed logs are non-missing? Keep simple:
N_logs <- format(sum(complete.cases(stack[, c("ln_rn","ln_lpn","ln_cna","ln_total")])), big.mark = ",")

# ------------------------------ Write LaTeX table (HPPD + Log(HPPD)) ------------------------------
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