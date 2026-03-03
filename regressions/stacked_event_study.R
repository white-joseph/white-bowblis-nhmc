# C:/Repositories/white-bowblis-nhmc/regressions/stacked_event_study.R
# Baseline Stacked DiD Event Study + Joint Wald pretrend table (LEVELS ONLY)
# - Stacked design: controls are never-treated + not-yet-treated (relative to cohort)
# - Baseline donut: drop rel in {-3,-2,-1} for treated cohort units; reference rel = -4
# - Saves iplots (RN/LPN/CNA/Total) WITH titles (in R)
# - Removes donut dots from plots via iplot(drop=...)
# - Writes Wald table fragment + QA tex to outputs/tables
# - Uses facility-only clustering for stacked (lighter + stable)

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(MASS)  # ginv
})

options(scipen = 999, digits = 4)

# ------------------------------ Plot font (Times / newtx-like) ------------------------------
set_plot_font <- function() {
  par(family = "Times New Roman")
}
set_plot_font()

# ------------------------------ Paths ------------------------------
panel_fp  <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_plots <- "C:/Repositories/white-bowblis-nhmc/outputs/plots"
out_tabs  <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
dir.create(out_plots, showWarnings = FALSE, recursive = TRUE)
dir.create(out_tabs,  showWarnings = FALSE, recursive = TRUE)

# ------------------------------ Load ------------------------------
keep_cols <- c(
  "cms_certification_number","year_month","ym_date",
  "anticipation2","event_time","treatment","time","time_treated",
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

# ------------------------------ Controls + logs (logs needed only if you want them later) ------------------------------
controls_rhs <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)

mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)
df <- df %>%
  mutate(
    ln_rn    = mk_log(rn_hppd),
    ln_lpn   = mk_log(lpn_hppd),
    ln_cna   = mk_log(cna_hppd),
    ln_total = mk_log(total_hppd)
  )

outs_lvl <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")
nice_out <- c(rn_hppd="RN", lpn_hppd="LPN", cna_hppd="CNA", total_hppd="Total")

# ------------------------------ Cohort g_i (time_treated) ------------------------------
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
  left_join(g_df, by = "cms_certification_number") %>%
  mutate(ever_treated = as.integer(!is.na(g)))

cohorts <- sort(unique(df$g[!is.na(df$g)]))
cat("Unique cohorts (treated months):", length(cohorts), "\n")

# ------------------------------ Build stacked data ------------------------------
make_stacked_data <- function(data, cohorts_vec, L = 24L, R = 24L,
                              donut = TRUE, drop_set = -3:-1) {
  
  stacked <- lapply(cohorts_vec, function(g0) {
    
    d <- data %>%
      filter(time >= g0 - L, time <= g0 + R) %>%
      # treated cohort g0, never-treated, and later-treated (not-yet-treated relative to g0)
      filter(is.na(g) | g > g0 | g == g0) %>%
      mutate(
        cohort = as.integer(g0),
        rel = as.integer(time - g0),
        treated_stack = as.integer(!is.na(g) & g == g0),
        stack_id = interaction(cms_certification_number, cohort, drop = TRUE)
      )
    
    if (donut) {
      d <- d %>% filter(treated_stack == 0L | !(rel %in% drop_set))
    }
    
    d
  })
  
  bind_rows(stacked)
}

# ------------------------------ Stacked ES regression ------------------------------
run_stacked_es <- function(lhs, data_stacked, ref = -4L, window = c(-24L, 24L),
                           vcov_formula = ~ cms_certification_number) {
  
  fml <- as.formula(paste0(
    lhs, " ~ i(rel, treated_stack, ref = ", ref,
    ", keep = ", window[1], ":", window[2], ") + ",
    controls_rhs,
    " | stack_id + year_month + cohort"
  ))
  
  feols(
    fml,
    data = data_stacked,
    vcov = vcov_formula,   # facility-only (lighter + stable)
    lean = TRUE
  )
}

# ------------------------------ Plot helper (titles + drop donut points) ------------------------------
save_iplot <- function(model, fname, ref, window, ylab_txt,
                       title_txt = "", drop_donut_points = TRUE) {
  
  grDevices::cairo_pdf(
    filename = file.path(out_plots, fname),
    width  = 9.5,
    height = 6.2
  )
  on.exit(dev.off(), add = TRUE)
  set_plot_font()
  
  args <- list(
    object = model,
    ref  = ref,
    xlim = window,
    xlab = "Months relative to treatment",
    ylab = ylab_txt,
    main = title_txt,
    sub  = ""
  )
  
  if (drop_donut_points) {
    # Removes plotted points for τ ∈ {-3,-2,-1} in stacked ES
    args$drop <- "^rel::-(3|2|1):treated_stack$"
  }
  
  do.call(iplot, args)
}

# ============================================================
# PART 1: Baseline stacked ES (donut -3..-1), save plots
# ============================================================
L <- 24L
R <- 24L
event_window <- c(-24L, 24L)
ref_tau <- -4L

stack_base <- make_stacked_data(df, cohorts, L = L, R = R, donut = TRUE, drop_set = -3:-1)
cat("Stacked baseline rows:", nrow(stack_base), "\n")

mods_lvl <- lapply(outs_lvl, \(y) run_stacked_es(y, stack_base, ref = ref_tau, window = event_window))
names(mods_lvl) <- outs_lvl

save_iplot(mods_lvl[["rn_hppd"]],
           "stacked_es_rn_baseline.pdf",
           ref_tau, event_window, "RN HPPD",
           title_txt = "Stacked DiD Event Study: RN HPPD (Donut, ref = -4)",
           drop_donut_points = TRUE)

save_iplot(mods_lvl[["lpn_hppd"]],
           "stacked_es_lpn_baseline.pdf",
           ref_tau, event_window, "LPN HPPD",
           title_txt = "Stacked DiD Event Study: LPN HPPD (Donut, ref = -4)",
           drop_donut_points = TRUE)

save_iplot(mods_lvl[["cna_hppd"]],
           "stacked_es_cna_baseline.pdf",
           ref_tau, event_window, "CNA HPPD",
           title_txt = "Stacked DiD Event Study: CNA HPPD (Donut, ref = -4)",
           drop_donut_points = TRUE)

save_iplot(mods_lvl[["total_hppd"]],
           "stacked_es_total_baseline.pdf",
           ref_tau, event_window, "Total HPPD",
           title_txt = "Stacked DiD Event Study: Total Nursing HPPD (Donut, ref = -4)",
           drop_donut_points = TRUE)

cat("Saved stacked baseline plots to: ", out_plots, "\n", sep = "")

# ============================================================
# PART 2: Joint Wald pretrend table for STACKED ES (LEVELS ONLY)
# Two rows:
#   1) 2 Year Window with Donut  (uses existing baseline models in mods_lvl)
#   2) 1 Year Window with Donut  (fits LEVELS only on WIN=12)
# ============================================================

# ---- coefficient picker for stacked ES (rel::k:treated_stack) ----
.es_pick <- function(mod, var = "rel", trt = "treated_stack") {
  cn <- names(coef(mod))
  if (is.null(cn) || !length(cn)) return(list(names = character(0), taus = integer(0)))
  pat <- sprintf("^%s::[-]?[0-9]+:%s$", var, trt)
  es_names <- grep(pat, cn, value = TRUE)
  get_tau <- function(s) as.integer(regmatches(s, regexpr("-?[0-9]+", s)))
  taus <- vapply(es_names, get_tau, integer(1))
  names(taus) <- es_names
  list(names = es_names, taus = taus)
}

pretrend_wald <- function(mod, ref_tau, from, to, var = "rel", trt = "treated_stack") {
  if (is.null(mod)) return(list(note = "Model is NULL"))
  es <- .es_pick(mod, var, trt)
  if (!length(es$names)) return(list(note = "No ES coefficients found"))
  
  pre_idx <- es$taus < 0L & es$taus != ref_tau & es$taus >= from & es$taus <= to
  pre_names <- names(es$taus)[pre_idx]
  if (!length(pre_names)) return(list(note = "No preperiod coefficients in window"))
  
  b <- coef(mod)[pre_names]
  V <- vcov(mod)[pre_names, pre_names, drop = FALSE]
  
  W <- as.numeric(t(b) %*% MASS::ginv(V) %*% b)
  df_w <- qr(V)$rank
  pval <- pchisq(W, df = df_w, lower.tail = FALSE)
  
  list(statistic = W, df = df_w, p.value = pval)
}

fmt_wald_cell <- function(res) {
  if (!is.null(res$note)) return("$\\,$")
  sprintf("$%.2f$ (%d) [%.4f]", res$statistic, res$df, res$p.value)
}

mk_row <- function(rowlabel, reslist) {
  cells <- vapply(outs_lvl, function(y) fmt_wald_cell(reslist[[y]]), character(1))
  paste0(rowlabel, " & ", paste(cells, collapse = " & "), " \\\\")
}

# ------------------------------
# Row 1: 2-year donut (reuse existing baseline models)
# donut drops -3..-1, ref=-4, test -24..-5
# ------------------------------
wald_24 <- lapply(outs_lvl, function(y) {
  pretrend_wald(mods_lvl[[y]], ref_tau = -4L, from = -24L, to = -5L)
})
names(wald_24) <- outs_lvl
N_24 <- nrow(stack_base)

# ------------------------------
# Row 2: 1-year donut (build smaller stack; LEVELS only; fit 4 models)
# donut drops -3..-1, ref=-4, test -12..-5
# ------------------------------
stack_12 <- make_stacked_data(df, cohorts, L = 12L, R = 12L, donut = TRUE, drop_set = -3:-1)
N_12 <- nrow(stack_12)

mods_12 <- lapply(outs_lvl, \(y) run_stacked_es(y, stack_12, ref = -4L, window = c(-12L, 12L)))
names(mods_12) <- outs_lvl

wald_12 <- lapply(outs_lvl, function(y) {
  pretrend_wald(mods_12[[y]], ref_tau = -4L, from = -12L, to = -5L)
})
names(wald_12) <- outs_lvl

rm(stack_12, mods_12); gc()

# ------------------------------ LaTeX table (Levels only) ------------------------------
notes_windows <- paste0(
  "\\item Tested windows and reference periods: ",
  "2 Year Donut tests $\\tau=-24$ to $\\tau=-5$ with reference $\\tau=-4$ (dropping $\\tau=-3,-2,-1$); ",
  "1 Year Donut tests $\\tau=-12$ to $\\tau=-5$ with reference $\\tau=-4$ (dropping $\\tau=-3,-2,-1$)."
)

notes_N <- paste0(
  "\\item Sample sizes (stacked rows): 2 Year Donut ($N=", format(N_24, big.mark=","), "$); ",
  "1 Year Donut ($N=", format(N_12, big.mark=","), "$)."
)

wald_caption <- "Joint Wald Tests of Pre-trends (Stacked Event Study) --- Levels"
wald_label   <- "tab:pretrend-wald-tests-stacked-levels"

wald_tab <- c(
  "\\begingroup",
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  sprintf("\\caption{%s}", wald_caption),
  sprintf("\\label{%s}", wald_label),
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "",
  "\\begin{tabularx}{\\textwidth}{@{} l YYYY @{} }",
  "\\toprule",
  " & \\multicolumn{4}{c}{\\textbf{Outcomes}} \\\\",
  "\\cmidrule(lr){2-5}",
  sprintf(" & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} \\\\",
          nice_out[["rn_hppd"]], nice_out[["lpn_hppd"]], nice_out[["cna_hppd"]], nice_out[["total_hppd"]]),
  "\\midrule",
  mk_row("2 Year Window with Donut", wald_24),
  mk_row("1 Year Window with Donut", wald_12),
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the Wald $\\chi^2$ statistic for the joint null that all pre-treatment event-time coefficients equal zero, followed by degrees of freedom in parentheses and the p-value in brackets.",
  notes_windows,
  notes_N,
  "\\item All specifications include stacked-unit fixed effects (facility-by-cohort), calendar-month fixed effects, cohort fixed effects, and baseline covariates. Standard errors are clustered at the facility level.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "\\endgroup",
  ""
)

frag_path <- file.path(out_tabs, "pretrend_wald_tests_stacked_levels_fragment.tex")
writeLines(wald_tab, frag_path, useBytes = TRUE)

qa_doc <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{array}",
  "\\usepackage{newtxtext}",
  "\\usepackage{newtxmath}",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "\\begin{document}",
  wald_tab,
  "\\end{document}"
)
qa_path <- file.path(out_tabs, "pretrend_wald_tests_stacked_levels_QA.tex")
writeLines(qa_doc, qa_path, useBytes = TRUE)

cat("\n[write] ", normalizePath(frag_path, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(qa_path,   winslash = "\\"), "\n", sep = "")
cat("Done.\n")