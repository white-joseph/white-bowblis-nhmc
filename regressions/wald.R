# Console-only Wald pretrend tests with event-time window [-12, 12]
# + LaTeX table output (same format/style as your -24 version)
#
# Panel A: With anticipation (ref typically -1) -> tests taus -12..-2
# Panel B: Without anticipation II (drop -3,-2,-1; ref typically -4) -> tests taus -12..-5
#
# Outputs:
#   - outputs/tables/pretrend_wald_tests_fragment.tex
#   - outputs/tables/pretrend_wald_tests_QA.tex

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
  library(MASS)  # ginv
})

options(scipen = 999, digits = 4)

# ------------------------------ Paths ------------------------------
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_dir  <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------ Load + prep ------------------------------
keep_cols <- c(
  "cms_certification_number","year_month","event_time","treatment",
  "government","non_profit","chain","beds",
  "occupancy_rate","pct_medicare","pct_medicaid",
  "cm_q_state_2","cm_q_state_3","cm_q_state_4",
  "rn_hppd","lpn_hppd","cna_hppd","total_hppd"
)

df <- read_csv(panel_fp, show_col_types = FALSE, col_select = all_of(keep_cols)) %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.factor(year_month)
  )

# Ever-treated & cap event_time to [-12, 12]
WIN <- 12L

df <- df %>%
  group_by(cms_certification_number) %>%
  mutate(ever_treated = as.integer(any(treatment == 1, na.rm = TRUE) | any(!is.na(event_time)))) %>%
  ungroup() %>%
  mutate(
    event_time_capped = dplyr::case_when(
      ever_treated == 1L & !is.na(event_time) ~ pmin(pmax(as.integer(event_time), -WIN), WIN),
      TRUE ~ 9999L
    )
  )

# Logs (for Panel B of the LaTeX table, matching your prior format)
mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)
df <- df %>%
  mutate(
    ln_rn    = mk_log(rn_hppd),
    ln_lpn   = mk_log(lpn_hppd),
    ln_cna   = mk_log(cna_hppd),
    ln_total = mk_log(total_hppd)
  )

# Controls RHS
controls_rhs <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)

# Helper: pick a valid reference that exists in the treated event-time support
pick_ref <- function(dat, desired = NULL) {
  ev <- sort(unique(dat$event_time_capped[dat$ever_treated == 1L]))
  ev <- ev[is.finite(ev) & ev != 9999L]
  if (!length(ev)) stop("No treated event times found.")
  if (!is.null(desired) && desired %in% ev) return(as.integer(desired))
  if (-1L %in% ev) return(-1L)
  if (-4L %in% ev) return(-4L)
  negs <- ev[ev < 0L]
  if (length(negs)) return(max(negs))
  return(ev[1])
}

# Event study TWFE
run_es_twfe <- function(lhs, data, ref_val, win = WIN) {
  fml <- as.formula(paste0(
    lhs, " ~ i(event_time_capped, ever_treated, ref = ", ref_val, ", keep = -", win, ":", win, ") + ",
    controls_rhs,
    " | cms_certification_number + year_month"
  ))
  feols(
    fml,
    data = data,
    vcov = ~ cms_certification_number + year_month,
    lean = TRUE
  )
}

# Pick ES coefficient names and taus from a fixest model
.es_pick <- function(mod, var = "event_time_capped", trt = "ever_treated") {
  cn <- names(coef(mod))
  if (is.null(cn) || !length(cn)) return(list(names = character(0), taus = integer(0)))
  pat <- sprintf("^%s::[-]?[0-9]+:%s$", var, trt)
  es_names <- grep(pat, cn, value = TRUE)
  get_tau <- function(s) as.integer(regmatches(s, regexpr("-?[0-9]+", s)))
  taus <- vapply(es_names, get_tau, integer(1))
  names(taus) <- es_names
  list(names = es_names, taus = taus)
}

# Joint Wald test: H0 all pre-treatment betas in [from,to] are zero (excluding ref)
pretrend_wald <- function(mod, ref_tau, from, to,
                          var = "event_time_capped", trt = "ever_treated") {
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
  
  list(statistic = W, df = df_w, p.value = pval, window = c(from, to))
}

# Pretty printer (console)
print_panel <- function(panel_name, data, ref, test_from, test_to) {
  outcomes <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")
  nice <- c(rn_hppd="RN", lpn_hppd="LPN", cna_hppd="CNA", total_hppd="Total")
  
  cat("\n============================================================\n")
  cat(panel_name, "\n")
  cat("Event-time window kept: [", -WIN, ", ", WIN, "]\n", sep = "")
  cat("Reference period: tau = ", ref, "\n", sep = "")
  cat("Joint pretrend window tested: tau = ", test_from, " .. ", test_to, "\n", sep = "")
  cat("N rows: ", format(nrow(data), big.mark = ","), "\n", sep = "")
  cat("============================================================\n")
  
  for (y in outcomes) {
    mod <- tryCatch(run_es_twfe(y, data, ref), error = function(e) NULL)
    res <- pretrend_wald(mod, ref_tau = ref, from = test_from, to = test_to)
    
    if (!is.null(res$note)) {
      cat(sprintf("%-6s  %s\n", nice[[y]], paste0("NA (", res$note, ")")))
    } else {
      cat(sprintf(
        "%-6s  Wald Chi^2 = %8.2f  df = %2d  p = %.4f\n",
        nice[[y]], res$statistic, res$df, res$p.value
      ))
    }
  }
}

# ------------------------------ Panels ------------------------------
# Panel A: With anticipation
dat_with <- df
ref_with <- pick_ref(dat_with, desired = -1L)
# With anticipation: test -12..-2 (exclude ref -1)
win_with <- c(-WIN, -2L)

# Panel B: Without anticipation II
skip2 <- c(-3L,-2L,-1L)
dat_wo2 <- df %>% filter(!(ever_treated == 1L & event_time_capped %in% skip2))
ref_wo2 <- pick_ref(dat_wo2, desired = -4L)
# Without anticipation: ref -4; test -12..-5 (since -3,-2,-1 dropped)
win_wo2 <- c(-WIN, -5L)

# ------------------------------ Run + print (console) ------------------------------
print_panel("Panel A: With anticipation", dat_with, ref_with, win_with[1], win_with[2])
print_panel("Panel B: Without anticipation (drop tau = -3,-2,-1)", dat_wo2, ref_wo2, win_wo2[1], win_wo2[2])

# ======================================================================
# LaTeX table output (same look/structure as your -24 Wald table)
#   Panel A: Levels
#   Panel B: Logs
# Rows: With anticipation / Without anticipation
# Cells: Chi^2 (df) [p-value]
# ======================================================================

outs_lvl <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")
nice_out <- c(rn_hppd="RN", lpn_hppd="LPN", cna_hppd="CNA", total_hppd="Total")
outs_log <- c(rn_hppd="ln_rn", lpn_hppd="ln_lpn", cna_hppd="ln_cna", total_hppd="ln_total")

# Fit ES models (levels)
mods_lvl_with <- lapply(outs_lvl, \(y) tryCatch(run_es_twfe(y, dat_with, ref_with), error = function(e) NULL))
names(mods_lvl_with) <- outs_lvl
mods_lvl_wo2  <- lapply(outs_lvl, \(y) tryCatch(run_es_twfe(y, dat_wo2,  ref_wo2),  error = function(e) NULL))
names(mods_lvl_wo2)  <- outs_lvl

# Fit ES models (logs)
mods_log_with <- list()
mods_log_wo2  <- list()
for (y in outs_lvl) {
  lhs <- outs_log[[y]]
  mods_log_with[[y]] <- if (!all(is.na(dat_with[[lhs]]))) tryCatch(run_es_twfe(lhs, dat_with, ref_with), error = function(e) NULL) else NULL
  mods_log_wo2[[y]]  <- if (!all(is.na(dat_wo2[[lhs]])))  tryCatch(run_es_twfe(lhs, dat_wo2,  ref_wo2),  error = function(e) NULL) else NULL
}

# Wald tests by outcome
wald_lvl_with <- lapply(outs_lvl, \(y) pretrend_wald(mods_lvl_with[[y]], ref_tau = ref_with, from = win_with[1], to = win_with[2]))
names(wald_lvl_with) <- outs_lvl
wald_lvl_wo2  <- lapply(outs_lvl, \(y) pretrend_wald(mods_lvl_wo2[[y]],  ref_tau = ref_wo2,  from = win_wo2[1],  to = win_wo2[2]))
names(wald_lvl_wo2)  <- outs_lvl

wald_log_with <- lapply(outs_lvl, \(y) pretrend_wald(mods_log_with[[y]], ref_tau = ref_with, from = win_with[1], to = win_with[2]))
names(wald_log_with) <- outs_lvl
wald_log_wo2  <- lapply(outs_lvl, \(y) pretrend_wald(mods_log_wo2[[y]],  ref_tau = ref_wo2,  from = win_wo2[1],  to = win_wo2[2]))
names(wald_log_wo2)  <- outs_lvl

fmt_wald_cell <- function(res) {
  if (!is.null(res$note)) return("$\\,$")
  sprintf("$%.2f$ (%d) [%.4f]", res$statistic, res$df, res$p.value)
}

mk_row <- function(rowlabel, reslist) {
  cells <- vapply(outs_lvl, \(y) fmt_wald_cell(reslist[[y]]), character(1))
  paste0(rowlabel, " & ", paste(cells, collapse = " & "), " \\\\")
}

N_with <- nrow(dat_with)
N_wo2  <- nrow(dat_wo2)

wald_caption <- "Joint Wald Tests of Pre-trends (Event Study)"
wald_label   <- "tab:pretrend-wald-tests"

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
  
  "\\multicolumn{5}{@{}l}{\\textbf{Panel A: Levels (HPPD)}} \\\\[2pt]",
  mk_row("With anticipation",    wald_lvl_with),
  mk_row("Without anticipation", wald_lvl_wo2),
  
  "\\addlinespace[6pt]",
  "\\multicolumn{5}{@{}l}{\\textbf{Panel B: Logs (HPPD)}} \\\\[2pt]",
  mk_row("With anticipation",    wald_log_with),
  mk_row("Without anticipation", wald_log_wo2),
  
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the Wald $\\chi^2$ statistic for the joint null that all pre-treatment event-time coefficients equal zero, followed by degrees of freedom in parentheses and the p-value in brackets.",
  sprintf("\\item Tested windows: With anticipation tests $\\tau=%d$ to $\\tau=%d$; Without anticipation tests $\\tau=%d$ to $\\tau=%d$ (since $\\tau=-3,-2,-1$ are dropped and $\\tau=-4$ is the reference).",
          win_with[1], win_with[2], win_wo2[1], win_wo2[2]),
  sprintf("\\item Reference periods: With anticipation uses $\\tau=%d$; Without anticipation uses $\\tau=%d$.", ref_with, ref_wo2),
  sprintf("\\item Sample sizes (rows): With anticipation ($N=%s$); Without anticipation ($N=%s$).",
          format(N_with, big.mark=","), format(N_wo2, big.mark=",")),
  "\\item All specifications include facility and month fixed effects and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "\\endgroup",
  ""
)

wald_frag_path <- file.path(out_dir, "pretrend_wald_tests_fragment.tex")
writeLines(wald_tab, wald_frag_path, useBytes = TRUE)

wald_qa_doc <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{array}",
  "\\usepackage{caption}",
  "\\usepackage{makecell}",
  "\\usepackage{newtxtext}",
  "\\usepackage{newtxmath}",
  "\\captionsetup{labelfont=bf, font=small}",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "\\begin{document}",
  wald_tab,
  "\\end{document}"
)

wald_qa_path <- file.path(out_dir, "pretrend_wald_tests_QA.tex")
writeLines(wald_qa_doc, wald_qa_path, useBytes = TRUE)

cat("\n[write] ", normalizePath(wald_frag_path, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(wald_qa_path,   winslash = "\\"), "\n", sep = "")
cat("\nDone.\n")