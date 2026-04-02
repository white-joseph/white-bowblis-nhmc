# ================================================================
# MCR timing event-study preview + Wald table
#
# Produces:
#   1) Preview plots (not saved)
#      - Levels, with anticipation
#      - Levels, without anticipation
#      - Logs, with anticipation
#      - Logs, without anticipation
#
#   2) Joint Wald pretrend LaTeX table
#      - Panel A: Levels (HPPD)
#      - Panel B: Logs (HPPD)
#      - Rows:
#          * 2 Year Full Pre-Window
#          * 2 Year Window with Donut
#
# Writes:
#   - outputs/tables/pretrend_wald_tests_date_mcr_fragment.tex
#   - outputs/tables/pretrend_wald_tests_date_mcr_QA.tex
# ================================================================

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
  library(MASS)   # ginv()
})

options(scipen = 999, digits = 4)

# ------------------------------ Paths ------------------------------
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel_date_mcr.csv"
out_dir  <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------ Plot font ------------------------------
set_plot_font <- function() {
  fam <- "Times New Roman"
  par(family = fam)
}
set_plot_font()

# ------------------------------ Load ------------------------------
keep_cols <- c(
  "cms_certification_number","year_month","anticipation2",
  "event_time","treatment",
  "government","non_profit","chain","beds",
  "occupancy_rate","pct_medicare","pct_medicaid",
  "cm_q_state_2","cm_q_state_3","cm_q_state_4",
  "rn_hppd","lpn_hppd","cna_hppd","total_hppd"
)

df0 <- read_csv(panel_fp, show_col_types = FALSE, col_select = any_of(keep_cols)) %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month_chr = as.character(year_month),
    year_month = as.factor(year_month_chr),
    ym_date = as.Date(paste0(gsub("/", "-", year_month_chr), "-01"))
  ) %>%
  group_by(cms_certification_number) %>%
  mutate(
    ever_treated = as.integer(any(treatment == 1, na.rm = TRUE) | any(!is.na(event_time)))
  ) %>%
  ungroup()

# ------------------------------ Helpers ------------------------------
mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)

prep_df <- function(dat, WIN = 24L) {
  dat %>%
    mutate(
      event_time_capped = dplyr::case_when(
        ever_treated == 1L & !is.na(event_time) ~ pmin(pmax(as.integer(event_time), -as.integer(WIN)), as.integer(WIN)),
        TRUE ~ 9999L
      ),
      ln_rn    = mk_log(rn_hppd),
      ln_lpn   = mk_log(lpn_hppd),
      ln_cna   = mk_log(cna_hppd),
      ln_total = mk_log(total_hppd)
    )
}

candidate_controls <- c(
  "government","non_profit","chain","beds",
  "occupancy_rate","pct_medicare","pct_medicaid",
  "cm_q_state_2","cm_q_state_3","cm_q_state_4"
)

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

run_es_twfe <- function(lhs, data, ref_val, WIN = 24L) {
  controls <- intersect(candidate_controls, names(data))
  
  es_term <- paste0(
    "i(event_time_capped, ever_treated, ref = ", ref_val,
    ", keep = -", as.integer(WIN), ":", as.integer(WIN), ")"
  )
  
  rhs <- c(es_term, controls)
  rhs_txt <- paste(rhs, collapse = " + ")
  
  fml <- as.formula(paste0(
    lhs, " ~ ", rhs_txt,
    " | cms_certification_number + year_month"
  ))
  
  feols(
    fml,
    data = data,
    vcov = ~ cms_certification_number + year_month,
    lean = TRUE
  )
}

# ------------------------------ Pretrend helpers ------------------------------
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

pretrend_wald <- function(mod, ref_tau, from, to,
                          var = "event_time_capped", trt = "ever_treated") {
  if (is.null(mod)) {
    return(list(note = "Model is NULL"))
  }
  
  es <- .es_pick(mod, var, trt)
  if (!length(es$names)) {
    return(list(note = "No ES coefficients found"))
  }
  
  pre_idx <- es$taus < 0L & es$taus != ref_tau & es$taus >= from & es$taus <= to
  pre_names <- names(es$taus)[pre_idx]
  
  if (!length(pre_names)) {
    return(list(note = "No preperiod coefficients in window"))
  }
  
  b <- coef(mod)[pre_names]
  V <- vcov(mod)[pre_names, pre_names, drop = FALSE]
  
  W <- as.numeric(t(b) %*% MASS::ginv(V) %*% b)
  df_w <- qr(V)$rank
  pval <- pchisq(W, df = df_w, lower.tail = FALSE)
  
  list(
    statistic = W,
    df = df_w,
    p.value = pval,
    tested_taus = sort(unique(es$taus[pre_idx])),
    n_constraints = length(pre_names),
    window = c(from, to),
    note = NULL
  )
}

print_pretrend <- function(title, res) {
  cat("\n================ ", title, " ================\n", sep = "")
  if (!is.null(res$note)) {
    cat("[info] ", res$note, "\n", sep = "")
    return(invisible(NULL))
  }
  cat(sprintf("Joint Wald: W = %.3f on %d df  =>  p = %.4g\n",
              res$statistic, res$df, res$p.value))
  cat("Tested pre τ: ", paste(res$tested_taus, collapse = ", "), "\n", sep = "")
}

wald_df_from_list <- function(res_list, spec_label) {
  data.frame(
    specification = spec_label,
    outcome = names(res_list),
    wald_stat = sapply(res_list, function(x) ifelse(is.null(x$statistic), NA_real_, x$statistic)),
    df = sapply(res_list, function(x) ifelse(is.null(x$df), NA_integer_, x$df)),
    p_value = sapply(res_list, function(x) ifelse(is.null(x$p.value), NA_real_, x$p.value)),
    tested_taus = sapply(res_list, function(x) {
      if (is.null(x$tested_taus) || length(x$tested_taus) == 0) return(NA_character_)
      paste(x$tested_taus, collapse = ", ")
    }),
    n_constraints = sapply(res_list, function(x) ifelse(is.null(x$n_constraints), NA_integer_, x$n_constraints)),
    row.names = NULL
  )
}

fmt_wald_cell <- function(res) {
  if (is.null(res)) return("$\\,$")
  if (!is.null(res$note)) return("$\\,$")
  sprintf("$%.2f$ (%d) [%.4f]", res$statistic, res$df, res$p.value)
}

# ------------------------------ Outcomes ------------------------------
outs_lvl <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")
outs_log <- c("ln_rn","ln_lpn","ln_cna","ln_total")

nice_out <- c(
  rn_hppd    = "RN",
  lpn_hppd   = "LPN",
  cna_hppd   = "CNA",
  total_hppd = "Total"
)

# ------------------------------ Fitting block ------------------------------
fit_block <- function(tag, data, desired_ref = -1L, WIN = 24L) {
  cat("\n\n", strrep("=", 84), "\nBLOCK: ", tag, "\n", strrep("=", 84), "\n", sep = "")
  
  ref <- pick_ref(data, desired = desired_ref)
  cat("Reference used: t = ", ref, "\n", sep = "")
  
  mods_lvl <- lapply(outs_lvl, function(y) run_es_twfe(y, data, ref_val = ref, WIN = WIN))
  names(mods_lvl) <- outs_lvl
  
  mods_log <- lapply(outs_log, function(y) run_es_twfe(y, data, ref_val = ref, WIN = WIN))
  names(mods_log) <- outs_log
  
  invisible(list(
    levels = mods_lvl,
    logs   = mods_log,
    ref    = ref,
    tag    = tag,
    WIN    = WIN
  ))
}

# ------------------------------ Plot helpers ------------------------------
plot_block_levels <- function(mod_obj, title_prefix, event_window = c(-24L, 24L)) {
  ref <- mod_obj$ref
  
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  
  par(mfrow = c(2, 2))
  set_plot_font()
  
  iplot(
    mod_obj$levels[["rn_hppd"]],
    ref = ref, xlim = event_window,
    xlab = "Months relative to treatment", ylab = "RN HPPD",
    main = paste0(title_prefix, ": RN"),
    sub = ""
  )
  
  iplot(
    mod_obj$levels[["lpn_hppd"]],
    ref = ref, xlim = event_window,
    xlab = "Months relative to treatment", ylab = "LPN HPPD",
    main = paste0(title_prefix, ": LPN"),
    sub = ""
  )
  
  iplot(
    mod_obj$levels[["cna_hppd"]],
    ref = ref, xlim = event_window,
    xlab = "Months relative to treatment", ylab = "CNA HPPD",
    main = paste0(title_prefix, ": CNA"),
    sub = ""
  )
  
  iplot(
    mod_obj$levels[["total_hppd"]],
    ref = ref, xlim = event_window,
    xlab = "Months relative to treatment", ylab = "Total HPPD",
    main = paste0(title_prefix, ": Total"),
    sub = ""
  )
}

plot_block_logs <- function(mod_obj, title_prefix, event_window = c(-24L, 24L)) {
  ref <- mod_obj$ref
  
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  
  par(mfrow = c(2, 2))
  set_plot_font()
  
  iplot(
    mod_obj$logs[["ln_rn"]],
    ref = ref, xlim = event_window,
    xlab = "Months relative to treatment", ylab = "Log(RN HPPD)",
    main = paste0(title_prefix, ": Log RN"),
    sub = ""
  )
  
  iplot(
    mod_obj$logs[["ln_lpn"]],
    ref = ref, xlim = event_window,
    xlab = "Months relative to treatment", ylab = "Log(LPN HPPD)",
    main = paste0(title_prefix, ": Log LPN"),
    sub = ""
  )
  
  iplot(
    mod_obj$logs[["ln_cna"]],
    ref = ref, xlim = event_window,
    xlab = "Months relative to treatment", ylab = "Log(CNA HPPD)",
    main = paste0(title_prefix, ": Log CNA"),
    sub = ""
  )
  
  iplot(
    mod_obj$logs[["ln_total"]],
    ref = ref, xlim = event_window,
    xlab = "Months relative to treatment", ylab = "Log(Total HPPD)",
    main = paste0(title_prefix, ": Log Total"),
    sub = ""
  )
}

# ------------------------------ Samples ------------------------------
WIN <- 24L

S_full <- prep_df(df0, WIN = WIN)
S_noant <- S_full %>% filter(anticipation2 == 0)

# ------------------------------ Run models ------------------------------
mods_full <- fit_block(
  tag = "MCR timing — WITH anticipation",
  data = S_full,
  desired_ref = -1L,
  WIN = WIN
)

mods_noant <- fit_block(
  tag = "MCR timing — WITHOUT anticipation",
  data = S_noant,
  desired_ref = -4L,
  WIN = WIN
)

# ------------------------------ Wald tests ------------------------------
# With anticipation: 2-year full prewindow, ref = -1, test tau = -24,...,-2
wald_full_levels <- lapply(
  mods_full$levels,
  function(m) pretrend_wald(m, ref_tau = mods_full$ref, from = -24L, to = -2L)
)
wald_full_logs <- lapply(
  mods_full$logs,
  function(m) pretrend_wald(m, ref_tau = mods_full$ref, from = -24L, to = -2L)
)

# Without anticipation: 2-year donut prewindow, ref = -4, test tau = -24,...,-5
wald_noant_levels <- lapply(
  mods_noant$levels,
  function(m) pretrend_wald(m, ref_tau = mods_noant$ref, from = -24L, to = -5L)
)
wald_noant_logs <- lapply(
  mods_noant$logs,
  function(m) pretrend_wald(m, ref_tau = mods_noant$ref, from = -24L, to = -5L)
)

cat("\n\n", strrep("=", 84), "\nJOINT WALD PRETREND TESTS — LEVELS\n", strrep("=", 84), "\n", sep = "")
for (nm in names(wald_full_levels)) {
  print_pretrend(paste("WITH anticipation —", nm), wald_full_levels[[nm]])
}
for (nm in names(wald_noant_levels)) {
  print_pretrend(paste("WITHOUT anticipation —", nm), wald_noant_levels[[nm]])
}

cat("\n\n", strrep("=", 84), "\nJOINT WALD PRETREND TESTS — LOGS\n", strrep("=", 84), "\n", sep = "")
for (nm in names(wald_full_logs)) {
  print_pretrend(paste("WITH anticipation —", nm), wald_full_logs[[nm]])
}
for (nm in names(wald_noant_logs)) {
  print_pretrend(paste("WITHOUT anticipation —", nm), wald_noant_logs[[nm]])
}

wald_levels_table <- bind_rows(
  wald_df_from_list(wald_full_levels,  "2 Year Full Pre-Window"),
  wald_df_from_list(wald_noant_levels, "2 Year Window with Donut")
)

wald_logs_table <- bind_rows(
  wald_df_from_list(wald_full_logs,  "2 Year Full Pre-Window"),
  wald_df_from_list(wald_noant_logs, "2 Year Window with Donut")
)

cat("\n\n================ WALD SUMMARY TABLE: LEVELS ================\n")
print(wald_levels_table, row.names = FALSE)

cat("\n\n================ WALD SUMMARY TABLE: LOGS ================\n")
print(wald_logs_table, row.names = FALSE)

# ------------------------------ LaTeX Wald table ------------------------------
mk_row <- function(rowlabel, reslist, keys) {
  cells <- vapply(keys, function(k) {
    obj <- reslist[[k]]
    if (is.null(obj)) return("$\\,$")
    fmt_wald_cell(obj)
  }, character(1))
  
  paste0(rowlabel, " & ", paste(cells, collapse = " & "), " \\\\")
}

N_full  <- nrow(S_full)
N_noant <- nrow(S_noant)

wald_caption <- "Joint Wald Tests of Pre-trends Using MCR Event Timing"
wald_label   <- "tab:pretrend-wald-tests-date-mcr"

notes_windows <- paste0(
  "\\item Tested windows and reference periods: ",
  "2 Year Full Pre-Window tests $\\tau=-24$ to $\\tau=-2$ with reference $\\tau=", mods_full$ref, "$; ",
  "2 Year Window with Donut tests $\\tau=-24$ to $\\tau=-5$ with reference $\\tau=", mods_noant$ref, "$ (dropping $\\tau=-3,-2,-1$)."
)

notes_N <- paste0(
  "\\item Sample sizes (rows): ",
  "2 Year Full Pre-Window ($N=", format(N_full, big.mark = ","), "$); ",
  "2 Year Window with Donut ($N=", format(N_noant, big.mark = ","), "$)."
)

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
  sprintf(
    " & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} \\\\",
    nice_out[["rn_hppd"]], nice_out[["lpn_hppd"]], nice_out[["cna_hppd"]], nice_out[["total_hppd"]]
  ),
  "\\midrule",
  
  "\\multicolumn{5}{@{}l}{\\textbf{Panel A: Levels (HPPD)}} \\\\[2pt]",
  mk_row("2 Year Full Pre-Window",   wald_full_levels,  keys = outs_lvl),
  mk_row("2 Year Window with Donut", wald_noant_levels, keys = outs_lvl),
  
  "\\addlinespace[6pt]",
  "\\multicolumn{5}{@{}l}{\\textbf{Panel B: Logs (HPPD)}} \\\\[2pt]",
  mk_row("2 Year Full Pre-Window",   wald_full_logs,  keys = outs_log),
  mk_row("2 Year Window with Donut", wald_noant_logs, keys = outs_log),
  
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the Wald $\\chi^2$ statistic for the joint null that all pre-treatment event-time coefficients equal zero, followed by degrees of freedom in parentheses and the p-value in brackets.",
  notes_windows,
  notes_N,
  "\\item All specifications include facility and month fixed effects and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "\\endgroup",
  ""
)

wald_frag_path <- file.path(out_dir, "pretrend_wald_tests_date_mcr_fragment.tex")
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

wald_qa_path <- file.path(out_dir, "pretrend_wald_tests_date_mcr_QA.tex")
writeLines(wald_qa_doc, wald_qa_path, useBytes = TRUE)

cat("\n[write] ", normalizePath(wald_frag_path, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(wald_qa_path,   winslash = "\\"), "\n", sep = "")

# ------------------------------ Preview plots only ------------------------------
plot_block_levels(mods_full,  "MCR timing with anticipation",    event_window = c(-24L, 24L))
plot_block_levels(mods_noant, "MCR timing without anticipation", event_window = c(-24L, 24L))

plot_block_logs(mods_full,  "MCR timing with anticipation",    event_window = c(-24L, 24L))
plot_block_logs(mods_noant, "MCR timing without anticipation", event_window = c(-24L, 24L))

cat("\nPreview plots and Wald table generation completed for MCR timing panel.\n")