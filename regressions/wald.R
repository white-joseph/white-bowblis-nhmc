# ================================================================
# Joint Wald pretrend tests in ONE table with THREE rows per panel:
#
# Levels (HPPD):
#   1) With anticipation
#   2) Without anticipation (-24)   [event-time window kept: -24..24]
#   3) Without anticipation (-12)   [event-time window kept: -12..12]
#
# Logs (HPPD): same three rows
#
# Outputs (same filenames as your existing Wald script):
#   - outputs/tables/pretrend_wald_tests_fragment.tex
#   - outputs/tables/pretrend_wald_tests_QA.tex
# ================================================================

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

# ------------------------------ Load ------------------------------
keep_cols <- c(
  "cms_certification_number","year_month","event_time","treatment",
  "government","non_profit","chain","beds",
  "occupancy_rate","pct_medicare","pct_medicaid",
  "cm_q_state_2","cm_q_state_3","cm_q_state_4",
  "rn_hppd","lpn_hppd","cna_hppd","total_hppd"
)

df0 <- read_csv(panel_fp, show_col_types = FALSE, col_select = all_of(keep_cols)) %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.factor(year_month)
  ) %>%
  group_by(cms_certification_number) %>%
  mutate(ever_treated = as.integer(any(treatment == 1, na.rm = TRUE) | any(!is.na(event_time)))) %>%
  ungroup()

# ------------------------------ Helpers ------------------------------
mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)

# cap event_time to [-WIN, WIN] for treated; set 9999 for never-treated
prep_df <- function(dat, WIN) {
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

controls_rhs <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
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

run_es_twfe <- function(lhs, data, ref_val, WIN) {
  fml <- as.formula(paste0(
    lhs,
    " ~ i(event_time_capped, ever_treated, ref = ", ref_val, ", keep = -", as.integer(WIN), ":", as.integer(WIN), ") + ",
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

fmt_wald_cell <- function(res) {
  if (!is.null(res$note)) return("$\\,$")
  sprintf("$%.2f$ (%d) [%.4f]", res$statistic, res$df, res$p.value)
}

outs_lvl <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")
nice_out <- c(rn_hppd="RN", lpn_hppd="LPN", cna_hppd="CNA", total_hppd="Total")
outs_log <- c(rn_hppd="ln_rn", lpn_hppd="ln_lpn", cna_hppd="ln_cna", total_hppd="ln_total")

# ------------------------------ Build three specs ------------------------------
skip2 <- c(-3L,-2L,-1L)

# Spec 1: WITH anticipation, WIN=24
WIN_A <- 24L
dat_with_24 <- prep_df(df0, WIN_A)
ref_with_24 <- pick_ref(dat_with_24, desired = -1L)
test_with_24 <- c(-WIN_A, -2L)  # test -24..-2 (exclude ref -1)

# Spec 2: WITHOUT anticipation, WIN=24 (drop -3,-2,-1), ref -4, test -24..-5
dat_wo_24 <- dat_with_24 %>%
  filter(!(ever_treated == 1L & event_time_capped %in% skip2))
ref_wo_24 <- pick_ref(dat_wo_24, desired = -4L)
test_wo_24 <- c(-WIN_A, -5L)

# Spec 3: WITHOUT anticipation, WIN=12 (drop -3,-2,-1), ref -4, test -12..-5
WIN_C <- 12L
dat_wo_12 <- prep_df(df0, WIN_C) %>%
  filter(!(ever_treated == 1L & event_time_capped %in% skip2))
ref_wo_12 <- pick_ref(dat_wo_12, desired = -4L)
test_wo_12 <- c(-WIN_C, -5L)

specs <- list(
  with_anticip = list(
    row_label = "2 Year Full Pre-Window",
    dat = dat_with_24,
    WIN = WIN_A,
    ref = ref_with_24,
    test_from = test_with_24[1],
    test_to   = test_with_24[2]
  ),
  wo_anticip_24 = list(
    row_label = "2 Year Window with Donut",
    dat = dat_wo_24,
    WIN = WIN_A,
    ref = ref_wo_24,
    test_from = test_wo_24[1],
    test_to   = test_wo_24[2]
  ),
  wo_anticip_12 = list(
    row_label = "1 Year Window with Donut",
    dat = dat_wo_12,
    WIN = WIN_C,
    ref = ref_wo_12,
    test_from = test_wo_12[1],
    test_to   = test_wo_12[2]
  )
)

# ------------------------------ Fit + Wald tests ------------------------------
fit_models_for_spec <- function(sp, is_log = FALSE) {
  mods <- list()
  for (y in outs_lvl) {
    lhs <- if (!is_log) y else outs_log[[y]]
    if (is_log && (lhs %in% names(sp$dat)) && all(is.na(sp$dat[[lhs]]))) {
      mods[[y]] <- NULL
    } else {
      mods[[y]] <- tryCatch(run_es_twfe(lhs, sp$dat, sp$ref, sp$WIN), error = function(e) NULL)
    }
  }
  mods
}

wald_for_spec <- function(mods, sp) {
  res <- lapply(outs_lvl, function(y) {
    pretrend_wald(mods[[y]], ref_tau = sp$ref, from = sp$test_from, to = sp$test_to)
  })
  names(res) <- outs_lvl
  res
}

wald_lvl <- list()
wald_log <- list()
N_rows   <- list()

for (nm in names(specs)) {
  sp <- specs[[nm]]
  
  mods_lvl <- fit_models_for_spec(sp, is_log = FALSE)
  mods_log <- fit_models_for_spec(sp, is_log = TRUE)
  
  wald_lvl[[nm]] <- wald_for_spec(mods_lvl, sp)
  wald_log[[nm]] <- wald_for_spec(mods_log, sp)
  
  N_rows[[nm]] <- nrow(sp$dat)
  cat("[spec]", nm, "|", sp$row_label, "| N =", format(N_rows[[nm]], big.mark=","), "\n")
}

mk_row <- function(rowlabel, reslist) {
  cells <- vapply(outs_lvl, function(y) fmt_wald_cell(reslist[[y]]), character(1))
  paste0(rowlabel, " & ", paste(cells, collapse = " & "), " \\\\")
}

# ------------------------------ LaTeX table ------------------------------
wald_caption <- "Joint Wald Tests of Pre-trends (Event Study)"
wald_label   <- "tab:pretrend-wald-tests"

# Notes: spell out tested windows + refs for each row (keeps table standalone)
notes_windows <- paste0(
  "\\item Tested windows and reference periods: ",
  "2 Year Full Pre-Window tests $\\tau=", specs$with_anticip$test_from, "$ to $\\tau=", specs$with_anticip$test_to,
  "$ with reference $\\tau=", specs$with_anticip$ref, "$; ",
  "2 Year Window with Donut tests $\\tau=", specs$wo_anticip_24$test_from, "$ to $\\tau=", specs$wo_anticip_24$test_to,
  "$ with reference $\\tau=", specs$wo_anticip_24$ref, "$ (dropping $\\tau=-3,-2,-1$); ",
  "1 Year Window with Donut tests $\\tau=", specs$wo_anticip_12$test_from, "$ to $\\tau=", specs$wo_anticip_12$test_to,
  "$ with reference $\\tau=", specs$wo_anticip_12$ref, "$ (dropping $\\tau=-3,-2,-1$)."
)

notes_N <- paste0(
  "\\item Sample sizes (rows): ",
  "2 Year Full Pre-Window ($N=", format(N_rows$with_anticip, big.mark=","), "$); ",
  "2 Year Window with Donut ($N=", format(N_rows$wo_anticip_24, big.mark=","), "$); ",
  "1 Year Window with Donut ($N=", format(N_rows$wo_anticip_12, big.mark=","), "$)."
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
  sprintf(" & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} \\\\",
          nice_out[["rn_hppd"]], nice_out[["lpn_hppd"]], nice_out[["cna_hppd"]], nice_out[["total_hppd"]]),
  "\\midrule",
  
  "\\multicolumn{5}{@{}l}{\\textbf{Panel A: Levels (HPPD)}} \\\\[2pt]",
  mk_row(specs$with_anticip$row_label,      wald_lvl$with_anticip),
  mk_row(specs$wo_anticip_24$row_label,     wald_lvl$wo_anticip_24),
  mk_row(specs$wo_anticip_12$row_label,     wald_lvl$wo_anticip_12),
  
  "\\addlinespace[6pt]",
  "\\multicolumn{5}{@{}l}{\\textbf{Panel B: Logs (HPPD)}} \\\\[2pt]",
  mk_row(specs$with_anticip$row_label,      wald_log$with_anticip),
  mk_row(specs$wo_anticip_24$row_label,     wald_log$wo_anticip_24),
  mk_row(specs$wo_anticip_12$row_label,     wald_log$wo_anticip_12),
  
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
cat("Done.\n")