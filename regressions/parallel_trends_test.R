# C:/Repositories/white-bowblis-nhmc/regressions/pretrend_pvals_by_tau.R
# Table: Pretrend p-values by event time (tau), levels only
# Panel A: With anticipation (ref typically -1)
# Panel B: Without anticipation II (drop t=-3,-2,-1; ref typically -4)
#
# Outputs:
#   - outputs/tables/pretrend_pvals_by_tau_fragment.tex
#   - outputs/tables/pretrend_pvals_by_tau_QA.tex

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
})

options(scipen = 999, digits = 3)

# ------------------------------ Paths ------------------------------
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_dir  <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------ Load + prep ------------------------------
keep_cols <- c(
  "cms_certification_number","year_month","anticipation2",
  "event_time","treatment",
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

# Ever-treated & cap event_time
df <- df %>%
  group_by(cms_certification_number) %>%
  mutate(ever_treated = as.integer(any(treatment == 1, na.rm = TRUE) | any(!is.na(event_time)))) %>%
  ungroup() %>%
  mutate(
    event_time_capped = dplyr::case_when(
      ever_treated == 1L & !is.na(event_time) ~ pmin(pmax(as.integer(event_time), -24L), 24L),
      TRUE ~ 9999L
    )
  )

mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)

df <- df %>%
  mutate(
    ln_rn    = mk_log(rn_hppd),
    ln_lpn   = mk_log(lpn_hppd),
    ln_cna   = mk_log(cna_hppd),
    ln_total = mk_log(total_hppd)
  )

# Controls
controls_rhs <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)

# Helper: choose a valid ref present in data
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

run_es_twfe <- function(lhs, data, ref_val) {
  fml <- as.formula(paste0(
    lhs, " ~ i(event_time_capped, ever_treated, ref = ", ref_val, ", keep = -24:24) + ",
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

# ------------------------------ Specs ------------------------------
# Panel A: With anticipation (full sample)
ref_with <- pick_ref(df, desired = -1L)

# Panel B: Without anticipation II: drop treated bins t in {-3,-2,-1}
skip2 <- c(-3L,-2L,-1L)
df_wo2 <- df %>% filter(!(ever_treated == 1L & event_time_capped %in% skip2))
ref_wo2 <- pick_ref(df_wo2, desired = -4L)

specs <- list(
  A = list(data = df,     ref = ref_with, label = "Panel A: With Anticipation"),
  B = list(data = df_wo2, ref = ref_wo2,  label = "Panel B: Without Anticipation ($t=-3,-2,-1$ dropped)")
)

# Outcomes (levels only)
outs <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")
nice_out <- c(rn_hppd="RN", lpn_hppd="LPN", cna_hppd="CNA", total_hppd="Total")

# ------------------------------ Extract per-tau p-values ------------------------------
# Term naming in fixest for i(var, trt): "event_time_capped::<tau>:ever_treated"
term_name <- function(tau) sprintf("event_time_capped::%d:ever_treated", as.integer(tau))

pval_for_tau <- function(mod, tau) {
  if (is.null(mod)) return(NA_real_)
  tn <- term_name(tau)
  ct <- summary(mod)$coeftable
  if (is.null(ct) || !nrow(ct) || !(tn %in% rownames(ct))) return(NA_real_)
  unname(ct[tn, "Pr(>|t|)"])
}

fmt_p <- function(p) {
  if (is.na(p)) return("$\\,$")
  sprintf("%.4f", p)
}

# Build one panel's body rows: tau rows + p-values by outcome
build_panel_rows <- function(panel_key) {
  sp  <- specs[[panel_key]]
  dat <- sp$data
  ref <- sp$ref
  
  # Fit one ES model per outcome (levels only)
  mods <- lapply(outs, \(y) tryCatch(run_es_twfe(y, dat, ref), error = function(e) NULL))
  names(mods) <- outs
  
  # Tau sequence you requested:
  taus <- if (panel_key == "A") -24L:-1L else -24L:-4L
  
  rows <- character(0)
  for (tau in taus) {
    # Row label
    tau_lab <- if (tau == ref) sprintf("$\\tau=%d$ (Ref.)", tau) else sprintf("$\\tau=%d$", tau)
    
    # Cells
    cells <- vapply(outs, function(y) {
      if (tau == ref) return("\\textit{Ref.}")
      fmt_p(pval_for_tau(mods[[y]], tau))
    }, character(1))
    
    rows <- c(rows, paste0(tau_lab, " & ", paste(cells, collapse = " & "), " \\\\"))
  }
  
  list(
    rows = rows,
    ref  = ref,
    N    = nrow(dat),
    label = sp$label
  )
}

panelA <- build_panel_rows("A")
panelB <- build_panel_rows("B")

# ------------------------------ LaTeX table (fragment) ------------------------------
caption_txt <- "Pre-treatment Event-Time Coefficient p-values by $\\tau$ (Levels Only)"
label_txt   <- "tab:pretrend-pvals-by-tau"

tab <- c(
  "\\begingroup",
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  sprintf("\\caption{%s}", caption_txt),
  sprintf("\\label{%s}", label_txt),
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "",
  "\\begin{tabularx}{\\textwidth}{@{} l YYYY @{} }",
  "\\toprule",
  " & \\multicolumn{4}{c}{\\textbf{Outcomes (HPPD)}} \\\\",
  "\\cmidrule(lr){2-5}",
  sprintf(" & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} \\\\",
          nice_out[["rn_hppd"]], nice_out[["lpn_hppd"]], nice_out[["cna_hppd"]], nice_out[["total_hppd"]]),
  "\\midrule",
  sprintf("\\multicolumn{5}{@{}l}{\\textbf{%s}} \\\\[2pt]", specs$A$label),
  panelA$rows,
  "\\addlinespace[6pt]",
  sprintf("\\multicolumn{5}{@{}l}{\\textbf{%s}} \\\\[2pt]", specs$B$label),
  panelB$rows,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the p-value for the event-time coefficient at $\\tau$ in the TWFE event-study specification with two-way clustered standard errors (by facility and month).",
  sprintf("\\item Reference periods: Panel~A uses $\\tau=%d$ as the reference; Panel~B uses $\\tau=%d$ as the reference.", panelA$ref, panelB$ref),
  sprintf("\\item Sample sizes: Panel~A ($N=%s$); Panel~B ($N=%s$).",
          format(panelA$N, big.mark=","), format(panelB$N, big.mark=",")),
  "\\item All specifications include facility and month fixed effects and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "\\endgroup",
  ""
)

frag_path <- file.path(out_dir, "pretrend_pvals_by_tau_fragment.tex")
writeLines(tab, frag_path, useBytes = TRUE)

# ------------------------------ Standalone QA LaTeX doc ------------------------------
qa_doc <- c(
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
  tab,
  "\\end{document}"
)

qa_path <- file.path(out_dir, "pretrend_pvals_by_tau_QA.tex")
writeLines(qa_doc, qa_path, useBytes = TRUE)

cat("[write] ", normalizePath(frag_path, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(qa_path,   winslash = "\\"), "\n", sep = "")

# ======================================================================
# NEW: Joint Wald pretrend test table (Panels = Levels / Logs)
# Panel A: Levels (with vs without anticipation)
# Panel B: Logs   (with vs without anticipation)
#
# NOTE: Windows differ mechanically because refs/dropped months differ:
#  - With anticipation (ref ~ -1): tests taus -24..-2
#  - Without anticipation II (drop -3,-2,-1; ref ~ -4): tests taus -24..-5
#
# Produces:
#   - outputs/tables/pretrend_wald_tests_fragment.tex
#   - outputs/tables/pretrend_wald_tests_QA.tex
# ======================================================================

suppressPackageStartupMessages({
  library(MASS)   # ginv
})

# ---- Helpers to pick ES coef names/taus and run joint Wald ----
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

# ---- Build "with" and "without" datasets ----
dat_with <- df

skip2 <- c(-3L,-2L,-1L)
dat_wo2 <- df %>% filter(!(ever_treated == 1L & event_time_capped %in% skip2))

# References
ref_with <- pick_ref(dat_with, desired = -1L)
ref_wo2  <- pick_ref(dat_wo2,  desired = -4L)

# Pre windows (see note at top)
win_with <- c(-24L, -2L)
win_wo2  <- c(-24L, -5L)

# Outcomes
outs_lvl <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")
nice_out <- c(rn_hppd="RN", lpn_hppd="LPN", cna_hppd="CNA", total_hppd="Total")

# Log outcomes (match your other scripts)
outs_log <- c(rn_hppd="ln_rn", lpn_hppd="ln_lpn", cna_hppd="ln_cna", total_hppd="ln_total")

# Fit ES models (levels)
mods_lvl_with <- lapply(outs_lvl, \(y) tryCatch(run_es_twfe(y, dat_with, ref_with), error = function(e) NULL))
names(mods_lvl_with) <- outs_lvl
mods_lvl_wo2  <- lapply(outs_lvl, \(y) tryCatch(run_es_twfe(y, dat_wo2,  ref_wo2),  error = function(e) NULL))
names(mods_lvl_wo2)  <- outs_lvl

# Fit ES models (logs) — if the log column is all NA, keep NULL
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

# Row builder
mk_row <- function(rowlabel, reslist) {
  cells <- vapply(outs_lvl, \(y) fmt_wald_cell(reslist[[y]]), character(1))
  paste0(rowlabel, " & ", paste(cells, collapse = " & "), " \\\\")
}

# Sample sizes
N_with_lvl <- nrow(dat_with)
N_wo2_lvl  <- nrow(dat_wo2)

# If you want, you can compute N for logs as "non-missing log outcome rows", but keep simple:
N_with_log <- N_with_lvl
N_wo2_log  <- N_wo2_lvl

# ------------------------------ LaTeX table (fragment) ------------------------------
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
          format(N_with_lvl, big.mark=","), format(N_wo2_lvl, big.mark=",")),
  "\\item All specifications include facility and month fixed effects and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "\\endgroup",
  ""
)

wald_frag_path <- file.path(out_dir, "pretrend_wald_tests_fragment.tex")
writeLines(wald_tab, wald_frag_path, useBytes = TRUE)

# ------------------------------ Standalone QA LaTeX doc ------------------------------
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

cat("[write] ", normalizePath(wald_frag_path, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(wald_qa_path,   winslash = "\\"), "\n", sep = "")
cat("Done.\n")
