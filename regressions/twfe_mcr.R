suppressPackageStartupMessages({
  library(fixest)
  library(dplyr)
  library(readr)
})

# =========================================================
# MCR timing robustness table
# - Baseline/full sample only
# - Panel A: With anticipation
# - Panel B: Without anticipation
# - Each panel has HPRD and Log(HPRD)
# =========================================================

panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel_date_mcr.csv"
out_dir  <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------ Load + prep ------------------
df <- read_csv(panel_fp, show_col_types = FALSE) %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month_chr = as.character(year_month),
    year_month = as.factor(year_month_chr),
    ym_date = as.Date(paste0(gsub("/", "-", year_month_chr), "-01"))
  )

# safe logs
mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)

df <- df %>%
  mutate(
    ln_rn    = mk_log(rn_hppd),
    ln_lpn   = mk_log(lpn_hppd),
    ln_cna   = mk_log(cna_hppd),
    ln_total = mk_log(total_hppd)
  )

# ------------------ model setup ------------------
outs_order <- c("rn_hppd", "lpn_hppd", "cna_hppd", "total_hppd")
log_map <- c(
  rn_hppd    = "ln_rn",
  lpn_hppd   = "ln_lpn",
  cna_hppd   = "ln_cna",
  total_hppd = "ln_total"
)

candidate_controls <- c(
  "government", "non_profit", "chain", "beds",
  "occupancy_rate", "pct_medicare", "pct_medicaid",
  "cm_q_state_2", "cm_q_state_3", "cm_q_state_4"
)

controls <- intersect(candidate_controls, names(df))
rhs <- paste(c("post", controls), collapse = " + ")

make_fml <- function(lhs) {
  as.formula(sprintf(
    "%s ~ %s | cms_certification_number + year_month",
    lhs, rhs
  ))
}

vc <- ~ cms_certification_number + year_month

# ------------------ fitters ------------------
fit_block <- function(dat, drop_anticipation = FALSE) {
  dsub <- dat
  
  if (drop_anticipation) {
    dsub <- dsub %>% filter(anticipation2 == 0)
  }
  
  res <- list(level = list(), log = list())
  
  for (y in outs_order) {
    res$level[[y]] <- feols(
      make_fml(y),
      data = dsub,
      vcov = vc,
      lean = TRUE
    )
    
    lncol <- unname(log_map[[y]])
    if (lncol %in% names(dsub) && !all(is.na(dsub[[lncol]]))) {
      res$log[[y]] <- feols(
        make_fml(lncol),
        data = dsub,
        vcov = vc,
        lean = TRUE
      )
    } else {
      res$log[[y]] <- NULL
    }
  }
  
  res
}

# ------------------ formatting helpers ------------------
coef_se_star <- function(mod, term = "post") {
  if (is.null(mod)) return(list(coef = NA, se = NA, stars = ""))
  
  sm <- summary(mod)
  
  if (!(term %in% rownames(sm$coeftable))) {
    return(list(coef = NA, se = NA, stars = ""))
  }
  
  b  <- unname(coef(mod)[term])
  se <- unname(sm$coeftable[term, "Std. Error"])
  p  <- unname(sm$coeftable[term, "Pr(>|t|)"])
  
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
  
  list(coef = b, se = se, stars = stars)
}

fmt_est <- function(b, se, stars) {
  if (is.na(b) || is.na(se)) return("\\est{$\\,$}{$\\,$}{}")
  bstr  <- sprintf("%.3f", b)
  if (b > 0) bstr <- paste0("\\phantom{-}", bstr)
  sestr <- sprintf("%.3f", se)
  sprintf("\\est{$%s$}{$%s$}{%s}", bstr, sestr, stars)
}

build_row <- function(model_set) {
  paste(
    lapply(outs_order, function(y) {
      s <- coef_se_star(model_set[[y]])
      fmt_est(s$coef, s$se, s$stars)
    }),
    collapse = " & "
  )
}

# ------------------ sample-size helpers ------------------
get_n_vec <- function(model_set) {
  sapply(outs_order, function(y) {
    mod <- model_set[[y]]
    if (is.null(mod)) return(NA_integer_)
    nobs(mod)
  })
}

fmt_n_note <- function(nvec) {
  paste(
    sprintf("%s=%s",
            c("RN", "LPN", "CNA", "Total"),
            format(nvec, big.mark = ",")),
    collapse = "; "
  )
}

# ------------------ build table ------------------
build_mcr_baseline_table <- function(res_with, res_without, label, caption) {
  row_A_hprd <- build_row(res_with$level)
  row_A_log  <- build_row(res_with$log)
  row_B_hprd <- build_row(res_without$level)
  row_B_log  <- build_row(res_without$log)
  
  nA_levels <- fmt_n_note(get_n_vec(res_with$level))
  nA_logs   <- fmt_n_note(get_n_vec(res_with$log))
  nB_levels <- fmt_n_note(get_n_vec(res_without$level))
  nB_logs   <- fmt_n_note(get_n_vec(res_without$log))
  
  c(
    "\\begingroup",
    "\\begin{table}[!ht]",
    "\\centering",
    "\\begin{threeparttable}",
    sprintf("\\caption{%s}", caption),
    sprintf("\\label{%s}", label),
    "\\small",
    "\\setlength{\\tabcolsep}{6pt}",
    "",
    "\\begin{tabularx}{\\textwidth}{@{} l YYYY @{} }",
    "\\toprule",
    " & \\multicolumn{4}{c}{\\textbf{Outcomes}} \\\\",
    "\\cmidrule(lr){2-5}",
    " & \\textbf{RN} & \\textbf{LPN} & \\textbf{CNA} & \\textbf{Total} \\\\",
    "\\midrule",
    "\\multicolumn{5}{@{}l}{\\textbf{Panel A: With anticipation}} \\\\[2pt]",
    paste0("HPRD & ", row_A_hprd, " \\\\"),
    "\\addlinespace[3pt]",
    paste0("Log(HPRD) & ", row_A_log, " \\\\"),
    "",
    "\\addlinespace[5pt]",
    "\\multicolumn{5}{@{}l}{\\textbf{Panel B: Without anticipation}} \\\\[2pt]",
    paste0("HPRD & ", row_B_hprd, " \\\\"),
    "\\addlinespace[3pt]",
    paste0("Log(HPRD) & ", row_B_log, " \\\\"),
    "\\bottomrule",
    "\\end{tabularx}",
    "",
    "\\begin{tablenotes}[flushleft]",
    "\\footnotesize",
    "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post} with two-way clustered standard errors (by facility and month) in parentheses.",
    "\\item Panel~A uses the full baseline sample with anticipation periods included. Panel~B excludes observations with \\textit{anticipation2}=1 (that is, event time $\\in \\{-3,-2,-1\\}$).",
    "\\item All specifications include facility and month fixed effects and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators, when available in the panel.",
    sprintf("\\item Estimation sample sizes for Panel~A HPRD models: %s.", nA_levels),
    sprintf("\\item Estimation sample sizes for Panel~A log(HPRD) models: %s.", nA_logs),
    sprintf("\\item Estimation sample sizes for Panel~B HPRD models: %s.", nB_levels),
    sprintf("\\item Estimation sample sizes for Panel~B log(HPRD) models: %s.", nB_logs),
    "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
    "\\end{tablenotes}",
    "\\end{threeparttable}",
    "\\end{table}",
    "\\endgroup",
    ""
  )
}

# ------------------ run models ------------------
fits_with    <- fit_block(df, drop_anticipation = FALSE)
fits_without <- fit_block(df, drop_anticipation = TRUE)

tab_mcr <- build_mcr_baseline_table(
  res_with    = fits_with,
  res_without = fits_without,
  label   = "tab:twfe-post-full-date-mcr",
  caption = "Two-Way Fixed Effects Estimates of \\textit{post} on Staffing Outcomes Using MCR Event Timing"
)

# ------------------ write outputs ------------------
tab_path  <- file.path(out_dir, "twfe_post_full_date_mcr.tex")
frag_path <- file.path(out_dir, "twfe_tables_all_date_mcr.tex")
qa_path   <- file.path(out_dir, "twfe_tables_QA_date_mcr.tex")

writeLines(tab_mcr, tab_path,  useBytes = TRUE)
writeLines(tab_mcr, frag_path, useBytes = TRUE)

full_doc <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{array}",
  "\\usepackage{caption}",
  "\\usepackage{makecell}",
  "\\usepackage{graphicx}",
  "\\captionsetup{labelfont=bf, font=small}",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "\\newcommand{\\sym}[1]{\\rlap{$^{#1}$}}",
  "\\newcommand{\\est}[3]{\\makecell[c]{#1\\sym{#3}\\\\ \\footnotesize(#2)}}",
  "\\begin{document}",
  tab_mcr,
  "\\end{document}"
)

writeLines(full_doc, qa_path, useBytes = TRUE)

cat("[write] ", normalizePath(tab_path,  winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(frag_path, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(qa_path,   winslash = "\\"), "\n", sep = "")
cat("Done.\n")