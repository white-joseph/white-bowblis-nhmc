source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

# -----------------------------------------------------------------------------
# Output path
# -----------------------------------------------------------------------------
out_dir <- out_tables_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Load panel
# -----------------------------------------------------------------------------
df <- load_staffing_panel()

# -----------------------------------------------------------------------------
# Controls / RHS / vcov
# -----------------------------------------------------------------------------
controls_rhs <- make_controls_rhs(df)
rhs <- if (controls_rhs == "1") "post" else paste("post +", controls_rhs)
vc <- ~ cms_certification_number + year_month

# -----------------------------------------------------------------------------
# Outcomes and dataset windows
# -----------------------------------------------------------------------------
outs_order <- staffing_outcomes

df_wo <- drop_anticipation_window(df)

datasets <- list(
  full = df_wo,
  prepandemic = sample_prepandemic(df_wo),
  pandemic = sample_pandemic(df_wo)
)

# Baseline chain / non-chain classification from January 2017
jan2017_chain <- df %>%
  filter(year_month == "2017/01") %>%
  distinct(cms_certification_number, chain)

chain_ccns <- jan2017_chain %>%
  filter(chain == 1) %>%
  pull(cms_certification_number)

nonchain_ccns <- jan2017_chain %>%
  filter(chain == 0) %>%
  pull(cms_certification_number)

datasets$baseline_chain_2017q1 <- df_wo %>%
  filter(cms_certification_number %in% chain_ccns)

datasets$baseline_nonchain_2017q1 <- df_wo %>%
  filter(cms_certification_number %in% nonchain_ccns)

# -----------------------------------------------------------------------------
# Formula helper
# -----------------------------------------------------------------------------
make_fml <- function(lhs) {
  as.formula(
    sprintf("%s ~ %s | cms_certification_number + year_month", lhs, rhs)
  )
}

# -----------------------------------------------------------------------------
# Fit blocks
# -----------------------------------------------------------------------------
fit_block_without_only <- function(dsub) {
  res <- list(level = list(), log = list())
  
  for (y in outs_order) {
    # level
    if (y %in% names(dsub) && !all(is.na(dsub[[y]]))) {
      res$level[[y]] <- feols(make_fml(y), data = dsub, vcov = vc, lean = TRUE)
    } else {
      res$level[[y]] <- NULL
    }
    
    # log
    lncol <- unname(log_outcome_map[[y]])
    if (!is.null(lncol) && lncol %in% names(dsub) && !all(is.na(dsub[[lncol]]))) {
      res$log[[y]] <- feols(make_fml(lncol), data = dsub, vcov = vc, lean = TRUE)
    } else {
      res$log[[y]] <- NULL
    }
  }
  
  res
}

# -----------------------------------------------------------------------------
# Table helpers
# -----------------------------------------------------------------------------
coef_se_star <- function(mod, term = "post") {
  if (is.null(mod)) return(list(coef = NA, se = NA, stars = ""))
  
  sm <- summary(mod)
  b <- unname(coef(mod)[term])
  se <- unname(sm$coeftable[term, "Std. Error"])
  p <- unname(sm$coeftable[term, "Pr(>|t|)"])
  
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
  bstr <- sprintf("%.3f", b)
  if (b > 0) bstr <- paste0("\\phantom{-}", bstr)
  sestr <- sprintf("%.3f", se)
  sprintf("\\est{$%s$}{$%s$}{%s}", bstr, sestr, stars)
}

build_row <- function(mset) {
  paste(
    lapply(outs_order, function(y) {
      s <- coef_se_star(mset[[y]])
      fmt_est(s$coef, s$se, s$stars)
    }),
    collapse = "  &  "
  )
}

# -----------------------------------------------------------------------------
# Table builders
# -----------------------------------------------------------------------------
one_table_fragment_without_only_twfe_post_full <- function(res_wo, dat_all, caption, label, notes_extra = NULL) {
  dat_wo <- drop_anticipation_window(dat_all)
  
  log_cols <- unname(log_outcome_map[outs_order])
  
  Ns_without <- list(
    levels = format(nrow(dat_wo), big.mark = ","),
    logs   = format(sum(rowSums(!is.na(dat_wo[, log_cols, drop = FALSE])) > 0), big.mark = ",")
  )
  
  row_HPPD <- build_row(res_wo$level)
  row_LOG  <- build_row(res_wo$log)
  
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
    paste0("HPRD & ", row_HPPD, " \\\\"),
    "\\addlinespace[3pt]",
    paste0("Log(HPRD) & ", row_LOG, " \\\\"),
    "\\bottomrule",
    "\\end{tabularx}",
    "",
    "\\begin{tablenotes}[flushleft]",
    "\\footnotesize",
    sprintf(
      "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post} with two-way clustered standard errors (by facility and month) in parentheses. The table reports staffing levels (HPRD) and log staffing levels (Log(HPRD)). Sample: Without anticipation ($N_{\\mathrm{HPRD}}=%s;\\ N_{\\mathrm{Log}}=%s$).",
      Ns_without$levels, Ns_without$logs
    ),
    "\\item All specifications include facility and month fixed effects and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators.",
    "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
    if (!is.null(notes_extra)) paste0("\\item ", notes_extra) else NULL,
    "\\end{tablenotes}",
    "\\end{threeparttable}",
    "\\end{table}",
    "\\endgroup",
    ""
  )
}

two_dataset_table_without_only <- function(res1, res2, dat1, dat2, cap, label, rowlabs, notes_extra = NULL) {
  log_cols <- unname(log_outcome_map[outs_order])
  
  Ns1 <- list(
    levels = format(nrow(dat1), big.mark = ","),
    logs   = format(sum(rowSums(!is.na(dat1[, log_cols, drop = FALSE])) > 0), big.mark = ",")
  )
  Ns2 <- list(
    levels = format(nrow(dat2), big.mark = ","),
    logs   = format(sum(rowSums(!is.na(dat2[, log_cols, drop = FALSE])) > 0), big.mark = ",")
  )
  
  rowA1 <- build_row(res1$level)
  rowA2 <- build_row(res2$level)
  rowB1 <- build_row(res1$log)
  rowB2 <- build_row(res2$log)
  
  c(
    "\\begingroup",
    "\\begin{table}[!ht]",
    "\\centering",
    "\\begin{threeparttable}",
    sprintf("\\caption{%s}", cap),
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
    "\\multicolumn{5}{@{}l}{\\textbf{Panel A: Staffing Levels in HPRD}} \\\\[2pt]",
    paste0(rowlabs[1], " & ", rowA1, " \\\\"),
    paste0(rowlabs[2], " & ", rowA2, " \\\\"),
    "",
    "\\addlinespace[3pt]",
    "\\multicolumn{5}{@{}l}{\\textbf{Panel B: Log Staffing Levels in HPRD}} \\\\[2pt]",
    paste0(rowlabs[1], " & ", rowB1, " \\\\"),
    paste0(rowlabs[2], " & ", rowB2, " \\\\"),
    "\\bottomrule",
    "\\end{tabularx}",
    "",
    "\\begin{tablenotes}[flushleft]",
    "\\footnotesize",
    sprintf(
      "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post} with two-way clustered standard errors (by facility and month) in parentheses. Panel~A reports levels (HPRD); Panel~B reports logs (HPRD). Sample sizes: Row~1 ($N_{\\mathrm{levels}}=%s;\\ N_{\\mathrm{logs}}=%s$), Row~2 ($N_{\\mathrm{levels}}=%s;\\ N_{\\mathrm{logs}}=%s$).",
      Ns1$levels, Ns1$logs, Ns2$levels, Ns2$logs
    ),
    "\\item All specifications include facility and month fixed effects and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators.",
    "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
    if (!is.null(notes_extra)) paste0("\\item ", notes_extra) else NULL,
    "\\end{tablenotes}",
    "\\end{threeparttable}",
    "\\end{table}",
    "\\endgroup",
    ""
  )
}

# -----------------------------------------------------------------------------
# Run models
# -----------------------------------------------------------------------------
fits_wo <- lapply(datasets, fit_block_without_only)

# -----------------------------------------------------------------------------
# Table 1: Baseline overall (WITHOUT anticipation only)
# -----------------------------------------------------------------------------
tab1 <- one_table_fragment_without_only_twfe_post_full(
  res_wo   = fits_wo$full,
  dat_all  = datasets$full,
  caption  = "Two-Way Fixed Effects Estimates of \\textit{post} on Staffing Outcomes (Baseline, Without anticipation)",
  label    = "tab:twfe-post-full"
)

# -----------------------------------------------------------------------------
# Table 2: Pre vs Post
# -----------------------------------------------------------------------------
tab2 <- two_dataset_table_without_only(
  res1 = fits_wo$prepandemic,
  res2 = fits_wo$pandemic,
  dat1 = datasets$prepandemic,
  dat2 = datasets$pandemic,
  cap = "TWFE Estimates of \\textit{post}: Pre- vs Post-pandemic Periods (Without anticipation)",
  label = "tab:twfe-prepost",
  rowlabs = c("Pre-Pandemic Period (2017/01 - 2019/12)", "Pandemic Period (2020/04 - 2024/06)"),
  notes_extra = "Pre-pandemic 2017/01--2019/12; Pandemic 2020/04--2024/06."
)

# -----------------------------------------------------------------------------
# Table 3: Chain vs Non-chain
# -----------------------------------------------------------------------------
tab3 <- two_dataset_table_without_only(
  res1 = fits_wo$baseline_chain_2017q1,
  res2 = fits_wo$baseline_nonchain_2017q1,
  dat1 = datasets$baseline_chain_2017q1,
  dat2 = datasets$baseline_nonchain_2017q1,
  cap = "TWFE Estimates of \\textit{post}: Chain vs Non-chain Facilities (Jan 2017 Baseline, Without anticipation)",
  label = "tab:twfe-chain-nonchain",
  rowlabs = c("Chain January 2017", "Non-chain January 2017"),
  notes_extra = "Baseline chain classification determined by facility status in January 2017."
)

# -----------------------------------------------------------------------------
# Write .tex outputs
# -----------------------------------------------------------------------------
tab1_path <- file.path(out_dir, "twfe_post_full.tex")
tab2_path <- file.path(out_dir, "twfe_prepost.tex")
tab3_path <- file.path(out_dir, "twfe_chain_nonchain.tex")

writeLines(tab1, tab1_path, useBytes = TRUE)
writeLines(tab2, tab2_path, useBytes = TRUE)
writeLines(tab3, tab3_path, useBytes = TRUE)

# Combined fragment
all_fragment <- c(tab1, tab2, tab3)
frag_path <- file.path(out_dir, "twfe_tables_all.tex")
writeLines(all_fragment, frag_path, useBytes = TRUE)

# Standalone QA doc
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
  "",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "\\newcommand{\\est}[3]{\\makecell[c]{$#1$ \\\\ $(#2)$ \\\\ #3}}",
  "",
  "\\begin{document}",
  tab1,
  "\\clearpage",
  tab2,
  "\\clearpage",
  tab3,
  "\\end{document}",
  ""
)

full_doc_path <- file.path(out_dir, "twfe_tables_preview.tex")
writeLines(full_doc, full_doc_path, useBytes = TRUE)

cat("\nSaved:\n")
cat(" -", tab1_path, "\n")
cat(" -", tab2_path, "\n")
cat(" -", tab3_path, "\n")
cat(" -", frag_path, "\n")
cat(" -", full_doc_path, "\n")
cat("\nDone.\n")