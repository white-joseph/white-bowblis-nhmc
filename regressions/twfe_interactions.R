# C:/Repositories/white-bowblis-nhmc/regressions/twfe_post_interactions.R
# PURPOSE:
#   Formal tests of subgroup differences for TWFE Post models:
#     (1) Chain vs Non-chain (baseline chain in 2017Q1)
#     (2) Pre-pandemic vs Pandemic (post x pandemic-period)
#   Uses the SAME TWFE framework as your tables:
#     - facility FE + month FE
#     - controls
#     - two-way clustered SEs (facility + month)
#     - without anticipation: anticipation2 == 0
#
# OUTPUT:
#   C:/Repositories/white-bowblis-nhmc/outputs/tables/twfe_post_chain_diff.tex
#   C:/Repositories/white-bowblis-nhmc/outputs/tables/twfe_post_pandemic_diff.tex
#   C:/Repositories/white-bowblis-nhmc/outputs/tables/twfe_post_interactions_QA.tex

suppressPackageStartupMessages({
  library(fixest)
  library(dplyr)
  library(readr)
  library(tibble)
})

panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_dir  <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------ Load + prep ------------------
df <- read_csv(panel_fp, show_col_types = FALSE) %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.character(year_month),
    ym_date = as.Date(paste0(gsub("/", "-", year_month), "-01"))
  )

# baseline chain status (2017Q1)
baseline_window <- df %>%
  filter(ym_date >= as.Date("2017-01-01"), ym_date <= as.Date("2017-03-31")) %>%
  arrange(cms_certification_number, ym_date) %>%
  group_by(cms_certification_number) %>%
  summarise(baseline_chain_2017Q1 = dplyr::first(chain), .groups = "drop")

df <- df %>% left_join(baseline_window, by = "cms_certification_number") %>%
  mutate(
    baseline_nonchain_2017Q1 = ifelse(is.na(baseline_chain_2017Q1), NA_integer_, 1L - as.integer(baseline_chain_2017Q1))
  )

# Pandemic period indicator (matches your earlier split; exclude Jan–Mar 2020)
df <- df %>%
  mutate(
    in_prepandemic = ym_date >= as.Date("2017-01-01") & ym_date <= as.Date("2019-12-31"),
    in_pandemic    = ym_date >= as.Date("2020-04-01") & ym_date <= as.Date("2024-06-30"),
    pandemic_period = as.integer(in_pandemic)
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
controls <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)

# FE + 2-way clustering
vc  <- ~ cms_certification_number + year_month
fe_part <- "| cms_certification_number + year_month"

outs_order <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

coef_se_star <- function(mod, term) {
  if (is.null(mod)) return(list(coef=NA, se=NA, p=NA, stars=""))
  ct <- summary(mod)$coeftable
  if (!(term %in% rownames(ct))) return(list(coef=NA, se=NA, p=NA, stars=""))
  b  <- unname(ct[term, "Estimate"])
  se <- unname(ct[term, "Std. Error"])
  p  <- unname(ct[term, "Pr(>|t|)"])
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(coef=b, se=se, p=p, stars=stars)
}

# linear combo: sum_j w_j * beta_j
lincom <- function(mod, weights_named) {
  if (is.null(mod)) return(list(est=NA, se=NA, p=NA, stars=""))
  b <- coef(mod)
  V <- vcov(mod)
  terms <- names(weights_named)
  if (!all(terms %in% names(b))) return(list(est=NA, se=NA, p=NA, stars=""))
  
  w <- weights_named[terms]
  est <- sum(w * b[terms])
  
  # Var = w' V w
  VV <- V[terms, terms, drop = FALSE]
  se <- sqrt(as.numeric(t(w) %*% VV %*% w))
  
  z  <- est / se
  p  <- 2 * (1 - pnorm(abs(z)))
  
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(est=est, se=se, p=p, stars=stars)
}

fmt_est <- function(b, se, stars) {
  if (is.na(b) || is.na(se)) return("\\est{$\\,$}{$\\,$}{}")
  bstr  <- sprintf("%.3f", b); if (b > 0) bstr <- paste0("\\phantom{-}", bstr)
  sestr <- sprintf("%.3f", se)
  sprintf("\\est{$%s$}{$%s$}{%s}", bstr, sestr, stars)
}

# Build one LaTeX row across outcomes for a given extractor
build_row_from <- function(mods_by_outcome, extractor_fn) {
  paste(lapply(outs_order, function(y) {
    out <- extractor_fn(mods_by_outcome[[y]])
    fmt_est(out$b, out$se, out$stars)
  }), collapse = "  &  ")
}

# ------------------------------------------------------------
# Fit interaction models (WITHOUT anticipation only)
# ------------------------------------------------------------

df_wo <- df %>%
  filter(anticipation2 == 0)

# For pandemic interaction: drop Jan-Mar 2020 to match your window definition
df_wo_pandemic <- df_wo %>%
  filter(in_prepandemic | in_pandemic) %>%
  mutate(pandemic_period = as.integer(in_pandemic))

# For chain interaction: require baseline chain defined
df_wo_chain <- df_wo %>%
  filter(!is.na(baseline_nonchain_2017Q1))

# formulas
make_fml_chain <- function(lhs) as.formula(sprintf(
  "%s ~ post + post:baseline_nonchain_2017Q1 + %s %s",
  lhs, controls, fe_part
))

make_fml_pandemic <- function(lhs) as.formula(sprintf(
  "%s ~ post + post:pandemic_period + %s %s",
  lhs, controls, fe_part
))

fit_all_outcomes <- function(dat, make_fml) {
  res <- list(level=list(), log=list())
  for (y in outs_order) {
    # levels
    res$level[[y]] <- feols(make_fml(y), data = dat, vcov = vc, lean = TRUE)
    
    # logs
    lncol <- paste0("ln_", sub("_hppd$","", y))
    if (lncol %in% names(dat) && !all(is.na(dat[[lncol]]))) {
      res$log[[y]] <- feols(make_fml(lncol), data = dat, vcov = vc, lean = TRUE)
    } else {
      res$log[[y]] <- NULL
    }
  }
  res
}

fits_chain   <- fit_all_outcomes(df_wo_chain,   make_fml_chain)
fits_pandemic<- fit_all_outcomes(df_wo_pandemic,make_fml_pandemic)

# ------------------------------------------------------------
# Table builders
# ------------------------------------------------------------

# extractor wrappers returning list(b,se,stars)
get_post <- function(mod) {
  s <- coef_se_star(mod, "post")
  list(b=s$coef, se=s$se, stars=s$stars)
}

get_diff_chain <- function(mod) {
  s <- coef_se_star(mod, "post:baseline_nonchain_2017Q1")
  list(b=s$coef, se=s$se, stars=s$stars)
}

get_nonchain_effect <- function(mod) {
  lc <- lincom(mod, c(post = 1, "post:baseline_nonchain_2017Q1" = 1))
  list(b=lc$est, se=lc$se, stars=lc$stars)
}

get_diff_pandemic <- function(mod) {
  s <- coef_se_star(mod, "post:pandemic_period")
  list(b=s$coef, se=s$se, stars=s$stars)
}

get_pandemic_effect <- function(mod) {
  lc <- lincom(mod, c(post = 1, "post:pandemic_period" = 1))
  list(b=lc$est, se=lc$se, stars=lc$stars)
}

two_panel_three_row_table <- function(fits, caption, label, rowlabs, notes_lines) {
  rowA1 <- build_row_from(fits$level, get_post)
  rowA2 <- build_row_from(fits$level, rowlabs$diff_getter)
  rowA3 <- build_row_from(fits$level, rowlabs$sum_getter)
  
  rowB1 <- build_row_from(fits$log, get_post)
  rowB2 <- build_row_from(fits$log, rowlabs$diff_getter)
  rowB3 <- build_row_from(fits$log, rowlabs$sum_getter)
  
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
    "\\multicolumn{5}{@{}l}{\\textbf{Panel A: Staffing Levels in HPPD}} \\\\[2pt]",
    paste0(rowlabs$A1, " & ", rowA1, " \\\\"),
    paste0(rowlabs$A2, " & ", rowA2, " \\\\"),
    paste0(rowlabs$A3, " & ", rowA3, " \\\\"),
    "\\addlinespace[4pt]",
    "\\multicolumn{5}{@{}l}{\\textbf{Panel B: Log Staffing Levels}} \\\\[2pt]",
    paste0(rowlabs$B1, " & ", rowB1, " \\\\"),
    paste0(rowlabs$B2, " & ", rowB2, " \\\\"),
    paste0(rowlabs$B3, " & ", rowB3, " \\\\"),
    "\\bottomrule",
    "\\end{tabularx}",
    "",
    "\\begin{tablenotes}[flushleft]",
    "\\footnotesize",
    notes_lines,
    "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
    "\\end{tablenotes}",
    "\\end{threeparttable}",
    "\\end{table}",
    "\\endgroup",
    ""
  )
}

# Notes blocks
notes_common <- c(
  "\\item \\textit{Notes:} Each cell reports the indicated coefficient with two-way clustered standard errors (by facility and month) in parentheses.",
  "\\item All specifications include facility and month fixed effects and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators."
)

# ------------------ Build: Chain diff table ------------------
tab_chain <- two_panel_three_row_table(
  fits = fits_chain,
  caption = "TWFE Post Estimates with Interaction: Chain vs Non-chain (Without anticipation)",
  label   = "tab:twfe-post-chain-diff",
  rowlabs = list(
    A1="Chain (baseline 2017Q1): \\textit{post}",
    A2="Difference (Non-chain $-$ Chain): \\textit{post} $\\times$ Non-chain",
    A3="Non-chain effect: \\textit{post} $+$ interaction",
    B1="Chain (baseline 2017Q1): \\textit{post}",
    B2="Difference (Non-chain $-$ Chain): \\textit{post} $\\times$ Non-chain",
    B3="Non-chain effect: \\textit{post} $+$ interaction",
    diff_getter = get_diff_chain,
    sum_getter  = get_nonchain_effect
  ),
  notes_lines = c(
    notes_common,
    sprintf("\\item Sample: facilities with non-missing baseline chain status (2017Q1), without anticipation ($N=%s$).",
            format(nrow(df_wo_chain), big.mark=","))
  )
)

# ------------------ Build: Pandemic diff table ------------------
tab_pandemic <- two_panel_three_row_table(
  fits = fits_pandemic,
  caption = "TWFE Post Estimates with Interaction: Pre-pandemic vs Pandemic (Without anticipation)",
  label   = "tab:twfe-post-pandemic-diff",
  rowlabs = list(
    A1="Pre-pandemic: \\textit{post}",
    A2="Difference (Pandemic $-$ Pre): \\textit{post} $\\times$ Pandemic",
    A3="Pandemic effect: \\textit{post} $+$ interaction",
    B1="Pre-pandemic: \\textit{post}",
    B2="Difference (Pandemic $-$ Pre): \\textit{post} $\\times$ Pandemic",
    B3="Pandemic effect: \\textit{post} $+$ interaction",
    diff_getter = get_diff_pandemic,
    sum_getter  = get_pandemic_effect
  ),
  notes_lines = c(
    notes_common,
    sprintf("\\item Sample: 2017/01--2019/12 and 2020/04--2024/06 (excluding 2020/01--2020/03), without anticipation ($N=%s$).",
            format(nrow(df_wo_pandemic), big.mark=","))
  )
)

# ------------------ write .tex ------------------
chain_path   <- file.path(out_dir, "twfe_post_chain_diff.tex")
pand_path    <- file.path(out_dir, "twfe_post_pandemic_diff.tex")
writeLines(tab_chain, chain_path, useBytes = TRUE)
writeLines(tab_pandemic, pand_path, useBytes = TRUE)

# QA doc to compile and inspect
qa_doc <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{array}",
  "\\usepackage{caption}",
  "\\usepackage{makecell}",
  "\\captionsetup{labelfont=bf, font=small}",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "\\newcommand{\\sym}[1]{\\rlap{$^{#1}$}}",
  "\\newcommand{\\est}[3]{\\makecell[c]{#1\\sym{#3}\\\\ \\footnotesize(#2)}}",
  "\\begin{document}",
  tab_chain,
  tab_pandemic,
  "\\end{document}"
)
qa_path <- file.path(out_dir, "twfe_post_interactions_QA.tex")
writeLines(qa_doc, qa_path, useBytes = TRUE)

cat("[write] ", normalizePath(chain_path, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(pand_path,  winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(qa_path,    winslash = "\\"), "\n", sep = "")
cat("Done.\n")