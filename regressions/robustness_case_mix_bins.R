# =============================================================================
# regressions/robustness_case_mix_bins.R
#
# Purpose:
#   Test whether the choice of case-mix control granularity (state vs.
#   national reference group; quartile vs. decile bins) meaningfully changes
#   the main TWFE post-regression results on staffing HPRD. The project's
#   current default (see _setup.R's preferred_case_mix_controls) is STATE
#   QUARTILES -- this reruns the identical specification with all four
#   combinations, changing ONLY the case-mix control set each time.
#
# Sample / specification held fixed across all four columns:
#   - Same outcomes: RN, LPN, CNA, Total HPRD
#   - Same base controls (government, non_profit, chain, beds, occupancy
#     rate, Medicare share, Medicaid share)
#   - Same facility + calendar-month fixed effects
#   - Same anticipation-window exclusion
#   - Same clustering (facility x calendar month)
#
# Output:
#   outputs/tables/robustness_case_mix_bins.tex
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(tibble)
})

options(scipen = 999, digits = 4)

out_dir <- out_tables_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tex_out_fp <- file.path(out_dir, "robustness_case_mix_bins.tex")

# -----------------------------------------------------------------------------
# Load panel
# -----------------------------------------------------------------------------
df <- load_staffing_panel()
df_wo <- drop_anticipation_window(df)

# -----------------------------------------------------------------------------
# Define the four case-mix control-set variants
# -----------------------------------------------------------------------------
case_mix_variants <- list(
  "State Quartile (current default)" = c("cm_q_state_2", "cm_q_state_3", "cm_q_state_4"),
  "State Decile" = paste0("cm_d_state_", 2:10),
  "National Quartile" = c("cm_q_nat_2", "cm_q_nat_3", "cm_q_nat_4"),
  "National Decile" = paste0("cm_d_nat_", 2:10)
)

# Confirm which variants are actually present in the panel before proceeding
for (nm in names(case_mix_variants)) {
  present <- intersect(case_mix_variants[[nm]], names(df_wo))
  missing <- setdiff(case_mix_variants[[nm]], names(df_wo))
  cat(sprintf("[check] %-35s present=%d, missing=%s\n", nm, length(present),
              if (length(missing) == 0) "none" else paste(missing, collapse = ", ")))
  case_mix_variants[[nm]] <- present
}

vc_month <- ~ cms_certification_number + year_month

build_rhs <- function(case_mix_cols) {
  ctrls <- c(base_controls, case_mix_cols)
  paste(c("post", ctrls), collapse = " + ")
}

fit_variant <- function(lhs, case_mix_cols) {
  rhs <- build_rhs(case_mix_cols)
  feols(
    as.formula(paste0(lhs, " ~ ", rhs, " | cms_certification_number + year_month")),
    data = df_wo, vcov = vc_month, lean = TRUE
  )
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
coef_se_star <- function(mod, term = "post") {
  ct <- summary(mod)$coeftable
  if (!(term %in% rownames(ct))) return(list(coef = NA, se = NA, stars = ""))
  b  <- unname(ct[term, "Estimate"])
  se <- unname(ct[term, "Std. Error"])
  p  <- unname(ct[term, "Pr(>|t|)"])
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(coef = b, se = se, stars = stars)
}

fmt_cell <- function(mod) {
  s <- coef_se_star(mod)
  if (is.na(s$coef) || is.na(s$se)) return("\\makecell[t]{-- \\\\ (--)}")
  b <- sprintf("%.4f", s$coef); if (s$coef > 0) b <- paste0("\\phantom{-}", b)
  se <- sprintf("%.4f", s$se)
  if (s$stars == "") {
    paste0("\\makecell[t]{$", b, "$ \\\\ $(", se, ")$}")
  } else {
    paste0("\\makecell[t]{$", b, "^{", s$stars, "}$ \\\\ $(", se, ")$}")
  }
}

# -----------------------------------------------------------------------------
# Fit all outcome x variant combinations
# -----------------------------------------------------------------------------
outcomes <- tibble::tribble(
  ~var,          ~label,
  "rn_hprd",     "RN HPRD",
  "lpn_hprd",    "LPN HPRD",
  "cna_hprd",    "CNA HPRD",
  "total_hprd",  "Total HPRD"
)

variant_names <- names(case_mix_variants)

results <- outcomes %>%
  rowwise() %>%
  mutate(
    cells = list(sapply(variant_names, function(vn) {
      fmt_cell(fit_variant(var, case_mix_variants[[vn]]))
    }))
  ) %>%
  ungroup()

table_rows <- sapply(seq_len(nrow(results)), function(i) {
  paste(results$label[i], paste(results$cells[[i]], collapse = " & "), sep = " & ")
})
table_rows <- paste0(table_rows, " \\\\")

n_obs <- format(nobs(fit_variant("rn_hprd", case_mix_variants[[1]])), big.mark = ",")

# -----------------------------------------------------------------------------
# Build LaTeX table
# -----------------------------------------------------------------------------

col_headers <- paste(variant_names, collapse = " & ")

tex_lines <- c(
  "\\documentclass[12pt]{article}",
  "",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{makecell}",
  "\\usepackage{array}",
  "\\usepackage{caption}",
  "\\usepackage{float}",
  "",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "",
  "\\begin{document}",
  "",
  "\\begin{table}[H]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Robustness: Case-Mix Control Granularity (State vs. National, Quartile vs. Decile)}",
  "\\label{tab:robustness-case-mix-bins}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l Y Y Y Y @{}}",
  "\\toprule",
  paste0("Outcome & ", col_headers, " \\\\"),
  "\\midrule",
  table_rows,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  sprintf("\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses. $N = %s$ for all models. Only the case-mix control set changes across columns -- base controls (beds, government, non-profit, chain, occupancy rate, Medicare share, Medicaid share), facility and calendar-month fixed effects, and the anticipation-window exclusion are identical across all four columns.", n_obs),
  "\\item ``State Quartile'' is the project's current default (see \\texttt{\\_setup.R}'s \\texttt{preferred\\_case\\_mix\\_controls}). Quartile bins use dummies for bins 2-4 (bin 1 omitted as reference); decile bins use dummies for bins 2-10 (bin 1 omitted as reference).",
  "\\item Standard errors are clustered two ways by facility and calendar month. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp, useBytes = TRUE)

cat("\n[write] ", normalizePath(tex_out_fp, winslash = "\\"), "\n", sep = "")
cat("Done.\n")
