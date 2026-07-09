# =============================================================================
# regressions/composition_checks_chain_nonchain_preevent.R
#
# Purpose:
#   Same composition-check outcomes as composition_checks_chain_nonchain.R
#   (occupancy rate, Medicare share, Medicaid share, average length of stay),
#   split by chain status -- but using each facility's OWN PRE-EVENT chain
#   status instead of a fixed January 2017 calendar baseline.
#
#   Motivation: the chain-transition check showed that ~47% of treated
#   facilities change chain status around their own ownership-change event
#   (34.9% left a chain, 11.7% joined one). A fixed 2017 baseline can
#   therefore mislabel a facility relative to its OWN transaction -- e.g., a
#   facility that was a chain in Jan 2017 but left the chain years before its
#   actual CHOW would be counted as "chain" under the old classification even
#   though it was independent at the time it was actually sold.
#
# Classification:
#   - TREATED facilities: mode of `chain` over event_time in [-12, -4]
#     (pre-event window, same convention as chain_transition_check.R and the
#     spare_capacity baseline window), i.e., each facility's chain status
#     right before ITS OWN ownership change.
#   - NEVER-TREATED facilities: no acquisition event exists to be "pre" of,
#     so they are classified by the MODE of `chain` across their ENTIRE
#     observed panel (their own typical/persistent status). This is a
#     different reference frame than the treated classification -- flagging
#     explicitly rather than glossing over it, same as the near/slack
#     with-never-treated script.
#
# Output:
#   outputs/tables/composition_checks_chain_nonchain_preevent.tex
#   outputs/tables/chain_preevent_classification_summary.csv
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(stringr)
  library(readr)
})

options(scipen = 999, digits = 4)

out_dir <- out_tables_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tex_out_fp <- file.path(out_dir, "composition_checks_chain_nonchain_preevent.tex")
class_summary_fp <- file.path(out_dir, "chain_preevent_classification_summary.csv")

# -----------------------------------------------------------------------------
# Load monthly staffing panel
# -----------------------------------------------------------------------------
df <- load_staffing_panel()
stopifnot(all(c("treated", "event_time", "chain", "spare_capacity") %in% names(df)))

if (!("year" %in% names(df))) {
  df <- df %>%
    mutate(year = as.integer(str_sub(as.character(year_month), 1, 4)))
}

get_mode <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_real_)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# -----------------------------------------------------------------------------
# Classification: pre-event chain status (treated) / whole-panel mode (never-treated)
# -----------------------------------------------------------------------------

treated_class <- df %>%
  filter(treated == 1, event_time >= -12, event_time <= -4) %>%
  group_by(cms_certification_number) %>%
  summarise(own_chain = get_mode(chain), .groups = "drop") %>%
  filter(!is.na(own_chain)) %>%
  mutate(classification_type = "pre_event_window")

nevertreated_class <- df %>%
  filter(treated == 0) %>%
  group_by(cms_certification_number) %>%
  summarise(own_chain = get_mode(chain), .groups = "drop") %>%
  filter(!is.na(own_chain)) %>%
  mutate(classification_type = "whole_panel_mode")

classification <- bind_rows(treated_class, nevertreated_class)

write_csv(classification, class_summary_fp)

cat("=== Pre-event / own chain classification ===\n")
cat(sprintf("Treated facilities classified (pre-event window):    %d\n", nrow(treated_class)))
cat(sprintf("Never-treated facilities classified (whole-panel):   %d\n", nrow(nevertreated_class)))
cat(sprintf(
  "Chain: %d facilities; Non-chain: %d facilities\n",
  sum(classification$own_chain == 1), sum(classification$own_chain == 0)
))

# -----------------------------------------------------------------------------
# Build analysis sample
# -----------------------------------------------------------------------------

df_wo <- drop_anticipation_window(df) %>%
  inner_join(
    classification %>% select(cms_certification_number, own_chain),
    by = "cms_certification_number"
  )

df_chain    <- df_wo %>% filter(own_chain == 1)
df_nonchain <- df_wo %>% filter(own_chain == 0)

n_chain_fac    <- n_distinct(df_chain$cms_certification_number)
n_nonchain_fac <- n_distinct(df_nonchain$cms_certification_number)

# -----------------------------------------------------------------------------
# Helper functions (same style as composition_checks_chain_nonchain.R)
# -----------------------------------------------------------------------------

coef_se_star <- function(mod, term = "post") {
  sm <- summary(mod)
  ct <- sm$coeftable
  if (!(term %in% rownames(ct))) {
    return(list(coef = NA_real_, se = NA_real_, p = NA_real_, stars = ""))
  }
  b  <- unname(ct[term, "Estimate"])
  se <- unname(ct[term, "Std. Error"])
  p  <- unname(ct[term, "Pr(>|t|)"])
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(coef = b, se = se, p = p, stars = stars)
}

fmt_est <- function(mod, term = "post") {
  s <- coef_se_star(mod, term)
  if (is.na(s$coef) || is.na(s$se)) {
    return("\\makecell[c]{-- \\\\ (--) }")
  }
  b <- sprintf("%.3f", s$coef)
  if (s$coef > 0) b <- paste0("\\phantom{-}", b)
  se <- sprintf("%.3f", s$se)
  if (s$stars == "") {
    paste0("\\makecell[c]{$", b, "$ \\\\ $(", se, ")$}")
  } else {
    paste0("\\makecell[c]{$", b, "^{", s$stars, "}$ \\\\ $(", se, ")$}")
  }
}

fmt_n <- function(mod) format(nobs(mod), big.mark = ",")

make_row <- function(label, mod_nocontrols, mod_controls) {
  paste(
    label,
    fmt_est(mod_nocontrols),
    fmt_est(mod_controls),
    fmt_n(mod_nocontrols),
    sep = " & "
  )
}

# -----------------------------------------------------------------------------
# Regression setup
# -----------------------------------------------------------------------------
vc_month <- ~ cms_certification_number + year_month

# Controls WITHOUT chain (constant within each subsample by construction).
controls_month <- c("beds", "government", "non_profit")

fit_outcome <- function(dat, lhs, controls = TRUE) {
  controls_avail <- intersect(controls_month, names(dat))
  if (controls) {
    rhs <- paste(c("post", controls_avail), collapse = " + ")
  } else {
    rhs <- "post"
  }
  feols(
    as.formula(paste0(lhs, " ~ ", rhs, " | cms_certification_number + year_month")),
    data = dat, vcov = vc_month, lean = FALSE
  )
}

outcomes <- tibble::tribble(
  ~var,             ~label,
  "occupancy_rate",  "Occupancy rate",
  "spare_capacity",  "Spare capacity",
  "pct_medicare",    "Medicare share",
  "pct_medicaid",    "Medicaid share",
  "avg_los_total",   "Average length of stay"
)

fit_group <- function(dat) {
  purrr::map(outcomes$var, function(v) {
    list(
      nocontrols = fit_outcome(dat, v, controls = FALSE),
      controls   = fit_outcome(dat, v, controls = TRUE)
    )
  }) %>% setNames(outcomes$var)
}

fits_chain    <- fit_group(df_chain)
fits_nonchain <- fit_group(df_nonchain)

rows_chain <- purrr::map2_chr(
  outcomes$var, outcomes$label,
  ~ make_row(.y, fits_chain[[.x]]$nocontrols, fits_chain[[.x]]$controls)
)
rows_nonchain <- purrr::map2_chr(
  outcomes$var, outcomes$label,
  ~ make_row(.y, fits_nonchain[[.x]]$nocontrols, fits_nonchain[[.x]]$controls)
)

# -----------------------------------------------------------------------------
# Build LaTeX table
# -----------------------------------------------------------------------------

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
  "\\caption{Effects of Ownership Change: Chain vs. Non-chain Facilities}",
  "\\label{tab:composition-checks-chain-nonchain-preevent}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l Y Y r @{}}",
  "\\toprule",
  "Outcome & No controls & Controls & Observations \\\\",
  "\\midrule",
  paste0("\\multicolumn{4}{@{}l}{\\textbf{Panel A: Chain, N = ", n_chain_fac, " facilities}} \\\\[2pt]"),
  paste0(rows_chain, " \\\\"),
  "\\addlinespace[6pt]",
  paste0("\\multicolumn{4}{@{}l}{\\textbf{Panel B: Non-chain, N = ", n_nonchain_fac, " facilities}} \\\\[2pt]"),
  paste0(rows_nonchain, " \\\\"),
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post}, with standard errors in parentheses. All models include facility and calendar-month fixed effects. The controls column adds beds, government ownership, and nonprofit ownership.",
  "\\item Standard errors are clustered two ways by facility and calendar month. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp, useBytes = TRUE)

cat("\n[write] ", normalizePath(tex_out_fp, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(class_summary_fp, winslash = "\\"), "\n", sep = "")
cat("Done.\n")
