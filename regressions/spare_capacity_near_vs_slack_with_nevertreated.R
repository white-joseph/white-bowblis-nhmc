# =============================================================================
# regressions/spare_capacity_near_vs_slack_with_nevertreated.R
#
# Purpose:
#   Compare the near-capacity vs. slack-capacity static regression under two
#   samples:
#     Panel A: TREATED-ONLY (same as spare_capacity_report.R / the version
#              already sent to advisors) -- treated-near-capacity vs.
#              treated-slack-capacity, compared to EACH OTHER.
#     Panel B: FULL PANEL, including never-treated facilities as additional
#              controls in the fixed-effects estimation. post is always 0
#              for never-treated facilities, so they do not mechanically
#              change the "post" or "post:near_capacity" coefficients on
#              their own -- but including them changes what the calendar
#              fixed effects (and therefore "post") are estimated relative
#              to, since the counterfactual trend no longer comes only from
#              other treated facilities.
#
# IMPORTANT ASYMMETRY -- read before interpreting Panel B:
#   Treated facilities are classified as near/slack-capacity using their OWN
#   PRE-ACQUISITION window (event_time in [-12,-4]) -- an event-relative
#   reference point that doesn't exist for a facility that was never
#   acquired. Never-treated facilities therefore CANNOT be classified the
#   same way. As a stand-in, they are classified using a FIXED CALENDAR
#   window (2017 Q1), the same convention this project already uses to
#   define baseline chain status. This means the two groups' classifications
#   are not perfectly comparable in timing -- flagging this explicitly
#   rather than glossing over it. Both treated and never-treated baseline
#   values are pooled together and split at ONE combined median so there is
#   a single, consistent near/slack cutoff across both groups.
#
# Output:
#   outputs/tables/spare_capacity_near_vs_slack_comparison.tex
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(tibble)
  library(readr)
})

options(scipen = 999, digits = 4)

out_dir <- out_tables_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tex_out_fp <- file.path(out_dir, "spare_capacity_near_vs_slack_comparison.tex")

# -----------------------------------------------------------------------------
# Load panel
# -----------------------------------------------------------------------------
df <- load_staffing_panel()
stopifnot(all(c("spare_capacity", "occupancy_rate", "treated") %in% names(df)))

# -----------------------------------------------------------------------------
# Baseline classification
# -----------------------------------------------------------------------------

# Treated facilities: average spare_capacity in event_time [-12,-4]
baseline_treated <- df %>%
  filter(treated == 1, event_time >= -12, event_time <= -4) %>%
  group_by(cms_certification_number) %>%
  summarise(
    baseline_spare_capacity = mean(spare_capacity, na.rm = TRUE),
    n_baseline_months = sum(!is.na(spare_capacity)),
    .groups = "drop"
  ) %>%
  filter(n_baseline_months > 0, is.finite(baseline_spare_capacity)) %>%
  mutate(baseline_type = "event_time_pre_acquisition")

# Never-treated facilities: no event_time exists, so use a FIXED CALENDAR
# baseline window instead (2017 Q1), matching this project's convention for
# other time-invariant baseline traits (e.g. baseline chain status).
baseline_never_treated <- df %>%
  filter(treated == 0, year_month %in% c("2017/01", "2017/02", "2017/03")) %>%
  group_by(cms_certification_number) %>%
  summarise(
    baseline_spare_capacity = mean(spare_capacity, na.rm = TRUE),
    n_baseline_months = sum(!is.na(spare_capacity)),
    .groups = "drop"
  ) %>%
  filter(n_baseline_months > 0, is.finite(baseline_spare_capacity)) %>%
  mutate(baseline_type = "fixed_2017q1")

baseline_all <- bind_rows(baseline_treated, baseline_never_treated)

# Single combined median across BOTH groups' baseline values, so there is one
# consistent near/slack cutoff even though the reference windows differ.
baseline_median <- median(baseline_all$baseline_spare_capacity, na.rm = TRUE)

baseline_all <- baseline_all %>%
  mutate(near_capacity = as.integer(baseline_spare_capacity <= baseline_median))

cat("=== Baseline classification ===\n")
cat(sprintf("Treated facilities with usable baseline (event_time -12 to -4): %d\n", nrow(baseline_treated)))
cat(sprintf("Never-treated facilities with usable baseline (2017 Q1):        %d\n", nrow(baseline_never_treated)))
cat(sprintf("Combined median spare_capacity used for split: %.3f\n", baseline_median))
cat(sprintf(
  "Near-capacity (<= median): %d facilities; Slack-capacity (> median): %d facilities\n",
  sum(baseline_all$near_capacity == 1), sum(baseline_all$near_capacity == 0)
))

# -----------------------------------------------------------------------------
# Build the two analysis samples
# -----------------------------------------------------------------------------

# Panel A: treated-only (as sent to advisors)
df_treated_only <- df %>%
  filter(treated == 1) %>%
  inner_join(
    baseline_all %>% select(cms_certification_number, near_capacity),
    by = "cms_certification_number"
  ) %>%
  drop_anticipation_window()

# Panel B: full panel, including never-treated facilities as controls
df_full <- df %>%
  inner_join(
    baseline_all %>% select(cms_certification_number, near_capacity),
    by = "cms_certification_number"
  ) %>%
  drop_anticipation_window()

cat(sprintf(
  "\n[Panel A: treated-only]     facility-months = %s, facilities = %d\n",
  format(nrow(df_treated_only), big.mark = ","), n_distinct(df_treated_only$cms_certification_number)
))
cat(sprintf(
  "[Panel B: full panel]       facility-months = %s, facilities = %d\n",
  format(nrow(df_full), big.mark = ","), n_distinct(df_full$cms_certification_number)
))

# -----------------------------------------------------------------------------
# Regression setup (shared)
# -----------------------------------------------------------------------------

vc <- as.formula(paste0("~ ", fe_unit, " + ", fe_time))
fe_part <- paste0("| ", fe_unit, " + ", fe_time)

outcome_related_excludes <- list(
  spare_capacity = c("occupancy_rate", "spare_capacity"),
  occupancy_rate = c("occupancy_rate", "spare_capacity")
)

controls_rhs_for <- function(dat, outcome) {
  ctrls <- get_controls(dat)
  ctrls <- setdiff(ctrls, outcome_related_excludes[[outcome]])
  if (length(ctrls) == 0) return("1")
  paste(ctrls, collapse = " + ")
}

make_static_fml <- function(dat, lhs) {
  ctrls <- controls_rhs_for(dat, lhs)
  rhs_ctrl_part <- if (ctrls == "1") "" else paste0(" + ", ctrls)
  as.formula(sprintf("%s ~ post + post:near_capacity%s %s", lhs, rhs_ctrl_part, fe_part))
}

coef_se_star <- function(mod, term) {
  ct <- summary(mod)$coeftable
  if (!(term %in% rownames(ct))) return(list(coef = NA, se = NA, stars = ""))
  b  <- unname(ct[term, "Estimate"])
  se <- unname(ct[term, "Std. Error"])
  p  <- unname(ct[term, "Pr(>|t|)"])
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(coef = b, se = se, stars = stars)
}

lincom <- function(mod, weights_named) {
  b <- coef(mod); V <- vcov(mod)
  terms <- names(weights_named)
  if (!all(terms %in% names(b))) return(list(est = NA, se = NA, stars = ""))
  w <- weights_named[terms]
  est <- sum(w * b[terms])
  VV <- V[terms, terms, drop = FALSE]
  se <- sqrt(as.numeric(t(w) %*% VV %*% w))
  p <- 2 * (1 - pnorm(abs(est / se)))
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(est = est, se = se, stars = stars)
}

fmt_cell <- function(b, se, stars) {
  if (is.na(b) || is.na(se)) return("\\makecell[c]{-- \\\\ (--)}")
  bstr <- sprintf("%.4f", b); if (b > 0) bstr <- paste0("\\phantom{-}", bstr)
  sestr <- sprintf("%.4f", se)
  paste0("\\makecell[c]{$", bstr, "^{", stars, "}$ \\\\ $(", sestr, ")$}")
}

fit_panel <- function(dat) {
  m_sc  <- feols(make_static_fml(dat, "spare_capacity"), data = dat, vcov = vc, lean = TRUE)
  m_occ <- feols(make_static_fml(dat, "occupancy_rate"), data = dat, vcov = vc, lean = TRUE)

  sc_slack <- coef_se_star(m_sc, "post")
  sc_diff  <- coef_se_star(m_sc, "post:near_capacity")
  sc_near  <- lincom(m_sc, c(post = 1, "post:near_capacity" = 1))

  occ_slack <- coef_se_star(m_occ, "post")
  occ_diff  <- coef_se_star(m_occ, "post:near_capacity")
  occ_near  <- lincom(m_occ, c(post = 1, "post:near_capacity" = 1))

  list(
    row_slack = paste(
      fmt_cell(sc_slack$coef, sc_slack$se, sc_slack$stars),
      fmt_cell(occ_slack$coef, occ_slack$se, occ_slack$stars),
      sep = " & "
    ),
    row_diff = paste(
      fmt_cell(sc_diff$coef, sc_diff$se, sc_diff$stars),
      fmt_cell(occ_diff$coef, occ_diff$se, occ_diff$stars),
      sep = " & "
    ),
    row_near = paste(
      fmt_cell(sc_near$est, sc_near$se, sc_near$stars),
      fmt_cell(occ_near$est, occ_near$se, occ_near$stars),
      sep = " & "
    ),
    n = format(nobs(m_sc), big.mark = ",")
  )
}

panelA <- fit_panel(df_treated_only)
panelB <- fit_panel(df_full)

# -----------------------------------------------------------------------------
# Build comparison table
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
  "\\caption{Near-Capacity vs. Slack-Capacity: Treated-Only vs. Including Never-Treated Controls}",
  "\\label{tab:spare-capacity-comparison}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "\\begin{tabularx}{\\textwidth}{@{} l Y Y @{}}",
  "\\toprule",
  " & \\textbf{Spare capacity} & \\textbf{Occupancy rate} \\\\",
  "\\midrule",
  "\\multicolumn{3}{@{}l}{\\textbf{Panel A: Treated-only (baseline specification)}} \\\\[2pt]",
  paste0("Slack-capacity: \\textit{post} & ", panelA$row_slack, " \\\\"),
  paste0("Difference (Near $-$ Slack) & ", panelA$row_diff, " \\\\"),
  paste0("Near-capacity effect & ", panelA$row_near, " \\\\"),
  paste0("Observations & \\multicolumn{2}{c}{", panelA$n, "} \\\\"),
  "\\addlinespace[6pt]",
  "\\multicolumn{3}{@{}l}{\\textbf{Panel B: Including never-treated facilities as controls}} \\\\[2pt]",
  paste0("Slack-capacity: \\textit{post} & ", panelB$row_slack, " \\\\"),
  paste0("Difference (Near $-$ Slack) & ", panelB$row_diff, " \\\\"),
  paste0("Near-capacity effect & ", panelB$row_near, " \\\\"),
  paste0("Observations & \\multicolumn{2}{c}{", panelB$n, "} \\\\"),
  "\\bottomrule",
  "\\end{tabularx}",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  sprintf(
    "\\item \\textit{Notes:} Combined median spare capacity used for the near/slack split: %.3f. Near-capacity $=$ baseline spare capacity $\\leq$ median; Slack-capacity $=$ above.",
    baseline_median
  ),
  "\\item \\textbf{Panel A} restricts the sample to ever-treated (CHOW) facilities, classified by their own pre-acquisition average spare capacity (event\\_time $\\in [-12,-4]$). This is the specification already sent to advisors.",
  "\\item \\textbf{Panel B} adds never-treated facilities to the estimation sample as controls. Because never-treated facilities have no acquisition event, they cannot be classified on a pre-acquisition window -- they are instead classified using a FIXED CALENDAR baseline (2017 Q1 average spare capacity), the same convention used elsewhere in this project for time-invariant baseline traits (e.g. chain status). \\textit{post} is always 0 for never-treated facilities, so they do not mechanically enter the post or interaction coefficients directly; including them changes what the calendar fixed effects (and therefore \\textit{post}) are estimated relative to.",
  "\\item All specifications include facility and calendar-month fixed effects and the standard control set (excluding occupancy rate/spare capacity from each other's controls). Anticipation window excluded. Standard errors clustered two ways by facility and calendar month.",
  "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "",
  "\\end{document}"
)

writeLines(tex_lines, tex_out_fp, useBytes = TRUE)

cat("\n[write] ", normalizePath(tex_out_fp, winslash = "\\"), "\n", sep = "")
cat("Done.\n")
