# =============================================================================
# regressions/post_staffing.R
#
# Purpose:
#   Two-way fixed effects estimates of `post` on staffing outcomes, reported
#   as a single 4x4 table:
#
#                      RN        LPN       CNA       Total
#     HPRD             .         .         .         .
#     Log(HPRD)        .         .         .         .
#     Raw hours        .         .         .         .
#     Log(raw hours)   .         .         .         .
#
#   Columns are staff type; rows are the outcome transformation. The HPRD and
#   Log(HPRD) rows reproduce the baseline table in twfe_post.R
#   (outputs/tables/twfe_post_full.tex). The two raw-hours rows are new: raw
#   hours are the HPRD numerator only, so they test whether the HPRD decline
#   is mechanically driven by the occupancy-rate increase in the denominator.
#
# Specification (identical across all 16 cells):
#   outcome ~ post + controls | cms_certification_number + year_month
#   - two-way clustered SEs (facility and month)
#   - anticipation window (event_time in -3, -2, -1) dropped
#   - `government`-ever facilities dropped upstream by load_staffing_panel()
#
# Outputs:
#   outputs/tables/post_staffing.tex  (fragment for \input)
#
# Paths:
#   Paths are resolved with the `here` package, which anchors on the project
#   root (identified by white-bowblis-nhmc.Rproj / .git). The script therefore
#   runs unchanged on any machine and from any working directory. here::i_am()
#   asserts that the detected root is the right one and fails with a clear
#   message if this file is run from an unrelated project.
#
#   The script also overrides the hardcoded paths that _setup.R still sets, so
#   sourcing _setup.R does not reintroduce a machine-specific path. Once
#   _setup.R is itself made portable, those overrides become redundant but
#   remain harmless.
# =============================================================================

# -----------------------------------------------------------------------------
# Portable project-root resolution
# -----------------------------------------------------------------------------
library(here)

here::i_am("regressions/post_staffing.R")

PROJECT_ROOT <- here::here()

source(here::here("regressions", "_setup.R"))

# Override the machine-specific paths _setup.R hardcodes. load_staffing_panel()
# takes `fp = panel_fp` as a lazily-evaluated default resolved from the global
# environment, so reassigning panel_fp here is what the loader actually uses.
project_root   <- PROJECT_ROOT
panel_fp       <- here::here("data", "clean", "staffing_panel.csv")
out_tables_dir <- here::here("outputs", "tables")
out_plots_dir  <- here::here("outputs", "plots")

dir.create(out_tables_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(tibble)
})

options(scipen = 999)

message(sprintf("[post_staffing] project root = %s", PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Load panel and build the estimation sample
# -----------------------------------------------------------------------------
df <- load_staffing_panel()

# Matches twfe_post.R: control set is resolved against the full panel, and the
# anticipation window is dropped from the estimation sample.
controls_rhs <- make_controls_rhs(df)
rhs <- if (controls_rhs == "1") "post" else paste("post +", controls_rhs)
vc  <- ~ cms_certification_number + year_month

df_wo <- drop_anticipation_window(df)

# -----------------------------------------------------------------------------
# Outcome grid: columns = staff type, rows = transformation
# -----------------------------------------------------------------------------
staff_grid <- tibble::tribble(
  ~staff,  ~hprd,        ~ln_hprd,   ~hours,            ~ln_hours,
  "RN",    "rn_hprd",    "ln_rn",    "rn_hours_month",  "ln_rn_hours",
  "LPN",   "lpn_hprd",   "ln_lpn",   "lpn_hours_month", "ln_lpn_hours",
  "CNA",   "cna_hprd",   "ln_cna",   "cna_hours_month", "ln_cna_hours",
  "Total", "total_hprd", "ln_total", "total_hours",     "ln_total_hours"
)

# All rows report 3 decimal places. big_mark is enabled only on the raw-hours
# row, where estimates are measured in hours per facility-month and can reach
# four figures; the HPRD and log rows never do.
row_specs <- list(
  list(key = "hprd",     label = "HPRD",           digits = 3, big_mark = FALSE),
  list(key = "ln_hprd",  label = "Log(HPRD)",      digits = 3, big_mark = FALSE),
  list(key = "hours",    label = "Raw hours",      digits = 3, big_mark = TRUE),
  list(key = "ln_hours", label = "Log(raw hours)", digits = 3, big_mark = FALSE)
)

# Fail loudly and early if the panel predates the raw-hours columns.
assert_has_cols(
  df_wo,
  c(staff_grid$hprd, staff_grid$ln_hprd, staff_grid$hours, staff_grid$ln_hours),
  "staffing_panel (post_staffing)"
)

# -----------------------------------------------------------------------------
# Estimation
# -----------------------------------------------------------------------------
make_fml <- function(lhs) {
  as.formula(
    sprintf("%s ~ %s | cms_certification_number + year_month", lhs, rhs)
  )
}

# Fit, extract, discard. Holding all 16 fitted objects is unnecessary and
# expensive on a ~550k-row panel, so only the scalars we report are retained.
fit_and_extract <- function(lhs, term = "post") {
  empty <- list(
    coef = NA_real_, se = NA_real_, p = NA_real_,
    stars = "", n = NA_integer_
  )

  if (!(lhs %in% names(df_wo)) || all(is.na(df_wo[[lhs]]))) {
    message(sprintf("[post_staffing] skipped %s (absent or all-missing)", lhs))
    return(empty)
  }

  mod <- feols(make_fml(lhs), data = df_wo, vcov = vc, lean = FALSE)
  ct <- summary(mod)$coeftable

  if (!(term %in% rownames(ct))) {
    message(sprintf("[post_staffing] term '%s' absent for %s", term, lhs))
    return(empty)
  }

  b  <- unname(ct[term, "Estimate"])
  se <- unname(ct[term, "Std. Error"])
  p  <- unname(ct[term, "Pr(>|t|)"])
  n  <- nobs(mod)

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

  rm(mod)

  list(coef = b, se = se, p = p, stars = stars, n = n)
}

results <- list()
for (spec in row_specs) {
  for (i in seq_len(nrow(staff_grid))) {
    staff <- staff_grid$staff[i]
    lhs   <- staff_grid[[spec$key]][i]

    message(sprintf("[post_staffing] fitting %-14s %-6s (%s)", spec$key, staff, lhs))

    res <- fit_and_extract(lhs)
    res$staff <- staff
    res$row_key <- spec$key
    res$row_label <- spec$label
    res$outcome <- lhs

    results[[paste(spec$key, staff, sep = "|")]] <- res
  }
}

# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------
fmt_est <- function(res, digits = 3, big_mark = FALSE) {
  if (is.na(res$coef) || is.na(res$se)) {
    return("\\est{$\\,$}{$\\,$}{}")
  }

  num <- function(x) {
    if (big_mark) {
      formatC(x, format = "f", digits = digits, big.mark = ",")
    } else {
      formatC(x, format = "f", digits = digits)
    }
  }

  bstr <- num(res$coef)
  # Keep positive and negative estimates vertically aligned in the column.
  if (res$coef > 0) bstr <- paste0("\\phantom{-}", bstr)
  sestr <- num(res$se)

  sprintf("\\est{$%s$}{$%s$}{%s}", bstr, sestr, res$stars)
}

build_row <- function(spec) {
  cells <- vapply(
    staff_grid$staff,
    function(s) {
      fmt_est(
        results[[paste(spec$key, s, sep = "|")]],
        digits = spec$digits,
        big_mark = spec$big_mark
      )
    },
    character(1)
  )
  paste(cells, collapse = "  &  ")
}

# Per-row observation counts: one number if the four staff types agree,
# otherwise the observed range.
#
# Returned in TEXT mode, not math mode. Inside $...$ LaTeX inserts a thin space
# after each comma ("550, 330") and typesets "--" as two minus signs rather
# than an en-dash, so the counts must stay outside the math delimiters.
fmt_row_n <- function(spec) {
  ns <- vapply(
    staff_grid$staff,
    function(s) as.numeric(results[[paste(spec$key, s, sep = "|")]]$n),
    numeric(1)
  )
  ns <- ns[!is.na(ns)]

  if (length(ns) == 0) return("--")

  if (length(unique(ns)) == 1) {
    format(ns[1], big.mark = ",")
  } else {
    sprintf(
      "%s--%s",
      format(min(ns), big.mark = ","),
      format(max(ns), big.mark = ",")
    )
  }
}

# -----------------------------------------------------------------------------
# Table fragment
# -----------------------------------------------------------------------------
n_note <- paste(
  vapply(
    row_specs,
    function(spec) sprintf("%s: $N$ = %s", spec$label, fmt_row_n(spec)),
    character(1)
  ),
  collapse = "; "
)

tab <- c(
  "\\begingroup",
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Two-Way Fixed Effects Estimates of \\textit{post} on Staffing Outcomes (HPRD and Raw Hours)}",
  "\\label{tab:post-staffing}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "",
  "\\begin{tabularx}{\\textwidth}{@{} l YYYY @{} }",
  "\\toprule",
  " & \\multicolumn{4}{c}{\\textbf{Staff Type}} \\\\",
  "\\cmidrule(lr){2-5}",
  " & \\textbf{RN} & \\textbf{LPN} & \\textbf{CNA} & \\textbf{Total} \\\\",
  "\\midrule",
  paste0(row_specs[[1]]$label, " & ", build_row(row_specs[[1]]), " \\\\"),
  "\\addlinespace[3pt]",
  paste0(row_specs[[2]]$label, " & ", build_row(row_specs[[2]]), " \\\\"),
  "\\addlinespace[6pt]",
  paste0(row_specs[[3]]$label, " & ", build_row(row_specs[[3]]), " \\\\"),
  "\\addlinespace[3pt]",
  paste0(row_specs[[4]]$label, " & ", build_row(row_specs[[4]]), " \\\\"),
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post} with two-way clustered standard errors (by facility and month) in parentheses. Columns are staff type; rows are the outcome transformation. HPRD is measured in hours per resident-day. Raw hours are total reported hours per facility-month, i.e. the HPRD numerator only, and are included to test whether the HPRD results are driven mechanically by changes in the resident-day denominator.",
  sprintf("\\item Sample excludes the anticipation window ($\\tau \\in \\{-3,-2,-1\\}$) and any facility ever government-owned. Observations by row --- %s. Log specifications drop facility-months with non-positive values.", n_note),
  "\\item All specifications include facility and month fixed effects and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators.",
  "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}",
  "\\endgroup",
  ""
)

tab_path <- file.path(out_tables_dir, "post_staffing.tex")
writeLines(tab, tab_path, useBytes = TRUE)

# -----------------------------------------------------------------------------
# Console summary (not written to disk)
# -----------------------------------------------------------------------------
coef_tbl <- bind_rows(lapply(results, function(r) {
  tibble::tibble(
    row_key    = r$row_key,
    row_label  = r$row_label,
    staff      = r$staff,
    outcome    = r$outcome,
    coef       = r$coef,
    se         = r$se,
    p_value    = r$p,
    stars      = r$stars,
    n_obs      = r$n
  )
})) %>%
  mutate(
    row_label = factor(row_label, levels = vapply(row_specs, `[[`, character(1), "label")),
    staff     = factor(staff, levels = staff_grid$staff)
  ) %>%
  arrange(row_label, staff)

cat("\n")
print(as.data.frame(coef_tbl %>% select(row_label, staff, coef, se, p_value, n_obs)))

cat("\nSaved:\n")
cat(" -", tab_path, "\n")
cat("\nDone.\n")
