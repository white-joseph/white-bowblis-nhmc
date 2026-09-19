# =============================================================================
# regressions/admin_staffing_post.R
#
# Estimates the effect of ownership change on the administrative and other
# nursing staffing categories reported in PBJ but excluded from the paper's
# three core direct-care measures, and on inclusive staffing definitions that
# those categories make constructible.
#
# Console output only; this script writes no files.
#
# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
#   outcome ~ post + beds | facility + calendar month
#
# Spec A as defined in _setup.R, matching the main staffing table in
# post_tables.R. Standard errors are two-way clustered by facility and
# calendar month. The anticipation window (event_time in -3, -2, -1) is
# excluded.
#
# -----------------------------------------------------------------------------
# Outcomes
# -----------------------------------------------------------------------------
# Panel A reports the administrative nursing categories as raw monthly
# hours: RN director of nursing, RN with administrative duties, LPN with
# administrative duties, and, when the panel was built with the non-nurse
# PBJ file, the facility administrator.
#
# Panel B compares the paper's direct-care RN, LPN, and total measures
# against inclusive variants that add the administrative roles sharing the
# same credential. CNA is reported once, since nurse aides in training and
# medication aides are distinct occupations rather than administrative
# versions of the CNA role and are not folded in. Both definitions are
# reported in raw hours and in hours per resident day.
#
# The direct-care measures remain the paper's primary ones. The paper's
# argument concerns labor available to residents, and folding administrative
# roles into the direct-care categories would also conceal the reallocation
# visible in Panel A, where direct-care hours and administrative hours move
# in opposite directions.
#
# Outcomes are reported in levels only. Several of these categories are
# exactly zero for a large share of facility-months, which makes a log
# specification uninformative; Panel C reports the share of non-zero
# observations for each so the level estimates can be read in context.
#
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
#   data/clean/staffing_panel.csv   via load_staffing_panel()
#
# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
#   regressions/_setup.R
#   R packages: dplyr, fixest, tibble
# =============================================================================

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(tibble)
})

options(scipen = 999, digits = 6)

SPEC <- "A"
ALWAYS_EXCLUDE <- "chain_at_start"

vc_month <- ~ cms_certification_number + year_month
fe_month <- "cms_certification_number + year_month"

# -----------------------------------------------------------------------------
# Load and construct
# -----------------------------------------------------------------------------
keep_cols <- c(
  "cms_certification_number", "year_month", "event_time", "post", "treated",
  "beds", "chain_at_start", "resident_days",
  staffing_outcomes, raw_hours_outcomes, admin_hours_outcomes
)

df_full <- load_staffing_panel()
df <- df_full %>% dplyr::select(dplyr::any_of(keep_cols))
rm(df_full); gc(verbose = FALSE)

present_admin <- intersect_existing(admin_hours_outcomes, df)
missing_admin <- setdiff(admin_hours_outcomes, present_admin)

if (length(missing_admin) > 0) {
  message(sprintf(
    "[admin] not present in this panel, skipped: %s",
    paste(missing_admin, collapse = ", ")
  ))
}

# Treat missing as zero ONLY for constructing the aggregate measures below.
# A facility-month with no reported hours in a category is a genuine zero in
# PBJ, not an unobserved value; but the individual-category regressions in
# Panel A use the column as-is, without this substitution.
zero_if_na <- function(x) ifelse(is.na(x), 0, x)

safe_hprd <- function(hours, days) {
  dplyr::if_else(!is.na(days) & days > 0, hours / days, NA_real_)
}

# Inclusive variants. Each adds the administrative categories that share the
# same credential as the direct-care category but perform administrative or
# supervisory work:
#   RN    + RN with administrative duties + RN director of nursing
#   LPN   + LPN with administrative duties
#   Total + both of the above
#
# Nurse aides in training and medication aides are deliberately NOT folded
# into CNA. They are distinct occupations with different scopes of practice
# rather than the same credential in an administrative role, so adding them
# would change what the category measures rather than broaden its
# definition. They remain available in the panel and are reported on their
# own in Panel A.
df <- df %>%
  dplyr::mutate(
    rn_hours_incl = rn_hours_month +
      zero_if_na(rnadmin_hours_month) +
      zero_if_na(rndon_hours_month),
    lpn_hours_incl = lpn_hours_month +
      zero_if_na(lpnadmin_hours_month),
    total_hours_incl = rn_hours_month + lpn_hours_month + cna_hours_month +
      zero_if_na(rnadmin_hours_month) +
      zero_if_na(rndon_hours_month) +
      zero_if_na(lpnadmin_hours_month),
    rn_hprd_incl    = safe_hprd(rn_hours_incl, resident_days),
    lpn_hprd_incl   = safe_hprd(lpn_hours_incl, resident_days),
    total_hprd_incl = safe_hprd(total_hours_incl, resident_days)
  )

df_wo <- drop_anticipation_window(df)
rm(df); gc(verbose = FALSE)

# -----------------------------------------------------------------------------
# Estimation and formatting helpers
# -----------------------------------------------------------------------------
fit_post <- function(dat, lhs) {
  if (!(lhs %in% names(dat))) return(NULL)
  rhs <- make_spec_rhs(dat, spec = SPEC, exclude = union(ALWAYS_EXCLUDE, lhs))
  tryCatch(
    feols(
      as.formula(paste0(lhs, " ~ ", rhs, " | ", fe_month)),
      data = dat, vcov = vc_month, lean = TRUE
    ),
    error = function(e) {
      message(sprintf("[warn] %s failed: %s", lhs, e$message))
      NULL
    }
  )
}

coef_row <- function(mod, term = "post") {
  if (is.null(mod)) return(list(b = NA_real_, se = NA_real_, p = NA_real_, n = NA_integer_))
  ct <- summary(mod)$coeftable
  if (!(term %in% rownames(ct))) {
    return(list(b = NA_real_, se = NA_real_, p = NA_real_, n = nobs(mod)))
  }
  list(
    b  = unname(ct[term, "Estimate"]),
    se = unname(ct[term, "Std. Error"]),
    p  = unname(ct[term, "Pr(>|t|)"]),
    n  = nobs(mod)
  )
}

stars <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.01) return("***")
  if (p < 0.05) return("**")
  if (p < 0.10) return("*")
  ""
}

# Baseline is the pre-transition mean among treated facilities, so the
# coefficient can be read as a proportional change off the relevant base
# rather than off a sample-wide mean that includes never-treated facilities.
baseline_mean <- function(dat, v) {
  if (!(v %in% names(dat))) return(NA_real_)
  x <- dat[[v]][dat$treated == 1 & !is.na(dat$event_time) & dat$event_time < 0]
  if (length(x) == 0) return(NA_real_)
  mean(x, na.rm = TRUE)
}

share_nonzero <- function(dat, v) {
  if (!(v %in% names(dat))) return(NA_real_)
  x <- dat[[v]]
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_real_)
  mean(x > 0)
}

fmt_num <- function(x, digits = 4) {
  if (is.na(x)) return("--")
  formatC(x, format = "f", digits = digits, big.mark = ",")
}

print_header <- function(title) {
  cat("\n")
  cat(strrep("=", 96), "\n", sep = "")
  cat(title, "\n", sep = "")
  cat(strrep("=", 96), "\n", sep = "")
  cat(sprintf(
    "%-34s %14s %14s %6s %12s %10s\n",
    "Outcome", "Coefficient", "Std. error", "", "Obs.", "Pct chg"
  ))
  cat(strrep("-", 96), "\n", sep = "")
}

print_result <- function(label, dat, v, digits = 4) {
  mod <- fit_post(dat, v)
  r <- coef_row(mod)
  base <- baseline_mean(dat, v)
  pct <- if (!is.na(r$b) && !is.na(base) && base != 0) 100 * r$b / base else NA_real_

  cat(sprintf(
    "%-34s %14s %14s %-6s %12s %10s\n",
    label,
    fmt_num(r$b, digits),
    fmt_num(r$se, digits),
    stars(r$p),
    if (is.na(r$n)) "--" else format(r$n, big.mark = ","),
    if (is.na(pct)) "--" else sprintf("%+.2f%%", pct)
  ))

  rm(mod); gc(verbose = FALSE)
  invisible(NULL)
}

# -----------------------------------------------------------------------------
# PANEL A: individual administrative and other nursing categories
# -----------------------------------------------------------------------------
panel_a <- tibble::tribble(
  ~var,                    ~label,
  "rndon_hours_month",     "RN director of nursing",
  "rnadmin_hours_month",   "RN with admin duties",
  "lpnadmin_hours_month",  "LPN with admin duties",
  "admin_hours_month",     "Facility administrator"
) %>% dplyr::filter(var %in% names(df_wo))

print_header("PANEL A: Administrative nursing categories (raw monthly hours)")
for (i in seq_len(nrow(panel_a))) {
  print_result(panel_a$label[i], df_wo, panel_a$var[i])
}

# -----------------------------------------------------------------------------
# PANEL B: direct-care measures vs. inclusive measures
# -----------------------------------------------------------------------------
print_header("PANEL B: Direct-care definition vs. inclusive definition")

cat("-- Raw monthly hours --\n")
print_result("RN hours (direct care)",      df_wo, "rn_hours_month")
print_result("RN hours (inclusive)",        df_wo, "rn_hours_incl")
print_result("LPN hours (direct care)",     df_wo, "lpn_hours_month")
print_result("LPN hours (inclusive)",       df_wo, "lpn_hours_incl")
print_result("CNA hours",                   df_wo, "cna_hours_month")
print_result("Total hours (direct care)",   df_wo, "total_hours")
print_result("Total hours (inclusive)",     df_wo, "total_hours_incl")

cat("\n-- Hours per resident day --\n")
print_result("RN HPRD (direct care)",       df_wo, "rn_hprd")
print_result("RN HPRD (inclusive)",         df_wo, "rn_hprd_incl")
print_result("LPN HPRD (direct care)",      df_wo, "lpn_hprd")
print_result("LPN HPRD (inclusive)",        df_wo, "lpn_hprd_incl")
print_result("CNA HPRD",                    df_wo, "cna_hprd")
print_result("Total HPRD (direct care)",    df_wo, "total_hprd")
print_result("Total HPRD (inclusive)",      df_wo, "total_hprd_incl")

# -----------------------------------------------------------------------------
# PANEL C: coverage and scale
#
# Reported because several of these categories are exactly zero for a large
# share of facility-months. A level coefficient estimated on a variable that
# is mostly zeros is dominated by the facilities that report the category at
# all, which is worth knowing before interpreting Panel A.
# -----------------------------------------------------------------------------
cat("\n")
cat(strrep("=", 96), "\n", sep = "")
cat("PANEL C: Coverage and scale (estimation sample)\n")
cat(strrep("=", 96), "\n", sep = "")
cat(sprintf(
  "%-34s %16s %18s %16s\n",
  "Variable", "Pct non-zero", "Mean (all obs.)", "Pre-period mean"
))
cat(strrep("-", 96), "\n", sep = "")

coverage_vars <- c(
  panel_a$var,
  "rn_hours_month", "rn_hours_incl",
  "lpn_hours_month", "lpn_hours_incl",
  "cna_hours_month",
  "total_hours", "total_hours_incl"
)
coverage_labs <- c(
  panel_a$label,
  "RN hours (direct care)", "RN hours (inclusive)",
  "LPN hours (direct care)", "LPN hours (inclusive)",
  "CNA hours",
  "Total hours (direct care)", "Total hours (inclusive)"
)

for (i in seq_along(coverage_vars)) {
  v <- coverage_vars[i]
  cat(sprintf(
    "%-34s %15s%% %18s %16s\n",
    coverage_labs[i],
    if (is.na(share_nonzero(df_wo, v))) "--" else sprintf("%.1f", 100 * share_nonzero(df_wo, v)),
    fmt_num(mean(df_wo[[v]], na.rm = TRUE), 1),
    fmt_num(baseline_mean(df_wo, v), 1)
  ))
}

# -----------------------------------------------------------------------------
# Share of each inclusive measure contributed by the excluded categories
# -----------------------------------------------------------------------------
cat("\n")
cat(strrep("-", 96), "\n", sep = "")
cat("Share of each inclusive measure contributed by the excluded categories,\n")
cat("evaluated at the pre-transition mean among treated facilities:\n\n")

for (pair in list(
  c("rn_hours_month",  "rn_hours_incl",    "RN"),
  c("lpn_hours_month", "lpn_hours_incl",   "LPN"),
  c("total_hours",     "total_hours_incl", "Total nurse")
)) {
  b_core <- baseline_mean(df_wo, pair[1])
  b_incl <- baseline_mean(df_wo, pair[2])
  if (!is.na(b_core) && !is.na(b_incl) && b_incl > 0) {
    cat(sprintf("  %-14s %5.1f%%\n", pair[3], 100 * (b_incl - b_core) / b_incl))
  }
}

cat("\nNotes: Spec A (post + beds), facility and calendar-month fixed effects,\n")
cat("standard errors two-way clustered by facility and calendar month.\n")
cat("Anticipation window (tau = -3, -2, -1) excluded. Levels only.\n")
cat("Pct chg is the coefficient relative to the pre-transition mean among\n")
cat("treated facilities. Significance: *** p<0.01, ** p<0.05, * p<0.10.\n\n")
