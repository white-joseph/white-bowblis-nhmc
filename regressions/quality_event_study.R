# =============================================================================
# quarterly_quality_event_study_wald.R
#
# Quarterly quality event studies:
# - saves all event-study plots
# - writes standalone LaTeX Wald-test document
# - outcome-specific sample windows
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(fixest)
  library(stringr)
  library(MASS)
})

options(scipen = 999, digits = 4)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
project_root <- "C:/Repositories/white-bowblis-nhmc"
panel_fp     <- file.path(project_root, "data", "clean", "quality_panel.csv")
plots_dir    <- file.path(project_root, "outputs", "plots")
tables_dir   <- file.path(project_root, "tables")

dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)

wald_tex_fp <- file.path(tables_dir, "quarterly_quality_pretrend_wald.tex")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
assert_has_cols <- function(df, cols, df_name = "data") {
  miss <- setdiff(cols, names(df))
  if (length(miss) > 0) {
    stop(
      sprintf("[%s] missing required columns: %s", df_name, paste(miss, collapse = ", ")),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

intersect_existing <- function(x, df) {
  intersect(x, names(df))
}

quarter_num <- function(x) {
  x <- toupper(trimws(as.character(x)))
  suppressWarnings(as.integer(str_extract(x, "[1-4]")))
}

year_quarter_index <- function(year, quarter) {
  qn <- quarter_num(quarter)
  yr <- suppressWarnings(as.integer(year))
  yr * 4L + qn
}

subset_window <- function(df, start_year, start_quarter, end_year, end_quarter) {
  start_idx <- start_year * 4L + start_quarter
  end_idx   <- end_year * 4L + end_quarter
  idx <- year_quarter_index(df$year, df$quarter)
  df[idx >= start_idx & idx <= end_idx, , drop = FALSE]
}

drop_tau_minus1 <- function(df) {
  df %>% filter(is.na(event_time) | event_time != -1)
}

prepare_event_study_data_quarterly <- function(df, min_et, max_et) {
  assert_has_cols(df, c("treated", "event_time"), "event_study_data")
  
  df %>%
    group_by(cms_certification_number) %>%
    mutate(
      ever_treated = as.integer(any(treated == 1, na.rm = TRUE) | any(!is.na(event_time)))
    ) %>%
    ungroup() %>%
    mutate(
      event_time_capped = case_when(
        ever_treated == 1L & !is.na(event_time) ~ pmin(pmax(as.integer(event_time), min_et), max_et),
        TRUE ~ 9999L
      )
    )
}

get_case_mix_controls <- function(df) {
  preferred <- intersect_existing(c("cm_q_state_2", "cm_q_state_3", "cm_q_state_4"), df)
  if (length(preferred) > 0) return(preferred)
  
  fallback <- intersect_existing(c("cm_q_nat_2", "cm_q_nat_3", "cm_q_nat_4"), df)
  fallback
}

get_controls <- function(df) {
  base_controls <- c(
    "government",
    "non_profit",
    "chain",
    "beds",
    "occupancy_rate",
    "pct_medicare",
    "pct_medicaid"
  )
  c(intersect_existing(base_controls, df), get_case_mix_controls(df))
}

make_controls_rhs <- function(df) {
  ctrls <- get_controls(df)
  if (length(ctrls) == 0) return("1")
  paste(ctrls, collapse = " + ")
}

pick_ref <- function(dat, desired = NULL) {
  ev <- sort(unique(dat$event_time_capped[dat$ever_treated == 1L]))
  ev <- ev[is.finite(ev) & ev != 9999L]
  if (!length(ev)) stop("No treated event times found.")
  if (!is.null(desired) && desired %in% ev) return(as.integer(desired))
  if (-1L %in% ev) return(-1L)
  negs <- ev[ev < 0L]
  if (length(negs)) return(max(negs))
  return(ev[1])
}

run_es_twfe <- function(lhs, data, controls_rhs, ref_val, window) {
  fml <- as.formula(paste0(
    lhs, " ~ i(event_time_capped, ever_treated, ref = ", ref_val,
    ", keep = ", window[1], ":", window[2], ") + ",
    controls_rhs,
    " | cms_certification_number + year_quarter"
  ))
  
  feols(
    fml = fml,
    data = data,
    vcov = ~ cms_certification_number + year_quarter,
    lean = TRUE
  )
}

pretrend_wald <- function(mod, ref_tau, from, to) {
  if (is.null(mod)) return(list(note = "Model is NULL"))
  
  cn <- names(coef(mod))
  if (is.null(cn) || !length(cn)) return(list(note = "No coefficients found"))
  
  pat <- "^event_time_capped::[-]?[0-9]+:ever_treated$"
  es_names <- grep(pat, cn, value = TRUE)
  if (!length(es_names)) return(list(note = "No event-study coefficients found"))
  
  get_tau <- function(s) as.integer(regmatches(s, regexpr("-?[0-9]+", s)))
  taus <- vapply(es_names, get_tau, integer(1))
  
  keep <- taus >= from & taus <= to & taus != ref_tau
  pre_names <- es_names[keep]
  pre_taus  <- taus[keep]
  
  if (!length(pre_names)) return(list(note = "No preperiod coefficients available"))
  
  b <- coef(mod)[pre_names]
  V <- vcov(mod)[pre_names, pre_names, drop = FALSE]
  Vinv <- tryCatch(solve(V), error = function(e) MASS::ginv(V))
  W <- as.numeric(t(b) %*% Vinv %*% b)
  df_w <- length(b)
  pval <- pchisq(W, df = df_w, lower.tail = FALSE)
  
  list(
    statistic = W,
    df = df_w,
    p.value = pval,
    taus = pre_taus
  )
}

# FIXED
fmt_wald_cell <- function(x) {
  if (is.null(x) || !is.null(x$note)) {
    return("\\makecell[c]{NA}")
  }
  
  out <- sprintf(
    "\\makecell[c]{%.2f (%d) \\\\ {[%.4f]}}",
    as.numeric(x$statistic),
    as.integer(x$df),
    as.numeric(x$p.value)
  )
  
  return(as.character(out))
}

escape_latex <- function(x) {
  x <- gsub("\\\\", "\\\\textbackslash{}", x)
  x <- gsub("([#$%&_{}])", "\\\\\\1", x, perl = TRUE)
  x <- gsub("~", "\\\\textasciitilde{}", x, fixed = TRUE)
  x <- gsub("\\^", "\\\\textasciicircum{}", x)
  x
}

# -----------------------------------------------------------------------------
# Plot helper
# -----------------------------------------------------------------------------
save_es_plot <- function(model, ref_val, file_stub, ylab_txt,
                         xlab_txt = "Quarters relative to treatment",
                         xlim_window = c(-8, 8),
                         out_dir = plots_dir) {
  if (is.null(model)) return(invisible(NULL))
  
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_fp <- file.path(out_dir, paste0(file_stub, ".pdf"))
  
  grDevices::cairo_pdf(
    filename = out_fp,
    width = 9.5,
    height = 6.2
  )
  on.exit(dev.off(), add = TRUE)
  
  par(family = "Times New Roman")
  
  iplot(
    model,
    ref  = ref_val,
    xlim = xlim_window,
    xlab = xlab_txt,
    ylab = ylab_txt,
    main = "",
    sub  = ""
  )
}

# -----------------------------------------------------------------------------
# Load panel
# -----------------------------------------------------------------------------
df0 <- readr::read_csv(panel_fp, show_col_types = FALSE)

required_cols <- c(
  "cms_certification_number",
  "year",
  "quarter",
  "treated",
  "event_time"
)
assert_has_cols(df0, required_cols, "quality_panel")

df0 <- df0 %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year = suppressWarnings(as.integer(year)),
    quarter = toupper(trimws(as.character(quarter))),
    year_quarter = paste0(year, "_", quarter)
  )

numeric_candidates <- c(
  "beds", "occupancy_rate", "pct_medicare", "pct_medicaid",
  "event_time", "government", "non_profit", "chain"
)
numeric_candidates <- intersect_existing(numeric_candidates, df0)

if (length(numeric_candidates) > 0) {
  df0 <- df0 %>%
    mutate(across(all_of(numeric_candidates), ~ suppressWarnings(as.numeric(.x))))
}

controls_rhs <- make_controls_rhs(df0)

# -----------------------------------------------------------------------------
# Outcome map with sample windows
# -----------------------------------------------------------------------------
outcome_windows <- list(
  qm_401 = c(2017L, 1L, 2024L, 2L),
  qm_404 = c(2017L, 1L, 2024L, 2L),
  qm_406 = c(2017L, 1L, 2024L, 2L),
  qm_407 = c(2017L, 1L, 2024L, 2L),
  qm_410 = c(2017L, 1L, 2024L, 2L),
  qm_419 = c(2017L, 1L, 2024L, 2L),
  qm_434 = c(2017L, 1L, 2024L, 2L),
  qm_452 = c(2017L, 1L, 2024L, 2L),
  
  qm_405 = c(2017L, 1L, 2023L, 3L),
  qm_451 = c(2017L, 1L, 2023L, 3L),
  qm_471 = c(2017L, 1L, 2023L, 3L),
  
  qm_453 = c(2018L, 1L, 2023L, 3L)
)

all_outcomes <- names(outcome_windows)
missing_outcomes <- setdiff(all_outcomes, names(df0))
if (length(missing_outcomes) > 0) {
  stop(sprintf("Missing requested outcomes: %s",
               paste(missing_outcomes, collapse = ", ")),
       call. = FALSE)
}

nice_out <- setNames(gsub("^qm_", "QM ", all_outcomes), all_outcomes)

# -----------------------------------------------------------------------------
# Specs
# -----------------------------------------------------------------------------
specs <- list(
  with_anticip = list(
    row_label = "2 Year Full Pre-Window",
    window = c(-8L, 8L),
    donut = FALSE,
    ref_desired = -1L,
    test_from = -8L,
    test_to = -1L,
    plot_tag = "full_pre_8q"
  ),
  wo_anticip_8 = list(
    row_label = "2 Year Window with Donut",
    window = c(-8L, 8L),
    donut = TRUE,
    ref_desired = -2L,
    test_from = -8L,
    test_to = -2L,
    plot_tag = "donut_pre_8q"
  ),
  wo_anticip_4 = list(
    row_label = "1 Year Window with Donut",
    window = c(-4L, 4L),
    donut = TRUE,
    ref_desired = -2L,
    test_from = -4L,
    test_to = -2L,
    plot_tag = "donut_pre_4q"
  )
)

# -----------------------------------------------------------------------------
# Run models and Wald tests
# -----------------------------------------------------------------------------
results <- list()
sample_sizes <- list()

for (outcome in all_outcomes) {
  cat("\n", strrep("=", 80), "\n", sep = "")
  cat("OUTCOME: ", outcome, "\n", sep = "")
  cat(strrep("=", 80), "\n", sep = "")
  
  win <- outcome_windows[[outcome]]
  dat_base <- subset_window(df0, win[1], win[2], win[3], win[4])
  
  results[[outcome]] <- list()
  sample_sizes[[outcome]] <- list()
  
  for (sp_nm in names(specs)) {
    sp <- specs[[sp_nm]]
    
    dat_sp <- dat_base
    if (isTRUE(sp$donut)) {
      dat_sp <- drop_tau_minus1(dat_sp)
    }
    
    dat_sp <- prepare_event_study_data_quarterly(
      dat_sp,
      min_et = sp$window[1],
      max_et = sp$window[2]
    ) %>%
      filter(!is.na(.data[[outcome]]))
    
    ref_val <- pick_ref(dat_sp, desired = sp$ref_desired)
    mod <- run_es_twfe(
      lhs = outcome,
      data = dat_sp,
      controls_rhs = controls_rhs,
      ref_val = ref_val,
      window = sp$window
    )
    
    wald <- pretrend_wald(
      mod,
      ref_tau = ref_val,
      from = sp$test_from,
      to = sp$test_to
    )
    
    results[[outcome]][[sp_nm]] <- list(
      model = mod,
      ref = ref_val,
      wald = wald
    )
    
    sample_sizes[[outcome]][[sp_nm]] <- nrow(dat_sp)
    
    cat("[spec] ", sp_nm,
        " | ref=", ref_val,
        " | N=", format(nrow(dat_sp), big.mark = ","),
        "\n", sep = "")
    
    plot_stub <- paste0("quality_es_", outcome, "_", sp$plot_tag)
    save_es_plot(
      model = mod,
      ref_val = ref_val,
      file_stub = plot_stub,
      ylab_txt = nice_out[[outcome]],
      xlim_window = sp$window,
      out_dir = plots_dir
    )
  }
}

# -----------------------------------------------------------------------------
# Group outcomes into LaTeX tables by valid sample window
# -----------------------------------------------------------------------------
group_full <- c("qm_401", "qm_404", "qm_406", "qm_407", "qm_410", "qm_419", "qm_434", "qm_452")
group_2017_2023q3 <- c("qm_405", "qm_451", "qm_471")
group_2018_2023q3 <- c("qm_453")

mk_row <- function(rowlabel, outcomes, spec_name) {
  cells <- vapply(
    outcomes,
    function(y) fmt_wald_cell(results[[y]][[spec_name]]$wald),
    FUN.VALUE = character(1)
  )
  paste0(rowlabel, " & ", paste(cells, collapse = " & "), " \\\\")
}

build_table_block <- function(outcomes, caption, label, notes_extra = NULL, landscape = FALSE) {
  n_out <- length(outcomes)
  colspec <- paste0("@{}l", paste(rep("c", n_out), collapse = ""), "@{}")
  header_line <- paste(vapply(outcomes, function(x) sprintf("\\textbf{%s}", nice_out[[x]]), character(1)),
                       collapse = " & ")
  
  Ns1 <- paste(vapply(outcomes, function(y) format(sample_sizes[[y]][["with_anticip"]], big.mark = ","), character(1)),
               collapse = ", ")
  Ns2 <- paste(vapply(outcomes, function(y) format(sample_sizes[[y]][["wo_anticip_8"]], big.mark = ","), character(1)),
               collapse = ", ")
  Ns3 <- paste(vapply(outcomes, function(y) format(sample_sizes[[y]][["wo_anticip_4"]], big.mark = ","), character(1)),
               collapse = ", ")
  
  open_env  <- if (landscape) "\\begin{landscape}" else NULL
  close_env <- if (landscape) "\\end{landscape}" else NULL
  
  c(
    "\\begingroup",
    open_env,
    "\\begin{table}[!ht]",
    "\\centering",
    "\\begin{threeparttable}",
    sprintf("\\caption{%s}", caption),
    sprintf("\\label{%s}", label),
    "\\small",
    "\\setlength{\\tabcolsep}{6pt}",
    "",
    sprintf("\\begin{tabular}{%s}", colspec),
    "\\toprule",
    paste0(" & \\multicolumn{", n_out, "}{c}{\\textbf{Outcomes}} \\\\"),
    sprintf("\\cmidrule(lr){2-%d}", n_out + 1),
    paste0(" & ", header_line, " \\\\"),
    "\\midrule",
    mk_row(specs$with_anticip$row_label, outcomes, "with_anticip"),
    mk_row(specs$wo_anticip_8$row_label, outcomes, "wo_anticip_8"),
    mk_row(specs$wo_anticip_4$row_label, outcomes, "wo_anticip_4"),
    "\\bottomrule",
    "\\end{tabular}",
    "",
    "\\begin{tablenotes}[flushleft]",
    "\\footnotesize",
    "\\item \\textit{Notes:} Each cell reports the Wald $\\chi^2$ statistic for the joint null that all pre-treatment event-time coefficients equal zero, followed by degrees of freedom in parentheses and the p-value in brackets.",
    "\\item The 2 Year Full Pre-Window tests $\\tau=-8$ to $\\tau=-1$ with reference $\\tau=-1$.",
    "\\item The 2 Year Window with Donut tests $\\tau=-8$ to $\\tau=-2$ with reference $\\tau=-2$, dropping $\\tau=-1$.",
    "\\item The 1 Year Window with Donut tests $\\tau=-4$ to $\\tau=-2$ with reference $\\tau=-2$, dropping $\\tau=-1$.",
    sprintf("\\item Sample sizes by outcome for the three rows are: Row 1 [%s]; Row 2 [%s]; Row 3 [%s].", Ns1, Ns2, Ns3),
    sprintf("\\item All specifications include facility and quarter fixed effects and covariates: %s.", escape_latex(controls_rhs)),
    if (!is.null(notes_extra)) paste0("\\item ", notes_extra) else NULL,
    "\\end{tablenotes}",
    "\\end{threeparttable}",
    "\\end{table}",
    close_env,
    "\\endgroup",
    ""
  )
}

tab1 <- build_table_block(
  outcomes = group_full,
  caption = "Joint Wald Tests of Pre-trends for Quarterly Quality Measures (2017~Q1--2024~Q2 Outcomes)",
  label = "tab:q_pretrend_wald_full",
  notes_extra = "Outcomes in this table are available over 2017~Q1--2024~Q2.",
  landscape = TRUE
)

tab2 <- build_table_block(
  outcomes = group_2017_2023q3,
  caption = "Joint Wald Tests of Pre-trends for Quarterly Quality Measures (2017~Q1--2023~Q3 Outcomes)",
  label = "tab:q_pretrend_wald_2017_2023q3",
  notes_extra = "Outcomes in this table are available over 2017~Q1--2023~Q3.",
  landscape = FALSE
)

tab3 <- build_table_block(
  outcomes = group_2018_2023q3,
  caption = "Joint Wald Tests of Pre-trends for Quarterly Quality Measure 453",
  label = "tab:q_pretrend_wald_453",
  notes_extra = "QM 453 is available over 2018~Q1--2023~Q3.",
  landscape = FALSE
)

# -----------------------------------------------------------------------------
# Write standalone LaTeX document
# -----------------------------------------------------------------------------
latex_doc <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{threeparttable}",
  "\\usepackage{array}",
  "\\usepackage{caption}",
  "\\usepackage{makecell}",
  "\\usepackage{pdflscape}",
  "\\usepackage{newtxtext}",
  "\\usepackage{newtxmath}",
  "\\captionsetup{labelfont=bf, font=small}",
  "\\begin{document}",
  tab1,
  "\\clearpage",
  tab2,
  "\\clearpage",
  tab3,
  "\\end{document}",
  ""
)

writeLines(latex_doc, wald_tex_fp, useBytes = TRUE)

cat("\n[write] ", normalizePath(wald_tex_fp, winslash = "\\"), "\n", sep = "")
cat("[plots] saved to ", normalizePath(plots_dir, winslash = "\\"), "\n", sep = "")
cat("Done.\n")