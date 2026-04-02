# C:/Repositories/white-bowblis-nhmc/regressions/twfe_event_study_date_mcr_preview.R
# Preview TWFE event-study plots using MCR event timing
# - Full sample WITH anticipation
# - Full sample WITHOUT anticipation
# - Shows plots only (does not save)
# - Adds titles
# - Prints joint Wald pretrend tests (no LaTeX table output)

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
  library(MASS)   # ginv()
})

options(scipen = 999, digits = 4)

# ------------------------------ Plot font ------------------------------
set_plot_font <- function() {
  fam <- "Times New Roman"
  par(family = fam)
}
set_plot_font()

# ------------------------------ Load ------------------------------
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel_date_mcr.csv"

keep_cols <- c(
  "cms_certification_number","year_month","anticipation2",
  "event_time","treatment",
  "time","time_treated",
  "government","non_profit","chain","beds",
  "occupancy_rate","pct_medicare","pct_medicaid",
  "cm_q_state_2","cm_q_state_3","cm_q_state_4",
  "rn_hppd","lpn_hppd","cna_hppd","total_hppd"
)

df <- read_csv(panel_fp, show_col_types = FALSE, col_select = all_of(keep_cols)) %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month_chr = as.character(year_month),
    year_month = as.factor(year_month_chr),
    ym_date = as.Date(paste0(gsub("/", "-", year_month_chr), "-01"))
  )

# ------------------------------ Treated window + logs ------------------------------
df <- df %>%
  group_by(cms_certification_number) %>%
  mutate(
    ever_treated = as.integer(any(treatment == 1, na.rm = TRUE) | any(!is.na(event_time)))
  ) %>%
  ungroup() %>%
  mutate(
    event_time_capped = case_when(
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

# ------------------------------ Controls ------------------------------
controls_rhs <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)

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

run_es_twfe <- function(lhs, data, ref_val, window = c(-24L, 24L)) {
  fml <- as.formula(paste0(
    lhs, " ~ i(event_time_capped, ever_treated, ref = ", ref_val,
    ", keep = ", window[1], ":", window[2], ") + ",
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

# ------------------------------ Pretrend helpers ------------------------------
.es_pick <- function(mod, var = "event_time_capped", trt = "ever_treated") {
  cn <- names(coef(mod))
  if (is.null(cn) || !length(cn)) return(list(names = character(0), taus = integer(0)))
  pat <- sprintf("^%s::-?\\d+:%s$", var, trt)
  es_names <- grep(pat, cn, value = TRUE)
  get_tau <- function(s) as.integer(regmatches(s, regexpr("-?\\d+", s)))
  taus <- vapply(es_names, get_tau, integer(1))
  names(taus) <- es_names
  list(names = es_names, taus = taus)
}

pretrend_wald <- function(mod, ref_tau, from = -Inf, to = -2,
                          var = "event_time_capped", trt = "ever_treated") {
  es <- .es_pick(mod, var, trt)
  pre_idx <- es$taus < 0L & es$taus != ref_tau & es$taus >= from & es$taus <= to
  pre_names <- names(es$taus)[pre_idx]
  if (!length(pre_names)) {
    return(list(
      statistic = NA_real_,
      df = NA_integer_,
      p.value = NA_real_,
      tested_taus = integer(0),
      n_constraints = 0L,
      note = "No preperiod coefficients in window"
    ))
  }
  
  b <- coef(mod)[pre_names]
  V <- vcov(mod)[pre_names, pre_names, drop = FALSE]
  W <- as.numeric(t(b) %*% MASS::ginv(V) %*% b)
  df_w <- qr(V)$rank
  pval <- pchisq(W, df = df_w, lower.tail = FALSE)
  
  list(
    statistic = W,
    df = df_w,
    p.value = pval,
    tested_taus = sort(unique(es$taus[pre_idx])),
    n_constraints = length(pre_names),
    note = NULL
  )
}

print_pretrend <- function(title, res) {
  cat("\n================ ", title, " ================\n", sep = "")
  if (!is.null(res$note)) {
    cat("[info] ", res$note, "\n", sep = "")
    return(invisible(NULL))
  }
  cat(sprintf("Joint Wald: W = %.3f on %d df  =>  p = %.4g\n",
              res$statistic, res$df, res$p.value))
  cat("Tested pre τ: ", paste(res$tested_taus, collapse = ", "), "\n", sep = "")
}

wald_df_from_list <- function(res_list, spec_label) {
  data.frame(
    specification = spec_label,
    outcome = names(res_list),
    wald_stat = sapply(res_list, function(x) x$statistic),
    df = sapply(res_list, function(x) x$df),
    p_value = sapply(res_list, function(x) x$p.value),
    tested_taus = sapply(res_list, function(x) {
      if (length(x$tested_taus) == 0) return(NA_character_)
      paste(x$tested_taus, collapse = ", ")
    }),
    n_constraints = sapply(res_list, function(x) x$n_constraints),
    row.names = NULL
  )
}

# ------------------------------ Outcomes ------------------------------
outs_lvl <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")
outs_log <- c("ln_rn","ln_lpn","ln_cna","ln_total")

# ------------------------------ Fitting block ------------------------------
fit_block <- function(tag, data, desired_ref = -1L, event_window = c(-24L, 24L)) {
  cat("\n\n", strrep("=", 84), "\nBLOCK: ", tag, "\n", strrep("=", 84), "\n", sep = "")
  ref <- pick_ref(data, desired = desired_ref)
  cat("Reference used: t = ", ref, "\n", sep = "")
  
  mods_lvl <- lapply(outs_lvl, function(y) run_es_twfe(y, data, ref_val = ref, window = event_window))
  names(mods_lvl) <- outs_lvl
  
  mods_log <- lapply(outs_log, function(y) run_es_twfe(y, data, ref_val = ref, window = event_window))
  names(mods_log) <- outs_log
  
  invisible(list(levels = mods_lvl, logs = mods_log, ref = ref, tag = tag))
}

# ------------------------------ Plot helpers ------------------------------
plot_block_levels <- function(mod_obj, title_prefix, event_window = c(-24L, 24L)) {
  ref <- mod_obj$ref
  
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  
  par(mfrow = c(2, 2))
  set_plot_font()
  
  iplot(mod_obj$levels[["rn_hppd"]],
        ref = ref, xlim = event_window,
        xlab = "Months relative to treatment", ylab = "RN HPRD",
        main = paste0(title_prefix, ": RN"), sub = "")
  
  iplot(mod_obj$levels[["lpn_hppd"]],
        ref = ref, xlim = event_window,
        xlab = "Months relative to treatment", ylab = "LPN HPRD",
        main = paste0(title_prefix, ": LPN"), sub = "")
  
  iplot(mod_obj$levels[["cna_hppd"]],
        ref = ref, xlim = event_window,
        xlab = "Months relative to treatment", ylab = "CNA HPRD",
        main = paste0(title_prefix, ": CNA"), sub = "")
  
  iplot(mod_obj$levels[["total_hppd"]],
        ref = ref, xlim = event_window,
        xlab = "Months relative to treatment", ylab = "Total HPRD",
        main = paste0(title_prefix, ": Total"), sub = "")
}

plot_block_logs <- function(mod_obj, title_prefix, event_window = c(-24L, 24L)) {
  ref <- mod_obj$ref
  
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  
  par(mfrow = c(2, 2))
  set_plot_font()
  
  iplot(mod_obj$logs[["ln_rn"]],
        ref = ref, xlim = event_window,
        xlab = "Months relative to treatment", ylab = "Log(RN HPRD)",
        main = paste0(title_prefix, ": Log RN"), sub = "")
  
  iplot(mod_obj$logs[["ln_lpn"]],
        ref = ref, xlim = event_window,
        xlab = "Months relative to treatment", ylab = "Log(LPN HPRD)",
        main = paste0(title_prefix, ": Log LPN"), sub = "")
  
  iplot(mod_obj$logs[["ln_cna"]],
        ref = ref, xlim = event_window,
        xlab = "Months relative to treatment", ylab = "Log(CNA HPRD)",
        main = paste0(title_prefix, ": Log CNA"), sub = "")
  
  iplot(mod_obj$logs[["ln_total"]],
        ref = ref, xlim = event_window,
        xlab = "Months relative to treatment", ylab = "Log(Total HPRD)",
        main = paste0(title_prefix, ": Log Total"), sub = "")
}

# ------------------------------ Samples ------------------------------
S_full  <- df
S_noant <- df %>% filter(anticipation2 == 0)

# ------------------------------ Run models ------------------------------
mods_full <- fit_block(
  tag = "MCR timing — WITH anticipation",
  data = S_full,
  desired_ref = -1L,
  event_window = c(-24L, 24L)
)

mods_noant <- fit_block(
  tag = "MCR timing — WITHOUT anticipation",
  data = S_noant,
  desired_ref = -4L,
  event_window = c(-24L, 24L)
)

# ------------------------------ Wald tests ------------------------------
# With anticipation: 2-year full prewindow, ref = -1, test tau = -24,...,-2
wald_full_levels <- lapply(
  mods_full$levels,
  function(m) pretrend_wald(m, ref_tau = mods_full$ref, from = -24L, to = -2L)
)
wald_full_logs <- lapply(
  mods_full$logs,
  function(m) pretrend_wald(m, ref_tau = mods_full$ref, from = -24L, to = -2L)
)

# Without anticipation: 2-year donut prewindow, ref = -4, test tau = -24,...,-5
wald_noant_levels <- lapply(
  mods_noant$levels,
  function(m) pretrend_wald(m, ref_tau = mods_noant$ref, from = -24L, to = -5L)
)
wald_noant_logs <- lapply(
  mods_noant$logs,
  function(m) pretrend_wald(m, ref_tau = mods_noant$ref, from = -24L, to = -5L)
)

cat("\n\n", strrep("=", 84), "\nJOINT WALD PRETREND TESTS — LEVELS\n", strrep("=", 84), "\n", sep = "")
for (nm in names(wald_full_levels)) {
  print_pretrend(paste("WITH anticipation —", nm), wald_full_levels[[nm]])
}
for (nm in names(wald_noant_levels)) {
  print_pretrend(paste("WITHOUT anticipation —", nm), wald_noant_levels[[nm]])
}

cat("\n\n", strrep("=", 84), "\nJOINT WALD PRETREND TESTS — LOGS\n", strrep("=", 84), "\n", sep = "")
for (nm in names(wald_full_logs)) {
  print_pretrend(paste("WITH anticipation —", nm), wald_full_logs[[nm]])
}
for (nm in names(wald_noant_logs)) {
  print_pretrend(paste("WITHOUT anticipation —", nm), wald_noant_logs[[nm]])
}

wald_levels_table <- bind_rows(
  wald_df_from_list(wald_full_levels,  "2 Year Window with Anticipation"),
  wald_df_from_list(wald_noant_levels, "2 Year Window with Donut")
)

wald_logs_table <- bind_rows(
  wald_df_from_list(wald_full_logs,  "2 Year Window with Anticipation"),
  wald_df_from_list(wald_noant_logs, "2 Year Window with Donut")
)

cat("\n\n================ WALD SUMMARY TABLE: LEVELS ================\n")
print(wald_levels_table, row.names = FALSE)

cat("\n\n================ WALD SUMMARY TABLE: LOGS ================\n")
print(wald_logs_table, row.names = FALSE)

# ------------------------------ Preview plots only ------------------------------
plot_block_levels(mods_full,  "MCR timing with anticipation",    event_window = c(-24L, 24L))
plot_block_levels(mods_noant, "MCR timing without anticipation", event_window = c(-24L, 24L))

plot_block_logs(mods_full,  "MCR timing with anticipation",    event_window = c(-24L, 24L))
plot_block_logs(mods_noant, "MCR timing without anticipation", event_window = c(-24L, 24L))

cat("\nPreview plots and Wald tests completed for MCR timing panel.\n")