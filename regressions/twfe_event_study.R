# C:/Repositories/white-bowblis-nhmc/regressions/twfe_event_study.R
# TWFE Event Study
# Specs in this file:
#   (A) WITH anticipation              (full sample)
#   (B) WITHOUT anticipation (drop t in {-3,-2,-1})
#   (C) Pre-pandemic (2017-01..2019-12) vs Pandemic (2020-04..2024-06),
#       each WITH and WITHOUT anticipation
#   (D) Robustness: change event-time window, change anticipation window
#
# Outcomes: RN, LPN, CNA, Total — in levels and logs (logs only if > 0)

source("C:/Repositories/white-bowblis-nhmc/regressions/_setup.R")

suppressPackageStartupMessages({
  library(MASS)   # for ginv() in pretrend tests if needed later
})

# ------------------------------ Plot font (Times / newtx-like) ------------------------------
set_plot_font <- function() {
  fam <- "Times New Roman"
  par(family = fam)
}
set_plot_font()

# ------------------------------ Paths ------------------------------
out_plots <- out_plots_dir
dir.create(out_plots, showWarnings = FALSE, recursive = TRUE)

presentation_dir <- file.path(project_root, "presentation")
dir.create(presentation_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------ 0) Load ------------------------------
df <- load_staffing_panel() %>%
  dplyr::mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.factor(year_month)
  ) %>%
  prepare_event_study_data(min_et = -24L, max_et = 24L)

# ------------------------------ 1) Treated window + logs ------------------------------
# load_staffing_panel() already created ln_rn, ln_lpn, ln_cna, ln_total where possible
# prepare_event_study_data() already created ever_treated and event_time_capped

# ------------------------------ 2) Controls (TWFE set) ------------------------------
controls_rhs <- make_controls_rhs(df)

# pick a valid ES reference (prefer -1, else -4, else nearest negative, else first)
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

# nearest-pre single-coefficient test helper
nearest_pre_test <- function(mod, ref_tau) {
  if (is.null(mod)) return(list(note = "Model is NULL"))
  
  cn <- names(coef(mod))
  if (is.null(cn) || !length(cn)) return(list(note = "No coefficients found"))
  
  pat <- "^event_time_capped::[-]?[0-9]+:ever_treated$"
  es_names <- grep(pat, cn, value = TRUE)
  if (!length(es_names)) return(list(note = "No event-study coefficients found"))
  
  get_tau <- function(s) as.integer(regmatches(s, regexpr("-?[0-9]+", s)))
  taus <- vapply(es_names, get_tau, integer(1))
  
  pre_names <- es_names[taus < 0 & taus != ref_tau]
  pre_taus  <- taus[taus < 0 & taus != ref_tau]
  
  if (!length(pre_names)) return(list(note = "No preperiod coefficients available"))
  
  target_tau <- max(pre_taus)
  target_name <- pre_names[which.max(pre_taus)]
  
  b <- unname(coef(mod)[target_name])
  se <- sqrt(diag(vcov(mod)))[target_name]
  z <- b / se
  pval <- 2 * pnorm(abs(z), lower.tail = FALSE)
  
  list(
    tau = target_tau,
    coef = b,
    se = se,
    z = z,
    p.value = pval,
    name = target_name
  )
}

print_pretrend <- function(title, res) {
  cat("\n", title, "\n", sep = "")
  if (!is.null(res$note)) {
    cat("NOTE: ", res$note, "\n", sep = "")
  } else if (!is.null(res$coef)) {
    cat(sprintf(
      "Nearest-pre (τ=%d): coef = %.4f, se = %.4f, z = %.2f, p = %.4g\n",
      res$tau, res$coef, res$se, res$z, res$p.value
    ))
    cat("Name: ", res$name, "\n", sep = "")
  }
}

# ------------------------------ 4) Outcomes ------------------------------
outs_lvl <- c("rn_hprd", "lpn_hprd", "cna_hprd", "total_hprd")
outs_log <- c("ln_rn", "ln_lpn", "ln_cna", "ln_total")

# ------------------------------ 5) Run models + SAVE PLOTS ------------------------------
fit_block <- function(tag, data, desired_ref = -1L, print_logs = TRUE,
                      save_dir = NULL, width_px = 1800, height_px = 1200, dpi = 200,
                      event_window = c(-24L, 24L)) {
  
  cat("\n\n", strrep("=", 84), "\nBLOCK: ", tag, "\n", strrep("=", 84), "\n", sep = "")
  ref <- pick_ref(data, desired = desired_ref)
  cat("Reference used: t = ", ref, "\n", sep = "")
  
  # LEVELS
  mods_lvl <- lapply(outs_lvl, \(y) run_es_twfe(y, data, ref_val = ref, window = event_window))
  names(mods_lvl) <- outs_lvl
  
  # LOGS
  mods_log <- lapply(outs_log, \(y) run_es_twfe(y, data, ref_val = ref, window = event_window))
  names(mods_log) <- outs_log
  
  # Print compact summaries for event-time coefficients only
  cat("\n--- Event-time coefficients (LEVELS) ---\n")
  lapply(mods_lvl, \(m) print(summary(m, keep = "^event_time_capped::")))
  if (print_logs) {
    cat("\n--- Event-time coefficients (LOGS) ---\n")
    lapply(mods_log, \(m) print(summary(m, keep = "^event_time_capped::")))
  }
  
  # Helper: save one iplot to file
  save_iplot <- function(model, fname, ylab_txt, main_txt) {
    if (!is.null(save_dir)) {
      dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)
      
      fname_pdf <- sub("\\.png$", ".pdf", fname, ignore.case = TRUE)
      if (!grepl("\\.pdf$", fname_pdf, ignore.case = TRUE)) {
        fname_pdf <- paste0(fname_pdf, ".pdf")
      }
      
      grDevices::cairo_pdf(
        filename = file.path(save_dir, fname_pdf),
        width  = 9.5,
        height = 6.2
      )
      on.exit(dev.off(), add = TRUE)
      
      set_plot_font()
      
      iplot(
        model,
        ref  = ref,
        xlim = event_window,
        xlab = "Months relative to treatment",
        ylab = ylab_txt,
        main = "",
        sub  = ""
      )
    }
  }
  
  tag_safe <- gsub("[^A-Za-z0-9]+", "_", tolower(tag))
  
  # RN
  iplot(mods_lvl[["rn_hprd"]], ref = ref, xlim = event_window,
        xlab = "Months relative to treatment", ylab = "RN HPRD",
        main = "", sub = "")
  save_iplot(mods_lvl[["rn_hprd"]],
             sprintf("twfe_es_rn_%s.pdf", tag_safe),
             "RN HPRD", paste0("TWFE ES: RN — ", tag))
  
  # LPN
  iplot(mods_lvl[["lpn_hprd"]], ref = ref, xlim = event_window,
        xlab = "Months relative to treatment", ylab = "LPN HPRD",
        main = "", sub = "")
  save_iplot(mods_lvl[["lpn_hprd"]],
             sprintf("twfe_es_lpn_%s.pdf", tag_safe),
             "LPN HPRD", paste0("TWFE ES: LPN — ", tag))
  
  # CNA
  iplot(mods_lvl[["cna_hprd"]], ref = ref, xlim = event_window,
        xlab = "Months relative to treatment", ylab = "CNA HPRD",
        main = "", sub = "")
  save_iplot(mods_lvl[["cna_hprd"]],
             sprintf("twfe_es_cna_%s.pdf", tag_safe),
             "CNA HPRD", paste0("TWFE ES: CNA — ", tag))
  
  # TOTAL
  iplot(mods_lvl[["total_hprd"]], ref = ref, xlim = event_window,
        xlab = "Months relative to treatment", ylab = "Total HPRD",
        main = "", sub = "")
  save_iplot(mods_lvl[["total_hprd"]],
             sprintf("twfe_es_total_%s.pdf", tag_safe),
             "Total HPRD", paste0("TWFE ES: Total — ", tag))
  
  # LOGS
  if (print_logs) {
    iplot(mods_log[["ln_rn"]], ref = ref, xlim = event_window,
          xlab = "Months relative to treatment", ylab = "Log(RN HPRD)",
          main = "", sub = "")
    save_iplot(mods_log[["ln_rn"]],
               sprintf("twfe_es_lnrn_%s.pdf", tag_safe),
               "Log(RN HPRD)", paste0("TWFE ES: log RN — ", tag))
    
    iplot(mods_log[["ln_lpn"]], ref = ref, xlim = event_window,
          xlab = "Months relative to treatment", ylab = "Log(LPN HPRD)",
          main = "", sub = "")
    save_iplot(mods_log[["ln_lpn"]],
               sprintf("twfe_es_lnlpn_%s.pdf", tag_safe),
               "Log(LPN HPRD)", paste0("TWFE ES: log LPN — ", tag))
    
    iplot(mods_log[["ln_cna"]], ref = ref, xlim = event_window,
          xlab = "Months relative to treatment", ylab = "Log(CNA HPRD)",
          main = "", sub = "")
    save_iplot(mods_log[["ln_cna"]],
               sprintf("twfe_es_lncna_%s.pdf", tag_safe),
               "Log(CNA HPRD)", paste0("TWFE ES: log CNA — ", tag))
    
    iplot(mods_log[["ln_total"]], ref = ref, xlim = event_window,
          xlab = "Months relative to treatment", ylab = "Log(Total HPRD)",
          main = "", sub = "")
    save_iplot(mods_log[["ln_total"]],
               sprintf("twfe_es_lntotal_%s.pdf", tag_safe),
               "Log(Total HPRD)", paste0("TWFE ES: log Total — ", tag))
  }
  
  invisible(list(levels = mods_lvl, logs = mods_log, ref = ref, tag = tag))
}

# ------------------------------ 6) Samples ------------------------------
S_full <- df
S_noant <- drop_anticipation_window(df)

S_pre <- sample_prepandemic(df)
S_pre_noant <- drop_anticipation_window(S_pre)

S_post <- sample_pandemic(df)
S_post_noant <- drop_anticipation_window(S_post)

# ------------------------------ 7) Run main blocks ------------------------------
res_full <- fit_block(
  tag = "WITH anticipation",
  data = S_full,
  desired_ref = -1L,
  print_logs = TRUE,
  save_dir = out_plots,
  event_window = c(-24L, 24L)
)

res_noant <- fit_block(
  tag = "WITHOUT anticipation",
  data = S_noant,
  desired_ref = -4L,
  print_logs = TRUE,
  save_dir = out_plots,
  event_window = c(-24L, 24L)
)

res_pre <- fit_block(
  tag = "Pre-pandemic WITH anticipation",
  data = S_pre,
  desired_ref = -1L,
  print_logs = TRUE,
  save_dir = out_plots,
  event_window = c(-24L, 24L)
)

res_pre_noant <- fit_block(
  tag = "Pre-pandemic WITHOUT anticipation",
  data = S_pre_noant,
  desired_ref = -4L,
  print_logs = TRUE,
  save_dir = out_plots,
  event_window = c(-24L, 24L)
)

res_post <- fit_block(
  tag = "Pandemic WITH anticipation",
  data = S_post,
  desired_ref = -1L,
  print_logs = TRUE,
  save_dir = out_plots,
  event_window = c(-24L, 24L)
)

res_post_noant <- fit_block(
  tag = "Pandemic WITHOUT anticipation",
  data = S_post_noant,
  desired_ref = -4L,
  print_logs = TRUE,
  save_dir = out_plots,
  event_window = c(-24L, 24L)
)

# ------------------------------ Extra presentation baseline plots ------------------------------
# Kept separate because you were already saving these to the presentation folder.
fit_block(
  tag = "Presentation baseline WITH anticipation",
  data = S_full,
  desired_ref = -1L,
  print_logs = FALSE,
  save_dir = presentation_dir,
  event_window = c(-24L, 24L)
)

fit_block(
  tag = "Presentation baseline WITHOUT anticipation",
  data = S_noant,
  desired_ref = -4L,
  print_logs = FALSE,
  save_dir = presentation_dir,
  event_window = c(-24L, 24L)
)

cat("\nSaved extra presentation baseline plots to:\n", presentation_dir, "\n", sep = "")

# ------------------------------ 8) TWFE robustness: event-window and anticipation-window ------------------------------
robust_specs <- list(
  list(
    name         = "noant_win_24",
    tag          = "Robustness: WITHOUT anticipation, window [-24,24]",
    data         = S_noant,
    desired_ref  = -4L,
    event_window = c(-24L, 24L)
  ),
  list(
    name         = "noant_win_18",
    tag          = "Robustness: WITHOUT anticipation, window [-18,18]",
    data         = S_noant,
    desired_ref  = -4L,
    event_window = c(-18L, 18L)
  ),
  list(
    name         = "noant_win_12",
    tag          = "Robustness: WITHOUT anticipation, window [-12,12]",
    data         = S_noant,
    desired_ref  = -4L,
    event_window = c(-12L, 12L)
  ),
  list(
    name         = "drop_m4_to_m1",
    tag          = "Robustness: drop t in {-4,-3,-2,-1}",
    data         = df %>% dplyr::filter(is.na(event_time) | !(event_time %in% -4:-1)),
    desired_ref  = -1L,
    event_window = c(-24L, 24L)
  ),
  list(
    name         = "drop_m2_to_m1",
    tag          = "Robustness: drop t in {-2,-1}",
    data         = df %>% dplyr::filter(is.na(event_time) | !(event_time %in% c(-2, -1))),
    desired_ref  = -1L,
    event_window = c(-24L, 24L)
  )
)

robust_results <- list()
for (sp in robust_specs) {
  cat("\n\n", strrep("-", 60), "\nROBUSTNESS BLOCK: ", sp$name, "\n",
      strrep("-", 60), "\n", sep = "")
  robust_results[[sp$name]] <- fit_block(
    tag          = sp$tag,
    data         = sp$data,
    desired_ref  = sp$desired_ref,
    print_logs   = FALSE,
    save_dir     = out_plots,
    event_window = sp$event_window
  )
}

cat("\nAll TWFE robustness blocks completed.\n")
cat("\nDone.\n")