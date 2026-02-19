# C:/Repositories/white-bowblis-nhmc/regressions/twfe_event_study_preonly_all4_plots.R
# TWFE Event Study (WITH anticipation, full sample)
# PURPOSE: Presentation-ready plots for RN/LPN/CNA/TOTAL staffing
#          plotting ONLY tau -24..0 and highlighting anticipation window (-3,-2,-1).
#
# OUTPUTS:
#   C:/Repositories/white-bowblis-nhmc/presentation/
#     - twfe_es_rn_preonly_tau_m24_to_0_highlight_anticipation.pdf/png
#     - twfe_es_lpn_preonly_tau_m24_to_0_highlight_anticipation.pdf/png
#     - twfe_es_cna_preonly_tau_m24_to_0_highlight_anticipation.pdf/png
#     - twfe_es_total_preonly_tau_m24_to_0_highlight_anticipation.pdf/png

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
})

# ------------------------------ Plot font (Times / newtx-like) ------------------------------
set_plot_font <- function() {
  fam <- "Times New Roman"
  par(family = fam)
}

# ------------------------------ Paths ------------------------------
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_pres <- "C:/Repositories/white-bowblis-nhmc/presentation"
dir.create(out_pres, showWarnings = FALSE, recursive = TRUE)

# ------------------------------ 0) Load ------------------------------
keep_cols <- c(
  "cms_certification_number","year_month","anticipation2",
  "event_time","treatment",
  "government","non_profit","chain","beds",
  "occupancy_rate","pct_medicare","pct_medicaid",
  "cm_q_state_2","cm_q_state_3","cm_q_state_4",
  "rn_hppd","lpn_hppd","cna_hppd","total_hppd"
)

df <- read_csv(panel_fp, show_col_types = FALSE, col_select = all_of(keep_cols)) %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.factor(year_month)
  )

# ------------------------------ 1) Treated window (cap event time to [-24, 24]) ------------------------------
df <- df %>%
  group_by(cms_certification_number) %>%
  mutate(ever_treated = as.integer(any(treatment == 1, na.rm = TRUE) | any(!is.na(event_time)))) %>%
  ungroup() %>%
  mutate(
    event_time_capped = case_when(
      ever_treated == 1L & !is.na(event_time) ~ pmin(pmax(as.integer(event_time), -24L), 24L),
      TRUE ~ 9999L
    )
  )

# ------------------------------ 2) Controls ------------------------------
controls_rhs <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)

# Pick a valid ES reference (prefer -1, else -4, else nearest negative, else first)
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

ref_tau <- pick_ref(df, desired = -1L)
cat("Reference period used (omitted tau): ", ref_tau, "\n", sep = "")

# ------------------------------ 3) Model runner ------------------------------
run_es_twfe <- function(lhs, data, ref_tau) {
  fml <- as.formula(paste0(
    lhs, " ~ i(event_time_capped, ever_treated, ref = ", ref_tau, ", keep = -24:24) + ",
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

# ------------------------------ 4) Extract ES coefficients ------------------------------
get_es_df <- function(mod,
                      var = "event_time_capped",
                      trt = "ever_treated",
                      ref_tau = NULL) {
  cn <- names(coef(mod))
  if (is.null(cn) || !length(cn)) stop("Model has no coefficients.")
  
  pat <- sprintf("^%s::-?\\d+:%s$", var, trt)
  es_names <- grep(pat, cn, value = TRUE)
  if (!length(es_names)) stop("No event-study coefficients found (pattern mismatch).")
  
  get_tau <- function(s) as.integer(regmatches(s, regexpr("-?\\d+", s)))
  taus <- vapply(es_names, get_tau, integer(1))
  
  b  <- coef(mod)[es_names]
  Vd <- diag(vcov(mod))[es_names]
  se <- sqrt(pmax(Vd, 0))
  
  df_es <- data.frame(
    tau = as.integer(taus),
    estimate = as.numeric(b),
    se = as.numeric(se),
    stringsAsFactors = FALSE
  )
  df_es$ci_lo <- df_es$estimate - 1.96 * df_es$se
  df_es$ci_hi <- df_es$estimate + 1.96 * df_es$se
  
  # Add omitted reference as a 0-effect point (no CI)
  if (!is.null(ref_tau) && !(ref_tau %in% df_es$tau)) {
    df_es <- rbind(
      df_es,
      data.frame(tau = as.integer(ref_tau), estimate = 0, se = NA_real_, ci_lo = NA_real_, ci_hi = NA_real_)
    )
  }
  
  df_es <- df_es[order(df_es$tau), ]
  rownames(df_es) <- NULL
  df_es
}

# ------------------------------ 5) Plot + save (pre-only, highlight anticipation) ------------------------------
save_preonly_plot <- function(mod,
                              ref_tau,
                              out_pdf,
                              out_png,
                              tau_min = -24L,
                              tau_max = 0L,
                              highlight_taus = -3:-1,
                              xlab = "Months relative to treatment",
                              ylab = "HPPD") {
  
  d <- get_es_df(mod, ref_tau = ref_tau)
  d <- d[d$tau >= tau_min & d$tau <= tau_max, ]
  if (!nrow(d)) stop("No coefficients in requested plotting window.")
  
  d$is_hi <- d$tau %in% highlight_taus
  
  pt_col <- ifelse(d$is_hi, "red3", "black")
  ci_col <- ifelse(d$is_hi, "red3", "gray40")
  has_ci <- is.finite(d$ci_lo) & is.finite(d$ci_hi)
  
  ylim <- range(c(d$ci_lo, d$ci_hi, 0), na.rm = TRUE)
  pad  <- 0.08 * diff(ylim)
  if (!is.finite(pad) || pad == 0) pad <- 0.1
  ylim <- c(ylim[1] - pad, ylim[2] + pad)
  
  draw_panel <- function() {
    set_plot_font()
    
    plot(d$tau, d$estimate,
         type = "n",
         xlim = c(tau_min, tau_max),
         ylim = ylim,
         xlab = xlab,
         ylab = ylab,
         axes = TRUE)
    
    # subtle shading for anticipation window
    if (length(highlight_taus)) {
      lo <- min(highlight_taus); hi <- max(highlight_taus)
      usr <- par("usr")
      rect(xleft = lo - 0.5, ybottom = usr[3],
           xright = hi + 0.5, ytop = usr[4],
           col = grDevices::adjustcolor("red", alpha.f = 0.06),
           border = NA)
    }
    
    abline(h = 0, lty = 2, col = "gray50")
    abline(v = 0, lty = 1, col = "gray50")
    
    segments(d$tau[has_ci], d$ci_lo[has_ci], d$tau[has_ci], d$ci_hi[has_ci],
             col = ci_col[has_ci], lwd = 2)
    
    points(d$tau, d$estimate, pch = 19, cex = 1.1, col = pt_col)
    
    legend("topleft",
           legend = c("Other pre-period months", "Anticipation months (-3 to -1)"),
           col    = c("black", "red3"),
           pch    = 19,
           bty    = "n",
           cex    = 0.95)
  }
  
  grDevices::cairo_pdf(out_pdf, width = 9.5, height = 6.2)
  draw_panel()
  grDevices::dev.off()
  
  grDevices::png(out_png, width = 1800, height = 1200, res = 200)
  draw_panel()
  grDevices::dev.off()
  
  invisible(d)
}

# ------------------------------ 6) Run models + save plots for all 4 ------------------------------
specs <- list(
  list(var = "rn_hppd",    stem = "rn",    ylab = "RN HPPD"),
  list(var = "lpn_hppd",   stem = "lpn",   ylab = "LPN HPPD"),
  list(var = "cna_hppd",   stem = "cna",   ylab = "CNA HPPD"),
  list(var = "total_hppd", stem = "total", ylab = "Total HPPD")
)

for (s in specs) {
  cat("\nEstimating + plotting: ", s$var, "\n", sep = "")
  
  mod <- run_es_twfe(lhs = s$var, data = df, ref_tau = ref_tau)
  
  pdf_fp <- file.path(out_pres, sprintf("twfe_es_%s_preonly_tau_m24_to_0_highlight_anticipation.pdf", s$stem))
  png_fp <- file.path(out_pres, sprintf("twfe_es_%s_preonly_tau_m24_to_0_highlight_anticipation.png", s$stem))
  
  save_preonly_plot(
    mod = mod,
    ref_tau = ref_tau,
    out_pdf = pdf_fp,
    out_png = png_fp,
    tau_min = -24L,
    tau_max = 0L,
    highlight_taus = -3:-1,
    xlab = "Months relative to treatment",
    ylab = s$ylab
  )
  
  cat("Saved:\n")
  cat(" - ", pdf_fp, "\n", sep = "")
  cat(" - ", png_fp, "\n", sep = "")
}

cat("\nDone.\n")