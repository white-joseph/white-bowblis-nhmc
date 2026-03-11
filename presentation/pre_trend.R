# C:/Repositories/white-bowblis-nhmc/regressions/twfe_event_study_preonly_all4_plots.R
# TWFE Event Study (WITH anticipation, full sample)
# PURPOSE: Presentation-ready plots for RN/LPN/CNA/TOTAL staffing
#          plotting ONLY tau -24..0 and highlighting anticipation window (-3,-2,-1).
#
# OUTPUTS:
#   C:/Repositories/white-bowblis-nhmc/presentation/
#     - twfe_es_rn_preonly_tau_m24_to_0_highlight_anticipation_presentation.pdf
#     - twfe_es_lpn_preonly_tau_m24_to_0_highlight_anticipation_presentation.pdf
#     - twfe_es_cna_preonly_tau_m24_to_0_highlight_anticipation_presentation.pdf
#     - twfe_es_total_preonly_tau_m24_to_0_highlight_anticipation_presentation.pdf

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
})

# ------------------------------ Presentation plot style ------------------------------
# Match the slide-style formatting used in the other presentation plots.
set_presentation_plot_style <- function() {
  par(
    family   = "sans",
    cex.axis = 1.30,
    mar      = c(6.0, 6.2, 1.2, 1.2),  # a bit more room for manual axis titles
    mgp      = c(2.0, 0.85, 0),        # tick labels spacing
    tcl      = -0.3
  )
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
      data.frame(
        tau = as.integer(ref_tau),
        estimate = 0,
        se = NA_real_,
        ci_lo = NA_real_,
        ci_hi = NA_real_
      )
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
                              tau_min = -24L,
                              tau_max = 0L,
                              highlight_taus = -3:-1,
                              xlab = "Months relative to treatment",
                              ylab = "HPRD") {
  
  d <- get_es_df(mod, ref_tau = ref_tau)
  d <- d[d$tau >= tau_min & d$tau <= tau_max, ]
  if (!nrow(d)) stop("No coefficients in requested plotting window.")
  
  d$is_hi <- d$tau %in% highlight_taus
  has_ci <- is.finite(d$ci_lo) & is.finite(d$ci_hi)
  
  draw_panel <- function() {
    old_par <- par(no.readonly = TRUE)
    on.exit(par(old_par), add = TRUE)
    
    set_presentation_plot_style()
    
    # Base plot with no built-in axis titles
    iplot(
      mod,
      ref  = ref_tau,
      xlim = c(tau_min, tau_max),
      xlab = "",
      ylab = "",
      main = "",
      sub  = ""
    )
    
    # subtle anticipation-window shading
    if (length(highlight_taus)) {
      lo <- min(highlight_taus)
      hi <- max(highlight_taus)
      usr <- par("usr")
      rect(
        xleft   = lo - 0.5,
        ybottom = usr[3],
        xright  = hi + 0.5,
        ytop    = usr[4],
        col     = grDevices::adjustcolor("red", alpha.f = 0.06),
        border  = NA
      )
      
      # keep reference lines visible
      abline(h = 0, lty = 2, col = "gray50")
      abline(v = 0, lty = 1, col = "gray50")
    }
    
    # highlighted anticipation coefficients
    d_hi <- d[d$is_hi, ]
    has_ci_hi <- is.finite(d_hi$ci_lo) & is.finite(d_hi$ci_hi)
    
    if (nrow(d_hi) > 0) {
      segments(
        d_hi$tau[has_ci_hi], d_hi$ci_lo[has_ci_hi],
        d_hi$tau[has_ci_hi], d_hi$ci_hi[has_ci_hi],
        col = "red3",
        lwd = 2
      )
      
      points(
        d_hi$tau, d_hi$estimate,
        pch = 19,
        cex = 1.0,
        col = "red3"
      )
    }
    
    # manual axis titles with more separation
    mtext(
      text   = xlab,
      side   = 1,
      line   = 3.6,
      cex    = 1.45,
      family = "sans"
    )
    
    mtext(
      text   = ylab,
      side   = 2,
      line   = 3.9,
      cex    = 1.45,
      family = "sans"
    )
  }
  
  grDevices::cairo_pdf(out_pdf, width = 9.5, height = 6.2)
  draw_panel()
  grDevices::dev.off()
  
  invisible(d)
}

# ------------------------------ 6) Run models + save plots for all 4 ------------------------------
specs <- list(
  list(var = "rn_hppd",    stem = "rn",    ylab = "RN HPRD"),
  list(var = "lpn_hppd",   stem = "lpn",   ylab = "LPN HPRD"),
  list(var = "cna_hppd",   stem = "cna",   ylab = "CNA HPRD"),
  list(var = "total_hppd", stem = "total", ylab = "Total HPRD")
)

for (s in specs) {
  cat("\nEstimating + plotting: ", s$var, "\n", sep = "")
  
  mod <- run_es_twfe(lhs = s$var, data = df, ref_tau = ref_tau)
  
  pdf_fp <- file.path(
    out_pres,
    sprintf("twfe_es_%s_preonly_tau_m24_to_0_highlight_anticipation_presentation.pdf", s$stem)
  )
  
  save_preonly_plot(
    mod = mod,
    ref_tau = ref_tau,
    out_pdf = pdf_fp,
    tau_min = -24L,
    tau_max = 0L,
    highlight_taus = -3:-1,
    xlab = "Months relative to treatment",
    ylab = s$ylab
  )
  
  cat("Saved:\n")
  cat(" - ", pdf_fp, "\n", sep = "")
}

cat("\nDone.\n")