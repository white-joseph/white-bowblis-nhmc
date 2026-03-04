# C:/Repositories/white-bowblis-nhmc/regressions/twfe_event_study_interactions_noant.R
# PURPOSE:
#   Event-study DiD heterogeneity via interactions (ONLY "WITHOUT anticipation"),
#   with plots styled like your existing TWFE event-study plots:
#     - points + 95% CI (no connecting lines)
#     - same minimal/base look
#     - BOTH series use circles (same shape), differentiated ONLY by color
#
# OUTPUT (PDFs, no PNGs):
#   C:/Repositories/white-bowblis-nhmc/outputs/plots/
#     es_chain_vs_nonchain_<outcome>_noant.pdf
#     es_pre_vs_pandemic_<outcome>_noant.pdf

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(stringr)
  library(tibble)
  library(scales)
})

# ------------------------------ Plot font (Times / newtx-like) ------------------------------
set_plot_font <- function() par(family = "Times New Roman")
set_plot_font()

# ------------------------------ Paths ------------------------------
panel_fp  <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_plots <- "C:/Repositories/white-bowblis-nhmc/outputs/plots"
dir.create(out_plots, showWarnings = FALSE, recursive = TRUE)

# ------------------------------ Load ------------------------------
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
    year_month = as.factor(year_month),
    ym_date = as.Date(paste0(gsub("/", "-", as.character(year_month)), "-01"))
  )

# ------------------------------ Event time capping + ever-treated ------------------------------
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

# ------------------------------ Baseline chain status (2017Q1) ------------------------------
baseline_window <- df %>%
  filter(ym_date >= as.Date("2017-01-01"), ym_date <= as.Date("2017-03-31")) %>%
  arrange(cms_certification_number, ym_date) %>%
  group_by(cms_certification_number) %>%
  summarise(baseline_chain_2017Q1 = dplyr::first(chain), .groups = "drop")

df <- df %>%
  left_join(baseline_window, by = "cms_certification_number") %>%
  mutate(
    baseline_chain_2017Q1 = as.integer(baseline_chain_2017Q1),
    baseline_nonchain_2017Q1 = ifelse(is.na(baseline_chain_2017Q1), NA_integer_, 1L - baseline_chain_2017Q1)
  )

# ------------------------------ Period indicators ------------------------------
df <- df %>%
  mutate(
    pre_period  = ym_date >= as.Date("2017-01-01") & ym_date <= as.Date("2019-12-31"),
    pand_period = ym_date >= as.Date("2020-04-01") & ym_date <= as.Date("2024-06-30")
  )

# ------------------------------ Controls (same as your TWFE set) ------------------------------
controls_rhs <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)

# ------------------------------ Helpers ------------------------------
pick_ref_simple <- function(desired, data) {
  ev <- sort(unique(data$event_time_capped[data$ever_treated == 1L]))
  ev <- ev[is.finite(ev) & ev != 9999L]
  if (!length(ev)) stop("No treated event times found.")
  if (desired %in% ev) return(as.integer(desired))
  if (-4L %in% ev) return(-4L)
  if (-1L %in% ev) return(-1L)
  negs <- ev[ev < 0L]
  if (length(negs)) return(max(negs))
  ev[1]
}

extract_es_series <- function(mod, var = "event_time_capped", treat_names, labels) {
  ct <- as.data.frame(summary(mod)$coeftable)
  ct$term <- rownames(ct)
  
  out <- list()
  for (j in seq_along(treat_names)) {
    tn  <- treat_names[j]
    lab <- labels[j]
    pat <- paste0("^", var, "::(-?\\d+):", tn, "$")
    m <- str_match(ct$term, pat)
    keep <- !is.na(m[, 2])
    if (!any(keep)) next
    
    dd <- tibble(
      tau = as.integer(m[keep, 2]),
      estimate = ct$Estimate[keep],
      se = ct$`Std. Error`[keep],
      group = lab
    ) %>%
      mutate(
        ci_lo = estimate - 1.96 * se,
        ci_hi = estimate + 1.96 * se
      ) %>%
      arrange(tau)
    
    out[[length(out) + 1]] <- dd
  }
  bind_rows(out)
}

tight_ylims <- function(dd, pad_frac = 0.12) {
  lo <- min(dd$ci_lo, na.rm = TRUE)
  hi <- max(dd$ci_hi, na.rm = TRUE)
  rng <- hi - lo
  pad <- ifelse(rng == 0, 0.05, pad_frac * rng)
  c(lo - pad, hi + pad)
}

# Plot styled to match your standard TWFE ES plots: points + CI, no lines.
# - both series are circles (shape fixed)
# - differentiate series only by color
plot_es_overlay <- function(dd, out_pdf, ylab,
                            colors = c("Chain (baseline 2017Q1)" = "#1f77b4",
                                       "Non-chain (baseline 2017Q1)" = "#d62728"),
                            x_breaks = seq(-24, 24, by = 6),
                            show_legend = TRUE) {
  dd <- dd %>% mutate(group = factor(group, levels = names(colors)))
  yl <- tight_ylims(dd)
  
  p <- ggplot(dd, aes(x = tau, y = estimate, color = group)) +
    geom_hline(yintercept = 0, linewidth = 0.4) +
    # fixest iplot has a "treatment month" divider; we use a subtle line at 0
    geom_vline(xintercept = 0, linewidth = 0.35, linetype = "dashed") +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.25, linewidth = 0.35) +
    geom_point(shape = 16, size = 2.1) +
    scale_x_continuous(breaks = x_breaks) +
    scale_y_continuous(limits = yl, breaks = pretty_breaks(n = 6)) +
    scale_color_manual(values = colors) +
    labs(x = "Months relative to treatment", y = ylab, color = NULL) +
    theme_minimal(base_size = 14, base_family = "Times New Roman") +
    theme(
      panel.grid.minor = element_blank(),
      legend.position = if (show_legend) "top" else "none"
    )
  
  ggsave(out_pdf, p, width = 9.5, height = 6.2, device = cairo_pdf)
  message("Saved: ", normalizePath(out_pdf, winslash = "\\"))
  invisible(p)
}

# ------------------------------ Model runners (INTERACTIONS ONLY) ------------------------------
run_es_chain_vs_nonchain <- function(data, lhs, ref_val) {
  d <- data %>% filter(!is.na(baseline_chain_2017Q1)) %>%
    mutate(
      treat_chain    = treatment * baseline_chain_2017Q1,
      treat_nonchain = treatment * baseline_nonchain_2017Q1
    )
  
  fml <- as.formula(paste0(
    lhs, " ~ ",
    "i(event_time_capped, treat_chain, ref = ", ref_val, ", keep = -24:24) + ",
    "i(event_time_capped, treat_nonchain, ref = ", ref_val, ", keep = -24:24) + ",
    controls_rhs,
    " | cms_certification_number + year_month"
  ))
  
  feols(fml, data = d, vcov = ~ cms_certification_number + year_month, lean = TRUE)
}

run_es_pre_vs_pandemic <- function(data, lhs, ref_val) {
  d <- data %>%
    filter(pre_period | pand_period) %>%
    mutate(
      treat_pre  = treatment * as.integer(pre_period),
      treat_pand = treatment * as.integer(pand_period)
    )
  
  fml <- as.formula(paste0(
    lhs, " ~ ",
    "i(event_time_capped, treat_pre, ref = ", ref_val, ", keep = -24:24) + ",
    "i(event_time_capped, treat_pand, ref = ", ref_val, ", keep = -24:24) + ",
    controls_rhs,
    " | cms_certification_number + year_month"
  ))
  
  feols(fml, data = d, vcov = ~ cms_certification_number + year_month, lean = TRUE)
}

# ------------------------------ ONLY WITHOUT anticipation ------------------------------
S_noant <- df %>% filter(anticipation2 == 0)
ref <- pick_ref_simple(-4L, S_noant)
cat("Running interaction ES (WITHOUT anticipation). Reference tau = ", ref, "\n", sep = "")

outs_lvl <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")

# Colors (edit these if you want different ones)
cols_chain <- c(
  "Chain (baseline 2017Q1)"    = "#1f77b4",  # blue
  "Non-chain (baseline 2017Q1)"= "#d62728"   # red
)
cols_period <- c(
  "Pre-pandemic (2017–2019)" = "#1f77b4",    # blue
  "Pandemic (2020–2024)"     = "#d62728"     # red
)

for (y in outs_lvl) {
  ylab <- paste0(toupper(gsub("_hppd","", y)), " HPPD")
  
  # ---- Chain vs Non-chain ----
  mod_c <- run_es_chain_vs_nonchain(S_noant, y, ref_val = ref)
  dd_c <- extract_es_series(
    mod_c,
    treat_names = c("treat_chain", "treat_nonchain"),
    labels      = c("Chain (baseline 2017Q1)", "Non-chain (baseline 2017Q1)")
  )
  
  out_c <- file.path(out_plots, paste0("es_chain_vs_nonchain_", y, "_noant.pdf"))
  plot_es_overlay(dd_c, out_pdf = out_c, ylab = ylab, colors = cols_chain)
  
  # ---- Pre vs Pandemic ----
  mod_p <- run_es_pre_vs_pandemic(S_noant, y, ref_val = ref)
  dd_p <- extract_es_series(
    mod_p,
    treat_names = c("treat_pre", "treat_pand"),
    labels      = c("Pre-pandemic (2017–2019)", "Pandemic (2020–2024)")
  )
  
  out_p <- file.path(out_plots, paste0("es_pre_vs_pandemic_", y, "_noant.pdf"))
  plot_es_overlay(dd_p, out_pdf = out_p, ylab = ylab, colors = cols_period)
}

cat("\nDone.\n")