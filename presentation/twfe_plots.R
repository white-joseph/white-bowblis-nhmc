suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(scales)
  library(tibble)
})

OUT_DIR <- "C:/Repositories/white-bowblis-nhmc/presentation"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

theme_set(theme_minimal(base_size = 14, base_family = "Times New Roman"))

# -----------------------------
# Helper: compute tight y-lims from 95% CI
# -----------------------------
tight_ylims <- function(df, pad_frac = 0.12) {
  lo <- min(df$estimate - 1.96 * df$se, na.rm = TRUE)
  hi <- max(df$estimate + 1.96 * df$se, na.rm = TRUE)
  rng <- hi - lo
  pad <- ifelse(rng == 0, 0.05, pad_frac * rng)
  c(lo - pad, hi + pad)
}

# -----------------------------
# Plot one panel (either HPPD or Log(HPPD))
#   PDF only (no PNG)
# -----------------------------
plot_twfe_bars_onepanel <- function(df, panel_keep,
                                    title, out_stub,
                                    ylab = "Estimated effect of ownership change (post)",
                                    bar_width = 0.38,
                                    dodge_width = 0.62,
                                    out_width = 10,
                                    out_height = 6.2,
                                    fill_values = c(
                                      "Pre-pandemic (2017–2019)" = "#2C7FB8",
                                      "Pandemic (2020–2024)"     = "#DE2D26",
                                      "Chain (Jan 2017)"         = "#2C7FB8",
                                      "Non-chain (Jan 2017)"     = "#DE2D26",
                                      "Baseline"                 = "#2C7FB8"
                                    )) {
  
  d <- df %>%
    filter(panel == panel_keep) %>%
    mutate(
      outcome = factor(outcome, levels = c("RN", "LPN", "CNA", "Total")),
      group   = if (is.character(group)) factor(group, levels = unique(group)) else group,
      ci_lo   = estimate - 1.96 * se,
      ci_hi   = estimate + 1.96 * se
    )
  
  yl <- tight_ylims(d)
  
  pal <- fill_values[names(fill_values) %in% as.character(unique(d$group))]
  
  p <- ggplot(d, aes(x = outcome, y = estimate, fill = group)) +
    geom_hline(yintercept = 0, linewidth = 0.4) +
    geom_col(position = position_dodge(width = dodge_width), width = bar_width) +
    geom_errorbar(
      aes(ymin = ci_lo, ymax = ci_hi),
      position = position_dodge(width = dodge_width),
      width = 0.12,
      linewidth = 0.4
    ) +
    scale_y_continuous(limits = yl, breaks = pretty_breaks(n = 6)) +
    scale_fill_manual(values = pal, drop = FALSE) +
    labs(x = NULL, y = ylab, title = title, fill = NULL) +
    theme(
      panel.grid.minor = element_blank(),
      legend.position  = "top",
      axis.text.x      = element_text(size = 12),
      plot.title       = element_text(hjust = 0.5),
      plot.title.position = "plot"
    )
  
  out_pdf <- file.path(OUT_DIR, paste0(out_stub, ".pdf"))
  ggsave(out_pdf, p, width = out_width, height = out_height, device = cairo_pdf)
  
  message("Saved:\n- ", normalizePath(out_pdf))
  invisible(p)
}

# ============================================================
# DATA (same numbers you already have)
# ============================================================

df_tab3 <- tribble(
  ~panel,      ~group,     ~outcome, ~estimate, ~se,
  "HPPD",      "Baseline", "RN",     -0.032,    0.005,
  "HPPD",      "Baseline", "LPN",      0.000,   0.006,
  "HPPD",      "Baseline", "CNA",     -0.054,   0.009,
  "HPPD",      "Baseline", "Total",   -0.086,   0.013,
  "Log(HPPD)", "Baseline", "RN",     -0.092,    0.016,
  "Log(HPPD)", "Baseline", "LPN",      0.003,   0.009,
  "Log(HPPD)", "Baseline", "CNA",     -0.028,   0.005,
  "Log(HPPD)", "Baseline", "Total",   -0.026,   0.004
)

df_tab5 <- tribble(
  ~panel,      ~group,                 ~outcome, ~estimate, ~se,
  "HPPD",      "Chain (Jan 2017)",     "RN",     -0.031,    0.007,
  "HPPD",      "Chain (Jan 2017)",     "LPN",      0.003,   0.008,
  "HPPD",      "Chain (Jan 2017)",     "CNA",     -0.034,   0.011,
  "HPPD",      "Chain (Jan 2017)",     "Total",   -0.062,   0.015,
  "HPPD",      "Non-chain (Jan 2017)", "RN",     -0.037,    0.009,
  "HPPD",      "Non-chain (Jan 2017)", "LPN",      0.003,   0.011,
  "HPPD",      "Non-chain (Jan 2017)", "CNA",     -0.079,   0.019,
  "HPPD",      "Non-chain (Jan 2017)", "Total",   -0.113,   0.025,
  "Log(HPPD)", "Chain (Jan 2017)",     "RN",     -0.099,    0.021,
  "Log(HPPD)", "Chain (Jan 2017)",     "LPN",      0.008,   0.013,
  "Log(HPPD)", "Chain (Jan 2017)",     "CNA",     -0.019,   0.006,
  "Log(HPPD)", "Chain (Jan 2017)",     "Total",   -0.020,   0.005,
  "Log(HPPD)", "Non-chain (Jan 2017)", "RN",     -0.085,    0.030,
  "Log(HPPD)", "Non-chain (Jan 2017)", "LPN",      0.013,   0.017,
  "Log(HPPD)", "Non-chain (Jan 2017)", "CNA",     -0.037,   0.010,
  "Log(HPPD)", "Non-chain (Jan 2017)", "Total",   -0.032,   0.008
)

df_tab4 <- tribble(
  ~panel,      ~group,                     ~outcome, ~estimate, ~se,
  "HPPD",      "Pre-pandemic (2017–2019)", "RN",     -0.029,    0.006,
  "HPPD",      "Pre-pandemic (2017–2019)", "LPN",    -0.016,    0.008,
  "HPPD",      "Pre-pandemic (2017–2019)", "CNA",    -0.063,    0.013,
  "HPPD",      "Pre-pandemic (2017–2019)", "Total",  -0.108,    0.019,
  "HPPD",      "Pandemic (2020–2024)",     "RN",     -0.018,    0.007,
  "HPPD",      "Pandemic (2020–2024)",     "LPN",      0.000,   0.008,
  "HPPD",      "Pandemic (2020–2024)",     "CNA",     -0.041,   0.015,
  "HPPD",      "Pandemic (2020–2024)",     "Total",   -0.058,   0.020,
  "Log(HPPD)", "Pre-pandemic (2017–2019)", "RN",     -0.068,    0.024,
  "Log(HPPD)", "Pre-pandemic (2017–2019)", "LPN",    -0.009,    0.012,
  "Log(HPPD)", "Pre-pandemic (2017–2019)", "CNA",    -0.032,    0.006,
  "Log(HPPD)", "Pre-pandemic (2017–2019)", "Total",  -0.033,    0.006,
  "Log(HPPD)", "Pandemic (2020–2024)",     "RN",     -0.063,    0.021,
  "Log(HPPD)", "Pandemic (2020–2024)",     "LPN",      0.002,   0.011,
  "Log(HPPD)", "Pandemic (2020–2024)",     "CNA",     -0.020,   0.008,
  "Log(HPPD)", "Pandemic (2020–2024)",     "Total",   -0.019,   0.006
) %>%
  mutate(group = factor(group, levels = c("Pre-pandemic (2017–2019)", "Pandemic (2020–2024)")))

# ============================================================
# MAKE PLOTS (two per table)
# ============================================================

plot_twfe_bars_onepanel(df_tab3, "HPPD",
                        title = "TWFE Post Estimates: Baseline",
                        out_stub = "twfe_post_baseline_hppd_bars"
)

plot_twfe_bars_onepanel(df_tab3, "Log(HPPD)",
                        title = "TWFE Post Estimates: Baseline",
                        out_stub = "twfe_post_baseline_loghppd_bars",
                        ylab = "Estimated effect of ownership change (post, log points)"
)

plot_twfe_bars_onepanel(df_tab5, "HPPD",
                        title = "TWFE Post Estimates: Chain vs Non-chain",
                        out_stub = "twfe_post_chain_nonchain_hppd_bars"
)

plot_twfe_bars_onepanel(df_tab5, "Log(HPPD)",
                        title = "TWFE Post Estimates: Chain vs Non-chain",
                        out_stub = "twfe_post_chain_nonchain_loghppd_bars",
                        ylab = "Estimated effect of ownership change (post, log points)"
)

plot_twfe_bars_onepanel(df_tab4, "HPPD",
                        title = "TWFE Post Estimates: Pre-pandemic vs Pandemic",
                        out_stub = "twfe_post_prepandemic_pandemic_hppd_bars"
)

plot_twfe_bars_onepanel(df_tab4, "Log(HPPD)",
                        title = "TWFE Post Estimates: Pre-pandemic vs Pandemic",
                        out_stub = "twfe_post_prepandemic_pandemic_loghppd_bars",
                        ylab = "Estimated effect of ownership change (post, log points)"
)