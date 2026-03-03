# C:/Repositories/white-bowblis-nhmc/regressions/stacked_event_study.R
# Stacked DiD Event Study (cohort-by-cohort stacking)
# - Builds cohort-specific samples where controls are never-treated or not-yet-treated (relative to cohort)
# - Drops treated-as-controls by construction
# - Supports donut (drop rel in {-3,-2,-1} for treated cohort only)
# - Produces fixest iplots like your TWFE script

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
set_plot_font()

# ------------------------------ 0) Load ------------------------------
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_plots <- "C:/Repositories/white-bowblis-nhmc/outputs/plots"
dir.create(out_plots, showWarnings = FALSE, recursive = TRUE)

keep_cols <- c(
  "cms_certification_number","year_month","ym_date",
  "anticipation2","event_time","treatment","time","time_treated",
  "government","non_profit","chain","beds",
  "occupancy_rate","pct_medicare","pct_medicaid",
  "cm_q_state_2","cm_q_state_3","cm_q_state_4",
  "rn_hppd","lpn_hppd","cna_hppd","total_hppd"
)

df <- read_csv(panel_fp, show_col_types = FALSE) %>%
  select(any_of(keep_cols)) %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.factor(year_month)
  )

# ------------------------------ 1) Controls + logs ------------------------------
controls_rhs <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)

mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)
df <- df %>%
  mutate(
    ln_rn    = mk_log(rn_hppd),
    ln_lpn   = mk_log(lpn_hppd),
    ln_cna   = mk_log(cna_hppd),
    ln_total = mk_log(total_hppd)
  )

outs_lvl <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")
outs_log <- c("ln_rn","ln_lpn","ln_cna","ln_total")

# ------------------------------ 2) Build cohort (g_i) at facility level ------------------------------
# g_i = unique time_treated for treated facilities; NA for never-treated
g_df <- df %>%
  group_by(cms_certification_number) %>%
  summarise(
    g = {
      tt <- unique(time_treated[!is.na(time_treated)])
      if (length(tt) == 1) as.integer(tt) else NA_integer_
    },
    .groups = "drop"
  )

df <- df %>%
  left_join(g_df, by = "cms_certification_number") %>%
  mutate(
    ever_treated = as.integer(!is.na(g))
  )

cohorts <- sort(unique(df$g[!is.na(df$g)]))
cat("Unique cohorts (treated months):", length(cohorts), "\n")

# ------------------------------ 3) Stacking function ------------------------------
make_stacked_data <- function(data, L = 24L, R = 24L, donut = TRUE, drop_set = -3:-1) {
  
  # Build cohort-specific stacks:
  # - Treated in stack: facilities with g == g0
  # - Controls in stack: never-treated (g is NA) OR treated later (g > g0)
  # - Keep only times in [g0-L, g0+R]
  # - rel = time - g0
  # - Drop donut leads ONLY for treated cohort units (treated_stack==1)
  
  stacked <- lapply(cohorts, function(g0) {
    d <- data %>%
      filter(time >= g0 - L, time <= g0 + R) %>%
      filter(is.na(g) | g > g0 | g == g0) %>%
      mutate(
        cohort = g0,
        rel = as.integer(time - g0),
        treated_stack = as.integer(!is.na(g) & g == g0),
        # unique id within stack for FE (facility duplicated across cohorts)
        stack_id = interaction(cms_certification_number, cohort, drop = TRUE)
      )
    
    if (donut) {
      d <- d %>% filter(treated_stack == 0L | !(rel %in% drop_set))
    }
    d
  })
  
  bind_rows(stacked)
}

# ------------------------------ 4) Stacked event-study regression + plots ------------------------------
run_stacked_es <- function(lhs, data_stacked, ref = -4L, window = c(-24L, 24L)) {
  
  fml <- as.formula(paste0(
    lhs, " ~ i(rel, treated_stack, ref = ", ref,
    ", keep = ", window[1], ":", window[2], ") + ",
    controls_rhs,
    " | stack_id + year_month + cohort"
  ))
  
  feols(
    fml,
    data = data_stacked,
    vcov = ~ cms_certification_number + year_month,  # cluster on original facility + calendar month
    lean = TRUE
  )
}

save_iplot <- function(model, fname, ref, window, ylab_txt) {
  grDevices::cairo_pdf(
    filename = file.path(out_plots, fname),
    width  = 9.5,
    height = 6.2
  )
  on.exit(dev.off(), add = TRUE)
  set_plot_font()
  iplot(
    model,
    ref  = ref,
    xlim = window,
    xlab = "Months relative to treatment",
    ylab = ylab_txt,
    main = "",
    sub  = ""
  )
}

# ------------------------------ 5) Build stacked data and run baseline ------------------------------
L <- 24L
R <- 24L
ref_tau <- -4L
event_window <- c(-24L, 24L)

# Baseline stacked = donut drops {-3,-2,-1} for treated cohort only
stack_base <- make_stacked_data(df, L = L, R = R, donut = TRUE, drop_set = -3:-1)

cat("Stacked baseline rows:", nrow(stack_base), "\n")

mods_lvl <- lapply(outs_lvl, \(y) run_stacked_es(y, stack_base, ref = ref_tau, window = event_window))
names(mods_lvl) <- outs_lvl

# Save plots (levels)
save_iplot(mods_lvl[["rn_hppd"]],    "stacked_es_rn_baseline.pdf",    ref_tau, event_window, "RN HPPD")
save_iplot(mods_lvl[["lpn_hppd"]],   "stacked_es_lpn_baseline.pdf",   ref_tau, event_window, "LPN HPPD")
save_iplot(mods_lvl[["cna_hppd"]],   "stacked_es_cna_baseline.pdf",   ref_tau, event_window, "CNA HPPD")
save_iplot(mods_lvl[["total_hppd"]], "stacked_es_total_baseline.pdf", ref_tau, event_window, "Total HPPD")

cat("Saved stacked baseline plots to: ", out_plots, "\n", sep = "")

# ------------------------------ 6) Optional: stacked donut sensitivity (like your robustness) ------------------------------
stack_wide_donut <- make_stacked_data(df, L = L, R = R, donut = TRUE, drop_set = -4:-1)
stack_small_donut <- make_stacked_data(df, L = L, R = R, donut = TRUE, drop_set = -2:-1)

m_total_wide  <- run_stacked_es("total_hppd", stack_wide_donut,  ref = ref_tau, window = event_window)
m_total_small <- run_stacked_es("total_hppd", stack_small_donut, ref = ref_tau, window = event_window)

save_iplot(m_total_wide,  "stacked_es_total_drop_m4_to_m1.pdf", ref_tau, event_window, "Total HPPD")
save_iplot(m_total_small, "stacked_es_total_drop_m2_to_m1.pdf", ref_tau, event_window, "Total HPPD")

cat("Done.\n")

# After stack_base is created:
rhs_vars <- c(
  "government","non_profit","chain","beds",
  "occupancy_rate","pct_medicare","pct_medicaid",
  "cm_q_state_2","cm_q_state_3","cm_q_state_4",
  "rel","treated_stack","year_month","stack_id","cohort"
)

miss_counts <- stack_base %>%
  summarise(across(all_of(rhs_vars), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to="var", values_to="n_miss") %>%
  arrange(desc(n_miss))

print(miss_counts)
print(head(miss_counts, 15))

# How many rows have ANY missing on RHS?
cat("Rows with any RHS missing: ",
    sum(!complete.cases(stack_base[, rhs_vars])), "\n", sep="")
cat("Share with any RHS missing: ",
    mean(!complete.cases(stack_base[, rhs_vars])), "\n", sep="")