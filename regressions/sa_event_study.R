# C:/Repositories/white-bowblis-nhmc/regressions/sa_event_study_noant_fast.R
# Sun–Abraham (fixest::sunab) — NO ANTICIPATION ONLY (FAST VERSION)
# Key speedups:
#   - restrict window to [-12,12] initially
#   - bin relative time (bin.rel = "bin::2")
#   - run ONE outcome first (RN) to confirm it works
#   - cluster by facility only first (swap to 2-way later)

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
})

panel_fp  <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_plots <- "C:/Repositories/white-bowblis-nhmc/outputs/plots"
dir.create(out_plots, showWarnings = FALSE, recursive = TRUE)

keep_cols <- c(
  "cms_certification_number","year_month","anticipation2",
  "event_time","treatment","post",
  "time","time_treated",
  "government","non_profit","chain","beds",
  "occupancy_rate","pct_medicare","pct_medicaid",
  "cm_q_state_2","cm_q_state_3","cm_q_state_4",
  "rn_hppd","lpn_hppd","cna_hppd","total_hppd"
)

df <- read_csv(panel_fp, show_col_types = FALSE, col_select = all_of(keep_cols)) %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.factor(year_month),
    ym_date = as.Date(paste0(gsub("/", "-", as.character(year_month)), "-01")),
    time = suppressWarnings(as.integer(time)),
    time_treated = suppressWarnings(as.integer(time_treated)),
    event_time = suppressWarnings(as.integer(event_time)),
    treatment = ifelse(is.na(treatment), 0L, as.integer(treatment))
  )

# If time is weird, rebuild from ym_date
n_time <- n_distinct(df$time[!is.na(df$time)])
if (is.na(n_time) || n_time < 2L || n_time > 300L) {
  message("[SA] Rebuilding time from year_month because n_distinct(time) = ", n_time)
  df <- df %>% mutate(time = as.integer(factor(ym_date, levels = sort(unique(ym_date)))))
  message("[SA] New n_distinct(time) = ", n_distinct(df$time))
} else {
  message("[SA] Using existing time. n_distinct(time) = ", n_time)
}

# Drop treated facilities with missing time_treated (your 22)
bad_fac <- df %>%
  group_by(cms_certification_number) %>%
  summarise(
    treated_group = max(treatment, na.rm = TRUE),
    has_time_treated = any(!is.na(time_treated)),
    .groups = "drop"
  ) %>%
  filter(treated_group == 1L, !has_time_treated)

message("[SA] Treated facilities with missing time_treated (dropping): ", nrow(bad_fac))
if (nrow(bad_fac) > 0) {
  df <- df %>% filter(!(cms_certification_number %in% bad_fac$cms_certification_number))
}

# Build cohort: MUST be non-missing for sunab -> set controls cohort=0
facility_cohort <- df %>%
  group_by(cms_certification_number) %>%
  summarise(
    treated_group = max(treatment, na.rm = TRUE),
    cohort = ifelse(max(treatment, na.rm = TRUE) == 1L,
                    max(time_treated, na.rm = TRUE),
                    0L),
    .groups = "drop"
  )

df <- df %>%
  left_join(facility_cohort, by = "cms_certification_number") %>%
  mutate(
    treated_group = as.integer(treated_group),
    cohort = as.integer(cohort)
  )

# Donut sample: drop treated rows with event_time in {-3,-2,-1}
S_noant <- df %>%
  filter(treated_group == 0L | is.na(event_time) | !(event_time %in% c(-3L,-2L,-1L)))

# Controls
controls_rhs <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)

# Window + binning
EVENT_WINDOW <- c(-12L, 12L)          # start small; expand after it works
REF_P        <- -4L
BIN_REL      <- "bin::2"              # 2-month bins

# Estimate ONE outcome first (RN)
fml_rn <- as.formula(paste0(
  "rn_hppd ~ sunab(cohort, time, ref.p = ", REF_P, ", bin.rel = '", BIN_REL, "') + ",
  controls_rhs,
  " | cms_certification_number + year_month"
))

# Keep ALL controls; only window-filter treated by relative time
S_noant_w <- S_noant %>%
  mutate(rel_time = ifelse(treated_group == 1L, time - cohort, NA_integer_)) %>%
  filter(treated_group == 0L | is.na(rel_time) | (rel_time >= EVENT_WINDOW[1] & rel_time <= EVENT_WINDOW[2]))

message("[SA] Estimation rows: ", nrow(S_noant_w))
message("[SA] Treated rows in window: ", sum(S_noant_w$treated_group == 1L, na.rm = TRUE))
message("[SA] Control rows: ", sum(S_noant_w$treated_group == 0L, na.rm = TRUE))

# Run (cluster by facility first for speed)
est_rn <- feols(
  fml_rn,
  data = S_noant_w,
  vcov = ~ cms_certification_number,
  lean = TRUE
)

print(summary(est_rn, keep = "sunab\\("))

# Plot + save
png(file.path(out_plots, "sa_noant_fast_rn.png"), width = 1800, height = 1200, res = 200)
iplot(est_rn, xlim = EVENT_WINDOW,
      xlab = "Months relative to treatment",
      ylab = "RN HPPD",
      main = "SA Event Study (No anticipation) — RN (binned rel time)")
dev.off()

cat("\nDone. If this runs fast, expand EVENT_WINDOW and then add LPN/CNA/Total.\n")