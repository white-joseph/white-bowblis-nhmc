# C:/Repositories/white-bowblis-nhmc/presentation/presentation_visuals.R
# PURPOSE:
#   Plot validated ownership changes over time aggregated to QUARTERS.
#
# OUTPUT:
#   C:/Repositories/white-bowblis-nhmc/presentation/validated_changes_by_quarter.pdf
#   C:/Repositories/white-bowblis-nhmc/presentation/validated_changes_by_quarter.png

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(lubridate)
  library(ggplot2)
})

CHOW_FP <- "C:/Repositories/white-bowblis-nhmc/data/interim/chow.csv"
OUT_DIR <- "C:/Repositories/white-bowblis-nhmc/presentation"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

theme_set(theme_minimal(base_size = 14, base_family = "Times New Roman"))

df <- read_csv(CHOW_FP, show_col_types = FALSE)

# -----------------------------
# Helper: first valid Date across candidate columns for a single row (as a named list)
# -----------------------------
first_valid_date_row <- function(row_list, cols) {
  for (c in cols) {
    if (!is.null(row_list[[c]])) {
      d <- suppressWarnings(as.Date(row_list[[c]]))
      if (!is.na(d)) return(d)
    }
  }
  as.Date(NA)
}

# -----------------------------
# Validated changes: exactly 1 in NHC AND 1 in MCR
# -----------------------------
date_cols_nhc <- names(df)[str_detect(names(df), "^nh_compare_chow_.*_date$")]
date_cols_mcr <- names(df)[str_detect(names(df), "^mcr_chow_.*_date$")]

if (length(date_cols_nhc) == 0) stop("No NHC date columns found (expected nh_compare_chow_*_date).")
if (length(date_cols_mcr) == 0) stop("No MCR date columns found (expected mcr_chow_*_date).")

df_11 <- df %>%
  filter(n_chow_nh_compare == 1, n_chow_mcr == 1)

cat(sprintf("\nValidated 1x1 facilities: %s\n", format(nrow(df_11), big.mark = ",")))

# Compute dates without rowwise()/cur_data()
rows <- split(df_11, seq_len(nrow(df_11)))
nhc_dates <- as.Date(vapply(rows, first_valid_date_row, as.Date(NA), cols = date_cols_nhc))
mcr_dates <- as.Date(vapply(rows, first_valid_date_row, as.Date(NA), cols = date_cols_mcr))

df_11 <- df_11 %>%
  mutate(
    nhc_date = nhc_dates,
    mcr_date = mcr_dates,
    date_gap_days = as.integer(nhc_date - mcr_date)
  )

cat("\nDate gap (NHC - MCR) in days, summary:\n")
print(summary(df_11$date_gap_days))

# Study window
start <- as.Date("2017-01-01")
end   <- as.Date("2024-06-30")

df_11_win <- df_11 %>%
  filter(!is.na(nhc_date), nhc_date >= start, nhc_date <= end)

cat(sprintf(
  "\nValidated 1x1 changes within study window (by NHC date): %s\n",
  format(nrow(df_11_win), big.mark = ",")
))

# -----------------------------
# Aggregate to QUARTER
# -----------------------------
df_11_win <- df_11_win %>%
  mutate(qtr_date = floor_date(nhc_date, unit = "quarter"))

all_quarters <- tibble(
  qtr_date = seq(from = floor_date(start, "quarter"),
                 to   = floor_date(end, "quarter"),
                 by   = "quarter")
)

quarterly <- df_11_win %>%
  count(qtr_date, name = "n") %>%
  right_join(all_quarters, by = "qtr_date") %>%
  mutate(n = ifelse(is.na(n), 0L, n)) %>%   # no replace_na() needed
  arrange(qtr_date) %>%
  mutate(qtr_label = paste0(year(qtr_date), " Q", quarter(qtr_date)))

cat("\n=== Quarterly validated change counts (NHC=1 & MCR=1, within window) ===\n")
print(quarterly %>% select(qtr_label, n))

# -----------------------------
# Plot
# -----------------------------
p_q <- ggplot(quarterly, aes(x = qtr_date, y = n)) +
  geom_col() +
  scale_x_date(
    date_breaks = "1 year",
    date_labels = "%Y"
  ) +
  labs(
    x = NULL,
    y = "Number of Ownership Changes",
    title = "Ownership Changes by Quarter"
  ) +
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 11),
    plot.title = element_text(hjust = 0.45)
  )

out_pdf <- file.path(OUT_DIR, "validated_changes_by_quarter.pdf")
out_png <- file.path(OUT_DIR, "validated_changes_by_quarter.png")

ggsave(filename = out_pdf, plot = p_q, width = 10, height = 4.2, device = cairo_pdf)
ggsave(filename = out_png, plot = p_q, width = 10, height = 4.2, dpi = 300)

cat(sprintf("\nSaved:\n- %s\n- %s\nDone.\n",
            normalizePath(out_pdf),
            normalizePath(out_png)))