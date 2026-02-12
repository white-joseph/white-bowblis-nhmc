suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(lubridate)
  library(ggplot2)
})

CHOW_FP <- "C:/Repositories/white-bowblis-nhmc/data/interim/chow.csv"
OUT_DIR <- "C:/Repositories/white-bowblis-nhmc/presentation"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ------------------------------ Plot font (Times / newtx-like) ------------------------------
# Match the style idea from your TWFE script but for ggplot objects.
set_plot_font_gg <- function() {
  fam <- "Times New Roman"
  theme_set(theme_minimal(base_size = 12, base_family = fam))
  # If Times New Roman isn't available, ggplot will typically fall back; if you want:
  # theme_set(theme_minimal(base_size = 12, base_family = "Times"))
}
set_plot_font_gg()

df <- read_csv(CHOW_FP, show_col_types = FALSE)

# -----------------------------
# Helpers
# -----------------------------
bin_0_1_2p <- function(x) {
  if (is.na(x)) return(NA_character_)
  x <- as.integer(x)
  if (x <= 0) return("0")
  if (x == 1) return("1")
  return("2+")
}

first_valid_date <- function(row_df, cols) {
  for (c in cols) {
    if (c %in% names(row_df)) {
      d <- suppressWarnings(as.Date(row_df[[c]][1]))
      if (!is.na(d)) return(d)
    }
  }
  as.Date(NA)
}

# -----------------------------
# Cross-tab (0/1/2+) + "white tiles" plot with red boxes
# -----------------------------
df <- df %>%
  mutate(
    nhc_bin = vapply(n_chow_nh_compare, bin_0_1_2p, character(1)),
    mcr_bin = vapply(n_chow_mcr,       bin_0_1_2p, character(1))
  )

ct <- df %>%
  count(nhc_bin, mcr_bin, name = "n") %>%
  mutate(
    nhc_bin = factor(nhc_bin, levels = c("0","1","2+")),
    mcr_bin = factor(mcr_bin, levels = c("0","1","2+"))
  ) %>%
  complete(nhc_bin, mcr_bin, fill = list(n = 0)) %>%
  arrange(nhc_bin, mcr_bin)

cat("\n=== Cross-tab (0/1/2+) ===\n")
print(ct %>% pivot_wider(names_from = mcr_bin, values_from = n))

rects <- tibble::tribble(
  ~xmin, ~xmax, ~ymin, ~ymax,
  0.5,   1.5,   0.5,   1.5,   # (0,0)
  1.5,   2.5,   1.5,   2.5    # (1,1)
)

p_ct <- ggplot(ct, aes(x = mcr_bin, y = nhc_bin)) +
  geom_tile(fill = "white", color = "grey70", linewidth = 0.6) +
  geom_text(aes(label = format(n, big.mark = ",")), size = 4) +
  geom_rect(
    data = rects,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = NA,
    color = "red",
    linewidth = 1.2
  ) +
  labs(
    x = "HCRIS/MCR changes (0 / 1 / 2+)",
    y = "NHC changes (0 / 1 / 2+)",
    title = "Cross-tab of Ownership Changes: NHC vs HCRIS/MCR"
  ) +
  theme(
    panel.grid = element_blank()
  )

ggsave(
  filename = file.path(OUT_DIR, "crosstab_nhc_vs_mcr_white_redboxes.pdf"),
  plot = p_ct, width = 7, height = 5, device = cairo_pdf
)

# -----------------------------
# Validated changes: exactly 1 in NHC AND 1 in MCR
# -----------------------------
date_cols_nhc <- names(df)[str_detect(names(df), "^nh_compare_chow_.*_date$")]
date_cols_mcr <- names(df)[str_detect(names(df), "^mcr_chow_.*_date$")]

if (length(date_cols_nhc) == 0) stop("No NHC date columns found (expected nh_compare_chow_*_date).")
if (length(date_cols_mcr) == 0) stop("No MCR date columns found (expected mcr_chow_*_date).")

df_11 <- df %>%
  filter(n_chow_nh_compare == 1, n_chow_mcr == 1) %>%
  mutate(row_id = row_number())

cat(sprintf("\nValidated 1x1 facilities: %s\n", format(nrow(df_11), big.mark = ",")))

df_11 <- df_11 %>%
  rowwise() %>%
  mutate(
    nhc_date = first_valid_date(cur_data(), date_cols_nhc),
    mcr_date = first_valid_date(cur_data(), date_cols_mcr),
    date_gap_days = as.integer(nhc_date - mcr_date)
  ) %>%
  ungroup()

cat("\nDate gap (NHC - MCR) in days, summary:\n")
print(summary(df_11$date_gap_days))

start <- as.Date("2017-01-01")
end   <- as.Date("2024-06-30")

df_11_win <- df_11 %>%
  filter(!is.na(nhc_date), nhc_date >= start, nhc_date <= end)

cat(sprintf(
  "\nValidated 1x1 changes within study window (by NHC date): %s\n",
  format(nrow(df_11_win), big.mark = ",")
))

# -----------------------------
# Bar chart by MONTH
# -----------------------------
df_11_win <- df_11_win %>%
  mutate(year_month = format(floor_date(nhc_date, "month"), "%Y-%m"))

all_months <- tibble(year_month = format(seq(from = floor_date(start, "month"),
                                             to   = floor_date(end, "month"),
                                             by   = "month"), "%Y-%m"))

monthly <- df_11_win %>%
  count(year_month, name = "n") %>%
  right_join(all_months, by = "year_month") %>%
  mutate(n = replace_na(n, 0L)) %>%
  arrange(year_month)

p_m <- ggplot(monthly, aes(x = year_month, y = n)) +
  geom_col() +
  labs(
    x = NULL,
    y = "# validated ownership changes",
    title = "Validated Ownership Changes by Month (NHC=1 and HCRIS/MCR=1)"
  ) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 7))

ggsave(
  filename = file.path(OUT_DIR, "validated_changes_by_month.pdf"),
  plot = p_m, width = 10, height = 4, device = cairo_pdf
)

# -----------------------------
# Bar chart by YEAR
# -----------------------------
yearly <- df_11_win %>%
  mutate(year = year(nhc_date)) %>%
  count(year, name = "n") %>%
  right_join(tibble(year = 2017:2024), by = "year") %>%
  mutate(n = replace_na(n, 0L)) %>%
  arrange(year)

p_y <- ggplot(yearly, aes(x = factor(year), y = n)) +
  geom_col() +
  labs(
    x = "Year",
    y = "# validated ownership changes",
    title = "Validated Ownership Changes by Year (NHC=1 and HCRIS/MCR=1)"
  )

ggsave(
  filename = file.path(OUT_DIR, "validated_changes_by_year.pdf"),
  plot = p_y, width = 7, height = 4, device = cairo_pdf
)

cat("\n=== Yearly validated change counts (NHC=1 & MCR=1, within window) ===\n")
print(yearly)

cat(sprintf("\nSaved figures to: %s\nDone.\n", normalizePath(OUT_DIR)))
