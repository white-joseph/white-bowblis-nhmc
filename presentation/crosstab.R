# C:/Repositories/white-bowblis-nhmc/presentation/crosstab.R

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

CHOW_FP <- "C:/Repositories/white-bowblis-nhmc/data/interim/chow.csv"
ANL_FP  <- "C:/Repositories/white-bowblis-nhmc/data/clean/analytical_panel.csv"
OUT_DIR <- "C:/Repositories/white-bowblis-nhmc/presentation"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ------------------------------ Plot font (Times / newtx-like) ------------------------------
theme_set(theme_minimal(base_size = 18, base_family = "Times New Roman"))

# ------------------------------ Helpers ------------------------------
normalize_ccn <- function(x) {
  s <- toupper(trimws(as.character(x)))
  s <- gsub("[[:space:]/\\.-]", "", s)
  is_digits <- grepl("^\\d+$", s)
  s[is_digits] <- sprintf("%06d", as.integer(s[is_digits]))
  s[s == ""] <- NA_character_
  s
}

bin_0_1_2p <- function(x) {
  if (is.na(x)) return(NA_character_)
  x <- as.integer(x)
  if (x <= 0) return("0")
  if (x == 1) return("1")
  return("2+")
}

pct_str <- function(num, den, digits = 1) {
  out <- rep("(NA)", length(num))
  ok <- !is.na(num) & !is.na(den) & den != 0
  out[ok] <- paste0(
    "(",
    format(round(100 * num[ok] / den, digits), nsmall = digits),
    "%)"
  )
  out
}

# left-pad a character vector to a fixed width (for visual centering in proportional fonts)
lpad <- function(x, width) {
  x <- as.character(x)
  pad_n <- pmax(0, width - nchar(x, type = "chars"))
  paste0(strrep(" ", pad_n), x)
}

# ------------------------------ Load final analytical CCNs ------------------------------
anl <- read_csv(ANL_FP, show_col_types = FALSE)
stopifnot("cms_certification_number" %in% names(anl))

final_ccns <- anl %>%
  transmute(ccn = normalize_ccn(cms_certification_number)) %>%
  filter(!is.na(ccn)) %>%
  distinct(ccn) %>%
  pull(ccn)

cat(sprintf("[final sample] unique CCNs in analytical panel: %s\n",
            format(length(final_ccns), big.mark=",")))

# ------------------------------ Load chow, build bins (ALL CCNs) ------------------------------
chow <- read_csv(CHOW_FP, show_col_types = FALSE) %>%
  mutate(ccn = normalize_ccn(cms_certification_number))

need <- c("n_chow_nh_compare", "n_chow_mcr")
miss <- setdiff(need, names(chow))
if (length(miss) > 0) stop("chow.csv missing: ", paste(miss, collapse=", "))

chow_bins <- chow %>%
  filter(!is.na(ccn)) %>%
  transmute(
    ccn,
    n_nhc = suppressWarnings(as.integer(n_chow_nh_compare)),
    n_mcr = suppressWarnings(as.integer(n_chow_mcr)),
    nhc_bin = vapply(n_nhc, bin_0_1_2p, character(1)),
    mcr_bin = vapply(n_mcr, bin_0_1_2p, character(1))
  ) %>%
  distinct(ccn, .keep_all = TRUE)

cat(sprintf("[overlap] chow CCNs that appear in analytical panel: %s\n",
            format(sum(chow_bins$ccn %in% final_ccns), big.mark=",")))

# ------------------------------ Hybrid counts ------------------------------
panel_part <- chow_bins %>%
  filter(ccn %in% final_ccns) %>%
  count(nhc_bin, mcr_bin, name = "n_panel")

nonpanel_part <- chow_bins %>%
  filter(!(ccn %in% final_ccns)) %>%
  count(nhc_bin, mcr_bin, name = "n_nonpanel")

ct <- full_join(panel_part, nonpanel_part, by = c("nhc_bin","mcr_bin")) %>%
  mutate(
    n_panel    = replace_na(n_panel, 0L),
    n_nonpanel = replace_na(n_nonpanel, 0L),
    n = n_nonpanel,
    n = ifelse(nhc_bin == "0" & mcr_bin == "0", n_panel, n),
    n = ifelse(nhc_bin == "1" & mcr_bin == "1", n_panel, n),
    nhc_bin = factor(nhc_bin, levels = c("0","1","2+")),
    mcr_bin = factor(mcr_bin, levels = c("0","1","2+"))
  ) %>%
  complete(nhc_bin, mcr_bin, fill = list(n = 0L, n_panel = 0L, n_nonpanel = 0L)) %>%
  arrange(nhc_bin, mcr_bin)

cat("\n=== Hybrid cross-tab counts (panel for 0/0 & 1/1; chow for all others) ===\n")
print(ct %>% select(nhc_bin, mcr_bin, n) %>% pivot_wider(names_from = mcr_bin, values_from = n))

# ------------------------------ Label components ------------------------------
total_all_cells <- sum(ct$n, na.rm = TRUE)

ct <- ct %>%
  mutate(
    n_fmt_raw   = format(n, big.mark = ","),
    pct_fmt_raw = pct_str(n, total_all_cells, digits = 1)
  )

# compute a consistent label width from the longest COUNT string
LABEL_WIDTH <- max(nchar(ct$n_fmt_raw, type = "chars"), na.rm = TRUE)

ct <- ct %>%
  mutate(
    n_fmt   = lpad(n_fmt_raw, LABEL_WIDTH),
    pct_fmt = lpad(pct_fmt_raw, LABEL_WIDTH)
  )

# ------------------------------ Plot styling ------------------------------
rects <- tibble::tribble(
  ~xmin, ~xmax, ~ymin, ~ymax,
  0.5,   1.5,   0.5,   1.5,   # (0,0)
  1.5,   2.5,   1.5,   2.5    # (1,1)
)

COUNT_SIZE <- 7.0
PCT_SIZE   <- 4.3

p_ct <- ggplot(ct, aes(x = mcr_bin, y = nhc_bin)) +
  geom_tile(fill = "white", color = "black", linewidth = 0.7) +
  
  # Count: above center
  geom_text(
    aes(label = n_fmt),
    size = COUNT_SIZE,
    nudge_y = +0.12
  ) +
  
  # Percent: below center
  geom_text(
    aes(label = pct_fmt),
    size = PCT_SIZE,
    nudge_y = -0.16
  ) +
  
  geom_rect(
    data = rects,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = NA,
    color = "red",
    linewidth = 1.4
  ) +
  labs(
    x = "Count of changes in HCRIS",
    y = "Count of changes in NHC",
    title = "Cross-tab of Ownership Changes: NHC vs HCRIS"
  ) +
  theme(
    panel.grid = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 22),
    axis.title = element_text(size = 18),
    axis.text  = element_text(size = 16)
  )

out_pdf <- file.path(OUT_DIR, "crosstab_hybrid_panel00_11_chow_rest.pdf")
out_png <- file.path(OUT_DIR, "crosstab_hybrid_panel00_11_chow_rest.png")

ggsave(filename = out_pdf, plot = p_ct, width = 7.8, height = 5.6, device = cairo_pdf)
ggsave(filename = out_png, plot = p_ct, width = 7.8, height = 5.6, dpi = 300)

cat(sprintf("\nSaved:\n- %s\n- %s\n",
            normalizePath(out_pdf),
            normalizePath(out_png)))