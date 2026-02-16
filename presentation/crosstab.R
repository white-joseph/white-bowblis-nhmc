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
theme_set(theme_minimal(base_size = 12, base_family = "Times New Roman"))

# ------------------------------ Helpers ------------------------------
normalize_ccn <- function(x) {
  s <- toupper(trimws(as.character(x)))
  s <- gsub("[[:space:]/\\.-]", "", s)  # safe on Windows
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

# ------------------------------ Load final analytical CCNs ------------------------------
anl <- read_csv(ANL_FP, show_col_types = FALSE)
stopifnot("cms_certification_number" %in% names(anl))

final_ccns <- anl %>%
  transmute(ccn = normalize_ccn(cms_certification_number)) %>%
  filter(!is.na(ccn)) %>%
  distinct(ccn) %>%
  pull(ccn)

n_final <- length(final_ccns)
cat(sprintf("[final sample] unique CCNs in analytical panel: %s\n", format(n_final, big.mark=",")))

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
  distinct(ccn, .keep_all = TRUE)  # one row per CCN

# sanity: how many chow CCNs are in final panel?
cat(sprintf("[overlap] chow CCNs that appear in analytical panel: %s\n",
            format(sum(chow_bins$ccn %in% final_ccns), big.mark=",")))

# ------------------------------
# Hybrid counts:
#   - (0,0) and (1,1): count ONLY among final analytical CCNs
#   - everything else: count among CCNs NOT in analytical panel
# ------------------------------
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
    # default: use non-panel counts
    n = n_nonpanel,
    # override for the two “care about” cells using final analytical-panel counts
    n = ifelse(nhc_bin == "0" & mcr_bin == "0", n_panel, n),
    n = ifelse(nhc_bin == "1" & mcr_bin == "1", n_panel, n),
    nhc_bin = factor(nhc_bin, levels = c("0","1","2+")),
    mcr_bin = factor(mcr_bin, levels = c("0","1","2+"))
  ) %>%
  complete(nhc_bin, mcr_bin, fill = list(n = 0L, n_panel = 0L, n_nonpanel = 0L)) %>%
  arrange(nhc_bin, mcr_bin)

cat("\n=== Hybrid cross-tab counts (panel for 0/0 & 1/1; chow for all others) ===\n")
print(ct %>% select(nhc_bin, mcr_bin, n) %>% pivot_wider(names_from = mcr_bin, values_from = n))

# Check that panel CCNs are only in 0/0 or 1/1 (should be true given your pipeline)
panel_cells <- panel_part %>% mutate(cell = paste0(nhc_bin, "-", mcr_bin))
if (any(!(panel_cells$cell %in% c("0-0","1-1")))) {
  cat("\n[warning] Some final-panel CCNs are not in (0,0) or (1,1) by chow bins.\n")
  print(panel_part)
}

# ------------------------------ Plot styling ------------------------------
rects <- tibble::tribble(
  ~xmin, ~xmax, ~ymin, ~ymax,
  0.5,   1.5,   0.5,   1.5,   # (0,0)
  1.5,   2.5,   1.5,   2.5    # (1,1)
)

p_ct <- ggplot(ct, aes(x = mcr_bin, y = nhc_bin)) +
  geom_tile(fill = "white", color = "black", linewidth = 0.6) +
  geom_text(aes(label = format(n, big.mark=",")), size = 4) +
  geom_rect(
    data = rects,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = NA,
    color = "red",
    linewidth = 1.2
  ) +
  labs(
    x = "Count of changes in HCRIS",
    y = "Count of changes in NHC",
    title = "Cross-tab of Ownership Changes: NHC vs HCRIS"
  ) +
  theme(
    panel.grid = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )

ggsave(
  filename = file.path(OUT_DIR, "crosstab_hybrid_panel00_11_chow_rest.pdf"),
  plot = p_ct,
  width = 7, height = 5,
  device = cairo_pdf
)

cat(sprintf("\nSaved: %s\n", normalizePath(file.path(OUT_DIR, "crosstab_hybrid_panel00_11_chow_rest.pdf"))))