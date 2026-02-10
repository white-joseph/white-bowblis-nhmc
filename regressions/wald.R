# Console-only Wald pretrend tests with event-time window [-12, 12]
# Panel A: With anticipation (ref typically -1) -> test taus -12..-2
# Panel B: Without anticipation II (drop -3,-2,-1; ref typically -4) -> test taus -12..-5
#
# Outputs: printed to console only

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
  library(MASS)  # ginv
})

options(scipen = 999, digits = 4)

# ------------------------------ Paths ------------------------------
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"

# ------------------------------ Load + prep ------------------------------
keep_cols <- c(
  "cms_certification_number","year_month","event_time","treatment",
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

# Ever-treated & cap event_time to [-12, 12]
WIN <- 12L

df <- df %>%
  group_by(cms_certification_number) %>%
  mutate(ever_treated = as.integer(any(treatment == 1, na.rm = TRUE) | any(!is.na(event_time)))) %>%
  ungroup() %>%
  mutate(
    event_time_capped = dplyr::case_when(
      ever_treated == 1L & !is.na(event_time) ~ pmin(pmax(as.integer(event_time), -WIN), WIN),
      TRUE ~ 9999L
    )
  )

# Controls RHS
controls_rhs <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)

# Helper: pick a valid reference that exists in the treated event-time support
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

# Event study TWFE (levels)
run_es_twfe <- function(lhs, data, ref_val, win = WIN) {
  fml <- as.formula(paste0(
    lhs, " ~ i(event_time_capped, ever_treated, ref = ", ref_val, ", keep = -", win, ":", win, ") + ",
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

# Pick ES coefficient names and taus from a fixest model
.es_pick <- function(mod, var = "event_time_capped", trt = "ever_treated") {
  cn <- names(coef(mod))
  if (is.null(cn) || !length(cn)) return(list(names = character(0), taus = integer(0)))
  pat <- sprintf("^%s::[-]?[0-9]+:%s$", var, trt)
  es_names <- grep(pat, cn, value = TRUE)
  get_tau <- function(s) as.integer(regmatches(s, regexpr("-?[0-9]+", s)))
  taus <- vapply(es_names, get_tau, integer(1))
  names(taus) <- es_names
  list(names = es_names, taus = taus)
}

# Joint Wald test: H0 all pre-treatment betas in [from,to] are zero (excluding ref)
pretrend_wald <- function(mod, ref_tau, from, to,
                          var = "event_time_capped", trt = "ever_treated") {
  if (is.null(mod)) return(list(note = "Model is NULL"))
  es <- .es_pick(mod, var, trt)
  if (!length(es$names)) return(list(note = "No ES coefficients found"))
  
  pre_idx <- es$taus < 0L & es$taus != ref_tau & es$taus >= from & es$taus <= to
  pre_names <- names(es$taus)[pre_idx]
  if (!length(pre_names)) return(list(note = "No preperiod coefficients in window"))
  
  b <- coef(mod)[pre_names]
  V <- vcov(mod)[pre_names, pre_names, drop = FALSE]
  
  W <- as.numeric(t(b) %*% MASS::ginv(V) %*% b)
  df_w <- qr(V)$rank
  pval <- pchisq(W, df = df_w, lower.tail = FALSE)
  
  list(statistic = W, df = df_w, p.value = pval, window = c(from, to))
}

# Pretty printer
print_panel <- function(panel_name, data, ref, test_from, test_to) {
  outcomes <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")
  nice <- c(rn_hppd="RN", lpn_hppd="LPN", cna_hppd="CNA", total_hppd="Total")
  
  cat("\n============================================================\n")
  cat(panel_name, "\n")
  cat("Event-time window kept: [", -WIN, ", ", WIN, "]\n", sep = "")
  cat("Reference period: tau = ", ref, "\n", sep = "")
  cat("Joint pretrend window tested: tau = ", test_from, " .. ", test_to, "\n", sep = "")
  cat("N rows: ", format(nrow(data), big.mark = ","), "\n", sep = "")
  cat("============================================================\n")
  
  for (y in outcomes) {
    mod <- tryCatch(run_es_twfe(y, data, ref), error = function(e) NULL)
    res <- pretrend_wald(mod, ref_tau = ref, from = test_from, to = test_to)
    
    if (!is.null(res$note)) {
      cat(sprintf("%-6s  %s\n", nice[[y]], paste0("NA (", res$note, ")")))
    } else {
      cat(sprintf(
        "%-6s  Wald Chi^2 = %8.2f  df = %2d  p = %.4f\n",
        nice[[y]], res$statistic, res$df, res$p.value
      ))
    }
  }
}

# ------------------------------ Panels ------------------------------
# Panel A: With anticipation
dat_with <- df
ref_with <- pick_ref(dat_with, desired = -1L)
# With anticipation: test -12..-2 (exclude ref -1)
win_with <- c(-WIN, -2L)

# Panel B: Without anticipation II
skip2 <- c(-3L,-2L,-1L)
dat_wo2 <- df %>% filter(!(ever_treated == 1L & event_time_capped %in% skip2))
ref_wo2 <- pick_ref(dat_wo2, desired = -4L)
# Without anticipation: ref -4; test -12..-5 (since -3,-2,-1 dropped)
win_wo2 <- c(-WIN, -5L)

# ------------------------------ Run + print ------------------------------
print_panel("Panel A: With anticipation", dat_with, ref_with, win_with[1], win_with[2])
print_panel("Panel B: Without anticipation (drop tau = -3,-2,-1)", dat_wo2, ref_wo2, win_wo2[1], win_wo2[2])

cat("\nDone.\n")
