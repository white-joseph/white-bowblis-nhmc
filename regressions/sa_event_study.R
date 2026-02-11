# Sun–Abraham IW event study (WITHOUT anticipation only)
library(data.table)
library(fixest)

df <- fread("C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv")

# --- id/time/cohort prep ---
df[, cms_certification_number := as.character(cms_certification_number)]
df[, time := as.integer(time)]
df[, time_treated := as.integer(time_treated)]

# sunab(): never-treated cohort must be NA
df[, g_sa := fifelse(time_treated == 0L, NA_integer_, time_treated)]

# --- enforce "without anticipation" sample restriction ---
# If anticipation2 exists and equals 1 for anticipation months, keep only 0/NA.
if ("anticipation2" %in% names(df)) {
  df <- df[is.na(anticipation2) | anticipation2 == 0]
} else {
  warning("anticipation2 not found; running on full sample (cannot enforce 'without anticipation').")
}

# --- run Sun–Abraham IW event study (levels) ---
m_rn <- feols(
  rn_hppd ~ sunab(g_sa, time) | cms_certification_number + time,
  data    = df,
  cluster = ~ cms_certification_number  # or two-way: ~ cms_certification_number + t_index
)

summary(m_rn)
iplot(m_rn, ref.line = 0)

# --- optional: loop across outcomes ---
outs <- c("rn_hppd", "lpn_hppd", "cna_hppd", "total_hppd")

mods <- lapply(outs, function(y){
  feols(
    as.formula(paste0(y, " ~ sunab(g_sa, time) | cms_certification_number + time")),
    data    = df,
    cluster = ~ cms_certification_number
  )
})
names(mods) <- outs

etable(mods, se = "cluster")