# Ownership Changes in Skilled Nursing Facilities

Code and paper for "Measuring Ownership Changes and Operating Responses in Skilled Nursing Facilities" (Joseph White, John Bowblis, Charles C. Moul).

The project identifies substantive ownership changes in U.S. skilled nursing facilities (2017–2024) by verifying cost-report (HCRIS) change dates against monthly Nursing Home Compare ownership records. It then estimates how staffing, resident census, payer mix, occupancy, length of stay, and quality change after a transfer using staggered event-study and two-way fixed effects designs.

## Repository structure

| Folder | Contents |
|---|---|
| `src/` | Python pipeline: cleans raw CMS data and builds the analysis panels |
| `regressions/` | R scripts: estimation, tables, and figures |
| `data/` | `raw/` (not tracked), `interim/`, `clean/` |
| `outputs/` | Generated tables (`.tex`) and figures (`.pdf`) |
| `paper/` | Thesis and writing sample (LaTeX) |

## Data

All data are public CMS files and are not stored in the repository:

- Payroll-Based Journal (PBJ) daily nurse staffing
- Healthcare Cost Report Information System (HCRIS) Medicare cost reports
- Nursing Home Compare archive (ownership, provider information, quality measures)

Place raw files in `data/raw/` or set the `NH_DATA_DIR` environment variable. Expected subfolders are defined in `src/config.py`.

## Replication

1. **Build the panels (Python).** Run `src/run_pipeline.ipynb`, or run the scripts in order: `01`–`06`, then `quality-metrics.py` and `quarterly_panel.py`. Output: `data/clean/staffing_panel.csv` and `data/clean/quality_panel.csv`.
2. **Estimate (R).** Run `regressions/run_all.R`. Tables and figures are written to `outputs/`.
3. **Compile the paper.** Build `paper/ma_thesis.tex` with XeLaTeX and BibTeX.

R scripts use an absolute project path (`project_root` in `regressions/_setup.R` and the paths in `run_all.R`); update these if the repository is not at `C:/Repositories/white-bowblis-nhmc`.

## Requirements

- **Python 3:** pandas, numpy, pyreadstat
- **R:** fixest, dplyr, readr, stringr, tibble, purrr, ggplot2, MASS
- **LaTeX:** XeLaTeX distribution with BibTeX
