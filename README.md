# KL & Selangor Apartment Rent Prediction (6001CMD)

Regression project: predict the monthly rent (RM) of apartment listings in Kuala Lumpur and Selangor
from listing details (size, rooms, location, furnishing, facilities). Metric: MAE in RM.
Models: three regression models (Random Forest Regressor, Support Vector Regressor, Gradient Boosting Regressor), tuned with cross-validation;
the two with the lowest CV error (Gradient Boosting + Random Forest) are combined in a Voting ensemble (Stacking was also tested).

## Folder layout
- `data/raw/` original CSV (never edited); `data/processed/rent_clean.csv` cleaned table
- `data/processed/rent_model_ready.csv` the same rows as the models see them (blanks filled, encoded, scaled; for inspection)
- `src/rental.py` cleaning, preprocessing, tuning and evaluation functions
- `rent_analysis.ipynb` the full analysis (EDA, preparation, models, tuning, ensemble, evaluation)
- `rent_analysis.py` the same analysis as a plain Python script
- `figures/` all plots (PNG), `results/` all tables (CSV)
- `report/submission_forms` the group report:
  - `6001CMD_GroupProject_Group6.docx` current version (Arial 12, 1.5 spacing, Word heading numbering)

  - `notes/` change log (`CHANGES_v3.md`) and round-2 notes (`NOTES_round2.md`)
  - `archive/` older drafts (v1, v2, v3)
- `archive/` older notebooks (not part of the submission)

## Extra checks added for the report (v3/v4)
These cells are in the notebook and the script; their outputs are:
- `results/ablation_extra.csv` alternative preprocessing choices (report Table 6, added rows)
- `results/ensemble_pairs.csv` every pair of tuned models as a voting ensemble (Section 6.1)
- `results/scaling.csv` fit time on growing training samples (Section 7.3)
- `results/error_by_area.csv` ensemble error by area (Table 12)
- `figures/fig_learning_curves.png` learning curves (Figure 13)

## How to run
Install the libraries once, then use either the notebook or the script. Run from the project root.
```
pip install -r requirements.txt
```
Notebook: open `rent_analysis.ipynb` in VS Code (Jupyter extension) and choose "Run All", or run it from the terminal:
```
python -m jupyter nbconvert --to notebook --execute --inplace rent_analysis.ipynb --ExecutePreprocessor.timeout=-1
```
Script (prints the tables, saves the figures):
```
python rent_analysis.py
```
Python 3.12.4 was used. A full run took about 50 minutes on 6 CPU threads before the extra checks were added; with them it takes over an hour.
The random seeds are fixed, so every run gives the same numbers and figures. Only the timing columns change.

## Data
Kaggle "Rental Pricing Dataset, Malaysia" by ariewijaya, scraped from mudah.my
(19,991 listings, 14 columns).
https://www.kaggle.com/datasets/ariewijaya/rent-pricing-kuala-lumpur-malaysi
