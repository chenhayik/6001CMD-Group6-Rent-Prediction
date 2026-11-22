"""Plain Python version of rent_analysis.ipynb. Runs the whole analysis (parts B to G).

Run from the project folder:  python rent_analysis.py
Figures are saved to figures/ and tables to results/. A full run takes over an hour (the extra checks added for the report make it longer than the original 50 minutes).
"""

# ===== Rent Prediction for KL & Selangor Apartments (6001CMD) =====
# ===== A. Problem Definition =====
# Predict `monthly_rent` (RM per month) of apartment listings in Kuala Lumpur and Selangor from the listing details
# (size, rooms, location, furnishing, facilities). This is supervised regression and the business metric is **MAE in
# RM**; RMSE and R² are reported as well.
# Data: Kaggle "Rent Pricing Kuala Lumpur Malaysia" (ariewijaya), scraped from mudah.my. The code lives in
# `src/rental.py`.

# What:   Imports, plot style and output folders for the whole analysis.
# Input:  Helper functions from src/rental.py.
# Output: figures/, results/ and data/processed/ folders; RAW path and BLUE colour.
# How:    Adds src to the import path, sets one seaborn theme, creates the folders if missing.
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, train_test_split

sys.path.insert(0, 'src')
from rental import (BASE_MODELS, CAT, CV, NUM, SCORING, build_preprocessor, evaluate,
                    load_and_clean, make_model, parse, tune)

sns.set_theme(style='whitegrid', context='notebook', font_scale=1.0)
plt.rcParams.update({'savefig.dpi': 150, 'savefig.bbox': 'tight'})
for folder in ['figures', 'results', 'data/processed']:
    Path(folder).mkdir(parents=True, exist_ok=True)

RAW = 'data/raw/mudah-apartment-kl-selangor.csv'
BLUE = "steelblue"
pd.set_option('display.width', 200, 'display.max_columns', 30)   # print full tables in the terminal

# ===== B. Exploratory Data Analysis =====

# What:   First look at the raw data.
# Input:  Raw CSV (RAW), parsed by parse() in src/rental.py.
# Output: Printed row count, repost count and a table of type, unique values and example per column.
# How:    Counts rows that are identical when ads_id is ignored (the same listing posted again).
orig = pd.read_csv(RAW)   # untouched raw columns
raw = parse(orig)         # rent, size and rooms as numbers, nothing removed
print('Rows, columns:', raw.shape)
print('Exact duplicate listings in the raw data (ignoring ads_id):', raw.drop(columns='ads_id').duplicated().sum())

print(pd.DataFrame({'dtype': orig.dtypes.astype(str), 'n_unique': orig.nunique(),
              'example': orig.apply(lambda c: c.dropna().iloc[0])}).rename_axis('column'))

# What:   Descriptive statistics of the numeric columns.
# Input:  Parsed raw data (raw).
# Output: results/descriptive_stats.csv, also printed.
# How:    pandas describe() plus skewness.
# Notes:  Run on the raw data on purpose, so impossible values show up.
num_cols = ['monthly_rent', 'size_sqft', 'rooms', 'bathroom', 'parking', 'completion_year']
stats = raw[num_cols].describe().T.assign(skew=raw[num_cols].skew()).round(2)
stats.to_csv('results/descriptive_stats.csv')
print(stats)

# What:   Missing value analysis.
# Input:  Untouched raw columns (orig).
# Output: results/missing.csv and figures/fig_missing.png.
# How:    Percentage of empty cells per column, sorted from most to least missing.
missing = (orig.isna().mean() * 100).round(2).sort_values(ascending=False).rename('missing_pct').rename_axis('column')
missing.to_csv('results/missing.csv')

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=missing.values, y=missing.index, color=BLUE, ax=ax)
ax.bar_label(ax.containers[0], fmt='%.2f%%', padding=3)
ax.set(title='Missing values per column', xlabel='Missing (% of listings)', ylabel='Column', xlim=(0, 55))
plt.savefig('figures/fig_missing.png')
plt.close()

# What:   Distribution of the target (monthly rent), before and after a log transform.
# Input:  Parsed raw rent (raw['monthly_rent']).
# Output: figures/fig_rent_distribution.png.
# How:    Two histograms; values above the 99th percentile are left out so the main part is readable.
# Notes:  Rent is right-skewed and much more symmetric on the log scale, so the models train on log1p(rent).
rent = raw['monthly_rent'].dropna()
cap = rent.quantile(0.99)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(rent[rent <= cap], bins=50, color=BLUE, ax=axes[0])
axes[0].set(title=f'Monthly rent, up to the 99th percentile (RM {cap:,.0f})', xlabel='Monthly rent (RM)', ylabel='Listings')
sns.histplot(np.log1p(rent[rent <= cap]), bins=50, color=BLUE, ax=axes[1])
axes[1].set(title='log(1 + monthly rent), up to the 99th percentile', xlabel='log(1 + rent in RM)', ylabel='Listings')
plt.savefig('figures/fig_rent_distribution.png')
plt.close()

# What:   Outlier analysis for rent and size.
# Input:  Parsed raw data (raw).
# Output: results/outliers.csv and figures/fig_outliers_boxplot.png.
# How:    Counts values outside the 1.5 x IQR fences and outside the domain range, then draws log-scale boxplots.
# Notes:  The IQR rule flags many real luxury units, so cleaning uses the domain range instead.
# IQR rule vs the domain range used for cleaning (rent RM 300-10,000, size 200-5,000 sq.ft.)
rows = []
for col, low, high in [('monthly_rent', 300, 10_000), ('size_sqft', 200, 5_000)]:
    v = raw[col].dropna()
    q1, q3 = v.quantile([0.25, 0.75])
    fence = 1.5 * (q3 - q1)
    rows.append({'variable': col, 'iqr_outliers': int(((v < q1 - fence) | (v > q3 + fence)).sum()),
                 'outside_domain_range': int(((v < low) | (v > high)).sum()),
                 'domain_range': f'{low:,} - {high:,}', 'min': v.min(), 'max': v.max()})
outliers = pd.DataFrame(rows)
outliers.to_csv('results/outliers.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(y=raw['monthly_rent'], color=BLUE, ax=axes[0])
axes[0].set(title='Monthly rent (log scale)', ylabel='Monthly rent (RM, log scale)', yscale='log')
sns.boxplot(y=raw['size_sqft'], color=BLUE, ax=axes[1])
axes[1].set(title='Size (log scale)', ylabel='Size (sq.ft., log scale)', yscale='log')
plt.savefig('figures/fig_outliers_boxplot.png')
plt.close()
print(outliers)

# What:   Cleaned dataset and a summary of the categorical features.
# Input:  Raw CSV via load_and_clean() in src/rental.py.
# Output: df (clean table), log (rows left after each step), fac_cols (facility flag names); printed summary.
# How:    describe() gives the number of categories and the most common one per text column.
# Notes:  The rest of the EDA uses the cleaned data.
df, log = load_and_clean(RAW)
fac_cols = [c for c in df if c.startswith('f_')]
print(df[CAT].describe().T)

# What:   Distributions of the numeric features.
# Input:  Cleaned data (df[NUM]).
# Output: figures/fig_numeric_distributions.png.
# How:    One pandas histogram per numeric column.
axes = df[NUM].hist(bins=30, figsize=(13, 7), color=BLUE)
for ax in axes.flat:
    ax.set_ylabel('Listings')
plt.suptitle('Distributions of the numeric features (cleaned data)')
plt.tight_layout()
plt.savefig('figures/fig_numeric_distributions.png')
plt.close()

# What:   Correlation between the numeric features and rent.
# Input:  Cleaned data (df).
# Output: results/correlation.csv and figures/fig_correlation.png.
# How:    Spearman rank correlation shown as a heatmap.
corr = df[NUM + ['monthly_rent']].corr(method='spearman').round(2)   # Spearman because rent and size are skewed
corr.to_csv('results/correlation.csv')

fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={'size': 12}, ax=ax)
ax.set_title('Spearman correlation of numeric features and rent')
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.savefig('figures/fig_correlation.png')
plt.close()

# What:   Rent by property type and by furnishing.
# Input:  Cleaned data (df).
# Output: figures/fig_rent_by_category.png.
# How:    Boxplots for the 6 most common property types and the 3 furnishing levels, extreme points hidden.
top_types = df['property_type'].value_counts().index[:6]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={'width_ratios': [3, 2]})
sns.boxplot(data=df[df['property_type'].isin(top_types)], x='property_type', y='monthly_rent', order=top_types,
            showfliers=False, color=BLUE, ax=axes[0])
axes[0].set(title='Rent by property type (6 most common)', xlabel='Property type', ylabel='Monthly rent (RM)')
axes[0].tick_params(axis='x', rotation=20)
sns.boxplot(data=df, x='furnished', y='monthly_rent', order=['Not Furnished', 'Partially Furnished', 'Fully Furnished'],
            showfliers=False, color=BLUE, ax=axes[1])
axes[1].set(title='Rent by furnishing', xlabel='Furnishing', ylabel='Monthly rent (RM)')
plt.savefig('figures/fig_rent_by_category.png')
plt.close()

# What:   Median rent of the 15 most listed areas.
# Input:  Cleaned data (df).
# Output: figures/fig_rent_by_area.png.
# How:    Groups by area, takes the median rent and sorts the bars.
top_areas = df['area'].value_counts().index[:15]
area_median = df[df['area'].isin(top_areas)].groupby('area')['monthly_rent'].median().sort_values()

fig, ax = plt.subplots(figsize=(8, 6.5))
area_median.plot.barh(color=BLUE, ax=ax)
ax.bar_label(ax.containers[0], fmt='%.0f', padding=3)
ax.set(title='Median rent of the 15 most listed areas', xlabel='Median monthly rent (RM)', ylabel='Area', xlim=(0, area_median.max() * 1.1))
plt.savefig('figures/fig_rent_by_area.png')
plt.close()

# What:   Rent difference linked to each facility.
# Input:  Cleaned data (df) and the facility flags (fac_cols).
# Output: figures/fig_facilities.png.
# How:    Median rent of listings with the facility minus median rent of listings without it.
# Notes:  Facilities come with certain property types, so this is an association, not the effect of adding one.
effect = pd.Series({f.removeprefix('f_'): df.groupby(f)['monthly_rent'].median().diff().iloc[-1] for f in fac_cols}).sort_values()

fig, ax = plt.subplots(figsize=(8, 7))
effect.plot.barh(color=BLUE, ax=ax)
ax.set(title='Median rent with vs without each facility', xlabel='Median rent difference (RM, with minus without)', ylabel='Facility')
plt.savefig('figures/fig_facilities.png')
plt.close()

# ===== C. Data Preparation =====

# What:   Cleaning log, processed dataset and the train/test split.
# Input:  df and log from load_and_clean().
# Output: results/cleaning_log.csv, data/processed/rent_clean.csv, X_train, X_test, y_train, y_test.
# How:    Rows removed per step = difference between steps; random 80/20 split with a fixed seed.
# Notes:  Duplicates were dropped before the split, so no listing is in both parts.
cleaning = pd.Series(log, name='rows').rename_axis('step').to_frame()
cleaning['removed'] = (cleaning['rows'].shift() - cleaning['rows']).fillna(0).astype(int)
cleaning.to_csv('results/cleaning_log.csv')
df.to_csv('data/processed/rent_clean.csv', index=False)

X, y = df.drop(columns='monthly_rent'), df['monthly_rent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train:', X_train.shape, ' Test:', X_test.shape, ' Facility flags:', len(fac_cols))
print(f'{X_train.shape[1]} input columns -> {build_preprocessor().fit_transform(X_train).shape[1]} features after preprocessing')
print(cleaning)

# What:   Model-ready copy of the data, as the models see it.
# Input:  X, y and the training rows (X_train); build_preprocessor() in src/rental.py.
# Output: data/processed/rent_model_ready.csv (119 feature columns, monthly_rent and a train/test label).
# How:    Fits the preprocessing on the training rows, applies it to all rows and marks each row train or test.
# Notes:  For inspection only. Each model repeats these steps inside its own pipeline.
# The same data as the models see it: blanks filled, "was missing" flags added, text one-hot encoded, numbers scaled.
# Fitted on the training rows only, then applied to all rows. Rent stays in RM; the log is applied inside the model.
prep = build_preprocessor().fit(X_train)
ready = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out()).round(4)
ready['monthly_rent'] = y
ready['split'] = np.where(X.index.isin(X_train.index), 'train', 'test')
ready.to_csv('data/processed/rent_model_ready.csv', index=False)
print('Model-ready table:', ready.shape)
print(ready.head())

# ===== Preprocessing ablation =====

# What:   Test of how much each preprocessing step matters.
# Input:  Training data and the three default models from BASE_MODELS in src/rental.py.
# Output: results/ablation.csv and figures/fig_ablation.png.
# How:    5-fold CV MAE of each model with one step removed at a time, next to a median-rent dummy baseline.
configs = {'Full pipeline': (X_train, {}),
           'No log target': (X_train, {'log_target': False}),
           'No scaling': (X_train, {'scale': False}),
           'No facility flags': (X_train[NUM + CAT], {}),
           'No area feature': (X_train.drop(columns='area'), {})}

ablation = pd.DataFrame(index=pd.Index(list(configs), name='config'), columns=list(BASE_MODELS), dtype=float)
for config, (data, options) in configs.items():
    for name, (estimator, _, _) in BASE_MODELS.items():
        scores = cross_val_score(make_model(estimator, **options), data, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
        ablation.loc[config, name] = -scores.mean()

dummy_cv = -cross_val_score(DummyRegressor(strategy='median'), X_train, y_train, cv=CV, scoring=SCORING)
ablation.loc['Dummy (median)'] = dummy_cv.mean()
ablation = ablation.round(2)
ablation.to_csv('results/ablation.csv')

ax = ablation.iloc[:-1].plot.bar(figsize=(11, 5.5), rot=15, width=0.8)
ax.axhline(ablation.loc['Dummy (median)'].iloc[0], color='grey', ls='--', label='Dummy (median) baseline')
ax.set(title='Preprocessing ablation, 5-fold CV MAE (lower is better)', xlabel='Pipeline configuration', ylabel='CV MAE (RM)')
ax.legend(title='Model', loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('figures/fig_ablation.png')
plt.close()
print(ablation)

# What:   Extra preprocessing checks (report Table 6, added rows).
# Input:  Training data and the three default base models.
# Output: results/ablation_extra.csv
# How:    Switches one choice at a time (no missing flag, no rare-area grouping, complete cases only,
#         Yeo-Johnson / Box-Cox target, duplicates kept) and scores each with the same 5-fold CV.
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler


def variant_model(estimator, indicator=True, min_frequency=20, target='log'):
    """Same pipeline as make_model, with one choice switched."""
    num = make_pipeline(SimpleImputer(strategy='median', add_indicator=indicator), StandardScaler())
    onehot = OneHotEncoder(handle_unknown='infrequent_if_exist' if min_frequency else 'ignore',
                           min_frequency=min_frequency, sparse_output=False)
    prep = ColumnTransformer([('num', num, NUM), ('cat', onehot, make_column_selector(dtype_include=object))],
                             remainder='passthrough')
    pipe = Pipeline([('prep', prep), ('model', estimator)])
    if target == 'log':
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return TransformedTargetRegressor(regressor=pipe, transformer=PowerTransformer(method=target))


d = raw[raw['monthly_rent'].between(300, 10_000) & raw['size_sqft'].between(200, 5_000)].dropna(subset=['monthly_rent', 'rooms'])
d = d.assign(area=d['location'].str.split(' - ').str[1])
flags = pd.concat([d[c].fillna('').str.get_dummies(sep=', ') for c in ['facilities', 'additional_facilities']], axis=1)
flags = flags.loc[:, ~flags.columns.str.fullmatch(r'\d*')].add_prefix('f_')
d = pd.concat([d, flags], axis=1)[['monthly_rent'] + NUM + CAT + list(flags.columns)].reset_index(drop=True)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(d.drop(columns='monthly_rent'), d['monthly_rent'], test_size=0.2, random_state=42)
train_keys = set(map(tuple, pd.concat([yd_train, Xd_train], axis=1).astype(str).values))
copies = sum(tuple(r) in train_keys for r in pd.concat([yd_test, Xd_test], axis=1).astype(str).values)
print(f'With duplicates kept, {copies} of {len(Xd_test)} test listings have an exact copy in the training set')

complete = X_train[['completion_year', 'parking']].notna().all(axis=1)
print('Complete-case training rows:', complete.sum(), 'of', len(X_train))

variants = {'No missing indicator': (X_train, y_train, {'indicator': False}),
            'No rare-area grouping': (X_train, y_train, {'min_frequency': None}),
            'Complete cases only': (X_train[complete], y_train[complete], {}),
            'Yeo-Johnson target (not log)': (X_train, y_train, {'target': 'yeo-johnson'}),
            'Box-Cox target (not log)': (X_train, y_train, {'target': 'box-cox'}),
            'Duplicates kept': (Xd_train, yd_train, {})}
extra = pd.DataFrame(index=pd.Index(list(variants), name='config'), columns=list(BASE_MODELS), dtype=float)
for config, (Xv, yv, options) in variants.items():
    for name, (estimator, _, _) in BASE_MODELS.items():
        extra.loc[config, name] = -cross_val_score(variant_model(estimator, **options), Xv, yv, cv=CV,
                                                   scoring=SCORING, n_jobs=-1).mean()
extra = extra.round(2)
extra.to_csv('results/ablation_extra.csv')
print(extra)

# ===== D. Base Models =====

# What:   The three base models and their search spaces.
# Input:  BASE_MODELS from src/rental.py.
# Output: Printed model, search type and values searched.
# How:    Loops over the dictionary.
for name, (estimator, space, search) in BASE_MODELS.items():
    print(f'{name} ({search} search): {estimator}\n    {space}')

# ===== E. Hyperparameter Optimisation =====

# What:   Hyperparameter tuning of the three base models.
# Input:  Training data, via tune() in src/rental.py.
# Output: tuned (best fitted model per name), results/tuning.csv and figures/fig_tuning.png.
# How:    Compares the default CV MAE with the best CV MAE found by grid or random search.
tuned, rows = {}, []
for name in BASE_MODELS:
    tuned[name], info = tune(name, X_train, y_train)
    rows.append(info)
    print(f"{name}: CV MAE default {info['default_cv_mae']:.1f} -> tuned {info['tuned_cv_mae']:.1f} RM, search {info['search_seconds']:.0f}s")

tuning = pd.DataFrame(rows).set_index('model')
tuning.round(4).to_csv('results/tuning.csv')

ax = tuning[['default_cv_mae', 'tuned_cv_mae']].plot.bar(figsize=(8, 5), rot=0, width=0.7)
for bars in ax.containers:
    ax.bar_label(bars, fmt='%.1f', padding=2)
ax.set(title='Default vs tuned 5-fold CV MAE (lower is better)', xlabel='Model', ylabel='CV MAE (RM)',
       ylim=(0, tuning[['default_cv_mae', 'tuned_cv_mae']].max().max() * 1.12))
ax.legend(['Default', 'Tuned'], loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('figures/fig_tuning.png')
plt.close()
print(tuning.round(4))

# ===== F. Ensemble =====

# What:   Ensemble of the two best base models.
# Input:  Tuned models (tuned) and their CV MAE (tuning).
# Output: ensemble (chosen model), ens_label and results/ensemble_compare.csv.
# How:    Takes the two lowest CV MAE models, scores Voting and Stacking with 5-fold CV, keeps the better one.
# Notes:  Only the training set is used here, never the test set.
top2 = tuning['tuned_cv_mae'].nsmallest(2).index   # the third model is left out
members = [(name, tuned[name]) for name in top2]
# StackingRegressor's default final estimator is RidgeCV
candidates = {'Voting': VotingRegressor(members), 'Stacking': StackingRegressor(members, cv=5)}

rows = []
for technique, model in candidates.items():
    scores = -cross_val_score(model, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
    rows.append({'technique': technique, 'members': ' + '.join(top2), 'cv_mae': scores.mean(), 'cv_mae_std': scores.std()})

ens_compare = pd.DataFrame(rows)
winner = ens_compare['cv_mae'].idxmin()
ens_compare['chosen'] = ens_compare.index == winner
ens_compare.round(4).to_csv('results/ensemble_compare.csv', index=False)
ensemble = candidates[ens_compare.loc[winner, 'technique']]
ens_label = f"Ensemble ({ens_compare.loc[winner, 'technique']})"
print('Chosen ensemble:', ens_label, 'of', ' + '.join(top2))
print(ens_compare.round(4))

# What:   Check of every pair of tuned models as a voting ensemble.
# Input:  The three tuned models and the training data.
# Output: results/ensemble_pairs.csv
# How:    Scores each pair with the same 5-fold CV (the report keeps the two best single models).
from itertools import combinations

pairs = []
for a, b in combinations(tuning['tuned_cv_mae'].sort_values().index, 2):
    scores = -cross_val_score(VotingRegressor([(a, tuned[a]), (b, tuned[b])]), X_train, y_train,
                              cv=CV, scoring=SCORING, n_jobs=-1)
    pairs.append({'pair': f'{a} + {b}', 'cv_mae': scores.mean(), 'cv_mae_std': scores.std()})
pairs = pd.DataFrame(pairs).round(2)
pairs.to_csv('results/ensemble_pairs.csv', index=False)
print(pairs)

# ===== G. Evaluation =====

# What:   Final scores on the test set.
# Input:  The three tuned models, the ensemble and a dummy baseline, via evaluate() in src/rental.py.
# Output: results/test_metrics.csv and preds (test predictions per model).
# How:    Each model is fitted on the training set and scored once on the test set (MAE, MSE, RMSE, R2, timings).
models = {**tuned, ens_label: ensemble}
rows, preds = [], {}
for name, model in models.items():
    row, preds[name] = evaluate(name, model, X_train, y_train, X_test, y_test)
    rows.append(row)
row, preds['Dummy (median)'] = evaluate('Dummy (median)', DummyRegressor(strategy='median'), X_train, y_train, X_test, y_test)
rows.append(row)

test_metrics = pd.DataFrame(rows).set_index('model').round(4)
test_metrics.to_csv('results/test_metrics.csv')
print(test_metrics)

# What:   Check of how similar the base models' mistakes are.
# Input:  Test predictions (preds) of the three tuned models.
# Output: results/residual_corr.csv.
# How:    Pearson correlation between the models' test errors.
# Notes:  A high correlation means the models make the same mistakes, so averaging them helps less.
# Evidence for ensemble diversity: how correlated are the tuned models' test errors?
residuals = pd.DataFrame({name: y_test - preds[name] for name in tuned})
residual_corr = residuals.corr().round(3)
residual_corr.to_csv('results/residual_corr.csv')
print(residual_corr)

# What:   Predicted against actual rent for the four models.
# Input:  y_test and preds.
# Output: figures/fig_pred_vs_actual.png.
# How:    One scatter plot per model with the y = x line; points on the line are perfect predictions.
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for ax, name in zip(axes.flat, models):
    ax.scatter(y_test, preds[name], s=6, alpha=0.3, color=BLUE)
    ax.plot([0, 10_000], [0, 10_000], color='red', ls='--', label='y = x')
    ax.set(xlim=(0, 10_000), ylim=(0, 10_000), title=f"{name} (R$^2$ = {test_metrics.loc[name, 'test_r2']:.3f})",
           xlabel='Actual rent (RM)', ylabel='Predicted rent (RM)')
    ax.legend()
plt.tight_layout()
plt.savefig('figures/fig_pred_vs_actual.png')
plt.close()

# What:   Residual plots of the ensemble.
# Input:  y_test and the ensemble's predictions.
# Output: figures/fig_residuals.png.
# How:    Residual (actual - predicted) against the prediction, and a histogram of the residuals.
residual = y_test - preds[ens_label]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].scatter(preds[ens_label], residual, s=6, alpha=0.3, color=BLUE)
axes[0].axhline(0, color='red', ls='--')
axes[0].set(title=f'{ens_label}: residual vs predicted', xlabel='Predicted rent (RM)', ylabel='Residual = actual - predicted (RM)')
sns.histplot(residual, bins=60, color=BLUE, ax=axes[1])
axes[1].set(title=f'{ens_label}: residual distribution', xlabel='Residual (RM)', ylabel='Listings')
plt.savefig('figures/fig_residuals.png')
plt.close()

# What:   Learning curves of the three tuned models (report Figure 13).
# Input:  The three tuned models and the training data.
# Output: figures/fig_learning_curves.png
# How:    Training and 5-fold validation MAE at 10%, 25%, 50%, 75% and 100% of the training rows.
from sklearn.model_selection import learning_curve

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
for ax, (name, model) in zip(axes, tuned.items()):
    sizes, train_s, valid_s = learning_curve(model, X_train, y_train, train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
                                             cv=CV, scoring=SCORING, n_jobs=-1)
    ax.plot(sizes, -train_s.mean(1), 'o-', color=BLUE, label='Training MAE')
    ax.plot(sizes, -valid_s.mean(1), 's-', color='darkorange', label='Validation MAE (5-fold CV)')
    ax.set(title=name, xlabel='Training listings', ylabel='MAE (RM)' if ax is axes[0] else '')
axes[0].legend()
plt.tight_layout()
plt.savefig('figures/fig_learning_curves.png')
plt.close()

# What:   Significance test of the ensemble against each base model.
# Input:  Absolute test error per listing (from preds).
# Output: results/significance.csv.
# How:    Paired Wilcoxon signed-rank test; a small p-value means the difference is unlikely to be luck.
# Paired Wilcoxon test on per-listing absolute errors: chosen ensemble vs each tuned base model
error_ens = (y_test - preds[ens_label]).abs()
rows = []
for name in tuned:
    error_base = (y_test - preds[name]).abs()
    rows.append({'model_a': ens_label, 'model_b': name,
                 'mae_a': error_ens.mean(), 'mae_b': error_base.mean(), 'mae_diff': error_ens.mean() - error_base.mean(),
                 'median_abs_error_diff': (error_ens - error_base).median(),
                 'p_value': wilcoxon(error_ens, error_base).pvalue})
significance = pd.DataFrame(rows).round(4)
significance.to_csv('results/significance.csv', index=False)
print(significance)

# What:   Scalability check (report Section 7.3, Listing 7).
# Input:  The three tuned models and the training data.
# Output: results/scaling.csv
# How:    Times a fresh fit of each tuned model on 2,000, 4,000, 8,000 and all training rows.
import time
from sklearn.base import clone

rows = []
for n in [2000, 4000, 8000, len(X_train)]:
    for name, model in tuned.items():
        start = time.perf_counter()
        clone(model).fit(X_train.iloc[:n], y_train.iloc[:n])
        rows.append({'n': n, 'model': name, 'fit_seconds': time.perf_counter() - start})
scaling = pd.DataFrame(rows).pivot(index='n', columns='model', values='fit_seconds').round(2)
scaling.to_csv('results/scaling.csv')
print(scaling)

# What:   Error of the ensemble for cheap, typical and expensive units.
# Input:  y_test and the ensemble's predictions.
# Output: results/error_by_band.csv.
# How:    Cuts the actual rent into four bands, then takes the MAE and the share within 20% per band.
# Error of the chosen ensemble by rent band; within_20pct = share of listings with error <= 20% of the actual rent
labels = ['RM 300-999', 'RM 1,000-1,999', 'RM 2,000-3,999', 'RM 4,000-10,000']
errors = pd.DataFrame({'band': pd.cut(y_test, [300, 1000, 2000, 4000, 10_001], right=False, labels=labels).astype(str),
                       'abs_error': (y_test - preds[ens_label]).abs()})
errors['within_20pct'] = 100 * (errors['abs_error'] <= 0.2 * y_test)

by_band = (pd.concat([errors, errors.assign(band='All')]).groupby('band')
           .agg(n=('abs_error', 'size'), mae=('abs_error', 'mean'), within_20pct=('within_20pct', 'mean'))
           .reindex(labels + ['All']).round(4))
by_band.to_csv('results/error_by_band.csv')
print(by_band)

# What:   Error of the ensemble by area (report Table 12).
# Input:  Test set and the ensemble's predictions.
# Output: results/error_by_area.csv
# How:    Areas with at least 30 test listings; bias = mean(predicted - actual), negative = under-priced.
e = pd.DataFrame({'area': X_test['area'], 'actual': y_test, 'pred': preds[ens_label]})
e['abs_err'] = (e['actual'] - e['pred']).abs()
e['pct_err'] = 100 * e['abs_err'] / e['actual']
e['signed'] = e['pred'] - e['actual']
by_area = (e.groupby('area').agg(n=('abs_err', 'size'), median_rent=('actual', 'median'), mae=('abs_err', 'mean'),
                                 mape=('pct_err', 'mean'), bias=('signed', 'mean'))
           .query('n >= 30').round(1).sort_values('mape', ascending=False))
by_area.to_csv('results/error_by_area.csv')
print('Areas:', len(by_area), ' corr(median rent, bias):', round(by_area['median_rent'].corr(by_area['bias']), 2))
print(pd.concat([by_area.head(5), by_area.tail(5)]))

# What:   Which features the ensemble relies on.
# Input:  Fitted ensemble and the test set.
# Output: results/importance.csv and figures/fig_importance.png (top 15 features).
# How:    Shuffles one feature at a time (5 repeats) and measures how much the test MAE rises.
imp = permutation_importance(ensemble, X_test, y_test, scoring=SCORING, n_repeats=5, random_state=42, n_jobs=1)
importance = (pd.DataFrame({'feature': X_test.columns, 'mae_increase_rm': imp.importances_mean, 'std': imp.importances_std})
              .sort_values('mae_increase_rm', ascending=False).head(15).round(4))
importance.to_csv('results/importance.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.barh(importance['feature'][::-1], importance['mae_increase_rm'][::-1], xerr=importance['std'][::-1], color=BLUE)
ax.set(title=f'Permutation importance on the test set: {ens_label}', xlabel='Increase in MAE when the feature is shuffled (RM)', ylabel='Feature')
plt.savefig('figures/fig_importance.png')
plt.close()
print(importance)

# What:   Final summary.
# Input:  test_metrics.
# Output: Printed test MAE, RMSE and R2 of the ensemble and its gain over the baseline.
# How:    Compares the ensemble's test MAE with the dummy baseline's.
top = test_metrics.loc[ens_label]
base = test_metrics.loc['Dummy (median)', 'test_mae']
print(f"Test MAE RM {top['test_mae']:.1f}, RMSE RM {top['test_rmse']:.1f}, R2 {top['test_r2']:.3f} ({ens_label})")
print(f"Median-rent baseline MAE RM {base:.1f}, so the ensemble cuts the error by {100 * (1 - top['test_mae'] / base):.0f}%")
