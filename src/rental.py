"""Cleaning, preprocessing and model helpers for the KL / Selangor rent project."""
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

NUM = ['completion_year', 'rooms', 'parking', 'bathroom', 'size_sqft']
CAT = ['property_type', 'furnished', 'area', 'region']
# facility flag columns are all named f_<facility>

CV = KFold(5, shuffle=True, random_state=42)
SCORING = 'neg_mean_absolute_error'  # MAE in RM is the business metric
PREFIX = 'regressor__model__'  # TransformedTargetRegressor -> pipeline step 'model'

# name -> (estimator, search space with short names, search type)
BASE_MODELS = {
    'Random Forest': (
        RandomForestRegressor(random_state=42, n_jobs=1),  # the search already runs the folds in parallel
        {'n_estimators': [100, 200, 400], 'max_depth': [None, 10, 20, 30],
         'min_samples_leaf': [1, 2, 4], 'max_features': [1.0, 'sqrt', 0.5]},
        'random'),
    'SVR': (
        SVR(),
        # C= penalty, gamma= impact of each training listing, epsilon= width of the tube
        {'C': [0.3, 1, 3, 10], 'gamma': ['scale', 0.01, 0.03], 'epsilon': [0.02, 0.05]},
        'grid'),
    'Gradient Boosting': (
        HistGradientBoostingRegressor(random_state=42),
        # min_samples_leaf: Each leaf must represent that many listings
        # l2_regularization: shrinks corrections from small leaves; higher = more cautious, less overfitting
        {'learning_rate': [0.01, 0.03, 0.05, 0.1], 'max_iter': [500, 1000, 2000],
         'max_leaf_nodes': [31, 63, 127], 'min_samples_leaf': [5, 10, 20, 50],
         'l2_regularization': [0, 0.1, 1]},
        'random'),
}


def parse(df):
    """
    What:   Turns the rent, size and rooms text columns into numbers.
    Input:  Raw listings DataFrame as read from the CSV.
    Output: Copy with numeric monthly_rent, size_sqft and rooms (size text dropped).
    How:    Strips non-digits from rent and size; rooms via to_numeric(errors='coerce').
    Notes:  Shared by the EDA and load_and_clean so the parsing is defined once.
    """
    df = df.copy()
    df['monthly_rent'] = pd.to_numeric(df['monthly_rent'].str.replace(r'\D', '', regex=True))
    df['size_sqft'] = pd.to_numeric(df['size'].str.replace(r'\D', '', regex=True))
    df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')  # "More than 10" -> NaN
    return df.drop(columns='size')


def load_and_clean(path):
    """
    What:   Cleaner that turns raw mudah.my listings into a model-ready table.
    Input:  Raw CSV path (data/raw/mudah-apartment-kl-selangor.csv).
    Output: (clean DataFrame, dict of row counts after each cleaning step).
    How:    Parses rent/size text, applies the range rule, adds area and facility flags, drops duplicates.
    Notes:  Duplicates are judged on the modelled columns and dropped before the train/test split to avoid leakage.
            Range rule is domain based, not IQR.
    """
    df = parse(pd.read_csv(path))
    log = {'raw': len(df)}

    df = df[df['monthly_rent'].between(300, 10_000) & df['size_sqft'].between(200, 5_000)]
    log['after_range_filter'] = len(df)

    df = df.dropna(subset=['monthly_rent', 'rooms'])
    log['after_missing_rooms'] = len(df)

    df = df.assign(area=df['location'].str.split(' - ').str[1])
    flags = pd.concat([df[c].fillna('').str.get_dummies(sep=', ')
                       for c in ['facilities', 'additional_facilities']], axis=1)
    flags = flags.loc[:, ~flags.columns.str.fullmatch(r'\d*')].add_prefix('f_')  # drop junk tokens

    df = pd.concat([df, flags], axis=1)[['monthly_rent'] + NUM + CAT + list(flags.columns)]
    df = df.drop_duplicates().reset_index(drop=True)  # reposts and identical twin listings
    log['after_duplicates'] = len(df)
    return df, log


def build_preprocessor(scale=True):
    """
    What:   Column-wise preprocessing shared by every model.
    Input:  scale on/off for the numeric columns.
    Output: Unfitted ColumnTransformer.
    How:    Numeric: median impute + missing flag + scale. Text columns: one-hot. 0/1 facility flags pass through.
    Notes:  Categories seen under 20 times are grouped as 'infrequent'. No categorical NaNs remain after cleaning.
    """
    num_steps = [SimpleImputer(strategy='median', add_indicator=True)]
    if scale:
        num_steps.append(StandardScaler())
    onehot = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20, sparse_output=False)
    return ColumnTransformer([('num', make_pipeline(*num_steps), NUM),
                              ('cat', onehot, make_column_selector(dtype_include=object))],
                             remainder='passthrough')  # the 0/1 facility flags pass through unchanged


def make_model(estimator, log_target=True, scale=True):
    """
    What:   Full model = preprocessing + estimator, predicting in RM.
    Input:  Estimator and the preprocessing switches.
    Output: Unfitted pipeline (wrapped in TransformedTargetRegressor if log_target).
    How:    Trains on log1p(rent) and converts predictions back with expm1. The estimator step is named 'model'.
    Notes:  Rent is right-skewed, so the log target stops large rents dominating the fit.
    """
    model = Pipeline([('prep', build_preprocessor(scale)), ('model', estimator)])
    if log_target:
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return model


def tune(name, X, y):
    """
    What:   Tunes one base model and measures what tuning gained over the defaults.
    Input:  Model name (key of BASE_MODELS), training X and y.
    Output: (best fitted pipeline, dict with default/tuned CV MAE, best params, search time).
    How:    cross_val_score for defaults, then Grid/RandomizedSearchCV on the same 5-fold CV.
    Notes:  Tuned CV MAE is the score the search picked on, so it is slightly optimistic.
    """
    estimator, params, kind = BASE_MODELS[name]
    model = make_model(estimator)
    default_mae = -cross_val_score(model, X, y, cv=CV, scoring=SCORING, n_jobs=-1).mean()

    grid = {PREFIX + k: v for k, v in params.items()}
    if kind == 'random':
        search = RandomizedSearchCV(model, grid, n_iter=20, cv=CV, scoring=SCORING, n_jobs=-1, random_state=42)
    else:
        search = GridSearchCV(model, grid, cv=CV, scoring=SCORING, n_jobs=-1)
    start = time.perf_counter()
    search.fit(X, y)
    seconds = time.perf_counter() - start

    tuned_mae = -search.best_score_
    info = {'model': name,
            'default_cv_mae': default_mae,
            'tuned_cv_mae': tuned_mae,
            'tuned_cv_mae_std': search.cv_results_['std_test_score'][search.best_index_],
            'improvement_pct': 100 * (default_mae - tuned_mae) / default_mae,
            'best_params': {k.removeprefix(PREFIX): v for k, v in search.best_params_.items()},
            'search_seconds': seconds}
    return search.best_estimator_, info


def evaluate(name, model, X_train, y_train, X_test, y_test):
    """
    What:   Scores one model on the held-out test set, with timings.
    Input:  Model name, unfitted or fitted model, train and test data.
    Output: (dict of metrics and timings, test predictions in RM).
    How:    Refits on the training set, predicts the test set, computes MAE/MSE/RMSE/R2.
    Notes:  Train MAE/R2 are included so over-fitting is visible next to the test scores.
    """
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    pred = model.predict(X_test)
    predict_seconds = time.perf_counter() - start

    train_pred = model.predict(X_train)
    mse = mean_squared_error(y_test, pred)
    row = {'model': name,
           'test_mae': mean_absolute_error(y_test, pred),
           'test_mse': mse,
           'test_rmse': np.sqrt(mse),
           'test_r2': r2_score(y_test, pred),
           'train_mae': mean_absolute_error(y_train, train_pred),
           'train_r2': r2_score(y_train, train_pred),
           'fit_seconds': fit_seconds,
           'predict_seconds': predict_seconds}
    return row, pred
