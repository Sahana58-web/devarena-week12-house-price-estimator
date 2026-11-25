"""Model building, comparison and evaluation."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from .features import FEATURES, NEW_NUMERIC, NUMERIC, AddFeatures, make_preprocessor


def split_data(df, random_state=42):
    """80/20 split, keeping the share of each location the same in both parts."""
    return train_test_split(df[FEATURES], df['Price'], test_size=0.2, random_state=random_state, stratify=df['Location'])


def build_final_model():
    """The chosen model: scaled numbers and one-hot categories, all pairwise products, then Linear Regression."""
    return Pipeline([('prep', make_preprocessor(NUMERIC)),
                     ('poly', PolynomialFeatures(2, include_bias=False)),
                     ('model', LinearRegression())])


def build_candidates():
    """The five models compared in the notebook."""
    return {
        '1. Linear Regression (basic features)': Pipeline([('prep', make_preprocessor(NUMERIC)), ('model', LinearRegression())]),
        '2. Linear Regression (+ engineered features)': Pipeline([('add', AddFeatures()),
                                                                  ('prep', make_preprocessor(NUMERIC + NEW_NUMERIC)),
                                                                  ('model', LinearRegression())]),
        '3. Polynomial degree 2 + Linear Regression': build_final_model(),
        '4. Random Forest (basic features)': Pipeline([('prep', make_preprocessor(NUMERIC)),
                                                       ('model', RandomForestRegressor(n_estimators=200, random_state=42))]),
        '5. Gradient Boosting (basic features)': Pipeline([('prep', make_preprocessor(NUMERIC)),
                                                           ('model', GradientBoostingRegressor(random_state=42))]),
    }


def evaluate(model, X, y):
    """R2, MAE, RMSE, MAPE and the share of predictions within 5% and 10% of the price."""
    pred = model.predict(X)
    rel = np.abs(pred - y) / y
    return {'R2': r2_score(y, pred), 'MAE': mean_absolute_error(y, pred), 'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'MAPE %': float(rel.mean() * 100), 'Within 5% of price': float((rel <= 0.05).mean() * 100),
            'Within 10% of price': float((rel <= 0.10).mean() * 100)}


def cross_validated_scores(model, X, y, folds=5):
    """Mean cross-validated scores on the training data."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    s = cross_validate(model, X, y, cv=cv, scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error',
                                                    'rmse': 'neg_root_mean_squared_error',
                                                    'mape': 'neg_mean_absolute_percentage_error'})
    return {'R2': s['test_r2'].mean(), 'MAE': -s['test_mae'].mean(), 'RMSE': -s['test_rmse'].mean(),
            'MAPE %': -s['test_mape'].mean() * 100}


def compare_models(X_train, y_train):
    """Cross-validated comparison table of all candidate models."""
    rows = [{'Model': name, **cross_validated_scores(model, X_train, y_train)} for name, model in build_candidates().items()]
    return pd.DataFrame(rows)
