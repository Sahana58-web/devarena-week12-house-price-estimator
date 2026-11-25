"""Feature engineering and preprocessing."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ['Area', 'Bedrooms', 'Bathrooms', 'Age', 'Location', 'Property_Type']
CATEGORICAL = ['Location', 'Property_Type']
NUMERIC = ['Area', 'Bedrooms', 'Bathrooms', 'Age']
NEW_NUMERIC = ['Total_rooms', 'Area_per_bedroom', 'Is_new', 'Area_in_City_Center', 'Area_in_Suburb', 'Area_in_Rural']


class AddFeatures(BaseEstimator, TransformerMixin):
    """Builds new features from the raw columns, so a new property can be scored in the same way."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Total_rooms'] = X['Bedrooms'] + X['Bathrooms']
        X['Area_per_bedroom'] = X['Area'] / X['Bedrooms']
        X['Is_new'] = (X['Age'] <= 10).astype(int)
        for loc in ['City Center', 'Suburb', 'Rural']:
            X['Area_in_' + loc.replace(' ', '_')] = X['Area'] * (X['Location'] == loc)
        return X


def make_preprocessor(numeric_columns):
    """Scales the numbers and one-hot encodes the categories."""
    return ColumnTransformer([('numbers', StandardScaler(), numeric_columns),
                              ('categories', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL)])
