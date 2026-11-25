"""Loading and validating the house price data."""
from pathlib import Path

import pandas as pd

VALID_LOCATIONS = ['City Center', 'Suburb', 'Rural']
VALID_TYPES = ['House', 'Villa', 'Apartment']
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'house_data.csv'


def load_data(path=None):
    """Reads the house data from a CSV file (default: data/house_data.csv)."""
    return pd.read_csv(path or DEFAULT_DATA_PATH)


def validate_data(df):
    """Runs the data quality checks. Returns a list of (check name, passed) pairs."""
    return [
        ('At least 100 properties', len(df) >= 100),
        ('No missing values', int(df.isnull().sum().sum()) == 0),
        ('No duplicate rows', int(df.duplicated().sum()) == 0),
        ('Property_ID is unique', bool(df['Property_ID'].is_unique)),
        ('Area between 100 and 10,000', bool(df['Area'].between(100, 10000).all())),
        ('Bedrooms between 1 and 10', bool(df['Bedrooms'].between(1, 10).all())),
        ('Bathrooms between 1 and 10', bool(df['Bathrooms'].between(1, 10).all())),
        ('Age is 0 or more', bool((df['Age'] >= 0).all())),
        ('Price is above 0', bool((df['Price'] > 0).all())),
        ('Location has 3 known values', set(df['Location']) == set(VALID_LOCATIONS)),
        ('Property_Type has 3 known values', set(df['Property_Type']) == set(VALID_TYPES)),
    ]
