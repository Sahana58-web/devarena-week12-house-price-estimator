"""The prediction function used by the web app and the API."""
import json

import numpy as np
import pandas as pd

from .data_processing import VALID_LOCATIONS, VALID_TYPES, load_data
from .model import build_final_model, split_data

MIN_RANGE_BAND = 0.03          # the shown price range is never narrower than +/- 3%


def _number(value, name, minimum):
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} is required")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if np.isnan(number) or np.isinf(number):
        raise ValueError(f"{name} must be a real number")
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


class PricePredictor:
    """Trains the model from the data file and estimates prices for new properties."""

    def __init__(self, data_path=None):
        self.data = load_data(data_path)
        X_train, X_test, y_train, y_test = split_data(self.data)
        check = build_final_model().fit(X_train, y_train)                    # used only to measure the typical error
        relative = np.abs(check.predict(X_test) - y_test) / y_test
        self.range_band = max(float(np.quantile(relative, 0.9)), MIN_RANGE_BAND)
        self.model = build_final_model().fit(self.data[X_train.columns], self.data['Price'])   # deployed model uses all properties
        self.ranges = {'Area': (int(self.data['Area'].min()), int(self.data['Area'].max())),
                       'Bedrooms': (int(self.data['Bedrooms'].min()), int(self.data['Bedrooms'].max())),
                       'Bathrooms': (int(self.data['Bathrooms'].min()), int(self.data['Bathrooms'].max())),
                       'Age': (int(self.data['Age'].min()), int(self.data['Age'].max()))}

    def predict_price(self, area, bedrooms, bathrooms, age, location, property_type):
        """Estimates the price of one property. Raises ValueError for invalid input."""
        values = {'Area': _number(area, 'Area', 1), 'Bedrooms': _number(bedrooms, 'Bedrooms', 1),
                  'Bathrooms': _number(bathrooms, 'Bathrooms', 1), 'Age': _number(age, 'Age', 0)}
        if location not in VALID_LOCATIONS:
            raise ValueError(f"Location must be one of {VALID_LOCATIONS}")
        if property_type not in VALID_TYPES:
            raise ValueError(f"Property_Type must be one of {VALID_TYPES}")
        warnings = []
        for name, value in values.items():
            low, high = self.ranges[name]
            if value < low or value > high:
                warnings.append(f"{name} {value:g} is outside the range the model learned ({low} to {high}); the estimate may be unreliable")
        row = pd.DataFrame([{**values, 'Location': location, 'Property_Type': property_type}])
        estimate = float(self.model.predict(row)[0])
        if estimate <= 0:
            warnings.append("The estimate is not positive; the inputs are far from the properties the model learned")
        return {'estimate': round(estimate), 'low': round(estimate * (1 - self.range_band)),
                'high': round(estimate * (1 + self.range_band)), 'warnings': warnings}

    def predict_api(self, request_text):
        """JSON text in, JSON text out."""
        try:
            result = self.predict_price(**json.loads(request_text))
            return json.dumps({'status': 'ok', **result})
        except (ValueError, TypeError, json.JSONDecodeError) as problem:
            return json.dumps({'status': 'error', 'message': str(problem)})
