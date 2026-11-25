"""Tests for the data checks, the model and the prediction function. Run with: python -m pytest src"""
import json

import pytest

from src.data_processing import load_data, validate_data
from src.model import build_final_model, evaluate, split_data
from src.predict import PricePredictor


@pytest.fixture(scope='module')
def predictor():
    return PricePredictor()


def test_data_passes_all_validation_checks():
    assert all(ok for _, ok in validate_data(load_data()))


def test_split_sizes_and_untouched_test_set():
    X_train, X_test, _, _ = split_data(load_data())
    assert (len(X_train), len(X_test)) == (240, 60)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_model_meets_success_metrics_on_unseen_properties():
    X_train, X_test, y_train, y_test = split_data(load_data())
    scores = evaluate(build_final_model().fit(X_train, y_train), X_test, y_test)
    assert scores['R2'] >= 0.95
    assert scores['MAPE %'] <= 10


def test_valid_property_gets_positive_estimate(predictor):
    assert predictor.predict_price(2000, 3, 2, 10, 'Suburb', 'House')['estimate'] > 0


def test_bigger_area_costs_more(predictor):
    small = predictor.predict_price(2000, 3, 2, 10, 'Suburb', 'House')['estimate']
    assert predictor.predict_price(3000, 3, 2, 10, 'Suburb', 'House')['estimate'] > small


def test_city_center_costs_more_than_rural(predictor):
    city = predictor.predict_price(2000, 3, 2, 10, 'City Center', 'House')['estimate']
    assert city > predictor.predict_price(2000, 3, 2, 10, 'Rural', 'House')['estimate']


def test_older_property_costs_less(predictor):
    new = predictor.predict_price(2000, 3, 2, 10, 'Suburb', 'House')['estimate']
    assert predictor.predict_price(2000, 3, 2, 40, 'Suburb', 'House')['estimate'] < new


@pytest.mark.parametrize('bad', [{'location': 'Mars'}, {'property_type': 'Castle'}, {'area': -100},
                                 {'bedrooms': None}, {'age': 'old'}, {'area': float('nan')}])
def test_invalid_input_is_rejected(predictor, bad):
    args = dict(area=2000, bedrooms=3, bathrooms=2, age=10, location='Suburb', property_type='House')
    args.update(bad)
    with pytest.raises(ValueError):
        predictor.predict_price(**args)


def test_area_outside_learned_range_gives_warning(predictor):
    assert predictor.predict_price(20000, 3, 2, 10, 'Suburb', 'House')['warnings']


def test_range_is_at_least_three_percent(predictor):
    r = predictor.predict_price(2000, 3, 2, 10, 'Suburb', 'House')
    assert (r['high'] - r['low']) / r['estimate'] >= 0.0599


def test_api_ok_and_error(predictor):
    good = json.loads(predictor.predict_api(json.dumps({'area': 3000, 'bedrooms': 3, 'bathrooms': 2, 'age': 10,
                                                         'location': 'City Center', 'property_type': 'Villa'})))
    assert good['status'] == 'ok' and good['estimate'] > 0
    assert json.loads(predictor.predict_api('{"area": 3000}'))['status'] == 'error'
    assert json.loads(predictor.predict_api('not json'))['status'] == 'error'
