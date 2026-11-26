"""Tests for the web app and the API. Run with: python -m pytest deployment"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from deployment.api import app

APP_FILE = str(Path(__file__).resolve().parent / 'app.py')
client = TestClient(app)


def test_health():
    assert client.get('/health').json() == {'status': 'ok'}


def test_predict_ok():
    body = {'area': 3000, 'bedrooms': 3, 'bathrooms': 2, 'age': 10, 'location': 'City Center', 'property_type': 'Villa'}
    response = client.post('/predict', json=body)
    assert response.status_code == 200 and response.json()['estimate'] > 0


def test_predict_bad_location_is_rejected():
    body = {'area': 3000, 'bedrooms': 3, 'bathrooms': 2, 'age': 10, 'location': 'Mars', 'property_type': 'Villa'}
    assert client.post('/predict', json=body).status_code == 422


def test_app_starts_and_shows_estimate():
    at = AppTest.from_file(APP_FILE, default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert 'Estimated price' in at.metric[0].label and at.metric[0].value != ''
