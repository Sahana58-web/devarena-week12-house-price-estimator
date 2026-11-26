# Deployment

The model is deployed in two ways: a **web app** for agents and an **API** for other systems. Both use `src/predict.py`, and both train the model from `data/house_data.csv` when they start, so no saved model file is needed.

## Run the web app on your computer
```bash
pip install -r requirements.txt
streamlit run deployment/app.py
```
The app has three tabs: *Estimate a price*, *Market insights* and *About the model*.

## Run the API on your computer
```bash
uvicorn deployment.api:app --reload
```
Open http://127.0.0.1:8000/docs to try it. Example request:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"area": 3000, "bedrooms": 3, "bathrooms": 2, "age": 10, "location": "City Center", "property_type": "Villa"}'
```
Response:
```json
{"status": "ok", "estimate": 37000000, "low": 35890000, "high": 38110000, "warnings": []}
```
An invalid request (for example an unknown location) returns HTTP code 422 with a message.

## Publish the web app for free (Streamlit Community Cloud)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** (or **New app**) and choose **Deploy a public app from GitHub**.
3. Choose this repository, the branch `main`, and the main file path `deployment/app.py`.
4. Click **Deploy** and wait a few minutes. Streamlit installs the packages from `deployment/requirements.txt`.
5. Copy the app link and put it in the main `README.md`.

## Tests
```bash
python -m pytest deployment
```
They check the API (health, a good request, a bad request) and that the web app starts and shows an estimate.
