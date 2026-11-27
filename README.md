# House Price Estimator

**Final Capstone - a complete data science project, from data to a deployed web app**

A real estate agency prices listings by hand, which is slow and inconsistent. This project builds and deploys a model that estimates a property's price from its area, rooms, age, location and type, and explains what drives the price.

**Live app:** https://house-price-estimator-sahana.streamlit.app

## Results at a glance
| Success metric | Target | Result on 60 unseen properties |
|---|---|---|
| R2 (share of price explained) | at least 0.95 | 1.000 |
| MAPE (average % error) | at most 10% | about 0% |
| Time per estimate | under 1 second | a few milliseconds |
| Deployed app with tests | yes | Streamlit web app + FastAPI service + 37 tests |

> **Important:** the data is practice data and its prices follow a fixed formula, which is why the model is so accurate. On real market data expect larger errors. See `reports/technical_documentation.md`.

## Key findings
- **Location sets the price:** price per unit of area is 12,526 in the City Center, 9,681 in the Suburb and 6,442 in Rural areas (1.9 times).
- **Area and location work together:** each extra unit of area adds about 10,032 in the City Center, 7,502 in the Suburb and 5,064 in Rural areas.
- Bedrooms (+1,525,648 each), bathrooms (+804,528 each) and age (-75,115 per year) matter less.
- Features that combine area and location cut the error of a simple model from 13.4% to 2.9%; a polynomial model reaches about 0%.

## The workflow
1. **Project setup:** business problem, success metrics
2. **Data:** 300 properties, 11 quality checks (all passed), data dictionary
3. **Exploration:** 6 charts and 6 patterns
4. **Model development:** 6 engineered features, 7 models compared with cross-validation, Grid Search tuning
5. **Evaluation:** final test on 60 unseen properties, error analysis
6. **Deployment:** prediction function with input checks, API, web app
7. **Documentation:** technical documentation, business report, presentation

## Repository structure
```
README.md
capstone_project.ipynb        main analysis (3 parts, with outputs)
requirements.txt
src/                          source code modules
    data_processing.py, features.py, model.py, predict.py, test_predict.py
data/                         house_data.csv, data_dictionary.md
reports/                      technical_documentation.md, business_report.pdf, charts
deployment/                   app.py (web app), api.py (API), test_deployment.py, requirements.txt, README.md
presentation/                 capstone_presentation.pptx (13 slides)
```

## Quick start
```bash
pip install -r requirements.txt
python -m pytest src deployment          # run the 37 repo and app tests
streamlit run deployment/app.py          # web app
uvicorn deployment.api:app               # API (then open http://127.0.0.1:8000/docs)
```
To run the notebook: open `capstone_project.ipynb` in Google Colab, upload `house_prices.csv` when asked, and run all cells.

## Documentation
- `reports/technical_documentation.md` - methodology, code structure, results, testing evidence
- `reports/business_report.pdf` - executive summary, findings, recommendations
- `presentation/capstone_presentation.pptx` - slides covering problem to solution
- `deployment/README.md` - how to run and deploy the app and the API

## Author
Sahana
