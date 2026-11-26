"""House Price Estimator: a simple web app (Streamlit).

Run locally:   streamlit run deployment/app.py
The app trains the model from data/house_data.csv when it starts, so it needs no saved model file.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from src.data_processing import VALID_LOCATIONS, VALID_TYPES
from src.predict import PricePredictor

st.set_page_config(page_title='House Price Estimator', page_icon='🏠', layout='centered')


@st.cache_resource
def get_predictor():
    return PricePredictor()


predictor = get_predictor()
data = predictor.data

st.title('🏠 House Price Estimator')
st.write('Enter the details of a property to get an instant price estimate. '
         'Built for a real estate agency to price listings quickly and consistently.')

tab_estimate, tab_market, tab_about = st.tabs(['Estimate a price', 'Market insights', 'About the model'])

with tab_estimate:
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input('Area', min_value=100, max_value=10000, value=2500, step=50)
        bedrooms = st.number_input('Bedrooms', min_value=1, max_value=10, value=3, step=1)
        bathrooms = st.number_input('Bathrooms', min_value=1, max_value=10, value=2, step=1)
    with col2:
        age = st.number_input('Age of the property (years)', min_value=0, max_value=100, value=10, step=1)
        location = st.selectbox('Location', VALID_LOCATIONS)
        property_type = st.selectbox('Property type', VALID_TYPES)

    if st.button('Estimate price', type='primary'):
        try:
            result = predictor.predict_price(area, bedrooms, bathrooms, age, location, property_type)
            st.metric('Estimated price', f"{result['estimate']:,}")
            st.write(f"Likely range: **{result['low']:,}** to **{result['high']:,}**")
            for warning in result['warnings']:
                st.warning(warning)
        except ValueError as problem:
            st.error(str(problem))

with tab_market:
    st.subheader('Price per unit of area by location')
    per_area = (data['Price'] / data['Area']).groupby(data['Location']).mean().reindex(VALID_LOCATIONS)
    st.bar_chart(per_area.round(0))
    st.subheader('Median price by location and property type')
    median_table = data.pivot_table(index='Location', columns='Property_Type', values='Price', aggfunc='median').reindex(VALID_LOCATIONS)
    st.dataframe(median_table.round(0).astype(int))
    st.subheader('Area against price')
    st.scatter_chart(data, x='Area', y='Price', color='Location')

with tab_about:
    st.subheader('How it works')
    st.write('The model learns from 300 properties. It combines the area, rooms, age, location and type, and it lets the '
             'features work together (for example, area is worth more in the City Center than in Rural areas).')
    st.write(f'The price range shown is the estimate plus or minus {predictor.range_band * 100:.0f}%.')
    st.subheader('Accuracy on 60 unseen properties')
    st.write('R2 = 1.000 and average error of about 0%. The model meets the project targets (R2 at least 0.95, error at most 10%).')
    st.info('Important: the data in this project is practice data, and its prices follow a fixed formula, which is why the '
            'model is so accurate. With real market data, expect larger errors and always treat the estimate as a guide.')
