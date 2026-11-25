# Data Dictionary: `house_data.csv`

300 properties, 8 columns. Source: `house_prices.csv`, provided with the course. It is practice data.

| Column | Type | Meaning | Values seen |
|---|---|---|---|
| Property_ID | Text | Unique code of the property (not used by the model) | PROP0001 to PROP0300 |
| Area | Number | Floor area of the property | 520 to 4,999 |
| Bedrooms | Number | Number of bedrooms | 1 to 5 |
| Bathrooms | Number | Number of bathrooms | 1 to 3 |
| Age | Number | Age of the property in years | 0 to 49 |
| Location | Category | Where the property is | City Center, Suburb, Rural |
| Property_Type | Category | Kind of property | House, Villa, Apartment |
| Price | Number | Sale price (the value to predict; the currency is not stated in the file) | 3,695,000 to 58,700,000 |

## Quality checks (all passed)
At least 100 properties; no missing values; no duplicate rows; unique Property_ID; Area, Bedrooms, Bathrooms and Age within sensible limits; Price above 0; Location and Property_Type contain only the 3 known values each.

## Summary
| | Area | Bedrooms | Bathrooms | Age | Price |
|---|---|---|---|---|---|
| Mean | 2,759.7 | 3.0 | 2.0 | 25.0 | 24,883,658 |
| Median | 2,738 | 3 | 2 | 25.5 | 22,365,000 |

Locations: Suburb 105, Rural 98 and City Center 97 properties (about a third each). Property types are also balanced: House 107, Villa 102 and Apartment 91.
