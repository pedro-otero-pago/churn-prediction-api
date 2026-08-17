# churn-prediction-api
REST API that predicts customer churn from a trained scikit-learn model, containerized with Docker for one-command deployment.

## Exploratory Data Analysis

Run `python src/eda.py` to regenerate the exploratory plots (saved to
`reports/`): boxplots of tenure and MonthlyCharges by churn status, a
correlation heatmap of the numeric features, and grouped bar charts for
the categorical and service-related features.

Key findings: customers who churn tend to have shorter tenure, higher
monthly charges, month-to-month contracts, electronic check as payment
method, and fiber optic internet. TotalCharges was dropped due to a
0.83 correlation with tenure; gender, PhoneService, and MultipleLines
were dropped for showing little to no relationship with churn.

## Project Structure
- `data_processing.py` — loads the raw CSV, cleans TotalCharges
  (converting blank values for zero-tenure customers to 0), and drops
  columns found to be unused or redundant during the EDA.
- `eda.py` — standalone script that explores the full (uncleaned-of-
  columns) dataset and saves exploratory plots to `reports/`.