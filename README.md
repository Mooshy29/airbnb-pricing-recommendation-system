# Airbnb Pricing Recommendation System

A machine learning and business intelligence project using public Airbnb listing data to identify potentially underpriced and overpriced listings, generate market-based pricing recommendations, and support host-success outreach decisions.

> Academic project. This repository is not affiliated with or endorsed by Airbnb.

## Project Overview

This project explores how machine learning can be used to identify Airbnb listings that may be systematically mispriced relative to comparable listings.

We combined multi-year, multi-market Airbnb listing data from 2019, 2020, and 2023, cleaned and harmonized different schemas, engineered pricing-related features, compared multiple regression models, and built an interactive Streamlit decision-support dashboard.

The cleaned dataset contained more than 42,000 listings across multiple U.S. markets.

## Business Problem

The main question behind the project was:

**Which listings appear materially underpriced or overpriced, and how can those recommendations be explained in a way that is useful for business decision-making?**

Rather than building an automated pricing engine, the project focuses on a recommendation system that helps prioritize listings for outreach and provides interpretable reasons behind each suggested price.

## Data Preparation & Feature Engineering

The workflow included:

- Cleaning and harmonizing multiple Airbnb datasets
- Parsing and standardizing nightly prices
- Handling missing values and inconsistent schemas
- Capping extreme price outliers
- Creating location and market features
- Engineering availability and review activity features
- Creating amenity indicators and amenity scores
- Incorporating host and listing characteristics
- Adding year-based pricing regime information

## Exploratory Data Analysis

The EDA examined how nightly prices vary across:

- Markets and neighborhoods
- Room types
- Listing capacity
- Amenities
- Availability
- Review activity
- Geographic location
- Different pricing periods

The analysis showed strong nonlinear relationships and interactions, motivating the use of tree-based ensemble models.

## Modeling

The following approaches were compared:

- Naive segment-median baseline
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost

The final model was a tuned XGBoost regressor.

### Model Performance

The tuned XGBoost model achieved approximately:

- **$58 MAE per night**
- **18.3% lower MAE than the naive baseline**

Model performance was evaluated on a held-out test set, with additional cross-validation and segment-level error analysis.

## Model Interpretability

To make the recommendations more understandable, the project used:

- **Permutation Importance** for global feature importance
- **SHAP** for explaining individual predictions

This allows the model to show which listing characteristics contributed most to a specific pricing recommendation.

## Mispricing Classification

Listings were classified based on the difference between actual and predicted price:

- **Underpriced**
- **Fairly Priced**
- **Overpriced**

Threshold sensitivity analysis was used to compare different mispricing cutoffs and balance the number of flagged listings against model uncertainty.

## Streamlit Dashboard

An interactive Streamlit dashboard was built to turn the model into a business decision-support tool.

The dashboard includes:

- Predicted market-consistent nightly price
- Current listed price
- Pricing gap
- Underpriced / Fairly Priced / Overpriced classification
- SHAP-based explanations
- Listing and market filters
- Adoption-rate sensitivity analysis
- Model performance views
- Ranked outreach lists
- CSV export functionality

## Business Analysis

The project also included:

- Mispricing threshold sensitivity analysis
- Estimated marketplace impact scenarios
- Prioritized outreach lists
- A proposed 90-day A/B test design

The A/B test was designed to evaluate whether explainable pricing recommendations could improve host adoption and marketplace value before broader rollout.

## Repository Files

### `airbnb_pricing_analysis.ipynb`
Main analysis notebook containing data cleaning, EDA, feature engineering, model development, model evaluation, interpretability, and business analysis.

### `app.py`
Streamlit decision-support dashboard.

### `airbnb_price_model.pkl`
Serialized trained model and supporting metadata used by the Streamlit application.

### `mispricing_sample_output.csv`
Sample model output containing actual prices, predicted prices, mispricing gaps, and pricing classifications.

### `model_metrics.csv`
Model comparison and evaluation results.

### `top_500_pilot_listings.csv`
Prioritized list of underpriced listings generated for the proposed outreach pilot.

### `airbnb_pricing_final_report.pdf`
Full project report.

### `airbnb_pricing_presentation.pdf`
Final project presentation.

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- Streamlit
- Plotly
- Matplotlib
- Jupyter Notebook

## Limitations

This project should be interpreted as a decision-support analysis rather than an automated pricing system.

Key limitations include:

- Listed price is not the same as realized booking revenue
- The data is observational rather than causal
- Pricing behavior varies across markets
- Historical data includes different market regimes
- Important factors such as photography quality, calendar dynamics, host responsiveness, and booking behavior are not fully captured
- SHAP explains model predictions but does not establish causal effects

## Contributors

Team 3:

- Ghilliann Lou
- Jianxiong Zhu
- Ruimin Pei
- Thomas Zhang
- Yuxuan Jin
