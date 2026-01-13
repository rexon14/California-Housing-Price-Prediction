# California Housing Price Prediction

Machine learning regression model to predict median house values in California districts.

## Business Context

**Stakeholder:** CalHome Realty - Pricing Team

**Problem:** Manual property appraisals are time-consuming, inconsistent, and average ~$50,000 error per valuation.

**Solution:** ML model providing fast, accurate price predictions to support listing decisions.

## Dataset

- **Source:** 1990 California Census
- **Records:** 14,448 districts
- **Features:** 10 (location, housing characteristics, demographics)
- **Target:** `median_house_value`

| Feature | Description |
|---------|-------------|
| longitude, latitude | Geographic coordinates |
| housing_median_age | Median age of houses in district |
| total_rooms, total_bedrooms | Housing capacity metrics |
| population, households | District demographics |
| median_income | Median income (scaled) |
| ocean_proximity | Categorical: INLAND, NEAR BAY, NEAR OCEAN, <1H OCEAN, ISLAND |

## Methodology

### Preprocessing
- Missing values: Median imputation (total_bedrooms)
- Outliers: Retained (real market variation)
- Feature engineering: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`
- Encoding: One-hot encoding (ocean_proximity)
- Scaling: StandardScaler

### Model Selection
Evaluated 7 algorithms via 5-fold cross-validation:
- Linear Regression, Ridge, Lasso
- Decision Tree
- Random Forest, Gradient Boosting, XGBoost

**Final Model:** Selected based on lowest MAE in cross-validation.

### Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **MAE** (Primary) | Average dollar error — directly interpretable for business |
| **RMSE** (Secondary) | Monitors large prediction errors |
| **R²** (Supporting) | Overall explanatory power |

## Results

| Metric | Target | Achieved |
|--------|--------|----------|
| MAE | < $45,000 | ✓ |
| RMSE | < $65,000 | ✓ |
| R² | > 0.65 | ✓ |

**Business Impact:**
- ~36% error reduction vs manual appraisal
- Projected $270,000 yearly savings (600 transactions)

## Model Limitations

- Cannot predict values above $500,000 (data capped)
- Trained on 1990 data — retrain with current data for production
- California districts only
- District-level predictions, not property-specific

## Repository Structure

```
├── README.md
├── california_housing_prediction.ipynb    # Full analysis notebook
├── data_california_house.csv              # Dataset
└── saved_models/
    ├── california_housing_model.pkl       # Trained model
    ├── scaler.pkl                         # Preprocessing scaler
    └── feature_names.pkl                  # Feature column order
```

## Usage

```python
import pickle
import pandas as pd

# Load model and preprocessors
with open('saved_models/california_housing_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('saved_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('saved_models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Predict
new_data_scaled = scaler.transform(new_data[feature_names])
prediction = model.predict(new_data_scaled)
```

## Tools & Libraries

- Python 3.9
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- XGBoost

## Author

**Dennis Schira**
<br>_JCDSAH-024 ~ Purwadhika Data Science Bootcamp_
