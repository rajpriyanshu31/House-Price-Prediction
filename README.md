# 🏠 House Price Prediction

A Machine Learning web application that predicts house prices using a tuned Random Forest Regression model.

## 🚀 Features

- House price prediction using Random Forest
- Hyperparameter tuning
- 5-Fold Cross Validation
- Model performance metrics
- Feature importance visualization
- Frontend and backend input validation
- Clean Flask web interface
- Property input summary
- Human-readable predicted price

## 🤖 Machine Learning Model

The project uses the California Housing dataset and a tuned Random Forest Regressor.

### Model Performance

- R² Score: **80.62%**
- MAE: **0.3266**
- RMSE: **0.5039**
- 5-Fold Cross-Validation RMSE: **0.5103**

## 📊 Input Features

The model uses 8 features:

1. Median Income
2. House Age
3. Average Rooms
4. Average Bedrooms
5. Population
6. Average Occupancy
7. Latitude
8. Longitude

## 🧠 Feature Importance

The Random Forest model identifies the following important features:

| Feature | Importance |
|---|---:|
| MedInc | 52.6% |
| AveOccup | 13.8% |
| Latitude | 8.9% |
| Longitude | 8.8% |
| HouseAge | 5.4% |
| AveRooms | 4.4% |
| Population | 3.1% |
| AveBedrms | 3.0% |

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Flask
- HTML
- CSS
- Joblib

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── app/
│   ├── app.py
│   └── templates/
│       ├── index.html
│       └── result.html
│
├── data/
│   └── california_housing.csv
│
├── models/
│   └── house_price_model.pkl
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│
├── requirements.txt
└── README.md
