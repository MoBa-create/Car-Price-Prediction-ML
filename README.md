# Car Price Prediction (Supervised Learning - Regression)

A Machine Learning project demonstrating end-to-end regression modeling to predict used car prices based on vehicle features including manufacturing year, mileage, engine capacity, fuel type, and transmission.

## 📌 Key Highlights
* Feature encoding for categorical variables (`Fuel_Type`, `Transmission`) via Dummy Encoding (`pd.get_dummies`).
* Model evaluation leveraging **$R^2$ Score** and **Root Mean Squared Error (RMSE)**.
* Dynamic local path handling (`BASE_DIR`) for cross-platform compatibility.
* Automated serialization of trained models (`.pkl`) and processed datasets (`.csv`).

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Scikit-Learn, Pandas, NumPy, Joblib

## 🚀 Usage
1. Clone repository:
   ```bash
   git clone https://github.com/MoBa-create/Car-Price-Prediction-ML.git

2 . Install dependencies:
	pip install -r requirements.txt

3 . Run the regression model script:
	python car_price_prediction.py