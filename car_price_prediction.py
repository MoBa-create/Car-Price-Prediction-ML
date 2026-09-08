from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

cars = {
    "Year": [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
    "Kms_Driven": [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100],
    "Fuel_Type": ["Diesel", "Diesel", "Diesel", "Diesel", "Diesel", "Petrol", "Petrol", "Petrol", "Petrol", "Petrol"],
    "Transmission": ["Manual", "Manual", "Manual", "Manual", "Manual", "Automatic", "Automatic", "Automatic", "Automatic", "Automatic"],
    "Engine_CC": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
    "Price": [10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000]
}

df = pd.DataFrame(cars)

df_encoded = pd.get_dummies(df, drop_first=True)

x = df_encoded.drop(columns=["Price"])
y = df_encoded["Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=== predictions ===")
print(y_pred)
print("=== actual values ===")
print(y_test.values)
print("=== r2 ===")
print(r2)
print("=== rmse ===")
print(rmse)

joblib.dump(model, os.path.join(OUTPUTS_DIR, "car_price_model.pkl"))
df_encoded.to_csv(os.path.join(OUTPUTS_DIR, "processed_car_data.csv"), index=False)