import json
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib

# Load and flatten public_cases.json
with open("public_cases.json", "r") as f:
    raw_data = json.load(f)

records = []
for row in raw_data:
    flat_row = row["input"]
    flat_row["expected_output"] = row["expected_output"]
    records.append(flat_row)

df = pd.DataFrame(records)

# Feature engineering with breakaway points
df["days_0"] = (df["trip_duration_days"] <= 2).astype(int)
df["days_1"] = ((df["trip_duration_days"] > 2) & (df["trip_duration_days"] <= 6)).astype(int)
df["days_2"] = (df["trip_duration_days"] > 6).astype(int)

df["receipt_0"] = (df["total_receipts_amount"] <= 540.03).astype(float)
df["receipt_1"] = ((df["total_receipts_amount"] > 540.03) & (df["total_receipts_amount"] <= 1526.58)).astype(float)
df["receipt_2"] = (df["total_receipts_amount"] > 1526.58).astype(float)

df["miles_l100"] = (df["miles_traveled"] <= 100).astype(float)
df["miles_m100_400"] = ((df["miles_traveled"] > 100) & (df["miles_traveled"] <= 400)).astype(float)
df["miles_g400"] = (df["miles_traveled"] > 400).astype(float)

df["miles_per_day"] = df["miles_traveled"] / df["trip_duration_days"].replace(0, 0.1)
df["receipt_per_day"] = df["total_receipts_amount"] / df["trip_duration_days"].replace(0, 0.1)
df["receipt_per_mile"] = df["total_receipts_amount"] / df["miles_traveled"].replace(0, 0.1)

df["mpd_break"] = (df["miles_per_day"] > 150).astype(int)
df["rpd_break"] = (df["receipt_per_day"] > 90).astype(int)
df["rpm_break"] = (df["receipt_per_mile"] > 0.45).astype(int)

features = [
    "trip_duration_days", "miles_traveled", "total_receipts_amount",
    "days_0", "days_1", "days_2",
    "receipt_0", "receipt_1", "receipt_2",
    "miles_l100", "miles_m100_400", "miles_g400",
    "miles_per_day", "receipt_per_day", "receipt_per_mile",
    "mpd_break", "rpd_break", "rpm_break"
]

X = df[features]
y = df["expected_output"]

# Train the model on full dataset
model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X, y)

# Evaluate
y_pred = model.predict(X)
print("Train MAE:", mean_absolute_error(y, y_pred))

# Save
joblib.dump(model, "model.joblib")
