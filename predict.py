import sys
import pandas as pd
import joblib
import xgboost as xgb

# Load model
model = joblib.load("model.joblib")

# Read inputs
trip_duration_days = float(sys.argv[1])
miles_traveled = float(sys.argv[2])
total_receipts_amount = float(sys.argv[3])

# Derived features
days_0 = int(trip_duration_days <= 2)
days_1 = int(2 < trip_duration_days <= 6)
days_2 = int(trip_duration_days > 6)

receipt_0 = float(total_receipts_amount <= 540.03)
receipt_1 = float(540.03 < total_receipts_amount <= 1526.58)
receipt_2 = float(total_receipts_amount > 1526.58)

miles_l100 = float(miles_traveled <= 100)
miles_m100_400 = float(100 < miles_traveled <= 400)
miles_g400 = float(miles_traveled > 400)

miles_per_day = miles_traveled / trip_duration_days if trip_duration_days != 0 else 0
receipt_per_day = total_receipts_amount / trip_duration_days if trip_duration_days != 0 else 0
receipt_per_mile = total_receipts_amount / miles_traveled if miles_traveled != 0 else 0

mpd_break = int(miles_per_day > 150)
rpd_break = int(receipt_per_day > 90)
rpm_break = int(receipt_per_mile > 0.45)

input_features = pd.DataFrame([[
    trip_duration_days, miles_traveled, total_receipts_amount,
    days_0, days_1, days_2,
    receipt_0, receipt_1, receipt_2,
    miles_l100, miles_m100_400, miles_g400,
    miles_per_day, receipt_per_day, receipt_per_mile,
    mpd_break, rpd_break, rpm_break
]], columns=[
    "trip_duration_days", "miles_traveled", "total_receipts_amount",
    "days_0", "days_1", "days_2",
    "receipt_0", "receipt_1", "receipt_2",
    "miles_l100", "miles_m100_400", "miles_g400",
    "miles_per_day", "receipt_per_day", "receipt_per_mile",
    "mpd_break", "rpd_break", "rpm_break"
])

# Predict
prediction = model.predict(input_features)[0]
print(prediction)
