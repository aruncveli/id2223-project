import hopsworks
import pandas as pd
import tensorflow as tf
import joblib

api_key = ""
project = hopsworks.login(project="ID2223HT24G4" )
fs = project.get_feature_store()


# Get the data from the feature store
feature_view = fs.get_feature_view(name="rate_vix_bzf_omx_feature_view", version=1)
data = feature_view.get_batch_data()
data['date'] = pd.to_datetime(data['date'])
data.sort_values(by='date', inplace=True)

# Select the last 30 rows to form your input sequence
latest_30_df = data.tail(30).copy()
latest_30_df = latest_30_df.fillna(method='ffill')  # fill forward if there are NaNs
latest_30_df = latest_30_df.fillna(method='bfill')  # fill backward if there are NaNs

print("Latest 30 rows:\n", latest_30_df.head(30))

# Load the most recent model version
mr = project.get_model_registry()
model_versions = mr.get_models("lstm_rate_vix_omx_bzf_7day_v2")
model_to_load = model_versions[-1]  # Load the newest version
model_dir = model_to_load.download()

model_file = model_dir + "/rates_lstm_model_v2.keras"
loaded_model = tf.keras.models.load_model(model_file)

scaler_X_file = model_dir + "/scaler_x_v2.pkl"
scaler_y_file = model_dir + "/scaler_y_v2.pkl"

loaded_scaler_X = joblib.load(scaler_X_file)
loaded_scaler_y = joblib.load(scaler_y_file)

print("Model and scalers loaded successfully from:", model_dir)

######################
# 3) SCALING & PREDICTING
######################
FEATURE_COLS = ["rate", "vix", "omx", "bzf"]  # same order used in training

# Transform the last 30 rows
latest_30_values = latest_30_df[FEATURE_COLS].values  # shape (30, 4)
latest_30_scaled = loaded_scaler_X.transform(latest_30_values)

# Reshape to (1, 30, 4) for a single prediction batch
X_latest = latest_30_scaled.reshape(1, 30, 4)

# Predict => shape (1, 7) if you have a 7-output model
raw_pred = loaded_model.predict(X_latest)
# Invert scaling back to original rate scale => shape (1, 7)
y_pred_orig = loaded_scaler_y.inverse_transform(raw_pred)

# Extract day+1 and day+7 from that single row
# day+1 => index 0, day+7 => index 6
pred_rate_day1 = y_pred_orig[0, 0]
pred_rate_day7 = y_pred_orig[0, 6]

print("Predicted rate for 1 day ahead:", pred_rate_day1)
print("Predicted rate for 7 days ahead:", pred_rate_day7)

# Build small dataframes to insert into the feature store
forecast_df_pred_1day = pd.DataFrame({
    "date": [latest_30_df['date'].values[-1] + pd.Timedelta(days=1)],
    "rate": [pred_rate_day1]
})
forecast_df_pred_7days = pd.DataFrame({
    "date": [latest_30_df['date'].values[-1] + pd.Timedelta(days=7)],
    "rate": [pred_rate_day7]
})

print("1-day forecast row:\n", forecast_df_pred_1day)
print("7-day forecast row:\n", forecast_df_pred_7days)

######################
# 4) INSERTING THE FORECASTS INTO FEATURE GROUPS
######################
fg_forcast_1day = fs.get_or_create_feature_group(
    name="rate_vix_forecast_1day",
    version=1,
    description="Forecast for rate for 1 day ahead",
    primary_key=['date'],
)
fg_forcast_7days = fs.get_or_create_feature_group(
    name="rate_vix_forecast_7days",
    version=1,
    description="Forecast for rate for 7 days ahead",
    primary_key=['date'],
)

fg_forcast_1day.insert(forecast_df_pred_1day)
fg_forcast_7days.insert(forecast_df_pred_7days)

print("Forecast rows inserted into feature store!")


######################
# 5) BAR PLOT: 1 DAY VS 7 DAYS AHEAD
######################
import matplotlib.pyplot as plt
import os
predicted_7_days = y_pred_orig[0, :]  # shape (7,)
days = list(range(1, 8))

plt.figure(figsize=(8, 6))
plt.plot(days, predicted_7_days, marker="o", linestyle="-", color="blue")
plt.title("Predicted Rate for Each of the Next 7 Days")
plt.xlabel("Day Ahead")
plt.ylabel("Predicted Rate")
plt.xticks(days, [f"Day+{d}" for d in days])

plt.grid(linestyle="--", alpha=0.7)

# Save & show
plot_dir = "docs"
os.makedirs(plot_dir, exist_ok=True)
plot_file = os.path.join(plot_dir, "predicted_rates_7days_line.png")
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.show()
######################
# 6) COMPARE 1-DAY-AHEAD FORECASTS WITH ACTUAL
######################
rate_vix_forecast_1day_fg = fs.get_feature_group("rate_vix_forecast_1day", version=1)
rate_vix_forecast_1day_df = rate_vix_forecast_1day_fg.read()

# The 'sek_to_usd' feature group presumably has the actual daily rates
sek_to_usd_fg = fs.get_feature_group("sek_to_usd", version=1)
sek_to_usd_df = sek_to_usd_fg.read()

# Convert 'date' to datetime
rate_vix_forecast_1day_df['date'] = pd.to_datetime(rate_vix_forecast_1day_df['date'])
sek_to_usd_df['date'] = pd.to_datetime(sek_to_usd_df['date'])

# Merge on 'date'
comparison_1d_df = pd.merge(
    rate_vix_forecast_1day_df,
    sek_to_usd_df[['date', 'rate']],
    on="date",
    how="inner",
    suffixes=('_predicted', '_actual')
)

# Sort by date
comparison_1d_df.sort_values(by='date', inplace=True)

# Plot actual vs predicted (1 day)
plt.figure(figsize=(12, 6))
plt.plot(comparison_1d_df['date'], comparison_1d_df['rate_actual'], marker='o', label="Actual Rate", color="blue")
plt.plot(comparison_1d_df['date'], comparison_1d_df['rate_predicted'], marker='x', label="Predicted Rate (1 Day Ahead)", color="orange")
plt.title("Actual vs Predicted Rate (1 Day Ahead)")
plt.xlabel("Date")
plt.ylabel("Rate")
plt.xticks(rotation=45)
plt.legend()
plt.grid()

plot_path_1d = os.path.join(plot_dir, "1_day_actual_vs_prediction.png")
plt.savefig(plot_path_1d, dpi=300, bbox_inches="tight")
plt.show()
print(f"1-Day Actual vs Prediction graph saved to {plot_path_1d}")


######################
# 7) COMPARE 7-DAY-AHEAD FORECASTS WITH ACTUAL
######################
rate_vix_forecast_7days_fg = fs.get_feature_group("rate_vix_forecast_7days", version=1)
rate_vix_forecast_7days_df = rate_vix_forecast_7days_fg.read()

# We already have sek_to_usd_fg => sek_to_usd_df
rate_vix_forecast_7days_df['date'] = pd.to_datetime(rate_vix_forecast_7days_df['date'])
sek_to_usd_df['date'] = pd.to_datetime(sek_to_usd_df['date'])

comparison_7d_df = pd.merge(
    rate_vix_forecast_7days_df,
    sek_to_usd_df[['date', 'rate']],
    on="date",
    how="inner",
    suffixes=('_predicted', '_actual')
)

#sort
comparison_7d_df.sort_values(by='date', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(comparison_7d_df['date'], comparison_7d_df['rate_actual'], marker='o', label="Actual Rate", color="blue")
plt.plot(comparison_7d_df['date'], comparison_7d_df['rate_predicted'], marker='x', label="Predicted Rate (7 Days Ahead)", color="orange")
plt.title("Actual vs Predicted Rate (7 Days Ahead)")
plt.xlabel("Date")
plt.ylabel("Rate")
plt.xticks(rotation=45)
plt.legend()
plt.grid()

plot_path_7d = os.path.join(plot_dir, "7_day_actual_vs_prediction.png")
plt.savefig(plot_path_7d, dpi=300, bbox_inches="tight")
plt.show()
print(f"7-Day Actual vs Prediction graph saved to {plot_path_7d}")
