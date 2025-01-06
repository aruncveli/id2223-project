import hopsworks
import pandas as pd
import tensorflow as tf
import joblib

api_key = ""
project = hopsworks.login(project="ID2223HT24G4" )
fs = project.get_feature_store()


feature_view = fs.get_feature_view(name="rate_vix_bzf_omx_feature_view", version=1)
data = feature_view.get_batch_data()
data['date'] = pd.to_datetime(data['date'])
data.sort_values(by='date', inplace=True)



latest_30_df = data.tail(30).copy()
latest_30_df = latest_30_df.fillna(method='ffill')

print("Latest 30 rows:\n", latest_30_df)


mr = project.get_model_registry()
model_versions = mr.get_models("lstm_rate_vix_omx_bzf_7day")
model_to_load = model_versions[-1]  # Load the latest version

model_dir = model_to_load.download()
model_file = model_dir + "/rates_lstm_model.keras"
loaded_model = tf.keras.models.load_model(model_file)

scaler_X_file = model_dir + "/scaler_x.pkl"
scaler_y_file = model_dir + "/scaler_y.pkl"
loaded_scaler_X = joblib.load(scaler_X_file)
loaded_scaler_y = joblib.load(scaler_y_file)

print("Model and scalers loaded successfully!")

FEATURE_COLS = ["rate", "vix", "omx", "bzf"]  # Ensure the column order matches training
latest_30_values = latest_30_df[FEATURE_COLS].values  # shape (30, 4)
latest_30_scaled = loaded_scaler_X.transform(latest_30_values)

X_latest = latest_30_scaled.reshape(1, 30, 4)

raw_pred = loaded_model.predict(X_latest)
y_pred_orig = loaded_scaler_y.inverse_transform(raw_pred)

pred_rate_1day = y_pred_orig[0][0]
pred_rate_7days = y_pred_orig[-1][-1]

print("Predicted rate for 1 day:", pred_rate_1day)
print("Predicted rate for 7 days:", pred_rate_7days)

forecast_df_pred_7days = pd.DataFrame({
    "date": latest_30_df['date'].values[-1] + pd.Timedelta(days=7),
    "pred_rate": pred_rate_7days
}, index=[0])

forecast_df_pred_7days.reset_index(drop=True, inplace=True)

forecast_df_pred_1day = pd.DataFrame({
    "date": latest_30_df['date'].values[-1] + pd.Timedelta(days=1),
    "pred_rate": pred_rate_1day
}, index=[0])

forecast_df_pred_1day.reset_index(drop=True, inplace=True)

forecast_df_pred_1day.columns = ['date', 'rate']
forecast_df_pred_7days.columns = ['date', 'rate']
# Example:
print(forecast_df_pred_1day)
print(forecast_df_pred_7days)

fg_forcast_1day = fs.get_or_create_feature_group(
    name="rate_vix_forecast_1day",
    version=1,
    description="Forecast for rate and vix for 1 day",
    primary_key=['date'],
)

fg_forcast_7days = fs.get_or_create_feature_group(
    name="rate_vix_forecast_7days",
    version=1,
    description="Forecast for rate and vix for 7 days",
    primary_key=['date'],
)

fg_forcast_1day.insert(forecast_df_pred_1day)
fg_forcast_7days.insert(forecast_df_pred_7days)

import matplotlib.pyplot as plt
import os

# Directory to save the plot
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Create the plot
plt.figure(figsize=(8, 6))
plt.bar(["1 Day Ahead", "7 Days Ahead"], [pred_rate_1day, pred_rate_7days], color=["blue", "orange"])
plt.title("Predicted Rate for 1 Day and 7 Days Ahead")
plt.ylabel("Predicted Rate")
plt.ylim(min(pred_rate_1day, pred_rate_7days) - 0.1, max(pred_rate_1day, pred_rate_7days) + 0.1)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Annotate the bar with the exact value
for i, value in enumerate([pred_rate_1day, pred_rate_7days]):
    plt.text(i, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=10)

# Save the plot
plot_file = os.path.join(plot_dir, "predicted_rates.png")
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.show()





# Fetch data from the feature groups
# 1-day predictions from "rate_vix_forecast_1day"
rate_vix_forecast_1day_fg = fs.get_feature_group("rate_vix_forecast_1day", version=1)
rate_vix_forecast_1day_df = rate_vix_forecast_1day_fg.read()

# Actual rates from "sek_to_usd"
sek_to_usd_fg = fs.get_feature_group("sek_to_usd", version=1)
sek_to_usd_df = sek_to_usd_fg.read()

# Convert 'date' columns to datetime for both feature groups
rate_vix_forecast_1day_df['date'] = pd.to_datetime(rate_vix_forecast_1day_df['date'])
sek_to_usd_df['date'] = pd.to_datetime(sek_to_usd_df['date'])

# Merge the two dataframes on 'date' for comparison
comparison_df = pd.merge(
    rate_vix_forecast_1day_df,
    sek_to_usd_df[['date', 'rate']],
    on="date",
    how="inner",
    suffixes=('_predicted', '_actual')
)

# Plot Actual vs Predicted rates
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['date'], comparison_df['rate_actual'], marker='o', label="Actual Rate", color="blue")
plt.plot(comparison_df['date'], comparison_df['rate_predicted'], marker='x', label="Predicted Rate (1 Day Ahead)", color="orange")
plt.title("Actual vs Predicted Rate (1 Day Ahead)")
plt.xlabel("Date")
plt.ylabel("Rate")
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Save the plot
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, "1_day_actual_vs_prediction.png")
plt.savefig(plot_path)
plt.show()

print(f"1-Day Actual vs Prediction graph saved to {plot_path}")


# Fetch data from the feature groups
# 1-day predictions from "rate_vix_forecast_1day"
rate_vix_forecast_7days_fg = fs.get_feature_group("rate_vix_forecast_7days", version=1)
rate_vix_forecast_7days_df = rate_vix_forecast_7days_fg.read()

# Actual rates from "sek_to_usd"
sek_to_usd_fg = fs.get_feature_group("sek_to_usd", version=1)
sek_to_usd_df = sek_to_usd_fg.read()

# Convert 'date' columns to datetime for both feature groups
rate_vix_forecast_7days_df['date'] = pd.to_datetime(rate_vix_forecast_7days_df['date'])
sek_to_usd_df['date'] = pd.to_datetime(sek_to_usd_df['date'])

# Merge the two dataframes on 'date' for comparison
comparison_df = pd.merge(
    rate_vix_forecast_7days_df,
    sek_to_usd_df[['date', 'rate']],
    on="date",
    how="inner",
    suffixes=('_predicted', '_actual')
)

# Plot Actual vs Predicted rates
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['date'], comparison_df['rate_actual'], marker='o', label="Actual Rate", color="blue")
plt.plot(comparison_df['date'], comparison_df['rate_predicted'], marker='x', label="Predicted Rate (7 Day Ahead)", color="orange")
plt.title("Actual vs Predicted Rate (7 Day Ahead)")
plt.xlabel("Date")
plt.ylabel("Rate")
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Save the plot
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, "7_day_actual_vs_prediction.png")
plt.savefig(plot_path)
plt.show()

print(f"7-Day Actual vs Prediction graph saved to {plot_path}")
