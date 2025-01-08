import yfinance as yf
import hopsworks
from datetime import datetime, timedelta
import pandas as pd

# Function to fetch today's VIX data
def fetch_latest_vix():
    today = datetime.utcnow().date()
    last_trading_day = today - timedelta(days=1)

    while last_trading_day.weekday() > 4:  # Skip weekends (Saturday=5, Sunday=6)
        last_trading_day -= timedelta(days=1)

    vix = yf.download("^VIX", start=last_trading_day.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"), interval="1d")

    if vix.empty:
        print(f"No new data available for VIX. Checked for {last_trading_day}.")
        return None

    vix = vix[['Close']].rename(columns={'Close': 'vix'})
    vix.index = pd.to_datetime(vix.index)
    vix.reset_index(inplace=True)
    vix.rename(columns={'Date': 'date'}, inplace=True)
    vix.columns = vix.columns.map(lambda col: col if isinstance(col, str) else col[0])

    print("Fetched VIX data:")
    print(vix.head())

    return vix
def update_vix_feature_group(vix_data):
    if vix_data is None or vix_data.empty:
        print("No data to insert into the feature group.")
        return

    project = hopsworks.login(project="ID2223HT24G4")
    fs = project.get_feature_store()
    vix_feature_group = fs.get_feature_group(name="vix", version=1)
    vix_feature_group.insert(vix_data)

    print("Successfully updated the VIX feature group!")

if __name__ == "__main__":
    vix_data = fetch_latest_vix()
    update_vix_feature_group(vix_data)
