import yfinance as yf
import hopsworks
from datetime import datetime, timedelta
import pandas as pd

# Function to fetch today's OMX data
def fetch_latest_omx():
    today = datetime.utcnow().date()
    start_trading_day = today + timedelta(days=1)

    while last_trading_day.weekday() > 4:  # Skip weekends (Saturday=5, Sunday=6)
        last_trading_day -= timedelta(days=1)

    omx = yf.download("^OMX", start=today.strftime("%Y-%m-%d"), end=start_trading_day.strftime("%Y-%m-%d"), interval="1d")

    if omx.empty:
        print(f"No new data available for OMX. Checked for {last_trading_day}.")
        return None

    omx = omx[['Close']].rename(columns={'Close': 'omx'})
    omx.index = pd.to_datetime(omx.index)
    omx.reset_index(inplace=True)
    omx.rename(columns={'Date': 'date'}, inplace=True)
    omx.columns = omx.columns.map(lambda col: col if isinstance(col, str) else col[0])

    print("Fetched OMX data:")
    print(omx.head())

    return omx
def update_omx_feature_group(omx_data):
    if omx_data is None or omx_data.empty:
        print("No data to insert into the feature group.")
        return

    project = hopsworks.login(project="ID2223HT24G4")
    fs = project.get_feature_store()
    omx_feature_group = fs.get_feature_group(name="omx", version=1)
    omx_feature_group.insert(omx_data)

    print("Successfully updated the OMX feature group!")

if __name__ == "__main__":
    omx_data = fetch_latest_omx()
    update_omx_feature_group(omx_data)
