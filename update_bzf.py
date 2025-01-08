import yfinance as yf
import hopsworks
from datetime import datetime, timedelta
import pandas as pd

# Function to fetch today's BZ=F data
def fetch_latest_bzf():
    today = datetime.utcnow().date()
    last_trading_day = today - timedelta(days=1)

    while last_trading_day.weekday() > 4:  # Skip weekends (Saturday=5, Sunday=6)
        last_trading_day -= timedelta(days=1)

    bzf = yf.download("BZ=F", start=last_trading_day.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"), interval="1d")

    if bzf.empty:
        print(f"No new data available for BZ=F. Checked for {last_trading_day}.")
        return None

    bzf = bzf[['Close']].rename(columns={'Close': 'bzf'})
    bzf.index = pd.to_datetime(bzf.index)
    bzf.reset_index(inplace=True)
    bzf.rename(columns={'Date': 'date'}, inplace=True)
    bzf.columns = bzf.columns.map(lambda col: col if isinstance(col, str) else col[0])

    print("Fetched BZ=F data:")
    print(bzf.head())

    return bzf


def update_bzf_feature_group(bzf_data):
    if bzf_data is None or bzf_data.empty:
        print("No data to insert into the feature group.")
        return

    project = hopsworks.login(project="ID2223HT24G4")
    fs = project.get_feature_store()
    bzf_feature_group = fs.get_feature_group(name="bzf", version=1)
    bzf_feature_group.insert(bzf_data)

    print("Successfully updated the BZ=F feature group!")

if __name__ == "__main__":
    bzf_data = fetch_latest_bzf()
    update_bzf_feature_group(bzf_data)
