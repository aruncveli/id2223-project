import yfinance as yf
import hopsworks
from datetime import datetime, timedelta
import pandas as pd

# Function to fetch today's SEK to USD rate
def fetch_latest_sek_to_usd():
    today = datetime.utcnow().date()
    start_trading_day = today + timedelta(days=1)

    while last_trading_day.weekday() > 4:  # Skip weekends (Saturday=5, Sunday=6)
        last_trading_day -= timedelta(days=1)

    sek_to_usd = yf.download("SEK=X", start=today.strftime("%Y-%m-%d"), end=start_trading_day.strftime("%Y-%m-%d"), interval="1d")

    if sek_to_usd.empty:
        print(f"No new data available for VIX. Checked for {last_trading_day}.")
        return None

    sek_to_usd = sek_to_usd[['Close']].rename(columns={'Close': 'rate'})
    sek_to_usd.index = pd.to_datetime(sek_to_usd.index)
    sek_to_usd.reset_index(inplace=True)
    sek_to_usd.rename(columns={'Date': 'date'}, inplace=True)
    sek_to_usd.columns = sek_to_usd.columns.map(lambda col: col if isinstance(col, str) else col[0])

    print("Fetched SEK to USD data:")
    print(sek_to_usd.head())

    return sek_to_usd

def update_sek_to_usd_feature_group(sek_to_usd):
    if sek_to_usd is None or sek_to_usd.empty:
        print("No data to insert into the feature group.")
        return

    project = hopsworks.login(project="ID2223HT24G4")
    fs = project.get_feature_store()
    vix_feature_group = fs.get_feature_group(name="sek_to_usd", version=1)
    vix_feature_group.insert(sek_to_usd)

    print("Successfully updated the sek_to_usd feature group!")

if __name__ == "__main__":
    sek_to_usd = fetch_latest_sek_to_usd()
    update_sek_to_usd_feature_group(sek_to_usd)
