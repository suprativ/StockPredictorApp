from alpha_vantage.timeseries import TimeSeries
import time

# Load your Alpha Vantage API Key
API_KEY = 'YOUR_ALPHA_VANTAGE_API_KEY'

def check_data_availability():
    # Initialize Alpha Vantage TimeSeries object
    ts = TimeSeries(key=API_KEY, output_format='pandas')

    # Load symbols from the file
    with open('constituents_symbols.txt', 'r') as file:
        lines = file.readlines()

    # Check each symbol for data availability
    for line in lines:
        try:
            symbol, name = line.strip().split(',')
            print(f"Checking {symbol} - {name}")
            data, _ = ts.get_daily(symbol=symbol, outputsize='compact')

            if not data.empty:
                print(f"Data available for {symbol}")
            else:
                print(f"No data for {symbol}")

        except Exception as e:
            print(f"Error for {symbol}: {e}")

        # Respect Alpha Vantage API rate limits
        time.sleep(12)  # 5 requests per minute limit

if __name__ == "__main__":
    check_data_availability()