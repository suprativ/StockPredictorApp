# This is app.py

import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to avoid GUI warnings
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import yfinance as yf
import io
import base64
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Use Regressor for continuous data

app = Flask(__name__)

# Suprativ variable for copyright
suprativ_app_name = "Suprativ Stock Predictor"

def get_data(symbol, is_forex=False):
    if is_forex:
        symbol = symbol + "=X"  # Append '=X' for forex symbols
    data = yf.Ticker(symbol)
    hist = data.history(period="5y")
    return hist

# FIX: Function to predict for multiple time periods with error handling
def predict_data(symbol, is_forex=False, days_ahead=1):
    data = get_data(symbol, is_forex)
    if data.empty:
        return None, None, None, None

    # FIX: Ensure there's enough data for the prediction
    if len(data) < days_ahead:
        return None, None, None, None  # Not enough data for the selected period

    # Prepare data for prediction
    data['Prediction'] = data['Close'].shift(-days_ahead)
    X = np.array(data[['Close']])
    X = X[:-days_ahead]
    y = np.array(data['Prediction'])
    y = y[:-days_ahead]

    # FIX: Ensure there's enough data for train-test split
    if len(X) == 0 or len(y) == 0:
        return None, None, None, None

    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    except ValueError:
        return None, None, None, None  # Handle insufficient data for splitting

    model = RandomForestRegressor()  # Use Regressor for continuous data
    model.fit(X_train, y_train)

    # Predict
    accuracy = model.score(X_test, y_test)
    prediction = model.predict(X_test)
    price_direction = "rise" if prediction[-1] > data['Close'].iloc[-1] else "fall"

    return accuracy, prediction, data, price_direction

@app.route('/', methods=['GET', 'POST'])
def index():
    with open('available_stocks.txt', 'r') as f:
        stocks = [line.strip().split(',') for line in f.readlines()]
    with open('available_forex.txt', 'r') as f:
        forex = [line.strip().split(',') for line in f.readlines()]

    stock_graph_url = None
    stock_accuracy = None
    stock_price_direction = None
    stock_recommendation = None

    forex_graph_url = None
    forex_accuracy = None
    forex_price_direction = None
    forex_recommendation = None

    # FIX: Limit time periods to 1d, 1w, 1m, 6m, 1y, 2y, 3y
    time_periods = [1, 7, 30, 180, 365, 730, 1095]  # Days for 1d, 1w, 1m, 6m, 1y, 2y, 3y
    selected_period = 1  # Default to 1 day

    if request.method == 'POST':
        if 'stock_symbol' in request.form:
            symbol = request.form['stock_symbol']
            selected_period = int(request.form.get('time_period', 1))  # Get selected time period
            accuracy, prediction, data, price_direction = predict_data(symbol, is_forex=False, days_ahead=selected_period)
            if data is not None and not data.empty:
                plt.figure(figsize=(10, 5))
                plt.plot(data['Close'])
                plt.title(f"{symbol} Stock Price Prediction")
                plt.xlabel('Date')
                plt.ylabel('Close Price')
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                stock_graph_url = base64.b64encode(img.getvalue()).decode()
                stock_accuracy = f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A"
                stock_price_direction = f"The price is predicted to {price_direction} in {selected_period} days." if price_direction else "N/A"
                stock_recommendation = "Invest" if price_direction == "rise" else "Do not invest" if price_direction else "N/A"

        if 'forex_symbol' in request.form:
            symbol = request.form['forex_symbol']
            selected_period = int(request.form.get('time_period', 1))  # Get selected time period
            accuracy, prediction, data, price_direction = predict_data(symbol, is_forex=True, days_ahead=selected_period)
            if data is not None and not data.empty:
                plt.figure(figsize=(10, 5))
                plt.plot(data['Close'])
                plt.title(f"{symbol} Forex Price Prediction")
                plt.xlabel('Date')
                plt.ylabel('Close Price')
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                forex_graph_url = base64.b64encode(img.getvalue()).decode()
                forex_accuracy = f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A"
                forex_price_direction = f"The price is predicted to {price_direction} in {selected_period} days." if price_direction else "N/A"
                forex_recommendation = "Invest" if price_direction == "rise" else "Do not invest" if price_direction else "N/A"

    return render_template(
        'index.html',
        stocks=stocks,
        forex=forex,
        app_name=suprativ_app_name,
        stock_graph_url=stock_graph_url,
        stock_accuracy=stock_accuracy,
        stock_price_direction=stock_price_direction,
        stock_recommendation=stock_recommendation,
        forex_graph_url=forex_graph_url,
        forex_accuracy=forex_accuracy,
        forex_price_direction=forex_price_direction,
        forex_recommendation=forex_recommendation,
        time_periods=time_periods,  # Pass time periods to the template
        selected_period=selected_period  # Pass selected period to the template
    )

if __name__ == '__main__':
    app.run(debug=False)  # Disable debug mode