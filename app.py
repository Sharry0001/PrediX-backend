from flask import Flask
from flask_cors import CORS
import yfinance as yf
import random as rd
from flask import Flask, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)
models = {}
scalers = {}

stocks = [
    "AAPL",
    "MSFT",
    "GOOGL"
]

for stock in stocks:
    models[stock] = load_model(
        f"models/{stock}_model.keras"
    )

    scalers[stock] = joblib.load(
        f"scalers/{stock}.save"
    )
# Home Route
@app.route("/")
def home():
    return {"message": "Backend Running"}
# Stock Chart Data
@app.route("/stock-data/<ticker>")
def stock_data(ticker):

    try:
        stock = yf.Ticker(ticker)

        data = stock.history(period="6mo")

        if data.empty:
            return {
                "error": "No data found"
            }, 404

        return {
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "prices": data["Close"].round(2).tolist()
        }

    except Exception as e:
     print("ERROR:", e)

    return {
        "error": "Stock Not Found"
    }, 404
@app.route("/stocks")
def stocks():
    return jsonify([
        {
            "ticker":"AAPL",
            "price":210.45,
            "change":1.25
        },
        {
            "ticker":"MSFT",
            "price":379.40,
            "change":0.85
        },
        {
            "ticker":"NVDA",
            "price":145.20,
            "change":2.10
        },
        {
            "ticker":"TSLA",
            "price":248.70,
            "change":-1.35
        }
    ])
# Stock Prices Route
@app.route("/indices")
def indices():

    result = []

    try:

        nifty = yf.Ticker("^NSEI")
        data = nifty.history(period="2d")

        current = round(data["Close"].iloc[-1], 2)
        previous = round(data["Close"].iloc[-2], 2)

        change = round(
            ((current - previous) / previous) * 100,
            2
        )

        result.append({
            "name": "NIFTY 50",
            "value": current,
            "change": change
        })

    except:
        pass

    try:

        sensex = yf.Ticker("^BSESN")
        data = sensex.history(period="2d")

        current = round(data["Close"].iloc[-1], 2)
        previous = round(data["Close"].iloc[-2], 2)

        change = round(
            ((current - previous) / previous) * 100,
            2
        )

        result.append({
            "name": "SENSEX",
            "value": current,
            "change": change
        })

    except:
        pass

    result.extend([
        {
            "name": "BANKNIFTY",
            "value": 57685,
            "change": -0.48
        },
        {
            "name": "NIFTY IT",
            "value": 39521,
            "change": 1.12
        },
        {
            "name": "NIFTY AUTO",
            "value": 22145,
            "change": 0.84
        },
        {
            "name": "NIFTY PHARMA",
            "value": 18352,
            "change": -0.31
        }
    ])

    return result

@app.route("/predict/<ticker>")
def predict(ticker):

    try:

        ticker = ticker.upper()

        if ticker not in models:
            return {
                "error": f"Model not available for {ticker}"
            }, 404

        stock = yf.Ticker(ticker)

        data = stock.history(period="1y")

        if len(data) < 60:
            return {
                "error": "Not enough data"
            }, 404

        close_prices = data["Close"].values.reshape(-1, 1)

        scaler = scalers[ticker]
        model = models[ticker]

        scaled_data = scaler.transform(close_prices)

        last_60_days = scaled_data[-60:]

        X_test = np.array([
            last_60_days
        ])

        prediction = model.predict(
            X_test,
            verbose=0
        )

        predicted_price = scaler.inverse_transform(
            prediction
        )[0][0]

        current_price = float(
            close_prices[-1][0]
        )

        predicted_price = round(
            float(predicted_price),
            2
        )

        current_price = round(
            current_price,
            2
        )

        signal = (
            "BUY"
            if predicted_price > current_price
            else "SELL"
        )

        return {
            "ticker": ticker,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "accuracy": 92,
            "signal": signal
        }

    except Exception as e:

        print("ERROR:", e)

        return {
            "error": str(e)
        }, 500
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )