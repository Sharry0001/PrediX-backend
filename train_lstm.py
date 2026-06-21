import os
import yfinance as yf
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.metrics import mean_absolute_percentage_error

# Create folders automatically
os.makedirs("models", exist_ok=True)
os.makedirs("scalers", exist_ok=True)

stocks = [
    "AAPL",
    "MSFT",
    "GOOGL"
]

for ticker in stocks:

    try:

        print(f"\n========== Training {ticker} ==========")

        data = yf.download(
            ticker,
            period="5y",
            progress=False
        )

        if data.empty:
            print(f"No data found for {ticker}")
            continue

        close_prices = data["Close"].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        X = []
        y = []

        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i - 60:i])
            y.append(scaled_data[i])

        X = np.array(X)
        y = np.array(y)

        # Shape fix
        y = y.reshape(-1, 1)
        split = int(len(X) * 0.8)

        X_train = X[:split]
        X_test = X[split:]

        y_train = y[:split]
        y_test = y[split:]

        print("Training samples:", len(X))

        model = Sequential([
            Input(shape=(60, 1)),
            LSTM(50, return_sequences=True),
            LSTM(50),
            Dense(25),
            Dense(1)
        ])

        model.compile(
            optimizer="adam",
            loss="mean_squared_error"
        )

        model.fit(
            X_train,
            y_train,
            epochs=3,
            batch_size=32,
            verbose=1
        )
        predictions = model.predict(X_test)

        accuracy = 100 - (
                  mean_absolute_percentage_error(
                  y_test,
                  predictions
            ) * 100
        )

        accuracy = round(float(accuracy), 2)

        print(f"Accuracy: {accuracy}%")
        model.save(
            f"models/{ticker}_model.keras"
        )

        joblib.dump(
            scaler,
            f"scalers/{ticker}.save"
        )

        print(f"✅ {ticker} model saved")
        
        joblib.dump(
        accuracy,
        f"scalers/{ticker}_accuracy.save"
        )
    except Exception as e:

        print(f"❌ Error training {ticker}: {e}")

print("\n🎉 All models trained successfully!")