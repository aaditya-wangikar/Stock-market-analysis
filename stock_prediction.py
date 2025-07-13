# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 18:10:20 2025

@author: lenovo
"""

import os
import warnings
import pdb
from datetime import timedelta
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from statsmodels.formula.api import ols
from statsmodels.tsa.arima.model import ARIMA

import yfinance as yf

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
import tensorflow as tf
import ta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
warnings.filterwarnings('ignore')

def display_top_stocks(tickers, period):
    # Download last 6 months of data
    data = yf.download(tickers, period=period, interval="1d")['Close']

    # Drop columns with all NaNs (if any ticker didn't download)
    data = data.dropna(axis=1, how='all')

    # Calculate percentage return
    returns = (data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100
    returns = returns.sort_values(ascending=False)

    # Display best performing stocks
    print(f"📈 Best Performing Stocks in {period}:")
    print(returns)
    returns.to_csv(f"returns_{period}.csv", header=["% Return"])
    
    # Compute volatility
    daily_returns = data.pct_change().dropna()

    volatility = daily_returns.std() * 100  # Daily volatility in %
    annual_volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized
    
    # Combine into DataFrame
    vol_df = pd.DataFrame({
        "Daily Volatility (%)": volatility.round(2),
        "Annualized Volatility (%)": annual_volatility.round(2)
    }).sort_values(by="Annualized Volatility (%)", ascending=False)
    
    # Save to CSV
    vol_df.to_csv(f"Files//volatility_{period}.csv")
    
    # Print
    print(f"📊 Volatility (period: {period}) saved to 'volatility_{period}.csv'")
    print(vol_df)
    
    
    
    # Get top 10 performers
    top_10 = returns.sort_values(ascending=False).head(10).index.tolist()

    # Normalize data for top 10
    normalized = data[top_10] / data[top_10].iloc[0] * 100
    
    # Plot
    plt.figure(figsize=(12, 6))
    for col in normalized.columns:
        plt.plot(normalized.index, normalized[col], label=col)

    plt.title(f"Top 10 Performing Stocks ({period})")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (Start = 100)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot 10 most stable stocks
    top_stable = vol_df.tail(10)
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_stable.index, top_stable["Annualized Volatility (%)"], color='teal')
    plt.xlabel("Annualized Volatility (%)")
    plt.title("10 Most Stable Stocks (Lowest Volatility)")
    plt.gca().invert_yaxis()  # lowest volatility at top
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()
    
    # --- Combine into DataFrame ---
    df = pd.DataFrame({
        "Returns (%)": returns.round(2),
        "Annualized Volatility (%)": annual_volatility.round(2)
    })
    
    # --- Sharpe Ratio (returns / risk) ---
    df["Sharpe Ratio"] = (df["Returns (%)"] / df["Annualized Volatility (%)"]).round(2)
    
    # --- Selection Criteria ---
    median_vol = df["Annualized Volatility (%)"].median()
    
    # Top 5 by Sharpe
    top_sharpe = df.sort_values(by="Sharpe Ratio", ascending=False).head(5)
    
    # High-return, low-volatility filter
    filtered = df[(df["Returns (%)"] > 5) & (df["Annualized Volatility (%)"] < median_vol)]
    
    # --- Save Results ---
    df.to_csv(f"Files//full_analysis_{period}.csv")
    top_sharpe.to_csv(f"top_sharpe_{period}.csv")
    filtered.to_csv(f"Files//high_return_low_volatility_{period}.csv")
    
    # --- Display Output ---
    print(f"\n✅ Full analysis saved to 'full_analysis_{period}.csv'")
    print("\n📊 Top 5 by Sharpe Ratio:")
    print(top_sharpe)
    
    print("\n📉 High-return, Low-volatility Picks:")
    print(filtered)
    
    # Top 3 by Sharpe and Returns
    top_sharpe_tickers = df.sort_values(by="Sharpe Ratio", ascending=False).head(3).index
    top_return_tickers = df.sort_values(by="Returns (%)", ascending=False).head(3).index
    
    # Assign colors
    colors = []
    for ticker in df.index:
        if ticker in top_sharpe_tickers:
            colors.append('green')
        elif ticker in top_return_tickers:
            colors.append('blue')
        else:
            colors.append('gray')
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Sharpe Ratio"], df["Returns (%)"], c=colors, s=100, edgecolors='black')
    
    # Annotate each point
    for ticker in df.index:
        plt.annotate(ticker, 
                     (df.loc[ticker, "Sharpe Ratio"], df.loc[ticker, "Returns (%)"]),
                     textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9)
    
    # Labels and styling
    plt.xlabel("Sharpe Ratio (Return / Volatility)")
    plt.ylabel("Returns (%)")
    plt.title("Sharpe Ratio vs Returns (Top Performers Highlighted)")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def moving_average(tickers):
    df = yf.download(tickers, period="5y", interval="1d")
    discription = df.describe()

    # Calculate 50 DMA and 200 DMA
    df['DMA_50'] = df['Close'].rolling(window=50).mean()
    df['DMA_200'] = df['Close'].rolling(window=200).mean()

    # Plot closing prices vs 50DMA and 200DMA
    sns.set_style('darkgrid')
    plt.figure(figsize = (7,5), dpi = 150)
    plt.title(f'Closing Prices vs 50 DMA & 200 DMA for {tickers[0].split(".")[0]}')
    plt.plot(df['Close'],label = 'Closing Price')
    plt.plot(df['DMA_50'],label = 'DMA_50')
    plt.plot(df['DMA_200'],label = 'DMA_200')
    plt.legend()
    plt.savefig(f'Files//Closing_Prices_{tickers[0].split(".")[0]}.png')

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'].to_numpy().flatten(), mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['DMA_50'].to_numpy().flatten(), mode='lines', name='DMA_50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['DMA_200'].to_numpy().flatten(), mode='lines', name='DMA_200'))
    
    fig.update_layout(
        title={
        'text': f'Closing Prices vs 50 DMA & 200 DMA for {tickers[0].split(".")[0]}',
        'x': 0.5,  # Center the title
        'xanchor': 'center'
        },
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        template='plotly_white',
        font=dict(
        family='Times New Roman',
        size=14
        ),
        autosize=True,
        height=800,
        width=1400
    )
    fig.write_html(f'Files//Closing_Prices_{tickers[0].split(".")[0]}.html', auto_open=True)

    # Analyze correlation
    plt.figure(figsize = (7,7), dpi = 150)
    sns.heatmap(df.corr(),annot=True)

    # Plot distplot
    sns.set_style('darkgrid')
    plt.figure(figsize = (7,5), dpi = 150)
    plt.title('Distplot 50DMA')
    sns.distplot(df['DMA_50'])

    sns.set_style('darkgrid')
    plt.figure(figsize = (7,5), dpi = 150)
    plt.title('Distplot 200DMA')
    sns.distplot(df['DMA_200'])


    df['Signal'] = 0
    df.loc[df['DMA_50'] > df['DMA_200'], 'Signal'] = 1
    df.loc[df['DMA_50'] < df['DMA_200'], 'Signal'] = -1
    # Detect crossover points
    df['Crossover'] = df['Signal'].diff()
    # Show crossover dates
    crossovers = df[df['Crossover'].abs() == 2]  # 2 = change from 1 to -1 or vice versa

    # Golden Cross = 50_DMA crosses above 200_DMA → Signal changes from -1 to 1 (Crossover = 2)
    # Death Cross = 50_DMA crosses below 200_DMA → Signal changes from 1 to -1 (Crossover = -2)

    for date, row in crossovers.iterrows():
        if row['Crossover'].to_numpy()[0] == 2:
            print(f"🔔 Golden Cross on {date.date()} — Possible Uptrend!")
        elif row['Crossover'].to_numpy()[0] == -2:
            print(f"⚠️ Death Cross on {date.date()}  — Possible Downtrend!")


    # model = ARIMA(df['Close'], order=(5, 1, 0))  # (p,d,q) — can be auto-tuned
    # model_fit = model.fit()

    # forecast_steps = 5
    # forecast = model_fit.forecast(steps=forecast_steps)

    # # Step 4: Create date range for future days
    # last_date = df.index[-1]
    # future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='B')

    # # Step 5: Print forecasted prices with dates
    # print(f"\n📅 Predicted closing prices for {tickers}:")
    # for date, price in zip(future_dates, forecast):
    #     print(f"{date.date()} → ₹{price:.2f}")

def support_resistance(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    df.dropna(inplace=True)
    
    # Identify local minima and maxima for support and resistance
    n = 5  # Number of points to consider for local extrema
    # Find extrema
    min_idx = argrelextrema(df['Low'].values, np.less_equal, order=n)[0]
    max_idx = argrelextrema(df['High'].values, np.greater_equal, order=n)[0]
    
    # Fill NaN initially
    df['min'] = np.nan
    df['max'] = np.nan
    
    # ✅ Use iloc for position-based assignment
    df.iloc[min_idx, df.columns.get_loc('min')] = df.iloc[min_idx, df.columns.get_loc('Low')]
    df.iloc[max_idx, df.columns.get_loc('max')] = df.iloc[max_idx, df.columns.get_loc('High')]
    
    df1 = df.copy()
    # Remove NaN for support and resistance lines
    df1.columns = [''.join(col) if isinstance(col, tuple) else col for col in df.columns]
    support_points = df1.dropna(subset=['min'])
    resistance_points = df1.dropna(subset=['max'])
    
    # Initialize candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlestick"
    )])
    
    # Add support line
    if len(support_points) >= 2:
        support_line = np.polyfit(support_points.index.map(pd.Timestamp.toordinal), support_points['min'], 1)
        support_trend = np.poly1d(support_line)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=support_trend(df.index.map(pd.Timestamp.toordinal)),
            mode='lines',
            line=dict(color='green', dash='dash'),
            name='Support Trendline'
        ))
    
    # Add resistance line
    if len(resistance_points) >= 2:
        resistance_line = np.polyfit(resistance_points.index.map(pd.Timestamp.toordinal), resistance_points['max'], 1)
        resistance_trend = np.poly1d(resistance_line)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=resistance_trend(df.index.map(pd.Timestamp.toordinal)),
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Resistance Trendline'
        ))
    
    # Convert dates to ordinal for fitting
    x_dates = df.index.map(pd.Timestamp.toordinal)
    
    # Support trendline
    if len(support_points) >= 2:
        support_fit = np.polyfit(support_points.index.map(pd.Timestamp.toordinal), support_points['min'], 1)
        support_trend = np.poly1d(support_fit)
        support_line = support_trend(x_dates)
    else:
        support_line = [np.nan] * len(df)
    
    # Resistance trendline
    if len(resistance_points) >= 2:
        resistance_fit = np.polyfit(resistance_points.index.map(pd.Timestamp.toordinal), resistance_points['max'], 1)
        resistance_trend = np.poly1d(resistance_fit)
        resistance_line = resistance_trend(x_dates)
    else:
        resistance_line = [np.nan] * len(df)
    
    # --- Step 5: Plot everything ---
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Close', color='black')
    plt.scatter(df.index, df['min'], label='Support Points', color='green', marker='^')
    plt.scatter(df.index, df['max'], label='Resistance Points', color='red', marker='v')
    
    # Add trendlines
    plt.plot(df.index, support_line, color='green', linestyle='--', label='Support Line')
    plt.plot(df.index, resistance_line, color='red', linestyle='--', label='Resistance Line')
    
    print("""
            📉 Black line: Daily close price

            🟢 Dotted green line: Support trendline
            
            🔴 Dotted red line: Resistance trendline
            
            ⬆️ Green triangles: Local support points
            
            🔻 Red triangles: Local resistance points
          """)
    
    plt.title(f"Support and Resistance Trendlines - {ticker[0].split('.')[0]}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"Files//Support_Resistance_Trendlines_{ticker[0].split('.')[0]}.png")
    plt.show()
    
def RSI(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    df = df.reset_index()
    df.dropna(inplace=True)
    # Compute daily price change ---
    delta = df['Close'].diff()
    # Separate gains and losses ---
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Compute exponential moving average (recommended for RSI) ---
    period = 14
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Use exponential average (Wilder's method)
    avg_gain = avg_gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = avg_loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # --- Step 5: Calculate RS and RSI ---
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    print("""
The Relative Strength Index (RSI) is a technical indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.
          
          RSI Value	      Signal
          ---------------------------
          >70	          Overbought
          <30             Oversold
          45–55	          Neutral / Trend
          """)
    
    # --- Step 6: Plot RSI ---
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['RSI'], label='RSI', color='orange')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title(f"RSI - {ticker[0]}")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Files//RSI_{ticker[0]}.png")
    plt.show()

def intraday(ticker):
    
    # --- 1. Get intraday data ---
    df = yf.download(ticker, period="1d", interval="5m")
    df.dropna(inplace=True)
    
    # --- 2. Compute Indicators ---
    
    # EMA
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # VWAP
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Cum_TP_Volume'] = (df['Typical_Price'].to_numpy() * df['Volume'].to_numpy().flatten()).cumsum()
    df['Cum_Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cum_TP_Volume'] / df['Cum_Volume']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'].to_numpy()+(2 * df['Close'].rolling(window=20).std()).to_numpy().flatten()
    df['BB_Lower'] = df['BB_Middle'].to_numpy()-(2 * df['Close'].rolling(window=20).std()).to_numpy().flatten()
    
    # --- Generate Buy/Sell signals based on EMA crossover ---
    df['Signal_EMA'] = np.where((df['EMA_9'] > df['EMA_21']) & (df['EMA_9'].shift(1) <= df['EMA_21'].shift(1)), 'Buy',
                         np.where((df['EMA_9'] < df['EMA_21']) & (df['EMA_9'].shift(1) >= df['EMA_21'].shift(1)), 'Sell', ''))
    
    # --- RSI signals ---
    df['Signal_RSI'] = np.where(df['RSI_14'] < 30, 'Buy (RSI Oversold)',
                         np.where(df['RSI_14'] > 70, 'Sell (RSI Overbought)', ''))
    # --- VWAP and Bollinger Band alerts ---
    df['Alert_VWAP'] = np.where(df['Close'].to_numpy().flatten() > df['VWAP'].to_numpy(), 'Above VWAP', 
                         np.where(df['Close'].to_numpy().flatten() < df['VWAP'].to_numpy(), 'Below VWAP', ''))
    df['Alert_BB'] = np.where(df['Close'].to_numpy().flatten() > df['BB_Upper'].to_numpy().flatten(), 'Touching Upper BB',
                       np.where(df['Close'].to_numpy().flatten() < df['BB_Lower'].to_numpy().flatten(), 'Touching Lower BB', ''))
    
    # --- Display last signals ---
    latest_signals = df[['Close', 'EMA_9', 'EMA_21', 'RSI_14', 'VWAP', 'BB_Upper', 'BB_Lower',
                         'Signal_EMA', 'Signal_RSI', 'Alert_VWAP', 'Alert_BB']].dropna().tail(10)
    
    print("\n🔔 Latest Signals and Alerts:")
    print(latest_signals)
    
    # --- 3. Plotting ---
    
    plt.figure(figsize=(16, 10))
    
    # --- Subplot 1: Price + EMA + VWAP + Bollinger Bands ---
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Close'], label='Close', color='black')
    plt.plot(df.index, df['EMA_9'], label='EMA 9', color='blue')
    plt.plot(df.index, df['EMA_21'], label='EMA 21', color='red')
    plt.plot(df.index, df['VWAP'], label='VWAP', color='orange', linestyle='--')
    plt.plot(df.index, df['BB_Upper'], label='BB Upper', color='green', linestyle='--', alpha=0.4)
    plt.plot(df.index, df['BB_Lower'], label='BB Lower', color='green', linestyle='--', alpha=0.4)
    plt.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='green', alpha=0.1)
    plt.title(f"{ticker}[0] - Intraday Indicators")
    plt.legend()
    plt.grid()
    
    # --- Subplot 2: RSI ---
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['RSI_14'], label='RSI (14)', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought')
    plt.axhline(30, color='green', linestyle='--', label='Oversold')
    plt.title("RSI")
    plt.legend()
    plt.grid()
    
    # --- Subplot 3: MACD ---
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['MACD'], label='MACD', color='blue')
    plt.plot(df.index, df['Signal'], label='Signal Line', color='orange')
    plt.fill_between(df.index, df['MACD'] - df['Signal'], color='gray', alpha=0.2)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title("MACD")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(f"Files//{ticker[0]} - Intraday Indicators.png")
    plt.show()

    # --- Optional: Plot Close price with Buy/Sell points (EMA crossover) ---
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='black')
    plt.plot(df.index, df['EMA_9'], label='EMA 9', color='blue', alpha=0.6)
    plt.plot(df.index, df['EMA_21'], label='EMA 21', color='red', alpha=0.6)
    
    # Mark buy/sell points
    buy_signals = df[df['Signal_EMA'] == 'Buy']
    sell_signals = df[df['Signal_EMA'] == 'Sell']
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', s=100)
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', s=100)
    
    plt.title(f"{ticker} - Intraday Signals")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def predict_future_values(ticker):
    df = yf.download(ticker, period='1y')[['Close', 'Volume']].dropna()

    # Set seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # --- 1. Preprocess Data ---
    df = yf.download(ticker, period='1y')
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df.dropna(inplace=True)
    
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # --- 2. Prepare Sequences ---
    sequence_length = 60
    X, y = [], []
    
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i][0])  # Close
    
    X, y = np.array(X), np.array(y)
    
    # --- 3. Define LSTM with Dropout Enabled at Inference ---
    def create_mc_dropout_model(input_shape):
        inputs = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True)(inputs)
        x = Dropout(0.3)(x, training=True)  # Dropout active during inference
        x = LSTM(64)(x)
        x = Dropout(0.3)(x, training=True)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        return model
    
    model = create_mc_dropout_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=25, batch_size=32, verbose=1)
    
    # --- 4. Recursive Forecast with MC Dropout ---
    n_days = 7
    n_simulations = 30
    last_seq = scaled_data[-sequence_length:]
    future_predictions = []
    
    for day in range(n_days):
        preds = []
        for _ in range(n_simulations):
            input_seq = last_seq.reshape(1, sequence_length, len(features))
            pred = model.predict(input_seq, verbose=0)[0][0]
            preds.append(pred)
    
        mean_pred = np.mean(preds)
        std_pred = np.std(preds)
    
        # Inverse scale Close price
        dummy = np.zeros((1, len(features)))
        dummy[0][0] = mean_pred
        price = scaler.inverse_transform(dummy)[0][0]
    
        # Confidence ~ low std → high confidence
        confidence = max(0.0, 1 - std_pred * 10)  # crude scale from 1 to 0
    
        future_predictions.append((price, confidence))
    
        # Add the predicted row back into the sequence
        new_row = last_seq[-1].copy()
        new_row[0] = mean_pred
        last_seq = np.vstack([last_seq[1:], new_row])
    
    # --- 5. Display Results ---
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_days)
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': [p[0] for p in future_predictions],
        'Confidence (0-1)': [round(p[1], 3) for p in future_predictions]
    })
    
    print(forecast_df)
    forecast_df.to_csv(f'Files//Forecasted_vales_{ticker[0].split(".")[0]}.csv')
    
    # --- 6. Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[-60:], df['Close'].iloc[-60:], label='Historical', linewidth=2)
    plt.plot(forecast_df['Date'], forecast_df['Predicted_Close'], marker='o', linestyle='--', color='orange', label='Forecast')
    plt.fill_between(forecast_df['Date'],
                     forecast_df['Predicted_Close'] * (1 - 0.05),
                     forecast_df['Predicted_Close'] * (1 + 0.05),
                     alpha=0.2, color='orange', label='±5% band')
    plt.title(f'{ticker[0].split(".")[0]} - Next 7 Day Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Predicted Close Price (INR)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{ticker[0].split(".")[0]}_Next7_Day_Price_Forecast.png')
    plt.show()
    
    threshold_pct = 0.005  # 0.5%
    confidence_threshold = 0.85
    
    signals = []
    for i in range(1, len(forecast_df)):
        prev_price = forecast_df.loc[i-1, 'Predicted_Close']
        curr_price = forecast_df.loc[i, 'Predicted_Close']
        confidence = forecast_df.loc[i, 'Confidence (0-1)']
        
        price_change_pct = (curr_price - prev_price) / prev_price
        
        if (price_change_pct > threshold_pct) and (confidence >= confidence_threshold):
            signals.append('BUY')
        elif (price_change_pct < -threshold_pct) and (confidence >= confidence_threshold):
            signals.append('SELL')
        else:
            signals.append('HOLD')
    
    # First day has no previous price to compare
    signals.insert(0, 'HOLD')
    
    forecast_df['Signal'] = signals
    plt.figure(figsize=(10, 5))
    plt.plot(forecast_df['Date'], forecast_df['Predicted_Close'], marker='o', label='Forecasted Price', color='navy')
    
    # Mark BUY and SELL
    for i in range(len(forecast_df)):
        if forecast_df.loc[i, 'Signal'] == 'BUY':
            plt.scatter(forecast_df.loc[i, 'Date'], forecast_df.loc[i, 'Predicted_Close'], color='green', label='BUY' if i == 0 else "", marker='^', s=100)
        elif forecast_df.loc[i, 'Signal'] == 'SELL':
            plt.scatter(forecast_df.loc[i, 'Date'], forecast_df.loc[i, 'Predicted_Close'], color='red', label='SELL' if i == 0 else "", marker='v', s=100)
    
    plt.title(f"{ticker} - 7-Day Forecast with Buy/Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Predicted Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()