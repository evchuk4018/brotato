import time
import random
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
MODEL_FILE = "smart_trader.keras"
TICKERS = [
    "AMD", "INTC", "CSCO", "ORCL", "IBM", "F", "GM", "TM", 
    "JPM", "BAC", "WFC", "C", "GS", "KO", "PEP", "MCD", 
    "SBUX", "WMT", "TGT", "DIS", "NFLX", "PFE", "JNJ", 
    "MRK", "XOM", "CVX", "BA", "GE", "CAT"
]
BLACKLIST = ["NVDA", "AAPL", "MSFT", "GOOG", "AMZN"] 

# Fake Wallet
STARTING_CASH = 10000.0
CASH = STARTING_CASH
PORTFOLIO = {} # Stores shares: {"F": 10, "AMD": 5}
LOOKBACK = 60 

def get_live_data(ticker):
    """Fetches enough recent data to normalize and create a sequence."""
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        
        if 'Close' in df.columns:
            if isinstance(df.columns, pd.MultiIndex):
                series = df['Close'].iloc[:, 0]
            else:
                series = df['Close']
            return series.dropna().values
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def get_portfolio_value():
    """Calculates total net worth (Cash + Stock Value)."""
    total_value = CASH
    print("\n--- Auditing Portfolio ---")
    for ticker, shares in PORTFOLIO.items():
        try:
            # We fetch a quick price check for accuracy
            data = get_live_data(ticker)
            if data is not None:
                current_price = data[-1]
                val = shares * current_price
                total_value += val
                print(f"{ticker}: {shares} shares @ ${current_price:.2f} = ${val:.2f}")
        except:
            print(f"Could not value {ticker}")
    
    profit = total_value - STARTING_CASH
    print(f"TOTAL NET WORTH: ${total_value:.2f} (Profit: ${profit:.2f})")
    print("--------------------------")
    return total_value

def predict_and_trade(model, ticker):
    global CASH, PORTFOLIO
    
    # 1. Get Data
    prices = get_live_data(ticker)
    if prices is None or len(prices) < LOOKBACK + 2:
        print(f"Not enough data for {ticker}. Skipping.")
        return

    # 2. Live Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_reshaped = prices.reshape(-1, 1)
    scaler.fit(prices_reshaped) 
    
    scaled_prices = scaler.transform(prices_reshaped)
    
    # --- ONLINE LEARNING (Real-time Training) ---
    # Before predicting, let's teach the bot the LATEST pattern it just saw.
    # We take the sequence ending Yesterday to predict Today (the last known data point).
    try:
        # X: Days -61 to -1 (The sequence leading up to the latest price)
        # Y: Day -1 (The latest price itself)
        train_x = scaled_prices[-(LOOKBACK+1):-1]
        train_y = scaled_prices[-1]
        
        # Reshape for AI
        train_x = np.reshape(train_x, (1, LOOKBACK, 1))
        train_y = np.reshape(train_y, (1, 1))
        
        # Train for 1 epoch (Quick refresher)
        model.fit(train_x, train_y, epochs=1, verbose=0)
        print(f"[Brain] Updated with latest data for {ticker}")
    except Exception as e:
        print(f"[Brain] Skip training: {e}")
    # --------------------------------------------

    # 3. Create the input sequence (Last 60 days)
    last_60_days = scaled_prices[-LOOKBACK:]
    input_seq = np.reshape(last_60_days, (1, LOOKBACK, 1))
    
    # 4. The Brain Thinks
    predicted_scaled_price = model.predict(input_seq, verbose=0)
    predicted_price = scaler.inverse_transform(predicted_scaled_price)[0][0]
    current_price = prices[-1]
    
    print(f"\n--- Analysis: {ticker} ---")
    print(f"Current Price: ${current_price:.2f}")
    print(f"AI Prediction: ${predicted_price:.2f}")
    
    # 5. The Decision Logic
    threshold = current_price * 1.01 
    
    if predicted_price > threshold:
        print(f"Decision: BUY (Expected > 1% Gain)")
        if CASH > current_price:
            shares_to_buy = int(CASH // current_price)
            shares_to_buy = min(shares_to_buy, 10) 
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                CASH -= cost
                PORTFOLIO[ticker] = PORTFOLIO.get(ticker, 0) + shares_to_buy
                print(f"*** BOUGHT {shares_to_buy} shares of {ticker} ***")
    
    elif predicted_price < current_price:
        print(f"Decision: SELL (Price expected to drop)")
        if ticker in PORTFOLIO and PORTFOLIO[ticker] > 0:
            shares_to_sell = PORTFOLIO[ticker]
            revenue = shares_to_sell * current_price
            CASH += revenue
            del PORTFOLIO[ticker]
            print(f"*** SOLD {shares_to_sell} shares of {ticker} ***")
    else:
        print("Decision: HOLD")

# --- MAIN LOOP ---
print("Loading Brain...")
model = load_model(MODEL_FILE)
print("Brain Loaded. Starting Trading Session.")
print(f"Initial Cash: ${CASH}")

cycle_count = 0

try:
    while True:
        target_ticker = random.choice(TICKERS)
        if target_ticker in BLACKLIST: continue
        
        predict_and_trade(model, target_ticker)
        
        # Every 5 cycles, do a full audit to see Success/Failure
        cycle_count += 1
        if cycle_count % 5 == 0 and PORTFOLIO:
            get_portfolio_value()
        else:
             print(f"[Wallet] Cash: ${CASH:.2f} | Holdings: {len(PORTFOLIO)} stocks")

        print("\nScanning next stock in 5 seconds...")
        time.sleep(5)

except KeyboardInterrupt:
    print("\nBot stopped by user.")