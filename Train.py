import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# --- CONFIGURATION ---
# 1. The "Blacklist" (Top 10 Stocks to IGNORE as of 2026)
BLACKLIST = [
    "NVDA", "AAPL", "MSFT", "GOOG", "GOOGL", 
    "AMZN", "META", "TSLA", "BRK-B", "LLY", "AVGO"
]

# 2. The "Universe" (The stocks we WILL study)
# We use a diverse mix of 30 well-known companies (Tech, Food, Banks, Cars)
# excluding the top 10.
TICKERS = [
    "AMD", "INTC", "CSCO", "ORCL", "IBM",      # Tech
    "F", "GM", "TM",                           # Cars
    "JPM", "BAC", "WFC", "C", "GS",            # Banks
    "KO", "PEP", "MCD", "SBUX", "WMT", "TGT",  # Consumer
    "DIS", "NFLX",                             # Media
    "PFE", "JNJ", "MRK",                       # Pharma
    "XOM", "CVX",                              # Energy
    "BA", "GE", "CAT"                          # Industrial
]

# Settings
LOOKBACK = 60        # The brain looks at the past 60 days to predict day 61
DAYS_TO_FETCH = 730  # 2 years of data
TEST_SIZE = 0.2      # Keep 20% of data secret to test the bot later

print(f"--- PREPARING TO TRAIN ON {len(TICKERS)} STOCKS ---")
print(f"Ignoring Giants: {BLACKLIST}")

# --- STEP 1: DATA HARVESTING ---
all_data = []

for ticker in TICKERS:
    if ticker in BLACKLIST:
        print(f"Skipping {ticker} (Blacklisted)...")
        continue
    
    print(f"Downloading data for: {ticker}...")
    try:
        # Fetch daily data
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        
        # We only care about the 'Close' price
        if 'Close' in df.columns:
            # yfinance sometimes returns a double-index column, we fix that:
            if isinstance(df.columns, pd.MultiIndex):
                series = df['Close'].iloc[:, 0]
            else:
                series = df['Close']
            
            # Clean data (drop missing values)
            series = series.dropna()
            
            # Add to our massive list
            if len(series) > LOOKBACK:
                all_data.append(series.values)
            else:
                print(f"Not enough data for {ticker}")
                
    except Exception as e:
        print(f"Error getting {ticker}: {e}")

print(f"\nSuccessfully loaded data for {len(all_data)} stocks.")

# --- STEP 2: NORMALIZATION (SQUISHING) ---
# We chain all stocks together to train one "Generalist" brain
# But we must normalize them individually so $10 stock looks like $1000 stock
scaler = MinMaxScaler(feature_range=(0, 1))

x_train, y_train = [], []

print("Creating Flashcards (Sequences)...")

for stock_prices in all_data:
    # 1. Reshape to (rows, 1) for the scaler
    stock_prices = stock_prices.reshape(-1, 1)
    
    # 2. Squish prices between 0 and 1
    scaled_prices = scaler.fit_transform(stock_prices)
    
    # 3. Cut into clips (Sequences)
    # If we have 100 days, and Lookback is 60:
    # Clip 1: Days 0-59 -> Predict Day 60
    # Clip 2: Days 1-60 -> Predict Day 61
    for i in range(LOOKBACK, len(scaled_prices)):
        x_train.append(scaled_prices[i-LOOKBACK:i, 0])
        y_train.append(scaled_prices[i, 0])

# Convert list to numpy arrays (Computer speak)
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape x_train for LSTM [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

print(f"Generated {len(x_train)} flashcards for the AI to study.")

# --- STEP 3: BUILDING THE BRAIN ---
print("\nBuilding the Neural Network...")

model = Sequential()

# Layer 1: LSTM (The Memory Layer)
# return_sequences=True because we are stacking another LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2)) # Randomly forget 20% of things to prevent over-studying (overfitting)

# Layer 2: LSTM
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Layer 3: Dense (The Decision Layer)
model.add(Dense(units=25)) # Simplify thoughts
model.add(Dense(units=1))  # Final Output: The Predicted Price

# Compile the Brain
model.compile(optimizer='adam', loss='mean_squared_error')

# --- STEP 4: THE GYM (TRAINING) ---
print("\n--- STARTING TRAINING ---")
# Epochs = How many times to read the whole dataset
# Batch Size = How many flashcards to look at before updating the brain
model.fit(x_train, y_train, batch_size=32, epochs=25)

# --- STEP 5: GRADUATION ---
print("\nTraining Complete.")
model.save("smart_trader.keras")
print("Brain saved as 'smart_trader.keras'. Ready for Phase 3.")