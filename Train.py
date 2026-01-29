import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
DAYS_TO_FETCH = 1095 # 3 years of data
TEST_SIZE = 0.2      # Keep 20% of data secret to test the bot later
FEATURE_COLUMNS = [
    "Close",
    "Return",
    "LogReturn",
    "Volatility",
    "EMA_12",
    "EMA_26",
    "RSI_14",
    "Volume_Z",
]

def _compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def build_features(df):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"]).diff()
    df["Volatility"] = df["Return"].rolling(14).std()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["RSI_14"] = _compute_rsi(df["Close"], window=14)
    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    df["Volume_Z"] = (df["Volume"] - vol_mean) / vol_std
    df = df.dropna()
    return df

def build_sequences(scaled_features, lookback, target_idx, split_index):
    x_train, y_train, x_val, y_val = [], [], [], []
    for i in range(lookback, len(scaled_features)):
        seq = scaled_features[i - lookback:i, :]
        target = scaled_features[i, target_idx]
        if i < split_index:
            x_train.append(seq)
            y_train.append(target)
        else:
            x_val.append(seq)
            y_val.append(target)
    return (
        np.array(x_train),
        np.array(y_train),
        np.array(x_val),
        np.array(y_val),
    )

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
        # Fetch daily data (auto_adjust handles splits/dividends)
        df = yf.download(
            ticker,
            period=f"{DAYS_TO_FETCH}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = {"Close", "Volume"}
        if required_cols.issubset(df.columns):
            df = df.dropna()
            df = build_features(df)
            if len(df) > LOOKBACK:
                all_data.append(df)
            else:
                print(f"Not enough data for {ticker}")
        else:
            print(f"Missing columns for {ticker}")

    except Exception as e:
        print(f"Error getting {ticker}: {e}")

print(f"\nSuccessfully loaded data for {len(all_data)} stocks.")

# --- STEP 2: NORMALIZATION + WALK-FORWARD SPLIT ---
# We normalize per stock to reduce scale effects, then split by time
x_train, y_train, x_val, y_val = [], [], [], []

print("Creating Flashcards (Sequences)...")

for df in all_data:
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = df[FEATURE_COLUMNS].values
    scaled = scaler.fit_transform(features)

    split_index = int(len(scaled) * (1 - TEST_SIZE))
    t_x, t_y, v_x, v_y = build_sequences(
        scaled,
        LOOKBACK,
        target_idx=FEATURE_COLUMNS.index("Close"),
        split_index=split_index,
    )
    if len(t_x) > 0:
        x_train.append(t_x)
        y_train.append(t_y)
    if len(v_x) > 0:
        x_val.append(v_x)
        y_val.append(v_y)

if not x_train:
    raise ValueError("No training data was generated. Check data availability.")

x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

if x_val:
    x_val = np.concatenate(x_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
else:
    x_val, y_val = None, None

print(f"Generated {len(x_train)} training flashcards.")
if x_val is not None:
    print(f"Generated {len(x_val)} validation flashcards.")

# --- STEP 3: BUILDING THE BRAIN ---
print("\nBuilding the Neural Network...")

model = Sequential()

# Layer 1: LSTM (The Memory Layer)
# return_sequences=True because we are stacking another LSTM layer
model.add(LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2)) # Randomly forget 20% of things to prevent over-studying (overfitting)

# Layer 2: LSTM
model.add(LSTM(units=64, return_sequences=False))
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
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
]

fit_kwargs = {
    "batch_size": 32,
    "epochs": 40,
    "callbacks": callbacks,
    "verbose": 1,
    "shuffle": False,
}

if x_val is not None:
    fit_kwargs["validation_data"] = (x_val, y_val)

model.fit(x_train, y_train, **fit_kwargs)

if x_val is not None:
    loss = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation MSE: {loss:.6f}")

# --- STEP 5: GRADUATION ---
print("\nTraining Complete.")
model.save("smart_trader.keras")
print("Brain saved as 'smart_trader.keras'. Ready for Phase 3.")