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
BLACKLIST = [
    "NVDA", "AAPL", "MSFT", "GOOG", "GOOGL",
    "AMZN", "META", "TSLA", "BRK-B", "LLY", "AVGO"
] 

# Fake Wallet
STARTING_CASH = 10000.0
CASH = STARTING_CASH
PORTFOLIO = {} # Stores position info: {"F": {"shares": 10, "entry": 12.3, ...}}
LOOKBACK = 60 

# Risk Controls
MAX_POSITIONS = 5
MAX_POSITION_PCT = 0.2      # Max 20% of equity in one ticker
MAX_TOTAL_EXPOSURE = 0.8    # Max 80% of equity invested
STOP_LOSS_PCT = 0.03        # 3% stop loss
TAKE_PROFIT_PCT = 0.05      # 5% take profit
MAX_DRAWDOWN_PCT = 0.1      # 10% drawdown limit

# Execution Simulation
SLIPPAGE_BPS = 5            # 5 basis points
FEE_PER_TRADE = 1.00        # $1 flat fee

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

equity_high = STARTING_CASH
TRADES = []

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

def get_live_data(ticker):
    """Fetches enough recent data to normalize and create a sequence."""
    try:
        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = {"Close", "Volume"}
        if required_cols.issubset(df.columns):
            return df.dropna()
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
    return None

def get_latest_price(ticker):
    df = get_live_data(ticker)
    if df is None or df.empty:
        return None
    return df["Close"].iloc[-1]

def get_total_exposure():
    exposure = 0.0
    for ticker, pos in PORTFOLIO.items():
        price = get_latest_price(ticker)
        if price is not None:
            exposure += pos["shares"] * price
    return exposure

def get_portfolio_value():
    """Calculates total net worth (Cash + Stock Value)."""
    total_value = CASH
    print("\n--- Auditing Portfolio ---")
    for ticker, pos in PORTFOLIO.items():
        try:
            current_price = get_latest_price(ticker)
            if current_price is not None:
                val = pos["shares"] * current_price
                total_value += val
                print(f"{ticker}: {pos['shares']} shares @ ${current_price:.2f} = ${val:.2f}")
        except:
            print(f"Could not value {ticker}")
    
    profit = total_value - STARTING_CASH
    print(f"TOTAL NET WORTH: ${total_value:.2f} (Profit: ${profit:.2f})")
    print("--------------------------")
    return total_value

def apply_execution_price(price, side):
    slip = SLIPPAGE_BPS / 10000
    if side == "buy":
        return price * (1 + slip)
    return price * (1 - slip)

def check_risk_controls():
    global CASH
    to_close = []
    for ticker, pos in PORTFOLIO.items():
        current_price = get_latest_price(ticker)
        if current_price is None:
            continue
        stop_price = pos["entry"] * (1 - STOP_LOSS_PCT)
        take_price = pos["entry"] * (1 + TAKE_PROFIT_PCT)
        if current_price <= stop_price or current_price >= take_price:
            to_close.append((ticker, current_price))

    for ticker, current_price in to_close:
        pos = PORTFOLIO[ticker]
        exec_price = apply_execution_price(current_price, "sell")
        revenue = pos["shares"] * exec_price - FEE_PER_TRADE
        CASH += max(revenue, 0)
        TRADES.append({
            "ticker": ticker,
            "side": "SELL",
            "shares": pos["shares"],
            "price": exec_price,
            "reason": "risk-control",
        })
        del PORTFOLIO[ticker]

def predict_and_trade(model, ticker):
    global CASH, PORTFOLIO
    
    # 1. Get Data
    df = get_live_data(ticker)
    if df is None or len(df) < LOOKBACK + 30:
        print(f"Not enough data for {ticker}. Skipping.")
        return

    df = build_features(df)
    if len(df) < LOOKBACK + 1:
        print(f"Not enough feature data for {ticker}. Skipping.")
        return

    # 2. Live Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = df[FEATURE_COLUMNS].values
    scaler.fit(features)
    scaled_features = scaler.transform(features)
    
    # --- ONLINE LEARNING (Real-time Training) ---
    # Before predicting, let's teach the bot the LATEST pattern it just saw.
    # We take the sequence ending Yesterday to predict Today (the last known data point).
    try:
        # X: Days -61 to -1 (The sequence leading up to the latest price)
        # Y: Day -1 (The latest price itself)
        train_x = scaled_features[-(LOOKBACK+1):-1]
        train_y = scaled_features[-1][FEATURE_COLUMNS.index("Close")]
        
        # Reshape for AI
        train_x = np.reshape(train_x, (1, LOOKBACK, len(FEATURE_COLUMNS)))
        train_y = np.reshape(train_y, (1, 1))
        
        # Train for 1 epoch (Quick refresher)
        model.fit(train_x, train_y, epochs=1, verbose=0)
        print(f"[Brain] Updated with latest data for {ticker}")
    except Exception as e:
        print(f"[Brain] Skip training: {e}")
    # --------------------------------------------

    # 3. Create the input sequence (Last 60 days)
    last_60_days = scaled_features[-LOOKBACK:]
    input_seq = np.reshape(last_60_days, (1, LOOKBACK, len(FEATURE_COLUMNS)))
    
    # 4. The Brain Thinks
    predicted_scaled_price = model.predict(input_seq, verbose=0)
    # Only invert the Close column
    close_min = scaler.data_min_[FEATURE_COLUMNS.index("Close")]
    close_max = scaler.data_max_[FEATURE_COLUMNS.index("Close")]
    predicted_price = predicted_scaled_price[0][0] * (close_max - close_min) + close_min
    current_price = df["Close"].iloc[-1]
    
    print(f"\n--- Analysis: {ticker} ---")
    print(f"Current Price: ${current_price:.2f}")
    print(f"AI Prediction: ${predicted_price:.2f}")
    
    # 5. The Decision Logic
    threshold = current_price * 1.01
    
    if predicted_price > threshold:
        print(f"Decision: BUY (Expected > 1% Gain)")
        equity = get_portfolio_value()
        exposure = get_total_exposure()
        if len(PORTFOLIO) >= MAX_POSITIONS:
            print("Max positions reached.")
            return
        if exposure / max(equity, 1) >= MAX_TOTAL_EXPOSURE:
            print("Max total exposure reached.")
            return

        max_position_value = equity * MAX_POSITION_PCT
        exec_price = apply_execution_price(current_price, "buy")
        max_shares = int(max_position_value // exec_price)
        cash_shares = int((CASH - FEE_PER_TRADE) // exec_price)
        shares_to_buy = min(max_shares, cash_shares)

        if shares_to_buy > 0:
            cost = shares_to_buy * exec_price + FEE_PER_TRADE
            CASH -= cost
            PORTFOLIO[ticker] = {
                "shares": shares_to_buy,
                "entry": exec_price,
            }
            TRADES.append({
                "ticker": ticker,
                "side": "BUY",
                "shares": shares_to_buy,
                "price": exec_price,
                "reason": "signal",
            })
            print(f"*** BOUGHT {shares_to_buy} shares of {ticker} @ ${exec_price:.2f} ***")
    
    elif predicted_price < current_price:
        print(f"Decision: SELL (Price expected to drop)")
        if ticker in PORTFOLIO and PORTFOLIO[ticker]["shares"] > 0:
            shares_to_sell = PORTFOLIO[ticker]["shares"]
            exec_price = apply_execution_price(current_price, "sell")
            revenue = shares_to_sell * exec_price - FEE_PER_TRADE
            CASH += max(revenue, 0)
            del PORTFOLIO[ticker]
            TRADES.append({
                "ticker": ticker,
                "side": "SELL",
                "shares": shares_to_sell,
                "price": exec_price,
                "reason": "signal",
            })
            print(f"*** SOLD {shares_to_sell} shares of {ticker} @ ${exec_price:.2f} ***")
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

        check_risk_controls()
        current_equity = get_portfolio_value()
        equity_high = max(equity_high, current_equity)
        if current_equity < equity_high * (1 - MAX_DRAWDOWN_PCT):
            print("Drawdown limit hit. Pausing trades this cycle.")
            time.sleep(5)
            continue
        
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