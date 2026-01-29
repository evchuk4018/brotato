🧠 AI Stock Trader (LSTM Neural Network)

A "Zero-Risk" Python trading bot that uses deep learning to predict stock movements. It trains on historical data and trades with a simulated "Fake Wallet" on the live market.

Strategy: Generalist LSTM (Long Short-Term Memory) Neural Net with feature engineering.
Universe: S&P 500 subset (excluding top 10 giants like NVDA/AAPL).

🛠️ Setup (The Workshop)

1. Prerequisites

You need Python 3.10 or newer.

2. Installation

Install the required heavy machinery (TensorFlow, yfinance, etc.):

pip install -r requirements.txt


🏫 Phase 1: Training (The School)

Before the bot can trade, it must learn what a stock chart looks like. We teach it by showing it 2 years of history for 30 different companies.

Run the Trainer:

python Train.py


What happens:

Downloads adjusted data for ~30 stocks (excluding the blacklist) using splits/dividends adjusted prices.

Builds features (returns, volatility, EMA, RSI, volume z-score) and normalizes per stock.

Creates "Flashcards": sequences of 60 days to predict day 61.

Trains the Neural Net with walk-forward validation and early stopping.

Output:

Generates a brain file named smart_trader.keras.

Note: You only need to run this once (or whenever you want to do a major brain reset).

💼 Phase 2: Live Trading (The Job)

Now we hook the brain up to real-time data. The bot acts as a "Paper Trader" with $10,000 fake cash.

Run the Bot:

python live_bot.py


The Loop:

Scans: Picks a random stock from the watchlist.

Online Learning: Crucial Step. It looks at the most recent data for that specific stock and runs a quick training session (1 epoch) to adapt to the latest trends.

Predicts: Guesses the next price.

Acts: Buys if predicted gain > 1%. Sells if predicted price < current price.
Execution: Simulates slippage and fees for paper trading realism.
Risk Controls: Max positions, max exposure, stop-loss/take-profit, and drawdown limits.

The Scoreboard:

Every 5 cycles, the bot pauses to calculate your Total Net Worth (Cash + Current Value of Shares).

🚫 The Blacklist

We explicitly avoid the "Mag 7" and other giants because their movements are often disconnected from standard technical patterns due to index fund flows.

Blacklisted: NVDA, AAPL, MSFT, GOOG, AMZN, META, TSLA.

📂 File Structure

train.py: The teacher. Downloads history and builds the smart_trader.keras model.

live_bot.py: The trader. Loads the model, connects to Yahoo Finance, and manages the fake wallet.

requirements.txt: List of Python libraries needed.

smart_trader.keras: The saved brain (generated after running train.py).