# AI Coding Agent Instructions for Brotato

## Project Overview
Brotato is an AI-powered stock trading bot that uses an LSTM neural network to predict stock movements. The project is divided into two main phases:
1. **Training Phase**: The bot learns from historical stock data to build a predictive model.
2. **Live Trading Phase**: The bot uses the trained model to simulate real-time trading with a fake wallet.

### Key Components
- **`Train.py`**: Handles data collection, preprocessing, and training the LSTM model. Outputs the trained model as `smart_trader.keras`.
- **`live_bot.py`**: Uses the trained model for live trading simulation, fetching real-time data and making buy/sell decisions.
- **`requirements.txt`**: Lists dependencies, including TensorFlow and yfinance.
- **`smart_trader.keras`**: The trained model file generated during the training phase.

## Developer Workflows

### Training the Model
1. Ensure Python 3.10+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python Train.py
   ```
   - Downloads historical data for ~30 stocks.
   - Preprocesses data into sequences of 60 days.
   - Trains the LSTM model for 25 epochs.
   - Outputs the trained model as `smart_trader.keras`.

### Live Trading Simulation
1. Ensure the trained model (`smart_trader.keras`) exists.
2. Run the live trading bot:
   ```bash
   python live_bot.py
   ```
   - Fetches real-time stock data.
   - Performs online learning (1 epoch) to adapt to recent trends.
   - Simulates trades based on predictions.
   - Calculates and displays total net worth every 5 cycles.

## Project-Specific Conventions
- **Stock Blacklist**: Avoids "Mag 7" stocks (e.g., NVDA, AAPL) due to atypical movement patterns.
- **Online Learning**: The bot retrains on recent data during live trading to stay updated with market trends.
- **Fake Wallet**: Simulates trading with $10,000 to evaluate performance without real financial risk.

## External Dependencies
- **TensorFlow**: For building and training the LSTM model.
- **yfinance**: For fetching historical and real-time stock data.

## Notes for AI Agents
- Focus on maintaining the separation of concerns between training (`Train.py`) and live trading (`live_bot.py`).
- When modifying `Train.py`, ensure compatibility with the `smart_trader.keras` model format expected by `live_bot.py`.
- Any changes to the stock blacklist or trading logic should be well-documented in the README.
- Test changes thoroughly in both training and live trading phases to ensure stability.

## Example Commands
- Training: `python Train.py`
- Live Trading: `python live_bot.py`