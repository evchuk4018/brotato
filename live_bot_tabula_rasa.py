"""
Tabula Rasa Trading Bot - A DQN-based trading agent that learns from scratch.

This bot starts with ZERO knowledge (random weights) and learns to trade
through real-time reinforcement learning on 1-minute market data.

Key Philosophy:
- No pre-trained models loaded
- Training and trading happen simultaneously (Online Learning)
- Every minute of market data is a training example
- Uses Deep Q-Network (DQN) with experience replay
"""

import time
import random
from collections import deque
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Trading Universe (avoiding "Mag 7" due to atypical patterns)
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

# Agent Hyperparameters
LOOKBACK = 60           # 60 minutes of data for each state
N_FEATURES = 5          # OHLCV features
STATE_SHAPE = (LOOKBACK, N_FEATURES)
N_ACTIONS = 3           # BUY=0, HOLD=1, SELL=2

# DQN Parameters
EPSILON_START = 1.0     # Start with 100% exploration
EPSILON_MIN = 0.01      # Minimum exploration (1%)
EPSILON_DECAY = 0.995   # Decay rate per episode
GAMMA = 0.95            # Discount factor for future rewards
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100  # Update target network every N steps

# Wallet & Risk Management
STARTING_CASH = 10000.0
MAX_POSITION_PCT = 0.2      # Max 20% of equity in one ticker
MAX_TOTAL_EXPOSURE = 0.8    # Max 80% of equity invested
MAX_POSITIONS = 5
STOP_LOSS_PCT = 0.03        # 3% stop loss

    def act(self, state: np.ndarray, training: bool = True) -> int:
    """
    Main entry point for the Tabula Rasa trading bot.
    Implements:
    - Stock selection bias (pick stock with most experience/trades)
    - Improved reward logic to penalize inactivity and invalid actions, encourage successful trades
    """
    print("=" * 60)
    print("TABULA RASA TRADING BOT")
    print("Starting with ZERO knowledge - Learning from scratch!")
    print("=" * 60)

    # Initialize components
    agent = DQNAgent(STATE_SHAPE, N_ACTIONS)
    wallet = TradingExecutor(STARTING_CASH)

    # Track metrics
    cycle_count = 0
    total_loss = 0
    loss_count = 0

    # Experience/trade count per ticker
    ticker_experience = {t: 0 for t in TICKERS if t not in BLACKLIST}
    ticker_trades = {t: 0 for t in TICKERS if t not in BLACKLIST}

    print(f"\n[START] Monitoring {len(ticker_experience)} stocks")
    print(f"[START] Trading interval: {TRADING_INTERVAL} seconds")
    print("[START] Press Ctrl+C to stop\n")

    # Track current position for learning
    current_ticker = None
    previous_state = None
    previous_action = None
    previous_value = wallet.get_portfolio_value()

    try:
        while True:
            cycle_count += 1
            print(f"\n{'='*20} CYCLE {cycle_count} {'='*20}")

            # 1. Select ticker with most experience/trades
            # If all zero, fallback to random
            max_exp = max(ticker_experience.values())
            max_trade = max(ticker_trades.values())
            candidates = [t for t in ticker_experience if ticker_experience[t] == max_exp or ticker_trades[t] == max_trade]
            if max_exp == 0 and max_trade == 0:
                ticker = random.choice(list(ticker_experience.keys()))
            else:
                ticker = random.choice(candidates)
            print(f"\n[SCAN] Analyzing: {ticker}")

            # 2. Get current market state
            result = get_market_state(ticker, LOOKBACK)
            if result is None:
                print(f"[SKIP] No data for {ticker}")
                time.sleep(5)
                continue

            current_state, raw_data = result
            current_price = float(raw_data['Close'].iloc[-1])
            print(f"[DATA] Current price: ${current_price:.2f}")

            # 3. Check risk controls
            wallet.check_risk_controls()

            # 4. If we have a previous experience, learn from it
            if previous_state is not None and previous_action is not None:
                current_value = wallet.get_portfolio_value()
                # Improved reward logic
                trade_valid = True
                # Penalize HOLD if value unchanged or decreased
                if previous_action == 1 and current_value <= previous_value:
                    reward = -0.1
                # Penalize SELL if no shares were held
                elif previous_action == 2 and ticker_trades.get(current_ticker, 0) == 0:
                    reward = -10.0
                # Encourage BUY if successful
                elif previous_action == 0 and ticker_trades.get(current_ticker, 0) > 0:
                    reward = 0.1
                # Small negative reward for inactivity
                elif current_value == previous_value:
                    reward = -0.05
                else:
                    reward = calculate_reward(previous_value, current_value, previous_action, trade_valid)

                # Store experience
                agent.remember(previous_state, previous_action, reward, current_state, False)
                ticker_experience[current_ticker] += 1
                print(f"[LEARN] Reward: {reward:+.2f} | Memory: {len(agent.memory)}")

                # Train on batch
                loss = agent.replay(BATCH_SIZE)
                if loss is not None:
                    total_loss += loss
                    loss_count += 1
                    avg_loss = total_loss / loss_count
                    print(f"[TRAIN] Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f}")

                previous_value = current_value

            # 5. Agent decides action
            action = agent.act(current_state, training=True)

            # 6. Execute the trade
            success, message = wallet.execute_trade(action, ticker, current_price)
            print(f"[TRADE] {message}")
            if success and action in [0,2]:
                ticker_trades[ticker] += 1

            # 7. Store state for next learning cycle
            previous_state = current_state
            previous_action = action
            current_ticker = ticker

            # 8. Print status every 5 cycles
            if cycle_count % 5 == 0:
                wallet.print_summary()
                stats = agent.get_stats()
                print(f"[STATS] Epsilon: {stats['epsilon']:.2%} | Steps: {stats['training_steps']}")
                print(f"[EXPERIENCE] Top tickers: {sorted(ticker_experience.items(), key=lambda x: -x[1])[:3]}")
                print(f"[TRADES] Top tickers: {sorted(ticker_trades.items(), key=lambda x: -x[1])[:3]}")
            else:
                value = wallet.get_portfolio_value()
                pnl = value - STARTING_CASH
                print(f"[WALLET] Value: ${value:,.2f} | P&L: ${pnl:+,.2f}")

            # 9. Wait for next cycle
            print(f"\n[WAIT] Next scan in {TRADING_INTERVAL} seconds...")
            time.sleep(TRADING_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n[STOP] Bot stopped by user")
        wallet.print_summary()

        stats = agent.get_stats()
        print("\n[FINAL STATS]")
        print(f"  Total training steps: {stats['training_steps']}")
        print(f"  Final epsilon: {stats['epsilon']:.2%}")
        print(f"  Total reward: {stats['total_reward']:.2f}")
        print(f"  Memory size: {stats['memory_size']}")

        if loss_count > 0:
            print(f"  Average loss: {total_loss / loss_count:.4f}")
        """
        Select an action using epsilon-greedy strategy.
        
        Args:
            state: Current market state (normalized OHLCV sequence)
            training: If True, use epsilon-greedy; if False, greedy only
            
        Returns:
            Action: 0=BUY, 1=HOLD, 2=SELL
        """
        # Exploration: Random action
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
            print(f"[BRAIN] EXPLORING: Random action -> {ActionSpace.to_string(action)}")
            return action
        
        # Exploitation: Best action from model
        state_reshaped = state.reshape(1, *self.state_shape)
        q_values = self.model.predict(state_reshaped, verbose=0)[0]
        action = int(np.argmax(q_values))
        print(f"[BRAIN] EXPLOITING: Q-values={q_values.round(3)} -> {ActionSpace.to_string(action)}")
        return action
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay buffer."""
        self.memory.add(state, action, reward, next_state, done)
        self.total_reward += reward
    
    def replay(self, batch_size: int = BATCH_SIZE) -> Optional[float]:
        """
        Train the network on a batch of experiences (Experience Replay).
        
        Uses Double DQN:
        - Main network selects the best action
        - Target network evaluates the Q-value of that action
        
        Returns:
            Training loss, or None if not enough samples
        """
        if not self.memory.is_ready(batch_size):
            return None
        
        # Sample random batch
        batch = self.memory.sample(batch_size)
        
        # Prepare training data
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Current Q-values
        current_q = self.model.predict(states, verbose=0)
        
        # Double DQN: Use main network to select action, target to evaluate
        next_q_main = self.model.predict(next_states, verbose=0)
        next_q_target = self.target_model.predict(next_states, verbose=0)
        
        # Compute targets
        best_actions = np.argmax(next_q_main, axis=1)
        targets = current_q.copy()
        
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * next_q_target[i, best_actions[i]]
        
        # Train
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        loss = history.history['loss'][0]
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_steps += 1
        
        # Update target network periodically
        if self.training_steps % TARGET_UPDATE_FREQ == 0:
            self.update_target_network()
            print(f"[BRAIN] Target network updated (step {self.training_steps})")
        
        return loss
    
    def update_target_network(self) -> None:
        """Copy weights from main network to target network."""
        self.target_model.set_weights(self.model.get_weights())
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            "training_steps": self.training_steps,
            "epsilon": self.epsilon,
            "total_reward": self.total_reward,
            "memory_size": len(self.memory)
        }


# =============================================================================
# DATA MODULE (THE EYES)
# =============================================================================

def get_market_state(ticker: str, lookback: int = LOOKBACK) -> Optional[Tuple[np.ndarray, pd.DataFrame]]:
    """
    Fetch and normalize recent market data for a ticker.
    
    Args:
        ticker: Stock symbol
        lookback: Number of minutes to fetch
        
    Returns:
        Tuple of (normalized_state, raw_dataframe) or None if error
    """
    try:
        # Fetch 1-minute data (need extra for buffer)
        df = yf.download(
            ticker,
            period="1d",  # Last trading day
            interval="1m",
            auto_adjust=True,
            progress=False
        )
        
        if df.empty:
            # Try 5-day period for more data
            df = yf.download(
                ticker,
                period="5d",
                interval="1m",
                auto_adjust=True,
                progress=False
            )
        
        if df.empty or len(df) < lookback:
            print(f"[DATA] Not enough data for {ticker}: got {len(df)} candles, need {lookback}")
            return None
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Select OHLCV features
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in feature_cols):
            print(f"[DATA] Missing columns for {ticker}")
            return None
        
        df = df[feature_cols].dropna()
        
        if len(df) < lookback:
            print(f"[DATA] Not enough clean data for {ticker}")
            return None
        
        # Take last `lookback` candles
        raw_df = df.tail(lookback).copy()
        
        # Rolling window normalization (Min-Max scaling on current window)
        normalized = normalize_data(raw_df)
        
        return normalized, raw_df
        
    except Exception as e:
        print(f"[DATA] Error fetching {ticker}: {e}")
        return None


def normalize_data(df: pd.DataFrame) -> np.ndarray:
    """
    Normalize OHLCV data using Min-Max scaling on the current window.
    
    This is rolling normalization - uses only the data in the current window,
    not a global scaler, to adapt to current market conditions.
    """
    data = df.values.astype(float)
    
    # Min-Max normalize each column independently
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    
    # Avoid division by zero
    range_vals = data_max - data_min
    range_vals[range_vals == 0] = 1
    
    normalized = (data - data_min) / range_vals
    
    return normalized


def get_latest_price(ticker: str) -> Optional[float]:
    """Quick fetch of the latest price for a ticker."""
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return float(data['Close'].iloc[-1])
    except Exception as e:
        print(f"[DATA] Error getting price for {ticker}: {e}")
        return None


# =============================================================================
# TRADING EXECUTOR (THE WALLET)
# =============================================================================

class TradingExecutor:
    """
    Handles trade execution with a simulated wallet.
    
    Tracks cash, positions, and applies realistic execution costs.
    """
    
    def __init__(self, starting_cash: float = STARTING_CASH):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.portfolio: Dict[str, Dict] = {}  # {"TICKER": {"shares": int, "entry_price": float}}
        self.trades: List[Dict] = []
        
        print(f"[WALLET] Initialized with ${self.cash:,.2f}")
    
    def execute_trade(self, action: int, ticker: str, current_price: float) -> Tuple[bool, str]:
        """
        Execute a trading action.
        
        Args:
            action: 0=BUY, 1=HOLD, 2=SELL
            ticker: Stock symbol
            current_price: Current market price
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if action == ActionSpace.HOLD:
            return True, "HOLD - No action taken"
        
        elif action == ActionSpace.BUY:
            return self._execute_buy(ticker, current_price)
        
        elif action == ActionSpace.SELL:
            return self._execute_sell(ticker, current_price)
        
        return False, "Unknown action"
    
    def _execute_buy(self, ticker: str, price: float) -> Tuple[bool, str]:
        """Execute a buy order with risk controls."""
        # Check position limits
        if len(self.portfolio) >= MAX_POSITIONS:
            return False, "Max positions reached"
        
        if ticker in self.portfolio:
            return False, "Already holding position"
        
        # Calculate position size
        equity = self.get_portfolio_value()
        max_position_value = equity * MAX_POSITION_PCT
        
        # Check exposure limits
        current_exposure = self._get_total_exposure()
        if current_exposure / max(equity, 1) >= MAX_TOTAL_EXPOSURE:
            return False, "Max exposure reached"
        
        # Apply slippage
        exec_price = price * (1 + SLIPPAGE_BPS / 10000)
        
        # Calculate shares to buy
        available_cash = self.cash - FEE_PER_TRADE
        max_shares = int(max_position_value / exec_price)
        affordable_shares = int(available_cash / exec_price)
        shares_to_buy = min(max_shares, affordable_shares)
        
        if shares_to_buy <= 0:
            return False, "Insufficient funds"
        
        # Execute
        cost = shares_to_buy * exec_price + FEE_PER_TRADE
        self.cash -= cost
        self.portfolio[ticker] = {
            "shares": shares_to_buy,
            "entry_price": exec_price
        }
        
        self.trades.append({
            "action": "BUY",
            "ticker": ticker,
            "shares": shares_to_buy,
            "price": exec_price,
            "timestamp": time.time()
        })
        
        return True, f"BOUGHT {shares_to_buy} shares @ ${exec_price:.2f}"
    
    def _execute_sell(self, ticker: str, price: float) -> Tuple[bool, str]:
        """Execute a sell order."""
        if ticker not in self.portfolio:
            return False, "No position to sell"
        
        pos = self.portfolio[ticker]
        if pos["shares"] <= 0:
            return False, "No shares to sell"
        
        # Apply slippage
        exec_price = price * (1 - SLIPPAGE_BPS / 10000)
        
        # Execute
        revenue = pos["shares"] * exec_price - FEE_PER_TRADE
        self.cash += max(revenue, 0)
        
        pnl = (exec_price - pos["entry_price"]) * pos["shares"] - FEE_PER_TRADE
        
        self.trades.append({
            "action": "SELL",
            "ticker": ticker,
            "shares": pos["shares"],
            "price": exec_price,
            "pnl": pnl,
            "timestamp": time.time()
        })
        
        del self.portfolio[ticker]
        
        return True, f"SOLD {pos['shares']} shares @ ${exec_price:.2f} (PnL: ${pnl:+.2f})"
    
    def _get_total_exposure(self) -> float:
        """Calculate total value in stock positions."""
        total = 0
        for ticker, pos in self.portfolio.items():
            price = get_latest_price(ticker)
            if price is not None:
                total += pos["shares"] * price
        return total
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + positions)."""
        return self.cash + self._get_total_exposure()
    
    def check_risk_controls(self) -> None:
        """Check and execute stop-loss / take-profit orders."""
        to_close = []
        
        for ticker, pos in self.portfolio.items():
            current_price = get_latest_price(ticker)
            if current_price is None:
                continue
            
            entry = pos["entry_price"]
            pnl_pct = (current_price - entry) / entry
            
            if pnl_pct <= -STOP_LOSS_PCT:
                to_close.append((ticker, current_price, "STOP-LOSS"))
            elif pnl_pct >= TAKE_PROFIT_PCT:
                to_close.append((ticker, current_price, "TAKE-PROFIT"))
        
        for ticker, price, reason in to_close:
            success, msg = self._execute_sell(ticker, price)
            if success:
                print(f"[RISK] {reason}: {msg}")
    
    def print_summary(self) -> None:
        """Print portfolio summary."""
        value = self.get_portfolio_value()
        pnl = value - self.starting_cash
        pnl_pct = (pnl / self.starting_cash) * 100
        
        print("\n" + "=" * 50)
        print("PORTFOLIO SUMMARY")
        print("=" * 50)
        print(f"Cash:       ${self.cash:,.2f}")
        print(f"Positions:  {len(self.portfolio)}")
        for ticker, pos in self.portfolio.items():
            price = get_latest_price(ticker) or pos["entry_price"]
            value_pos = pos["shares"] * price
            print(f"  - {ticker}: {pos['shares']} shares @ ${price:.2f} = ${value_pos:,.2f}")
        print(f"Total:      ${value:,.2f}")
        print(f"P&L:        ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"Trades:     {len(self.trades)}")
        print("=" * 50 + "\n")


# =============================================================================
# REWARD CALCULATOR
# =============================================================================

def calculate_reward(prev_value: float, curr_value: float, 
                     action: int, trade_valid: bool) -> float:
    """
    Calculate the reward signal for the agent.
    
    Reward Structure:
    - Invalid trade (e.g., selling nothing): -10 (harsh penalty)
    - Portfolio value increased: +1 (encourage profitable actions)
    - Portfolio value decreased: -1 (discourage losses)
    - Neutral (no change): 0
    
    Args:
        prev_value: Portfolio value before action
        curr_value: Portfolio value after action
        action: The action taken (0=BUY, 1=HOLD, 2=SELL)
        trade_valid: Whether the trade was successfully executed
        
    Returns:
        Reward value
    """
    # Heavy penalty for invalid trades
    if not trade_valid and action != ActionSpace.HOLD:
        return -10.0
    
    # Calculate PnL
    pnl = curr_value - prev_value
    
    # Normalize to -1, 0, +1 with some thresholds
    if pnl > 10:  # Significant profit
        return 1.0
    elif pnl < -10:  # Significant loss
        return -1.0
    else:
        # Proportional reward for smaller changes
        return pnl / 100  # Small nudge in the right direction


# =============================================================================
# MAIN TRADING LOOP
# =============================================================================

def run_trading_bot():
    """
    Main entry point for the Tabula Rasa trading bot.
    
    The bot will:
    1. Initialize with random weights (no pre-training)
    2. Observe the market state
    3. Choose an action (explore vs exploit)
    4. Execute the trade
    5. Calculate reward
    6. Learn from the experience
    7. Repeat
    """
    print("=" * 60)
    print("TABULA RASA TRADING BOT")
    print("Starting with ZERO knowledge - Learning from scratch!")
    print("=" * 60)
    
    # Initialize components
    agent = DQNAgent(STATE_SHAPE, N_ACTIONS)
    wallet = TradingExecutor(STARTING_CASH)
    
    # Track metrics
    cycle_count = 0
    total_loss = 0
    loss_count = 0
    
    print(f"\n[START] Monitoring {len(TICKERS)} stocks")
    print(f"[START] Trading interval: {TRADING_INTERVAL} seconds")
    print("[START] Press Ctrl+C to stop\n")
    
    # Track current position for learning
    current_ticker = None
    previous_state = None
    previous_action = None
    previous_value = wallet.get_portfolio_value()
    
    try:
        while True:
            cycle_count += 1
            print(f"\n{'='*20} CYCLE {cycle_count} {'='*20}")
            
            # 1. Select a random ticker to analyze
            ticker = random.choice([t for t in TICKERS if t not in BLACKLIST])
            print(f"\n[SCAN] Analyzing: {ticker}")
            
            # 2. Get current market state
            result = get_market_state(ticker, LOOKBACK)
            if result is None:
                print(f"[SKIP] No data for {ticker}")
                time.sleep(5)
                continue
            
            current_state, raw_data = result
            current_price = float(raw_data['Close'].iloc[-1])
            print(f"[DATA] Current price: ${current_price:.2f}")
            
            # 3. Check risk controls
            wallet.check_risk_controls()
            
            # 4. If we have a previous experience, learn from it
            if previous_state is not None and previous_action is not None:
                # Calculate reward based on portfolio change
                current_value = wallet.get_portfolio_value()
                reward = calculate_reward(
                    previous_value, 
                    current_value, 
                    previous_action, 
                    True  # Simplified - assuming last trade was valid
                )
                
                # Store experience
                agent.remember(previous_state, previous_action, reward, current_state, False)
                print(f"[LEARN] Reward: {reward:+.2f} | Memory: {len(agent.memory)}")
                
                # Train on batch
                loss = agent.replay(BATCH_SIZE)
                if loss is not None:
                    total_loss += loss
                    loss_count += 1
                    avg_loss = total_loss / loss_count
                    print(f"[TRAIN] Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f}")
                
                previous_value = current_value
            
            # 5. Agent decides action
            action = agent.act(current_state, training=True)
            
            # 6. Execute the trade
            success, message = wallet.execute_trade(action, ticker, current_price)
            print(f"[TRADE] {message}")
            
            # 7. Store state for next learning cycle
            previous_state = current_state
            previous_action = action
            current_ticker = ticker
            
            # 8. Print status every 5 cycles
            if cycle_count % 5 == 0:
                wallet.print_summary()
                stats = agent.get_stats()
                print(f"[STATS] Epsilon: {stats['epsilon']:.2%} | Steps: {stats['training_steps']}")
            else:
                value = wallet.get_portfolio_value()
                pnl = value - STARTING_CASH
                print(f"[WALLET] Value: ${value:,.2f} | P&L: ${pnl:+,.2f}")
            
            # 9. Wait for next cycle
            print(f"\n[WAIT] Next scan in {TRADING_INTERVAL} seconds...")
            time.sleep(TRADING_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n[STOP] Bot stopped by user")
        wallet.print_summary()
        
        stats = agent.get_stats()
        print("\n[FINAL STATS]")
        print(f"  Total training steps: {stats['training_steps']}")
        print(f"  Final epsilon: {stats['epsilon']:.2%}")
        print(f"  Total reward: {stats['total_reward']:.2f}")
        print(f"  Memory size: {stats['memory_size']}")
        
        if loss_count > 0:
            print(f"  Average loss: {total_loss / loss_count:.4f}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_trading_bot()
