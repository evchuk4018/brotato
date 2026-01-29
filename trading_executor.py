"""
Trading Executor Module for DQN-based Stock Trading Bot.

This module handles trade execution, reward calculation, and portfolio management
for a reinforcement learning stock trading system.
"""

from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Wallet Configuration
STARTING_CASH: float = 10000.0

# Trading Costs
SLIPPAGE_BPS: float = 5.0  # 5 basis points = 0.05%
TRADE_FEE: float = 1.0  # $1 per trade

# Actions
ACTION_BUY: int = 0
ACTION_HOLD: int = 1
ACTION_SELL: int = 2

# Risk Controls
MAX_POSITION_PCT: float = 0.20  # Max 20% in one position
MAX_EXPOSURE_PCT: float = 0.80  # Max 80% total exposure
STOP_LOSS_PCT: float = 0.03  # 3% stop loss
TAKE_PROFIT_PCT: float = 0.05  # 5% take profit

# Reward Values
REWARD_PROFIT: float = 1.0
REWARD_LOSS: float = -1.0
REWARD_INVALID_TRADE: float = -10.0
REWARD_NEUTRAL: float = 0.0


# =============================================================================
# PORTFOLIO STATE
# =============================================================================

class TradingExecutor:
    """
    Manages trade execution, portfolio tracking, and reward calculation
    for the DQN trading agent.
    """

    def __init__(self, starting_cash: float = STARTING_CASH):
        """
        Initialize the trading executor with a fake wallet.

        Args:
            starting_cash: Initial cash balance (default: $10,000)
        """
        self.cash: float = starting_cash
        self.starting_cash: float = starting_cash
        self.portfolio: Dict[str, Dict[str, float]] = {}
        # Portfolio format: {"ticker": {"shares": int, "entry_price": float}}
        self.trade_history: list = []
        self.current_prices: Dict[str, float] = {}

    def reset(self) -> None:
        """Reset the wallet to initial state."""
        self.cash = self.starting_cash
        self.portfolio = {}
        self.trade_history = []
        self.current_prices = {}
        logger.info(f"Portfolio reset. Cash: ${self.cash:,.2f}")

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current market prices for all tracked tickers.

        Args:
            prices: Dictionary mapping ticker symbols to current prices
        """
        self.current_prices.update(prices)

    def get_portfolio_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate total portfolio value (cash + stock positions).

        Args:
            prices: Optional dictionary of current prices. If None, uses stored prices.

        Returns:
            Total equity value as float
        """
        if prices is not None:
            self.update_prices(prices)

        stock_value = 0.0
        for ticker, position in self.portfolio.items():
            if ticker in self.current_prices:
                stock_value += position["shares"] * self.current_prices[ticker]
            else:
                # Use entry price if current price not available
                stock_value += position["shares"] * position["entry_price"]
                logger.warning(f"No current price for {ticker}, using entry price")

        total_value = self.cash + stock_value
        return total_value

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """
        Apply slippage to trade price.

        Args:
            price: Base price
            is_buy: True if buying (slippage increases price), False if selling

        Returns:
            Adjusted price after slippage
        """
        slippage_multiplier = SLIPPAGE_BPS / 10000.0  # Convert basis points to decimal
        if is_buy:
            return price * (1 + slippage_multiplier)
        else:
            return price * (1 - slippage_multiplier)

    def _check_position_limit(self, ticker: str, trade_value: float) -> bool:
        """
        Check if trade would exceed maximum position size (20%).

        Args:
            ticker: Stock ticker symbol
            trade_value: Value of the proposed trade

        Returns:
            True if trade is within limits, False otherwise
        """
        total_value = self.get_portfolio_value()
        if total_value <= 0:
            return False

        # Calculate new position value after trade
        current_position_value = 0.0
        if ticker in self.portfolio and ticker in self.current_prices:
            current_position_value = (
                self.portfolio[ticker]["shares"] * self.current_prices[ticker]
            )

        new_position_value = current_position_value + trade_value
        position_pct = new_position_value / total_value

        if position_pct > MAX_POSITION_PCT:
            logger.warning(
                f"Position limit exceeded for {ticker}: "
                f"{position_pct:.1%} > {MAX_POSITION_PCT:.1%}"
            )
            return False
        return True

    def _check_exposure_limit(self, additional_exposure: float) -> bool:
        """
        Check if trade would exceed maximum total exposure (80%).

        Args:
            additional_exposure: Additional stock exposure from proposed trade

        Returns:
            True if trade is within limits, False otherwise
        """
        total_value = self.get_portfolio_value()
        if total_value <= 0:
            return False

        # Calculate current stock exposure
        current_exposure = sum(
            pos["shares"] * self.current_prices.get(ticker, pos["entry_price"])
            for ticker, pos in self.portfolio.items()
        )

        new_exposure = current_exposure + additional_exposure
        exposure_pct = new_exposure / total_value

        if exposure_pct > MAX_EXPOSURE_PCT:
            logger.warning(
                f"Exposure limit exceeded: {exposure_pct:.1%} > {MAX_EXPOSURE_PCT:.1%}"
            )
            return False
        return True

    def _check_stop_loss(self, ticker: str, current_price: float) -> bool:
        """
        Check if position has hit stop loss threshold.

        Args:
            ticker: Stock ticker symbol
            current_price: Current market price

        Returns:
            True if stop loss triggered, False otherwise
        """
        if ticker not in self.portfolio:
            return False

        entry_price = self.portfolio[ticker]["entry_price"]
        loss_pct = (entry_price - current_price) / entry_price

        if loss_pct >= STOP_LOSS_PCT:
            logger.warning(
                f"STOP LOSS triggered for {ticker}: "
                f"{loss_pct:.1%} loss (threshold: {STOP_LOSS_PCT:.1%})"
            )
            return True
        return False

    def _check_take_profit(self, ticker: str, current_price: float) -> bool:
        """
        Check if position has hit take profit threshold.

        Args:
            ticker: Stock ticker symbol
            current_price: Current market price

        Returns:
            True if take profit triggered, False otherwise
        """
        if ticker not in self.portfolio:
            return False

        entry_price = self.portfolio[ticker]["entry_price"]
        profit_pct = (current_price - entry_price) / entry_price

        if profit_pct >= TAKE_PROFIT_PCT:
            logger.info(
                f"TAKE PROFIT triggered for {ticker}: "
                f"{profit_pct:.1%} gain (threshold: {TAKE_PROFIT_PCT:.1%})"
            )
            return True
        return False

    def execute_trade(
        self, action: int, ticker: str, current_price: float
    ) -> Tuple[bool, str]:
        """
        Execute a trade action (BUY, HOLD, or SELL).

        Args:
            action: Trade action (0=BUY, 1=HOLD, 2=SELL)
            ticker: Stock ticker symbol
            current_price: Current market price

        Returns:
            Tuple of (success: bool, message: str)
        """
        # Update current price
        self.current_prices[ticker] = current_price

        # Check for automatic stop loss / take profit triggers
        if ticker in self.portfolio:
            if self._check_stop_loss(ticker, current_price):
                # Force sell on stop loss
                action = ACTION_SELL
            elif self._check_take_profit(ticker, current_price):
                # Force sell on take profit
                action = ACTION_SELL

        # HOLD action - always valid
        if action == ACTION_HOLD:
            logger.debug(f"HOLD {ticker} @ ${current_price:.2f}")
            return True, "Hold executed"

        # BUY action
        elif action == ACTION_BUY:
            return self._execute_buy(ticker, current_price)

        # SELL action
        elif action == ACTION_SELL:
            return self._execute_sell(ticker, current_price)

        else:
            logger.error(f"Invalid action: {action}")
            return False, f"Invalid action: {action}"

    def _execute_buy(self, ticker: str, current_price: float) -> Tuple[bool, str]:
        """
        Execute a buy order.

        Args:
            ticker: Stock ticker symbol
            current_price: Current market price

        Returns:
            Tuple of (success: bool, message: str)
        """
        # Apply slippage (price goes up when buying)
        adjusted_price = self._apply_slippage(current_price, is_buy=True)

        # Calculate maximum shares we can buy with available cash
        total_value = self.get_portfolio_value()
        max_position_value = total_value * MAX_POSITION_PCT

        # Get current position value if any
        current_position_value = 0.0
        if ticker in self.portfolio:
            current_position_value = (
                self.portfolio[ticker]["shares"] * current_price
            )

        # Available room for this position
        available_position_value = max_position_value - current_position_value

        # Also check exposure limit
        current_exposure = sum(
            pos["shares"] * self.current_prices.get(t, pos["entry_price"])
            for t, pos in self.portfolio.items()
        )
        max_additional_exposure = (total_value * MAX_EXPOSURE_PCT) - current_exposure
        available_position_value = min(available_position_value, max_additional_exposure)

        # Calculate shares to buy (use 10% of available cash, respecting limits)
        buy_budget = min(self.cash * 0.10, available_position_value)
        buy_budget -= TRADE_FEE  # Account for fee

        if buy_budget <= 0:
            logger.warning(f"Insufficient funds or position limits reached for {ticker}")
            return False, "Insufficient funds or position limits reached"

        shares_to_buy = int(buy_budget / adjusted_price)

        if shares_to_buy <= 0:
            logger.warning(f"Cannot afford any shares of {ticker}")
            return False, "Cannot afford any shares"

        # Check risk controls
        trade_value = shares_to_buy * adjusted_price
        if not self._check_position_limit(ticker, trade_value):
            return False, "Position limit exceeded"
        if not self._check_exposure_limit(trade_value):
            return False, "Exposure limit exceeded"

        # Execute the buy
        total_cost = (shares_to_buy * adjusted_price) + TRADE_FEE
        self.cash -= total_cost

        if ticker in self.portfolio:
            # Average down/up the position
            old_shares = self.portfolio[ticker]["shares"]
            old_entry = self.portfolio[ticker]["entry_price"]
            new_shares = old_shares + shares_to_buy
            # Calculate weighted average entry price
            new_entry = (
                (old_shares * old_entry) + (shares_to_buy * adjusted_price)
            ) / new_shares
            self.portfolio[ticker] = {"shares": new_shares, "entry_price": new_entry}
        else:
            self.portfolio[ticker] = {
                "shares": shares_to_buy,
                "entry_price": adjusted_price,
            }

        # Record trade
        self.trade_history.append({
            "action": "BUY",
            "ticker": ticker,
            "shares": shares_to_buy,
            "price": adjusted_price,
            "fee": TRADE_FEE,
            "total_cost": total_cost,
        })

        logger.info(
            f"BUY {shares_to_buy} shares of {ticker} @ ${adjusted_price:.2f} "
            f"(Total: ${total_cost:.2f}, Fee: ${TRADE_FEE})"
        )
        return True, f"Bought {shares_to_buy} shares"

    def _execute_sell(self, ticker: str, current_price: float) -> Tuple[bool, str]:
        """
        Execute a sell order.

        Args:
            ticker: Stock ticker symbol
            current_price: Current market price

        Returns:
            Tuple of (success: bool, message: str)
        """
        # Check if we have shares to sell
        if ticker not in self.portfolio or self.portfolio[ticker]["shares"] <= 0:
            logger.warning(f"No shares of {ticker} to sell")
            return False, "No shares to sell"

        # Apply slippage (price goes down when selling)
        adjusted_price = self._apply_slippage(current_price, is_buy=False)

        # Sell all shares
        shares_to_sell = self.portfolio[ticker]["shares"]
        gross_proceeds = shares_to_sell * adjusted_price
        net_proceeds = gross_proceeds - TRADE_FEE

        # Calculate profit/loss
        entry_price = self.portfolio[ticker]["entry_price"]
        profit_loss = (adjusted_price - entry_price) * shares_to_sell - TRADE_FEE

        # Update cash and remove position
        self.cash += net_proceeds
        del self.portfolio[ticker]

        # Record trade
        self.trade_history.append({
            "action": "SELL",
            "ticker": ticker,
            "shares": shares_to_sell,
            "price": adjusted_price,
            "fee": TRADE_FEE,
            "net_proceeds": net_proceeds,
            "profit_loss": profit_loss,
        })

        logger.info(
            f"SELL {shares_to_sell} shares of {ticker} @ ${adjusted_price:.2f} "
            f"(Net: ${net_proceeds:.2f}, P/L: ${profit_loss:+.2f})"
        )
        return True, f"Sold {shares_to_sell} shares, P/L: ${profit_loss:+.2f}"

    def calculate_reward(
        self,
        previous_value: float,
        current_value: float,
        action: int,
        trade_valid: bool,
    ) -> float:
        """
        Calculate reward for the DQN agent based on trade outcome.

        Args:
            previous_value: Portfolio value before action
            current_value: Portfolio value after action
            action: The action taken (0=BUY, 1=HOLD, 2=SELL)
            trade_valid: Whether the trade was successfully executed

        Returns:
            Reward value as float
        """
        # Invalid trade penalty (e.g., selling 0 shares, buying with no cash)
        if not trade_valid and action != ACTION_HOLD:
            logger.debug(f"Invalid trade penalty: {REWARD_INVALID_TRADE}")
            return REWARD_INVALID_TRADE

        # Calculate value change
        value_change = current_value - previous_value

        # Determine reward based on value change
        if value_change > 0:
            reward = REWARD_PROFIT
            logger.debug(f"Profit reward: {reward} (value change: ${value_change:+.2f})")
        elif value_change < 0:
            reward = REWARD_LOSS
            logger.debug(f"Loss penalty: {reward} (value change: ${value_change:+.2f})")
        else:
            reward = REWARD_NEUTRAL
            logger.debug(f"Neutral reward: {reward}")

        return reward

    def get_state_summary(self) -> Dict:
        """
        Get a summary of current portfolio state.

        Returns:
            Dictionary containing portfolio state information
        """
        total_value = self.get_portfolio_value()
        stock_value = total_value - self.cash
        exposure_pct = (stock_value / total_value * 100) if total_value > 0 else 0

        return {
            "cash": self.cash,
            "stock_value": stock_value,
            "total_value": total_value,
            "exposure_pct": exposure_pct,
            "positions": len(self.portfolio),
            "portfolio": dict(self.portfolio),
            "total_trades": len(self.trade_history),
            "pnl": total_value - self.starting_cash,
            "pnl_pct": ((total_value / self.starting_cash) - 1) * 100,
        }

    def print_summary(self) -> None:
        """Print a formatted summary of the portfolio state."""
        summary = self.get_state_summary()
        print("\n" + "=" * 50)
        print("PORTFOLIO SUMMARY")
        print("=" * 50)
        print(f"Cash:           ${summary['cash']:>12,.2f}")
        print(f"Stock Value:    ${summary['stock_value']:>12,.2f}")
        print(f"Total Value:    ${summary['total_value']:>12,.2f}")
        print(f"Exposure:       {summary['exposure_pct']:>12.1f}%")
        print(f"Positions:      {summary['positions']:>12}")
        print(f"Total Trades:   {summary['total_trades']:>12}")
        print("-" * 50)
        print(f"P/L:            ${summary['pnl']:>+12,.2f}")
        print(f"P/L %:          {summary['pnl_pct']:>+12.2f}%")
        print("=" * 50)

        if self.portfolio:
            print("\nOPEN POSITIONS:")
            print("-" * 50)
            for ticker, pos in self.portfolio.items():
                current_price = self.current_prices.get(ticker, pos["entry_price"])
                position_value = pos["shares"] * current_price
                unrealized_pnl = (current_price - pos["entry_price"]) * pos["shares"]
                print(
                    f"  {ticker:6} | {pos['shares']:>6} shares | "
                    f"Entry: ${pos['entry_price']:>8.2f} | "
                    f"Current: ${current_price:>8.2f} | "
                    f"P/L: ${unrealized_pnl:>+8.2f}"
                )
        print()


# =============================================================================
# CONVENIENCE FUNCTIONS (Module-level access)
# =============================================================================

# Global executor instance for simple usage
_executor = TradingExecutor()


def reset_portfolio() -> None:
    """Reset the global portfolio to initial state."""
    _executor.reset()


def execute_trade(action: int, ticker: str, current_price: float) -> bool:
    """
    Execute a trade using the global executor.

    Args:
        action: Trade action (0=BUY, 1=HOLD, 2=SELL)
        ticker: Stock ticker symbol
        current_price: Current market price

    Returns:
        Success/failure boolean
    """
    success, _ = _executor.execute_trade(action, ticker, current_price)
    return success


def calculate_reward(
    previous_value: float,
    current_value: float,
    action: int,
    trade_valid: bool,
) -> float:
    """
    Calculate reward using the global executor.

    Args:
        previous_value: Portfolio value before action
        current_value: Portfolio value after action
        action: The action taken (0=BUY, 1=HOLD, 2=SELL)
        trade_valid: Whether the trade was successfully executed

    Returns:
        Reward value as float
    """
    return _executor.calculate_reward(previous_value, current_value, action, trade_valid)


def get_portfolio_value(prices: Optional[Dict[str, float]] = None) -> float:
    """
    Get total portfolio value using the global executor.

    Args:
        prices: Optional dictionary of current prices

    Returns:
        Total equity value as float
    """
    return _executor.get_portfolio_value(prices)


def get_cash() -> float:
    """Get current cash balance."""
    return _executor.cash


def get_portfolio() -> Dict[str, Dict[str, float]]:
    """Get current portfolio positions."""
    return dict(_executor.portfolio)


def print_summary() -> None:
    """Print portfolio summary."""
    _executor.print_summary()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example usage demonstration
    print("DQN Trading Executor - Example Usage\n")

    # Create a new executor
    executor = TradingExecutor(starting_cash=10000.0)
    print(f"Initial cash: ${executor.cash:,.2f}")

    # Simulate some trades
    test_ticker = "AAPL"
    test_prices = [150.0, 152.0, 148.0, 155.0, 153.0]

    for i, price in enumerate(test_prices):
        print(f"\n--- Step {i + 1}: {test_ticker} @ ${price:.2f} ---")

        # Get previous value
        prev_value = executor.get_portfolio_value({test_ticker: price})

        # Alternate between buy, hold, sell
        if i == 0:
            action = ACTION_BUY
        elif i == len(test_prices) - 1:
            action = ACTION_SELL
        else:
            action = ACTION_HOLD

        # Execute trade
        success, message = executor.execute_trade(action, test_ticker, price)
        print(f"Action: {['BUY', 'HOLD', 'SELL'][action]} -> {message}")

        # Get new value and calculate reward
        curr_value = executor.get_portfolio_value({test_ticker: price})
        reward = executor.calculate_reward(prev_value, curr_value, action, success)
        print(f"Reward: {reward}")

    # Print final summary
    executor.print_summary()
