"""
Data Module for Real-Time Stock Trading Bot

This module provides functions for fetching and preprocessing 1-minute interval
stock data from Yahoo Finance for use with LSTM-based trading models.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional, Tuple
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV columns using Min-Max scaling on the current window.
    
    Uses rolling normalization where min/max values are computed from the
    current data window rather than a static global scaler.
    
    Args:
        df: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)
        
    Returns:
        DataFrame with normalized values in range [0, 1]
        
    Raises:
        ValueError: If required columns are missing or data is empty
    """
    if df.empty:
        raise ValueError("Cannot normalize empty DataFrame")
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create a copy to avoid modifying original
    normalized_df = df[required_columns].copy()
    
    # Normalize each column using its own min/max from the current window
    for column in required_columns:
        col_min = normalized_df[column].min()
        col_max = normalized_df[column].max()
        
        # Handle case where min == max (constant values)
        if col_max - col_min == 0:
            normalized_df[column] = 0.5  # Set to middle of range
        else:
            normalized_df[column] = (normalized_df[column] - col_min) / (col_max - col_min)
    
    return normalized_df


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns that yfinance sometimes returns.
    
    Args:
        df: DataFrame potentially with MultiIndex columns
        
    Returns:
        DataFrame with flattened column names
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Take the first level of the MultiIndex (the actual column names)
        df.columns = df.columns.get_level_values(0)
    return df


def _handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing data points in the DataFrame.
    
    Uses forward fill followed by backward fill to handle gaps,
    then drops any remaining NaN rows.
    
    Args:
        df: DataFrame with potential missing values
        
    Returns:
        DataFrame with missing values handled
    """
    if df.empty:
        return df
    
    # Forward fill then backward fill
    df = df.ffill().bfill()
    
    # Drop any remaining NaN rows (edge case)
    df = df.dropna()
    
    return df


def get_market_state(
    ticker: str,
    lookback: int = 60
) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Fetch the last N minutes of OHLCV data and prepare it for LSTM input.
    
    Fetches 1-minute interval data from Yahoo Finance, normalizes it using
    rolling window scaling, and formats it for LSTM input.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        lookback: Number of minutes of historical data to fetch (default: 60)
        
    Returns:
        Tuple containing:
            - numpy array shaped (1, lookback, 5) for LSTM input, or None on error
            - raw DataFrame with OHLCV data, or None on error
            
    Note:
        The LSTM input format is (samples, timesteps, features) where:
        - samples = 1 (single prediction)
        - timesteps = lookback (number of time steps)
        - features = 5 (Open, High, Low, Close, Volume)
    """
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Fetch 1-minute data
        # Request extra data to account for gaps and ensure we have enough
        # yfinance only allows 1m data for last 7 days
        df = stock.history(period="1d", interval="1m")
        
        if df.empty:
            # Try fetching more data if 1 day wasn't enough
            logger.warning(f"No data for {ticker} with 1d period, trying 5d")
            df = stock.history(period="5d", interval="1m")
        
        if df.empty:
            logger.error(f"No data available for {ticker}. Market may be closed.")
            return None, None
        
        # Flatten MultiIndex columns if present
        df = _flatten_multiindex_columns(df)
        
        # Select only OHLCV columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Verify all required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in data: {missing_cols}")
            return None, None
        
        df = df[required_columns]
        
        # Handle missing data
        df = _handle_missing_data(df)
        
        if len(df) < lookback:
            logger.warning(
                f"Insufficient data for {ticker}: got {len(df)} rows, "
                f"need {lookback}. Using available data."
            )
            if len(df) == 0:
                return None, None
            # Use whatever data we have
            lookback = len(df)
        
        # Get the last 'lookback' rows
        df = df.tail(lookback).copy()
        raw_df = df.copy()
        
        # Normalize the data using rolling window scaling
        normalized_df = normalize_data(df)
        
        # Convert to numpy array and reshape for LSTM
        # Shape: (samples, timesteps, features) = (1, lookback, 5)
        lstm_input = normalized_df.values.reshape(1, lookback, 5)
        
        logger.info(
            f"Successfully fetched {len(df)} minutes of data for {ticker}. "
            f"LSTM input shape: {lstm_input.shape}"
        )
        
        return lstm_input, raw_df
        
    except Exception as e:
        logger.error(f"Error fetching market state for {ticker}: {str(e)}")
        return None, None


def get_latest_price(ticker: str) -> Optional[float]:
    """
    Fetch the most recent price for a stock ticker.
    
    Provides a quick lookup for the current/latest trading price.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        
    Returns:
        The latest closing price as a float, or None on error
        
    Note:
        During market hours, this returns the most recent 1-minute close.
        Outside market hours, this returns the last available price.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Try to get the most recent 1-minute data
        df = stock.history(period="1d", interval="1m")
        
        if df.empty:
            # Fall back to daily data if 1m not available
            logger.warning(f"No 1m data for {ticker}, trying daily data")
            df = stock.history(period="5d", interval="1d")
        
        if df.empty:
            logger.error(f"No price data available for {ticker}")
            return None
        
        # Flatten MultiIndex if present
        df = _flatten_multiindex_columns(df)
        
        # Get the latest close price
        latest_price = float(df['Close'].iloc[-1])
        
        logger.debug(f"Latest price for {ticker}: ${latest_price:.2f}")
        
        return latest_price
        
    except Exception as e:
        logger.error(f"Error fetching latest price for {ticker}: {str(e)}")
        return None


def get_batch_market_state(
    tickers: list[str],
    lookback: int = 60
) -> dict[str, Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]]:
    """
    Fetch market state for multiple tickers.
    
    Convenience function for fetching data for multiple stocks at once.
    
    Args:
        tickers: List of stock ticker symbols
        lookback: Number of minutes of historical data to fetch
        
    Returns:
        Dictionary mapping ticker symbols to their (lstm_input, raw_df) tuples
    """
    results = {}
    for ticker in tickers:
        results[ticker] = get_market_state(ticker, lookback)
    return results


def is_market_open() -> bool:
    """
    Check if the US stock market is currently open.
    
    Uses a simple time-based check for NYSE/NASDAQ hours.
    Does not account for holidays.
    
    Returns:
        True if market is likely open, False otherwise
    """
    try:
        # Get current time in Eastern Time
        from zoneinfo import ZoneInfo
        eastern = ZoneInfo('America/New_York')
        now = datetime.now(eastern)
        
        # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
        is_market_hours = market_open <= now <= market_close
        
        return is_weekday and is_market_hours
        
    except Exception as e:
        logger.warning(f"Could not determine market status: {e}")
        return False


def validate_ticker(ticker: str) -> bool:
    """
    Validate that a ticker symbol exists and has data.
    
    Args:
        ticker: Stock ticker symbol to validate
        
    Returns:
        True if ticker is valid and has data, False otherwise
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if we got valid info back
        return info is not None and len(info) > 0
        
    except Exception:
        return False


# Example usage and testing
if __name__ == "__main__":
    # Test the module with a sample ticker
    test_ticker = "SPY"
    
    print(f"Testing data module with {test_ticker}...")
    print(f"Market open: {is_market_open()}")
    
    # Test get_latest_price
    price = get_latest_price(test_ticker)
    if price:
        print(f"Latest price for {test_ticker}: ${price:.2f}")
    else:
        print(f"Could not fetch price for {test_ticker}")
    
    # Test get_market_state
    lstm_input, raw_df = get_market_state(test_ticker, lookback=60)
    
    if lstm_input is not None:
        print(f"\nLSTM input shape: {lstm_input.shape}")
        print(f"Raw data shape: {raw_df.shape}")
        print(f"\nRaw data (last 5 rows):\n{raw_df.tail()}")
        print(f"\nNormalized values range: [{lstm_input.min():.4f}, {lstm_input.max():.4f}]")
    else:
        print(f"Could not fetch market state for {test_ticker}")
