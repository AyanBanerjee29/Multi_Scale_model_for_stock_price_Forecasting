import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os

def download_data(tickers, start_date, end_date):
    """Downloads historical data for a list of tickers from yfinance."""
    print(f"Downloading data for: {', '.join(tickers)}...")
    # Use "auto_adjust=True" to get adjusted close prices, which is standard
    data = yf.download(tickers, start=start_date, end=end_date, interval="1d", auto_adjust=True)
    data.index = pd.to_datetime(data.index.date)
    return data

def calculate_technical_indicators(nifty_data):
    """Calculates all technical indicators on the Nifty 50 data."""
    print("Calculating technical indicators...")
    
    # Ensure columns are lowercase for pandas-ta
    nifty_data.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    }, inplace=True, errors='ignore')

    # Momentum
    nifty_data['RSI_14'] = nifty_data.ta.rsi(length=14)
    
    macd = nifty_data.ta.macd(fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        nifty_data['MACD_12_26_9'] = macd[f'MACD_{12}_{26}_{9}']
        nifty_data['MACD_HIST_12_26_9'] = macd[f'MACDh_{12}_{26}_{9}']
        nifty_data['MACD_SIGNAL_12_26_9'] = macd[f'MACDs_{12}_{26}_{9}']
    else:
        print("Warning: MACD calculation failed.")

    stoch = nifty_data.ta.stoch(k=14, d=3, smooth_k=3)
    if stoch is not None and not stoch.empty:
        nifty_data['STOCH_K_14_3_3'] = stoch[f'STOCHk_{14}_{3}_{3}']
        nifty_data['STOCH_D_14_3_3'] = stoch[f'STOCHd_{14}_{3}_{3}']
    else:
        print("Warning: Stochastic Oscillator calculation failed.")

    # Trend
    nifty_data['SMA_20'] = nifty_data.ta.sma(length=20)
    nifty_data['SMA_50'] = nifty_data.ta.sma(length=50)
    nifty_data['SMA_200'] = nifty_data.ta.sma(length=200)
    nifty_data['EMA_12'] = nifty_data.ta.ema(length=12)
    nifty_data['EMA_26'] = nifty_data.ta.ema(length=26)
    
    adx = nifty_data.ta.adx(length=14)
    if adx is not None and not adx.empty:
        nifty_data['ADX_14'] = adx[f'ADX_{14}']
    else:
        print("Warning: ADX calculation failed.")

    # Volatility
    bbands = nifty_data.ta.bbands(length=20, std=2)
    if bbands is not None and not bbands.empty:
        # Use f-strings with floats to match pandas-ta column names
        lower_col = f'BBL_{20}_{2.0}'
        mid_col = f'BBM_{20}_{2.0}'
        upper_col = f'BBU_{20}_{2.0}'
        width_col = f'BBB_{20}_{2.0}'
        
        if lower_col in bbands.columns:
            nifty_data['BBL_20_2'] = bbands[lower_col]
            nifty_data['BBM_20_2'] = bbands[mid_col]
            nifty_data['BBU_20_2'] = bbands[upper_col]
            nifty_data['BBB_20_2'] = bbands[width_col]
        else:
            print(f"Warning: Could not find expected Bollinger Bands columns (e.g., {lower_col}). Found: {bbands.columns}. Skipping BBands.")
    else:
        print("Warning: bbands calculation returned None or Empty. Skipping BBands.")
        
    nifty_data['ATR_14'] = nifty_data.ta.atr(length=14)

    # Volume
    nifty_data['OBV'] = nifty_data.ta.obv()

    # Other Transforms
    nifty_data['RET_DAILY'] = nifty_data['close'].pct_change()
    nifty_data['LOG_RET'] = nifty_data.ta.log_return(length=1)
    nifty_data['HIGH_LOW_SPREAD'] = nifty_data['high'] - nifty_data['low']
    nifty_data['OPEN_CLOSE_SPREAD'] = nifty_data['open'] - nifty_data['close']

    return nifty_data

def main():
    # --- 1. Configuration ---
    START_DATE = "2010-01-01"
    END_DATE = "2023-12-31"
    
    TICKERS = [
        '^NSEI', '^INDIAVIX', '^GSPC', '^IXIC', '^DJI', 
        '^N225', '^HSI', '^FTSE', '^GDAXI', 
        'INR=X', 'BZ=F', 'GC=F'
    ]
    
    # Define paths for macro data (you must create these files)
    # Place them in the 'Dataset' folder for simplicity
    REPO_RATE_FILE = 'Dataset/rbi_repo_rate.csv' # Example: Date,Repo_Rate
    CPI_FILE = 'Dataset/india_cpi.csv'           # Example: Date,CPI
    
    OUTPUT_FILE = "Dataset/NIFTY50_features_wide.csv"
    
    # --- 2. Download yfinance Data ---
    all_data = download_data(TICKERS, START_DATE, END_DATE)
    
    # --- 3. Process Nifty 50 Data ---
    # Select OHLCV for Nifty
    nifty_ohlcv = all_data.loc[:, (['Open', 'High', 'Low', 'Close', 'Volume'], '^NSEI')]
    nifty_ohlcv.columns = nifty_ohlcv.columns.droplevel(1) # Drop the '^NSEI' level
    
    # --- THIS IS THE FIX ---
    # Drop all rows where Nifty 50 has NaN values (i.e., non-trading days)
    # This must be done BEFORE calculating technical indicators.
    nifty_ohlcv.dropna(inplace=True)
    # ----------------------
    
    # Calculate TA features
    nifty_with_ta = calculate_technical_indicators(nifty_ohlcv.copy())

    # --- 4. Prepare Other Features ---
    print("Preparing other market features...")
    # Get 'Close' prices for all other tickers
    other_features = all_data['Close'].drop(columns='^NSEI', errors='ignore')
    
    # Rename columns for clarity
    col_rename_map = {
        '^INDIAVIX': 'VIX_Close',
        '^GSPC': 'SP500_Close',
        '^IXIC': 'NASDAQ_Close',
        '^DJI': 'DOW_Close',
        '^N225': 'NIKKEI_Close',
        '^HSI': 'HANGSENG_Close',
        '^FTSE': 'FTSE_Close',
        '^GDAXI': 'DAX_Close',
        'INR=X': 'USDINR_Close',
        'BZ=F': 'BRENT_Close',
        'GC=F': 'GOLD_Close'
    }
    other_features.rename(columns=col_rename_map, inplace=True, errors='ignore')
    
    # --- 5. Combine Nifty TA + Other Market Features ---
    # This join aligns all other market data to Nifty's trading days
    combined_df = nifty_with_ta.join(other_features)
    
    # --- 6. Load and Merge Macro Data ---
    # IMPORTANT: You must create these CSV files manually.
    # The CSVs must have a 'Date' column and a value column.
    
    try:
        print("Loading and merging macro data...")
        
        # Load Repo Rate Data
        repo_df = pd.read_csv(REPO_RATE_FILE, parse_dates=['Date'])
        repo_df.set_index('Date', inplace=True)
        repo_df.index = repo_df.index.tz_localize(None) # Ensure timezone-naive
        repo_df = repo_df.reindex(combined_df.index, method='ffill')
        combined_df = combined_df.join(repo_df)

        # Load CPI Data
        cpi_df = pd.read_csv(CPI_FILE, parse_dates=['Date'])
        cpi_df.set_index('Date', inplace=True)
        cpi_df.index = cpi_df.index.tz_localize(None) # Ensure timezone-naive
        cpi_df = cpi_df.reindex(combined_df.index, method='ffill')
        combined_df = combined_df.join(cpi_df)
        
        print("Macro data merged.")
        
    except FileNotFoundError as e:
        print(f"Warning: Macro data file not found ({e}). Skipping macro features.")
        print("To include macro data, create 'rbi_repo_rate.csv' and 'india_cpi.csv' in 'Dataset/'.")

    # --- 7. Final Cleaning ---
    print("Cleaning final dataset (forward-filling and dropping NaNs)...")
    
    # Forward-fill gaps from holidays (e.g., US market closed when India was open)
    combined_df.fillna(method='ffill', inplace=True)
    
    # Drop any remaining NaNs (e.g., from the start of TA calculations like SMA_200)
    initial_rows = len(combined_df)
    combined_df.dropna(inplace=True)
    final_rows = len(combined_df)
    
    print(f"Dropped {initial_rows - final_rows} rows with NaNs (due to initial indicator warm-up).")
    
    # Ensure 'close' (the target) is the first column for data_utils.py
    target_col = 'close'
    if target_col in combined_df.columns:
        cols = [target_col] + [col for col in combined_df.columns if col != target_col]
        combined_df = combined_df[cols]
    else:
        print(f"Warning: '{target_col}' not found. The 'inverse_transform' function may not work as expected.")

    # --- 8. Save to CSV ---
    print(f"Saving final dataset to {OUTPUT_FILE}...")
    combined_df.to_csv(OUTPUT_FILE)
    print(f"Successfully created {OUTPUT_FILE} with {len(combined_df.columns)} features.")

if __name__ == "__main__":
    main()
