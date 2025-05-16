import pandas as pd
import numpy as np

def add_technical_indicators(df):
    if df.empty:
        print("WARNING: Input DataFrame to add_technical_indicators is empty.")
        return df

    required_ohlcv_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
    for col in required_ohlcv_cols:
        if col not in df.columns:
            raise ValueError(f"ERROR: Column '{col}' is missing from the input DataFrame for feature engineering.")

    df_out = df.copy()

    close = df_out['close_price']
    high = df_out['high_price']
    low = df_out['low_price']
    volume = df_out['volume']

    # Moving Averages
    df_out['SMA_10'] = close.rolling(window=10).mean()
    df_out['SMA_20'] = close.rolling(window=20).mean()
    df_out['EMA_10'] = close.ewm(span=10, adjust=False).mean()
    df_out['EMA_20'] = close.ewm(span=20, adjust=False).mean()

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df_out['MACD'] = ema_12 - ema_26
    df_out['MACD_signal'] = df_out['MACD'].ewm(span=9, adjust=False).mean()
    df_out['MACD_hist'] = df_out['MACD'] - df_out['MACD_signal']

    # RSI
    delta = close.diff(1)
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss
    df_out['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    df_out['RSI'].fillna(50, inplace=True)

    # Stochastic Oscillator
    low_min_14 = low.rolling(window=14).min()
    high_max_14 = high.rolling(window=14).max()
    df_out['%K'] = 100 * ((close - low_min_14) / (high_max_14 - low_min_14))
    df_out['%K'].fillna(50, inplace=True)
    df_out['%D'] = df_out['%K'].rolling(window=3).mean()

    # Bollinger Bands
    df_out['BB_mid'] = close.rolling(window=20).mean()
    std_dev_20 = close.rolling(window=20).std()
    df_out['BB_upper'] = df_out['BB_mid'] + (std_dev_20 * 2)
    df_out['BB_lower'] = df_out['BB_mid'] - (std_dev_20 * 2)
    df_out['BB_width'] = (df_out['BB_upper'] - df_out['BB_lower']) / df_out['BB_mid']

    # ATR
    high_low = high - low
    high_close_prev = np.abs(high - close.shift(1))
    low_close_prev = np.abs(low - close.shift(1))
    tr = pd.DataFrame({'hl': high_low, 'hc': high_close_prev, 'lc': low_close_prev}).max(axis=1)
    df_out['ATR'] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    # OBV
    df_out['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()

    # Price and Volume Changes
    df_out['close_price_change_pct'] = close.pct_change() * 100
    df_out['volume_change_pct'] = volume.replace(0, np.nan).pct_change().replace(np.nan, 0) * 100

    # Time-based Features
    df_out['hour'] = df_out.index.hour
    df_out['day_of_week'] = df_out.index.dayofweek
    df_out['day_of_month'] = df_out.index.day
    df_out['month_of_year'] = df_out.index.month

    df_out['hour_sin'] = np.sin(2 * np.pi * df_out['hour'] / 24)
    df_out['hour_cos'] = np.cos(2 * np.pi * df_out['hour'] / 24)
    df_out['day_of_week_sin'] = np.sin(2 * np.pi * df_out['day_of_week'] / 7)
    df_out['day_of_week_cos'] = np.cos(2 * np.pi * df_out['day_of_week'] / 7)
    df_out['month_sin'] = np.sin(2 * np.pi * df_out['month_of_year'] / 12)
    df_out['month_cos'] = np.cos(2 * np.pi * df_out['month_of_year'] / 12)

    # Lagged Features
    df_out['lag_close_1'] = close.shift(1)
    df_out['lag_volume_1'] = volume.shift(1)

    # Handle NaNs & Infs
    df_out.replace([np.inf, -np.inf], np.nan, inplace=True)
    initial_nan_rows_to_drop = 20  # Based on SMA_20
    df_out = df_out.iloc[initial_nan_rows_to_drop:]
    df_out.fillna(method='ffill', inplace=True)
    df_out.fillna(method='bfill', inplace=True)

    if df_out.isnull().any().any():
        print(f"WARNING: NaNs still present: {df_out.columns[df_out.isnull().any()].tolist()}")

    return df_out