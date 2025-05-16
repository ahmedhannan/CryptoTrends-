import pandas as pd
import os
from config import EXCEL_DATA_DIRECTORY, EXCEL_COLUMN_MAP

def load_ohlcv_from_excel(cryptocurrency_name, start_date_str=None, end_date_str=None):
    """
    Loads historical OHLCV data for a specific cryptocurrency from an Excel file.
    The Excel file is expected to be named <cryptocurrency_name>.xlsx and located in EXCEL_DATA_DIRECTORY.
    Data is taken from the first sheet of the Excel file.

    Args:
        cryptocurrency_name (str): Name of the crypto, corresponding to the Excel filename (without .xlsx).
        start_date_str (str, optional): 'YYYY-MM-DD' format for filtering start date.
        end_date_str (str, optional): 'YYYY-MM-DD' format for filtering end date.

    Returns:
        pd.DataFrame: OHLCV data indexed by a 'timestamp' column, or empty DataFrame on error.
                      Columns will be standardized to 'timestamp', 'open_price', 'high_price',
                      'low_price', 'close_price', 'volume' based on EXCEL_COLUMN_MAP.
    """
    filename = f"{cryptocurrency_name}.xlsx"
    file_path = os.path.join(EXCEL_DATA_DIRECTORY, filename)

    if not os.path.exists(file_path):
        print(f"ERROR: Excel file not found for {cryptocurrency_name} at {file_path}")
        return pd.DataFrame()

    try:
        # Read the first sheet by default
        df = pd.read_excel(file_path, sheet_name=0)
        if df.empty:
            print(f"WARNING: Excel file {file_path} is empty for {cryptocurrency_name}")
            return pd.DataFrame()

        # Rename columns based on the mapping in config.py
        df.rename(columns=EXCEL_COLUMN_MAP, inplace=True)

        # Verify that all essential internal column names are present
        internal_required_cols = ['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
        missing_internal_cols = [col for col in internal_required_cols if col not in df.columns]
        if missing_internal_cols:
            print(f"ERROR: After mapping, missing required columns in {file_path} for {cryptocurrency_name}: {missing_internal_cols}")
            print(f"       Available columns after mapping: {df.columns.tolist()}")
            print(f"       Please check your EXCEL_COLUMN_MAP in config.py and Excel file headers.")
            return pd.DataFrame()

        # Convert 'timestamp' to datetime
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            print(f"ERROR: Could not convert 'timestamp' column to datetime for {cryptocurrency_name} in {file_path}. Error: {e}")
            print(f"       Ensure the timestamp column (mapped from '{next(k for k, v in EXCEL_COLUMN_MAP.items() if v == 'timestamp')}') contains valid dates/times.")
            return pd.DataFrame()

        # Set 'timestamp' as index
        df.set_index('timestamp', inplace=True)

        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # Debug: Print data range before filtering
        if not df.empty:
            print(f"INFO: Loaded {len(df)} rows for {cryptocurrency_name}. Data range: {df.index.min()} to {df.index.max()}")

        # Apply hourly frequency with forward-fill, but only if data exists
        if not df.empty:
            df = df.asfreq('h', method='ffill')
        else:
            print(f"WARNING: No data available for {cryptocurrency_name} after initial processing")
            return pd.DataFrame()

        # Convert OHLCV columns to numeric types
        numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove rows with NaN in essential columns
        df.dropna(subset=numeric_cols, inplace=True)

        # Filter by date range if specified
        if start_date_str or end_date_str:
            start_date = pd.to_datetime(start_date_str) if start_date_str else df.index.min()
            end_date = pd.to_datetime(end_date_str) if end_date_str else df.index.max()
            df = df.loc[start_date:end_date]
            if df.empty:
                print(f"WARNING: No data for {cryptocurrency_name} in date range {start_date} to {end_date}")

        # Final validation
        if df.empty:
            print(f"WARNING: No data found for {cryptocurrency_name} after processing and date filtering from {file_path}")
        else:
            print(f"INFO: Returning {len(df)} rows for {cryptocurrency_name} after filtering")

        return df

    except FileNotFoundError:
        print(f"ERROR: Excel file not found at {file_path}")
        return pd.DataFrame()
    except ValueError as ve:
        print(f"ERROR reading Excel file {file_path} for {cryptocurrency_name}: {ve}")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading data for {cryptocurrency_name} from {file_path}: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    from config import TARGET_CRYPTOS

    print(f"Attempting to load data from Excel directory: {os.path.abspath(EXCEL_DATA_DIRECTORY)}")

    if not TARGET_CRYPTOS:
        print("No TARGET_CRYPTOS defined in config.py. Please add at least one.")
    else:
        for test_crypto in TARGET_CRYPTOS:
            print(f"\n--- Testing data loading for: {test_crypto} ---")
            crypto_df = load_ohlcv_from_excel(test_crypto, start_date_str='2024-11-14', end_date_str='2025-05-14')
            if not crypto_df.empty:
                print(f"\nSuccessfully loaded {len(crypto_df)} records for {test_crypto}:")
                print("Head of the data:")
                print(crypto_df.head())
                print("\nTail of the data:")
                print(crypto_df.tail())
                print("\nData information:")
                crypto_df.info()
            else:
                print(f"\nFailed to load data for {test_crypto}. Check Excel file, EXCEL_COLUMN_MAP, and date formats.")