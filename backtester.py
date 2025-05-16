import pandas as pd
import numpy as np
import os
from portfolio_optimizer import PortfolioOptimizer
from excel_data_loader import load_ohlcv_from_excel
from feature_engineer import add_technical_indicators
from config import (
    TARGET_CRYPTOS,
    INITIAL_CAPITAL,
    TRANSACTION_COST_PCT,
    SEQUENCE_LENGTH,
    PREDICTION_HORIZON,
    EXCEL_DATA_DIRECTORY,
    EXCEL_COLUMN_MAP
)

class Backtester:
    def __init__(self, start_date, end_date, crypto_symbols=None, initial_capital=INITIAL_CAPITAL,
                 transaction_cost_pct=TRANSACTION_COST_PCT, sequence_len=SEQUENCE_LENGTH,
                 prediction_horizon=PREDICTION_HORIZON):
        self.start_date = start_date
        self.end_date = end_date
        self.crypto_symbols = crypto_symbols if crypto_symbols is not None else TARGET_CRYPTOS
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.sequence_len = sequence_len
        self.prediction_horizon = prediction_horizon
        self.ohlcv_data = {}
        self.featured_data = {}
        self.models = {'prediction': {}, 'anomaly': {}}
        self.portfolio_optimizer = None
        self.portfolio_history_df = pd.DataFrame()
        self.trades_log_list = []

    def _prepare_data_and_models(self):
        for crypto in self.crypto_symbols[:]:  # Copy to allow removal
            df = load_ohlcv_from_excel(crypto, start_date_str=self.start_date, end_date_str=self.end_date)
            if not df.empty:
                # Verify required columns
                required_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"ERROR: Missing columns for {crypto}: {missing_cols}")
                    self.crypto_symbols.remove(crypto)
                    continue
                self.ohlcv_data[crypto] = df
                # Apply technical indicators
                try:
                    self.featured_data[crypto] = add_technical_indicators(df)
                except ValueError as e:
                    print(f"Failed to engineer features for {crypto}: {e}")
                    self.crypto_symbols.remove(crypto)
            else:
                print(f"No data loaded for {crypto}. Removing from simulation.")
                self.crypto_symbols.remove(crypto)
        if not self.crypto_symbols:
            print("ERROR: No valid data for any cryptocurrencies.")
            return
        self.portfolio_optimizer = PortfolioOptimizer(
            list_of_cryptos=self.crypto_symbols,
            initial_portfolio_capital=self.initial_capital,
            cost_per_transaction_pct=self.transaction_cost_pct
        )

    def _get_point_in_time_volatility(self, data):
        if len(data) < 2:
            return 0.0
        returns = data['close_price'].pct_change().dropna()
        return returns.std() * np.sqrt(24) if len(returns) > 0 else 0.0

    def predict_future_prices(self, model, scalers, features, input_data):
        try:
            from prediction_model import predict_future_prices as predict
            return predict(model, scalers, features, input_data, main_target_feature_name='close_price')
        except Exception as e:
            print(f"Prediction error: {e}")
            return pd.DataFrame()

    def detect_anomalies(self, model, scaler, threshold, data):
        try:
            from anomaly_detector import detect_anomalies_with_autoencoder
            return detect_anomalies_with_autoencoder(model, scaler, threshold, data)
        except Exception as e:
            print(f"Anomaly detection error: {e}")
            return pd.DataFrame()

    def run(self, rebalance_interval_days=1):
        """
        Runs the vectorized backtest simulation.
        """
        self._prepare_data_and_models()
        if not self.crypto_symbols:
            print("No valid crypto symbols with data available.")
            return pd.DataFrame(), []

        print(f"--- Backtester: Running Simulation (Rebalance every {rebalance_interval_days} days) ---")

        # Log data details for each crypto
        for crypto in self.crypto_symbols:
            df = self.ohlcv_data[crypto]
            print(f"{crypto}: {len(df)} rows, {df.index.min()} to {df.index.max()}")

        # Create simulation dates (hourly)
        sim_dates = pd.date_range(
            start=pd.to_datetime(self.start_date),
            end=pd.to_datetime(self.end_date),
            freq='H'
        )

        if len(sim_dates) == 0:
            print("ERROR: No simulation dates generated. Check start_date and end_date.")
            return pd.DataFrame(), []

        portfolio_values_over_time = []
        current_cash = self.initial_capital
        current_units = {crypto: 0.0 for crypto in self.crypto_symbols}
        self.portfolio_optimizer.cash_balance = current_cash
        last_rebalance_sim_date = None

        for current_sim_date in sim_dates:
            # Get current prices
            current_day_prices = {}
            data_available = True
            for crypto in self.crypto_symbols:
                crypto_data = self.ohlcv_data[crypto]
                valid_data = crypto_data[crypto_data.index <= current_sim_date]
                if valid_data.empty:
                    print(f"  WARNING: No data for {crypto} before {current_sim_date}")
                    data_available = False
                    break

                # Find closest timestamp
                time_diffs = abs(valid_data.index - current_sim_date)
                closest_idx = time_diffs.argmin()
                closest_time = valid_data.index[closest_idx]
                time_diff = time_diffs[closest_idx]

                if time_diff <= pd.Timedelta(hours=1):
                    current_day_prices[crypto] = valid_data.loc[closest_time, 'close_price']
                    if time_diff > pd.Timedelta(minutes=1):
                        print(f"  INFO: Using closest price for {crypto} at {closest_time} (offset: {time_diff})")
                else:
                    # Interpolate within 2 hours
                    if time_diff <= pd.Timedelta(hours=2) and len(valid_data) > 1:
                        before_data = valid_data[valid_data.index < current_sim_date]
                        after_data = crypto_data[crypto_data.index > current_sim_date]
                        if not before_data.empty and not after_data.empty:
                            t1 = before_data.index[-1]
                            t2 = after_data.index[0]
                            p1 = before_data['close_price'].iloc[-1]
                            p2 = after_data['close_price'].iloc[0]
                            time_total = (t2 - t1).total_seconds()
                            time_frac = (current_sim_date - t1).total_seconds() / time_total
                            interpolated_price = p1 + (p2 - p1) * time_frac
                            current_day_prices[crypto] = interpolated_price
                            print(f"  INFO: Interpolated price {interpolated_price:.2f} for {crypto} at {current_sim_date}")
                        else:
                            print(f"  WARNING: Cannot interpolate for {crypto} at {current_sim_date}")
                            data_available = False
                            break
                    else:
                        print(f"  WARNING: No price for {crypto} within 2 hours of {current_sim_date}")
                        data_available = False
                        break

            if not data_available:
                print(f"  Skipping {current_sim_date}: Price data missing for one or more assets.")
                if portfolio_values_over_time:
                    portfolio_values_over_time.append(portfolio_values_over_time[-1].copy())
                continue

            # Update portfolio
            self.portfolio_optimizer.current_asset_holdings_units = current_units.copy()
            self.portfolio_optimizer.cash_balance = current_cash
            current_total_pv = self.portfolio_optimizer._update_portfolio_metrics_and_allocations(current_day_prices)

            # Log portfolio value
            log_entry = {'date': current_sim_date, 'portfolio_value': current_total_pv, 'cash': current_cash}
            for crypto in self.crypto_symbols:
                log_entry[f'{crypto}_units'] = current_units[crypto]
                log_entry[f'{crypto}_price'] = current_day_prices.get(crypto, np.nan)
            portfolio_values_over_time.append(log_entry)

            # Rebalance
            if last_rebalance_sim_date is None or (current_sim_date - last_rebalance_sim_date).days >= rebalance_interval_days:
                print(f"\n  Attempting rebalance on {current_sim_date}...")
                last_rebalance_sim_date = current_sim_date

                pit_predictions = {}
                pit_volatilities = {}
                pit_anomalies = {}

                for crypto in self.crypto_symbols:
                    data_for_features = self.featured_data[crypto][self.featured_data[crypto].index < current_sim_date]
                    if len(data_for_features) < self.sequence_len:
                        print(f"    Not enough historical feature data for {crypto} on {current_sim_date}. Skipping.")
                        pit_predictions[crypto] = pd.DataFrame()
                        pit_volatilities[crypto] = self._get_point_in_time_volatility(data_for_features.iloc[-120:])
                        pit_anomalies[crypto] = False
                        continue

                    # Prediction
                    if crypto in self.models['prediction']:
                        model_info = self.models['prediction'][crypto]
                        input_slice = data_for_features[model_info['features']].iloc[-self.sequence_len:]
                        if len(input_slice) == self.sequence_len:
                            pit_predictions[crypto] = self.predict_future_prices(
                                model_info['model'], model_info['scalers'], model_info['features'], input_slice
                            )
                        else:
                            pit_predictions[crypto] = pd.DataFrame()
                    else:
                        pit_predictions[crypto] = pd.DataFrame()

                    # Volatility
                    raw_data_slice = self.ohlcv_data[crypto][self.ohlcv_data[crypto].index < current_sim_date]
                    pit_volatilities[crypto] = self._get_point_in_time_volatility(raw_data_slice)

                    # Anomaly
                    if crypto in self.models['anomaly']:
                        anomaly_info = self.models['anomaly'][crypto]
                        recent_slice = data_for_features.iloc[-5:]
                        if not recent_slice.empty:
                            anomaly_df = self.detect_anomalies(
                                anomaly_info['model'], anomaly_info['scaler'], anomaly_info['threshold'], recent_slice
                            )
                            pit_anomalies[crypto] = anomaly_df['is_anomaly'].iloc[-1] if not anomaly_df.empty else False
                        else:
                            pit_anomalies[crypto] = False
                    else:
                        pit_anomalies[crypto] = False

                # Target allocations
                target_allocs = self.portfolio_optimizer.generate_target_percentage_allocations(
                    pit_predictions, pit_volatilities, current_day_prices, pit_anomalies
                )

                # Simulate trades
                self.portfolio_optimizer.current_asset_holdings_units = current_units.copy()
                self.portfolio_optimizer.cash_balance = current_cash
                self.portfolio_optimizer._update_portfolio_metrics_and_allocations(current_day_prices)
                trades_made_log = self.portfolio_optimizer.simulate_rebalance_to_targets(
                    target_allocs, current_day_prices
                )
                self.trades_log_list.extend([dict(item, date=current_sim_date) for item in trades_made_log])

                # Update state
                current_units = self.portfolio_optimizer.current_asset_holdings_units.copy()
                current_cash = self.portfolio_optimizer.cash_balance

                # Update log
                current_total_pv = self.portfolio_optimizer.total_portfolio_value_pv
                portfolio_values_over_time[-1]['portfolio_value'] = current_total_pv
                portfolio_values_over_time[-1]['cash'] = current_cash
                for crypto in self.crypto_symbols:
                    portfolio_values_over_time[-1][f'{crypto}_units'] = current_units[crypto]

        self.portfolio_history_df = pd.DataFrame(portfolio_values_over_time)
        if not self.portfolio_history_df.empty:
            self.portfolio_history_df.set_index('date', inplace=True)
        return self.portfolio_history_df, self.trades_log_list

if __name__ == "__main__":
    backtester_instance = Backtester(
        start_date="2024-11-14",
        end_date="2025-05-13",
        crypto_symbols=TARGET_CRYPTOS,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        sequence_len=SEQUENCE_LENGTH,
        prediction_horizon=PREDICTION_HORIZON
    )
    history_df, trades = backtester_instance.run(rebalance_interval_days=1)
    if not history_df.empty:
        print("\nPortfolio History:")
        print(history_df)
        print("\nTrades Log:")
        print(pd.DataFrame(trades))
    else:
        print("Backtest did not generate any history. Check data and model availability.")