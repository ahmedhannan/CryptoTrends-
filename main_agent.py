import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import json

from config import (
    TARGET_CRYPTOS, MODEL_SAVE_DIR, SEQUENCE_LENGTH, PREDICTION_HORIZON,
    VALIDATION_SET_SIZE_HOURS, TEST_SET_SIZE_HOURS, RISK_FREE_RATE, INITIAL_CAPITAL,
    TRANSACTION_COST_PCT, ANOMALY_FEATURE_COLS, LOG_FILE
)
from excel_data_loader import load_ohlcv_from_excel
from feature_engineer import add_technical_indicators
from prediction_model import train_forecasting_model, load_forecasting_model, predict_future_prices, TransformerEncoderLayer
from anomaly_detector import train_anomaly_detector_model, load_anomaly_detector_components, detect_anomalies_with_autoencoder
from portfolio_optimizer import PortfolioOptimizer
from alert_system import send_alert

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def get_historical_volatility_for_portfolio(crypto_ohlcv_df, lookback_window=60*24, trading_days_per_year=365):
    default_volatility = 0.75
    if crypto_ohlcv_df.empty or len(crypto_ohlcv_df) < lookback_window + 1:
        return default_volatility
    log_returns = np.log(crypto_ohlcv_df['close_price'] / crypto_ohlcv_df['close_price'].shift(1)).dropna()
    if log_returns.empty or len(log_returns) < 2:
        return default_volatility
    recent_log_returns = log_returns.iloc[-lookback_window:]
    if len(recent_log_returns) < 2:
        return default_volatility
    hourly_std_dev = recent_log_returns.std()
    annualized_vol = hourly_std_dev * np.sqrt(trading_days_per_year * 24)
    return annualized_vol if pd.notna(annualized_vol) and annualized_vol > 0 else default_volatility

def run_agent_orchestration_cycle():
    cycle_start_time = time.time()
    current_timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_timestamp_str}] --- Starting Crypto Agent Orchestration Cycle (Data Source: Excel) ---")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    all_cryptos_raw_data_dict = {}
    all_cryptos_featured_data_dict = {}
    latest_market_prices_dict = {}

    for crypto_symbol in TARGET_CRYPTOS:
        print(f"\nProcessing data for: {crypto_symbol}...")
        raw_data_df = load_ohlcv_from_excel(crypto_symbol)
        min_data_len_needed = SEQUENCE_LENGTH + PREDICTION_HORIZON + VALIDATION_SET_SIZE_HOURS + TEST_SET_SIZE_HOURS + 20
        if raw_data_df.empty or len(raw_data_df) < min_data_len_needed:
            msg = f"Insufficient historical data for {crypto_symbol} ({len(raw_data_df)} hours, need ~{min_data_len_needed}). Skipping."
            print(f"WARNING: {msg}")
            continue
        all_cryptos_raw_data_dict[crypto_symbol] = raw_data_df

        print(f"Adding technical features for {crypto_symbol}...")
        featured_data_df = add_technical_indicators(raw_data_df.copy())
        min_rows_needed = SEQUENCE_LENGTH + PREDICTION_HORIZON + 20
        if featured_data_df.empty or len(featured_data_df) < min_rows_needed:
            msg = f"After feature engineering, {crypto_symbol} has only {len(featured_data_df)} rows, need {min_rows_needed}. Skipping."
            print(f"ERROR: {msg}")
            continue
        all_cryptos_featured_data_dict[crypto_symbol] = featured_data_df

        if 'close_price' in featured_data_df.columns and not featured_data_df.empty:
            latest_market_prices_dict[crypto_symbol] = featured_data_df['close_price'].iloc[-1]
            print(f"Data preparation complete for {crypto_symbol}. Latest Price: ${latest_market_prices_dict[crypto_symbol]:.2f}")
        else:
            msg = f"'close_price' not found or empty for {crypto_symbol} after feature engineering. Skipping."
            print(f"ERROR: {msg}")
            all_cryptos_raw_data_dict.pop(crypto_symbol, None)
            all_cryptos_featured_data_dict.pop(crypto_symbol, None)
            continue

    active_cryptos_for_cycle = list(all_cryptos_featured_data_dict.keys())
    if not active_cryptos_for_cycle:
        msg = "No crypto data successfully processed in this cycle."
        print(f"CRITICAL: {msg}")
        return

    all_crypto_predictions_dict = {}
    all_crypto_anomaly_status_dict = {}
    all_crypto_historical_volatility_dict = {}

    for crypto_symbol in active_cryptos_for_cycle:
        print(f"\n--- {crypto_symbol}: Modeling Phase ---")
        current_featured_data_df = all_cryptos_featured_data_dict[crypto_symbol]

        print(f"Loading/Training Price Prediction Model for {crypto_symbol}...")
        pred_model, pred_scalers, pred_features_list = load_forecasting_model(crypto_symbol)
        if not pred_model:
            try:
                pred_model, pred_scalers, pred_features_list = train_forecasting_model(
                    crypto_symbol, current_featured_data_df.copy(), target_col_for_inverse_scaling='close_price'
                )
            except Exception as e_train_pred:
                msg = f"Price prediction model training FAILED for {crypto_symbol}. Error: {e_train_pred}"
                print(f"ERROR: {msg}")
                continue

        if pred_model and pred_scalers and pred_features_list:
            print(f"Generating price predictions for {crypto_symbol}...")
            recent_data_for_model_input = current_featured_data_df[pred_features_list].iloc[-SEQUENCE_LENGTH:]
            if len(recent_data_for_model_input) == SEQUENCE_LENGTH:
                predictions_df = predict_future_prices(
                    pred_model, pred_scalers, pred_features_list, recent_data_for_model_input, 'close_price'
                )
                all_crypto_predictions_dict[crypto_symbol] = predictions_df
                if not predictions_df.empty:
                    predicted_1st_step_price = predictions_df.iloc[0, 0]
                    send_alert(
                        subject_line="Price Prediction",
                        alert_body_text=f"Predicted next close (1-hour ahead): ${predicted_1st_step_price:.2f}\n"
                                        f"(Current actual: ${latest_market_prices_dict.get(crypto_symbol, 0):.2f})\n"
                                        f"Full horizon ({PREDICTION_HORIZON} hours) predictions available.",
                        severity_level="INFO",
                        crypto_context=crypto_symbol
                    )
            else:
                print(f"Not enough recent data points ({len(recent_data_for_model_input)}) for prediction input.")

        print(f"Loading/Training Anomaly Detection Model for {crypto_symbol}...")
        ad_model, ad_scaler, ad_threshold = load_anomaly_detector_components(crypto_symbol)
        if not ad_model:
            try:
                ad_model, ad_scaler, ad_threshold = train_anomaly_detector_model(crypto_symbol, current_featured_data_df.copy())
            except Exception as e_train_ad:
                msg = f"Anomaly detector model training FAILED for {crypto_symbol}. Error: {e_train_ad}"
                print(f"ERROR: {msg}")
                continue

        if ad_model and ad_scaler and ad_threshold is not None:
            print(f"Detecting anomalies for {crypto_symbol}...")
            recent_slice_for_anomaly_detection = current_featured_data_df.iloc[-10:]
            if not recent_slice_for_anomaly_detection.empty:
                data_with_anomalies_df = detect_anomalies_with_autoencoder(
                    ad_model, ad_scaler, ad_threshold, recent_slice_for_anomaly_detection
                )
                if not data_with_anomalies_df.empty and 'is_anomaly' in data_with_anomalies_df.columns:
                    latest_point_is_anomaly = data_with_anomalies_df['is_anomaly'].iloc[-1]
                    all_crypto_anomaly_status_dict[crypto_symbol] = latest_point_is_anomaly
                    if latest_point_is_anomaly:
                        latest_rec_error = data_with_anomalies_df['anomaly_reconstruction_error'].iloc[-1]
                        send_alert(
                            subject_line="Anomaly Detected!",
                            alert_body_text=f"Latest data point flagged as ANOMALY.\n"
                                            f"Reconstruction Error: {latest_rec_error:.4f} (Threshold: >{ad_threshold:.4f})",
                            severity_level="CRITICAL",
                            crypto_context=crypto_symbol
                        )
                else:
                    all_crypto_anomaly_status_dict[crypto_symbol] = False
            else:
                all_crypto_anomaly_status_dict[crypto_symbol] = False
        else:
            all_crypto_anomaly_status_dict[crypto_symbol] = False

        all_crypto_historical_volatility_dict[crypto_symbol] = get_historical_volatility_for_portfolio(
            all_cryptos_raw_data_dict[crypto_symbol]
        )
        print(f"Historical Volatility for {crypto_symbol}: {all_crypto_historical_volatility_dict[crypto_symbol]*100:.2f}%")

    optimizer_eligible_cryptos = [
        c for c in active_cryptos_for_cycle
        if c in all_crypto_predictions_dict and not all_crypto_predictions_dict[c].empty
        and c in latest_market_prices_dict and c in all_crypto_historical_volatility_dict
    ]

    if not optimizer_eligible_cryptos:
        msg = "No cryptocurrencies eligible for portfolio optimization."
        print(f"WARNING: {msg}")
    else:
        print(f"\n--- Portfolio Optimization Phase (Eligible: {optimizer_eligible_cryptos}) ---")
        filtered_predictions_for_opt = {k: v for k, v in all_crypto_predictions_dict.items() if k in optimizer_eligible_cryptos}
        filtered_latest_prices_for_opt = {k: v for k, v in latest_market_prices_dict.items() if k in optimizer_eligible_cryptos}
        filtered_volatilities_for_opt = {k: v for k, v in all_crypto_historical_volatility_dict.items() if k in optimizer_eligible_cryptos}
        filtered_anomalies_for_opt = {k: v for k, v in all_crypto_anomaly_status_dict.items() if k in optimizer_eligible_cryptos}

        if not filtered_predictions_for_opt:
            print("No valid predictions available for portfolio optimization.")
        else:
            portfolio_opt_instance = PortfolioOptimizer(
                list_of_cryptos=list(filtered_predictions_for_opt.keys()),
                initial_portfolio_capital=INITIAL_CAPITAL,
                cost_per_transaction_pct=TRANSACTION_COST_PCT
            )
            target_percentage_allocs = portfolio_opt_instance.generate_target_percentage_allocations(
                filtered_predictions_for_opt, filtered_volatilities_for_opt,
                filtered_latest_prices_for_opt, filtered_anomalies_for_opt
            )
            if target_percentage_allocs:
                simulated_trades = portfolio_opt_instance.simulate_rebalance_to_targets(
                    target_percentage_allocs, filtered_latest_prices_for_opt
                )
                if simulated_trades:
                    send_alert(
                        subject_line="Portfolio Rebalance Suggestions",
                        alert_body_text="\n".join(simulated_trades),
                        severity_level="ACTION"
                    )
                else:
                    send_alert(
                        subject_line="Portfolio Rebalance Suggestions",
                        alert_body_text="No rebalancing actions suggested.",
                        severity_level="ACTION"
                    )

    cycle_end_time = time.time()
    cycle_duration_seconds = cycle_end_time - cycle_start_time
    final_msg_body = f"Cycle finished.\nDuration: {cycle_duration_seconds:.2f} seconds.\nProcessed cryptos: {active_cryptos_for_cycle}"
    print(f"\n[{datetime.now()}] --- Crypto Agent Orchestration Cycle Completed (Duration: {cycle_duration_seconds:.2f}s) ---")

if __name__ == '__main__':
    run_agent_orchestration_cycle()