import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import os
import joblib

from config import (
    MODEL_SAVE_DIR, ANOMALY_FEATURE_COLS, ANOMALY_ENCODING_DIM,
    ANOMALY_EPOCHS, ANOMALY_BATCH_SIZE, ANOMALY_THRESHOLD_PERCENTILE, TARGET_CRYPTOS
)
from data_preprocessor import split_data_chronologically

def build_autoencoder_model(input_dimension, encoding_dimension=ANOMALY_ENCODING_DIM):
    """Builds a simple dense autoencoder model for anomaly detection."""
    input_layer = Input(shape=(input_dimension,), name="autoencoder_input")
    encoder = Dense(encoding_dimension * 2, activation="relu", name="encoder_dense1")(input_layer)
    encoder = Dense(encoding_dimension, activation="relu", name="encoder_latent_space")(encoder)
    decoder = Dense(encoding_dimension * 2, activation="relu", name="decoder_dense1")(encoder)
    output_layer = Dense(input_dimension, activation="linear", name="autoencoder_output")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=output_layer, name="autoencoder_anomaly_detector")
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

def train_anomaly_detector_model(crypto_name_for_log, crypto_featured_dataframe):
    """
    Trains an autoencoder model for anomaly detection for a specific cryptocurrency.
    Uses features specified in ANOMALY_FEATURE_COLS from the config.
    Saves the trained autoencoder, the scaler used for its features, and the anomaly threshold.
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_file_path = os.path.join(MODEL_SAVE_DIR, f"{crypto_name_for_log}_autoencoder_anomaly.keras")
    scaler_file_path = os.path.join(MODEL_SAVE_DIR, f"{crypto_name_for_log}_autoencoder_scaler.joblib")
    threshold_file_path = os.path.join(MODEL_SAVE_DIR, f"{crypto_name_for_log}_autoencoder_threshold.joblib")

    print(f"\n--- Training Anomaly Detector (Autoencoder) for {crypto_name_for_log} ---")

    if not all(col in crypto_featured_dataframe.columns for col in ANOMALY_FEATURE_COLS):
        missing_anomaly_cols = [col for col in ANOMALY_FEATURE_COLS if col not in crypto_featured_dataframe.columns]
        print(f"ERROR: Missing required columns for anomaly detection in {crypto_name_for_log}: {missing_anomaly_cols}. "
              f"Please ensure these are created in feature_engineer.py. Skipping training.")
        return None, None, None

    anomaly_input_data_df = crypto_featured_dataframe[ANOMALY_FEATURE_COLS].copy().dropna()
    if anomaly_input_data_df.empty or len(anomaly_input_data_df) < 50:
        print(f"WARNING: Not enough valid data points ({len(anomaly_input_data_df)}) for anomaly detection training on {crypto_name_for_log}. Skipping.")
        return None, None, None

    try:
        train_anomaly_df, _, _ = split_data_chronologically(anomaly_input_data_df, f"{crypto_name_for_log}_anomaly_split")
    except ValueError as e:
        print(f"ERROR splitting data for {crypto_name_for_log} anomaly detector: {e}. Training cannot proceed.")
        return None, None, None

    if train_anomaly_df.empty:
        print(f"ERROR: No training data available for {crypto_name_for_log} anomaly detector after split. Skipping.")
        return None, None, None

    feature_scaler = StandardScaler()
    train_features_scaled = feature_scaler.fit_transform(train_anomaly_df)

    autoencoder_model = build_autoencoder_model(input_dimension=train_features_scaled.shape[1])
    print(f"Autoencoder model summary for {crypto_name_for_log}:")
    autoencoder_model.summary()

    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    checkpoint_cb = ModelCheckpoint(model_file_path, monitor='val_loss', save_best_only=True, verbose=0)

    print(f"Fitting autoencoder for {crypto_name_for_log} with {len(train_features_scaled)} 'normal' samples...")
    history = autoencoder_model.fit(
        train_features_scaled, train_features_scaled,
        epochs=ANOMALY_EPOCHS,
        batch_size=ANOMALY_BATCH_SIZE,
        shuffle=True,
        validation_split=0.15,
        callbacks=[early_stopping_cb, checkpoint_cb],
        verbose=1
    )

    best_autoencoder_model = load_model(model_file_path)
    train_data_predictions = best_autoencoder_model.predict(train_features_scaled, verbose=0)
    train_reconstruction_mse = np.mean(np.power(train_features_scaled - train_data_predictions, 2), axis=1)

    calculated_anomaly_threshold = np.percentile(train_reconstruction_mse, ANOMALY_THRESHOLD_PERCENTILE)
    print(f"Anomaly threshold for {crypto_name_for_log}: {calculated_anomaly_threshold:.6f} (based on {ANOMALY_THRESHOLD_PERCENTILE}th percentile)")

    joblib.dump(feature_scaler, scaler_file_path)
    joblib.dump(calculated_anomaly_threshold, threshold_file_path)
    print(f"Autoencoder model, scaler, and threshold for {crypto_name_for_log} saved to '{MODEL_SAVE_DIR}'.")

    return best_autoencoder_model, feature_scaler, calculated_anomaly_threshold

def load_anomaly_detector_components(crypto_name_for_log):
    """Loads a pre-trained autoencoder model, its feature scaler, and anomaly threshold."""
    model_file_path = os.path.join(MODEL_SAVE_DIR, f"{crypto_name_for_log}_autoencoder_anomaly.keras")
    scaler_file_path = os.path.join(MODEL_SAVE_DIR, f"{crypto_name_for_log}_autoencoder_scaler.joblib")
    threshold_file_path = os.path.join(MODEL_SAVE_DIR, f"{crypto_name_for_log}_autoencoder_threshold.joblib")

    if not all(os.path.exists(p) for p in [model_file_path, scaler_file_path, threshold_file_path]):
        print(f"INFO: Anomaly detector model/scaler/threshold not found for {crypto_name_for_log} at '{MODEL_SAVE_DIR}'. Needs training.")
        return None, None, None

    print(f"Loading anomaly detector components for {crypto_name_for_log} from '{MODEL_SAVE_DIR}'...")
    try:
        model = load_model(model_file_path)
        scaler = joblib.load(scaler_file_path)
        threshold = joblib.load(threshold_file_path)
        print("Anomaly detector components loaded successfully.")
        return model, scaler, threshold
    except Exception as e:
        print(f"ERROR loading anomaly detector components for {crypto_name_for_log}: {e}")
        return None, None, None

def detect_anomalies_with_autoencoder(autoencoder_model, feature_scaler, anomaly_threshold, recent_data_with_features_df):
    """
    Detects anomalies in recent data using the trained autoencoder.
    """
    output_df_with_anomalies = recent_data_with_features_df.copy()

    if autoencoder_model is None or feature_scaler is None or anomaly_threshold is None:
        print("WARNING: Anomaly detector model, scaler, or threshold not available. Cannot detect anomalies.")
        output_df_with_anomalies['anomaly_reconstruction_error'] = 0.0
        output_df_with_anomalies['is_anomaly'] = False
        return output_df_with_anomalies

    if not all(col in output_df_with_anomalies.columns for col in ANOMALY_FEATURE_COLS):
        missing_cols = [col for col in ANOMALY_FEATURE_COLS if col not in output_df_with_anomalies.columns]
        print(f"ERROR: Missing columns required for anomaly detection: {missing_cols}. Cannot proceed.")
        output_df_with_anomalies['anomaly_reconstruction_error'] = pd.NA
        output_df_with_anomalies['is_anomaly'] = pd.NA
        return output_df_with_anomalies

    data_to_check_for_anomalies = output_df_with_anomalies[ANOMALY_FEATURE_COLS].copy().dropna()
    if data_to_check_for_anomalies.empty:
        print("No valid data points to check for anomalies after selecting features and dropping NaNs.")
        output_df_with_anomalies['anomaly_reconstruction_error'] = 0.0
        output_df_with_anomalies['is_anomaly'] = False
        return output_df_with_anomalies

    scaled_data_to_check = feature_scaler.transform(data_to_check_for_anomalies)
    reconstructed_data = autoencoder_model.predict(scaled_data_to_check, verbose=0)
    reconstruction_mse_per_sample = np.mean(np.power(scaled_data_to_check - reconstructed_data, 2), axis=1)

    anomaly_results_series_mse = pd.Series(reconstruction_mse_per_sample, index=data_to_check_for_anomalies.index)
    output_df_with_anomalies['anomaly_reconstruction_error'] = anomaly_results_series_mse
    output_df_with_anomalies['is_anomaly'] = output_df_with_anomalies['anomaly_reconstruction_error'] > anomaly_threshold

    output_df_with_anomalies['anomaly_reconstruction_error'].fillna(0.0, inplace=True)
    output_df_with_anomalies['is_anomaly'].fillna(False, inplace=True)

    return output_df_with_anomalies

if __name__ == '__main__':
    from feature_engineer import add_technical_indicators
    from excel_data_loader import load_ohlcv_from_excel

    if not TARGET_CRYPTOS:
        print("No TARGET_CRYPTOS in config.py for anomaly detector test.")
        exit()

    test_crypto_name = TARGET_CRYPTOS[0]
    print(f"--- Anomaly Detector (Autoencoder) Standalone Example for {test_crypto_name} ---")

    crypto_raw_data = load_ohlcv_from_excel(test_crypto_name)
    if crypto_raw_data.empty or len(crypto_raw_data) < 150:
        print(f"Not enough raw data for {test_crypto_name} to run anomaly detector example. Min 150 hours needed. Got {len(crypto_raw_data)}")
        exit()

    crypto_featured_data = add_technical_indicators(crypto_raw_data.copy())
    if crypto_featured_data.empty or not all(col in crypto_featured_data.columns for col in ANOMALY_FEATURE_COLS):
        print(f"Feature engineering failed or did not produce required anomaly features for {test_crypto_name}. Exiting.")
        exit()

    anomaly_model, anomaly_scaler, anomaly_thresh = load_anomaly_detector_components(test_crypto_name)
    if not anomaly_model:
        print(f"No pre-trained anomaly detector found for {test_crypto_name}. Training a new one...")
        anomaly_model, anomaly_scaler, anomaly_thresh = train_anomaly_detector_model(test_crypto_name, crypto_featured_data.copy())

    if anomaly_model and anomaly_scaler and anomaly_thresh is not None:
        print(f"\nAnomaly detector for {test_crypto_name} is ready. Threshold: {anomaly_thresh:.6f}")
        recent_data_slice_for_detection = crypto_featured_data.iloc[-60:]
        if recent_data_slice_for_detection.empty:
            print("No recent data available to detect anomalies.")
        else:
            print(f"\nDetecting anomalies on the last {len(recent_data_slice_for_detection)} data points for {test_crypto_name}...")
            data_with_anomaly_info = detect_anomalies_with_autoencoder(
                anomaly_model, anomaly_scaler, anomaly_thresh, recent_data_slice_for_detection
            )
            cols_to_show = ANOMALY_FEATURE_COLS + ['anomaly_reconstruction_error', 'is_anomaly']
            if 'close_price' in data_with_anomaly_info.columns:
                cols_to_show = ['close_price'] + cols_to_show
            print(data_with_anomaly_info[cols_to_show].tail(10))
            identified_anomalies = data_with_anomaly_info[data_with_anomaly_info['is_anomaly']]
            if not identified_anomalies.empty:
                print(f"\n{len(identified_anomalies)} anomalies identified in the recent data for {test_crypto_name}:")
                print(identified_anomalies[cols_to_show].tail())
            else:
                print(f"\nNo anomalies detected in the recent data for {test_crypto_name}.")
    else:
        print(f"ERROR: Could not train or load the anomaly detector for {test_crypto_name}.")