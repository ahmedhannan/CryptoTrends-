import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import joblib
import json

from config import (
    MODEL_SAVE_DIR, SEQUENCE_LENGTH, PREDICTION_HORIZON,
    D_MODEL, NUM_HEADS, FF_DIM, NUM_ENCODER_LAYERS, DROPOUT_RATE,
    EPOCHS, BATCH_SIZE
)
from data_preprocessor import split_data_chronologically, scale_data, create_sequences, inverse_scale_data

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.dropout_attention = Dropout(dropout_rate)
        self.layer_norm_attention = LayerNormalization(epsilon=1e-6)
        self.ffn_dense1 = Dense(ff_dim, activation="relu")
        self.ffn_dense2 = Dense(d_model)
        self.dropout_ffn = Dropout(dropout_rate)
        self.layer_norm_ffn = LayerNormalization(epsilon=1e-6)
        self.add_attention = Add()
        self.add_ffn = Add()

    def call(self, inputs, training=False):
        attention_output = self.multi_head_attention(query=inputs, value=inputs, key=inputs, training=training)
        attention_output = self.dropout_attention(attention_output, training=training)
        out1 = self.layer_norm_attention(self.add_attention([inputs, attention_output]))
        ffn_output = self.ffn_dense1(out1)
        ffn_output = self.ffn_dense2(ffn_output)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        out2 = self.layer_norm_ffn(self.add_ffn([out1, ffn_output]))
        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config

def build_transformer_forecaster(input_shape_tuple, num_encoder_blocks, model_dim, num_attention_heads,
                                feed_forward_dim, output_prediction_horizon, num_output_prediction_features,
                                model_dropout_rate=0.1):
    input_seq_len, num_input_features = input_shape_tuple
    inputs = Input(shape=input_shape_tuple)
    if num_input_features != model_dim:
        x = Dense(model_dim, activation='relu', name="input_projection")(inputs)
    else:
        x = inputs
    x = Dropout(model_dropout_rate, name="input_dropout")(x)
    position_embeddings = tf.keras.layers.Embedding(input_dim=input_seq_len, output_dim=model_dim, name="positional_embedding")(tf.range(start=0, limit=input_seq_len, delta=1))
    x = x + position_embeddings
    for i in range(num_encoder_blocks):
        encoder_layer = TransformerEncoderLayer(
            d_model=model_dim, num_heads=num_attention_heads, ff_dim=feed_forward_dim,
            dropout_rate=model_dropout_rate, name=f"transformer_encoder_layer_{i+1}"
        )
        x = encoder_layer(x)
    x = GlobalAveragePooling1D(name="global_avg_pooling")(x)
    x = Dropout(0.1, name="output_dropout_1")(x)
    x = Dense(128, activation="relu", name="output_dense_1")(x)
    x = Dropout(0.1, name="output_dropout_2")(x)
    final_output_units = output_prediction_horizon * num_output_prediction_features
    outputs = Dense(final_output_units, name="final_output_dense")(x)
    outputs = tf.keras.layers.Reshape((output_prediction_horizon, num_output_prediction_features), name="output_reshape")(outputs)
    model = Model(inputs=inputs, outputs=outputs, name="transformer_forecaster")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mean_squared_error',
                  metrics=['mean_absolute_error', tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_forecasting_model(crypto_name, all_features_dataframe, target_col_for_inverse_scaling='close_price'):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_file_path = os.path.join(MODEL_SAVE_DIR, f"{crypto_name}_transformer_forecaster.keras")
    scalers_file_path = os.path.join(MODEL_SAVE_DIR, f"{crypto_name}_transformer_scalers.joblib")
    features_list_path = os.path.join(MODEL_SAVE_DIR, f"{crypto_name}_transformer_features.json")

    print(f"\n--- Training Transformer Forecasting Model for {crypto_name} ---")

    model_input_feature_cols = all_features_dataframe.select_dtypes(include=np.number).columns.tolist()
    if not model_input_feature_cols:
        raise ValueError(f"No numerical feature columns found for {crypto_name}.")
    if target_col_for_inverse_scaling not in model_input_feature_cols:
        raise ValueError(f"Target column '{target_col_for_inverse_scaling}' not found for {crypto_name}.")
    print(f"Selected {len(model_input_feature_cols)} features: {model_input_feature_cols}")

    try:
        train_df, val_df, _ = split_data_chronologically(all_features_dataframe[model_input_feature_cols], crypto_name)
    except ValueError as e:
        print(f"ERROR splitting data for {crypto_name}: {e}.")
        return None, None, None

    scaled_train_df, fitted_scalers = scale_data(train_df, model_input_feature_cols)
    scaled_val_df = val_df.copy()
    for col in model_input_feature_cols:
        if col in fitted_scalers:
            scaled_val_df[col] = fitted_scalers[col].transform(val_df[[col]]).flatten()
        else:
            print(f"WARNING: Scaler for '{col}' not found in validation set.")

    X_train, y_train = create_sequences(scaled_train_df.values, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    X_val, y_val = create_sequences(scaled_val_df.values, SEQUENCE_LENGTH, PREDICTION_HORIZON)

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        print(f"ERROR: Not enough data to create sequences for {crypto_name}. Training cannot proceed.")
        return None, None, None

    print(f"Training sequences: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Validation sequences: X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    transformer_model = build_transformer_forecaster(
        input_shape_tuple=(SEQUENCE_LENGTH, X_train.shape[2]),
        num_encoder_blocks=NUM_ENCODER_LAYERS,
        model_dim=D_MODEL,
        num_attention_heads=NUM_HEADS,
        feed_forward_dim=FF_DIM,
        output_prediction_horizon=PREDICTION_HORIZON,
        num_output_prediction_features=y_train.shape[2],
        model_dropout_rate=DROPOUT_RATE
    )
    transformer_model.summary()

    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)
    checkpoint_cb = ModelCheckpoint(model_file_path, monitor='val_loss', save_best_only=True, verbose=1)

    history = transformer_model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping_cb, reduce_lr_cb, checkpoint_cb],
        verbose=1
    )

    joblib.dump(fitted_scalers, scalers_file_path)
    with open(features_list_path, 'w') as f_json:
        json.dump(model_input_feature_cols, f_json)

    best_model = load_model(model_file_path, custom_objects={'TransformerEncoderLayer': TransformerEncoderLayer})
    print(f"Transformer model, scalers, and features for {crypto_name} saved to '{MODEL_SAVE_DIR}'.")
    return best_model, fitted_scalers, model_input_feature_cols

def load_forecasting_model(crypto_name):
    model_file_path = os.path.join(MODEL_SAVE_DIR, f"{crypto_name}_transformer_forecaster.keras")
    scalers_file_path = os.path.join(MODEL_SAVE_DIR, f"{crypto_name}_transformer_scalers.joblib")
    features_list_path = os.path.join(MODEL_SAVE_DIR, f"{crypto_name}_transformer_features.json")

    if not all(os.path.exists(p) for p in [model_file_path, scalers_file_path, features_list_path]):
        print(f"INFO: Model, scalers, or features not found for {crypto_name} at '{MODEL_SAVE_DIR}'.")
        return None, None, None

    print(f"Loading Transformer model, scalers, and features for {crypto_name}...")
    try:
        model = load_model(model_file_path, custom_objects={'TransformerEncoderLayer': TransformerEncoderLayer})
        scalers = joblib.load(scalers_file_path)
        with open(features_list_path, 'r') as f_json:
            model_feature_columns = json.load(f_json)
        print("Model, scalers, and features loaded successfully.")
        return model, scalers, model_feature_columns
    except Exception as e:
        print(f"ERROR loading model for {crypto_name}: {e}")
        return None, None, None

def predict_future_prices(keras_model, data_scalers_dict, model_input_features_list,
                          recent_ohlcv_featured_df, main_target_feature_name='close_price'):
    if len(recent_ohlcv_featured_df) < SEQUENCE_LENGTH:
        print(f"ERROR: Need at least {SEQUENCE_LENGTH} recent data points, got {len(recent_ohlcv_featured_df)}.")
        return pd.DataFrame()

    try:
        input_df_for_model = recent_ohlcv_featured_df[model_input_features_list].copy()
    except KeyError as e:
        print(f"ERROR: Missing feature columns for prediction: {e}. Required: {model_input_features_list}")
        return pd.DataFrame()

    scaled_input_df_for_model = input_df_for_model.copy()
    for col_name in model_input_features_list:
        if col_name in data_scalers_dict:
            scaler = data_scalers_dict[col_name]
            scaled_input_df_for_model[col_name] = scaler.transform(input_df_for_model[[col_name]]).flatten()
        else:
            print(f"ERROR: Scaler for feature '{col_name}' not found.")
            return pd.DataFrame()

    last_input_sequence_scaled = scaled_input_df_for_model.values[-SEQUENCE_LENGTH:]
    last_input_sequence_scaled = np.expand_dims(last_input_sequence_scaled, axis=0)

    scaled_multi_feature_predictions = keras_model.predict(last_input_sequence_scaled, verbose=0)[0]

    try:
        target_feature_index_in_model = model_input_features_list.index(main_target_feature_name)
    except ValueError:
        print(f"ERROR: Target feature '{main_target_feature_name}' not found in model features.")
        return pd.DataFrame()

    predicted_target_feature_scaled_array = scaled_multi_feature_predictions[:, target_feature_index_in_model]
    predicted_target_feature_original_scale = inverse_scale_data(
        predicted_target_feature_scaled_array, main_target_feature_name, data_scalers_dict
    )

    last_known_timestamp = recent_ohlcv_featured_df.index[-1]
    future_prediction_timestamps = pd.date_range(
        start=last_known_timestamp + pd.Timedelta(hours=1),
        periods=PREDICTION_HORIZON,
        freq='H'
    )

    predictions_output_df = pd.DataFrame(
        data=predicted_target_feature_original_scale,
        index=future_prediction_timestamps,
        columns=[f'predicted_{main_target_feature_name}']
    )

    # Validate predictions
    last_actual_price = recent_ohlcv_featured_df[main_target_feature_name].iloc[-1]
    predicted_price = predictions_output_df.iloc[0, 0]
    price_change_pct = abs(predicted_price - last_actual_price) / last_actual_price * 100
    if price_change_pct > 10:
        print(f"WARNING: Predicted price change for {main_target_feature_name} is {price_change_pct:.2f}% in 1 hour, which is unusually high.")

    return predictions_output_df

if __name__ == '__main__':
    from feature_engineer import add_technical_indicators
    from excel_data_loader import load_ohlcv_from_excel
    from config import TARGET_CRYPTOS

    crypto_to_test = TARGET_CRYPTOS[0]
    print(f"--- Prediction Model Example Workflow for {crypto_to_test} ---")

    raw_crypto_data = load_ohlcv_from_excel(crypto_to_test)
    if raw_crypto_data.empty or len(raw_crypto_data) < 250:
        print(f"Not enough raw data for {crypto_to_test}. Exiting example.")
        exit()

    featured_crypto_data = add_technical_indicators(raw_crypto_data.copy())
    if featured_crypto_data.empty:
        print(f"Feature engineering failed for {crypto_to_test}. Exiting example.")
        exit()

    loaded_model, loaded_scalers, loaded_features = load_forecasting_model(crypto_to_test)
    if not loaded_model:
        print(f"No pre-trained model found for {crypto_to_test}. Training a new one...")
        loaded_model, loaded_scalers, loaded_features = train_forecasting_model(
            crypto_to_test, featured_crypto_data.copy(), target_col_for_inverse_scaling='close_price'
        )

    if loaded_model and loaded_scalers and loaded_features:
        recent_data_for_prediction = featured_crypto_data.iloc[-SEQUENCE_LENGTH:]
        if len(recent_data_for_prediction) < SEQUENCE_LENGTH:
            print(f"ERROR: Not enough recent data points ({len(recent_data_for_prediction)}).")
        else:
            print(f"\nMaking prediction for {crypto_to_test} using last {SEQUENCE_LENGTH} hours...")
            future_price_predictions_df = predict_future_prices(
                loaded_model, loaded_scalers, loaded_features, recent_data_for_prediction, 'close_price'
            )
            if not future_price_predictions_df.empty:
                print(f"\nFuture 'close_price' predictions for {crypto_to_test} ({PREDICTION_HORIZON} hours ahead):")
                print(future_price_predictions_df)
                print(f"\nLast actual close price: {featured_crypto_data['close_price'].iloc[-1]:.2f} on {featured_crypto_data.index[-1]}")
            else:
                print(f"Failed to generate predictions for {crypto_to_test}.")
    else:
        print(f"ERROR: Could not train or load the forecasting model for {crypto_to_test}.")