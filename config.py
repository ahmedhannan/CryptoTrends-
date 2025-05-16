EXCEL_DATA_DIRECTORY = 'crypto_data_excel/'

# -- Target Cryptocurrencies --
#  names should match  Excel file names (without .xlsx)
TARGET_CRYPTOS = [
    'bitcoin',
    'ethereum',
    'ripple',
    'binancecoin',
    'solana'
   
]


EXCEL_COLUMN_MAP = {
    'Date': 'timestamp',       
    'timestamp': 'timestamp',  
    'Time': 'timestamp',       
    # --- Map your OHLCV columns ---
    'Open': 'open_price',      
    'open_price': 'open_price',      
    'High': 'high_price',
    'high_price': 'high_price',
    'Low': 'low_price',
    'low_price': 'low_price',
    'Close': 'close_price',    
    'Adj Close': 'close_price',
    'close_price': 'close_price',
    'Volume': 'volume',       
    'volume': 'volume',
    'Vol.': 'volume',
    # Add any other mappings if  columns are named differently.
    
}


# -- Model Parameters --
SEQUENCE_LENGTH = 30  # Number of past data points (e.g., days) to use for prediction
PREDICTION_HORIZON = 1 # Number of future data points (e.g., days) to predict
TEST_SET_SIZE_HOURS = 720  # 30 days × 24 hours
VALIDATION_SET_SIZE_HOURS = 720  # 30 days × 24 hours

# Transformer Model Specific
D_MODEL = 64          # Dimensionality of the model's internal layers
NUM_HEADS = 4          # Number of attention heads in MultiHeadAttention
NUM_ENCODER_LAYERS = 2 # Number of Transformer encoder blocks
FF_DIM = 128           # Hidden layer size in the feed-forward network part of Transformer
DROPOUT_RATE = 0.1     # Dropout rate for regularization
EPOCHS = 30            # Number of training epochs (can be increased for better results)
BATCH_SIZE = 32        # Number of samples per gradient update

# Anomaly Detection (Autoencoder)
ANOMALY_FEATURE_COLS = ['close_price_change_pct', 'volume_change_pct'] # Features for anomaly model
ANOMALY_ENCODING_DIM = 8 # Latent space dimension for the autoencoder
ANOMALY_EPOCHS = 30
ANOMALY_BATCH_SIZE = 32
ANOMALY_THRESHOLD_PERCENTILE = 95 # Reconstruction errors above this percentile are considered anomalies

# -- Paths --
MODEL_SAVE_DIR = 'trained_models' # Directory to save/load trained models & scalers
LOG_FILE = 'agent_run.log'    # Log file for agent's operational messages

# -- Portfolio Optimization --
RISK_FREE_RATE = 0.01 # Annualized risk-free rate (e.g., 1% for government bonds)
INITIAL_CAPITAL = 10000 # Example starting capital for portfolio simulations
TRANSACTION_COST_PCT = 0.001 # Percentage cost per transaction (e.g., 0.1%)

# -- Alerting --
ENABLE_EMAIL_ALERTS = True # Set to True to enable email alerts (requires SMTP setup below)
ALERT_EMAIL_FROM = 'talha.asif67@gmail.com'
ALERT_EMAIL_TO = 'chaudaryhammad140@gmail.com' # Can be a list: ['email1@example.com', 'email2@example.com']
ALERT_SMTP_SERVER = 'smtp.gmail.com' # Example for Gmail
ALERT_SMTP_PORT = 587 # For TLS (or 465 for SSL)
ALERT_SMTP_USER = 'talha.asif67@gmail.com' # Your email username
ALERT_SMTP_PASSWORD = 'erhk riry ndtm wsom' # For Gmail, use an "App Password"

