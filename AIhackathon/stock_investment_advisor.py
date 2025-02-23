import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
import os

# Set page configuration
st.set_page_config(
    page_title="Stock Investment Advisor",
    page_icon="📈",
    layout="wide"
)

# Set global random seeds and deterministic behavior
def set_all_seeds():
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Force single thread operation
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Define the functions for our model
def get_stock_data(ticker, start_date='2020-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)

    def calculate_ma(prices, window):
        return pd.Series(prices).rolling(window=window).mean()

    def calculate_rsi(prices, periods=14):
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['MA5'] = calculate_ma(df['Close'], 5)
    df['MA20'] = calculate_ma(df['Close'], 20)
    df['RSI'] = calculate_rsi(df['Close'])
    df['Price_Momentum'] = df['Close'].pct_change(5)
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
    df['Upper_Channel'] = df['High'].rolling(20).max()
    df['Lower_Channel'] = df['Low'].rolling(20).min()
    df['Channel_Position'] = (df['Close'] - df['Lower_Channel']) / (df['Upper_Channel'] - df['Lower_Channel'])

    return df.fillna(method='ffill').fillna(0)

def prepare_data(data, look_back=60):
    set_all_seeds()  # Ensure consistent data preparation
    
    feature_columns = ['Close', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'RSI',
                      'Price_Momentum', 'Volatility', 'Channel_Position']

    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(data[feature_columns])

    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform(data[['Close']])

    X, y = [], []
    for i in range(look_back, len(scaled_features)):
        X.append(scaled_features[i - look_back:i])
        y.append(scaled_target[i, 0])

    X, y = np.array(X), np.array(y)

    # Use fixed train-test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler

def create_model(look_back, n_features):
    set_all_seeds()
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(look_back, n_features),
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal',
             bias_initializer='zeros'),
        Dropout(0.2),
        LSTM(50, return_sequences=False,
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal',
             bias_initializer='zeros'),
        Dropout(0.2),
        Dense(25, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
        Dense(1, kernel_initializer='glorot_uniform', bias_initializer='zeros')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Cache for storing trained models and scalers
if 'models' not in st.session_state:
    st.session_state.models = {}

def get_model_key(ticker, investment_amount, prediction_days, data_date):
    return f"{ticker}{investment_amount}{prediction_days}_{data_date}"

# Main app
def main():
    st.title("📈 Stock Investment Advisor")
    st.write("Let's analyze your potential investment using machine learning!")

    # Sidebar for user inputs
    st.sidebar.header("Investment Parameters")

    # Predefined list of popular stocks
    popular_stocks = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Google": "GOOGL",
        "Tesla": "TSLA",
        "Meta": "META",
        "NVIDIA": "NVDA",
        "Netflix": "NFLX",
        "Disney": "DIS",
        "Coca-Cola": "KO"
    }

    # Stock selection
    selected_company = st.sidebar.selectbox(
        "Select Company",
        options=list(popular_stocks.keys())
    )
    ticker = popular_stocks[selected_company]

    # Investment amount selection
    amount_options = [1000, 5000, 10000, 50000, 100000]
    investment_amount = st.sidebar.selectbox(
        "Investment Amount ($)",
        options=amount_options
    )

    # Time period selection
    period_options = [1, 7, 14, 21, 28]
    prediction_days = st.sidebar.selectbox(
        "Prediction Period (Days)",
        options=period_options
    )

    # Add analyze button
    if st.sidebar.button("Analyze Investment"):
        with st.spinner("Analyzing your investment..."):
            try:
                # Get stock data
                df = get_stock_data(ticker)
                data_date = df.index[-1].strftime('%Y-%m-%d')
                model_key = get_model_key(ticker, investment_amount, prediction_days, data_date)
                look_back = 60

                # Check if we already have a trained model for these parameters
                if model_key not in st.session_state.models:
                    set_all_seeds()
                    X_train, X_test, y_train, y_test, feature_scaler, target_scaler = prepare_data(df, look_back)
                    
                    n_features = X_train.shape[2]
                    model = create_model(look_back, n_features)
                    
                    # Use fixed random state for validation split
                    np.random.seed(42)
                    indices = np.random.permutation(len(X_train))
                    split_idx = int(0.9 * len(X_train))
                    train_indices = indices[:split_idx]
                    val_indices = indices[split_idx:]
                    
                    X_train_final = X_train[train_indices]
                    y_train_final = y_train[train_indices]
                    X_val = X_train[val_indices]
                    y_val = y_train[val_indices]
                    
                    # Train with fixed validation data
                    model.fit(
                        X_train_final, y_train_final,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        verbose=0,
                        shuffle=False
                    )
                    
                    st.session_state.models[model_key] = {
                        'model': model,
                        'feature_scaler': feature_scaler,
                        'target_scaler': target_scaler,
                        'last_sequence': X_test[-1:],
                        'current_price': df['Close'].iloc[-1]
                    }

                # Get the stored model and data
                stored_data = st.session_state.models[model_key]
                model = stored_data['model']
                target_scaler = stored_data['target_scaler']
                last_sequence = stored_data['last_sequence']
                current_price = stored_data['current_price']

                # Make predictions
                set_all_seeds()  # Ensure consistent predictions
                future_prices = []
                current_sequence = last_sequence.copy()

                for _ in range(prediction_days):
                    next_pred = model.predict(current_sequence, verbose=0)
                    future_prices.append(target_scaler.inverse_transform(next_pred)[0, 0])

                    new_row = current_sequence[0, -1:].copy()
                    new_row[0, 0] = next_pred.item()
                    current_sequence = np.roll(current_sequence, -1, axis=1)
                    current_sequence[0, -1] = new_row
                    
                final_predicted_price = future_prices[-1]
                potential_return = ((final_predicted_price - current_price) / current_price) * 100
                predicted_value = investment_amount * (1 + potential_return / 100)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Investment Summary")
                    st.metric(
                        label="Current Stock Price",
                        value=f"${current_price:.2f}"
                    )
                    st.metric(
                        label=f"Predicted Price ({prediction_days} days)",
                        value=f"${final_predicted_price:.2f}",
                        delta=f"{potential_return:.1f}%"
                    )

                with col2:
                    st.subheader("Return Analysis")
                    st.metric(
                        label="Investment Amount",
                        value=f"${investment_amount:,.2f}"
                    )
                    
                    value_difference = predicted_value - investment_amount
                    st.metric(
                        label="Predicted Value",
                        value=f"${predicted_value:,.2f}",
                        delta=f"{'-' if value_difference < 0 else ''}${abs(value_difference):,.2f}",
                        delta_color="normal"
                    )

                # Create price prediction chart
                dates = pd.date_range(
                    start=df.index[-1],
                    periods=prediction_days + 1,
                    freq='D'
                )

                fig = go.Figure()

                # Add historical prices
                fig.add_trace(go.Scatter(
                    x=df.index[-30:],
                    y=df['Close'].tail(30),
                    name='Historical Price',
                    line=dict(color='blue')
                ))

                # Add predicted prices
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=[current_price] + future_prices,
                    name='Predicted Price',
                    line=dict(color='red', dash='dash')
                ))

                fig.update_layout(
                    title=f"{selected_company} Stock Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Risk assessment
                st.subheader("Risk Assessment")
                risk_level = "High" if abs(potential_return) > 20 else "Medium" if abs(potential_return) > 10 else "Low"

                st.info(
                    f"""
                    Risk Level: {risk_level}
                    - Predicted Return: {potential_return:.1f}%
                    - Time Period: {prediction_days} days
                    - Stock Volatility: {df['Volatility'].tail(20).mean() * 100:.1f}%

                    Remember: Past performance does not guarantee future results. Always diversify your investments.
                    """
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
