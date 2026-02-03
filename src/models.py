"""
Step 3: Model Training
=======================
This module implements multiple forecasting models:
- ARIMA/SARIMA (Statistical baseline)
- Prophet (Facebook's forecasting tool)
- LSTM (Deep learning approach)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """
    ARIMA/SARIMA model for time series forecasting.
    
    ARIMA = AutoRegressive Integrated Moving Average
    - AR (p): Uses past values to predict future
    - I (d): Differencing to make series stationary
    - MA (q): Uses past forecast errors
    
    SARIMA adds seasonal components (P, D, Q, s)
    """
    
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)):
        """
        Args:
            order: (p, d, q) - AR order, differencing, MA order
            seasonal_order: (P, D, Q, s) - seasonal components
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
    
    def fit(self, train_data: pd.Series) -> 'ARIMAForecaster':
        """
        Fit SARIMA model to training data.
        
        Args:
            train_data: Time series with datetime index
        
        Returns:
            Self for chaining
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        print(f"Fitting SARIMA{self.order}x{self.seasonal_order}...")
        
        self.model = SARIMAX(
            train_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.fitted_model = self.model.fit(disp=False)
        print(f"SARIMA AIC: {self.fitted_model.aic:.2f}")
        
        return self
    
    def predict(self, steps: int) -> pd.Series:
        """
        Generate forecasts for specified number of steps.
        
        Args:
            steps: Number of hours to forecast
        
        Returns:
            Series with predicted values
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return {
            'model': 'SARIMA',
            'order': self.order,
            'seasonal_order': self.seasonal_order
        }


class ProphetForecaster:
    """
    Facebook Prophet model for time series forecasting.
    
    Prophet handles:
    - Trend changes (linear or logistic growth)
    - Yearly, weekly, daily seasonality
    - Holiday effects
    - Missing data and outliers
    """
    
    def __init__(self, yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = True,
                 changepoint_prior_scale: float = 0.05):
        """
        Args:
            yearly_seasonality: Include yearly patterns
            weekly_seasonality: Include weekly patterns
            daily_seasonality: Include daily patterns
            changepoint_prior_scale: Flexibility of trend (higher = more flexible)
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
    
    def fit(self, train_data: pd.DataFrame, 
            target_col: str = 'demand_mw') -> 'ProphetForecaster':
        """
        Fit Prophet model to training data.
        
        Args:
            train_data: DataFrame with datetime index
            target_col: Column to forecast
        
        Returns:
            Self for chaining
        """
        from prophet import Prophet
        
        print("Fitting Prophet model...")
        
        # Prophet requires specific column names
        df_prophet = train_data.reset_index()
        df_prophet = df_prophet.rename(columns={
            df_prophet.columns[0]: 'ds',  # datetime column
            target_col: 'y'
        })[['ds', 'y']]
        
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        # Add temperature as regressor if available
        if 'temperature_c' in train_data.columns:
            df_prophet['temperature'] = train_data['temperature_c'].values
            self.model.add_regressor('temperature')
        
        self.model.fit(df_prophet)
        print("Prophet model fitted successfully")
        
        return self
    
    def predict(self, periods: int, 
                future_temps: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate forecasts for specified periods.
        
        Args:
            periods: Number of hours to forecast
            future_temps: Future temperature values (if using temp regressor)
        
        Returns:
            DataFrame with predictions and uncertainty intervals
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        future = self.model.make_future_dataframe(periods=periods, freq='H')
        
        if future_temps is not None:
            future['temperature'] = future_temps
        
        forecast = self.model.predict(future)
        
        # Return only the forecast periods
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return {
            'model': 'Prophet',
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'changepoint_prior_scale': self.changepoint_prior_scale
        }


class LSTMForecaster:
    """
    LSTM (Long Short-Term Memory) neural network for time series.
    
    LSTM advantages:
    - Captures long-term dependencies
    - Handles non-linear patterns
    - Can use multiple input features
    """
    
    def __init__(self, lookback: int = 168, 
                 lstm_units: int = 64,
                 epochs: int = 50,
                 batch_size: int = 32):
        """
        Args:
            lookback: Number of past hours to use as input (168 = 1 week)
            lstm_units: Number of LSTM units
            epochs: Training epochs
            batch_size: Batch size for training
        """
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences for LSTM."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            y.append(data[i, 0])  # Predict first column (demand)
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM architecture."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, 
                 input_shape=input_shape),
            Dropout(0.2),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, train_data: pd.DataFrame,
            feature_cols: list,
            target_col: str = 'demand_mw') -> 'LSTMForecaster':
        """
        Fit LSTM model to training data.
        
        Args:
            train_data: DataFrame with features
            feature_cols: List of feature column names
            target_col: Column to forecast
        
        Returns:
            Self for chaining
        """
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.callbacks import EarlyStopping
        
        print("Preparing data for LSTM...")
        
        # Prepare features (target should be first column)
        cols = [target_col] + [c for c in feature_cols if c != target_col]
        data = train_data[cols].values
        
        # Scale data
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self._create_sequences(data_scaled)
        
        # Split for validation
        split = int(len(X) * 0.9)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        print(f"Training LSTM with {len(X_train)} samples...")
        
        # Build and train model
        self.model = self._build_model((self.lookback, len(cols)))
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        print(f"LSTM training complete. Final val_loss: {history.history['val_loss'][-1]:.4f}")
        
        return self
    
    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        """
        Generate predictions from input sequence.
        
        Args:
            input_sequence: Array of shape (lookback, n_features)
        
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Scale input
        input_scaled = self.scaler.transform(input_sequence)
        
        # Reshape for prediction
        input_reshaped = input_scaled.reshape(1, self.lookback, -1)
        
        # Predict (returns scaled value)
        pred_scaled = self.model.predict(input_reshaped, verbose=0)
        
        # Inverse scale (only for target column)
        dummy = np.zeros((1, input_sequence.shape[1]))
        dummy[0, 0] = pred_scaled[0, 0]
        pred = self.scaler.inverse_transform(dummy)[0, 0]
        
        return pred
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return {
            'model': 'LSTM',
            'lookback': self.lookback,
            'lstm_units': self.lstm_units,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }


def train_all_models(train_data: pd.DataFrame,
                     feature_cols: list,
                     target_col: str = 'demand_mw') -> Dict[str, Any]:
    """
    Train all forecasting models.
    
    Args:
        train_data: Training DataFrame
        feature_cols: Feature columns for LSTM
        target_col: Target column name
    
    Returns:
        Dictionary with trained models
    """
    models = {}
    
    # Train ARIMA (uses only target series)
    print("\n" + "="*50)
    print("Training ARIMA Model")
    print("="*50)
    arima = ARIMAForecaster(order=(2, 1, 2), seasonal_order=(1, 0, 1, 24))
    arima.fit(train_data[target_col])
    models['arima'] = arima
    
    # Train Prophet
    print("\n" + "="*50)
    print("Training Prophet Model")
    print("="*50)
    prophet = ProphetForecaster()
    prophet.fit(train_data, target_col)
    models['prophet'] = prophet
    
    # Train LSTM
    print("\n" + "="*50)
    print("Training LSTM Model")
    print("="*50)
    lstm = LSTMForecaster(lookback=168, lstm_units=64, epochs=20)
    lstm.fit(train_data, feature_cols, target_col)
    models['lstm'] = lstm
    
    print("\n" + "="*50)
    print("All models trained successfully!")
    print("="*50)
    
    return models


# Example usage
if __name__ == "__main__":
    from data_prep import generate_sample_data, create_train_test_split
    from features import create_all_features, get_feature_columns
    
    # Generate and prepare data
    df = generate_sample_data(periods=2000)  # Smaller for demo
    df = create_all_features(df)
    train_df, test_df = create_train_test_split(df, test_days=7)
    
    feature_cols = get_feature_columns(train_df)
    
    # Train models
    models = train_all_models(train_df, feature_cols)
    
    # Print model info
    for name, model in models.items():
        print(f"\n{name}: {model.get_params()}")
