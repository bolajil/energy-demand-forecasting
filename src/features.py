"""
Step 2: Feature Engineering
============================
This module creates features for time series forecasting including
lag features, rolling statistics, calendar features, and Fourier terms.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def add_lag_features(df: pd.DataFrame, target_col: str = 'demand_mw',
                     lags: List[int] = [1, 24, 48, 168]) -> pd.DataFrame:
    """
    Add lag features to capture temporal dependencies.
    
    Args:
        df: Input DataFrame with datetime index
        target_col: Column to create lags for
        lags: List of lag periods (hours)
            - 1: Previous hour
            - 24: Same hour yesterday
            - 48: Same hour 2 days ago
            - 168: Same hour last week
    
    Returns:
        DataFrame with lag features added
    """
    df_features = df.copy()
    
    for lag in lags:
        df_features[f'lag_{lag}h'] = df_features[target_col].shift(lag)
    
    return df_features


def add_rolling_features(df: pd.DataFrame, target_col: str = 'demand_mw',
                         windows: List[int] = [6, 12, 24, 168]) -> pd.DataFrame:
    """
    Add rolling window statistics.
    
    Args:
        df: Input DataFrame
        target_col: Column to compute rolling stats for
        windows: List of window sizes (hours)
    
    Returns:
        DataFrame with rolling features added
    """
    df_features = df.copy()
    
    for window in windows:
        # Rolling mean
        df_features[f'rolling_mean_{window}h'] = (
            df_features[target_col]
            .shift(1)  # Shift to avoid data leakage
            .rolling(window=window, min_periods=1)
            .mean()
        )
        
        # Rolling standard deviation
        df_features[f'rolling_std_{window}h'] = (
            df_features[target_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
        )
        
        # Rolling min/max
        df_features[f'rolling_min_{window}h'] = (
            df_features[target_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .min()
        )
        
        df_features[f'rolling_max_{window}h'] = (
            df_features[target_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .max()
        )
    
    return df_features


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features to capture seasonal patterns.
    
    Args:
        df: Input DataFrame with datetime index
    
    Returns:
        DataFrame with calendar features added
    """
    df_features = df.copy()
    
    # Basic calendar features
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['day_of_month'] = df_features.index.day
    df_features['day_of_year'] = df_features.index.dayofyear
    df_features['week_of_year'] = df_features.index.isocalendar().week.astype(int)
    df_features['month'] = df_features.index.month
    df_features['quarter'] = df_features.index.quarter
    
    # Binary features
    df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)
    df_features['is_month_start'] = df_features.index.is_month_start.astype(int)
    df_features['is_month_end'] = df_features.index.is_month_end.astype(int)
    
    # Time of day categories
    df_features['time_of_day'] = pd.cut(
        df_features['hour'],
        bins=[-1, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening']
    )
    
    return df_features


def add_fourier_features(df: pd.DataFrame, periods: List[int] = [24, 168, 8760],
                         n_terms: int = 2) -> pd.DataFrame:
    """
    Add Fourier terms to capture cyclical patterns.
    
    Args:
        df: Input DataFrame with datetime index
        periods: Cycle periods in hours
            - 24: Daily cycle
            - 168: Weekly cycle (24*7)
            - 8760: Yearly cycle (24*365)
        n_terms: Number of Fourier terms per period
    
    Returns:
        DataFrame with Fourier features added
    """
    df_features = df.copy()
    
    # Create hour of year for continuous time
    hour_of_year = (
        (df_features.index.dayofyear - 1) * 24 + df_features.index.hour
    )
    
    for period in periods:
        for k in range(1, n_terms + 1):
            period_name = {24: 'daily', 168: 'weekly', 8760: 'yearly'}.get(period, f'{period}h')
            
            # Sine term
            df_features[f'sin_{period_name}_{k}'] = np.sin(
                2 * np.pi * k * hour_of_year / period
            )
            
            # Cosine term
            df_features[f'cos_{period_name}_{k}'] = np.cos(
                2 * np.pi * k * hour_of_year / period
            )
    
    return df_features


def add_holiday_features(df: pd.DataFrame, 
                         holidays: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add holiday indicators.
    
    Args:
        df: Input DataFrame with datetime index
        holidays: List of holiday dates in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with holiday features added
    """
    df_features = df.copy()
    
    # Default US holidays (simplified)
    if holidays is None:
        holidays = [
            '2022-01-01', '2022-01-17', '2022-02-21', '2022-05-30',
            '2022-07-04', '2022-09-05', '2022-10-10', '2022-11-11',
            '2022-11-24', '2022-12-25', '2022-12-26',
            '2023-01-01', '2023-01-16', '2023-02-20', '2023-05-29',
            '2023-07-04', '2023-09-04', '2023-10-09', '2023-11-10',
            '2023-11-23', '2023-12-25', '2023-12-26'
        ]
    
    holiday_dates = pd.to_datetime(holidays).date
    df_features['is_holiday'] = df_features.index.date.isin(holiday_dates).astype(int)
    
    # Day before/after holiday
    holiday_set = set(holiday_dates)
    df_features['is_day_before_holiday'] = (
        (df_features.index + pd.Timedelta(days=1)).date.isin(holiday_set)
    ).astype(int)
    df_features['is_day_after_holiday'] = (
        (df_features.index - pd.Timedelta(days=1)).date.isin(holiday_set)
    ).astype(int)
    
    return df_features


def add_weather_features(df: pd.DataFrame, 
                         temp_col: str = 'temperature_c') -> pd.DataFrame:
    """
    Add weather-derived features.
    
    Args:
        df: Input DataFrame with temperature column
        temp_col: Name of temperature column
    
    Returns:
        DataFrame with weather features added
    """
    df_features = df.copy()
    
    if temp_col in df_features.columns:
        # Heating/cooling degree days (base 18°C)
        df_features['heating_degree'] = np.maximum(18 - df_features[temp_col], 0)
        df_features['cooling_degree'] = np.maximum(df_features[temp_col] - 18, 0)
        
        # Temperature categories
        df_features['temp_category'] = pd.cut(
            df_features[temp_col],
            bins=[-np.inf, 0, 10, 20, 30, np.inf],
            labels=['freezing', 'cold', 'mild', 'warm', 'hot']
        )
        
        # Temperature change from previous hour
        df_features['temp_change_1h'] = df_features[temp_col].diff(1)
        
        # Temperature anomaly (deviation from daily mean)
        daily_mean_temp = df_features[temp_col].resample('D').transform('mean')
        df_features['temp_anomaly'] = df_features[temp_col] - daily_mean_temp
    
    return df_features


def create_all_features(df: pd.DataFrame, 
                        target_col: str = 'demand_mw') -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
    
    Returns:
        DataFrame with all features
    """
    print("Adding lag features...")
    df = add_lag_features(df, target_col)
    
    print("Adding rolling features...")
    df = add_rolling_features(df, target_col)
    
    print("Adding calendar features...")
    df = add_calendar_features(df)
    
    print("Adding Fourier features...")
    df = add_fourier_features(df)
    
    print("Adding holiday features...")
    df = add_holiday_features(df)
    
    print("Adding weather features...")
    df = add_weather_features(df)
    
    # Drop rows with NaN from lag/rolling features
    initial_len = len(df)
    df = df.dropna()
    print(f"Dropped {initial_len - len(df)} rows with NaN values")
    
    print(f"Total features created: {len(df.columns)}")
    
    return df


def get_feature_columns(df: pd.DataFrame, 
                        exclude: List[str] = ['demand_mw', 'time_of_day', 'temp_category']) -> List[str]:
    """
    Get list of feature columns for modeling.
    
    Args:
        df: DataFrame with features
        exclude: Columns to exclude
    
    Returns:
        List of feature column names
    """
    feature_cols = [col for col in df.columns 
                    if col not in exclude 
                    and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    return feature_cols


# Example usage
if __name__ == "__main__":
    from data_prep import generate_sample_data
    
    # Generate sample data
    df = generate_sample_data()
    print(f"Original shape: {df.shape}")
    
    # Create all features
    df_features = create_all_features(df)
    print(f"Shape with features: {df_features.shape}")
    
    # Get feature columns
    feature_cols = get_feature_columns(df_features)
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols[:10]:
        print(f"  - {col}")
    print("  ...")
