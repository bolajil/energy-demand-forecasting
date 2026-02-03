"""
Step 1: Data Preparation and Loading
=====================================
This module handles data loading, cleaning, and initial preprocessing
for the energy demand forecasting project.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_data(start_date: str = "2022-01-01", periods: int = 8760) -> pd.DataFrame:
    """
    Generate synthetic energy demand data for demonstration.
    In production, replace this with actual data loading.
    
    Args:
        start_date: Start date for the time series
        periods: Number of hourly observations (8760 = 1 year)
    
    Returns:
        DataFrame with datetime index and demand/weather columns
    """
    np.random.seed(42)
    
    # Create datetime index
    date_range = pd.date_range(start=start_date, periods=periods, freq='H')
    
    # Base demand pattern (MW)
    base_demand = 10000
    
    # Daily pattern: higher during day, lower at night
    hour_pattern = np.array([
        0.7, 0.65, 0.6, 0.58, 0.6, 0.7,    # 00:00 - 05:00
        0.85, 1.0, 1.1, 1.15, 1.1, 1.05,   # 06:00 - 11:00
        1.0, 1.05, 1.1, 1.15, 1.2, 1.15,   # 12:00 - 17:00
        1.1, 1.0, 0.95, 0.9, 0.85, 0.75    # 18:00 - 23:00
    ])
    
    # Weekly pattern: lower on weekends
    day_pattern = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.8])  # Mon-Sun
    
    # Monthly/seasonal pattern
    month_pattern = np.array([
        1.1, 1.05, 0.95, 0.85, 0.9, 1.05,  # Jan-Jun (heating, then mild)
        1.15, 1.2, 1.0, 0.9, 0.95, 1.1     # Jul-Dec (cooling, then heating)
    ])
    
    # Generate demand
    demand = []
    temperatures = []
    
    for dt in date_range:
        hour_factor = hour_pattern[dt.hour]
        day_factor = day_pattern[dt.dayofweek]
        month_factor = month_pattern[dt.month - 1]
        
        # Temperature simulation (affects demand)
        base_temp = 15 + 10 * np.sin(2 * np.pi * (dt.dayofyear - 80) / 365)  # Seasonal
        temp = base_temp + np.random.normal(0, 3)  # Daily variation
        temperatures.append(temp)
        
        # Temperature effect on demand (heating/cooling)
        temp_effect = 1.0 + 0.02 * abs(temp - 18)  # Deviation from comfortable 18°C
        
        # Calculate demand with noise
        d = base_demand * hour_factor * day_factor * month_factor * temp_effect
        d += np.random.normal(0, d * 0.03)  # 3% random noise
        demand.append(d)
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': date_range,
        'demand_mw': demand,
        'temperature_c': temperatures,
        'hour': date_range.hour,
        'day_of_week': date_range.dayofweek,
        'month': date_range.month,
        'is_weekend': date_range.dayofweek >= 5
    })
    
    df.set_index('datetime', inplace=True)
    
    return df


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load energy demand data from CSV or generate sample data.
    
    Args:
        filepath: Path to CSV file (optional)
    
    Returns:
        Preprocessed DataFrame
    """
    if filepath:
        df = pd.read_csv(filepath, parse_dates=['datetime'], index_col='datetime')
    else:
        print("No data file provided. Generating sample data...")
        df = generate_sample_data()
    
    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Perform data quality checks.
    
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'date_range': f"{df.index.min()} to {df.index.max()}",
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': df.index.duplicated().sum(),
        'demand_stats': {
            'min': df['demand_mw'].min(),
            'max': df['demand_mw'].max(),
            'mean': df['demand_mw'].mean(),
            'std': df['demand_mw'].std()
        }
    }
    
    return quality_report


def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        method: 'interpolate', 'forward_fill', or 'mean'
    
    Returns:
        DataFrame with missing values handled
    """
    df_clean = df.copy()
    
    if method == 'interpolate':
        df_clean = df_clean.interpolate(method='time')
    elif method == 'forward_fill':
        df_clean = df_clean.fillna(method='ffill')
    elif method == 'mean':
        df_clean = df_clean.fillna(df_clean.mean())
    
    # Fill any remaining NaN at edges
    df_clean = df_clean.bfill().ffill()
    
    return df_clean


def detect_outliers(df: pd.DataFrame, column: str = 'demand_mw', 
                    method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in the demand data.
    
    Args:
        df: Input DataFrame
        column: Column to check for outliers
        method: 'iqr' or 'zscore'
        threshold: IQR multiplier or z-score threshold
    
    Returns:
        Boolean Series marking outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        outliers = abs(z_scores) > threshold
    
    return outliers


def create_train_test_split(df: pd.DataFrame, test_days: int = 30) -> tuple:
    """
    Split data into training and test sets (time-based split).
    
    Args:
        df: Input DataFrame
        test_days: Number of days for test set
    
    Returns:
        Tuple of (train_df, test_df)
    """
    split_date = df.index.max() - timedelta(days=test_days)
    
    train_df = df[df.index <= split_date].copy()
    test_df = df[df.index > split_date].copy()
    
    print(f"Training set: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Test set: {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")
    
    return train_df, test_df


# Example usage
if __name__ == "__main__":
    # Generate and explore sample data
    df = generate_sample_data()
    print("Data Shape:", df.shape)
    print("\nData Quality Report:")
    report = check_data_quality(df)
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # Detect outliers
    outliers = detect_outliers(df)
    print(f"\nOutliers detected: {outliers.sum()} ({outliers.mean()*100:.2f}%)")
    
    # Split data
    train, test = create_train_test_split(df)
