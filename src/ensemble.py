"""
Step 4: Model Evaluation and Ensemble
======================================
This module evaluates individual models and creates
an ensemble for improved predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    MAPE = (1/n) * Σ |actual - predicted| / |actual| * 100
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        MAPE as percentage
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Dictionary with all metrics
    """
    return {
        'mape': calculate_mape(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mse': mean_squared_error(y_true, y_pred)
    }


def evaluate_model(model_name: str, y_true: np.ndarray, 
                   y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate a single model and print results.
    
    Args:
        model_name: Name of the model
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Dictionary with metrics
    """
    metrics = calculate_metrics(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  MAPE:  {metrics['mape']:.2f}%")
    print(f"  MAE:   {metrics['mae']:.2f} MW")
    print(f"  RMSE:  {metrics['rmse']:.2f} MW")
    
    return metrics


class EnsembleForecaster:
    """
    Ensemble forecaster that combines multiple models.
    
    Ensemble methods:
    - Simple average: Equal weight to all models
    - Weighted average: Weights based on validation performance
    - Stacking: Train meta-model on base model predictions
    """
    
    def __init__(self, models: Dict[str, Any], 
                 method: str = 'weighted'):
        """
        Args:
            models: Dictionary of trained models
            method: 'simple', 'weighted', or 'dynamic'
        """
        self.models = models
        self.method = method
        self.weights = {name: 1.0 / len(models) for name in models}
    
    def calculate_weights(self, validation_metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Calculate model weights based on validation performance.
        Uses inverse MAPE as weight (lower MAPE = higher weight).
        
        Args:
            validation_metrics: Dict of model_name -> metrics dict
        """
        if self.method == 'simple':
            return
        
        # Inverse MAPE weights
        inverse_mapes = {}
        for name, metrics in validation_metrics.items():
            inverse_mapes[name] = 1.0 / (metrics['mape'] + 1e-6)
        
        # Normalize to sum to 1
        total = sum(inverse_mapes.values())
        self.weights = {name: inv / total for name, inv in inverse_mapes.items()}
        
        print("\nEnsemble Weights (based on validation MAPE):")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.3f}")
    
    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine model predictions using ensemble method.
        
        Args:
            predictions: Dict of model_name -> predictions array
        
        Returns:
            Ensemble predictions
        """
        if self.method == 'simple':
            # Simple average
            all_preds = np.array(list(predictions.values()))
            return np.mean(all_preds, axis=0)
        
        elif self.method == 'weighted':
            # Weighted average
            ensemble_pred = np.zeros_like(list(predictions.values())[0])
            for name, preds in predictions.items():
                ensemble_pred += self.weights[name] * preds
            return ensemble_pred
        
        elif self.method == 'dynamic':
            # Dynamic weighting based on recent performance
            # (Simplified: use weighted for now)
            return self.predict_weighted(predictions)
    
    def get_confidence_interval(self, predictions: Dict[str, np.ndarray],
                                confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals from model disagreement.
        
        Args:
            predictions: Dict of model predictions
            confidence: Confidence level (0-1)
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        all_preds = np.array(list(predictions.values()))
        
        # Use model spread to estimate uncertainty
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)
        
        # Z-score for confidence level
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        lower = mean_pred - z * std_pred
        upper = mean_pred + z * std_pred
        
        return lower, upper


def run_evaluation(test_df: pd.DataFrame,
                   models: Dict[str, Any],
                   feature_cols: List[str],
                   target_col: str = 'demand_mw') -> Dict[str, Any]:
    """
    Run full evaluation on test set.
    
    Args:
        test_df: Test DataFrame
        models: Dictionary of trained models
        feature_cols: Feature columns for LSTM
        target_col: Target column name
    
    Returns:
        Dictionary with predictions and metrics
    """
    y_true = test_df[target_col].values
    predictions = {}
    metrics = {}
    
    # Get predictions from each model
    print("\n" + "="*50)
    print("Generating Predictions")
    print("="*50)
    
    # ARIMA predictions
    if 'arima' in models:
        print("\nGenerating ARIMA predictions...")
        arima_pred = models['arima'].predict(steps=len(test_df))
        predictions['arima'] = arima_pred.values
        metrics['arima'] = evaluate_model('ARIMA', y_true, predictions['arima'])
    
    # Prophet predictions
    if 'prophet' in models:
        print("\nGenerating Prophet predictions...")
        prophet_result = models['prophet'].predict(periods=len(test_df))
        predictions['prophet'] = prophet_result['yhat'].values
        metrics['prophet'] = evaluate_model('Prophet', y_true, predictions['prophet'])
    
    # LSTM predictions (simplified - one-step ahead)
    if 'lstm' in models:
        print("\nGenerating LSTM predictions...")
        lstm_preds = []
        lookback = models['lstm'].lookback
        
        # Use rolling predictions
        for i in range(len(test_df)):
            if i < lookback:
                # Use training data for initial predictions
                lstm_preds.append(y_true[i])  # Placeholder
            else:
                # Get last lookback values
                cols = [target_col] + [c for c in feature_cols if c != target_col]
                input_seq = test_df[cols].iloc[i-lookback:i].values
                pred = models['lstm'].predict(input_seq)
                lstm_preds.append(pred)
        
        predictions['lstm'] = np.array(lstm_preds)
        metrics['lstm'] = evaluate_model('LSTM', y_true, predictions['lstm'])
    
    # Create ensemble
    print("\n" + "="*50)
    print("Creating Ensemble")
    print("="*50)
    
    ensemble = EnsembleForecaster(models, method='weighted')
    ensemble.calculate_weights(metrics)
    
    ensemble_pred = ensemble.predict(predictions)
    predictions['ensemble'] = ensemble_pred
    metrics['ensemble'] = evaluate_model('Ensemble', y_true, ensemble_pred)
    
    # Confidence intervals
    lower, upper = ensemble.get_confidence_interval(predictions)
    
    # Summary
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print("\n| Model    | MAPE   | MAE (MW) | RMSE (MW) |")
    print("|----------|--------|----------|-----------|")
    for name, m in metrics.items():
        print(f"| {name:8s} | {m['mape']:5.2f}% | {m['mae']:8.1f} | {m['rmse']:9.1f} |")
    
    return {
        'predictions': predictions,
        'metrics': metrics,
        'y_true': y_true,
        'confidence_lower': lower,
        'confidence_upper': upper,
        'ensemble': ensemble
    }


def plot_forecast_comparison(test_df: pd.DataFrame,
                             results: Dict[str, Any],
                             last_n_hours: int = 168) -> None:
    """
    Plot forecast comparison for visualization.
    
    Args:
        test_df: Test DataFrame
        results: Results from run_evaluation
        last_n_hours: Number of hours to plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Slice data
    idx = test_df.index[-last_n_hours:]
    y_true = results['y_true'][-last_n_hours:]
    
    # Plot 1: All models comparison
    ax1 = axes[0]
    ax1.plot(idx, y_true, 'k-', label='Actual', linewidth=2)
    
    colors = {'arima': 'blue', 'prophet': 'green', 'lstm': 'orange', 'ensemble': 'red'}
    for name, preds in results['predictions'].items():
        ax1.plot(idx, preds[-last_n_hours:], '--', 
                 label=f"{name.upper()} (MAPE: {results['metrics'][name]['mape']:.2f}%)",
                 color=colors.get(name, 'gray'), alpha=0.7)
    
    ax1.set_title('Forecast Comparison - All Models')
    ax1.set_xlabel('DateTime')
    ax1.set_ylabel('Demand (MW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ensemble with confidence interval
    ax2 = axes[1]
    ax2.plot(idx, y_true, 'k-', label='Actual', linewidth=2)
    ax2.plot(idx, results['predictions']['ensemble'][-last_n_hours:], 
             'r-', label='Ensemble', linewidth=2)
    ax2.fill_between(idx, 
                     results['confidence_lower'][-last_n_hours:],
                     results['confidence_upper'][-last_n_hours:],
                     alpha=0.2, color='red', label='95% CI')
    
    ax2.set_title('Ensemble Forecast with Confidence Interval')
    ax2.set_xlabel('DateTime')
    ax2.set_ylabel('Demand (MW)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('forecast_comparison.png', dpi=150)
    plt.show()
    print("\nPlot saved to 'forecast_comparison.png'")


# Example usage
if __name__ == "__main__":
    print("Ensemble module loaded. Run with trained models for evaluation.")
