"""
Step 5: GenAI Integration for Forecast Insights
=================================================
This module integrates Large Language Models (LLMs) to generate
human-readable insights from forecast data.

Features:
- Forecast interpretation in natural language
- Anomaly explanation
- Operational recommendations
- Natural language Q&A about forecasts
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd


class GenAIInsightEngine:
    """
    Engine for generating AI-powered insights from forecast data.
    
    Uses OpenAI API (GPT-4) for natural language generation.
    Can be adapted for other LLM providers (Anthropic, local models, etc.)
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if self.api_key:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            print(f"GenAI Engine initialized with model: {self.model}")
        except ImportError:
            print("OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
    
    def _call_llm(self, system_prompt: str, user_prompt: str, 
                  temperature: float = 0.7) -> str:
        """
        Make API call to LLM.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query/data
            temperature: Creativity level (0-1)
        
        Returns:
            Generated text response
        """
        if not self.client:
            return self._generate_fallback_response(user_prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API error: {e}")
            return self._generate_fallback_response(user_prompt)
    
    def _generate_fallback_response(self, context: str) -> str:
        """Generate rule-based response when API unavailable."""
        return "GenAI insight generation requires API key. Set OPENAI_API_KEY environment variable."
    
    def interpret_forecast(self, forecast_data: Dict[str, Any]) -> str:
        """
        Generate natural language interpretation of forecast.
        
        Args:
            forecast_data: Dictionary containing:
                - current_demand: Current demand in MW
                - predictions: List of predicted values (24h)
                - historical_avg: Historical average for this period
                - peak_time: Time of predicted peak
                - peak_value: Predicted peak demand
        
        Returns:
            Natural language forecast summary
        """
        system_prompt = """You are an expert energy analyst AI. Your role is to interpret 
electricity demand forecasts and provide clear, actionable insights for grid operators.

Be concise but thorough. Use specific numbers. Highlight any concerns."""

        user_prompt = f"""Analyze this energy demand forecast and provide a business summary:

Current Demand: {forecast_data.get('current_demand', 'N/A')} MW
Predicted Next 24 Hours: {forecast_data.get('predictions', [])}
Historical Average: {forecast_data.get('historical_avg', 'N/A')} MW
Predicted Peak Time: {forecast_data.get('peak_time', 'N/A')}
Predicted Peak Value: {forecast_data.get('peak_value', 'N/A')} MW
Weather Forecast: {forecast_data.get('weather', 'N/A')}

Provide:
1. Executive summary (2-3 sentences)
2. Key demand trends for next 24 hours
3. Peak demand analysis
4. Any anomalies or concerns
5. Operational recommendations"""

        return self._call_llm(system_prompt, user_prompt)
    
    def explain_anomaly(self, anomaly_data: Dict[str, Any]) -> str:
        """
        Generate explanation for detected demand anomaly.
        
        Args:
            anomaly_data: Dictionary containing:
                - expected: Expected demand
                - actual: Actual demand
                - deviation_pct: Percentage deviation
                - timestamp: When anomaly occurred
                - weather: Weather conditions
                - day_type: Weekday/weekend/holiday
        
        Returns:
            Natural language anomaly explanation
        """
        system_prompt = """You are an expert energy analyst AI specializing in anomaly detection.
Your role is to explain unusual demand patterns and suggest causes.

Consider: weather events, holidays, equipment issues, industrial activity, 
special events, data quality issues."""

        user_prompt = f"""An unusual energy demand pattern was detected:

Expected Demand: {anomaly_data.get('expected', 'N/A')} MW
Actual Demand: {anomaly_data.get('actual', 'N/A')} MW
Deviation: {anomaly_data.get('deviation_pct', 'N/A')}%
Time: {anomaly_data.get('timestamp', 'N/A')}
Weather: {anomaly_data.get('weather', 'N/A')}
Day Type: {anomaly_data.get('day_type', 'N/A')}

Explain:
1. Most likely cause(s) for this anomaly
2. Confidence level in explanation
3. Recommended actions
4. Whether this should trigger an alert"""

        return self._call_llm(system_prompt, user_prompt, temperature=0.5)
    
    def generate_recommendations(self, forecast_data: Dict[str, Any]) -> str:
        """
        Generate operational recommendations based on forecast.
        
        Args:
            forecast_data: Forecast data dictionary
        
        Returns:
            Operational recommendations
        """
        system_prompt = """You are an expert power grid operator AI. 
Generate specific, actionable recommendations for grid operations based on demand forecasts.

Focus on: generation scheduling, reserve margins, maintenance timing, 
demand response programs, and cost optimization."""

        user_prompt = f"""Based on this demand forecast, provide operational recommendations:

Forecast Summary:
- Peak Demand: {forecast_data.get('peak_value', 'N/A')} MW at {forecast_data.get('peak_time', 'N/A')}
- Minimum Demand: {forecast_data.get('min_value', 'N/A')} MW
- Average Demand: {forecast_data.get('avg_value', 'N/A')} MW
- Forecast MAPE: {forecast_data.get('model_mape', 'N/A')}%
- Weather: {forecast_data.get('weather', 'N/A')}

Current Status:
- Available Capacity: {forecast_data.get('capacity', 'N/A')} MW
- Reserve Margin: {forecast_data.get('reserve', 'N/A')}%

Provide 3-5 specific recommendations with priority levels."""

        return self._call_llm(system_prompt, user_prompt)
    
    def answer_query(self, query: str, context: Dict[str, Any]) -> str:
        """
        Answer natural language questions about forecasts.
        
        Args:
            query: User's question
            context: Forecast context data
        
        Returns:
            Answer to the query
        """
        system_prompt = """You are an AI assistant for energy demand forecasting.
Answer questions about forecasts clearly and accurately.
If you don't have enough data to answer, say so.
Use specific numbers from the provided context."""

        context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
        
        user_prompt = f"""Context (Forecast Data):
{context_str}

User Question: {query}

Provide a clear, specific answer."""

        return self._call_llm(system_prompt, user_prompt)


def create_forecast_summary(predictions: np.ndarray, 
                            timestamps: pd.DatetimeIndex,
                            actuals: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Create forecast summary for GenAI input.
    
    Args:
        predictions: Array of predicted values
        timestamps: DatetimeIndex for predictions
        actuals: Actual values (if available)
    
    Returns:
        Summary dictionary for GenAI
    """
    peak_idx = np.argmax(predictions)
    min_idx = np.argmin(predictions)
    
    summary = {
        'current_demand': round(predictions[0], 1),
        'predictions': [round(p, 1) for p in predictions[:24]],  # First 24 hours
        'peak_time': timestamps[peak_idx].strftime('%Y-%m-%d %H:%M'),
        'peak_value': round(predictions[peak_idx], 1),
        'min_time': timestamps[min_idx].strftime('%Y-%m-%d %H:%M'),
        'min_value': round(predictions[min_idx], 1),
        'avg_value': round(np.mean(predictions), 1),
        'std_value': round(np.std(predictions), 1)
    }
    
    if actuals is not None:
        from ensemble import calculate_mape
        summary['model_mape'] = round(calculate_mape(actuals, predictions), 2)
        summary['historical_avg'] = round(np.mean(actuals), 1)
    
    return summary


def detect_anomalies_for_genai(actuals: np.ndarray, 
                               predictions: np.ndarray,
                               timestamps: pd.DatetimeIndex,
                               threshold: float = 10.0) -> List[Dict[str, Any]]:
    """
    Detect anomalies and format for GenAI explanation.
    
    Args:
        actuals: Actual demand values
        predictions: Predicted values
        timestamps: DatetimeIndex
        threshold: Percentage threshold for anomaly
    
    Returns:
        List of anomaly dictionaries
    """
    anomalies = []
    
    errors = np.abs((actuals - predictions) / actuals) * 100
    
    for i, error in enumerate(errors):
        if error > threshold:
            anomaly = {
                'expected': round(predictions[i], 1),
                'actual': round(actuals[i], 1),
                'deviation_pct': round(error, 1),
                'timestamp': timestamps[i].strftime('%Y-%m-%d %H:%M'),
                'day_type': 'Weekend' if timestamps[i].dayofweek >= 5 else 'Weekday'
            }
            anomalies.append(anomaly)
    
    return anomalies


# Demo function showing GenAI integration flow
def demo_genai_integration():
    """
    Demonstrate the GenAI integration pipeline.
    """
    print("="*60)
    print("GenAI Integration Demo")
    print("="*60)
    
    # Initialize engine (will use fallback if no API key)
    engine = GenAIInsightEngine()
    
    # Sample forecast data
    forecast_data = {
        'current_demand': 12500,
        'predictions': [12500, 12800, 13200, 14100, 14800, 15200, 15100, 14600,
                       14200, 13800, 13500, 13200, 13000, 12800, 13100, 13500,
                       14200, 14800, 15000, 14500, 13800, 13200, 12800, 12400],
        'peak_time': '2024-01-15 17:00',
        'peak_value': 15200,
        'min_value': 12400,
        'avg_value': 13725,
        'historical_avg': 13500,
        'weather': 'Cold front expected, temperatures 5°C below normal',
        'model_mape': 3.2,
        'capacity': 18000,
        'reserve': 18.4
    }
    
    print("\n1. FORECAST INTERPRETATION")
    print("-" * 40)
    interpretation = engine.interpret_forecast(forecast_data)
    print(interpretation)
    
    # Sample anomaly
    anomaly_data = {
        'expected': 13000,
        'actual': 14800,
        'deviation_pct': 13.8,
        'timestamp': '2024-01-15 14:00',
        'weather': 'Sudden temperature drop to -5°C',
        'day_type': 'Weekday'
    }
    
    print("\n2. ANOMALY EXPLANATION")
    print("-" * 40)
    explanation = engine.explain_anomaly(anomaly_data)
    print(explanation)
    
    print("\n3. OPERATIONAL RECOMMENDATIONS")
    print("-" * 40)
    recommendations = engine.generate_recommendations(forecast_data)
    print(recommendations)
    
    print("\n4. NATURAL LANGUAGE Q&A")
    print("-" * 40)
    query = "What time should we start the backup generator tomorrow?"
    answer = engine.answer_query(query, forecast_data)
    print(f"Q: {query}")
    print(f"A: {answer}")


# Example usage
if __name__ == "__main__":
    demo_genai_integration()
