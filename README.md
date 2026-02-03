# Energy Demand Forecasting with GenAI Integration

A complete step-by-step project demonstrating time series forecasting for electricity demand, enhanced with GenAI-powered insights generation.

---

## Project Overview

| Aspect | Details |
|--------|---------|
| **Problem** | Predict 24-hour ahead electricity demand to optimize power generation |
| **ML Techniques** | ARIMA, Prophet, LSTM, Ensemble |
| **GenAI Integration** | LLM-powered forecast interpretation and anomaly explanations |
| **Tech Stack** | Python, Pandas, Prophet, TensorFlow, OpenAI API |

---

## Project Structure

```
energy-demand-forecasting/
├── README.md                 # This guide
├── requirements.txt          # Dependencies
├── data/
│   └── sample_energy_data.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_genai_integration.ipynb
├── src/
│   ├── data_prep.py          # Data loading and preprocessing
│   ├── features.py           # Feature engineering functions
│   ├── models.py             # Model training classes
│   ├── ensemble.py           # Ensemble predictions
│   └── genai_insights.py     # GenAI integration
└── app/
    └── forecast_app.py       # Streamlit dashboard
```

---

## Step-by-Step Implementation Guide

### Step 1: Data Preparation & EDA
**Goal**: Understand the data, identify patterns, handle missing values

Key tasks:
- Load hourly energy demand data
- Visualize trends, seasonality, and anomalies
- Check for missing values and outliers
- Analyze correlations with weather data

→ See `src/data_prep.py` and `notebooks/01_data_exploration.ipynb`

---

### Step 2: Feature Engineering
**Goal**: Create features that capture temporal patterns

Key features:
- **Lag features**: demand at t-1, t-24, t-168 (1 hour, 1 day, 1 week ago)
- **Rolling statistics**: 24-hour rolling mean and std
- **Calendar features**: hour, day of week, month, is_weekend, is_holiday
- **Fourier terms**: sin/cos transformations for cyclical patterns
- **Weather features**: temperature, humidity (if available)

→ See `src/features.py` and `notebooks/02_feature_engineering.ipynb`

---

### Step 3: Model Training
**Goal**: Train multiple models for comparison

Models implemented:
1. **ARIMA/SARIMA**: Statistical baseline for time series
2. **Prophet**: Facebook's forecasting tool with seasonality handling
3. **LSTM**: Deep learning for sequence patterns

→ See `src/models.py` and `notebooks/03_model_training.ipynb`

---

### Step 4: Model Evaluation & Ensemble
**Goal**: Compare models and create ensemble for best performance

Evaluation metrics:
- **MAPE**: Mean Absolute Percentage Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error

Ensemble approach:
- Weighted average based on validation performance
- Dynamic weight adjustment based on recent accuracy

→ See `src/ensemble.py` and `notebooks/04_model_evaluation.ipynb`

---

### Step 5: GenAI Integration
**Goal**: Use LLM to generate human-readable forecast insights

GenAI capabilities:
1. **Forecast Interpretation**: Convert numbers to business insights
2. **Anomaly Explanation**: Explain unusual demand patterns
3. **Recommendation Generation**: Suggest operational actions
4. **Natural Language Queries**: Ask questions about forecasts

→ See `src/genai_insights.py` and `notebooks/05_genai_integration.ipynb`

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set OpenAI API key (for GenAI features)
set OPENAI_API_KEY=your-api-key-here

# 3. Run the Streamlit app
streamlit run app/forecast_app.py
```

---

## GenAI Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER QUERY / FORECAST DATA                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FORECASTING PIPELINE                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                         │
│  │ ARIMA   │  │ Prophet │  │  LSTM   │ → Ensemble Prediction   │
│  └─────────┘  └─────────┘  └─────────┘                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GENAI INSIGHT ENGINE                         │
│                                                                 │
│  Forecast Data + Context  ──►  LLM (GPT-4)  ──►  Insights       │
│                                                                 │
│  • "Tomorrow's peak demand expected at 3PM (15,200 MW)"         │
│  • "Unusual spike detected - likely due to heat wave"          │
│  • "Recommend: Start additional turbine by 2PM"                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Sample GenAI Prompts Used

### Forecast Interpretation Prompt
```
You are an energy analyst AI. Given the following forecast data:
- Current demand: {current} MW
- Predicted demand (next 24h): {predictions}
- Historical average for this period: {historical_avg} MW
- Weather forecast: {weather}

Provide a concise business summary including:
1. Key demand trends for the next 24 hours
2. Peak demand time and magnitude
3. Comparison to historical patterns
4. Any anomalies or concerns
```

### Anomaly Explanation Prompt
```
An unusual energy demand pattern was detected:
- Expected: {expected} MW
- Actual: {actual} MW  
- Deviation: {deviation}%
- Time: {timestamp}
- Weather: {weather}
- Day type: {day_type}

Explain the most likely causes for this anomaly and recommend actions.
```

---

## Results & Metrics

| Model | MAPE | RMSE | MAE |
|-------|------|------|-----|
| ARIMA | 4.8% | 342 MW | 287 MW |
| Prophet | 4.2% | 298 MW | 251 MW |
| LSTM | 3.9% | 276 MW | 234 MW |
| **Ensemble** | **3.2%** | **241 MW** | **198 MW** |

---

## Next Steps

- [ ] Add real-time data ingestion
- [ ] Implement drift detection for model retraining
- [ ] Deploy to cloud with scheduled predictions
- [ ] Integrate with operational systems via API
# energy-demand-forecasting
