# Data Folder

Place your real energy demand data here.

## Required CSV Format

Your CSV file must have these columns:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `datetime` | datetime | Timestamp (YYYY-MM-DD HH:MM:SS) | ✅ Yes |
| `demand_mw` | float | Energy demand in megawatts | ✅ Yes |
| `temperature_c` | float | Temperature in Celsius | Optional |

## Sample CSV Structure

```csv
datetime,demand_mw,temperature_c
2024-01-01 00:00:00,10234.5,5.2
2024-01-01 01:00:00,9876.3,4.8
2024-01-01 02:00:00,9543.1,4.5
2024-01-01 03:00:00,9234.8,4.2
```

## How to Load Your Data

### Option 1: In Python scripts
```python
from src.data_prep import load_data

# Load real data
df = load_data('data/your_energy_data.csv')

# Or use demo data (no argument)
df = load_data()
```

### Option 2: Update the Streamlit app

Edit `app/forecast_app.py` line ~58, change:
```python
# FROM (demo data):
hist_df, forecast_df = generate_demo_data()

# TO (real data):
from src.data_prep import load_data
df = load_data('data/your_energy_data.csv')
```

## Data Location

```
energy-demand-forecasting/
├── data/
│   ├── README.md          ← You are here
│   ├── sample_template.csv ← Template file
│   └── your_data.csv      ← Put your real data here
```

## Data Sources

If you don't have real data, you can use public datasets:

1. **UCI ML Repository** - Individual Household Electric Power Consumption
   https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

2. **Kaggle** - Hourly Energy Consumption
   https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

3. **ERCOT** (Texas Grid) - Historical Load Data
   https://www.ercot.com/gridinfo/load/load_hist
