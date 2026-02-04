# CLAUDE.md - AI Assistant Guide for Dynamic Withdrawal

This document provides essential context for AI assistants working with this codebase.

## Project Overview

**Dynamic Withdrawal** is a retirement portfolio withdrawal rate backtesting framework. It evaluates sustainable withdrawal strategies using historical rolling window analysis on Korean and US benchmark assets.

**Primary Use Case**: Test how different withdrawal rates (e.g., 4% rule) would have performed historically across various portfolio compositions and time horizons.

## Tech Stack

- **Language**: Python 3.8+
- **Web Framework**: Streamlit (interactive dashboard)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn (CLI), plotly (web)
- **Excel Export**: openpyxl

## Project Structure

```
Dynamic_Withdrawl/
├── withdrawal_backtest.py   # Core backtest engine (4 main classes)
├── app.py                   # Streamlit web application
├── benchmark_data.pkl       # Historical benchmark data (2001-present)
├── benchmark_to_pkl.py      # Data conversion utility
├── requirements.txt         # Python dependencies
├── README.md                # User documentation (Korean/English)
└── CLAUDE.md                # This file
```

## Key Modules

### withdrawal_backtest.py - Core Engine

Four main classes handle the simulation:

1. **`PortfolioCalculator`** - Computes weighted portfolio returns from benchmark indices
   - `calculate_portfolio_returns()` - Weight-averaged daily returns
   - `add_portfolios_to_returns()` - Adds portfolio columns to DataFrame

2. **`DataPreprocessor`** - Data loading and preparation pipeline
   - Validates DatetimeIndex, sorting, nulls
   - Calculates daily returns from price levels
   - Identifies month-start trading days
   - `get_data()` returns `(returns_df, month_starts_series)`

3. **`WithdrawalSimulator`** - Core simulation engine
   - `simulate_single_path()` - Single rolling window simulation
   - `run_rolling_backtest()` - Full backtest across all valid start dates
   - `get_valid_horizons()` - Available horizon years given data

4. **`WithdrawalVisualizer`** - Matplotlib-based visualization
   - `plot_all_paths()` - All paths with success/failure coloring
   - `plot_terminal_values_by_start_date()` - Scatter of final values
   - `plot_failure_rate_by_start_month()` - Monthly stacked bars + trend

### app.py - Streamlit Application

Interactive web interface with:
- Parameter configuration (withdrawal rate, dates, horizon, guardrails)
- Tab-based analysis (individual portfolios vs. comparison)
- Interactive plotly charts
- Excel export functionality
- Session state management for results persistence

## Portfolio Definitions

Six pre-defined portfolios in `PORTFOLIOS` dict (withdrawal_backtest.py:41-108):

| Portfolio  | Target Return | Target Risk | Key Composition |
|------------|--------------|-------------|-----------------|
| Port_4.0%  | 4.0%         | 3.75%       | 88% Korean bonds |
| Port_5.0%  | 5.0%         | 4.18%       | 76% Korean bonds |
| Port_6.0%  | 6.0%         | 5.00%       | 63% Korean bonds |
| Port_7.0%  | 7.0%         | 6.05%       | 46% Korean bonds |
| Port_8.0%  | 8.0%         | 7.18%       | 35% US growth |
| Port_9.0%  | 9.0%         | 8.36%       | 43% US growth |

## Benchmark Assets

Eight asset classes (Korean names in data, mapped via `BENCHMARK_MAPPING`):

- 미국성장주 (US Growth Stocks)
- 국내주식 / 한국주식 (Korean Stocks)
- 미국국채 / 미국채권 (US Treasury)
- 미국외글로벌채권 (Non-US Global Bonds)
- 신흥국달러채권 (EM Dollar Bonds)
- 한국종합채권 (Korean Composite Bonds)
- 한국국고채10년 (Korean 10Y Treasury)
- 금 (Gold)

## Running the Application

### Streamlit Web App (Primary)
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Command-Line Tests
```bash
python withdrawal_backtest.py
# Runs test scenarios, outputs to /mnt/user-data/outputs/
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Key Algorithms

### Rolling Window Backtest
For each valid start date in historical data:
1. Calculate end date = start + horizon_years
2. Extract daily returns for the period
3. For each month:
   - Deduct monthly withdrawal at month start
   - Compound daily returns for remaining days
4. Record final portfolio value
5. Mark success/failure based on threshold

### Withdrawal Mechanism
- Monthly withdrawal = (annual_rate × initial_capital) / 12
- Applied on first trading day of each month
- Daily compounding between withdrawals
- Stops if portfolio reaches zero

### Failure Definition
- Default: Final value < Initial value (V_T < V_0)
- Configurable threshold (e.g., 50% of initial)

## Code Conventions

### Naming
- **Classes**: PascalCase (`DataPreprocessor`, `WithdrawalSimulator`)
- **Functions/Variables**: snake_case (`run_rolling_backtest`, `month_starts`)
- **Constants**: UPPER_SNAKE_CASE (`PORTFOLIOS`, `BENCHMARK_MAPPING`)
- **Portfolio names**: `Port_X.0%` format

### Documentation
- Docstrings with Parameters/Returns sections
- Korean comments for domain concepts
- Section separators with `# ====...====`

### Streamlit Patterns
- `@st.cache_data` for data loading functions
- Session state for backtest results persistence
- Tab-based UI organization
- Plotly for interactive charts

### Error Handling
- Graceful optional imports (tqdm)
- Data validation in DataPreprocessor
- User-friendly error messages in Streamlit

## Common Development Tasks

### Adding a New Portfolio
Edit `PORTFOLIOS` dict in `withdrawal_backtest.py`:
```python
'Port_10.0%': {
    'target_return': 10.0,
    'target_risk': 9.5,
    'weights': {
        '미국성장주': 50.0,
        '한국종합채권': 30.0,
        '금': 20.0
    }
}
```

### Adding a New Visualization
1. Add method to `WithdrawalVisualizer` class
2. For Streamlit, create corresponding function in `app.py` using plotly
3. Follow existing patterns for figure configuration

### Modifying Backtest Logic
Core logic is in `WithdrawalSimulator.simulate_single_path()` (withdrawal_backtest.py)

### Adding New Benchmark
1. Update `benchmark_to_pkl.py` with new data
2. Add mapping in `BENCHMARK_MAPPING`
3. Regenerate `benchmark_data.pkl`

## Testing Approach

No formal test framework. Testing is done via:
- Ad-hoc scenarios in `withdrawal_backtest.py` `__main__` block
- Visual inspection of generated plots
- Manual verification of statistics

When making changes, run:
```bash
python withdrawal_backtest.py
```
And verify outputs match expected behavior.

## Known Limitations

Current version does NOT support:
- Portfolio rebalancing (buy-and-hold only)
- Dynamic withdrawal strategies (guardrails are in Streamlit only)
- Monte Carlo simulations
- Regime switching models
- Forward-looking projections

## Data Format

`benchmark_data.pkl` structure:
- **Index**: DatetimeIndex (trading days only, 2001-present)
- **Columns**: Asset names in Korean
- **Values**: Price index levels (normalized to 100 base)

## Important Files to Read First

When working on this codebase, start with:
1. `withdrawal_backtest.py:1-120` - Constants and portfolio definitions
2. `withdrawal_backtest.py:200-400` - DataPreprocessor and WithdrawalSimulator
3. `app.py:1-100` - Streamlit setup and data loading
4. `README.md` - User documentation and examples

## Git Workflow

- PR-based development
- Feature branches named `claude/...`
- Commit messages in English
- Test changes locally before pushing
