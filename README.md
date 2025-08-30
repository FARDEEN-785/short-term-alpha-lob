# Short-Term Alpha Detection in Limit Order Books

A comprehensive quantitative research project investigating the predictability of high-frequency price movements using Limit Order Book (LOB) data. This project implements a complete pipeline from raw data processing and feature engineering to machine learning modeling and strategic backtesting with transaction costs.

## 🎯 Project Goal

To determine if machine learning models can extract statistically significant predictive signals from market microstructure data to generate profitable trading strategies, while rigorously accounting for real-world transaction costs and market efficiency constraints.

## 📊 Data Source

This project uses high-frequency Limit Order Book data from **LOBSTER** ([https://lobsterdata.com](https://lobsterdata.com)).

**To reproduce this project:**
1.  Create an account on [LOBSTER](https://lobsterdata.com)
2.  Download sample data for **AAPL** (Apple Inc.) for a specific date and time window
3.  Recommended configuration: 1-hour window, 10 order book levels
4.  Place the downloaded orderbook CSV file in the `data/` directory
5.  Rename the file to `AAPL_orderbook_with_features.csv`

*Note: Actual market data files are not included in this repository due to their large size and LOBSTER's licensing terms.*

## ⚙️ Project Architecture

short-term-alpha-lob/
│
├── data/                   # Folder for your data
│   ├── AAPL_orderbook_with_features.csv   # Your initial data file
│   ├── AAPL_orderbook_phase3_features.csv # Your processed features file
│   └── README.txt          # File explaining how to get the data
│
├── notebooks/              # Folder for Jupyter notebooks (Optional but good practice)
│   └── project_walkthrough.ipynb  # Your main showcase notebook
│
├── src/                    # Folder for source code (Optional but good practice)
│   ├── features.py         # Your feature engineering script
│   └── model.py            # Your modeling and backtesting script
│
├── .gitignore              # Tells Git what files to ignore (CRUCIAL)
├── LICENSE                 # Tells others how they can use your code
└── README.md               # The homepage of your project (CRUCIAL)



## 🔬 Methodology & Technical Approach

### 1. Feature Engineering
Transformed raw order book data into economically meaningful features:
- **Basic Features:** `MidPrice`, `Spread`, `WeightedMid` (volume-weighted midpoint)
- **Market Dynamics:** `Imbalance` (bid-ask volume imbalance), `TotalDepth` (total liquidity)
- **Advanced Metrics:** `OrderBook Slope` (price curvature), `Volatility` (calculated from log returns)
- **Temporal Features:** Sequential patterns across 20-time-step windows

### 2. Modeling Framework
Trained and evaluated multiple model architectures:
- **Baseline Model:** Logistic Regression with L2 regularization
- **Deep Learning Model:** LSTM (Long Short-Term Memory) network with 64 units
- **Validation:** Rigorous time-series train-test split to prevent data leakage

### 3. Target Formulation
- **Classification Task:** 3-class prediction (`Up`, `Down`, `Neutral`)
- **Thresholding:** Significance threshold based on average bid-ask spread
- **Prediction Horizon:** 10-step ahead forecasting

### 4. Backtesting Protocol
- **Realistic Assumptions:** Incorporates transaction costs (bid-ask spread)
- **Benchmarking:** Comparison against buy-and-hold strategy
- **Performance Metrics:** Cumulative returns, Sharpe ratio, maximum drawdown

## 📈 Key Results & Findings

### Model Performance
- **Accuracy:** Both models achieved ~33% accuracy on 3-class prediction, performing marginally better than random chance (33.3%)
- **Confusion Analysis:** Models showed strong bias toward predicting 'Neutral' class, with limited ability to identify directional moves
- **Benchmark Comparison:** LSTM complexity provided no significant advantage over simple Logistic Regression

### Economic Significance
- **Transaction Cost Impact:** The bid-ask spread eliminated any potential profitability from predictive signals
- **Strategy Performance:** Net-of-costs returns were negative or negligible across all configurations
- **Market Efficiency:** Results strongly support the efficient market hypothesis at high frequencies

### Technical Insights
- **Data Leakage Prevention:** Implemented rigorous scaling procedures after train-test splitting
- **Feature Importance:** Order book imbalance and depth showed some predictive value
- **Temporal Patterns:** Short-term momentum effects were detected but insufficient for profitability

## 🧠 Lessons Learned

### Quantitative Insights
1.  **Market Efficiency Dominates:** Price movements at high frequencies are largely unpredictable after accounting for costs
2.  **Costs Are Everything:** Transaction costs (spread) are the primary determinant of strategy viability
3.  **Simple Beats Complex:** Sophisticated models (LSTM) provided no advantage over simple linear models for this task

### Technical Skills Developed
- **Data Engineering:** Processing high-frequency financial data with Pandas/NumPy
- **Feature Engineering:** Creating economically meaningful features from raw order books
- **Machine Learning:** Implementing both traditional and deep learning models
- **Validation Rigor:** Preventing data leakage and ensuring reproducible results
- **Backtesting:** Building realistic trading simulations with transaction costs

### Professional Practices
- **Documentation:** Creating reproducible research with clear documentation
- **Version Control:** Professional Git workflow and repository organization
- **Result Interpretation:** Moving beyond accuracy metrics to economic significance

## 📊 Data Source

This project uses high-frequency Limit Order Book data from **LOBSTER** (https://lobsterdata.com).

**The data files are not included in this repository.** To run this project, you must download the data separately and place it in the `data/` folder. See the detailed instructions in `data/README.txt`.
