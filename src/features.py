import pandas as pd
import numpy as np 

# File paths
orderbook_file = r"D:\short-term-alpha-lob\data\AAPL_orderbook_with_features.csv"
output_file = r"D:\short-term-alpha-lob\data\AAPL_orderbook_features.csv"

# Define orderbook headers (top 4 levels)
orderbook_cols = []
for i in range(1, 5):
    orderbook_cols += [f"AskPrice{i}", f"AskSize{i}"]
for i in range(1, 5):
    orderbook_cols += [f"BidPrice{i}", f"BidSize{i}"]

# At the top of your file, after defining orderbook_cols
NUM_LEVELS = 4
# Load orderbook CSV
orderbook = pd.read_csv(orderbook_file, header=None, names=orderbook_cols)

# -------------------
# Phase 3: Feature Engineering
# -------------------

price_size_cols = [
    'AskPrice1','AskPrice2','AskPrice3','AskPrice4',
    'AskSize1','AskSize2','AskSize3','AskSize4',
    'BidPrice1','BidPrice2','BidPrice3','BidPrice4',
    'BidSize1','BidSize2','BidSize3','BidSize4'
]

# Convert to numeric, coerce errors to NaN
orderbook[price_size_cols] = orderbook[price_size_cols].apply(pd.to_numeric, errors='coerce')

# Optional: drop rows with NaN after conversion, or fill them
orderbook.dropna(subset=price_size_cols, inplace=True)


# 1️⃣ Mid-Price
orderbook['MidPrice'] = (orderbook['AskPrice1'] + orderbook['BidPrice1']) / 2

# 2️⃣ Spread
orderbook['Spread'] = (orderbook['AskPrice1'] - orderbook['BidPrice1']).abs()

# 3️⃣ Imbalance
orderbook['Imbalance'] = (
    orderbook[['BidSize1','BidSize2','BidSize3','BidSize4']].sum(axis=1) -
    orderbook[['AskSize1','AskSize2','AskSize3','AskSize4']].sum(axis=1)
) / (
    orderbook[['BidSize1','BidSize2','BidSize3','BidSize4']].sum(axis=1) +
    orderbook[['AskSize1','AskSize2','AskSize3','AskSize4']].sum(axis=1)
)

# 4️⃣ Volatility (short-term rolling std of LOG RETURNS) - MORE ACCURATE
# First, calculate log returns of the MidPrice
orderbook['MidPrice_Returns'] = np.log(orderbook['MidPrice'] / orderbook['MidPrice'].shift(1))
# Now, calculate rolling standard deviation of the returns
window_size = 20
orderbook['Volatility'] = orderbook['MidPrice_Returns'].rolling(window=window_size).std()
# Fill NaN values that result from the shift and rolling calculations
orderbook['Volatility'].fillna(method='bfill', inplace=True) # Backfill is often better than 0 for volatility

# 5️⃣ Weighted Mid-Price (Volume-weighted average of top of book)
orderbook['WeightedMid'] = (
    (orderbook['BidPrice1'] * orderbook['AskSize1'] + orderbook['AskPrice1'] * orderbook['BidSize1']) /
    (orderbook['AskSize1'] + orderbook['BidSize1'])
)

# 6️⃣ Order Book Slope (Average price change between levels - indicates "steepness" of the book)
# Calculate average spread between levels on Ask side
orderbook['AskSlope'] = (
    (orderbook['AskPrice1'] - orderbook['AskPrice2']).abs() +
    (orderbook['AskPrice2'] - orderbook['AskPrice3']).abs() +
    (orderbook['AskPrice3'] - orderbook['AskPrice4']).abs()
) / 3

# Calculate average spread between levels on Bid side
orderbook['BidSlope'] = (
    (orderbook['BidPrice1'] - orderbook['BidPrice2']).abs() +
    (orderbook['BidPrice2'] - orderbook['BidPrice3']).abs() +
    (orderbook['BidPrice3'] - orderbook['BidPrice4']).abs()
) / 3

# 7️⃣ Total Depth (Total liquidity available in the first 4 levels)
orderbook['TotalBidDepth'] = orderbook[['BidSize1','BidSize2','BidSize3','BidSize4']].sum(axis=1)
orderbook['TotalAskDepth'] = orderbook[['AskSize1','AskSize2','AskSize3','AskSize4']].sum(axis=1)


# -------------------
# Save Phase 3 dataset
# -------------------
orderbook.to_csv(output_file, index=False)
print("✅ Phase 3 features calculated and saved to CSV")
