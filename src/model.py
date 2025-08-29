# model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# -------------------------------
# 1. Load data
# -------------------------------
phase3_file = r"D:\short-term-alpha-lob\data\AAPL_orderbook_features.csv"
orderbook = pd.read_csv(phase3_file)

# Ensure numeric columns
feature_cols = [
    'MidPrice', 'Spread', 'Imbalance', 'Volatility',
    'WeightedMid', 'AskSlope', 'BidSlope',
    'TotalBidDepth', 'TotalAskDepth'
]
orderbook[feature_cols] = orderbook[feature_cols].apply(pd.to_numeric, errors='coerce')
orderbook.dropna(subset=feature_cols, inplace=True)

# -------------------------------
# 2. Define Target
# -------------------------------
N = 10  # prediction horizon
orderbook['FutureMid'] = orderbook['MidPrice'].shift(-N)
orderbook['Target_Regression'] = orderbook['FutureMid'] - orderbook['MidPrice'] # Keep for potential future use

def categorize(x, threshold=0.0002):  # Use a small threshold, e.g., 2 basis points
    if x > threshold:
        return 1
    elif x < -threshold:
        return -1
    else:
        return 0

orderbook['Target_Class'] = orderbook['Target_Regression'].apply(categorize)
orderbook = orderbook[:-N] # Remove last N rows

# Get features and target
X = orderbook[feature_cols].values
y_class = orderbook['Target_Class'].values

# -------------------------------
# 3. Train-Test Split (CRITICAL: Do this FIRST!)
# -------------------------------
# Split the indices to avoid data leakage
split_idx = int(0.8 * len(X))
X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
y_train_class, y_test_class = y_class[:split_idx], y_class[split_idx:]

print(f"Training samples: {X_train_raw.shape[0]}, Test samples: {X_test_raw.shape[0]}")

# -------------------------------
# 4. Scale Features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw) # Fit ONLY on training data
X_test_scaled = scaler.transform(X_test_raw)       # Transform test data using training fit

# -------------------------------
# 5. Create Sequences for LSTM
# -------------------------------
SEQ_LEN = 20

def create_sequences(X, y, seq_len=SEQ_LEN):
    sequences = []
    targets = []
    for i in range(len(X) - seq_len):
        sequences.append(X[i:i+seq_len])
        targets.append(y[i+seq_len])
    return np.array(sequences), np.array(targets)

# Create sequences for the LSTM
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_class)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_class)

# Convert labels to categorical for LSTM
y_train_cat = to_categorical(y_train_seq + 1, num_classes=3)  # map -1,0,1 to 0,1,2
y_test_cat = to_categorical(y_test_seq + 1, num_classes=3)

print(f"LSTM Training sequences: {X_train_seq.shape}")
print(f"LSTM Test sequences: {X_test_seq.shape}")

# -------------------------------
# 6. Benchmark with Logistic Regression
# -------------------------------
# Flatten the sequences for the simple model
def flatten_sequences(X_sequences):
    return X_sequences.reshape(X_sequences.shape[0], -1) # Flatten to (samples, seq_len * features)

X_train_flat = flatten_sequences(X_train_seq)
X_test_flat = flatten_sequences(X_test_seq)

# Train Logistic Regression
print("\n--- Training Logistic Regression Benchmark ---")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_flat, y_train_seq) # Use the non-categorical labels

y_pred_lr = lr_model.predict(X_test_flat)
lr_accuracy = accuracy_score(y_test_seq, y_pred_lr)
print(f"Logistic Regression Test Accuracy: {lr_accuracy:.4f}")

# -------------------------------
# 7. Build and Train LSTM Model
# -------------------------------
print("\n--- Training LSTM Model ---")
model = Sequential([
    LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train_seq, y_train_cat, epochs=10, batch_size=64, 
                    validation_split=0.1, shuffle=False, verbose=1)

# -------------------------------
# 8. Evaluate LSTM
# -------------------------------
print("\n--- Evaluating LSTM Model ---")
loss, accuracy = model.evaluate(X_test_seq, y_test_cat, verbose=0)
print(f"LSTM Test Accuracy: {accuracy:.4f}")

# Get predictions for detailed analysis
y_pred_proba = model.predict(X_test_seq)
y_pred_lstm = np.argmax(y_pred_proba, axis=1) - 1  # map back to -1,0,1

# -------------------------------
# -------------------------------
# 10. Simple Backtest with Transaction Costs
# -------------------------------
print("\n" + "="*50)
print("SIMPLE BACKTEST")
print("="*50)

# Assume we trade at the mid-price but pay the spread as a cost
# This is a conservative assumption
average_spread = orderbook['Spread'].mean()
print(f"Average Bid-Ask Spread: {average_spread:.6f}")

# Create a DataFrame for the test set for easier plotting
backtest_df = pd.DataFrame({
    'True_Label': y_test_seq,
    'Predicted_Label': y_pred_lstm,
    # We need the corresponding MidPrice for the test period
    # We get it by using the last element of each sequence
    'MidPrice': [X_test_seq[i, -1, 0] for i in range(len(X_test_seq))] 
}, index=range(len(y_test_seq)))

# Calculate daily returns based on predictions
# Strategy: Buy if prediction == 1, Sell if prediction == -1, Hold otherwise.
backtest_df['Strategy_Return'] = 0.0

for i in range(1, len(backtest_df)):
    predicted_signal = backtest_df.iloc[i]['Predicted_Label']
    prev_price = backtest_df.iloc[i-1]['MidPrice']
    current_price = backtest_df.iloc[i]['MidPrice']
    
    # Calculate the raw return
    raw_return = (current_price - prev_price) / prev_price
    
    # If we predicted UP, we go long and get the raw return
    if predicted_signal == 1:
        backtest_df.loc[i, 'Strategy_Return'] = raw_return
    # If we predicted DOWN, we go short and get the inverse return
    elif predicted_signal == -1:
        backtest_df.loc[i, 'Strategy_Return'] = -raw_return
    # Else, hold cash -> 0% return

# Subtract transaction costs (we pay the spread every time we trade)
# Identify trade entries (when signal changes)
backtest_df['Position'] = backtest_df['Predicted_Label'].replace({0: 0, 1: 1, -1: -1})
backtest_df['Trade'] = (backtest_df['Position'].diff() != 0).astype(int)
# Apply cost: Every time we trade, subtract half the spread (one-way cost)
backtest_df['Strategy_Return_Net'] = backtest_df['Strategy_Return'] - (backtest_df['Trade'] * (average_spread / 2))

# Calculate cumulative returns
backtest_df['Cumulative_Strategy_Return'] = (1 + backtest_df['Strategy_Return_Net']).cumprod()
backtest_df['Cumulative_Buy_Hold_Return'] = (1 + backtest_df['Strategy_Return']).cumprod() # Benchmark

# Plot the equity curve
plt.figure(figsize=(12, 6))
plt.plot(backtest_df['Cumulative_Strategy_Return'], label='Trading Strategy (Net of Costs)')
plt.plot(backtest_df['Cumulative_Buy_Hold_Return'], label='Buy & Hold Benchmark', alpha=0.7)
plt.title("Strategy Equity Curve")
plt.xlabel("Time Step")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

# Print final performance
final_return = backtest_df['Cumulative_Strategy_Return'].iloc[-1] - 1
final_bh_return = backtest_df['Cumulative_Buy_Hold_Return'].iloc[-1] - 1

print(f"\nFinal Strategy Return (Net): {final_return:.4%}")
print(f"Final Buy & Hold Return: {final_bh_return:.4%}")
# 1. Confusion Matrix
print("\n1. Confusion Matrix (LSTM):")
cm = confusion_matrix(y_test_seq, y_pred_lstm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Down (-1)', 'Neutral (0)', 'Up (1)'], 
            yticklabels=['Down (-1)', 'Neutral (0)', 'Up (1)'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix for LSTM Predictions')
plt.show()

# 2. Classification Report
print("\n2. Classification Report (LSTM):")
print(classification_report(y_test_seq, y_pred_lstm, 
                            target_names=['Down (-1)', 'Neutral (0)', 'Up (1)']))

# 3. Compare to Benchmark
print(f"\n3. Model Comparison:")
print(f"   Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"   LSTM Accuracy:               {accuracy:.4f}")

# 4. Compare to Random Guessing
class_counts = np.bincount(y_test_seq + 1) # Count occurrences of -1,0,1 (shifted to 0,1,2)
most_common_class_prob = np.max(class_counts) / np.sum(class_counts)
print(f"   Most Common Class (Neutral) Baseline: {most_common_class_prob:.4f}")