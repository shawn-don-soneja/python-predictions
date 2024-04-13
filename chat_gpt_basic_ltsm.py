import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import json
from datetime import datetime

"""
Removed code, for random data generation

# Generate some example historical data
# Replace this with your own historical data
np.random.seed(42)
X = np.arange(1, 101).reshape(-1, 1)
y = 2 * X + np.random.normal(scale=5, size=(100, 1))

"""

# Retrieve Sample Data from JSON
sample_data = open('sample_request_json.json')
data = json.load(sample_data)
X = data['data']['x']
y = data['data']['y']


# Convert all dates to DateTime before normalization? See impacts...
X = [datetime.strptime(xValue, '%m/%d/%Y') for xValue in X]


# Stored Historical Data
# {parameter from client request}

print(X)
print(y)

# Normalize the data
X = X / np.max(X)
y = y / np.max(y)

print(X)
print(y)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Convert data to sequences suitable for LSTM
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

seq_length = 10  # Adjust this based on your data characteristics
X_train_seq, y_train_seq = create_sequences(y_train, seq_length)
X_test_seq, y_test_seq = create_sequences(y_test, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=8, verbose=1)

# Make predictions on the test set
y_pred = model.predict(X_test_seq)

# ADD make predictions into the future
"""
Make sure that we train based on all historical

And predict based on defined, date range extension into the future

"""

# Visualize the results
# Not needed for Lambda script
plt.plot(y_test_seq, label='Actual Data')
plt.plot(y_pred, label='LSTM Prediction')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Trend Prediction using LSTM')
plt.legend()
plt.show()
