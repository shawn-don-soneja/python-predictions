import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import json

"""
Removed code, for random data generation

# Generate some example historical data
# Replace this with your own historical data
np.random.seed(42)
X = np.arange(1, 101).reshape(-1, 1)
y = 2 * X + np.random.normal(scale=5, size=(100, 1))

"""

# Retrieve Sample Data from JSON
with open('sample_request_json.json') as sample_data:
    data = json.load(sample_data)
X = np.array(data['data']['x'])
y = np.array(data['data']['y'])

# Print the data to verify
print("Data Received:")
print(X)
print(y)

# Normalize the data
print("Normalized Data")
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

# Expand dimensions to make it suitable for LSTM input
X_train_seq = np.expand_dims(X_train_seq, axis=-1)
X_test_seq = np.expand_dims(X_test_seq, axis=-1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Create a TensorFlow dataset and use .repeat() to avoid the OUT_OF_RANGE error
import tensorflow as tf

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train_seq))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(8).repeat()

# Calculate the steps per epoch
steps_per_epoch = len(X_train_seq) // 8

# Train the model
model.fit(train_dataset, epochs=50, steps_per_epoch=steps_per_epoch, verbose=1)

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
