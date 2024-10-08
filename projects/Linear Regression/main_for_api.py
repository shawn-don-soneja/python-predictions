import pandas as pandas
import numpy as numpy
from sklearn.linear_model import LinearRegression
import os
import json

directory_path = "../Chat GPT__LTSM Prediction/supporting_files/sample_request_json.json"  # Replace with your directory path

with open(directory_path) as sample_data:
    data = json.load(sample_data)

X = numpy.array(data['data']['x'])
Y = numpy.array(data['data']['y'])

# Print the data to verify
print("Data Received:")
print(X)
print(Y)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, Y)

# Make predictions
Y_pred = model.predict(X)

# Print the model parameters
print(f"Slope (Coefficient): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")