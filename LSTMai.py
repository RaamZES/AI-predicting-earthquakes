import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from cutter import *

# 1. Loading data from files
def load_data_from_csv(directory):
    data_frames = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

# Loading training and test datasets
train_data = load_data_from_csv('space_apps_2024_seismic_detection/data/mars/training/data')
test_data = load_data_from_csv('space_apps_2024_seismic_detection/data/mars/test/data')

# Checking loaded data
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Outputting the names of all columns for debugging
print("Train data columns:", train_data.columns.tolist())
print("Test data columns:", test_data.columns.tolist())

# Checking if data exists
if train_data.empty or test_data.empty:
    raise ValueError("Training or testing data is empty. Check the data directory.")

# 2. Creating noise data
np.random.seed(42)
noise = np.random.normal(loc=0.0, scale=1.0, size=(70000 + random.randint(1200,2500), 2))  # 2 columns: rel_time and velocity
noise_df = pd.DataFrame(noise, columns=['rel_time', 'velocity'])

# 3. Combining data
noise_df['earthquake'] = 0
train_data['earthquake'] = 1  # Assuming all records in earthquake data correspond to an earthquake

# Combining data
combined_data = pd.concat([train_data, noise_df], ignore_index=True)

# Checking combined data
print("Combined data shape:", combined_data.shape)

# 4. Data preprocessing
print("NaN values in combined data:\n", combined_data.isna().sum())

# Replacing NaN values with the mean value in the corresponding columns
combined_data['rel_time'].fillna(combined_data['rel_time'].mean(), inplace=True)
combined_data['velocity'].fillna(combined_data['velocity'].mean(), inplace=True)

# Checking after replacing NaN
print("Combined data shape after filling NaN with mean:", combined_data.shape)

# Checking for the presence of columns 'rel_time' and 'velocity'
print("Columns in combined data:", combined_data.columns.tolist())
if 'rel_time' not in combined_data.columns or 'velocity' not in combined_data.columns:
    raise ValueError("Columns 'rel_time' and 'velocity' are missing from the dataset.")

# Checking if there is data for normalization
if combined_data[['rel_time', 'velocity']].empty or combined_data[['rel_time', 'velocity']].isnull().all().all():
    raise ValueError("No valid data available for scaling. Check the columns 'rel_time' and 'velocity'.")

# Initializing MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Normalizing data
try:
    scaled_data = scaler.fit_transform(combined_data[['rel_time', 'velocity']])
except ValueError as e:
    print("Error during scaling:", e)
    print("Check if 'rel_time' and 'velocity' contain valid numeric data.")
    print("Combined data for scaling:\n", combined_data[['rel_time', 'velocity']])
    raise

# Creating time series
def create_dataset(data, labels=None, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        if labels is not None:
            y.append(labels[i + time_step - 1])  # Taking the last element in the sequence as the label
        else:
            y.append(0)  # For test data
    return np.array(X), np.array(y)

# Setting the time step
time_step = 10
X, y = create_dataset(scaled_data, combined_data['earthquake'].values, time_step)

# Reshaping data for LSTM
X = X.reshape(X.shape[0], X.shape[1], 2)  # 2 features: rel_time and velocity

# 5. Creating LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 2)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Using sigmoid for binary classification

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Training the model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# 7. Evaluating the model on the test dataset
# Printing the names of the test dataset columns
print("Test data columns for prediction:", test_data.columns.tolist())

# Renaming the columns in the test data
test_data.rename(columns={'rel_time(sec)': 'rel_time', 'velocity(c/s)': 'velocity'}, inplace=True)

# Checking if the required columns exist
if 'rel_time' not in test_data.columns or 'velocity' not in test_data.columns:
    raise ValueError("Columns 'rel_time' and 'velocity' are missing from the test data.")

# Replacing NaN values with the mean value in the corresponding columns
test_data['rel_time'].fillna(test_data['rel_time'].mean(), inplace=True)
test_data['velocity'].fillna(test_data['velocity'].mean(), inplace=True)

# Normalizing test data
test_scaled = scaler.transform(test_data[['rel_time', 'velocity']].dropna())

# Checking if there is data for prediction
if test_scaled.size == 0:
    raise ValueError("No valid data available in the test set for prediction.")

X_test, _ = create_dataset(test_scaled, None, time_step)  # Passing None as labels
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 2)

# Predictions
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)  # Converting probabilities to labels


# Function for prediction based on file
def predict_from_file(file_name):
    user_data = pd.read_csv(file_name)

    # Outputting the names of all columns for debugging
    print("Columns in the user dataset:", user_data.columns.tolist())

    # Renaming columns
    user_data.rename(columns={'rel_time(sec)': 'rel_time', 'velocity(c/s)': 'velocity'}, inplace=True)

    # Checking for the presence of required columns
    if 'rel_time' not in user_data.columns or 'velocity' not in user_data.columns:
        raise ValueError("Columns 'rel_time' and 'velocity' are missing from the dataset.")

    # Replacing NaN values with the mean value in the corresponding columns
    user_data['rel_time'].fillna(user_data['rel_time'].mean(), inplace=True)
    user_data['velocity'].fillna(user_data['velocity'].mean(), inplace=True)

    # Normalizing data
    user_scaled = scaler.transform(user_data[['rel_time', 'velocity']].dropna())

    # Checking for data availability for prediction
    if user_scaled.size == 0:
        raise ValueError("No valid data available in the user dataset for prediction.")

    # Creating time series
    user_X, _ = create_dataset(user_scaled, None, time_step)  # Passing None as labels
    user_X = user_X.reshape(user_X.shape[0], user_X.shape[1], 2)

    # Predictions
    user_predictions = model.predict(user_X)
    user_predictions = (user_predictions > 0.5).astype(int)

    # Checking for earthquake presence
    if user_predictions.any():
        print("There is an earthquake in the data!")
        main_cutter(file_name)
    else:
        print("No earthquakes detected in the data.")

# Infinite loop for requesting file name
while True:
    file_name = input("Enter the file name for prediction (or type 'exit' to quit): ")
    if file_name.lower() == 'exit':
        print("Exiting the program.")
        break
    try:
        predict_from_file(file_name)
    except Exception as e:
        print(f"An error occurred: {e}")