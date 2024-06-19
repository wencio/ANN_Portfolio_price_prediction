
import dask.dataframe as dd
from dask.distributed import Client
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def normalize_data(df):
    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['open', 'close', 'low', 'high', 'volume']])
    return scaler, scaled_data

def create_model():
    # Load data from a CSV file using Dask
    df = dd.read_csv('prices.csv')
    # Select relevant columns
    df = df[['date', 'open', 'close', 'low', 'high', 'volume']]
    # Set 'date' as index and compute the DataFrame
    df = df.set_index('date').compute()

    # Normalize the data
    scaler, scaled_data = normalize_data(df)

    # Create training and test datasets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Create datasets with time windows
    def create_dataset(data, time_step=1):
        X, Y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), :])
            Y.append(data[i + time_step, 1])  # Close price (index 1 in scaled_data)
        return np.array(X), np.array(Y)

    time_step = 60  # Use 60 days of data to predict the next day
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    # Reshape data to fit LSTM input [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, Y_train, batch_size=1, epochs=1)

    # Make predictions with test data
    predictions = model.predict(X_test)

    # Invert the normalization
    predictions = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], df.shape[1]-1)), predictions), axis=1))[:, -1]
    print(predictions)

    # Calculate the error
    Y_test_unscaled = scaler.inverse_transform(np.concatenate((np.zeros((Y_test.shape[0], df.shape[1]-1)), Y_test.reshape(-1, 1)), axis=1))[:, -1]
    rmse = np.sqrt(np.mean(((predictions - Y_test_unscaled) ** 2)))
    print(f"RMSE: {rmse}")

    return model

# Predict future prices with new data
def predict_future(model, time_step, df):
    # Normalize the new data
    scaler, scaled_data = normalize_data(df)
    # Take the last 'time_step' days and reshape
    new_data = scaled_data[-time_step:]  
    data = new_data.reshape(1, time_step, new_data.shape[1])
    # Make prediction
    prediction = model.predict(data)
    # Invert the normalization
    prediction = scaler.inverse_transform(np.concatenate((np.zeros((1, df.shape[1]-1)), prediction), axis=1))[:, -1]
    return prediction

