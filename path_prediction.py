# path_prediction.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def process_data(file_name):
    """
    Process the data file to extract x and y coordinates.
    
    Args:
    file_name (str): Path to the data file.
    
    Returns:
    np.array: Array of x and y coordinates.
    """
    with open(file_name, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        x_str, y_str = line.strip().split(',')
        x = int(x_str)
        y = int(y_str)
        data.append([x, y])
    data = np.array(data)
    return data

def knn_path_prediction(training_data_X, training_data_Y, testing_data_X, time_interval, k=1):
    """
    Predict path using K-Nearest Neighbors algorithm.
    
    Args:
    training_data_X (np.array): Training data features.
    training_data_Y (np.array): Training data labels.
    testing_data_X (np.array): Testing data features.
    time_interval (tuple): Time interval for prediction.
    k (int): Number of neighbors to use.
    """
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(training_data_X, training_data_Y)
    
    actual_path = testing_data_X[time_interval[0]:time_interval[1], :]
    predicted_path = knn_model.predict(testing_data_X)[time_interval[0]:time_interval[1], :]
    
    plt.plot(actual_path[:, 0], actual_path[:, 1], 'r-', label='Actual Path')
    plt.plot(predicted_path[:, 0], predicted_path[:, 1], 'b-', label='Predicted Path')
    plt.title('KNN Path Prediction')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.legend()
    plt.show()

def train_model(X_train, y_train, X_test, y_test):
    """
    Train a Decision Tree Regressor model and compute RMSE.
    
    Args:
    X_train (np.array): Training data features.
    y_train (np.array): Training data labels.
    X_test (np.array): Testing data features.
    y_test (np.array): Testing data labels.
    
    Returns:
    float: Root mean squared error of the model predictions.
    """
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))
    
    model = DecisionTreeRegressor()
    model.fit(X_train_flat, y_train)
    y_pred = model.predict(X_test_flat)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def create_lookback_dataset(X, y, lookback=1):
    """
    Create lookback dataset for time series prediction.
    
    Args:
    X (np.array): Features data.
    y (np.array): Labels data.
    lookback (int): Lookback period.
    
    Returns:
    tuple: Lookback features and labels.
    """
    X_lookback, y_lookback = [], []
    for i in range(len(X) - lookback):
        X_lookback.append(X[i:(i + lookback)])
        y_lookback.append(y[i + lookback])
    return np.array(X_lookback), np.array(y_lookback)

def decision_tree_path_prediction(training_data_X, training_data_Y, testing_data_X, testing_data_Y, lookback=3, start_index=500, end_index=680):
    """
    Predict path using Decision Tree Regressor.
    
    Args:
    training_data_X (np.array): Training data features.
    training_data_Y (np.array): Training data labels.
    testing_data_X (np.array): Testing data features.
    testing_data_Y (np.array): Testing data labels.
    lookback (int): Lookback period for creating lookback dataset.
    start_index (int): Starting index for prediction interval.
    end_index (int): Ending index for prediction interval.
    """
    X_train = training_data_X[:-(lookback)]
    y_train = training_data_Y[lookback:]

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    X_interval = testing_data_X[start_index:end_index - lookback]
    y_actual = testing_data_Y[start_index + lookback:end_index]

    y_pred = model.predict(X_interval)

    plt.figure(figsize=(8, 6))
    plt.plot(y_actual[:, 0], y_actual[:, 1], label='Actual Path')
    plt.plot(y_pred[:, 0], y_pred[:, 1], label='Predicted Path')
    plt.title('Decision Tree Path Prediction')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    path_test = "Dataset/Testing/test01.txt"
    path_training = "Dataset/Training/training_data.txt"

    training_data_X = process_data(path_training)
    training_data_Y = training_data_X[1:]
    training_data_X = training_data_X[:-1]
    testing_data_X = process_data(path_test)
    testing_data_Y = testing_data_X[1:]
    testing_data_X = testing_data_X[:-1]

    time_interval = (10 * 30, 16 * 30)

    # KNN Path Prediction
    knn_path_prediction(training_data_X, training_data_Y, testing_data_X, time_interval, k=1)

    # Decision Tree Path Prediction with Lookback Evaluation
    lookbacks = [1, 2, 3]
    rmse = []

    for lookback in lookbacks:
        X_train_lb, y_train_lb = create_lookback_dataset(training_data_X, training_data_Y, lookback)
        X_test_lb, y_test_lb = create_lookback_dataset(testing_data_X, testing_data_Y, lookback)
        result = train_model(X_train_lb, y_train_lb, X_test_lb, y_test_lb)
        rmse.append(result)
        print(f"Lookback {lookback} - Rmse: {result}")

    plt.plot(lookbacks, rmse, marker='o')
    plt.title('RMSE vs. Lookback Size')
    plt.xlabel('Lookback Size')
    plt.ylabel('RMSE')
    plt.xticks(lookbacks)
    plt.show()

    # Decision Tree Path Prediction with best lookback
    decision_tree_path_prediction(training_data_X, training_data_Y, testing_data_X, testing_data_Y, lookback=3, start_index=500, end_index=680)
