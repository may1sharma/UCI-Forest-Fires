from numpy.linalg import inv
import pandas as pd
import numpy as np


def main():
    # Read input files and separate X and Y
    train_data = pd.read_csv('train.csv')
    trainX = train_data[train_data['area'] > 0].iloc[:, :12]
    trainY = train_data[train_data['area'] > 0].iloc[:, 12]

    test_data = pd.read_csv('test.csv')
    testX = test_data[test_data['area'] > 0].iloc[:, :12]
    testY = test_data[test_data['area'] > 0].iloc[:, 12]

    # BONUS: Uncomment following for Non-linear regression
    # trainX = trainX ** 3 + 5 * (trainX ** 2)
    # testX = testX ** 3 + 5 * (testX ** 2)

    # Normalize each feature
    for i in range(trainX.shape[1]):
        normalize(trainX, i)
        normalize(testX, i)

    # Linear Regression using Ordinary Least Squares
    optimal_weights = OLS_solution(trainX, trainY)
    print("Optimal weights from OLS ")
    print(optimal_weights)

    # Test the model
    testX = np.array(testX).astype(float)
    nRows = testX.shape[0]
    x0 = np.ones((nRows, 1))
    testX = np.concatenate((x0, testX), axis=1)
    predictions = np.dot(testX, optimal_weights)

    testY = np.array(testY)
    diff = testY - predictions

    RSSerror = np.dot(diff, diff)
    print()
    print("Residual Sum of Squares Errors ", RSSerror)

    correlation = np.corrcoef(testY, predictions)
    print()
    print("Correlation between Actual and Predicted values ")
    print(correlation)


def normalize(df, col):
    _mean = np.mean(df.iloc[:, col])
    _std = np.std(df.iloc[:, col])
    df.iloc[:, col] = df.iloc[:, col] - _mean
    if _std:    # avoid / by 0
        df.iloc[:, col] /= _std


def OLS_solution(trainX, trainY):
    X = np.array(trainX).astype(float)
    nRows = X.shape[0]
    x0 = np.ones((nRows, 1))
    X = np.concatenate((x0, X), axis=1)
    Y = np.array(trainY)

    ols = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, Y))
    return ols


main()
