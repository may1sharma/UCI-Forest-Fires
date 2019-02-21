import math
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    # Read input files and separate X and Y
    train_data = pd.read_csv('train.csv')
    trainX = train_data.iloc[:, :12]
    trainY = train_data.iloc[:, 12]
    trainY[trainY.iloc[:] > 0] = 1
    trainY = trainY.astype(int)

    test_data = pd.read_csv('test.csv')
    testX = test_data.iloc[:, :12]
    testY = test_data.iloc[:, 12]
    testY[testY.iloc[:] > 0] = 1
    testY = testY.astype(int)

    # Normalize each feature
    for i in range(trainX.shape[1]):
        normalize(trainX, i)
        normalize(testX, i)

    # Find optimal value of hyperparameter k
    # k = crossValidate(trainX, trainY)
    k=11
    # Test the model
    predictions = []

    for x in range(len(testX)):
        neighbors = kNearest(trainX, trainY, testX.iloc[x], k)
        result = predict(neighbors)
        predictions.append(result)
    accu = calc_accuracy(testY.values, predictions)
    print('Test Accuracy:', repr(accu) + '%')


def normalize(df, col):
    max_value = df.iloc[:, col].max()
    min_value = df.iloc[:, col].min()
    if (max_value - min_value) != 0:
        df.iloc[:, col] = (df.iloc[:, col] - min_value) / (max_value - min_value)


def crossValidate(trainX, trainY):
    N = len(trainX)
    bestk = 0
    bestaccuracy = 0
    k_values = []
    allAccuracy = []

    for k in range(1, 21):
        print('Iteration k =', k)
        sum = 0
        for i in range(5):
            start = int((i*N)/5)
            end = int(((i+1)*N)/5)
            val_setX = trainX.iloc[start: end]
            val_setY = trainY.iloc[start: end]
            train_setX = trainX.drop(trainX.index[start: end])
            train_setY = trainY.drop(trainY.index[start: end])

            predictions = []
            for x in range(len(val_setX)):
                neighbors = kNearest(train_setX, train_setY, val_setX.iloc[x], k)
                result = predict(neighbors)
                predictions.append(result)

            sum += calc_accuracy(val_setY.values, predictions)

        avg_accuracy = sum/5
        k_values.append(k)
        allAccuracy.append(avg_accuracy)

        if avg_accuracy > bestaccuracy:
            bestaccuracy = avg_accuracy
            bestk = k

    plt.plot(k_values, allAccuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('K')
    plt.show()
    print('Best k value', bestk)
    return bestk


def kNearest(trainX, trainY, testX, k):
    distances = []
    for x in range(len(trainX)):
        dist = euclideanDist(testX, trainX.iloc[x])
        # BONUS
        # dist = manhattanDist(testX, trainX.iloc[x])
        # dist = combinationDist(testX, trainX.iloc[x])
        distances.append((trainY.iloc[x], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def predict(neighbors):
    zeros = 0
    ones = 0

    for val in neighbors:
        if val == 0:
            zeros += 1
        else:
            ones += 1
    return 0 if zeros >= ones else 1


def calc_accuracy(test_output, predictions):
    correct = 0
    for x in range(len(test_output)):
        if test_output[x] == predictions[x]:
            correct += 1
    return (correct / float(len(test_output))) * 100.0


def euclideanDist(instance1, instance2):
    distance = 0
    for x in range(len(instance1)):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def manhattanDist(instance1, instance2):
    distance = 0
    for x in range(len(instance1)):
        distance += abs(instance1[x] - instance2[x])
    return distance

def combinationDist(instance1, instance2):
    distance = 0
    for x in range(len(instance1)):
        if x<4:
            distance += 1 if instance1[x]!=instance2[x] else 0
        else:
            distance += abs(instance1[x] - instance2[x])
    return distance


main()
