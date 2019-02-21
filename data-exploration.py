import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')
print(data.head())

for i in range(12):
    feature_name = data.columns[i]

    # Plotting Histogram of feature feature_name for part b
    plt.hist(data.iloc[:, i])
    plt.ylabel('Frequency')
    plt.xlabel(feature_name)
    plt.title('Histogram of feature ' + feature_name)
    plt.show()

    # Plotting Scatter plot of feature feature_name and log(burned area + 1)
    plt.scatter(data.iloc[:, i], np.log(data.iloc[:, 12] + 1))
    plt.ylabel('log(burned area + 1)')
    plt.xlabel(feature_name)
    plt.title('Scatter plot of log(burned area +1) vs ' + feature_name)
    plt.show()


# Plotting histogram of output variable for part c
plt.hist(data.iloc[:, 12])
plt.ylabel('Frequency')
plt.xlabel('area')
plt.title('Histogram of output variable area')
plt.show()

plt.hist(np.log(data.iloc[:, 12] + 1))
plt.ylabel('Frequency')
plt.xlabel('log(area) + 1')
plt.title('Histogram of logarithm of output variable area + 1')
plt.show()
