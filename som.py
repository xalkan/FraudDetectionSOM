# Self Organizing Map for Fraud Detection

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
# separate labels to test against later
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
a = 1

# training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
# mid = mean inter-neuron distance between neurons
# higher mid = outlier = encoded in color
from pylab import bone, pcolor, colorbar, plot, show
bone()
som_mid_matrix = som.distance_map().T
pcolor(som_mid_matrix)    # T is transpose of the mid matrix
colorbar()
# highlight customers 
# red - customers who didnt get approval
# green - customers who got approval
markers = ['o', 's']
colors = ['r', 'g']

for i, customer in enumerate(X):
    # winning node of the customer at i
    winning_node = som.winner(customer)
    # plot marker at the center of the node
    # y[customer] is the approval label, either 0 or 1
    plot(winning_node[0] + 0.5, winning_node[1] + 0.5,
         markers[y[i]], markeredgecolor = colors[y[i]],
         markerfacecolor = 'None', markersize = 10, markeredgewidth = 2 )

show()


# finding the frauds - those outliers who got approval
# mappings is a dict with key = som coordinates and value = array of customers

mappings = som.win_map(X)
frauds = []
# getting frauds from the som mid matrix
for row in range(len(som_mid_matrix)):
    for col in range(len(som_mid_matrix[0])):
        if som_mid_matrix[row][col] > 0.9:
            for customer in mappings[(row, col)]:
                frauds.append(customer)

# revert normalization
frauds = sc.inverse_transform(frauds)

