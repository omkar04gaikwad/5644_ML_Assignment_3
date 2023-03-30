import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, multivariate_normal as mvn
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

from variable import samples, classes, gmmparam, num_classes, train_sample, test_sample

np.set_printoptions(suppress=True)

np.random.seed(7)

def generate_data(sample, gmmparam):
    n = gmmparam['meanvectors'].shape[1]
    X = np.zeros([sample, n])
    labels = np.zeros(sample)
    u = np.random.rand(sample)
    threshold = np.cumsum(gmmparam['priors'])
    threshold = np.insert(threshold, 0, 0)
    L = np.array(range(len(gmmparam['priors'])))
    for l in L:
        indices = np.argwhere((threshold[l] <= u) & (u <= threshold[l+1]))[:, 0]
        N_labels = len(indices)
        labels[indices] = l * np.ones(N_labels)
        X[indices, :] = mvn.rvs(gmmparam['meanvectors'][l], gmmparam['covariancematrices'][l], N_labels)
    return X, labels

def plot_data(X, labels):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[labels == 0, 0], X[labels == 0, 1], X[labels== 0, 2], c='r', label="Class 0")
    ax.scatter(X[labels == 1, 0], X[labels == 1, 1], X[labels == 1, 2], c='b', label="Class 1")
    ax.scatter(X[labels == 2, 0], X[labels == 2, 1], X[labels == 2, 2], c='g', label="Class 2")
    ax.scatter(X[labels == 3, 0], X[labels == 3, 1], X[labels == 3, 2], c='k', label="Class 3")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$x_3$")
    plt.title("Data and True Class Labels")
    plt.legend()
    plt.tight_layout()
    plt.show()

def generate_datasample():
    X_train = []
    Y_train = []

    for N_i in train_sample:
        print("Generating the Training data set for sample = {}".format(N_i))

        X_i, Y_i = generate_data(N_i, gmmparam)

        X_train.append(X_i)
        Y_train.append(Y_i)

    print("Generating the Training data set for sample = {}".format(test_sample))
    X_test, Y_test = generate_data(test_sample, gmmparam)
    return X_train, Y_train, X_test, Y_test

X, labels = generate_data(samples,gmmparam)
print("Generated Data:")
plot_data(X, labels)
X_train, Y_train, X_test, Y_test = generate_datasample()
print("All Datasets generated")