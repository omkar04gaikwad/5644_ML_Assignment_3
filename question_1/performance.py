import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, multivariate_normal as mvn
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

from variable import samples, classes, gmmparam, num_classes, train_sample, test_sample
from generate_data import X_train, Y_train, X_test, Y_test, X, labels
from mlp import TwolayerMLP

import model

# First conver test set data to tensor suitable for PyTorch models
X_test_tensor = torch.FloatTensor(X_test)
pr_error_list = []

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
# Estimate loss (probability of error) for each trained MLP model by testing on the test data set
print("Probability of error results summarized below per trained MLP: \n")
print("\t # of Training Samples \t Pr(error)")
for i in range(len(X_train)):
    # Evaluate the neural network on the test set
    predictions = model.predict_model(model.trained_models[i], X_test_tensor)
    # Compute the probability of error estimates
    prob_error = np.sum(predictions != Y_test) / len(Y_test)
    print("\t\t %d \t\t   %.3f" % (train_sample[i], prob_error))
    pr_error_list.append(prob_error)

plt.axhline(y=model.minimum_probability_error, color="black", linestyle="--", label="Min. Pr(error)")
ax.plot(np.log10(train_sample), pr_error_list)
ax.set_title("No. Training Samples vs Test Set Pr(error)")
ax.set_xlabel("MLP Classifier")
ax.set_ylabel("Pr(error)")


ax.legend()
plt.show()