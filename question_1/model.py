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
class_condition_likelihoods = np.array([mvn.pdf(X_test, gmmparam['meanvectors'][i], gmmparam['covariancematrices'][i]) for i in range(classes)])
decisions = np.argmax(class_condition_likelihoods, axis=0)
wrong_samples = sum(decisions != Y_test)
minimum_probability_error = (wrong_samples / test_sample)
print("Probability of Error on Test Set using the true Data pdf = {}".format(minimum_probability_error))

def train_model(model, data, labels, optimizer, criterion=nn.CrossEntropyLoss(), num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, loss

def predict_model(model, data):
    model.eval()
    with torch.no_grad():
        predicted_labels = model(data)
        predicted_labels = predicted_labels.detach().numpy()
        return np.argmax(predicted_labels, 1)

def k_fold_cv_perceptrons(K, P_list, data, labels):
    """
    Performs k-fold cross-validation to select the optimal number of perceptrons for a two-layer MLP
    model based on the minimum validation error.

    Args:
    K (int): The number of folds for k-fold cross-validation.
    P_list (list): A list of integers representing the number of perceptrons to test.
    data (numpy.ndarray): The dataset features.
    labels (numpy.ndarray): The dataset labels.

    Returns:
    optimal_P (int): The optimal number of perceptrons.
    error_valid_m (numpy.ndarray): The mean validation error across K folds for each value of P.
    """
    kf = KFold(n_splits=K, shuffle=True)

    error_valid_mk = np.zeros((len(P_list), K))

    for p_idx, p in enumerate(P_list):
        for fold_idx, (train_indices, valid_indices) in enumerate(kf.split(data)):
            X_train, y_train = torch.FloatTensor(data[train_indices]), torch.LongTensor(labels[train_indices])
            X_valid, y_valid = torch.FloatTensor(data[valid_indices]), labels[valid_indices]

            model = TwolayerMLP(X_train.shape[1], p, classes)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            model, _ = train_model(model, X_train, y_train, optimizer)

            predictions = predict_model(model, X_valid)
            error_valid_mk[p_idx, fold_idx] = np.sum(predictions != y_valid) / len(y_valid)

    error_valid_m = np.mean(error_valid_mk, axis=1)
    optimal_P = P_list[np.argmin(error_valid_m)]

    return optimal_P, error_valid_m


K = 10
P_list = [2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512]
best_P_list = []

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
print("\t# of Training Samples \tBest # of Perceptrons \tPr(error)")
for i in range(len(X_train)):
    P_best, P_CV_err = k_fold_cv_perceptrons(K, P_list, X_train[i], Y_train[i])
    best_P_list.append(P_best)
    print("\t\t %d \t\t\t %d \t\t  %.3f" % (train_sample[i], P_best, np.min(P_CV_err)))
    ax.plot(P_list, P_CV_err, label="Sample_test = {}".format(train_sample[i]))

plt.axhline(y=minimum_probability_error, color="black", linestyle="--", label="Min. Pr(error)")
ax.set_title("No. Perceptrons vs Cross-Validation Pr(error)")
ax.set_xlabel(r"$P$")
ax.set_ylabel("Pr(error)")
ax.legend()
plt.show()


trained_models = []
num_restarts = 10
for i in range(len(X_train)):
    print("Training model for N = {}".format(X_train[i].shape[0]))
    X_i = torch.FloatTensor(X_train[i])
    y_i = torch.LongTensor(Y_train[i])
    best_models = []
    best_losses = []
    # Remove chances of falling into suboptimal local minima
    for r in range(num_restarts):
        model = TwolayerMLP(X_i.shape[1], best_P_list[i], classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # Trained model
        model, loss = train_model(model, X_i, y_i, optimizer)
        best_models.append(model)
        best_losses.append(loss.detach().item())

    # Add best model from multiple restarts to list
    trained_models.append(best_models[np.argmin(best_losses)])
