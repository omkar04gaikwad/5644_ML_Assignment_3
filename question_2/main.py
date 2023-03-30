import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import warnings
from variable import true_gmm_param, n_experiments, n_samples, n_splits, model_orders, freq, generate_data
warnings.filterwarnings("ignore")

def kfold_validation(freq):
    for exp in range(n_experiments):
        for i, n in enumerate(n_samples):
            X = generate_data(n)
            cv_scores = np.zeros((n_splits, len(model_orders)))
            kf = KFold(n_splits=n_splits)
            for j, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                for k, order in enumerate(model_orders):
                    gmm = GaussianMixture(n_components=order, covariance_type='full')
                    gmm.fit(X_train)
                    cv_scores[j, k] = gmm.score(X_test)
            freq[i] = np.bincount(np.argmax(cv_scores, axis=1), minlength=len(model_orders))
        avg_freq = freq / ((exp+1)*n_splits)
        print(f"Frequencies for experiment = {exp+1}")
        for i, n in enumerate(n_samples):
            print(f"n_samples={n}")
            for j, order in enumerate(model_orders):
                print(f"  order = {order} : Frequency = {avg_freq[i,j]:.4f}")
    freq /= (n_experiments * n_splits)
    return freq

def plot_bar(freq):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    width = 0.1
    x = np.arange(len(model_orders))
    for i, n in enumerate(n_samples):
        ax.bar(x+i*width, freq[i], width, label=f"{n} samples")
    ax.set_xticks(x+2*width)
    ax.set_xticklabels(model_orders)
    ax.set_xlabel('Model Order')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.show()
freqs = kfold_validation(freq)
plot_bar(freqs)