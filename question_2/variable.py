import numpy as np
import warnings
warnings.filterwarnings("ignore")

true_gmm_param = {
    'means': np.array([[0, 0], [2, 2], [0, 4], [4, 0]]),
    'covariancematrices':np.array([[[1, 0], [0, 1]], 
                 [[1, 0.5], [0.5, 1]], 
                 [[1, -0.7], [-0.5, 1]], 
                 [[1, 0], [0, 1]]]),
    'weights': np.array([0.2, 0.3, 0.1, 0.4])
}

n_experiments = 30
n_splits = 10
n_samples = [10, 100, 1000, 10000]
model_orders = [1, 2, 3, 4, 5, 6]

freq = np.zeros((len(n_samples), len(model_orders)))

def generate_data(n_samples):
    data_1 = []
    for k in range(len(true_gmm_param['weights'])):
        samples = np.random.multivariate_normal(true_gmm_param['means'][k], true_gmm_param['covariancematrices'][k], int(n_samples * true_gmm_param['weights'][k]))
        data_1.append(samples)
    data = np.vstack(data_1)
    return data
