import numpy as np

samples = 10000
classes = 4

gmmparam = {
    'priors': np.ones(classes) / classes,
    'meanvectors': np.array([[0, 0, 0],
               [0, 1, 1],
               [1, 0, 1],
               [1, 1, 0]]),
    'covariancematrices': np.array([[[1.5, 0.3, 0.3],
                   [0.3, 1, 0.5],
                   [0.3, 0.5, 1]],
                  
                  [[1.5, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]],
                  
                  [[1, 0.2, 0.2],
                   [0.2, 1, 0],
                   [0.2, 0, 1]],
                  
                  [[1, 0, 0.2],
                   [0, 1.5, 0],
                   [0.2, 0, 1.5]]])
}

num_classes = len(gmmparam['priors'])

train_sample = [100, 200, 500, 1000, 2000, 5000]

test_sample = 10000

