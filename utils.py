'''
All source for numerical utility functions.

Sources: 
[1] Kornblith et al., Similarity of Neural Network Representations Revisited (2019)
'''

import numpy as np
import scipy.linalg as LA
import torch
from torch import nn

def centering_matrix(size: int=2) -> torch.tensor:
    # Returns the centering matrix Hn of size n.
    return torch.eye(size) - 1/size * torch.ones((size, size))

def HSIC(K: torch.tensor, L: torch.tensor):
    # Returns the empirical estimator of the Hilbert-Schmidt
    # independence criterion. K and L are matrices 
    assert(K.size()[0] == L.size()[0], "Error: K and L do not have the same number of examples.")
    n = K.size()[0]
    H_n = centering_matrix(size=n).cuda()
    product = K @ H_n @ L @ H_n

    return 1/((n - 1)**2) * torch.trace(product)

def CKA(X: torch.tensor, Y: torch.tensor, kernel='linear'):
    # Returns the Centered Kernel Alignment similarity metric
    # of feature matrices K, L (assumed to be passed through kernels)
    assert(X.size()[0] == Y.size()[0], "Error: X1 and X2 do not have the same number of examples.")

    # kernel: one str of {linear, rbf}

    if kernel == 'rbf':
        raise NotImplementedError()
    elif kernel == 'linear': # dot-product similarity of all examples 
        K = X @ X.T
        L = Y @ Y.T 
    KL = HSIC(K, L)
    KK = HSIC(K,K)
    LL = HSIC(L,L)

    return KL/torch.sqrt(KK * LL)