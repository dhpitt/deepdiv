'''
All source for numerical utility functions.

Sources: 
[1] Kornblith et al., Similarity of Neural Network Representations Revisited (2019)
[2] Ding et al., Grounding Representation Similarity with Statistical Testing (2021)
'''

from typing import List

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

def OrthogonalProcrustes(A: torch.tensor, B: torch.tensor):
    '''
    OP similarity = ||A||_F^2 + ||B||_F^2 - 2 ||AtB||_*

    A, B: tensors with same number of rows (examples)
    returns the Orthogonal Procrustes matrix, which is a robust measure of similarity
        according to [2]
    '''
    assert (A.size()[0] == B.size()[0], "Numbers of examples don't match!")

    _, S, _ = torch.linalg.svd(A.T @ B)

    return torch.trace(A * A.T) + torch.trace(B @ B.T) - 2 * torch.trace(S)

def pairwise_CKA(representations: List[torch.tensor]) -> torch.tensor:
    '''
    representations: list of tensors of dim (n_examples, rep_dim)
    returns the sum of CKA applied to each pair of representations.
    '''
    n = len(representations)
    total = 0 
    for i in range(n):
        for j in range(i+1, n):
            total += CKA(representations[i], representations[j])
    return total

def pairwise_OP(representations: List[torch.tensor]) -> torch.tensor:
    '''
    representations: list of tensors of dim (n_examples, rep_dim)
    returns the sum of Orthogonal Procrustes similarity applied to each pair of representations.
    '''
    n = len(representations)
    total = 0
    for i in range(n):
        for j in range(i+1, n):
            total += OrthogonalProcrustes(representations[i], representations[j])
    return total

def pairwise_inner_product(representations: List[torch.tensor]) -> torch.tensor:
    '''
    representations: list of tensors of dim (n_examples, rep_dim)
    returns the matrix S (n_examples, n_examples), where S[i,j] is the sum of
        all inner product similarities 
    '''
    n = len(representations)
    S = torch.zeros(size=representations[0].size())
    for i in range(n):
        for j in range(i+1, n):
            S += representations[i] @ representations[j].T
    return S

# def query_by_committee(preds: torch.tensor) -> torch.tensor:
#     '''
#     preds: a set of predictions of shape (n_models, n_examples)
#         where preds[i, :] is a vector of n_examples predictions
#     returns the most-voted-for example among the committee of models for each example.
#     '''