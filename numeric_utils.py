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

    return torch.trace(A @ A.T) + torch.trace(B @ B.T) - 2 * torch.trace(S)

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

def cosine_similarity_matrix(features: torch.tensor) -> torch.tensor:
    '''
    features: a matrix (n_examples, feature_dim) of feature vectors
    returns cosine similarity matrix where S[i,j] = feature_i . feature_j / (||f_i|| * ||f_j||)
    '''
    n = torch.linalg.norm(features, dim=1).unsqueeze(1)
    nT = n.T
    sim = features @ features.T
    sim = sim * 1/n # * = hadamard product
    return sim * 1/nT

def avg_similarity(S: torch.tensor) -> torch.tensor:
    '''
    returns avg. entry of (S - tr(S))/ 2
    average of all off-diagonal entries (cosine distances from 
    '''
    n = S.size()[0]
    upperTri = torch.triu(S, diagonal=1).cuda() # all cosine similarities that aren't self to self
    avg = torch.sum(upperTri) / ((n**2 - n)/2) # number of off-diagonal elements divided by 2
    return avg

def get_avg_similarity_per_example(representations: torch.tensor) -> torch.tensor:
    '''
    representations: array of shape (num_mc_samples, num_examples, rep_dim)
    returns the average similarity per example
    '''
    n_examples = representations.size()[1]
    similarities = torch.empty(size=(n_examples,)).cuda()
    for i in range(n_examples):
        example_features = torch.squeeze(representations[:,i,:])
        cosine_sims = cosine_similarity_matrix(example_features)
        similarities[i] = avg_similarity(cosine_sims)
    return similarities

def getIntermediateActivation(name, activation_dict):
    '''
    name: intermediate layer name in nn.Module.modules()
    activation_dict: output dictionary
    Return the activation of an intermediate layer using
        nn.module.register_forward_hook
    '''
    def hook(model, input, output):
        activation_dict[name] = output
    return hook
