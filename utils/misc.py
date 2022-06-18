#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from itertools import chain, combinations
from collections import Counter
from scipy.linalg import null_space
import scipy.stats as stats

NODENUM_TO_LEN = {n: (n * (n - 1)) // 2 for n in range(100)}
LEN_TO_NODENUM = {v: k for k, v in NODENUM_TO_LEN.items()}

def normalised_kendall_tau_distance(group_order1, group_order2):
    '''
    extended kendall_tau_distance to group
    please see https://en.wikipedia.org/wiki/Kendall_tau_distance for details
    @param group_order1: list of lists, e.g., [[0], [1, 2], [3, 4]], items of inner lists are node names
    @param group_order2: list of lists, e.g., [[0, 1], [2, 3, 4]], items of inner lists are node names, should be same set as 1's
    @return:
    '''
    def _sign(id1, id2):
        if id1 < id2: return -1
        elif id1 == id2: return 0
        return 1
    nodes1_set = {x for xx in group_order1 for x in xx}
    nodes2_set = {x for xx in group_order2 for x in xx}
    assert nodes1_set == nodes2_set, "should be group ordering of a same variables set"
    n = len(nodes1_set)
    value1_find_id = {v: k for k, onegroup in enumerate(group_order1) for v in onegroup}
    value2_find_id = {v: k for k, onegroup in enumerate(group_order2) for v in onegroup}
    ndisordered = 0
    for nodei, nodej in combinations(nodes1_set, 2):
        nodei_grp1_id, nodej_grp1_id = value1_find_id[nodei], value1_find_id[nodej]
        nodei_grp2_id, nodej_grp2_id = value2_find_id[nodei], value2_find_id[nodej]
        if _sign(nodei_grp1_id, nodej_grp1_id) != _sign(nodei_grp2_id, nodej_grp2_id):
            ndisordered += 1
    return 2 * ndisordered / (n * (n - 1))

def _id_order_to_ordered_groups(id_to_order_dict):
    order_to_ids = {}
    for id, order in id_to_order_dict.items():
        if order in order_to_ids:
            order_to_ids[order].append(id)
        else:
            order_to_ids[order] = [id]
    return [order_to_ids[order] for order in sorted(order_to_ids.keys())]

def gen_truncated_gaussian(mu, sigma, lower, upper, numsamples):
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return X.rvs(numsamples)

def standard_scaler(X, stdval=1.):
    return stdval * (X - X.mean()) / X.std()

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def get_rank(mat, eps):
    if mat.size == 0: return 0
    u, s, vh = np.linalg.svd(mat)
    s_trhd = np.max(s) * max(mat.shape) * eps
    return (s > s_trhd).sum()

def get_null_space(A):
    '''
    :param A: 2d numpy array. Solve the homogeneous linear equations Ax=0. dim(x)=#cols(A)=n
        rank(A)<n <=> has infinitely many nonzero solution (if A square, <=> det(A)==0)
        rank(A)==n <=> has only zero solution (if A square, <=> det(A)!=0)
    :return null_space(A): return in shape (n, n-rank(A))  //if A full column rank return in shape (n, 0)
        columns of null_space(A) is set of orthonormal basis for the null space of A
            : Z=null_space(A); np.allclose(Z.T.dot(Z), np.eye(Z.shape[1]))
        any linear combination of columns of null_space(A) is a solution for Ax=0
            : np.allclose(A.dot(Z), 0)
    '''
    return null_space(A)

def partition(collection):
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        # put `first` in its own subset
        yield [[first]] + smaller

def get_nonlinear_covariances(data_Z, data_Y, orderk):
    orkerk_to_func = {
        2: lambda z: z,
        3: lambda z: z ** 2,
        4: lambda z: z ** 3,
        5: lambda z: np.abs(z),
        6: lambda z: np.exp(z),
        7: lambda z: np.log(np.abs(z)),
        8: lambda z: np.sin(z),
        9: lambda z: np.cos(z),
        10: lambda z: 1 / (1 + np.exp(-z)),
        11: lambda z: np.tanh(z)
    }
    data_Z_transformed = orkerk_to_func[orderk](data_Z)
    return data_Z_transformed.dot(data_Y.T) / data_Z.shape[1] - \
           data_Z_transformed.mean(axis=1)[:,None].dot(data_Y.mean(axis=1)[None,:])
