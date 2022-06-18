#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from causallearn.search.FCMBased.lingam.hsic import hsic_test_gamma
from utils.ISA import ISA_ZY
from utils.graph import DiGraph
from utils.misc import *

class Solver(object):
    def __init__(self, datadir='./data/NSR_3/1111111111_0'):
        self.datadir = datadir
        self.observ_data = np.load(os.path.join(datadir, 'noise_data', 'observ_data.npy'))
        self.nodenum = self.observ_data.shape[0]
        self.truth_B = np.load(os.path.join(datadir, 'graph_param', 'B.npy'))
        self.truth_A = np.load(os.path.join(datadir, 'graph_param', 'A.npy'))
        core_A_01 = (self.truth_A != 0).astype(int)
        self.truth_graph = DiGraph(generatedAdjmat=core_A_01.T)
        self.observ_data_2D_cumulants = np.load(
            os.path.join(datadir, 'cumulants', 'observ_data_2D_cumulants_empirical.npy'))

        self.rank_eps = 0.004
        self.pval_alpha = 0.05
        self.svrt_alpha = 0.85
        self.distance_alpha = 0.10

    def TIN_2steps(self, Z, Y):
        data_Z, data_Y = self.observ_data[Z], self.observ_data[Y]

        raw_s = np.ones((self.nodenum - 1, len(Y))) * -1 # use -1 for padding
        raw_vh = np.zeros((self.nodenum - 1, len(Y), len(Y)))
        pvalues = np.zeros((self.nodenum - 1, len(Y), len(Z)))
        severities = np.zeros((self.nodenum - 1, len(Y), len(Z)))
        # each row is the order k's |Y| omega solutions' indpendence to Z
        # note that it is **all the |Y| solutions**, not only the the rank ones

        cov_matrices = []
        for orderk in range(2, self.nodenum + 1):
            cov_matrices.append(get_nonlinear_covariances(data_Z, data_Y, orderk))
            stacked_Psi = np.vstack(cov_matrices)
            u, s, vh = np.linalg.svd(stacked_Psi, full_matrices=True)
            wTYs = vh @ data_Y
            raw_s[orderk - 2, :len(s)] = np.copy(s)
            raw_vh[orderk - 2] = np.copy(vh)
            for omegaid, wTYrow in enumerate(wTYs):
                hsic_stats = [hsic_test_gamma(data_Z_row, wTYrow) for data_Z_row in data_Z]
                svrts, pvals = zip(*hsic_stats)
                pvalues[orderk - 2, omegaid] = np.array(pvals)
                severities[orderk - 2, omegaid] = np.array(svrts) / data_Z.shape[1] * 1000
        return raw_s, raw_vh, pvalues, severities

    def TIN_subsets(self, Z, Y):
        def _exists_independence(Z, Yprime):
            _, _, pvalues, _ = self.TIN_2steps(Z, Yprime)
            min_pvals = np.min(pvalues, axis=2) # for each w, the minimum independence achieved by wTY and every z in Z
            print(min_pvals)
            return np.max(min_pvals) # use max for "exists"

        for tin_k in range(len(Y)):
            cond1 = np.percentile([_exists_independence(Z, list(Yprime)) for Yprime in combinations(Y, tin_k + 1)], 10) > self.pval_alpha
            # or tin_k == len(Y)
            cond2 = tin_k == 0 or \
                np.percentile([_exists_independence(Z, list(Yprime)) for Yprime in combinations(Y, tin_k)], 90) < self.pval_alpha
            if cond1 and cond2:
                # a more accurate version (instead of hardcode cond1 and cond2) is implemented by decision tree with input of pvalues
                return tin_k
        return len(Y)

    def TIN_ISA(self, Z, Y):
        def _get_true_Omega(Z, Y):
            AncZ = list(self.truth_graph.getAncestorsOfSet(set(Z)))
            return null_space(self.truth_B[Y][:, AncZ].T)

        def _process_mixing_matrix(Z, Y, WYY):
            ns = _get_true_Omega(Z, Y)
            distance_to_Omega_ZY = []
            for wrow in WYY:  # wrow is vector with norm 1
                coefs = wrow.T @ ns
                projection = ns @ coefs.T
                distance_to_Omega_ZY.append(np.linalg.norm(projection - wrow))

            whitened_data = self.observ_data
            whitened_data = whitened_data - whitened_data.mean(axis=1)[:, None]
            std_XX = whitened_data.std(axis=1, ddof=1)
            whitened_data = np.diag(1. / std_XX) @ whitened_data
            data_Z, data_Y = whitened_data[Z], whitened_data[Y]

            wTY_Z_hsic_pvalues = []
            wTY_Z_hsic_severities = []
            wTYs = WYY @ data_Y  # test each row: whether it's an w that satisfies wTY ind. Z
            for rid, wTYrow in enumerate(wTYs):
                hsic_stats = [hsic_test_gamma(data_Z_row, wTYrow) for data_Z_row in data_Z]
                svrts, pvals = zip(*hsic_stats)
                wTY_Z_hsic_pvalues.append(pvals)
                wTY_Z_hsic_severities.append(tuple(svt / data_Z.shape[1] * 1000 for svt in svrts))
            return {
                'distance_to_Omega_ZY': distance_to_Omega_ZY,
                'wTY_Z_hsic_pvalues': wTY_Z_hsic_pvalues,
                'wTY_Z_hsic_severities': wTY_Z_hsic_severities
            }

        data_Z, data_Y = self.observ_data[Z], self.observ_data[Y]
        _, W_init = ISA_ZY(data_Z, data_Y)
        init_stats = _process_mixing_matrix(Z, Y, W_init)
        return init_stats

    def TIN_rank(self, Z, Y):
        def _get_stacked_Psi(orderk, Z, Y):
            return np.vstack([self.observ_data_2D_cumulants[kid - 2][Z][:, Y] for kid in range(2, orderk + 1)])

        def _find_first_stop_value(ranklist):
            for i in range(1, len(ranklist)):
                if ranklist[i] == ranklist[i - 1]:
                    return ranklist[i]
            return ranklist[-1]

        ranklist = [] # processed by the epsilon e.g., 1e-3, not fully finetuned
        raw_s = np.ones((self.nodenum - 1, len(Y))) * -1 # use -1 for padding
        for orderk in range(2, self.nodenum + 1):
            stacked_Psi = _get_stacked_Psi(orderk, Z, Y)
            u, s, vh = np.linalg.svd(stacked_Psi, full_matrices=True)
            s_trhd = np.max(s) * max(stacked_Psi.shape) * self.rank_eps
            rank = int((s > s_trhd).sum())
            ranklist.append(rank)
            raw_s[orderk - 2, :len(s)] = np.copy(s)
        return _find_first_stop_value(ranklist)


if __name__ == '__main__':
    nodenum = 5
    slvr = Solver(datadir='./data/NSR_3/1111111111_0')
    for z in range(5):
        Z = [z]
        Y = [_ for _ in range(5) if _ != z]
        print(z, slvr.TIN_ISA(Z, Y))