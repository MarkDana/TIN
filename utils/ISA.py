#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numba as nb

@nb.njit('(int_[:,::1], float64[:,::1], int_, int_)')
def get_pr(idx, r, mmax, n):
    samplesize, numvars = idx.shape
    res = np.zeros((mmax, numvars), dtype=np.float64)
    for i in range(samplesize):
        for j in range(numvars):
            res[idx[i, j] - 1, j] += (1 - r[i, j]) ** 2 / 2
            res[idx[i, j], j] += 0.5 + r[i, j] * (1 - r[i, j])
            res[idx[i, j] + 1, j] += r[i, j] ** 2 / 2
    return res / n

@nb.njit('(int_[:,::1], float64[:,::1], float64[:,::1], float64)')
def get_psi(idx, logp, r, bandwidth):
    samplesize, numvars = idx.shape
    res = np.zeros((numvars, samplesize), dtype=np.float64)
    for i in range(samplesize):
        for j in range(numvars):
            res[j, i] += logp[idx[i, j] - 1, j] * (1 - r[i, j]) + \
                         logp[idx[i, j], j] * (2 * r[i, j] - 1) - \
                         logp[idx[i, j] + 1, j] * r[i, j]
    return res / bandwidth

def scorecond(data):
    '''
    % Syntaxe [psi, entropy] = scorecond(data, bdwidth, cova)
    %
    % Estimate the conditional score function defined as minus the
    % gradient of the conditional density of of a random variable X_p
    % given x_{p-1}, \dots, x_{q+1}. The values of these random variables are
    % provided in the n x p array data.
    %
    % The estimator is based on the partial derivatives of the
    % conditional entropy with respect to the data values, the entropy
    % being estimated through the kernel density estimate _after a
    % prewhitening operation_, with the kernel being the density of the
    % sum of 3 independent uniform random variables in [.5,.5]. The kernel
    % bandwidth is set to bdwidth*(standard deviation) and the density is
    % evaluated at bandwidth apart. bdwidth defaults to
    %   2*(11*sqrt(pi)/20)^((p-q)/(p-q+4))*(4/(3*n))^(1/(p-q+4)
    % (n = sample size), which is optimal _for estimating a normal density_
    %
    % If cova (a p x p matrix) is present, it is used as the covariance
    % matrix of the data and the mean is assume 0. This prevents
    % recomputation of the mean and covariance if the data is centered
    % and/or prewhitenned.
    %
    % The score function is computed at the data points and returned in
    % psi.
    @param data: in shape (n, p), note that n is sample size, and p is #vars (different from `X' later)，
                 the p columns are calculated parallely
    @param q:
    @param bdwidth:
    @return: psi, entropy, where the results of p columns are returned parallely
    '''
    n, numvars = data.shape
    bdwidth = 2 * (11 * np.sqrt(np.pi) / 20) ** (1 / 5) * (4 / (3 * n)) ** (1 / 5)  # repeated calculated for many times though

    # prewhitening
    data = data - data.mean(axis=0)
    T = np.sqrt((data * data).mean(axis=0)) # in shape (p,), same as data.std(axis=0)
    data = data / T

    # # Grouping the data into cells, idx gives the index of the cell
    # # % containing a datum, r gives its relative distance to the leftmost
    # # % border of the cell
    r = data / bdwidth
    idx = np.floor(r).astype(int)
    r = r - idx
    idx = idx - idx.min(axis=0) + 1  # 0 <= idx-1
    # r, idx are in shape (n, numvars)

    # % Compute the probabilities at grid cells
    # % The kernel function is
    # %        1/2 + (1/2+u)(1/2-u) |u| <= 1/2
    # % k(u) = (3/2 - |u|)^2/2 1/2 <= |u| <= 3/2
    # %        0 otherwise
    # %
    # % The contribution to the probability at i-1, i, i+1 by a
    # % data point at distance r (0 <= r < 1) to i, are respectively:
    # % (1-r)^2/2, 1/2 + r*(1-r), r^2/2
    # % The derivative of the contribution to the probability at i-1, i, i+1
    # % by a data point at distance r to i are respectively: r-1, 1-2r, r
    #
    # % The array ker contains the contributions to the probability of cells
    # % The array kerp contains the gradient of these contributions
    # % The array ix contains the indexes of the cells, arranged in
    # % _lexicographic order_
    pr = get_pr(idx, r, idx.max() + 2, n)

    logp = np.log(pr, out=np.zeros_like(pr), where=(pr != 0))  # to contain log(cond. prob.)
    # entropy = np.log(bdwidth * T) - (pr * logp).sum(axis=0)  # in shape (numvars,)

    psi = get_psi(idx, logp, r, bdwidth)
    psi = psi - psi.mean(axis=1)[:, None]  # center psi, in shape (numvars, n)
    lam = (psi.T * data).sum(axis=0) / n - 1  # correction, lam in shape (numvars,)
    psi = ((psi.T - data * lam) / T).T

    return psi

def estim_beta_pham(x):
    '''
    @param x: data rows (k, T), k is numvars, T is sample size
    @return: beta in shape (k, T), the same shape as psi, and data
    '''
    return -1. * scorecond(np.copy(x.T, order='C'))

def adaptive_size(grad_new, grad_old, eta_old, z_old):
    alpha = 0 # 0.7
    up = 1.05 # 1.1 1.05
    down = 0.8 # 0.4 0.5 0.34 0.5
    z = grad_new + alpha * z_old
    etaup = (grad_new * grad_old) >= 0
    eta = eta_old * (up * etaup + down * (1 - etaup))
    eta[eta >= 0.03] = 0.03 # min(eta, 0.03)
    return eta, z

def natural_grad_Adasize_Mask_regu(X, Mask, regu):
    N, T = X.shape
    mu = 3e-3 # 3e-3 # original matlab code: 3e-3
    itmax = 5000 # 10000 #18000 # 18000
    Tol = 1e-6 # 1e-4, now smaller. otherwise early stopped
    num_edges = Mask.sum()

    # initilization of W
    WW = np.eye(N, N)
    for i in range(N):
        Ind_i = np.where(Mask[i] != 0)[0]
        X_Ind_i = X[Ind_i]
        WW[i, Ind_i] = -0.5 * (X[i] @ X_Ind_i.T) @ np.linalg.pinv(X_Ind_i @ X_Ind_i.T)
    W = 0.5 * (WW + WW.T)
    W = W + np.diag(1 - np.diag(W))

    z = np.zeros((N, N))
    eta = mu * np.ones_like(W)
    y_psi = np.zeros_like(X)
    y_psi0 = np.zeros_like(X)
    Grad_W_o = None

    init_avg_gradient_curve = []
    init_loss_curve = []

    for iter in range(itmax):
        # if iter % 100 == 0: print(f'natural_grad_Adasize_Mask_regu: ======== {iter}/{itmax} ========')
        y = W @ X
        argsort_y = np.argsort(y, axis=1)
        # update W: linear ICA with marginal score function estimated from data...
        if iter % 12 == 0:
            y_psi = np.copy(estim_beta_pham(y))
            y_psi0 = np.take_along_axis(y_psi, argsort_y, axis=1)
        else:
            y_psi[(np.tile(np.arange(N), (T, 1)).T, argsort_y)] = np.copy(y_psi0)
        ##################################################################

        # with regularization to make W small
        Grad_W_n = y_psi @ X.T / float(T) + np.linalg.inv(W.T) - 2 * regu * W
        if iter == 0: Grad_W_o = np.copy(Grad_W_n)
        eta, z = adaptive_size(Grad_W_n, Grad_W_o, eta, z)
        delta_W = eta * z
        W = W + delta_W * Mask

        avg_gradient = np.abs(Grad_W_n * Mask).sum() / num_edges
        init_avg_gradient_curve.append(avg_gradient)
        if avg_gradient < Tol: break

        Grad_W_o = np.copy(Grad_W_n)

    return W, np.array(init_avg_gradient_curve), np.array(init_loss_curve)

def sparseica_W_adasize_Alasso_mask_regu(lamda, Mask, X, regu):
    ''' ICA with SCAD penalized entries of the de-mixing matrix
    @param lamda: float, usually lamda is set to a constant times log(T), where T is sample size
    @param Mask: N*N 0 1 matrix, only updates the 1 entries on gradient
        in 2-step CD, if no mask, it's set to ones(N,N) - eye(N)
    @param X: data matrix in shape N*T, where N is the number of nodes, T is sample size. don't need to be whitened
    @param regu: float, e.g., 0.00, 0.002, 0.01, 0.05
    @return:
    '''
    N, T = X.shape
    XX = X - X.mean(axis=1)[:, None]
    # To avoid instability
    std_XX = XX.std(axis=1, ddof=1) # note that the std function in matlab is sample stddev, so ddof=1 here
    XX = np.diag(1. / std_XX) @ XX
    Refine = True
    num_edges = Mask.sum()

    # learning rate
    mu = 1e-6 # 1e-6
    beta = 0 # 1
    m = 60 # for approximate the derivative of |.|
    itmax = 0 # i.e., now we don't use penal in ISA for TIN. 8000 # 10000 # 15000 # 15000 # 10000
    Tol = 1e-6

    # initiliazation
    # print('Initialization....')
    WW, init_avg_gradient_curve, init_loss_curve = natural_grad_Adasize_Mask_regu(XX, Mask, regu)

    omega1 = 1. / np.abs(WW[Mask != 0])
    # to avoid instability
    Upper = 3 * omega1.mean()
    omega1[omega1 > Upper] = Upper
    omega = np.zeros((N, N))
    omega[Mask != 0] = omega1
    W = np.copy(WW)

    z = np.zeros((N, N))
    eta = mu * np.ones_like(W)
    W_old = W + np.eye(N)
    grad_new = np.copy(W_old)
    y_psi = np.zeros_like(XX)
    y_psi0 = np.zeros_like(XX)
    grad_old = None
    y = np.zeros_like(XX)

    penal_avg_gradient_curve = []
    penal_loss_curve = []

    # print('Penalization....')
    for iter in range(itmax):
        # if iter % 100 == 0: print(f'sparseica_W_adasize_Alasso_mask_regu: ======== {iter}/{itmax} ========')
        y = W @ XX
        avg_gradient = np.abs(grad_new * Mask).sum() / num_edges
        penal_avg_gradient_curve.append(avg_gradient)
        if avg_gradient < Tol:
            if Refine:
                Mask = np.abs(W) > 0.01
                Mask[np.diag_indices(N)] = 0
                lamda = 0.
                Refine = False
            else:
                break

        # update W: linear ICA with marginal score function estimated from data...
        argsort_y = np.argsort(y, axis=1)
        if iter % 8 == 0:
            y_psi = np.copy(estim_beta_pham(y))
            y_psi0 = np.take_along_axis(y_psi, argsort_y, axis=1)
        else:
            y_psi[(np.tile(np.arange(N), (T, 1)).T, argsort_y)] = np.copy(y_psi0)

        dev = omega * np.tanh(m * W)

        # with additional regularization
        regu_l1 = regu / 2.
        grad_new = y_psi @ XX.T / T + np.linalg.inv(W.T) - \
                     4 * beta * (np.diag(np.diag(y @ y.T / T)) - np.eye(N)) * (y @ XX.T / T) - \
                        dev * lamda / T - \
                            2 * regu_l1 * W # seems that it should be 0-mean-1-std XX here. in original code, it's X here?
        if iter == 0: grad_old = np.copy(grad_new)

        # adaptive size
        eta, z = adaptive_size(grad_new, grad_old, eta, z)
        delta_W = eta * z
        W = W + 0.9 * delta_W * Mask
        grad_old = np.copy(grad_new)

    # re-scaling
    W = np.diag(std_XX) @ W @ np.diag(1. / std_XX)
    WW = np.diag(std_XX) @ WW @ np.diag(1. / std_XX)    # WW is returned by initialization
    y = np.diag(std_XX) @ y
    Score = omega * np.abs(W)
    return y, W, WW, Score, \
           init_avg_gradient_curve, init_loss_curve, \
           np.array(penal_avg_gradient_curve), np.array(penal_loss_curve)

def ISA_ZY(data_Z, data_Y):
    '''
    @param data_Z: np array in shape (|Z|, T), where T is sample size
    @param data_Y: np array in shape (|Y|, T), where T is sample size
    @return: WY: the W subblock in shape (|Y|, |Y|)
    '''
    dimZ = data_Z.shape[0]
    X = np.vstack([data_Z, data_Y])
    N, T = X.shape
    regu = 0. # 0.05
    Mask = np.zeros((N, N))
    Mask[dimZ:, dimZ:] = 1

    y, W, WW, Score, \
    init_avg_gradient_curve, init_loss_curve, \
    penal_avg_gradient_curve, penal_loss_curve = sparseica_W_adasize_Alasso_mask_regu(np.log(T) * 4, Mask, X, regu)

    # W is after penalization, and WW is after initialization
    WYY = W[dimZ:, dimZ:]
    WWYY = WW[dimZ:, dimZ:]
    return WYY / np.linalg.norm(WYY, axis=1)[:, None], \
           WWYY / np.linalg.norm(WWYY, axis=1)[:, None]