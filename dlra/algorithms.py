import numpy as np
from numpy.linalg import lstsq, solve
import warnings
from mscode.methods.algorithms import ista, homp, iht_mix, ista_nn
from mscode.methods.proxs import SoftT, HardT
from matplotlib import pyplot as plt

import tensorly as tl
from tensorly.random import random_cp
from tensorly.base import unfold
from tensorly.tenalg import khatri_rao, mode_dot
from tensorly.cp_tensor import (cp_to_tensor, CPTensor,
                         unfolding_dot_khatri_rao, cp_norm,
                         cp_normalize, validate_cp_rank)
from tensorly.decomposition._cp import initialize_cp, error_calc
from tensorly.tenalg.proximal import hals_nnls
import copy


# Adapting tensorly's ALS for DCPD or DMF


def palm_Dparafac(tensor, rank, D, k, step_rel=0.5, n_iter_max=100, init='svd', X0=None, tol=1e-8, random_state=None,
            verbose=0, return_errors=False,
            cvg_criterion='abs_rec_error', nonnegative=False, accelerate=True):
    """Dictionary-based Parafac Decomposition via PALM
    Computes a rank-`rank` decomposition of `tensor` such that::

        tensor = [|weights; factors[0], ..., factors[-1] |].

    with factors[0] = DX, for input dictionary D and columnwise k-sparse X.
    This solves approximately the following problem

        :math:`minimize\; \\frac{1}{2} \\|tensor - DX(B\\odot C)^T\\|_F^2\; wrt\; X, B, C \; s.t. \; \\|X_i\\|_0\\leq k`

    Constrained mode is necessarily the first one in this implementation.

    Parameters
    ----------
    tensor : ndarray
        Input data
    rank  : int
        Number of components.
    D : ndarray
        Dictionary for the first mode
    k : int
        Columnwise sparsity level for the first mode
    lamb_rel: float in [0,1]. Default: 0.1
        Amount of regularization for MSC Fista solver.
    step_rel : float in [0,1]. Default: 0.1
        Percentage of theoretical largest stepsize actually used. Set smaller than 1 if divergence occurs.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random', CPTensor}, optional
        Type of factor matrix initialization.
        If a CPTensor is passed, this is directly used for initalization.
        See `initialize_factors`.
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
        of shape (rank, ), which will contain the norms of the factors
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found a local minimum when the
        reconstruction error is less than `tol`.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
       Stopping criterion for ALS, works if `tol` is not None. If 'rec_error',  ALS stops at current iteration if ``(previous rec_error - current rec_error) < tol``. If 'abs_rec_error', ALS terminates when `|previous rec_error - current rec_error| < tol`.
    accelerate : boolean, default True
        If true, uses iPALM instead of PALM.

    Returns
    -------
    CPTensor : (weight, factors)
        Estimated DCPD tensor in cp_tensor tensorly format.

    X : ndarray
        The sparse coefficients of the first factor.

    S : numpy array
        Support of X.

    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    """

    # initial error
    weights, factors = initialize_cp(tensor, rank, init=init, svd='numpy_svd',
                                 random_state=random_state)

    if X0 is None:
        if nonnegative:
             X = np.random.rand(D.shape[1],rank)
        else:
             X = np.random.randn(D.shape[1],rank)
    else:
        X = X0

    # Precomputations
    #DtY = mode_dot(tensor, D.T, 0)
    DtD = D.T@D
    Dttensor = mode_dot(tensor,D.T,0)
    #Y = unfold(tensor,0)

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    beta=0
    X_old = np.copy(X)
    factors_old = copy.deepcopy(factors)

    for iteration in range(n_iter_max):

        if verbose > 1:
            print("Starting iteration", iteration + 1)

        if accelerate:
            beta = (iteration-1)/(iteration+2)

        # Updating X
        mttkrp = unfolding_dot_khatri_rao(Dttensor, (None,factors), 0)
        pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
        for i, factor in enumerate(factors):
            if i != 0:
                pseudo_inverse = pseudo_inverse * tl.dot(tl.transpose(factor), factor)

        # Stepsize computation, following PALM
        eta = step_rel/(tl.norm(DtD)*tl.norm(pseudo_inverse))

        Z = X + beta*(X - X_old)
        X_old = np.copy(X)
        X = HardT(Z - eta * (DtD@Z@pseudo_inverse - mttkrp), k)
        factors[0] = D@X

        # Updating other factors
        for mode in range(1,tensor.ndim):
            if verbose > 1:
                print("Mode", mode, "of", tl.ndim(tensor))

            pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
            for i, factor in enumerate(factors):
                #if i==0:
                #    pseudo_inverse = pseudo_inverse * tl.dot(tl.dot(X.T, DtD), X)
                if i != mode:
                    pseudo_inverse = pseudo_inverse * tl.dot(tl.transpose(factor), factor)

            # Concatenating X in front instead of the first factor, to use precomputed DtY
            #factors_mod = [X] + factors[1:]
            mttkrp = unfolding_dot_khatri_rao(tensor, (None, factors), mode)

            # Stepsize computation following PALM and acceleration
            eta = step_rel/tl.norm(pseudo_inverse)
            Z = factors[mode] + beta*(factors[mode] - factors_old[mode])
            factors_old[mode] = np.copy(factors[mode])

            if nonnegative:
                factors[mode] = tl.clip(Z - eta * (Z@pseudo_inverse - mttkrp), 0)
            else:
                factors[mode] = Z - eta * (Z@pseudo_inverse - mttkrp)

        # Calculate the current unnormalized error if we need it
        if (tol or return_errors):
            unnorml_rec_error, tensor, norm_tensor = error_calc(tensor, norm_tensor, weights, factors, 0, None, mttkrp)

        rec_error = unnorml_rec_error / norm_tensor
        rec_errors.append(rec_error)

        if tol:

            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if verbose:
                    print("iteration {}, reconstruction error: {}, decrease = {}, unnormalized = {}".format(iteration, rec_error, rec_error_decrease, unnorml_rec_error))

                if cvg_criterion == 'abs_rec_error':
                    stop_flag = abs(rec_error_decrease) < tol
                elif cvg_criterion == 'rec_error':
                    stop_flag = rec_error_decrease < tol
                else:
                    raise TypeError("Unknown convergence criterion")

                if stop_flag:
                    if verbose:
                        print("PARAFAC converged after {} iterations".format(iteration))
                    break

            else:
                if verbose:
                    print('reconstruction error={}'.format(rec_errors[-1]))

    cp_tensor = CPTensor((weights, factors))

    # Estimating support (np arrays with supports)
    Sx = np.argsort(np.abs(X), 0)
    # truncating the support
    Sx = Sx[-k:, :]

    if return_errors:
        return cp_tensor, X, Sx, rec_errors
    else:
        return cp_tensor, X, Sx

def dlra_parafac(tensor, rank, D, k, lamb_rel=0.01, n_iter_max=100, init='svd', X0=None, tol=1e-8, random_state=None,
            verbose=0, return_errors=False,
            cvg_criterion='abs_rec_error',
            fixed_modes=None, inner_iter_max=100, nonnegative=False, tau=20, itermax_calib=10):
    """Dictionary-based CANDECOMP/PARAFAC decomposition via alternating least squares (ALS) and Mixed Sparse Coding (MSC)
    Computes a rank-`rank` decomposition of `tensor` such that::

        tensor = [|weights; factors[0], ..., factors[-1] |].

    with factors[0] = DX, for input dictionary D and columnwise k-sparse X.
    This solves approximately the following problem

        :math:`minimize\; \\frac{1}{2} \\|tensor - DX(B\\odot C)^T\\|_F^2\; wrt\; X, B, C \; s.t. \; \\|X_i\\|_0\\leq k`

    Constrained mode is necessarily the first one in this implementation.

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    D : list of ndarray
        Dictionary for the first mode, or dictionary modes. E.g. [D1, D2] constraints modes 0 and 1.
    k : list of int
        Columnwise sparsity level for the first mode or for dictionary modes
    lamb_rel: list of float in [0,1]. Default: 0.1
        Amount of regularization for MSC Fista solver.
    n_iter_max : int
        Maximum number of
    X0 : list of arrays
        Initial values for the sparse coefficients.
    init : {'svd', 'random', CPTensor}, optional
        Type of factor matrix initialization.
        If a CPTensor is passed, this is directly used for initalization.
        See `initialize_factors`.
    tol : float, optional
        (Default: 1e-8) Relative reconstruction error tolerance. The
        algorithm is considered to have found a local minimum when the
        reconstruction error is less than `tol`.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
       Stopping criterion for ALS, works if `tol` is not None.
       If 'rec_error',  ALS stops at current iteration if ``(previous rec_error - current rec_error) < tol``.
       If 'abs_rec_error', ALS terminates when `|previous rec_error - current rec_error| < tol`.
    fixed_modes : list, default is None
        A list of modes for which the initial value is not modified.
        The last mode cannot be fixed due to error computation.
    nonnegative: bool
        Set to True to have nn constraints everywhere, with nnls solver HALS as implemented in Tensorly.

    Returns
    -------
    CPTensor : (weight, factors)
        Estimated DCPD tensor in cp_tensor tensorly format.

    X : numpy array
        The sparse coefficients of the first factor.

    S : numpy array
        Support of X

    errors : list
        A list of reconstruction errors at each iteration of the algorithms.
    """
    weights, factors = initialize_cp(tensor, rank, init=init, svd='numpy_svd',
                                 random_state=random_state)

    nbr_dic = len(D)

    if X0 is None:
        if nonnegative:
            X = [np.random.rand(D[l].shape[1],rank) for l in range(nbr_dic)]
        else:
            X = [np.random.randn(D[l].shape[1],rank) for l in range(nbr_dic)]
    else:
        X = X0

    Sx = [None for l in range(nbr_dic)]

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    if fixed_modes is None:
        fixed_modes = []

    if tl.ndim(tensor)-1 in fixed_modes:
        warnings.warn('You asked for fixing the last mode, which is not supported.\n The last mode will not be fixed. Consider using tl.moveaxis()')
        fixed_modes.remove(tl.ndim(tensor)-1)
    modes_list = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]

    # Precomputations
    DtY = [mode_dot(tensor, D[l].T, l) for l in range(nbr_dic)]
    DtD = [D[l].T@D[l] for l in range(nbr_dic)]
    Y = [unfold(tensor,l) for l in range(nbr_dic)]
    # Turning lamb_rel into a list
    lamb_rel = [lamb_rel[l]*np.ones(rank) for l in range(nbr_dic)]
    # to compute stepsizes
    singvalD = [np.linalg.svd(DtD[l])[1][0] for l in range(nbr_dic)]
    # store best
    err_best = np.Inf

    for iteration in range(n_iter_max):

        if verbose > 1:
            print("Starting iteration", iteration + 1)
        for mode in modes_list:
            if verbose > 1:
                print("Mode", mode, "of", tl.ndim(tensor))

            pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse = pseudo_inverse * tl.dot(tl.transpose(factor), factor)

            if mode>nbr_dic-1:
                mttkrp = unfolding_dot_khatri_rao(tensor, (None, factors), mode)
                if nonnegative:
                    factor = hals_nnls(mttkrp.T, pseudo_inverse, factors[mode].T, n_iter_max=100)[0]
                    factor = factor.T
                else:
                    factor = tl.transpose(tl.solve(tl.transpose(pseudo_inverse), tl.transpose(mttkrp)))

            else:
                # BtB is pseudo-inverse
                singvalB = np.linalg.svd(pseudo_inverse)[1][0]
                eta = 1/singvalD[mode]/singvalB
                # first mode update using a Mixed Sparse Coding solver
                # suboptimal :( rewrite ista for innerproducts
                mttkrp = unfolding_dot_khatri_rao(DtY[mode], (None,factors),mode)
                B = khatri_rao(factors, skip_matrix=mode)

                # Tuning lambda
                sparsity_balance=np.zeros(rank)
                iter_calib=0

                # First ISTA without final projection
                while np.sum(sparsity_balance)<rank and iter_calib<itermax_calib:
                    nz=[]
                    iter_calib+=1
                    if nonnegative:
                        X[mode],_,rec,_,X_old = ista_nn(Y[mode], D[mode], B, lamb_rel[mode], k=None, itermax=1000, verbose=False, X0=X[mode], DtD=DtD[mode], DtYB=mttkrp, BtB=pseudo_inverse, eta=eta, tol=1e-8, return_old=True)
                    else:
                        X[mode],_,rec,_,X_old = ista(Y[mode], D[mode], B, lamb_rel[mode], k=None, itermax=1000, verbose=False, X0=X[mode], DtD=DtD[mode], DtYB=mttkrp, BtB=pseudo_inverse, eta=eta, tol=1e-8, return_old=True)

                    # Automatic lambda tuning
                    for i in range(rank):
                        # first we get the nnz numbers
                        nz.append(np.sum(np.abs(X_old[:,i])>0))

                    for i in range(rank):
                        # then we increase lambda if needed, or decrease if needed
                        # tolerance on k is [k, k+10]
                        # We prefear that lambda is slightly too small to avoid picking zeroes arbitrarily in the unbiaising LS step
                        if nz[i]<k[mode]:
                            sparsity_balance[i]=0
                            lamb_rel[mode][i] = lamb_rel[mode][i]/1.3
                        elif nz[i]>(k[mode]+tau):
                            #elif nz[i]>(2*k):
                            sparsity_balance[i]=0
                            #lamb_rel[mode][i] = lamb_rel[mode][i]*1.01
                            # TODO Test
                            lamb_rel[mode][i] = np.minimum(lamb_rel[mode][i]*1.01,1)
                        else:
                            sparsity_balance[i]=1
                    #print(nz)
                    #print(lamb_rel)

                # Now the real ista
                if nonnegative:
                    X[mode],_,rec,Sx[mode],X_old = ista_nn(Y[mode], D[mode], B, lamb_rel[mode], k=k[mode], itermax=inner_iter_max, verbose=False, X0=X[mode], DtD=DtD[mode], DtYB=mttkrp, BtB=pseudo_inverse, eta=eta, tol=1e-8, return_old=True)
                else:
                    X[mode],_,rec,Sx[mode],X_old = ista(Y[mode], D[mode], B, lamb_rel[mode], k=k[mode], itermax=inner_iter_max, verbose=False, X0=X[mode], DtD=DtD[mode], DtYB=mttkrp, BtB=pseudo_inverse, eta=eta, tol=1e-8, return_old=True)

                if verbose:
                    nz = []
                    # printing sparsity levels for hand tuning
                    for i in range(rank):
                        nz.append(np.sum(np.abs(X_old[:,i])>0))
                    print('number of nonzeroes in X_old (before thres) is currently', nz)

                factor = D[mode]@X[mode]

            factors[mode] = factor
        # Calculate the current unnormalized error if we need it
        unnorml_rec_error, tensor, norm_tensor = error_calc(tensor, norm_tensor, weights, factors, 0, None, mttkrp)
        rec_error = unnorml_rec_error / norm_tensor
        rec_errors.append(rec_error)

        # TODO test
        if rec_error < err_best:
            X_best = np.copy(X)
            S_best = np.copy(Sx)
            err_best = rec_error
            cp_tensor = CPTensor(copy.deepcopy((weights,factors)))

        if tol:

            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if verbose:
                    print("iteration {}, reconstruction error: {}, decrease = {}, unnormalized = {}".format(iteration, rec_error, rec_error_decrease, unnorml_rec_error))

                if cvg_criterion == 'abs_rec_error':
                    stop_flag = abs(rec_error_decrease) < tol
                elif cvg_criterion == 'rec_error':
                    stop_flag = rec_error_decrease < tol
                else:
                    raise TypeError("Unknown convergence criterion")

                if stop_flag:
                    if verbose:
                        print("PARAFAC converged after {} iterations".format(iteration))
                    break

            else:
                if verbose:
                    print('reconstruction error={}'.format(rec_errors[-1]))

    if return_errors:
        return cp_tensor, X_best, S_best, rec_errors
    else:
        return cp_tensor, X_best, S_best


def dlra_mf(Y, rank, D, k, lamb_rel=0.01, n_iter_max=100, init='random', tol=1e-8,
            verbose=0, return_errors=False,
            cvg_criterion='abs_rec_error', inner_iter_max=20, method='ista', tau=20, itermax_calib=20):
    """Dictionary-based Matrix Factorization via alternating least squares (ALS) and Mixed Sparse Coding (MSC)
    Computes a rank-`rank` decomposition of Y such that
    :math:`Y \\approx DXB^T`
    with columnwise k-sparse X.

    Parameters
    ----------
    Y : numpy array
        input data matrix
    rank  : int
        Number of components.
    D : ndarray
        Dictionary for the first mode
    k : int
        Columnwise sparsity level for the first mode
    lamb_rel: float in [0,1], default 0.01
        Amount of regularization for MSC Fista solver.
    n_iter_max : int, default 100
        Maximum number of iteration
    init : {'random', CPTensor}, optional
        Type of factor matrix initialization. If a CPTensor is passed, this is directly used for initalization.
    tol : float, optional, default: 1e-8
        Relative reconstruction error tolerance. The
        algorithm is considered to have found a local minimum when the
        reconstruction error is less than `tol`.
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
        Stopping criterion for the outer iterations, works if `tol` is not None. If 'rec_error', iterations stop at current iteration if :math:`(previous rec_error - current rec_error) < tol`. If 'abs_rec_error', iterations terminate when :math:`|previous rec_error - current rec_error| < tol`.
    method : string, default 'ista'
        update strategy for the MSC problem. Can be 'ista', 'iht' or 'homp'.
    tau : int, default 20
        tolerance for the sparsity level of estimated X in the inner ista loops.
    itermax_calib : int, default 20
        maximum number of regularization calibration steps.

    Returns
    -------
    Decomposed Matrix : list of numpy arrays
        Estimated DMF factors [DX, B]

    X : numpy array
        The sparse coefficients of the first factor

    S : numpy array
        Support of X

    errors : list
        A list of reconstruction errors at each iteration of the algorithms.
    """

    # only need to initialize B
    if init=='random':
        B = np.random.randn(Y.shape[1],rank)
        X = np.random.randn(D.shape[1],rank)
        DX = D@X
    else:
        X = np.copy(init[0])
        DX = D@X
        B = np.copy(init[1])

    Ynorm = np.linalg.norm(Y,'fro')
    rec_errors = [np.linalg.norm(Y - DX@B.T, 'fro')/Ynorm]
    DtD = D.T@D
    DtY = D.T@Y
    r = X.shape[1]
    # Turning lamb_rel into a list
    lamb_rel = lamb_rel*np.ones(r)
    # to compute stepsizes
    singvalD = np.linalg.svd(DtD)[1][0]

    # Storing best results
    err_best = np.inf

    for iteration in range(n_iter_max):

        if verbose > 1:
            print("Starting iteration", iteration + 1)


        singvalB = np.linalg.svd(B.T@B)[1][0]
        eta = 1/singvalD/singvalB
        if method=='ista':
            sparsity_balance=np.zeros(r)
            iter_calib=0


            # First ISTA without final projection
            while np.sum(sparsity_balance)<r and iter_calib<itermax_calib:
                nz=[]
                iter_calib+=1
                X,_,rec,_,X_old = ista(Y, D, B, lamb_rel, k=None, itermax=100, verbose=False, X0=X, tol=1e-8, warning=False, DtD=DtD, DtY=DtY, return_old=True, eta=eta) # lambda hard to tune
                # Automatic lambda tuning
                for i in range(r):
                    # first we get the nnz numbers
                    nz.append(np.sum(np.abs(X_old[:,i])>0))

                for i in range(r):
                    # then we increase lambda if needed, or decrease if needed
                    # tolerance on k is [k, k+10]
                    # We prefear that lambda is slightly too small to avoid picking zeroes arbitrarily
                    if nz[i]<k:
                        sparsity_balance[i]=0
                        lamb_rel[i] = lamb_rel[i]/1.3
                    elif nz[i]>(k+tau):
                        #elif nz[i]>(3*k):
                        sparsity_balance[i]=0
                        lamb_rel[i] = np.minimum(lamb_rel[i]*1.01, 1)
                    else:
                        sparsity_balance[i]=1
                #print(nz)
                #print(lamb_rel)

            # Now the real ista
            X,_,rec,Sx,X_old = ista(Y, D, B, lamb_rel, k=k, itermax=inner_iter_max, verbose=False, X0=X, tol=1e-8, warning=False, DtD=DtD, DtY=DtY, return_old=True, eta=eta) # lambda hard to tune

            if verbose:
                nz = []
                # printing sparsity levels for hand tuning
                for i in range(X.shape[1]):
                    nz.append(np.sum(np.abs(X_old[:,i])>0))
                print('number of nonzeroes in X_old (before thres) is currently', nz)

        elif method=='homp':
            X, rec, Sx = homp(Y,D,B,k,Xin=X, itermax=50) # too slow for large k
        elif method=='iht':
            X,rec,Sx = iht_mix(Y,D,B,k,X_in=X,itermax=inner_iter_max, DtD=DtD, DtY=DtY, eta=eta) # not great
        else:
            print('method not supported')
            break
        DX = D@X

        DXtDX = X.T@DtD@X
        DXY = X.T@DtY
        Bt = np.linalg.solve(DXtDX, DXY)
        B = Bt.T

        # Calculate the current unnormalized error
        err = np.linalg.norm(Y - DX@B.T, 'fro')/Ynorm
        # check if better than before and storing best results
        if err < err_best:
            X_best = np.copy(X)
            B_best = np.copy(B)
            S_best = np.copy(Sx)
            err_best = err
        rec_errors.append(err)


        if tol:

            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if verbose:
                    print("iteration {}, reconstruction error: {}, decrease = {}".format(iteration, err, rec_error_decrease))

                if cvg_criterion == 'abs_rec_error':
                    stop_flag = abs(rec_error_decrease) < tol
                elif cvg_criterion == 'rec_error':
                    stop_flag = rec_error_decrease < tol
                else:
                    raise TypeError("Unknown convergence criterion")

                if stop_flag:
                    if verbose:
                        print("DMF converged after {} iterations".format(iteration))
                    break

            else:
                if verbose:
                    print('reconstruction error={}'.format(rec_errors[-1]))

    if return_errors:
        return [D@X_best, B_best], X_best, S_best, rec_errors
    else:
        return [D@X_best, B_best], X_best, S_best

def dlra_mf_iht(Y, rank, D, k, step_rel=0.5, n_iter_max=100, init='random', tol=1e-8,
            verbose=0, return_errors=False,
            cvg_criterion='abs_rec_error', accelerate=True):#, unbiais=False):
    """Dictionary-based Matrix Factorization via PALM
    solves approximately the following problem

        minimize 1/2 \|Y - D@X@B.T\|_F^2 wrt X, B s.t. \|X_i\|_0\leq k

    Parameters
    ----------
    Y : numpy array
        input data matrix
    rank : int
        Number of components.
    D : ndarray
        Dictionary for the first mode
    k : int
        Columnwise sparsity level for the first mode
    step_rel : float in [0,1]. Default: 0.5
        Percentage of theoretical largest stepsize actually used. Set smaller than 1 if divergence occurs.
    n_iter_max : int, default 100
        Maximum number of iteration
    init : {'random', [DX, B]]}, optional
        Type of factor matrix initialization. If a list of factors is passed, it is directly used for initalization. See `initialize_factors`.
    tol : float, default: 1e-8
        Relative reconstruction error tolerance. The algorithm is considered to have found a local minimum when the
        reconstruction error is less than `tol`.
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
       Stopping criterion for the outer iterations, works if `tol` is not None. If 'rec_error', iterations stop at current iteration if :math:`(previous rec_error - current rec_error) < tol`. If 'abs_rec_error', iterations terminate when :math:`|previous rec_error - current rec_error| < tol`.

    Returns
    -------
    Decomposed Matrix : list of numpy arrays
        Estimated DMF factors [DX, B]

    X : numpy array
        The sparse coefficients of the first factor

    S : numpy array
        Support of X

    errors : list
        A list of reconstruction errors at each iteration of the algorithms.
    """

    # only need to initialize B
    if init=='random':
        B = np.random.randn(Y.shape[1],rank)
        X = np.random.randn(D.shape[1],rank)
        DX = D@X
    else:
        X = np.copy(init[0])
        DX = D@X
        B = np.copy(init[1])

    Ynorm = np.linalg.norm(Y)
    rec_errors = [np.linalg.norm(Y - DX@B.T)/Ynorm]

    # Precomputations
    DtY = D.T@Y
    DtD = D.T@D
    X_old = np.copy(X)
    B_old = np.copy(B)

    # Default: no acceleration
    beta=0

    for iteration in range(n_iter_max):

        if verbose > 1:
            print("Starting iteration", iteration + 1)

        # Acceleration heuristic from iPALM
        if accelerate:
            beta = (iteration - 1)/(iteration + 2)

        # X proximal gradient update
        BtB = B.T@B
        eta = step_rel/np.linalg.norm(BtB)
        Zx = X + beta*(X - X_old)
        X_old = np.copy(X)
        X = HardT(Zx - eta * (DtD@Zx@(B.T@B) - DtY@B), k)
        DX = D@X

        # B gradient step
        DXtDX = DX.T@DX
        eta = step_rel/np.linalg.norm(DXtDX)
        Zb = B.T + beta*(B.T - B_old.T)
        B_old = np.copy(B)
        Bt = Zb - eta*(-X.T@DtY + DXtDX@Zb)
        B = Bt.T

        # Calculate the current unnormalized error if we need it
        if (tol or return_errors):
            err = np.linalg.norm(Y - DX@B.T)/Ynorm

        rec_errors.append(err)

        if tol:

            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if verbose:
                    print("iteration {}, reconstruction error: {}, decrease = {}".format(iteration, err, rec_error_decrease))

                if cvg_criterion == 'abs_rec_error':
                    stop_flag = abs(rec_error_decrease) < tol
                elif cvg_criterion == 'rec_error':
                    stop_flag = rec_error_decrease < tol
                else:
                    raise TypeError("Unknown convergence criterion")

                if stop_flag:
                    if verbose:
                        print("DMF converged after {} iterations".format(iteration))
                    break

            else:
                if verbose:
                    print('reconstruction error={}'.format(rec_errors[-1]))

    # Estimating support (np arrays with supports)
    Sx = np.argsort(np.abs(X), 0)
    # truncating the support
    Sx = Sx[-k:, :]

    return [DX, B], X, Sx, rec_errors
