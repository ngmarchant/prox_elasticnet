# Author: Neil G. Marchant <ngmarchant@gmail.com>
#
# License:

import numpy as np
import warnings
from scipy import sparse

from sklearn.linear_model.base import (LinearModel, _pre_fit)
from sklearn.base import RegressorMixin
from sklearn.utils.validation import (check_X_y, check_array)
from sklearn.externals.six.moves import xrange

import prox_fast

def enet_path(X, y, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
              precompute='auto', Xy=None, copy_X=True, coef_init=None,
              verbose=False, return_n_iter=False,
              check_input=True, **params):
    """Compute elastic net path with coordinate descent
    The elastic net optimization function varies for mono and multi-outputs.
    For mono-output tasks it is::
        1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
    For multi-output tasks it is::
        (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
    Where::
        ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}
    i.e. the sum of norm of each row.
    Read more in the :ref:`User Guide <elastic_net>`.
    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication. If ``y`` is mono-output then ``X``
        can be sparse.
    y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
        Target values
    l1_ratio : float, optional
        float between 0 and 1 passed to elastic net (scaling between
        l1 and l2 penalties). ``l1_ratio=1`` corresponds to the Lasso
    eps : float
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``
    n_alphas : int, optional
        Number of alphas along the regularization path
    alphas : ndarray, optional
        List of alphas where to compute the models.
        If None alphas are set automatically
    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.
    Xy : array-like, optional
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.
    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
    coef_init : array, shape (n_features, ) | None
        The initial values of the coefficients.
    verbose : bool or integer
        Amount of verbosity.
    params : kwargs
        keyword arguments passed to the coordinate descent solver.
    return_n_iter : bool
        whether to return the number of iterations or not.
    check_input : bool, default True
        Skip input validation checks, including the Gram matrix when provided
        assuming there are handled by the caller when check_input=False.
    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.
    coefs : array, shape (n_features, n_alphas) or \
            (n_outputs, n_features, n_alphas)
        Coefficients along the path.
    rel_errors : array, shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.
    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.
        (Is returned when ``return_n_iter`` is set to True).
    Notes
    -----
    See examples/plot_lasso_coordinate_descent_path.py for an example.
    See also
    --------
    MultiTaskElasticNet
    MultiTaskElasticNetCV
    ElasticNet
    ElasticNetCV
    """
    # We expect X and y to be already float64 Fortran ordered when bypassing
    # checks
    if check_input:
        X = check_array(X, 'csc', dtype=np.float64, order='F', copy=copy_X)
        y = check_array(y, 'csc', dtype=np.float64, order='F', copy=False,
                        ensure_2d=False)
        if Xy is not None:
            # Xy should be a 1d contiguous array or a 2D C ordered array
            Xy = check_array(Xy, dtype=np.float64, order='C', copy=False,
                             ensure_2d=False)
    n_samples, n_features = X.shape

    multi_output = False
    if y.ndim != 1:
        multi_output = True
        _, n_outputs = y.shape

    # MultiTaskElasticNet does not support sparse matrices
    if not multi_output and sparse.isspmatrix(X):
        if 'X_offset' in params:
            # As sparse matrices are not actually centered we need this
            # to be passed to the CD solver.
            X_sparse_scaling = params['X_offset'] / params['X_scale']
        else:
            X_sparse_scaling = np.zeros(n_features)

    # X should be normalized and fit already if function is called
    # from ElasticNet.fit
    if check_input:
        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, Xy, precompute, normalize=False,
                     fit_intercept=False, copy=False)
    if alphas is None:
        # No need to normalize of fit_intercept: it has been done
        # above
        alphas = _alpha_grid(X, y, Xy=Xy, l1_ratio=l1_ratio,
                             fit_intercept=False, eps=eps, n_alphas=n_alphas,
                             normalize=False, copy_X=False)
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 1000)
    rel_errors = np.empty(n_alphas)
    n_iters = []

    if not multi_output:
        coefs = np.empty((n_features, n_alphas), dtype=np.float64)
    else:
        coefs = np.empty((n_outputs, n_features, n_alphas),
                         dtype=np.float64)

    if coef_init is None:
        coef_ = np.asfortranarray(np.zeros(coefs.shape[:-1]))
    else:
        coef_ = np.asfortranarray(coef_init)

    for i, alpha in enumerate(alphas):
        l1_reg = alpha * l1_ratio * n_samples
        l2_reg = alpha * (1.0 - l1_ratio) * n_samples
        if not multi_output and sparse.isspmatrix(X):
            model = prox_fast.sparse_enet_prox_gradient(
                coef_, l1_reg, l2_reg, X.data, X.indices,
                X.indptr, y, X_sparse_scaling,
                max_iter, tol)
                # Remove multi_output option
        elif multi_output:
            model = cd_fast.enet_prox_gradient_multi_task(
                coef_, l1_reg, l2_reg, X, y, max_iter, tol)
        elif isinstance(precompute, np.ndarray):
            # We expect precompute to be already Fortran ordered when bypassing
            # checks
            if check_input:
                precompute = check_array(precompute, dtype=np.float64,
                                         order='C')
            model = prox_fast.enet_prox_gradient_gram(
                coef_, l1_reg, l2_reg, precompute, Xy, y, max_iter,
                tol)
        elif precompute is False:
            model = prox_fast.enet_prox_gradient(
                coef_, l1_reg, l2_reg, X, y, max_iter, tol)
        else:
            raise ValueError("Precompute should be one of True, False, "
                             "'auto' or array-like")
        coef_, rel_error_, n_iter_ = model
        coefs[..., i] = coef_
        rel_errors[i] = rel_error_
        n_iters.append(n_iter_)
        if rel_error_ > tol:
            warnings.warn('Objective did not converge.' +
                          ' You might want' +
                          ' to increase the number of iterations',
                          ConvergenceWarning)

        if verbose:
            if verbose > 2:
                print(model)
            elif verbose > 1:
                print('Path: %03i out of %03i' % (i, n_alphas))
            else:
                sys.stderr.write('.')

    if return_n_iter:
        return alphas, coefs, rel_errors, n_iters
    return alphas, coefs, rel_errors

class ElasticNet(LinearModel, RegressorMixin) :
    """Linear regression with combined L1 and L2 priors as regularizer.
    Minimizes the objective function::
            1 / (2 * n_samples) * ||y - Xw||^2_2
            + alpha * l1_ratio * ||w||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
    If you are interested in controlling the L1 and L2 penalty
    separately, keep in mind that this is equivalent to::
            a * L1 + b * L2
    where::
            alpha = a + b and l1_ratio = a / (a + b)
    The parameter l1_ratio corresponds to alpha in the glmnet R package while
    alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio
    = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable,
    unless you supply your own sequence of alpha.
    Read more in the :ref:`User Guide <elastic_net>`.
    Parameters
    ----------
    alpha : float
        Constant that multiplies the penalty terms. Defaults to 1.0
        See the notes for the exact mathematical meaning of this
        parameter.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the Lasso object is not advised
        and you should prefer the LinearRegression object.
    l1_ratio : float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.
    fit_intercept : bool
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.
    normalize : boolean, optional, default False
        If True, the regressors X will be normalized before regression.
        This parameter is ignored when `fit_intercept` is set to False.
        When the regressors are normalized, note that this makes the
        hyperparameters learnt more robust and almost independent of the number
        of samples. The same property is not valid for standardized data.
        However, if you wish to standardize, please use
        `preprocessing.StandardScaler` before calling `fit` on an estimator
        with `normalize=False`.
    precompute : True | False | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument. For sparse input
        this option is always ``True`` to preserve sparsity.
    max_iter : int, optional
        The maximum number of iterations
    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)
    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1) | \
            (n_targets, n_features)
        ``sparse_coef_`` is a readonly property derived from ``coef_``
    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.
    n_iter_ : array-like, shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    Notes
    -----
    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.
    """

    path = staticmethod(enet_path)

    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
             normalize=False, precompute=False, max_iter=1000,
             copy_X=True, tol=1e-4, warm_start=False):
        # Initialise parameters
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        # Initialise attributes
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y, check_input=True):

        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator", stacklevel=2)

        if check_input:
            # Ensure that X and y are float64 Fortran ordered arrays.
            # Also check for consistency in the dimensions, and that y doesn't
            # contain np.nan or np.inf entries.
            y = np.asarray(y, dtype=np.float64)
            X, y = check_X_y(X, y, accept_sparse='csc', dtype=np.float64,
                             order='F',
                             copy=self.copy_X and self.fit_intercept,
                             multi_output=True, y_numeric=True)
            y = check_array(y, dtype=np.float64, order='F', copy=False,
                            ensure_2d=False)

        # Centre and normalise the data
        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, None, self.precompute, self.normalize,
                     self.fit_intercept, copy=False)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]
        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        if not self.warm_start or self.coef_ is None:
            # Initial guess for coef_ vector is zero
            coef_ = np.zeros((n_targets, n_features), dtype=np.float64,
                             order='F')
        else:
            # Use previous value of coef_ vector as initial guess
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]

        rel_errors_ = np.zeros(n_targets, dtype=X.dtype)
        self.n_iter_ = []

        # Loop over different measurements of y (generalisation)
        for k in xrange(n_targets):
            if Xy is not None:
                this_Xy = Xy[:, k]
            else:
                this_Xy = None
            # Perform the optimisation
            _, this_coef, this_rel_error, this_iter = \
                self.path(X, y[:, k],
                          l1_ratio=self.l1_ratio, eps=None,
                          n_alphas=None, alphas=[self.alpha],
                          precompute=precompute, Xy=this_Xy,
                          fit_intercept=False, normalize=False, copy_X=True,
                          verbose=False, tol=self.tol,
                          X_offset=X_offset, X_scale=X_scale, return_n_iter=True,
                          coef_init=coef_[k], max_iter=self.max_iter,
                          check_input=False)
            coef_[k] = this_coef[:, 0]
            rel_errors_[k] = this_rel_error[0]
            self.n_iter_.append(this_iter[0])

        if n_targets == 1:
            self.n_iter_ = self.n_iter_[0]

        self.coef_, self.rel_error_ = map(np.squeeze, [coef_, rel_errors_])
        self._set_intercept(X_offset, y_offset, X_scale)

        # return self for chaining fit and predict calls
        return self
