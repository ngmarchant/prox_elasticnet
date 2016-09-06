# Author: Neil G. Marchant <ngmarchant@gmail.com>
#
# License:

import numpy as np
import warnings

from sklearn import (LinearModel, RegressorMixin, check_X_y, check_array,
                    _pre_fit)

from . import prox_fast

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
    positive : bool, optional
        When set to ``True``, forces the coefficients to be positive.
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
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
             normalize=False, precompute=False, max_iter=1000,
             copy_X=True, tol=1e-4, warm_start=False, positive=False):
        # Initialise parameters
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
#        self.warm_start = warm_start
#        self.positive = positive
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

        dual_gaps_ = np.zeros(n_targets, dtype=np.float64)
        self.n_iter_ = []

        # Loop over different measurements of y (generalisation)
        for k in xrange(n_targets):
            if Xy is not None:
                this_Xy = Xy[:, k]
            else:
                this_Xy = None
            # Perform the optimisation
            _, this_coef, this_dual_gap, this_iter = \
                self.path(X, y[:, k],
                          l1_ratio=self.l1_ratio, eps=None,
                          n_alphas=None, alphas=[self.alpha],
                          precompute=precompute, Xy=this_Xy,
                          fit_intercept=False, normalize=False, copy_X=True,
                          verbose=False, tol=self.tol, positive=self.positive,
                          X_offset=X_offset, X_scale=X_scale, return_n_iter=True,
                          coef_init=coef_[k], max_iter=self.max_iter,
                          random_state=self.random_state,
                          selection=self.selection,
                          check_input=False)
            coef_[k] = this_coef[:, 0]
            dual_gaps_[k] = this_dual_gap[0]
            self.n_iter_.append(this_iter[0])

        if n_targets == 1:
            self.n_iter_ = self.n_iter_[0]

        self.coef_, self.dual_gap_ = map(np.squeeze, [coef_, dual_gaps_])
        self._set_intercept(X_offset, y_offset, X_scale)

        # return self for chaining fit and predict calls
        return self
