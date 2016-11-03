# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Alexis Mignon <alexis.mignon@gmail.com>
#         Manoj Kumar <manojkumarsivaraj334@gmail.com>
#         Neil Marchant <ngmarchant@gmail.com>
#
# License: BSD clause 3

from distutils.version import LooseVersion
from sklearn import __version__ as sklearn_version
import numpy as np
import warnings
from scipy import sparse

new_sklearn_version = (LooseVersion(sklearn_version) > '0.17.1')

if new_sklearn_version:
    from sklearn.exceptions import ConvergenceWarning
else:
    from sklearn.utils import ConvergenceWarning
from sklearn.linear_model.base import (LinearModel, _pre_fit)
from sklearn.base import RegressorMixin
from sklearn.utils.validation import (check_X_y, check_array)
from sklearn.externals.six.moves import xrange

# Cross-validation
if new_sklearn_version:
    from sklearn.model_selection import check_cv
    from sklearn.linear_model.base import _preprocess_data
else:
    from sklearn.cross_validation import check_cv
    from sklearn.linear_model.base import (center_data, sparse_center_data)
    # Map _preprocess_data to old functions
    def _preprocess_data(X, y, fit_intercept, normalize, copy = True,
                            return_mean = False):
        if not return_mean:
            return center_data(X, y, fit_intercept, normalize, copy = False)
        else:
            return sparse_center_data(X, y, fit_intercept, normalize)
from abc import ABCMeta, abstractmethod
from sklearn.externals import six
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.validation import column_or_1d


from . import prox_fast

### Not my code ###
def _alpha_grid(X, y, Xy=None, l1_ratio=1.0, fit_intercept=True,
                eps=1e-3, n_alphas=100, normalize=False, copy_X=True):
    """ Compute the grid of alpha values for elastic net parameter search
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication
    y : ndarray, shape (n_samples,)
        Target values
    Xy : array-like, optional
        Xy = np.dot(X.T, y) that can be precomputed.
    l1_ratio : float
        The elastic net mixing parameter, with ``0 <= l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. ``For
        l1_ratio = 1`` it is an L1 penalty.  For ``0 < l1_ratio <
        1``, the penalty is a combination of L1 and L2.
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``
    n_alphas : int, optional
        Number of alphas along the regularization path
    fit_intercept : boolean, default True
        Whether to fit an intercept or not
    normalize : boolean, optional, default False
        If True, the regressors X will be normalized before regression.
        This parameter is ignored when `fit_intercept` is set to False.
        When the regressors are normalized, note that this makes the
        hyperparameters learnt more robust and almost independent of the number
        of samples. The same property is not valid for standardized data.
        However, if you wish to standardize, please use
        `preprocessing.StandardScaler` before calling `fit` on an estimator
        with `normalize=False`.
    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
    """
    n_samples = len(y)

    sparse_center = False
    if Xy is None:
        X_sparse = sparse.isspmatrix(X)
        sparse_center = X_sparse and (fit_intercept or normalize)
        X = check_array(X, 'csc',
                        copy=(copy_X and fit_intercept and not X_sparse))
        if not X_sparse:
            # X can be touched inplace thanks to the above line
            X, y, _, _, _ = _preprocess_data(X, y, fit_intercept,
                                             normalize, copy=False)
        Xy = safe_sparse_dot(X.T, y, dense_output=True)

        if sparse_center:
            # Workaround to find alpha_max for sparse matrices.
            # since we should not destroy the sparsity of such matrices.
            _, _, X_offset, _, X_scale = _preprocess_data(X, y, fit_intercept,
                                                      normalize,
                                                      return_mean=True)
            mean_dot = X_offset * np.sum(y)

    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]

    if sparse_center:
        if fit_intercept:
            Xy -= mean_dot[:, np.newaxis]
        if normalize:
            Xy /= X_scale[:, np.newaxis]

    alpha_max = (np.sqrt(np.sum(Xy ** 2, axis=1)).max() /
                 (n_samples * l1_ratio))

    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas

    return np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max),
            num=n_alphas)[::-1]

def _path_residuals(X, y, train, test, path, path_params, alphas=None,
                    l1_ratio=1, X_order=None, dtype=None):
    """Returns the MSE for the models computed by 'path'
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target values
    train : list of indices
        The indices of the train set
    test : list of indices
        The indices of the test set
    path : callable
        function returning a list of models on the path. See
        enet_path for an example of signature
    path_params : dictionary
        Parameters passed to the path function
    alphas : array-like, optional
        Array of float that is used for cross-validation. If not
        provided, computed using 'path'
    l1_ratio : float, optional
        float between 0 and 1 passed to ElasticNet (scaling between
        l1 and l2 penalties). For ``l1_ratio = 0`` the penalty is an
        L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty. For ``0
        < l1_ratio < 1``, the penalty is a combination of L1 and L2
    X_order : {'F', 'C', or None}, optional
        The order of the arrays expected by the path function to
        avoid memory copies
    dtype : a numpy dtype or None
        The dtype of the arrays expected by the path function to
        avoid memory copies
    """
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
    fit_intercept = path_params['fit_intercept']
    normalize = path_params['normalize']

    if y.ndim == 1:
        precompute = path_params['precompute']
    else:
        # No Gram variant of multi-task exists right now.
        # Fall back to default enet_multitask
        precompute = False

    X_train, y_train, X_offset, y_offset, X_scale, precompute, Xy = \
        _pre_fit(X_train, y_train, None, precompute, normalize, fit_intercept,
                 copy=False)

    path_params = path_params.copy()
    path_params['Xy'] = Xy
    path_params['X_offset'] = X_offset
    path_params['X_scale'] = X_scale
    path_params['precompute'] = precompute
    path_params['copy_X'] = False
    path_params['alphas'] = alphas

    if 'l1_ratio' in path_params:
        path_params['l1_ratio'] = l1_ratio

    # Do the ordering and type casting here, as if it is done in the path,
    # X is copied and a reference is kept here
    X_train = check_array(X_train, 'csc', dtype=dtype, order=X_order)
    alphas, coefs, _ = path(X_train, y_train, **path_params)
    del X_train, y_train

    if y.ndim == 1:
        # Doing this so that it becomes coherent with multioutput.
        coefs = coefs[np.newaxis, :, :]
        y_offset = np.atleast_1d(y_offset)
        y_test = y_test[:, np.newaxis]

    if normalize:
        nonzeros = np.flatnonzero(X_scale)
        coefs[:, nonzeros] /= X_scale[nonzeros][:, np.newaxis]

    intercepts = y_offset[:, np.newaxis] - np.dot(X_offset, coefs)
    if sparse.issparse(X_test):
        n_order, n_features, n_alphas = coefs.shape
        # Work around for sparse matices since coefs is a 3-D numpy array.
        coefs_feature_major = np.rollaxis(coefs, 1)
        feature_2d = np.reshape(coefs_feature_major, (n_features, -1))
        X_test_coefs = safe_sparse_dot(X_test, feature_2d)
        X_test_coefs = X_test_coefs.reshape(X_test.shape[0], n_order, -1)
    else:
        X_test_coefs = safe_sparse_dot(X_test, coefs)
    residues = X_test_coefs - y_test[:, :, np.newaxis]
    residues += intercepts
    this_mses = ((residues ** 2).mean(axis=0)).mean(axis=0)

    return this_mses

### My code ###

def enet_path(X, y, l1_ratio=0.5, eps=1e-3, eta = 0.5, init_step = 10,
              n_alphas=100, alphas=None, precompute='auto', Xy=None,
              copy_X=True, coef_init=None, verbose=False, return_n_iter=False,
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
    dual_gaps : array, shape (n_alphas,)
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

    # MultiTaskElasticNet does not support sparse matrices
    if sparse.isspmatrix(X):
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
    dual_gaps = np.empty(n_alphas)
    n_iters = []

    coefs = np.empty((n_features, n_alphas), dtype=np.float64)

    if coef_init is None:
        coef_ = np.asfortranarray(np.zeros(coefs.shape[:-1]))
    else:
        coef_ = np.asfortranarray(coef_init)

    for i, alpha in enumerate(alphas):
        l1_reg = alpha * l1_ratio * n_samples
        l2_reg = alpha * (1.0 - l1_ratio) * n_samples
        if sparse.isspmatrix(X):
            model = prox_fast.sparse_enet_prox_gradient(
                coef_, l1_reg, l2_reg, X.data, X.indices,
                X.indptr, y, X_sparse_scaling,
                max_iter, tol)
        elif isinstance(precompute, np.ndarray):
            # We expect precompute to be already Fortran ordered when bypassing
            # checks
            if check_input:
                precompute = check_array(precompute, dtype=np.float64,
                                         order='C')
            model = prox_fast.enet_prox_gradient_gram(
                coef_, l1_reg, l2_reg, precompute, Xy, y, max_iter, eta,
                init_step, tol)
        elif precompute is False:
            model = prox_fast.enet_prox_gradient(
                coef_, l1_reg, l2_reg, X, y, max_iter, eta, init_step, tol)
        else:
            raise ValueError("Precompute should be one of True, False, "
                             "'auto' or array-like")
        coef_, dual_gap_, tol_, n_iter_ = model
        coefs[..., i] = coef_
        dual_gaps[i] = dual_gap_
        n_iters.append(n_iter_)

        if dual_gap_ > tol_:
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
        return alphas, coefs, dual_gaps, n_iters
    return alphas, coefs, dual_gaps

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

    def __init__(self, alpha=1.0, l1_ratio=0.5, eta = 0.5, init_step = 10,
             fit_intercept=True, normalize=False, precompute=False,
             max_iter=1000, copy_X=True, tol=1e-4, warm_start=False):
        # Initialise parameters
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.eta = eta
        self.init_step = init_step
        self.warm_start = warm_start
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y, check_input=True):

        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator", stacklevel=2)

        if (isinstance(self.precompute, six.string_types) and
           self.precompute == 'auto'):
            warnings.warn("Setting precompute to 'auto', was found to be "
                          "slower even when n_samples > n_features. Hence "
                          "it will be removed in 0.18.",
                          DeprecationWarning, stacklevel=2)

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
            # Initial guess for coef_ is zero
            coef_ = np.zeros((n_targets, n_features), dtype=np.float64,
                             order='F')
        else:
            # Use previous value of coef_ as initial guess
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]

        dual_gaps_ = np.zeros(n_targets, dtype = np.float64)
        self.n_iter_ = []

        # Perform the optimisation
        for k in xrange(n_targets):
            if Xy is not None:
                this_Xy = Xy[:, k]
            else:
                this_Xy = None

            _, this_coef, this_dual_gap, this_iter  = \
                self.path(X, y[:, k],
                      l1_ratio=self.l1_ratio, eps=None, eta = self.eta,
                      init_step = self.init_step, n_alphas=None,
                      alphas=[self.alpha], precompute=precompute, Xy=this_Xy,
                      fit_intercept=False, normalize=False, copy_X=True,
                      verbose=False, tol=self.tol,
                      X_offset=X_offset, X_scale=X_scale, return_n_iter=True,
                      coef_init=coef_[k], max_iter=self.max_iter,
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

### Not my code
class LinearModelCV(six.with_metaclass(ABCMeta, LinearModel)):
    """Base class for iterative model fitting along a regularization path"""

    @abstractmethod
    def __init__(self, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True,
                 normalize=False, precompute='auto', max_iter=1000, tol=1e-4,
                 eta = 0.5, init_step = 10, copy_X=True, cv=None, verbose=False,
                 n_jobs=1):
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta
        self.init_step = init_step
        self.copy_X = copy_X
        self.cv = cv
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit linear model with coordinate descent
        Fit is on grid of alphas and best alpha estimated by cross-validation.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data. Pass directly as float64, Fortran-contiguous data
            to avoid unnecessary memory duplication. If y is mono-output,
            X can be sparse.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values
        """
        y = np.asarray(y, dtype=np.float64)
        if y.shape[0] == 0:
            raise ValueError("y has 0 samples: %r" % y)

        if hasattr(self, 'l1_ratio'):
            model_str = 'ElasticNet'
        else:
            model_str = 'Lasso'

        if isinstance(self, ElasticNetCV) or isinstance(self, LassoCV):
            if model_str == 'ElasticNet':
                model = ElasticNet()
            else:
                model = Lasso()
            if y.ndim > 1 and y.shape[1] > 1:
                raise ValueError("For multi-task outputs, use "
                                 "MultiTask%sCV" % (model_str))
            y = column_or_1d(y, warn=True)
        else:
            if sparse.isspmatrix(X):
                raise TypeError("X should be dense but a sparse matrix was"
                                "passed")
            elif y.ndim == 1:
                raise ValueError("For mono-task outputs, use "
                                 "%sCV" % (model_str))
            if model_str == 'ElasticNet':
                model = MultiTaskElasticNet()
            else:
                model = MultiTaskLasso()

        # This makes sure that there is no duplication in memory.
        # Dealing right with copy_X is important in the following:
        # Multiple functions touch X and subsamples of X and can induce a
        # lot of duplication of memory
        copy_X = self.copy_X and self.fit_intercept

        if isinstance(X, np.ndarray) or sparse.isspmatrix(X):
            # Keep a reference to X
            reference_to_old_X = X
            # Let us not impose fortran ordering or float64 so far: it is
            # not useful for the cross-validation loop and will be done
            # by the model fitting itself
            X = check_array(X, 'csc', copy=False)
            if sparse.isspmatrix(X):
                if (hasattr(reference_to_old_X, "data") and
                   not np.may_share_memory(reference_to_old_X.data, X.data)):
                    # X is a sparse matrix and has been copied
                    copy_X = False
            elif not np.may_share_memory(reference_to_old_X, X):
                # X has been copied
                copy_X = False
            del reference_to_old_X
        else:
            X = check_array(X, 'csc', dtype=np.float64, order='F', copy=copy_X)
            copy_X = False

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                             % (X.shape[0], y.shape[0]))

        # All LinearModelCV parameters except 'cv' are acceptable
        path_params = self.get_params()
        if 'l1_ratio' in path_params:
            l1_ratios = np.atleast_1d(path_params['l1_ratio'])
            # For the first path, we need to set l1_ratio
            path_params['l1_ratio'] = l1_ratios[0]
        else:
            l1_ratios = [1, ]
        path_params.pop('cv', None)
        path_params.pop('n_jobs', None)

        alphas = self.alphas
        n_l1_ratio = len(l1_ratios)
        if alphas is None:
            alphas = []
            for l1_ratio in l1_ratios:
                alphas.append(_alpha_grid(
                    X, y, l1_ratio=l1_ratio,
                    fit_intercept=self.fit_intercept,
                    eps=self.eps, n_alphas=self.n_alphas,
                    normalize=self.normalize,
                    copy_X=self.copy_X))
        else:
            # Making sure alphas is properly ordered.
            alphas = np.tile(np.sort(alphas)[::-1], (n_l1_ratio, 1))
        # We want n_alphas to be the number of alphas used for each l1_ratio.
        n_alphas = len(alphas[0])
        path_params.update({'n_alphas': n_alphas})

        path_params['copy_X'] = copy_X
        # We are not computing in parallel, we can modify X
        # inplace in the folds
        if not (self.n_jobs == 1 or self.n_jobs is None):
            path_params['copy_X'] = False

        # init cross-validation generator
        if new_sklearn_version:
            cv = check_cv(self.cv)
        else:
            cv = check_cv(self.cv, X)

        # Compute path for all folds and compute MSE to get the best alpha
        if new_sklearn_version:
            folds = list(cv.split(X))
        else:
            folds = list(cv)
        best_mse = np.inf

        # We do a double for loop folded in one, in order to be able to
        # iterate in parallel on l1_ratio and folds
        jobs = (delayed(_path_residuals)(X, y, train, test, self.path,
                                         path_params, alphas=this_alphas,
                                         l1_ratio=this_l1_ratio, X_order='F',
                                         dtype=np.float64)
                for this_l1_ratio, this_alphas in zip(l1_ratios, alphas)
                for train, test in folds)
        mse_paths = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             backend="threading")(jobs)
        mse_paths = np.reshape(mse_paths, (n_l1_ratio, len(folds), -1))
        mean_mse = np.mean(mse_paths, axis=1)
        self.mse_path_ = np.squeeze(np.rollaxis(mse_paths, 2, 1))
        for l1_ratio, l1_alphas, mse_alphas in zip(l1_ratios, alphas,
                                                   mean_mse):
            i_best_alpha = np.argmin(mse_alphas)
            this_best_mse = mse_alphas[i_best_alpha]
            if this_best_mse < best_mse:
                best_alpha = l1_alphas[i_best_alpha]
                best_l1_ratio = l1_ratio
                best_mse = this_best_mse

        self.l1_ratio_ = best_l1_ratio
        self.alpha_ = best_alpha
        if self.alphas is None:
            self.alphas_ = np.asarray(alphas)
            if n_l1_ratio == 1:
                self.alphas_ = self.alphas_[0]
        # Remove duplicate alphas in case alphas is provided.
        else:
            self.alphas_ = np.asarray(alphas[0])

        # Refit the model with the parameters selected
        common_params = dict((name, value)
                             for name, value in self.get_params().items()
                             if name in model.get_params())
        model.set_params(**common_params)
        model.alpha = best_alpha
        model.l1_ratio = best_l1_ratio
        model.copy_X = copy_X
        model.precompute = False
        model.fit(X, y)
        if not hasattr(self, 'l1_ratio'):
            del self.l1_ratio_
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        self.dual_gap_ = model.dual_gap_
        self.n_iter_ = model.n_iter_
        return self

class ElasticNetCV(LinearModelCV, RegressorMixin):
    """Elastic Net model with iterative fitting along a regularization path
    The best model is selected by cross-validation.
    Read more in the :ref:`User Guide <elastic_net>`.
    Parameters
    ----------
    l1_ratio : float or array of floats, optional
        float between 0 and 1 passed to ElasticNet (scaling between
        l1 and l2 penalties). For ``l1_ratio = 0``
        the penalty is an L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2
        This parameter can be a list, in which case the different
        values are tested by cross-validation and the one giving the best
        prediction score is used. Note that a good choice of list of
        values for l1_ratio is often to put more values close to 1
        (i.e. Lasso) and less close to 0 (i.e. Ridge), as in ``[.1, .5, .7,
        .9, .95, .99, 1]``
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.
    n_alphas : int, optional
        Number of alphas along the regularization path, used for each l1_ratio.
    alphas : numpy array, optional
        List of alphas where to compute the models.
        If None alphas are set automatically
    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.
    max_iter : int, optional
        The maximum number of iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    verbose : bool or integer
        Amount of verbosity.
    n_jobs : integer, optional
        Number of CPUs to use during the cross validation. If ``-1``, use
        all the CPUs.
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    normalize : boolean, optional, default False
        If True, the regressors X will be normalized before regression.
        This parameter is ignored when `fit_intercept` is set to False.
        When the regressors are normalized, note that this makes the
        hyperparameters learnt more robust and almost independent of the number
        of samples. The same property is not valid for standardized data.
        However, if you wish to standardize, please use
        `preprocessing.StandardScaler` before calling `fit` on an estimator
        with `normalize=False`.
    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
    Attributes
    ----------
    alpha_ : float
        The amount of penalization chosen by cross validation
    l1_ratio_ : float
        The compromise between l1 and l2 penalization chosen by
        cross validation
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        Parameter vector (w in the cost function formula),
    intercept_ : float | array, shape (n_targets, n_features)
        Independent term in the decision function.
    mse_path_ : array, shape (n_l1_ratio, n_alpha, n_folds)
        Mean square error for the test set on each fold, varying l1_ratio and
        alpha.
    alphas_ : numpy array, shape (n_alphas,) or (n_l1_ratio, n_alphas)
        The grid of alphas used for fitting, for each l1_ratio.
    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.
    Notes
    -----
    See examples/linear_model/lasso_path_with_crossvalidation.py
    for an example.
    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.
    The parameter l1_ratio corresponds to alpha in the glmnet R package
    while alpha corresponds to the lambda parameter in glmnet.
    More specifically, the optimization objective is::
        1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
    If you are interested in controlling the L1 and L2 penalty
    separately, keep in mind that this is equivalent to::
        a * L1 + b * L2
    for::
        alpha = a + b and l1_ratio = a / (a + b).
    See also
    --------
    enet_path
    ElasticNet
    """
    path = staticmethod(enet_path)

    def __init__(self, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, precompute='auto',
                 max_iter=1000, tol=1e-4, eta=0.5, init_step=10, cv=None,
                 copy_X=True, verbose=0, n_jobs=1):
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta
        self.init_step = init_step
        self.cv = cv
        self.copy_X = copy_X
        self.verbose = verbose
        self.n_jobs = n_jobs
