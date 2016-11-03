# Authors: Olivier Grisel <olivier.grisel@ensta.org>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

from sys import version_info

import numpy as np
from scipy import interpolate, sparse
from copy import deepcopy

from sklearn.datasets import load_boston
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import SkipTest
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_warns
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import TempMemmap

from prox_elasticnet import ElasticNet, ElasticNetCV, enet_path
from sklearn.utils import check_array


def check_warnings():
    if version_info < (2, 6):
        raise SkipTest("Testing for warnings is not supported in versions \
        older than Python 2.6")


def test_enet_zero():
    # Check that elastic net can handle zero data without crashing
    X = [[0], [0], [0]]
    y = [0, 0, 0]
    clf = ElasticNet(alpha=0.1).fit(X, y)
    pred = clf.predict([[1], [2], [3]])
    assert_array_almost_equal(clf.coef_, [0])
    assert_array_almost_equal(pred, [0, 0, 0])
    assert_almost_equal(clf.dual_gap_, 0)


def test_enet_toy():
    # Test ElasticNet for various parameters of alpha and l1_ratio.
    # Actually, the parameters alpha = 0 should not be allowed. However,
    # we test it as a border case.
    # ElasticNet is tested with and without precomputed Gram matrix

    X = np.array([[-1.], [0.], [1.]])
    Y = [-1, 0, 1]       # just a straight line
    T = [[2.], [3.], [4.]]  # test sample

    # this should be the same as lasso
    clf = ElasticNet(alpha=1e-8, l1_ratio=1.0)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1], decimal=3)
    assert_array_almost_equal(pred, [2, 3, 4], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0, decimal=3)

    clf = ElasticNet(alpha=0.5, l1_ratio=0.3, max_iter=100,
                     precompute=False)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0, decimal=3)

    clf.set_params(max_iter=100, precompute=True)
    clf.fit(X, Y)  # with Gram
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0, decimal=3)

    clf.set_params(max_iter=100, precompute=np.dot(X.T, X))
    clf.fit(X, Y)  # with Gram
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0, decimal=3)

    clf = ElasticNet(alpha=0.5, l1_ratio=0.5)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.45454], decimal=3)
    assert_array_almost_equal(pred, [0.9090, 1.3636, 1.8181], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0, decimal=3)


def build_dataset(n_samples=50, n_features=200, n_informative_features=10,
                  n_targets=1):
    """
    build an ill-posed linear regression problem with many noisy features and
    comparatively few samples
    """
    random_state = np.random.RandomState(0)
    if n_targets > 1:
        w = random_state.randn(n_features, n_targets)
    else:
        w = random_state.randn(n_features)
    w[n_informative_features:] = 0.0
    X = random_state.randn(n_samples, n_features)
    y = np.dot(X, w)
    X_test = random_state.randn(n_samples, n_features)
    y_test = np.dot(X_test, w)
    return X, y, X_test, y_test


def test_enet_path():
    # We use a large number of samples and of informative features so that
    # the l1_ratio selected is more toward ridge than lasso
    X, y, X_test, y_test = build_dataset(n_samples=200, n_features=100,
                                         n_informative_features=100)
    max_iter = 150

    # Here we have a small number of iterations, and thus the
    # ElasticNet might not converge. This is to speed up tests
    clf = ElasticNetCV(alphas=[0.01, 0.05, 0.1], eps=2e-3,
                       l1_ratio=[0.5, 0.7], cv=3,
                       max_iter=max_iter)
    ignore_warnings(clf.fit)(X, y)
    # Well-conditioned settings, we should have selected our
    # smallest penalty
    assert_almost_equal(clf.alpha_, min(clf.alphas_))
    # Non-sparse ground truth: we should have selected an elastic-net
    # that is closer to ridge than to lasso
    assert_equal(clf.l1_ratio_, min(clf.l1_ratio))

    clf = ElasticNetCV(alphas=[0.01, 0.05, 0.1], eps=2e-3,
                       l1_ratio=[0.5, 0.7], cv=3,
                       max_iter=max_iter, precompute=True)
    ignore_warnings(clf.fit)(X, y)

    # Well-conditioned settings, we should have selected our
    # smallest penalty
    assert_almost_equal(clf.alpha_, min(clf.alphas_))
    # Non-sparse ground truth: we should have selected an elastic-net
    # that is closer to ridge than to lasso
    assert_equal(clf.l1_ratio_, min(clf.l1_ratio))

    # We are in well-conditioned settings with low noise: we should
    # have a good test-set performance
    assert_greater(clf.score(X_test, y_test), 0.99)


def test_path_parameters():
    X, y, _, _ = build_dataset()
    max_iter = 100

    clf = ElasticNetCV(n_alphas=50, eps=1e-3, max_iter=max_iter,
                       l1_ratio=0.5, tol=1e-3)
    clf.fit(X, y)  # new params
    assert_almost_equal(0.5, clf.l1_ratio)
    assert_equal(50, clf.n_alphas)
    assert_equal(50, len(clf.alphas_))


def test_warm_start():
    X, y, _, _ = build_dataset()
    clf = ElasticNet(alpha=0.1, max_iter=5, warm_start=True)
    ignore_warnings(clf.fit)(X, y)
    ignore_warnings(clf.fit)(X, y)  # do a second round with 5 iterations

    clf2 = ElasticNet(alpha=0.1, max_iter=10)
    ignore_warnings(clf2.fit)(X, y)
    assert_array_almost_equal(clf2.coef_, clf.coef_, decimal=1)


def test_enet_alpha_warning():
    X = [[-1], [0], [1]]
    Y = [-1, 0, 1]       # just a straight line

    clf = ElasticNet(alpha=0)
    assert_warns(UserWarning, clf.fit, X, Y)


def test_uniform_targets():
    enet = ElasticNetCV(fit_intercept=True, n_alphas=3)

    rng = np.random.RandomState(0)

    X_train = rng.random_sample(size=(10, 3))
    X_test = rng.random_sample(size=(10, 3))

    y1 = np.empty(10)
    y2 = np.empty((10, 2))

    for y_values in (0, 5):
        y1.fill(y_values)
        assert_array_equal(enet.fit(X_train, y1).predict(X_test), y1)
        assert_array_equal(enet.alphas_, [np.finfo(float).resolution]*3)


def test_enet_readonly_data():
    X = np.array([[-1], [0], [1]])
    y = np.array([-1, 0, 1])   # just a straight line
    T = np.array([[2], [3], [4]])  # test sample
    with TempMemmap((X, y)) as (X, y):
        clf = ElasticNet(alpha=0.5)
        clf.fit(X, y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [0.454], decimal = 3)
        assert_array_almost_equal(pred, [0.909,  1.364,  1.818], decimal = 3)
        assert_almost_equal(clf.dual_gap_, 0, decimal = 3)


def test_enet_multitarget():
    n_targets = 3
    X, y, _, _ = build_dataset(n_samples=10, n_features=8,
                               n_informative_features=10, n_targets=n_targets)
    estimator = ElasticNet(alpha=0.01, fit_intercept=True)
    estimator.fit(X, y)
    coef, intercept, dual_gap = (estimator.coef_, estimator.intercept_,
                                 estimator.dual_gap_)

    for k in range(n_targets):
        estimator.fit(X, y[:, k])
        assert_array_almost_equal(coef[k, :], estimator.coef_)
        assert_array_almost_equal(intercept[k], estimator.intercept_)
        assert_array_almost_equal(dual_gap[k], estimator.dual_gap_)


def test_multioutput_enetcv_error():
    X = np.random.randn(10, 2)
    y = np.random.randn(10, 2)
    clf = ElasticNetCV()
    assert_raises(ValueError, clf.fit, X, y)


def test_precompute_invalid_argument():
    X, y, _, _ = build_dataset()
    clf = ElasticNetCV(precompute="invalid")

    assert_raises(ValueError, clf.fit, X, y)


def test_warm_start_convergence():
    X, y, _, _ = build_dataset()
    model = ElasticNet(alpha=1e-3, tol=1e-3).fit(X, y)
    n_iter_reference = model.n_iter_

    # This dataset is not trivial enough for the model to converge in one pass.
    assert_greater(n_iter_reference, 2)

    # Check that n_iter_ is invariant to multiple calls to fit
    # when warm_start=False, all else being equal.
    model.fit(X, y)
    n_iter_cold_start = model.n_iter_
    assert_equal(n_iter_cold_start, n_iter_reference)

    # Fit the same model again, using a warm start: the optimizer just performs
    # a single pass before checking that it has already converged
    model.set_params(warm_start=True)
    model.fit(X, y)
    n_iter_warm_start = model.n_iter_
    assert_equal(n_iter_warm_start, 1)


def test_warm_start_convergence_with_regularizer_decrement():
    boston = load_boston()
    X, y = boston.data, boston.target

    # Train a model to converge on a lightly regularized problem
    final_alpha = 1e-4
    low_reg_model = ElasticNet(alpha=final_alpha, max_iter = 500000).fit(X, y)

    # Fitting a new model on a more regularized version of the same problem.
    # Fitting with high regularization is easier it should converge faster
    # in general.
    high_reg_model = ElasticNet(alpha=final_alpha * 10, max_iter = 500000).fit(X, y)
    assert_greater(low_reg_model.n_iter_, high_reg_model.n_iter_)

    # Fit the solution to the original, less regularized version of the
    # problem but from the solution of the highly regularized variant of
    # the problem as a better starting point. This should also converge
    # faster than the original model that starts from zero.
    warm_low_reg_model = deepcopy(high_reg_model)
    warm_low_reg_model.set_params(warm_start=True, alpha=final_alpha)
    warm_low_reg_model.fit(X, y)
    assert_greater(low_reg_model.n_iter_, warm_low_reg_model.n_iter_)


def test_check_input_false():
    X, y, _, _ = build_dataset(n_samples=20, n_features=10)
    X = check_array(X, order='F', dtype='float64')
    y = check_array(X, order='F', dtype='float64')
    clf = ElasticNet(tol=1e-8)
    # Check that no error is raised if data is provided in the right format
    clf.fit(X, y, check_input=False)
    X = check_array(X, order='F', dtype='float32')
    clf.fit(X, y, check_input=True)
    # Check that an error is raised if data is provided in the wrong dtype,
    # because of check bypassing
    assert_raises(ValueError, clf.fit, X, y, check_input=False)

    # With no input checking, providing X in C order should result in false
    # computation
    X = check_array(X, order='C', dtype='float64')
    assert_raises(ValueError, clf.fit, X, y, check_input=False)


def test_overrided_gram_matrix():
    X, y, _, _ = build_dataset(n_samples=20, n_features=10)
    Gram = X.T.dot(X)
    clf = ElasticNet(tol=1e-8, precompute=Gram,
                     fit_intercept=True)
    assert_warns_message(UserWarning,
                         "Gram matrix was provided but X was centered"
                         " to fit intercept, "
                         "or X was normalized : recomputing Gram matrix.",
                         clf.fit, X, y)


def test_non_float_y():
    X = [[0, 0], [1, 1], [-1, -1]]
    y = [0, 1, 2]
    y_float = [0.0, 1.0, 2.0]

    clf = ElasticNet(fit_intercept=False)
    clf.fit(X, y)
    clf_float = ElasticNet(fit_intercept=False)
    clf_float.fit(X, y_float)
    assert_array_equal(clf.coef_, clf_float.coef_)
