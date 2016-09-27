# Author: Neil G. Marchant <ngmarchant@gmail.com>
#
# Licence:

# Initialize Numpy C-API
import numpy as np
cimport numpy as np
np.import_array()

# Use to calculate optimal step size
from numpy.linalg import norm

cimport cython
from cpython cimport bool
import warnings

from libc.math cimport fabs
from libc.stdlib cimport malloc, free

ctypedef np.float64_t DOUBLE

# -----------------------------------------------------------------------------
#  External declarations to C libraries
# -----------------------------------------------------------------------------
cdef extern from "math.h":
    double c_sqrt "sqrt"(double m) nogil


cdef extern from "cblas.h":
    enum CBLAS_ORDER:
        CblasRowMajor=101
        CblasColMajor=102
    enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113
        AtlasConj=114

    # y <- alpha * x + y
    void daxpy "cblas_daxpy"(int N, double alpha, double *X, int incX,
                             double *Y, int incY) nogil

    # out <- x^T y
    double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY
                             ) nogil

    # out <- norm(x,1)
    double dasum "cblas_dasum"(int N, double *X, int incX) nogil

    # A <- A + alpha * x * y^T
    void dger "cblas_dger"(CBLAS_ORDER Order, int M, int N, double alpha,
                double *X, int incX, double *Y, int incY, double *A, int lda) nogil

    # y <- alpha * A^T .* x + beta y
    void dgemv "cblas_dgemv"(CBLAS_ORDER Order,
                      CBLAS_TRANSPOSE TransA, int M, int N,
                      double alpha, double *A, int lda,
                      double *X, int incX, double beta,
                      double *Y, int incY) nogil

    # out <- norm(x, 2)
    double dnrm2 "cblas_dnrm2"(int N, double *X, int incX) nogil

    # y <- x
    void dcopy "cblas_dcopy"(int N, double *X, int incX, double *Y, int incY) nogil

    # x <- alpha * x
    void dscal "cblas_dscal"(int N, double alpha, double *X, int incX) nogil

    # C <- alpha*op(A)*op(B) + beta*C
    void dgemm "cblas_dgemm"(CBLAS_ORDER Order,
                       CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                       int m, int n, int k, double alpha, double *a, int lda,
                       double *b, int ldb, double beta, double *c, int ldc) nogil

# -----------------------------------------------------------------------------
#  Functions implemented in C (for speed)
# -----------------------------------------------------------------------------
cdef inline double fmax(double x, double y) nogil:
    """ Returns max(x, y) """
    if x > y:
        return x
    return y


cdef inline double fmin(double x, double y) nogil:
    """ Returns min(x, y) """
    if x < y:
        return x
    return y


cdef inline double abs_max(int n, double* a) nogil:
    """ Returns the maximum of the element-wise absolute value of vector a """
    cdef int i
    cdef double m = fabs(a[0])
    cdef double d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef inline double vmax(int n, double* a) nogil:
    """ Returns the maximum element of vector a """
    cdef int i
    cdef double m = a[0]
    cdef double d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


cdef inline double sq_loss(int n_features, int n_samples, double *X, double *y,
    double *v) nogil:
    """Evaluates the square loss function at vector v.
    The calculation performed is l <- 0.5 * norm(y - X v, 2)^2.
    """
    cdef double result
    cdef double *r = <double *> malloc(n_samples * sizeof(double))
    if not r:
        with gil:
            raise MemoryError()

    dgemv(CblasColMajor, CblasNoTrans, n_samples, n_features, -1.0, X,
          n_samples, v, 1, 0, r, 1)
    daxpy(n_samples, 1.0, y, 1, r, 1)
    result = 0.5 * ddot(n_samples, r, 1, r, 1)

    free(r)

    return result


cdef inline void grad_sq_loss(int n_features, double *XtX, double *Xty,
    double *v, double *g) nogil:
    """Evaluates the gradient of the square loss function at vector v and
    stores the result in g.
    The calculation performed is: g <- XtX * v - Xty.
    """
    dgemv(CblasRowMajor, CblasNoTrans, n_features, n_features, 1.0, XtX,
          n_features, v, 1, 0, g, 1)
    daxpy(n_features, -1.0, Xty, 1, g, 1)


cdef inline void soft_threshold(double thres, int n_features, double *v) nogil:
    """Applys the soft-thresholding operator S_thres to vector v.
    """
    cdef int i
    for i in range(0,n_features):
        if v[i] >= thres:
            v[i] = v[i] - thres
        elif v[i] <= -thres:
            v[i] = v[i] + thres
        else:
            v[i] = 0


cdef inline double sufficient_decrease(int n_features, int n_samples, double *X,
    double *y, double* v, double* w, double *g, double sigma) nogil:
    """ Checks whether there is a sufficient decrease in the objective for
    step size sigma.
    Returns True if (Term 1 + Term 2 + Term 3 > 0)
    where   Term 1 = sq_loss(w) - sq_loss(v),
            Term 2 = - (w - v)^T * g
            Term 3 = - norm(w - v, 2)^2/(2*sigma)
    Else returns False.
    """
    cdef double term1, term2, term3
    cdef double *wvdiff = <double *> malloc(n_features * sizeof(double))
    if not wvdiff:
        with gil:
            raise MemoryError()

    # term1 <- sq_loss(w) - sq_loss(v)
    term1 = sq_loss(n_features, n_samples, X, y, w) - sq_loss(n_features,
        n_samples, X, y, v)

    # wvdiff <- w - v
    dcopy(n_features, w, 1, wvdiff, 1)
    daxpy(n_features, -1.0, v, 1, wvdiff, 1)

    # term2 <- -(wvdiff)^T g
    term2 = - ddot(n_features, wvdiff, 1, g, 1)

    # term3 <- - norm(w - v, 2)^2/(2*sigma)
    term3 = - ddot(n_features, wvdiff, 1, wvdiff, 1) / (2 * sigma)

    free(wvdiff)

    return term1 + term2 + term3


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def enet_prox_gradient(np.ndarray[DOUBLE, ndim=1] w,
                            double alpha, double beta,
                            np.ndarray[DOUBLE, ndim=2, mode='fortran'] X,
                            np.ndarray[DOUBLE, ndim=1, mode='c'] y,
                            int max_iter, double tol):
    """Cython implementation of Fast Iterative Shrinkage Thresholding Algorithm
    (FISTA) for elastic-net regression.

        The objective function is:

        (1/2) * norm(X*w - y, 2)^2 + alpha norm(w, 1) + (beta/2) norm(w, 2)^2

    Parameters
    ----------
    w : np.ndarray, shape (n_features,)
        Initial guess for weights

    alpha : double
            l-1 regularization parameter

    beta : double
           l-2 regularization parameter

    X : Fortan-contiguous np.ndarray, shape (n_samples, n_features)
        Training data

    y : C-contiguous np.ndarray, shape (n_samples,)
        Target values

    max_iter : int
               Maximum number of iterations

    tol : double
          Stop if relative error < tol

    Returns
    -------
    w : np.ndarray, shape (n_parameters,)
        Optimal weight vector.

    rel_error : double
                Relative error achieved.

    n_iter : int
             Number of iterations performed.
    """

    # Initialize C variables
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]
    cdef int n_iter = 0
    cdef double t = 1.0
    cdef double t_prev
    cdef double sigma
    cdef double shrink_factor
    cdef double accel_parameter
    cdef double min_w_max
    cdef double d_w_max
    cdef double rel_error = float('inf')
    cdef np.ndarray[DOUBLE, ndim=1] v = np.empty(n_features)
    cdef np.ndarray[DOUBLE, ndim=1] g = np.empty(n_features)
    cdef np.ndarray[DOUBLE, ndim=1] delta_w = np.empty(n_features)
    cdef np.ndarray[DOUBLE, ndim=1] w_prev = np.empty(n_features)
    cdef np.ndarray[DOUBLE, ndim=2, mode='c'] XtX = np.empty([n_features, n_features])
    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] Xty = np.empty(n_features)

    # XtX <- X^T*X
    dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_features, n_features,
          n_samples, 1.0, <DOUBLE*>X.data, n_samples, <DOUBLE*>X.data, n_samples, 0,
          <DOUBLE*>XtX.data, n_features)

    # Xty <- X^T*y
    dgemv(CblasColMajor, CblasTrans, n_samples, n_features, 1.0, <DOUBLE*>X.data,
          n_samples, <DOUBLE*>y.data, 1, 0, <DOUBLE*>Xty.data, 1)

    # v <- w
    dcopy(n_features, <DOUBLE*>w.data, 1, <DOUBLE*>v.data, 1)

    # Determine optimal step size (1/Lipschitz constant)
    sigma = 1/norm(XtX, ord=2)

    # Set shrink factor
    shrink_factor = 1.0/(1.0 + sigma * beta)

    dcopy(n_features, <DOUBLE*>w.data, 1, <DOUBLE*>v.data, 1)
    dscal(n_features, 0.8, <DOUBLE*>v.data, 1)
    grad_sq_loss(n_features, <DOUBLE*>XtX.data, <DOUBLE*>Xty.data, <DOUBLE*>v.data,
                         <DOUBLE*>g.data)

    # Testing adaptive step size
    cdef double eta = 0.5
    sigma = 1
    cdef int m = 0
    print("Step size\n")
    print(sigma)

    with nogil:
        for n_iter in range(max_iter):
            # w_prev <- w
            dcopy(n_features, <DOUBLE*>w.data, 1, <DOUBLE*>w_prev.data, 1)

            # w <- v - sigma * (X^T * X * v - X^T * y)
            grad_sq_loss(n_features, <DOUBLE*>XtX.data, <DOUBLE*>Xty.data, <DOUBLE*>v.data,
                         <DOUBLE*>g.data)
            dcopy(n_features, <DOUBLE*>v.data, 1, <DOUBLE*>w.data, 1)
            daxpy(n_features, -sigma, <DOUBLE*>g.data, 1, <DOUBLE*>w.data, 1)

            # w <- S_{sigma*alpha}(w)
            soft_threshold(sigma*alpha, n_features, <DOUBLE*>w.data)

            # w <- shrink_factor * w
            dscal(n_features, shrink_factor, <DOUBLE*>w.data, 1)

            # Check for a sufficient decrease in
            while sufficient_decrease(n_features, n_samples,
                <DOUBLE*>X.data, <DOUBLE*> y.data, <DOUBLE*> v.data,
                <DOUBLE*> w.data, <DOUBLE*> g.data, sigma) > 0 :
                # Try smaller step size
                sigma = eta * sigma
                # Evaluate prox-grad step again with smaller sigma
                # w <- v - sigma * g
                dcopy(n_features, <DOUBLE*>v.data, 1, <DOUBLE*>w.data, 1)
                daxpy(n_features, -sigma, <DOUBLE*>g.data, 1, <DOUBLE*>w.data, 1)

                # w <- S_{sigma*alpha}(w)
                soft_threshold(sigma*alpha, n_features, <DOUBLE*>w.data)

                # w <- shrink_factor * w
                dscal(n_features, shrink_factor, <DOUBLE*>w.data, 1)

            # delta_w <- w - w_prev
            dcopy(n_features, <DOUBLE*>w.data, 1, <DOUBLE*>delta_w.data, 1)
            daxpy(n_features, -1.0, <DOUBLE*>w_prev.data, 1, <DOUBLE*>delta_w.data, 1)

            # calculate relative error
            min_w_max = fmin(vmax(n_features, <DOUBLE*>w.data),
                             vmax(n_features, <DOUBLE*>w_prev.data))
            d_w_max = abs_max(n_features, <DOUBLE*>delta_w.data)
            rel_error = d_w_max/min_w_max
            if rel_error < tol:
            	# Have converged to within tolerance
                break

            # update acceleration parameter
            t_prev = t
            t = (1.0 + c_sqrt(1.0 + 4.0 * t_prev * t_prev))/2.0
            accel_parameter = (t_prev - 1.0)/t

            # v <- w + accel_parameter * delta_w
            dcopy(n_features, <DOUBLE*>w.data, 1, <DOUBLE*>v.data, 1)
            daxpy(n_features, accel_parameter, <DOUBLE*>delta_w.data, 1, <DOUBLE*>v.data, 1)

    return w, rel_error, n_iter + 1
