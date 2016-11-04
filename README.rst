###############
prox_elasticnet
###############

A python package which implements the Elastic Net using the proximal gradient 
method.


Installation
============
To install from source, you must have the Python development headers, a C/C++ 
compiler, the ATLAS libraries and the following Python packages:
* NumPy
* SciPy
* sci-kit learn

To install these dependencies on a Debian-based system, run::

    sudo apt-get install build-essential python-dev python-setuptools \
                     python-numpy python-scipy python-sklearn \
                     libatlas-dev libatlas3gf-base

(replace ``python`` with ``python3`` if using Python 3).

On RHEL/Fedora and derivatives, run::

    sudo dnf install make automake gcc gcc-c++ python3-devel \
                 python3-setuptools python3-numpy python3-scipy \
                 python3-sklearn atlas-devel


Once these dependencies are installed, change to the directory of the source
code and run::

    make install

This will install the package into your Python user site directory.

Testing
=======
To run the unit tests, change to the directory of the source code and run::

    make test


Quickstart
==========
``prox_elasticnet`` uses the same syntax as sci-kit learn. Below is a toy 
example, demonstrating how to fit a model to a one-dimensional training 
dataset and make predictions on a test set. ::

    import numpy as np
    from prox_elasticnet import ElasticNet
    
    X_train = np.arange(100)
    y_train = X_train + np.random.normal(0,1,len(X))
    X_train = X_train[:, np.newaxis]
    X_test = np.arange(0.5, 10.5)
    X_test = X_test[:, np.newaxis]
    
    model = ElasticNet(alpha = 0.1).fit(X_train,y_train)
    model.predict(X_test)


Help
====
Documentation can be accessed from within a Python session by entering::

    `help(prox_elasticnet.ElasticNet)`

or::

    `help(prox_elasticnet.ElasticNet)`
    
You can also check out the Jupyter notebook `Demo.ipynb` included with the 
source code.
