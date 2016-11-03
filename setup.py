import os
from os.path import join

import numpy

from sklearn._build_utils import get_blas_info

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('prox_elasticnet', parent_package, top_path)

    cblas_libs, blas_info = get_blas_info()

    if os.name == 'posix':
        cblas_libs.append('m')
    
    # add cython extensions module
    config.add_extension('prox_fast', sources=['prox_fast.c'],
                         libraries=cblas_libs,
                         include_dirs=[join('..', 'src', 'cblas'),
                                       numpy.get_include(),
                                       blas_info.pop('include_dirs', [])],
                         extra_compile_args=blas_info.pop('extra_compile_args',
                                                          []), **blas_info)

    # add the test directory
    config.add_subpackage('tests')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    with open('README.rst') as f:
        readme = f.read()
    
    with open('LICENSE') as f:
        license = f.read()
    
    setup(
        version='0.0.1',
        description='A Python package which implements the Elastic Net using the proximal gradient method',
        long_description=readme,
        author='Neil G. Marchant',
        author_email='ngmarchant@gmail.com',
        url='https://github.com/ngmarchant/prox_elasticnet',
        license=license,
        **configuration(top_path='').todict())

