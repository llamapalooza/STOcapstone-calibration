from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='calibration',
    version='0.1',
    packages=[
        '', # The "root package": modules in the current directory
        'hESS',
        'simulation'
    ],
    ext_modules=cythonize('*.pyx'),
    
    requires=[
        "numpy (>= 1.6)",
        "scipy (>= 0.10)",
        "cv2 (>= 2.3)",     # OpenCV "new" Python bindings
        "cython (>= 0.14)",
        "sphinx",           # Only required for documentation. Comment out if not needed.
    ],
    
    include_dirs = [numpy.get_include()],
)
