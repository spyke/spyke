import numpy as np
import pyximport

#setup_args={'options': {'build_ext': {'compiler': 'mingw32'}}}

pyximport.install()
'''
setup_args={'include_dirs':[np.get_include()],
                              #'extra_compile_args':['-fopenmp'],
                              'libraries':[('gomp',)]}
                  )
'''

from testing import testing # .pyx file

ndi = np.array([9,19,29,39], dtype=np.uint32)
dims = np.array([10,20,30,40], dtype=np.uint32)
testing(ndi, dims)

