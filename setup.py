"""spyke installation script"""

from distutils.core import setup, Extension
import os
from Cython.Distutils import build_ext

#detect_cy = Extension('spyke.detect_cy',
#                       sources=['spyke/detect_cy.pyx'])
cython_test = Extension('spyke.cython_test',
                       sources=['spyke/cython_test.pyx'],
                       include_dirs=['/bin/Python25/Lib/site-packages/numpy/core/include'])


setup(name = 'spyke',
      version = '0.1',
      description = 'Multichannel spike viewer and sorter for Swindale Lab .srf files',
      author = 'Martin Spacek, Reza Lotun',
      author_email = 'mspacek at interchange ubc ca',
      url = 'http://swindale.ecc.ubc.ca',
      #long_description = '',
      packages = ['spyke', 'spyke.gui', 'spyke.gui.res'],
      cmdclass = {'build_ext': build_ext},
      ext_modules = [cython_test],
      )
