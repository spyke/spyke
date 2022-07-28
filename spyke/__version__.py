"""Define __version__ and enforce minimum library versions"""

from __future__ import division
from __future__ import print_function

import os
import sys
from distutils.version import LooseVersion

__version__ = '2.1' # incremented mostly to track significant changes to sort file format

# enforce minimum versions of various required libraries:
LIBVERSIONS = {'Python': '3.6.8',
               'Qt5': '5.12.8',
               'PyQt5': '5.14.1',
               'PyOpenGL': '3.1.0',
               #'setuptools': '39.0.0', # for checking packaged versions
               'IPython': '7.4.0',
               'numpy': '1.17.2',
               'scipy': '1.3.1',
               'matplotlib': '3.0.3',
               'cython': '0.29.13',
               'mdp': '3.5',
               'sklearn': '0.21.3',
               'pywt': '1.0.3',
               'jsonpickle': '1.2',
               'simplejson': '3.16.0',
              }

PYVER = sys.version_info.major
if PYVER < 3:
    raise RuntimeError("spyke requires Python 3.x")

# map library names to pip/conda package names, for those few which are not identical:
LIBNAME2PKGNAME = {'pywt': 'PyWavelets',
                   'skimage': 'scikit-image',
                   'sklearn': 'scikit-learn'
                  }

def get_python_version(libname):
    return os.sys.version.split(' ')[0]

def get_qt5_version(libname):
    from PyQt5.QtCore import QT_VERSION_STR
    return QT_VERSION_STR

def get_pyqt5_version(libname):
    from PyQt5.QtCore import PYQT_VERSION_STR
    return PYQT_VERSION_STR

def get_pyopengl_version(libname):
    import OpenGL
    return OpenGL.version.__version__

def get_generic_version(libname):
    exec('import ' + libname) # import full library names into namespace
    ver = eval(libname + '.__version__') # recommended attrib, according to PEP8
    return ver

def get_generic_pkg_version(libname):
    import pkg_resources # installed by setuptools package
    ver = pkg_resources.get_distribution(libname).version # packaged version
    return ver

LIBNAME2VERF = {'Python': get_python_version,
                'Qt5': get_qt5_version,
                'PyQt5': get_pyqt5_version,
                'PyOpenGL': get_pyopengl_version,
               }

def check_LIBVERSIONS(verbose=False):
    """Check that all minimum version requirements in LIBVERSIONS are met"""
    for libname, minver in LIBVERSIONS.items():
        verf = LIBNAME2VERF.get(libname, get_generic_version)
        # get current version of libname:
        ver = verf(libname)
        if verbose:
            print(libname, ver)
        if ver < LooseVersion(minver):
            msg = ('Please update %s from version %s to at least version %s\n'
                   % (libname, ver, minver))
            if libname in LIBNAME2VERF:
                sln = ''
            else:
                if libname in LIBNAME2PKGNAME: # libname and install package name differ
                    pkgname = LIBNAME2PKGNAME[libname]
                else:
                    pkgname = libname
                sln = ('Run `sudo pip%d install --upgrade %s` or `conda update %s` '
                       'at the command line' % (PYVER, pkgname, pkgname))
            raise RuntimeError(msg+sln)
        if libname == 'jsonpickle' and ver != LooseVersion(minver):
            msg = ('spyke currently requires exactly jsonpickle version %s, version %s is '
                   'currently installed\n' % (minver, ver))
            sln = ('Run `sudo pip%d install jsonpickle==%s`'% (PYVER, minver))
            raise RuntimeError(msg+sln)
