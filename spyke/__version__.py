"""Define __version__ and enforce minimum library versions"""

from __future__ import division
from __future__ import print_function

import os
import sys
from distutils.version import LooseVersion

__version__ = '2.0' # incremented mostly to track significant changes to sort file format

# enforce minimum versions of various required libraries:
PY2_LIBVERSIONS = {'Python': '2.7.15',
                   'Qt4': '4.8.7',
                   'PyQt4': '4.12.1',
                   'PyOpenGL': '3.1.0',
                   #'setuptools': '39.0.0', # for checking packaged versions
                   'IPython': '5.8.0',
                   'numpy': '1.16.5',
                   'scipy': '1.2.2',
                   'matplotlib': '2.2.3',
                   'cython': '0.29.13',
                   'mdp': '3.5',
                   'sklearn': '0.18.1',
                   'pywt': '1.0.3',
                   'jsonpickle': '1.2',
                   'simplejson': '3.16.0',
                  }

PY3_LIBVERSIONS = {'Python': '3.6.8',
                   'Qt4': '4.8.7',
                   'PyQt4': '4.12.1',
                   'PyOpenGL': '3.1.0',
                   #'setuptools': '39.0.0', # for checking packaged versions
                   'IPython': '7.4.0',
                   'numpy': '1.17.2',
                   'scipy': '1.3.1',
                   'matplotlib': '3.0.3',
                   'cython': '0.29.13',
                   'mdp': '3.5',
                   'sklearn': '0.21.1',
                   'pywt': '1.0.3',
                   'jsonpickle': '1.2',
                   'simplejson': '3.16.0',
                  }

PYVER = sys.version_info.major
PYVER2LIBVERSIONS = {2: PY2_LIBVERSIONS,
                     3: PY3_LIBVERSIONS}
LIBVERSIONS = PYVER2LIBVERSIONS[PYVER]

def get_python_version(libname):
    return os.sys.version.split(' ')[0]

def get_qt4_version(libname):
    from PyQt4.QtCore import QT_VERSION_STR
    return QT_VERSION_STR

def get_pyqt4_version(libname):
    from PyQt4.pyqtconfig import Configuration
    cfg = Configuration()
    return cfg.pyqt_version_str

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
                'Qt4': get_qt4_version,
                'PyQt4': get_pyqt4_version,
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
                sln = ('Run `sudo pip%d install --upgrade %s` or `conda update %s` '
                       'at the command line' % (PYVER, libname, libname))
            raise RuntimeError(msg+sln)
