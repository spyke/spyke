"""Minimal script I like to run on python startup"""

try:
    shell.run('from __future__ import division')
except: # we're running in some environment where shell isn't defined, like ipython
    pass

import __main__

import os
import sys
import gc
import numpy as np

from copy import copy
from pprint import pprint
printraw = sys.stdout.write # useful for raw printing

from numpy.random import rand, randn, randint
from numpy import arange, array, array as ar, asarray, asarray as aar, log, log2, log10, sqrt, zeros, ones, diff, concatenate, concatenate as cat, mean, median, std

# set some numpy options
np.set_printoptions(precision=3)
np.set_printoptions(threshold=1000)
np.set_printoptions(edgeitems=5)
np.set_printoptions(linewidth=150)
np.set_printoptions(suppress=True)

def who():
    """Print all names in namespace"""
    import __main__
    print __main__.__dict__.keys()

def whos():
    """Print all names in namespace, with details"""
    exec 'pprint(locals())' in globals()

def cd(path):
    """Change directories"""
    path = path.replace('~', os.environ['HOME']) # make '~' a shortcut to my home
    if path == '..': # go down one directory
        path = '\\'.join(os.getcwd().split('\\')[0:-1])
    try:
        os.chdir(os.getcwd() + path) # path is relative?
    except OSError:
        os.chdir(path) # nope, path is absolute

def pwd():
    """Returns working directory"""
    return os.getcwd()

def ls():
    """Returns directory contents in a list"""
    print pwd()
    return os.listdir(os.getcwd())

def ll():
    """Prints long-list directory contents"""
    print pwd()
    pprint(os.listdir(os.getcwd()))
    #buf = StringIO.StringIO()
    #pprint(os.listdir(os.getcwd()), stream=buf)
    #return buf.getvalue()

def src(obj):
    """Print object's source code"""
    try:
        import inspect
        source = inspect.getsource(obj)
    except TypeError: # probalby a builtin
        print obj
        return
    except IOError: # probably entered interactively, no source file
        print obj
        return
    print inspect.getfile(obj) + ':'
    print
    print source

try:
    clr = shell.clear
except NameError: # we're running in some environment where shell isn't defined, like ipython
    pass

def unload(modname):
    """Deletes all modules with 'modname' in their file path (mod.__file__).
    Also, deletes any objects that depended on modname"""
    import __main__
    md = __main__.__dict__
    print 'deleting %s' % modname
    for key, mod in sys.modules.items():
        try:
            if mod.__file__.count(modname):
                #print 'deleting', mod
                del sys.modules[key]
        except AttributeError: # some modules don't have a .__file__ attrib
            pass
    for key in md.keys(): # for all names in the namespace
        if repr(md[key]).count(modname): # if modname shows up in this object's repr, ie if this object depends on modname
            del md[key] # delete the object
            #print 'deleted object:', key

def refresh(modname):
    """Deletes all modules with 'modname' in their file path (mod.__file__).
    Also, deletes any objects that depended on modname. Then re-imports 'modname' as a module.
    'modname' need not have been previously imported.
    WARNING: Seems to cause problems in IPython"""
    import __main__
    md = __main__.__dict__
    unload(modname)
    exec('import '+modname) # bad boy
    #__import__(modname, globals(), locals(), []) # this is supposed to be equivalent to "import modname", yet accepts a string for modname
    md[modname] = eval(modname) # add the newly import module to the global namespace
    print '%s refreshed' % modname
    #n = gc.collect() # do a full garbage collection
    #print 'garbage collected, there were %d unreachable objects' % n

def c():
    """Clears all names added to the namespace after the '_original' point.
    Don't try running this in IPython!"""
    import __main__
    md = __main__.__dict__
    for key in md.keys():
        if key not in _original and key != '_original':
            del md[key]
            #print 'deleted md key:', key
    print 'namespace cleared'

def cf():
    """Closes all figures. This needs to be extended to include all neuropy generated wx.Frames,
    not just matplotlib figures"""
    import pylab as pl
    pl.close('all')
    print 'all figures closed'

def pickle(obj, fname):
    import cPickle
    pf = file(fname, 'wb')
    p = cPickle.Pickler(pf, protocol=-1)
    obj = copy(obj) # work on a copy, don't modify the original
    p.dump(obj)
    pf.close()

def unpickle(fname):
    import cPickle
    pf = file(fname, 'rb')
    u = cPickle.Unpickler(pf)
    obj = u.load()
    pf.close()
    return obj


_original = __main__.__dict__.keys()
