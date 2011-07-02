import pyximport

pyximport.install()

from testing import testing # .pyx file

#ndi = np.array([9,19,29,39], dtype=np.uint32)
#dims = np.array([10,20,30,40], dtype=np.uint32)
#testing(ndi, dims)
testing()
