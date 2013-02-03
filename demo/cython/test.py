import pyximport

pyximport.install(build_in_temp=False, inplace=True)

from testing import testing, testing2 # .pyx file

#ndi = np.array([9,19,29,39], dtype=np.uint32)
#dims = np.array([10,20,30,40], dtype=np.uint32)
#testing(ndi, dims)
#testing()
testing2()
