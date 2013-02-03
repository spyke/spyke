import numpy as np
import pyximport
pyximport.install(build_in_temp=False, inplace=True)

import cy_select
#data = np.float32(np.random.rand(10000001)) # odd number of values ensures there's a middle value
#data = np.float32([1,2,3,4,5])
data = np.int16(np.random.randint(-2**15, 2**15-1, 26000001))
cy_select.selectpy(data)
