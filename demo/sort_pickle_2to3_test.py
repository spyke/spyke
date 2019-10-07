pstring = f.read()
pbytes = bytes(pstring, encoding='latin1')
pickle.loads(pbytes, encoding='bytes')


#############




def get_data():
    import numpy as np
    import datetime
    dtype = [('x', np.uint16), ('y', np.float32)]
    arr = np.ones((2, 10), dtype=dtype)
    dt = datetime.datetime(2015, 10, 12, 13, 17, 42, 123456)
    data = {"ascii":"abc", "text":u"äbc", "intval":42, "arr":arr, "dt":dt}
    return data

## pickle:

## PY2:
import cPickle as pickle
data = get_data()
with open("pickle.test", "wb") as f:
    pickle.dump(data, f, protocol=-1)

## PY3:
import pickle
with open("pickle.test", "rb") as f:
    data = pickle.load(f) ## FAILS!
#UnicodeDecodeError: 'ascii' codec can't decode byte 0x80 in position 4: ordinal not in range(128)

## jsonpickle:

## PY2:
import jsonpickle
data = get_data()
with open("data.json", "w") as f:
    f.write(jsonpickle.encode(data))
>>> data
{'arr': array([[(1, 1.), (1, 1.), (1, 1.), (1, 1.), (1, 1.), (1, 1.), (1, 1.),
         (1, 1.), (1, 1.), (1, 1.)],
        [(1, 1.), (1, 1.), (1, 1.), (1, 1.), (1, 1.), (1, 1.), (1, 1.),
         (1, 1.), (1, 1.), (1, 1.)]], dtype=[('x', '<u2'), ('y', '<f4')]),
 'ascii': 'abc',
 'dt': datetime.datetime(2015, 10, 12, 13, 17, 42, 123456),
 'intval': 42,
 'text': u'\xe4bc'}

## PY3:
import jsonpickle
with open("data.json", "r") as f:
    data = jsonpickle.decode(f.read()) ## WORKS!
>>> data
{'arr': array([[(1, 1.), (1, 1.), (1, 1.), (1, 1.), (1, 1.), (1, 1.), (1, 1.),
         (1, 1.), (1, 1.), (1, 1.)],
        [(1, 1.), (1, 1.), (1, 1.), (1, 1.), (1, 1.), (1, 1.), (1, 1.),
         (1, 1.), (1, 1.), (1, 1.)]], dtype=[('x', '<u2'), ('y', '<f4')]),
 'ascii': 'abc',
 'dt': datetime.datetime(2015, 10, 12, 13, 17, 42, 123456),
 'intval': 42,
 'text': 'äbc'}
