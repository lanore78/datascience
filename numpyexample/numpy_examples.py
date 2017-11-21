
import numpy as np

class NumpyExample :

    def hello(self) :
        dir(np)
        a = np.arange(8).reshape(2, 2, 2)
        print(a)
        print(a.shape)
        print(a.ndim)
