import numpy as np
import time
from numpy.lib.stride_tricks import as_strided as ast
import keyboard

image = np.arange(100).reshape((4, 25))
w = 25
h = 4
dw = 5
dh = 2
x = np.add.reduceat(
    np.add.reduceat(image, np.arange(0, h, dh), axis=0),
    np.arange(0, w, dw), axis=1
)
print(x)