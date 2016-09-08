import numpy as np


outfile ="test.npy"
x = np.arange(10)
print x
np.save(outfile, x)
x=np.load(outfile)
print x