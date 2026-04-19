import sys
import numpy as np

data = np.load("data/shorts.npy")  # shape: (N, 784)

if data.shape == (124970, 784):
    sys.exit(0)
else:
    sys.exit(1)
