import numpy as np
import tempfile
import matplotlib.pyplot as plt
import os

def viz_data_slice(x, field_index = 0):
    half_size = int(np.ceil(np.shape(x)[3]/2))
    n_fields = int(np.shape(x)[4])
    fidx = np.clip(int(field_index), 0, n_fields)
    d2_tensor = x[0, :, :, half_size, fidx]
    plt.imshow(d2_tensor, interpolation='bilinear')
    plt.show()



def check_path(path):
    is_valid = os.path.isdir(path)
    if is_valid:
        return path
    else:
        return '../'
