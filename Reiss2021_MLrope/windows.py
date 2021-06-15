import pandas as pds
import datetime
import numpy as np
import time
import event as evt

'''
Utilities used for transforming a dataset in sliding window in a smart way,
less memory costing than what was done for the sliding windows
'''


def windowed(X, window):
    '''
    Using stride tricks to create a windowed view on the original
    data.
    '''
    shape = int((X.shape[0] - window) + 1), window, X.shape[1]
    strides = (X.strides[0],) + X.strides
    X_windowed = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    return X_windowed

