import pandas as pds
import numpy as np


def get_weights(serie, bins):
    a, b = np.histogram(serie, bins=bins)
    weights = 1/(a[1:]/np.sum(a[1:]))
    weights = np.insert(weights, 0,1)
    weights_Serie = pds.Series(index = serie.index, data=1)
    for i in range(1, bins):
        weights_Serie[(serie>b[i]) & (serie<b[i+1])] = weights[i]
    return weights_Serie