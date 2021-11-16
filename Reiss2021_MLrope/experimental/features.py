import scipy.constants as constants
import pandas as pds
import numpy as np

def computeBetawiki(data):
    '''
    compute Beta according to wiki
    '''
    try:
        data['beta'] = 1e6*data['np']*constants.Boltzmann*data['tp']/(np.square(1e-9*data['bt'])/(2*constants.mu_0))
    except KeyError:
        print('KeyError')
    return data
                                                               
def computePdyn(data):
    '''
    compute the evolution of the Beta for data
    data is a Pandas dataframe
    the function assume data already has ['Np','V'] features
    '''
    try:
        data['Pdyn'] = 1e12*constants.m_p*data['np']*data['vt']**2
    except KeyError:
        print('Error computing Pdyn, V or Np might not be loaded '
              'in dataframe')
        
def computeTexrat(data):
    '''
    compute the ratio of Tp/Tex
    '''
    try:
        data['texrat'] = data['tp']*1e-3/(np.square(0.031*data['vt']-5.1))
    except KeyError:
        print( 'Error computing Texrat')
        

        
def computeRollingStd(data, timeWindow, feature, center=False):
    '''
    for a given dataframe, compute the standard dev over
    a defined period of time (timeWindow) of a defined feature*
    ARGS :
    data : dataframe
    feature : feature in the dataframe we wish to compute the rolling mean from
                (string format)
    timeWindow : string that defines the length of the timeWindow (see pds doc)
    center : boolean to indicate if the point of the dataframe considered is
    center or end of the window
    '''
    name = feature+'std'
    data[name] = data[feature].rolling(timeWindow, center=center,
                                       min_periods=1).std()
    return data