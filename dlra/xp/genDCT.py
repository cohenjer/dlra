import numpy as np
from scipy.fftpack import dct
from numpy.matlib import repmat


def genDCT(dims, fact):
    '''
    Generates Discrete Consine truncated Transformations for the given sizes.
    fact decides on each mode how much overcompleteness we want.
    '''

    # Initialisation of the dictionary
    # Dictionary sizes
    di = fact*dims
    # Generating the DCT matrices
    D1 = dct(np.eye(di[0]))
    D2 = dct(np.eye(di[1]))
    # Truncating the DCT matrices
    D1 = D1[0:dims[0], :]
    D2 = D2[0:dims[1], :]
    # Normalizing after truncation
    D1 = D1*repmat(1/np.sqrt(np.sum(D1**2, 0)), dims[0], 1)
    D2 = D2*repmat(1/np.sqrt(np.sum(D2**2, 0)), dims[1], 1)
    # Creating the big dictionary (already normalized)
    Do = np.kron(D2, D1)
    return Do
