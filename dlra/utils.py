import numpy as np
import math
from scipy.interpolate import BSpline

def gen_BSplines(n, size_step, deg, shift=0):
    """
    Generates Bsplines on [1,n] where the knots are spaces by size_step and the polynomials are degree deg. Results are stored in a matrix D. Support of atoms has size (1+deg)*size_step. By default the grid for knots is [0,size_step, 2*size_step...] but it can be shifted.

    Parameters
    ----------
    n : int
        The splines are valued on [1,n]

    size-step : int
        The number of indices in [1,n] between each knot

    deg : int
        degree of the splines

    shift : int (default 0)
        Set a shift for the origin of the splines

    Returns
    -------
    D : numpy array
        A matrix containing the B-splines columnwise
    """

    n_over_m = size_step
    m = int(n/n_over_m + 2*(deg-1)+1) # adding left and right padding
    t = shift+np.linspace(-(deg-1)*n_over_m,n+(deg-1)*n_over_m,m)
    mySplines = BSpline(t,np.ones(n),deg)

    # todo: start before 0 to have nontrivial elements on the borders
    D = []
    for i in range(0,m-deg):
        Bspline = mySplines.basis_element(t[i:i+deg], extrapolate=False)
        D.append(Bspline(np.linspace(1,n,n)))
    D = np.array(D).T
    D[np.isnan(D)]=0
    return D


def sam(x,y):
    """
    Computes the Spectral Angular Mapper between two spectra x and y.

    Parameters
    ----------
    x : numpy array
        a spectra to compare with y

    y : numpy array
        a spectra to compare with x

    Returns
    -------
    out : float
        the SAM between x and y
    """

    return math.acos(np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y))
