from __future__ import division

import numpy as np
import ROOT

from matplotlib import pyplot as plt


def single_fit(
        spec, 
        fix=list(), 
        values=list(), 
        bounds=dict(),
        randomize=False, 
        nmax=100,
        tol=1e-2):
    """
    Perform a single fit using TMinuit.

    :param spec: ParSpec
        sepectrum object to use in the fit
    :param fix: [str]
        name of parameters to fix in the fit
    :param values: [float]
        custom central values
    :param bounds: {str: (float, float)}
        map parameter names to alternate down, up bounds
    :param randomize: bool
        randomize initial starting parameter values
    :param nmax: int
        maximum TMinuit fails before aborting
    :return: [float], float, ROOT.TMinimizer
        fit values, ll and minimizer object used to fit
    """
    # Do a vanilla fit (don't randomize parameters)
    minimizer = spec.build_minimizer()
    minimizer.SetTolerance(tol)

    # copy so they can be edited
    lows = np.array(spec.lows)
    highs = np.array(spec.highs)

    central = spec.central
    constraints = spec.constraints

    # Revised scales for all parameters
    for par, bound in bounds:
        ipar = spec.ipar(par)
        lows[ipar] = central[ipar] - abs(bound[0])
        highs[ipar] = central[ipar] + abs(bound[1])
        scale = abs(bound[1]-bound[0])
        minimizer.SetVariableStepSize(ipar, 1e-2*scale)

    # Build mask for parmaeters to fix, and indicate in minimizer
    ifixs = np.zeros(spec.npars, dtype=bool)
    for par in fix:
        ipar = spec.ipar(par)
        minimizer.FixVariable(ipar)
        ifixs[ipar] = True

    # Revised central values for all parameters
    if values:
        for i in range(spec.npars):
            minimizer.SetVariableValue(i, values[i])
    else:
        values = spec.central

    if randomize:
        x = spec.randomize_parameters(
            values, central, lows, highs, constraints)
        for i in range(spec.npars):
            minimizer.SetVariableValue(i, x[i])

    # Attempt the fit, TMinuit will fail sometimes
    nfails = 0  # keep track of failed fits
    while not minimizer.Minimize():
        nfails += 1
        if nfails >= nmax:
            raise RuntimeError("Failed minimization")
        if randomize:
            x = spec.randomize_parameters(
                values, central, lows, highs, constraints)
            for i in range(spec.npars):
                minimizer.SetVariableValue(i, x[i])

    minx = [minimizer.X()[i] for i in range(spec.npars)]
    ll = spec.ll(minx)

    return minx, ll, minimizer


def global_fit(spec, nfits=10, nmax=100, **kwargs):
    """
    Perform multiple fits and keep the best minimum.

    Accepts the same keyword arguments as `single_fit`.

    :param spec: ParSpec
        sepectrum object to use in the fit
    :param nfits: int
        number of successfull fit attempts from which to find a global minimum
    :return: [float], float, ROOT.TMinimizer
        fit parameters, ll and minimizer object used to fit
    """

    # Take control of the randomize parameter when passing along to single fit.
    # Note that nmax ia also not propagated as it is captured by this function.
    if 'randomize' in kwargs:
        del kwargs['randomize']

    best_x = None  # parameter values at global min
    best_ll = float('-inf')  # log likelihood at global min
    best_min = None  # keep minimizer object which reaches best min

    npass = 0
    nfails = 0

    while npass < nfits:
        try:
            minx, ll, minimizer = single_fit(spec, randomize=True, **kwargs)
            npass += 1  # once it succeeds, count the fit
        except RuntimeError:
            nfails += 1
            if nfails >= nmax:
                raise RuntimeError("Failed global minimization")
            continue  # if it fails, try again with different randomization

        if ll > best_ll:
            best_x = minx
            best_ll = ll
            best_min = minimizer

    return best_x, best_ll, best_min


def run_minos(spec, minimizer, pars=list(), verbose=False):
    """
    Find the points along each parameter value where the log likelihood is
    halved. For a normal distribution, this is the 1-sigma interval containing
    68.27% of the distribution.

    :param spec: ParSpec
        spectrum whose spectrum parameters are to be profiled
    :param minimizer: ROOT.TMinimizer
        minimier object which has found a minimum
    :param pars: [str]
        list of parameters on which to run Minos, or all if list is empty
    :param verbose: bool
        print information about parameter evaluations
    :return: [float], [float]
        distance to subtract and add to halve the log likelihood
    """
    if len(pars) == 0:
        pars = spec.pars

    # Lower and upper bounds for the parameters
    npars = len(pars)
    fit_down = [0] * npars
    fit_up = [0] * npars

    # Declare ROOT doubles which minos can write to by reference
    down = ROOT.Double(0)
    up = ROOT.Double(0)

    for i, par in enumerate(pars):
        ipar = spec.ipar(par)
        if minimizer.GetMinosError(ipar, down, up):
            # Note: important to cast the copy the ROOT variable, otherwise
            # the list will contain a reference to the value, which will change
            fit_down[i] = float(down)
            fit_up[i] = float(up)
        else:
            warnings.warn("Minos failed on %s" % par, RuntimeWarning)
        if verbose:
            print('...%s: %+.2e, %+.2e' % (par, float(down), float(up)))

    return fit_down, fit_up


def find_minima(spec, nsample=100, tol=1e-2, verbose=False, **kwargs):
    """
    Find individual local minima in the space.

    :param nsample: int
        number of samples of the likelihood space to explore minima
    :param tol: float
        consider two log likelihoods belong to different minima if they differ
        by more than this value
    :param verbose: bool
        print information about fit success
    :return: [float], [[float]], [[float]], float
        log likelihood at each minimum
        fit values of parameters at each minimum
        differenc of fit values to global minimum, scaled by uncertainty
        probability of finding the global minimum given a random initial point
    """
    if nsample <= 0:
        raise ValueError("Invalid number of samples")

    xs = list()  # minimized parameters of each fit
    lls = list()  # log likelihood of each fit

    best_ll = float('-inf')
    best_min = None   # keep track of the minimizer that reaches global

    isample = 0
    nfails = 0
    while isample < nsample:
        try:
            minx, ll, minimizer = single_fit(
                spec, 
                randomize=True if nfails+isample>0 else False, 
                nmax=1, 
                tol=tol,
                **kwargs)
            if verbose:
                print("...sample %d: %.2e" % (isample, ll))
        except RuntimeError:
            if verbose:
                print("...fit failed")
            nfails += 1
            continue

        isample += 1
        xs.append(minx)
        lls.append(ll)

        if ll > best_ll:
            best_min = minimizer
            best_ll = ll

    if verbose:
        print("...failure rate: %.2e" % (nfails/float(nsample+nfails)))

    # Compute the error for each parameter
    if not best_min.Hesse():
        warnings.warn("Failed to compute error marix", RuntimeWarning)
    errs = [best_min.Errors()[i] for i in range(spec.npars)]

    xs = np.array(xs)
    lls = np.array(lls)

    isort = np.argsort(lls)[::-1]
    xs = xs[isort]
    lls = lls[isort]

    # Build array of booleans, True if the log likelihood difference between
    # consecutive (sorted) minima exceeds the tolerance, False otherwise
    mins = np.fabs(lls[1:]-lls[:-1]) > tol
    # Convert to a list of indices of individual local minima from the samples
    imins = np.arange(1, nsample)[mins]
    imins = [0] + list(imins)

    # The number of samples that landed in the global minimum
    nglobal = nsample if len(imins) == 1 else imins[1]

    min_ll = [lls[i] for i in imins]
    min_x = [list(xs[i]) for i in imins]
    min_rel = [list((xs[i]-xs[0])/errs) for i in imins]

    return min_ll, min_x, min_rel, nglobal/nsample


def _make_bounds(spec, ipar, low, high):
    if low is None:
        low = spec.central[ipar]-spec.scales[ipar]*2
    if high is None:
        high = spec.central[ipar]+spec.scales[ipar]*2

    if low == high:
        low -= 1 
        high += 1

    return low, high


def slice1d(spec, x, par, low=None, high=None, nsteps=100):
    ipar = spec.ipar(par)
    low, high = _make_bounds(spec, ipar, low, high)
    vals = np.linspace(low, high, nsteps)

    lls = list()
    x = np.array(x)
    for i, val in enumerate(vals):
        x[ipar] = val
        lls.append((val, spec.ll(x)))

    return np.array(lls)


def slice2d(
        spec, 
        x, 
        par1, 
        par2, 
        low1=None, 
        high1=None, 
        low2=None, 
        high2=None, 
        nsteps=100):
    ipar1 = spec.ipar(par1)
    ipar2 = spec.ipar(par2)

    low1, high1 = _make_bounds(spec, ipar1, low1, high1)
    vals1 = np.linspace(low1, high1, nsteps)

    low2, high2 = _make_bounds(spec, ipar2, low2, high2)
    vals2 = np.linspace(low2, high2, nsteps)

    vals = [(v1, v2) for v1 in vals1 for v2 in vals2]

    lls = list()
    x = np.array(x)
    
    for i1, v1 in enumerate(vals1):
        x[ipar1] = v1
        for i2, v2 in enumerate(vals2):
            x[ipar2] = v2
            lls.append((v1, v2, spec.ll(x)))

    return np.array(lls)


def profile(
        spec, 
        par, 
        low=None, 
        high=None, 
        nsteps=100,
        **kwargs):
    ipar = spec.ipar(par)
    low, high = _make_bounds(spec, ipar, low, high)
    vals = np.linspace(low, high, nsteps)

    best, best_ll, minimizer = single_fit(spec, **kwargs)

    minimizer.FixVariable(ipar)

    lls = list()
    new_ll = best_ll
    new_x = best
    for i, val in enumerate(vals):
        minimizer.SetVariableValue(ipar, val)

        if not minimizer.Minimize():
            continue

        minx = [minimizer.X()[i] for i in range(spec.npars)]
        ll = spec.ll(minx)
        if ll > new_ll:
            new_ll = ll
            new_x = minx
        print(val, ll)
        lls.append((val, np.exp(ll)))

    return np.array(lls[1:])


def covm(minimizer):
    npars = int(minimizer.NDim())
    return np.array(
        [[minimizer.CovMatrix(i,j) 
        for j in range(npars)]
        for i in range(npars)])


def corrcoef(covm):
    covm = np.asarray(covm)
    sigs = np.diag(covm)**0.5  # grab the stdevs

    norms = np.copy(sigs)  # normalization for correlation matrix
    norms[~(sigs>0)] = 1  # ignore parameters with no var
    # Build correlation matrix
    corr = covm / (norms[np.newaxis, :] * norms[:, np.newaxis])

    return corr


def smallcorr(covm, ipois, threshold=0.01):
    corr = corrcoef(covm)

    np.set_printoptions(precision=3, suppress=True)
    
    # Select rows for the POIs only
    corr = corr[ipois]
    # Array of True where the correlation between a POI (row) and another
    # parameter (col) is below threshold
    below = np.fabs(corr) < threshold
    # Find columns that are always below threshold: parameters that don't
    # correlate strongly to any POI
    below = np.all(below, axis=0)

    for ipar in range(len(below)):
        print("%+.3f  %+.3f  %s" % (corr[0][ipar], corr[1][ipar], below[ipar]))

    return np.arange(len(below))[below]
