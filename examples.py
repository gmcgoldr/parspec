from __future__ import division
import warnings

import numpy as np
from matplotlib import pyplot as plt
import ROOT

import npinterval
from pymcmc import MCMC

import templates

# Try to set a nicer style for matplotlib, only for version > 1.5
_vmpl = list(map(int, plt.matplotlib.__version__.split('.')))
if _vmpl[0] > 1 or (_vmpl[0] == 1 and _vmpl[1] >= 5):
    plt.style.use('ggplot')


def draw_point_hist(y, x=None, **kwargs):
    """
    Draw a series of values as a histogram line. Each y-value is the count of
    a histogram bin. If x-values are also given, they are the bin centers. If
    no x-values are given, the y-values are bined at integers starting with 0.

    :param y: [float]
        series of bin counts for the histogram
    :param x: [float]
        bin centers for the histogram
    """
    if len(y) < 2:
        raise RuntimeError("Need at least 2 points for histogram")

    if x is None:
        # Use integers starting at 0
        x = np.arange(len(y))
    else:
        # Use given x, and ensure points are sorted
        x = np.array(x)
        isort = np.argsort(x)
        x = x[isort]
        y = y[isort]

    # Compute the bin edges from their centers
    in_edges = (x[1:]+x[:-1])/2.
    first = x[0] - (x[1]-x[0])/2.
    last = x[-1] + (x[-1]-x[-2])/2.
    # Convert to list to insert first and last
    edges = list(in_edges)
    edges.insert(0, first)
    edges.append(last)

    # Histogram y-values (two bin counts for each bin edge, plus one left-most
    # and one right-most point that go to zero)
    hy = np.zeros(2*len(y)+2)
    hx = np.zeros(2*len(y)+2)

    # Populate left edges with bin counts (don't use outside 0 edges)
    hy[1+0:-1:2] = y
    # Populate right edges with same counts (gets flat line accross)
    hy[1+1:-1:2] = y
    # Set the bin edges
    hx[1+0:-1:2] = edges[:-1]
    hx[1+1:-1:2] = edges[1:]
    # Set the outermost points to 0
    hy[0] = 0
    hy[-1] = 0
    hx[0] = edges[0]
    hx[-1] = edges[-1]

    lines = plt.plot(hx, hy, **kwargs)
    plt.xlim(hx[0], hx[-1])

    return lines[0]


def build_template_meas():
    """
    Build a template measurment object.

    :return: templates.TemplateMeasurement
    """

    meas = templates.TemplateMeasurement()
    meas.set_lumi(1, 0.02)

    # Base shape for the signal, triangular distribution
    base = np.array([10000, 12500, 15000, 12500, 10000], dtype=float)
    # Add a source for the signal
    src_sig = meas.new_source('sig', base)
    src_sig.use_lumi()  # impacted by luminosity
    src_sig.use_stats(1./(10*base**0.5))  # stat unc. from 10x MC
    src_sig.set_xsec(1, 0.95, 1.05)  # cross section constrained to +/-5%
    # Add a template: under the influence of parameter p, a linear slope is
    # added to the signal
    src_sig.add_template('p', base*[0.8, 0.9, 1, 1.1, 1.2])
    # Add highly asymmetric systematic uncertainty which looks a lot like the
    # signal. This is a challenging model to fit.
    src_sig.add_syst('s1', base*[0.94, 0.98, 1, 1.02, 1.06], 'up')
    src_sig.add_syst('s1', base*[0.97, 0.99, 1, 1.01, 1.03], 'down')
    # Add another systematic which doesn't look like the signal or the data
    # (should be constrained)
    src_sig.add_syst('s2', base*[1.02, 1.01, 1, 1.01, 1.02])

    # Add a flat-ish background (different shape from signal)
    base = np.array([1200, 1100, 1000, 1000, 1000], dtype=float)
    src_bg1 = meas.new_source('bg1', base)
    src_bg1.use_lumi()
    src_bg1.use_stats(1./(10*base)**0.5)
    src_bg1.set_xsec(1, 0.8, 1.1)
    # It is also impacted by systematic 2
    src_bg1.add_syst('s2', base*[1.02, 1.01, 1, 1.01, 1.02])

    # Add a background not impacted by lumi or stats (e.g. data driven)
    src_bg2 = meas.new_source('bg2', [1000, 1000, 1000, 1100, 1200])
    src_bg2.set_xsec(1, 0.9, 1.1)

    # Build the spectrum object
    meas.build()

    return meas


def make_pseudo(meas, systs=True, signal=True, stats=True):
    """
    Generate a plausible data spectrum for a pseudo-experiment in which the
    true underlying parameters are not known.

    :param meas: TemplateMeasurment
        measurement object whose spectrum is used to generate data
    :param systs: bool
        randomize systematic parameter values
    :param signal: bool
        randomize the signal value
    :param stats: bool
        poisson fluctuate data yields
    :return: [float], [float]
        pseudo-data and the true parameter values
    """

    # Get the scales for the paramters controlling the spectrum
    scales = meas.spec.scales
    # Randomize the true underlying values for constrained parameters
    true = list(meas.spec.central)

    if systs:
        # Vary the constrained parameters based on their priors
        for par in meas.spec.pars:
            if par in meas.spec.unconstrained:
                continue
            ipar = meas.spec.ipar(par)
            true[ipar] += np.random.normal(0, scales[ipar])

    if signal:
        # Also choose a random signal strength (unconstrained parameter)
        true[meas.spec.ipar('p')] = np.random.uniform(-1, 1)

    # Build the data spectrum that would be observed for those values
    data = meas.spec(true)

    if stats:
        # Poisson fluctuate yields (note that the statistical parameters
        # acount for fluctuation in simulated yields)
        data = np.random.poisson(data)

    return data, true


def single_fit(meas, randomize=False, nmax=100):
    """
    Perform a single fit using TMinuit.

    :param meas: TemplateMeasurement
        measurment object to fit
    :param randomize: bool
        randomize initial starting parameter values
    :param nmax: int
        maximum TMinuit fails before aborting
    :return: [float], float, ROOT.TMinimizer
        fit parameters, ll and minimizer object used to fit
    """
    # Do a vanilla fit (don't randomize parameters)
    minimizer = meas.spec.build_minimizer()

    if randomize:
        # Randomize initial values
        for ipar in range(meas.spec.npars):
            if meas.spec.scales[ipar] == 0:
                continue
            minimizer.SetVariableValue(
                ipar, 
                meas.spec.central[ipar] + 
                np.random.normal(0, meas.spec.scales[ipar]))

    # Attempt the fit, TMinuit will fail sometimes
    nfails = 0  # keep track of failed fits
    while not minimizer.Minimize():
        nfails += 1
        if nfails >= nmax:
            raise RuntimeError("Failed minimization")

    minx = [minimizer.X()[i] for i in range(meas.spec.npars)]
    ll = meas.spec.ll(minx)

    return minx, ll, minimizer


def global_fit(meas, ntrials=10, nmax=100):
    """
    Perform multiple fits and keep the best minimum.

    :param meas: TemplateMeasurement
        measurment object to fit
    :param ntrials: int
        number of successfull fit attempts from which to find a global minimum
    :param nmax: int
        maximum TMinuit fails before aborting
    :return: [float], float, ROOT.TMinimizer
        fit parameters, ll and minimizer object used to fit
    """

    best_x = None  # parameter values at global min
    best_ll = float('-inf')  # log likelihood at global min
    best_min = None  # keep minimizer object which reaches best min

    nfits = 0
    nfails = 0

    while nfits < 10:
        try:
            minx, ll, minimizer = single_fit(meas, randomize=True, nmax=1)
            nfits += 1  # once it succeeds, count the fit
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


def run_minos(meas, minimizer):
    """
    Find the points along each parameter value where the log likelihood is
    halved. For a normal distribution, this is the 1-sigma interval containing
    68.27% of the distribution.

    :param meas: TemplateMeasurement
        measurement whose spectrum parameters are to be profiled
    :param minimizer: ROOT.TMinimizer
        minimier object which has found a minimum
    :return: [float], [float]
        distance to subtract and add to halve the log likelihood
    """
    # Declare ROOT doubles which minos can write to by reference
    down = ROOT.Double(0)
    up = ROOT.Double(0)

    # Lower and upper bounds for the parameters
    fit_down = [0] * meas.spec.npars
    fit_up = [0] * meas.spec.npars

    for par in meas.spec.pars:
        ipar = meas.spec.ipar(par)
        if minimizer.GetMinosError(ipar, down, up):
            # Note: important to cast the copy the ROOT variable, otherwise
            # the list will contain a reference to the value, which will change
            fit_down[ipar] = float(down)
            fit_up[ipar] = float(up)
        else:
            warnings.warn("Minos failed on %s" % par, RuntimeWarning)

    return fit_down, fit_up


def run_mcmc(meas, x, nsamples, covm=None, scales=None):
    """
    Sample the likelihood space with a Markov Chain Monte Carlo.

    :param meas: TemplateMeasurement
        measurement whose spectrum likelihood space is to be probe
    :param x: [float]
        parameter values where to start the chain
    :param covm: [[float]]
        covariance matrix values if sampling transformed space
    :param scales: [float]
        parameter scales if not sampling transformed space
    :return: [float], [float], [float], pymcmc.MCMC
        posterior mean, lower CI, upper CI for each parameter, and the MCMC
        object used for sampling
    """
    mcmc = MCMC(meas.spec.npars)
    mcmc.set_values(x)

    if covm is not None and scales is None:
        scales, transform = np.linalg.eigh(covm)
        mcmc.set_transform(transform)
        mcmc.set_scales(scales**0.5)
    elif scales is not None:
        mcmc.set_scales(scales)
    else:
        raise ValueError("Must provide covariance OR scales")

    mcmc.rescale = 2  # good starting point
    mcmc.learn_scale(meas.spec.ll, 1000)

    mcmc.run(meas.spec.ll, nsamples)

    mean = list()
    mean_down = list()
    mean_up = list()

    for ipar in range(meas.spec.npars):
        mean.append(np.mean(mcmc.data[:, ipar]))
        low, high, _, _ = npinterval.interval(mcmc.data[:, ipar], 0.6827)
        mean_down.append(low-mean[-1])
        mean_up.append(high-mean[-1])

    return mean, mean_down, mean_up, mcmc


def measure_template(meas, draw=False):
    """
    Generate a fake data spectrum using a template measurement spectrum, and
    randomizing its parameters. Then fit this fake data to see if its true 
    underlying parameters can be recovered.

    :param meas: TemplateMeasurement
        measurment object to use
    :param draw: bool
        draw the fit spectrum
    :return: dict
        map each parameter to various measurement values
    """

    data, true = make_pseudo(meas)
    meas.spec.set_data(data)

    # First fit without randomization
    fit_first, _, _ = single_fit(meas, randomize=False)
    # Global fit with randomization to find better minimum
    minx, ll, minimizer = global_fit(meas, ntrials=10)

    # Compute the covariance matrix, never seen it fail, but warn in case
    if not minimizer.Hesse():
        warnings.warn("Failed to compute error marix", RuntimeWarning)

    fit_err = [minimizer.Errors()[i] for i in range(meas.spec.npars)]

    covm = np.array([
        [minimizer.CovMatrix(i,j) 
        for j in range(meas.spec.npars)] 
        for i in range(meas.spec.npars)])
    
    # Measure the confidence intervals with minos profiling
    fit_down, fit_up = run_minos(meas, minimizer)

    # Get a better estimate for the confidence intervals with MCMC sampling
    mean, mean_down, mean_up, mcmc = \
        run_mcmc(meas, minx, nsamples=1e5, covm=covm)

    results = dict()
    for par in meas.spec.pars:
        ipar = meas.spec.ipar(par)
        results[par] = dict()
        results[par]['true'] = true[ipar]
        results[par]['fit'] = minx[ipar]
        results[par]['fit_first'] = fit_first[ipar]
        results[par]['fit_err'] = fit_err[ipar]
        results[par]['fit_down'] = fit_down[ipar]
        results[par]['fit_up'] = fit_up[ipar]
        results[par]['mean'] = mean[ipar]
        results[par]['mean_down'] = mean_down[ipar]
        results[par]['mean_up'] = mean_up[ipar]
    
    if draw:
        nominal = meas.spec(meas.spec.central)
        l0 = draw_point_hist(
            np.zeros(meas.spec.npars), 
            label='nominal')
        ltrue = draw_point_hist(
            100*(data/nominal-1), 
            label='true')
        lfit = draw_point_hist(
            100*(meas.spec(minx)/nominal-1), 
            ls='--', label='fit')
        plt.legend(handles=[l0, ltrue, lfit])
        ylims = plt.ylim()
        yrange = ylims[1] - ylims[0]
        plt.ylim(ylims[0], ylims[0]+1.2*yrange)
        plt.savefig('spectrum_fit.png', format='png')
        plt.clf()

    return results


def draw_spectra(meas, normalize=True):
    """
    Draw a spectrum with a parameter fluctuated to +/- 1 sigma, for each
    parameter. Unconstrained parameters are set to +/- 1.

    :param meas: TemplateMeasurement
        measurement from which to draw the spctrum
    :param normalize: bool
        normalize fluctuated spectrum to the nominal one
    """

    for par in meas.spec.pars:
        ipar = meas.spec.ipar(par)
        info = meas.spec.parinfo(par)

        x = list(meas.spec.central)  # point where to draw spectrum

        # Draw a histogram of the spectrum with the central parameter values
        nominal = meas.spec(x)
        l0 = draw_point_hist(
            nominal if not normalize else np.zeros(len(nominal)),
            label='nominal')

        # Get the central values for the parameters
        low = info['low']
        high = info['high']
        
        # If the parameter is unconstrained, set its draw range to unity
        if low == high:
            low = info['central'] - 1
            high = info['central'] + 1

        # Shift the value for the current parameter to +1 and draw
        x[ipar] = high
        lhigh = draw_point_hist(
            meas.spec(x) if not normalize else 100*(meas.spec(x)/nominal-1),
            label=r'%s = +1$\sigma$' % par)

        # Shift to -1 and draw
        x[ipar] = low
        llow = draw_point_hist(
            meas.spec(x) if not normalize else 100*(meas.spec(x)/nominal-1),
            label=r'%s = -1$\sigma$' % par)

        plt.legend(handles=[l0, lhigh, llow])

        ylims = plt.ylim()
        yrange = ylims[1] - ylims[0]
        plt.ylim(ylims[0], ylims[0]+1.2*yrange)

        plt.savefig('spectrum_%s.png' % par, format='png')
        plt.clf()


if __name__ == '__main__':
    print("Building measurement...")
    meas = build_template_meas()

    print("Fitting templates...")
    results = measure_template(meas, draw=True)
    print(('%15s'+'  %6s'*9) % (
        "Parameter", 
        "True", 
        "Fit", 
        "First", 
        "Err", 
        "Down", 
        "Up", 
        "Mean",
        "Down", 
        "Up"))
    for par in meas.spec.pars:
        print(('%15s'+'  %+6.3f'*9) % (
            par, 
            results[par]['true'],
            results[par]['fit'],
            results[par]['fit_first'],
            results[par]['fit_err'],
            results[par]['fit_down'],
            results[par]['fit_up'],
            results[par]['mean'],
            results[par]['mean_down'],
            results[par]['mean_up']))

    print("Drawing spectrum...")
    draw_spectra(build_template_meas())

    print("Running pseudo-experiments...")
    print("Warning: this will take around 1 hour")
    with open('trials.csv', 'w') as fout:
        fout.write(', '.join([
            "True", 
            "Fit", 
            "First", 
            "Err", 
            "Down", 
            "Up", 
            "Mean",
            "Down", 
            "Up"]))
        fout.write('\n')
        fout.flush()
        for itrial in range(1000):
            try:
                results = measure_template(meas, draw=False)
            except RuntimeError:
                print("Measurement failed")
                continue
            fout.write(', '.join(map(str, [
                results['p']['true'],
                results['p']['fit'],
                results['p']['fit_first'],
                results['p']['fit_err'],
                results['p']['fit_down'],
                results['p']['fit_up'],
                results['p']['mean'],
                results['p']['mean_down'],
                results['p']['mean_up']])))
            fout.write('\n')
            fout.flush()
