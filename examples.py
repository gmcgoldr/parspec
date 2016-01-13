from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
# Try to set a nicer style for matplotlib, only for version > 1.5
_vmpl = list(map(int, plt.matplotlib.__version__.split('.')))
if _vmpl[0] > 1 or (_vmpl[0] == 1 and _vmpl[1] >= 5):
    plt.style.use('ggplot')

import npinterval
from pymcmc import MCMC

import templates


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

    # Add a signal source. The vector is its distribution, could be a TH1 (any
    # iterable object)
    src_sig = meas.new_source('sig', [1000, 1500, 2000, 1500, 1000])
    src_sig.use_lumi()  # impacted by luminosity
    src_sig.use_stats()  # underyling values subject to uncertainty
    src_sig.set_xsec(1, 0.99, 1.05)  # cross section constrained to +5%/-1%
    # Add fully asymmetric systematic uncertainty (same impact at +/-1 sigma)
    src_sig.add_syst('s1', [0, 100, 100, 100, 0], 'up')
    src_sig.add_syst('s1', [0, 100, 100, 100, 0], 'down')
    # Add a symmetric systematic uncertainty (modifies spectrum by a flat
    # positive slope when s2 is positive)
    src_sig.add_syst('s2', [-200, -100, 0, 100, 200])
    # Add a template (the parameter of interest). Increases contents of two
    # bins, so it doesn't look much like backgrounds and systs. The impact is
    # enhanced by a factor of 5.
    src_sig.add_template('0.5*p', [0, 0, 100, 500, 0], ['p'], [0.5])

    # Add a flat-ish background (different shape from signal)
    src_bg1 = meas.new_source('bg1', [600, 500, 600, 500, 600])
    src_bg1.use_lumi()
    src_bg1.use_stats()
    src_bg1.set_xsec(1, 0.8, 1.1)
    # Share one of the systematics with the background. Adds to bin 3 when
    # positive, adds to bin 1 when negative.
    src_bg1.add_syst('s2', [0, 0, 0, 200, 0], 'up')
    src_bg1.add_syst('s2', [0, 100, 0, 0, 0], 'down')

    # Add a background not impacted by lumi or stats (e.g. data driven)
    src_bg2 = meas.new_source('bg2', [200, 200, 200, 200, 200])
    src_bg2.set_xsec(1, 0.9, 1.1)

    # Build the spectrum object
    meas.build()

    return meas


def measure_template(meas, draw=False):
    """
    Generate a fake data spectrum using a template measurement spectrum, and
    randomizing its parameters. Then fit this fake data to see if its true 
    underlying parameters can be recovered.

    :return: dict
        mapping of parameter names to their results, or None if failed
    """

    # Map each parameter to a mapping of the following values
    results = {
        par: {
            'true': 0,  # true value used in simulating data
            'fit': 0,   # MLE from minimization
            'err': 0,   # uncertainty of minimization
            'mode': 0,   # Mode of the Bayesian posterior
            'low': 0,   # lower bound of 68.27% C.L.
            'high': 0}  # upper bound of 68.27% C.L.
        for par in meas.spec.pars()}

    # Get the scales for the paramters controlling the spectrum
    scales = meas.spec.scalesx()
    # Unconstrained parameters have sacles of 0
    constrained = (scales > 0)  # list of True if constrained

    # Randomize the true underlying values for constrained parameters
    true = meas.spec.centralx()
    true[constrained] += np.random.normal(0, scales[constrained])
    # Also choose a random signal strength (unconstrained parameter)
    true[meas.spec.ipar('p')] = np.random.uniform(-2, 2)

    # Store true values for parameters
    for par in meas.spec.pars():
        ipar = meas.spec.ipar(par)
        results[par]['true'] = true[ipar]

    # Show that the only unconstrained parameter is p
    assert(constrained[meas.spec.ipar('p')] == False)
    assert(np.sum(~constrained) == 1)

    # Build the data spectrum that would be observed for those values (note
    # that this includes statistical fluctuations)
    data = meas.spec(true)

    # Builder a ROOT minimizer
    minimizer = meas.spec.build_minimizer()

    # Tell the spectrum to compare to the data when computing the likelihood
    meas.spec.set_data(data)
    # Try to minimize, fail at 100 attempts
    niters = 0
    while not minimizer.Minimize():
        niters += 1
        if niters >= 100:
            return None
    # Try to measure covaraince, fail at 100 attempts
    niters = 0
    while not minimizer.Hesse():
        niters += 1
        if niters >= 100:
            return None

    # Get the minimization information
    best = [minimizer.X()[i] for i in range(meas.spec.npars())]
    errs = [minimizer.Errors()[i] for i in range(meas.spec.npars())]
    covm = np.array([
        [minimizer.CovMatrix(i,j) 
        for j in range(meas.spec.npars())] 
        for i in range(meas.spec.npars())])

    # Store minimization values for parameters
    for par in meas.spec.pars():
        ipar = meas.spec.ipar(par)
        results[par]['fit'] = best[ipar]
        results[par]['err'] = errs[ipar]

    # Get a better estimate for the confidence intervals with MCMC sampling
    mcmc = MCMC(meas.spec.npars())
    # Find a transformation of the parameter space which diagonalizes the
    # covariance matrix. In this space, each parameter is independent.
    scales, transform = np.linalg.eigh(covm)
    # Use this to help the MCMC sample the space more efficiently
    mcmc.set_transform(transform)
    # Set the scale of the parameters in the transformed space
    mcmc.set_scales(scales**0.5)
    mcmc.rescale = 2  # typically works well
    # Start the chain at the most likely parameter values  
    mcmc.set_values(best)
    # Try to learn the optimal rescale
    if not mcmc.learn_scale(meas.spec.ll):
        return None  # failed to converge in 100 steps
    if mcmc.rescale < 1.5 or mcmc.rescale > 3:
        return None  # converged to a bad scale, fit was probably bad

    # Take 100,000 samples of the space, this seems to give a few percent
    # variance on the confidence interval
    mcmc.run(meas.spec.ll, int(1e5))
    
    # Store confidence intervals for parameters
    for par in meas.spec.pars():
        ipar = meas.spec.ipar(par)
        # Estimate the mode of the sampled distribution
        mode = npinterval.half_sample_mode(mcmc.data[:, ipar])
        results[par]['mode'] = mode
        # Find the shortest interval between two samples containing 68%
        low, high, _, _ = npinterval.interval(mcmc.data[:, ipar], 0.6827)
        results[par]['low'] = low
        results[par]['high'] = high

    if draw:
        l0 = draw_point_hist(meas.spec(meas.spec.centralx()), label='nominal')
        ltrue = draw_point_hist(data, label='true')
        lfit = draw_point_hist(meas.spec(best), ls='--', label='fit')
        plt.legend(handles=[l0, ltrue, lfit])
        ylims = plt.ylim()
        yrange = ylims[1] - ylims[0]
        plt.ylim(ylims[0], ylims[0]+1.2*yrange)
        plt.savefig('spectrum_fit.png', format='png')
        plt.clf()

    return results


def draw_spectra(meas):
    """
    Draw the spectrum from a measurement object with parameter values set at 
    +/-1 sigma values.
    """

    for par in meas.spec.pars():
        ipar = meas.spec.ipar(par)

        scale_up = meas.spec.get_scale(par, 'up')
        scale_down = meas.spec.get_scale(par, 'down')

        # Get the central values for the parameters
        point = meas.spec.centralx()

        # Draw a histogram of the spectrum with the central parameter values
        l0 = draw_point_hist(meas.spec(point), label='nominal')
        
        # Set the scale of unconstrained parameters to +/- 1
        if scale_up == scale_down == 0:
            scale_up = 1
            scale_down = -1

        # Shift the value for the current parameter to +1 and draw
        point[ipar] = scale_up
        lup = draw_point_hist(
            meas.spec(point),
            label=r'%s = +1$\sigma$' % par)

        # Again at -1 (scaled)
        point[ipar] = scale_down
        ldown = draw_point_hist(
            meas.spec(point),
            label=r'%s = -1$\sigma$' % par)

        plt.legend(handles=[l0, lup, ldown])

        ylims = plt.ylim()
        yrange = ylims[1] - ylims[0]
        plt.ylim(ylims[0], ylims[0]+1.2*yrange)

        plt.savefig('spectrum_%s.png' % par, format='png')
        plt.clf()


if __name__ == '__main__':
    meas = build_template_meas()

    print("Fitting templates...")

    results = None
    while not results:
        print("Attempting a fit...")
        results = measure_template(meas, draw=True)

    print('%15s  %6s  %6s  %6s  %6s  %6s  %6s' % (
        "Parameter", 
        "True", 
        "Fit", 
        "Err", 
        "Mode",
        "Low", 
        "High"))

    for par in meas.spec.pars():
        print('%15s  %+6.3f  %+6.3f  %+6.3f  %+6.3f  %+6.3f  %+6.3f' % (
            par, 
            results[par]['true'],
            results[par]['fit'],
            results[par]['err'],
            results[par]['mode'],
            results[par]['low'],
            results[par]['high']))

    draw_spectra(build_template_meas())
