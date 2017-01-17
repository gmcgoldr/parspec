from __future__ import division
import warnings
import sys
import time

import numpy as np
from matplotlib import pyplot as plt
import ROOT

import npinterval
from pymcmc import MCMC

import templates
import minutils


def clean_pyplot():
    """Make pyplot nicer"""
    version = float('.'.join(plt.matplotlib.__version__.split('.')[:2]))
    if version >= 1.5:
        plt.style.use('ggplot')
    try:
        # Try to get the default colors in the new parameters
        plt.ccolors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    except KeyError:
        # Fall back to the old parameters
        plt.ccolors = list(plt.rcParams['axes.color_cycle'])
    plt.rc('lines', linewidth=2)
    plt.rc('patch', linewidth=2)


def rescale_plot(factor=0.25):
    low, high = plt.ylim()
    plt.ylim(low-factor*(high-low), high+factor*(high-low))


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

    lines = plt.plot(hx[1:-1], hy[1:-1], **kwargs)
    plt.xlim(hx[0], hx[-1])

    return lines[0]


def draw_spectrum(meas, x, rel=False, scale=True, **kwargs):
    """
    Convenience function to draw a spectrum.

    :param meas: templates.TemplateMeasurement
        measurement whose spectrum is drawn
    :param x: [float]
        parameter values for the spectrum
    :param rel: bool
        if True, normalize to the spectrum with default parameters
    :param name: str
        if given, store the figure at this path
    """

    if rel:
        s0 = meas.spec(meas.spec.central)
        s = meas.spec(x)
        line = draw_point_hist(100*(s/s0-1), **kwargs)
        plt.ylabel('Relative offset percentage')
        if scale:
            rescale_plot()

    else:
        line = draw_point_hist(meas.spec(x), **kwargs)
        plt.ylabel('Spctral value / bin')
        if scale:
            rescale_plot()
        plt.ylim(ymin=0)

    plt.xlabel('Bin')

    return line


def build_template_meas(name='example'):
    """
    Build a template measurment object.

    :return: templates.TemplateMeasurement
    """

    meas = templates.TemplateMeasurement(name)
    meas.set_lumi(1, 0.02)

    # Base shape for the signal, triangular distribution
    sig = np.array([10000, 12500, 15000, 12500, 10000], dtype=float)
    # Add a source for the signal
    src_sig = meas.new_source('sig', sig)
    src_sig.use_lumi()  # impacted by luminosity
    src_sig.use_stats(.1*(10*sig)**0.5)  # stat unc. from 10x MC
    src_sig.set_xsec(1, 0.95, 1.05)  # cross section constrained to +/-5%
    # Add a template: under the influence of parameter p, a linear slope is
    # added to the signal
    src_sig.add_template('p', sig*[-.2, -.1, 0, +.1, +.2])
    # Add highly asymmetric systematic uncertainty which looks a lot like the
    # signal. This is a challenging model to fit.
    src_sig.add_syst('s1', sig*[-.06, -.02, 0, +.02, +.06], polarity='up')
    src_sig.add_syst('s1', sig*[-.03, -.01, 0, +.01, +.03], polarity='down')
    # Add another systematic which doesn't look like the signal or the data
    # (should be constrained)
    src_sig.add_syst('s2', sig*[+.02, +.01, 0, +.01, +.02])

    # Add a flat-ish background (different shape from signal)
    bg1 = np.array([1600, 1300, 1000, 1000, 1000], dtype=float)
    src_bg1 = meas.new_source('bg1', bg1)
    src_bg1.use_lumi()
    src_bg1.use_stats(.1*(10*bg1)**0.5)
    src_bg1.set_xsec(1, 0.8, 1.1)
    # It is also impacted by systematic 2
    src_bg1.add_syst('s2', bg1*[+.02, +.01, 0, +.01, +.02])

    # Add a background not impacted by lumi or stats (e.g. data driven)
    bg2 = np.array([1000, 1000, 1000, 1300, 1600], dtype=float)
    src_bg2 = meas.new_source('bg2', bg2)
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
    truth = np.array(meas.spec.central)

    if systs:
        # Vary the constrained parameters based on their priors
        truth = meas.spec.randomize_parameters(
            meas.spec.central, 
            meas.spec.central, 
            meas.spec.lows, 
            meas.spec.highs,
            meas.spec.constraints)

    if signal:
        # Also choose a random signal strength (unconstrained parameter)
        truth[meas.spec.ipar('p')] = np.random.uniform(-1, 1)

    # Build the data spectrum that would be observed for those values
    data = meas.spec(truth)

    if stats:
        # Poisson fluctuate yields (note that the statistical parameters
        # acount for fluctuation in simulated yields)
        data = np.random.poisson(data)

    return data, truth


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
        mcmc.set_covm(covm)
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


def asses_space(meas):
    """
    Assess the structure of the likelihood space with all parameters shifted
    to +1 sigma.
    """
    truth = list(meas.spec.central)
    truth[meas.spec.ipar('syst_s1')] = 1
    data = meas.spec(truth)
    meas.spec.set_data(data)

    lls, xs, rels, prob = minutils.find_minima(meas.spec)

    print("Found %d minima with likelihoods:" % len(lls))
    print(', '.join(["%.3f" % l for l in lls]))

    print("Global minimum is found %.3f%% of the time" % (100*prob))

    l0 = draw_spectrum(meas, truth, True, label='truth', linestyle='--')
    l1 = draw_spectrum(meas, xs[0], True, label='fit')
    plt.legend(handles=[l0, l1])
    plt.savefig('spectrum-ll_0.pdf', format='pdf')
    plt.clf()

    for imin in range(1, len(lls)):
        print("Local minimum %.3f" % lls[imin])

        isort = np.argsort(np.fabs(rels[imin]))[::-1]
        par1 = meas.spec.pars[isort[0]]
        par2 = meas.spec.pars[isort[1]]
        print("Differs in %s, %s" % (par1, par2))

        print("%s_0: %.3f, %s_%d: %.3f" % (
            par1,
            xs[0][meas.spec.ipar(par1)],
            par1, imin,
            xs[imin][meas.spec.ipar(par1)]))

        print("%s_0: %.3f, %s_%d: %.3f" % (
            par2,
            xs[0][meas.spec.ipar(par2)],
            par2, imin,
            xs[imin][meas.spec.ipar(par2)]))

        draw_spectrum(meas, truth, True, label='truth', linestyle='--')
        draw_spectrum(meas, xs[imin], True, label='fit')
        plt.legend(handles=[l0, l1])
        plt.savefig('spectrum-ll_%d.pdf'%imin, format='pdf')
        plt.clf()

        vals = minutils.slice2d(meas.spec, xs[imin], par1, par2)
        plt.hist2d(
            vals.T[0], 
            vals.T[1], 
            weights=np.exp(vals.T[2]), 
            bins=len(vals)**0.5,
            normed=True)
    
        plt.xlabel(par1)
        plt.ylabel(par2)

        cbar = plt.colorbar()
        cbar.set_label('Likelihood density')

        plt.savefig('min_%d.pdf' % imin, format='pdf')
        plt.clf()


def measure_template(meas):
    """
    Generate a fake data spectrum using a template measurement spectrum, and
    randomizing its parameters. Then fit this fake data to see if its true 
    underlying parameters can be recovered.

    :return: dict
        map each parameter to various measurement values
    """
    # Make a pseudo-experiment
    data, truth = make_pseudo(meas)
    meas.spec.set_data(data)

    # First fit without randomization
    fit_first, _, _ = minutils.single_fit(meas.spec, randomize=False)
    # Global fit with randomization to find better minimum
    minx, ll, minimizer = minutils.global_fit(meas.spec, nfits=10)

    # Compute the covariance matrix, never seen it fail, but warn in case
    if not minimizer.Hesse():
        warnings.warn("Failed to compute error marix", RuntimeWarning)

    fit_err = [minimizer.Errors()[i] for i in range(meas.spec.npars)]

    covm = np.array([
        [minimizer.CovMatrix(i,j) 
        for j in range(meas.spec.npars)] 
        for i in range(meas.spec.npars)])
    
    # Measure the confidence intervals with minos profiling
    fit_down, fit_up, _ = minutils.run_minos(meas.spec, minimizer)

    # Get a better estimate for the confidence intervals with MCMC sampling
    mean, mean_down, mean_up, mcmc = \
        run_mcmc(meas, minx, nsamples=1e5, covm=covm)

    results = dict()
    for par in meas.spec.pars:
        ipar = meas.spec.ipar(par)
        results[par] = dict()
        results[par]['true'] = truth[ipar]
        results[par]['fit'] = minx[ipar]
        results[par]['fit_first'] = fit_first[ipar]
        results[par]['fit_err'] = fit_err[ipar]
        results[par]['fit_down'] = fit_down[ipar]
        results[par]['fit_up'] = fit_up[ipar]
        results[par]['mean'] = mean[ipar]
        results[par]['mean_down'] = mean_down[ipar]
        results[par]['mean_up'] = mean_up[ipar]

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

    # Draw the plain spectrum
    x = list(meas.spec.central)
    lfull = draw_spectrum(meas, x, scale=True, label='full')

    x[meas.spec.ipar('xsec_sig')] = 1
    x[meas.spec.ipar('xsec_bg1')] = 0
    x[meas.spec.ipar('xsec_bg2')] = 0
    lsig = draw_spectrum(meas, x, scale=False, label='sig')

    x[meas.spec.ipar('xsec_sig')] = 0
    x[meas.spec.ipar('xsec_bg1')] = 1
    x[meas.spec.ipar('xsec_bg2')] = 0
    lbg1 = draw_spectrum(meas, x, scale=False, label='bg1')

    x[meas.spec.ipar('xsec_sig')] = 0
    x[meas.spec.ipar('xsec_bg1')] = 0
    x[meas.spec.ipar('xsec_bg2')] = 1
    lbg2 = draw_spectrum(meas, x, scale=False, label='bg2')

    plt.legend(handles=[lfull, lsig, lbg1, lbg2])
    plt.savefig('spectrum.pdf', format='pdf')
    plt.clf()

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
            label=r'%s = %+.3e' % (par, high))

        # Shift to -1 and draw
        x[ipar] = low
        llow = draw_point_hist(
            meas.spec(x) if not normalize else 100*(meas.spec(x)/nominal-1),
            label=r'%s = %+.3e' % (par, low))

        plt.legend(handles=[l0, lhigh, llow])

        rescale_plot()
        plt.xlabel('Bin')
        plt.ylabel('Relative offset percentage')
        plt.savefig('spectrum-%s.pdf' % par, format='pdf')
        plt.clf()


if __name__ == '__main__':
    clean_pyplot()  # make pyplot nice
    np.random.seed(1234)  # get the same results each time

    print("Building measurement...")
    meas = build_template_meas()
    meas.spec.fixstats()

    print("Assessing minima...")
    asses_space(meas)

    print("Drawing spectrum...")
    draw_spectra(meas)

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
                results = measure_template(meas)
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
