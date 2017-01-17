import unittest
import math

import numpy as np
import numpy.testing
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from parspec import SpecBuilder
from parspec import Source
from parspec import ParSpec


def logPoisson(k, v, s):
    vv = np.array(k, dtype=float)
    vv[vv<1] = 1
    vv += s
    return -0.5 * (k-v)**2 / vv


class TestParSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup a single spectrum object for all tests"""

        # Builder accumulates data and builds the spectrum
        builder = SpecBuilder('Spectrum')

        ### Add a signal ###
        
        # Add a trinagular signal
        sig = [1000., 1100., 1200., 1100., 1000.]
        src_sig = Source(sig)
        # Indicate the bin contents in sig are subject to statistical
        # uncertainty, based on double the count (as if 2x MC was generated
        # then scaled down by 0.5)
        src_sig.use_stats(.5*(2*np.array(sig))**0.5)
        # Allow its scale to vary
        src_sig.set_expression(
            'lumi*xsec_sig',  # scale factor
            ['lumi', 'xsec_sig'],  # parameters are lumi and xsec
            ['xsec_sig', 'lumi'])  # dn/dlumi and dn/dxsec
        # Add to builder once configured
        builder.add_source(src_sig)
        # Constrain xsec with an asymmeric prior
        builder.set_prior('xsec_sig', 1, 0.9, 1.2, 'normal')
        # Constrain lumi with 5% uncertainty
        builder.set_prior('lumi', 1, 0.95, 1.05, 'lognormal')

        ### Add two systematic uncertinaties ###

        # Add systematic shape variation (a top hat)
        sig_syst1 = [0, 50, 50, 50, 0]
        # This is a shape which inherits the normalization from the signal
        src_sig_syst1_up = Source(sig_syst1, shapeof=src_sig)
        # Assume 1:1 statistical uncertainty on this shape
        src_sig_syst1_up.use_stats(np.array(sig_syst1)**0.5)
        # Control the amount of this variation with the parameter syst1, and
        # indicate that the shape applies only if syst1 >= 0. Note that 
        # parameter list and gradients can be omitted for simple sums
        src_sig_syst1_up.set_expression('syst1', polarity='up')
        # Make syst1 fully asymmetric: it has the same effect on the spectrum
        # when the parameter is positive as negative
        src_sig_syst1_down = Source(sig_syst1, shapeof=src_sig)
        src_sig_syst1_down.set_expression('syst1', polarity='down')
        builder.add_source(src_sig_syst1_up)
        builder.add_source(src_sig_syst1_down)
        # 1 sigma penality when this parameter gets to values +/- 1
        builder.set_prior('syst1', 0, -1, 1, 'normal')

        # Add a linear systematic variant
        sig_syst2 = [-100, -50, 0 , 50, 100]
        src_sig_syst2 = Source(sig_syst2, shapeof=src_sig)
        # This one is symmetrized: the value of syst2 simply scales
        src_sig_syst2.set_expression('syst2')
        builder.add_source(src_sig_syst2)
        builder.set_prior('syst2', 0, -1, 1, 'normal')

        ### Add a template (the parameter of interest) ###

        # Add shape to th3 signal, but won't be constrained
        sig_temp1 = [0, 0, 10, 100, 0]
        src_poi = Source(sig_temp1, shapeof=src_sig)
        # The parameter of interest is called p, and scales the template by
        # a factof of 5
        src_poi.set_expression('5*p', ['p'], ['5'])
        builder.add_source(src_poi)

        ### Add a background ###

        bg = [110, 100, 100, 100, 105]
        src_bg = Source(bg)
        src_bg.set_expression(
            'lumi*xsec_bg',
            ['lumi', 'xsec_bg'],
            ['xsec_bg', 'lumi'])
        builder.add_source(src_bg)
        builder.set_prior('xsec_bg', 1, 0.9, 1.1, 'normal')

        ### Share one of the systematics with the background ###

        bg_syst2 = [10, 20, 10, 20, 10]
        src_bg_syst2 = Source(bg_syst2, shapeof=src_bg)
        src_bg_syst2.set_expression('syst2')
        builder.add_source(src_bg_syst2)
        # Note that this parameter is already constrained

        ### Add a custom regularization for the free parameter ###

        builder.add_regularization(
            'std::pow(p-syst1, 2)',
            ['p', 'syst1'],
            ['2*(p-syst1)', '-2*(p-syst1)'])

        # Store the builder so that tests can use it or its contents
        cls.builder = builder
        cls.spec = builder.build()

    def test_pars(self):
        """Check if the spectrum returns the correct list of parameters"""
        np.testing.assert_equal(
            self.spec.pars,
            ['lumi', 
            'p', 
            'syst1', 
            'syst2', 
            'xsec_bg', 
            'xsec_sig'])

    def test_unconstrained(self):
        """Check that the spectrum returns the correct unconstrained pars"""
        np.testing.assert_equal(self.spec.unconstrained, ['p'])

    def test_central(self):
        """Check if the spectrum returns the correct central value"""
        # Paramters are:
        # lumi (centered at 1 to leave yields unchanged)
        # p (centered at 0 to not contribute)
        # syst1 (centered at 0 to not contribute)
        # syst2 (centered at 0 to not contribute)
        # xsec_sig (centered at 1 to leave yeilds unchanged)
        # xsec_bg (centered at 1 to leave yeilds unchanged)
        np.testing.assert_array_almost_equal(
            [1, 0, 0, 0, 1, 1],
            self.spec.central)

    def test_scales(self):
        """Check if the spectrum returns the correct scales"""
        # Check for all parameters
        for par in self.spec.pars:
            if par.startswith('stat'):
                continue
            ipar = self.spec.ipar(par)
            if par in self.builder._priors:
                # Constrained parameters are scaled by constraint
                low = self.builder._priors[par]['low']
                high = self.builder._priors[par]['high']
                scale = (high-low)/2.
            else:
                # Unconstrained parameters are not scaled
                scale = 0
            self.assertAlmostEqual(self.spec.scales[ipar], scale)


    def test_ipar(self):
        """Check parameter indices"""
        for ipar, par in enumerate(self.spec.pars):
            self.assertEqual(ipar, self.spec.ipar(par))

    def test_par_info(self):
        """Check parameter information"""
        # Check for all parameters
        for ipar, par in enumerate(self.spec.pars):
            info = self.spec.parinfo(par)
            # Should work with indices as well
            self.assertEqual(info, self.spec.parinfo(ipar))
            self.assertEqual(info['index'], ipar)
            self.assertEqual(info['name'], par)
            if par in self.spec.unconstrained:
                self.assertAlmostEqual(info['central'], 0)
                self.assertAlmostEqual(info['low'], 0)
                self.assertAlmostEqual(info['high'], 0)
                self.assertEqual(info['constraint'], 'none')
            else:
                prior = self.builder._priors[par]
                self.assertAlmostEqual(info['central'], prior['central'])
                self.assertAlmostEqual(info['low'], prior['low'])
                self.assertAlmostEqual(info['high'], prior['high'])
                if par == 'lumi':
                    self.assertEqual(info['constraint'], 'lognormal')
                else:
                    self.assertEqual(info['constraint'], 'normal')

    def test_spec_nom(self):
        """Check nominal spectrum"""
        # Nominal spectrum is source + background
        true = (
            self.builder._sources[0]._data +
            self.builder._sources[5]._data 
        )
        # Should get the same spectrum using central parameters
        pars = list(self.spec.central)
        comp = self.spec(pars)
        np.testing.assert_array_almost_equal(true, comp)

    def test_stats_nom(self):
        """Check nominal spectrum stats"""
        # stats is sum in quadrature of those provided
        true = (
            self.builder._sources[0]._stats**2 +
            self.builder._sources[1]._stats**2
        )
        # Should get the same spectrum using central parameters
        np.testing.assert_array_almost_equal(true, self.spec.stats)

    def test_spec_xsec(self):
        """Check spectrum with varied x-section"""
        # Modify cross section
        true = (
            1.2 * self.builder._sources[0]._data + 
            0.5 * self.builder._sources[5]._data 
        )
        pars = list(self.spec.central)
        pars[self.spec.ipar('xsec_sig')] = 1.2
        pars[self.spec.ipar('xsec_bg')] = 0.5
        comp = self.spec(pars)
        np.testing.assert_array_almost_equal(true, comp)

    def test_spec_lumi(self):
        """Check spectrum with varied luminosity"""
        # Modify luminosity and cross sections
        true = (
            0.8*1.2 * self.builder._sources[0]._data + 
            0.8*0.5 * self.builder._sources[5]._data 
        )
        pars = list(self.spec.central)
        pars[self.spec.ipar('xsec_sig')] = 1.2
        pars[self.spec.ipar('xsec_bg')] = 0.5
        pars[self.spec.ipar('lumi')] = 0.8
        comp = self.spec(pars)
        np.testing.assert_array_almost_equal(true, comp)

    def test_spec_syst1_up(self):
        """Check spectrum with positive systematic"""
        # Positive value for syst1
        true = (
            0.8*1.2 * self.builder._sources[0]._data + 
            0.8*0.5 * self.builder._sources[5]._data + 
            0.8*1.2*0.2 * self.builder._sources[1]._data
        )
        pars = list(self.spec.central)
        pars[self.spec.ipar('xsec_sig')] = 1.2
        pars[self.spec.ipar('xsec_bg')] = 0.5
        pars[self.spec.ipar('lumi')] = 0.8
        pars[self.spec.ipar('syst1')] = 0.2
        comp = self.spec(pars)
        np.testing.assert_array_almost_equal(true, comp)

    def test_spec_syst1_down(self):
        """Check spectrum with negative systematic"""
        # Negative value for syst1
        true = (
            0.8*1.2 * self.builder._sources[0]._data + 
            0.8*0.5 * self.builder._sources[5]._data + 
            -0.8*1.2*0.3 * self.builder._sources[2]._data  # notice diff. source
        )
        pars = list(self.spec.central)
        pars[self.spec.ipar('xsec_sig')] = 1.2
        pars[self.spec.ipar('xsec_bg')] = 0.5
        pars[self.spec.ipar('lumi')] = 0.8
        pars[self.spec.ipar('syst1')] = -0.3
        comp = self.spec(pars)
        np.testing.assert_array_almost_equal(true, comp)

    def move_pars(self, pars):
        """Move all types of parameters to non-trivial values"""
        pars[self.spec.ipar('xsec_sig')] = 1.2
        pars[self.spec.ipar('xsec_bg')] = 0.5
        pars[self.spec.ipar('lumi')] = 0.8
        pars[self.spec.ipar('syst1')] = +0.2
        pars[self.spec.ipar('syst2')] = -0.3
        pars[self.spec.ipar('p')] = 1.2

    def test_spec_varied(self):
        """Check spectrum with all parameters varied"""
        true = (
            # Add source with lumi=0.8 and xsec=1.2
            0.8*1.2 * self.builder._sources[0]._data +
            # Add a 0.2 contribution from syst1
            0.8*1.2 * +0.2 * self.builder._sources[1]._data +
            # Add a -0.3 contribution from syst2
            0.8*1.2 * -0.3 * self.builder._sources[3]._data +
            0.8*0.5 * self.builder._sources[5]._data + 
            0.8*0.5 * -0.3 * self.builder._sources[6]._data +
            # Source 4 is the template, with strenght 1.2 and scaled by 5
            # as this is the form of the factor for the template
            0.8*1.2 * 5*1.2 * self.builder._sources[4]._data
        )
        pars = list(self.spec.central)
        self.move_pars(pars)
        comp = self.spec(pars)
        np.testing.assert_array_almost_equal(true, comp)

    def test_ll_nom(self):
        """Check the nominal log likelihood"""
        pars = list(self.spec.central)
        nominal = self.spec(pars)
        self.spec.set_data(nominal)  # nominal data
        stats = np.array(self.spec.stats)
        # event with nominal, ll penalty from poisson normalization
        ll = 0  # log likelihood
        ll += np.sum(logPoisson(nominal, nominal, stats))
        self.assertAlmostEqual(ll, self.spec.ll(pars))

    def test_ll_poisson(self):
        """Check the log likelihood with varied yields"""
        # Modify a few bins in data and check for poisson likelihood drop
        pars = list(self.spec.central)
        nominal = self.spec(pars)
        data = np.copy(nominal)
        data[1] *= 1.1
        data[2] *= 0.5
        stats = np.array(self.spec.stats)
        ll = 0  # log likelihood
        ll += np.sum(logPoisson(data, nominal, stats))
        # Set the fluctuated data, and check the log likelihood to nominal
        self.spec.set_data(data)
        self.assertAlmostEqual(ll/self.spec.ll(pars), 1)

    def test_ll_reg(self):
        """Check the log likelihood with varied systematics"""
        # Now modify all parameters, and check all regularizations are also
        # contributing
        centre = self.spec.central
        pars = np.copy(centre)
        self.move_pars(pars)
        # Data includes the shifts, so penalty will be only due to priors
        data = self.spec(pars)
        self.spec.set_data(data)
        stats = np.array(self.spec.stats)
        ll = 0
        ll += np.sum(logPoisson(data, data, stats))
        for ipar, par in enumerate(self.spec.pars):
            # Don't regularize free parameters
            if par in self.spec.unconstrained:
                continue
            # Scale is parameter value at 1 sigma, so need to subtract centre
            if pars[ipar] >= centre[ipar]:
                bound = self.spec.parinfo(par)['high']
            else:
                bound = self.spec.parinfo(par)['low']
            prior = self.builder._priors.get(par, None)
            if prior is None or prior['constraint'] == 'normal':
                ll += -0.5 * \
                    (pars[ipar]-centre[ipar])**2 / \
                    (bound-centre[ipar])**2
            elif prior is not None and prior['constraint'] == 'lognormal':
                ll += -0.5 * \
                    (np.log(pars[ipar])-np.log(centre[ipar]))**2 / \
                    (np.log(bound)-np.log(centre[ipar]))**2
        # Add contribution from the custom regularization on p which is
        # (p-syst1)**2
        ll += (pars[self.spec.ipar('p')]-pars[self.spec.ipar('syst1')])**2
        
        self.assertAlmostEqual(ll/self.spec.ll(pars), 1)

    def test_ll_mix(self):
        """Check the log likelihood with varied parameters"""
        pars = list(self.spec.central)
        data = np.copy(self.spec(pars))  # data at nominal, causes stat penalty
        self.spec.set_data(data)
        pars[self.spec.ipar('xsec_sig')] = 1.2
        pars[self.spec.ipar('p')] = 1.2
        varied = self.spec(pars)  # nominal expectation (with shifts)
        stats = np.array(self.spec.stats)
        ll = 0
        ll += np.sum(logPoisson(data, varied, stats))
        ll += -0.5 * (1.2-1)**2 / (self.spec.parinfo('xsec_sig')['high']-1)**2
        # Add custom regularizationonce more
        ll += (pars[self.spec.ipar('p')]-pars[self.spec.ipar('syst1')])**2
        self.assertAlmostEqual(ll/self.spec.ll(pars), 1)

    def test_grads(self):
        """Test the computed gradients agree with numerical computation"""
        pars = np.array(self.spec.central, dtype='float64')
        data = np.copy(self.spec(pars))
        data *= 1.1  # move away from centre to ensure non-zero gradients
        self.spec.set_data(data)
        self.move_pars(pars)  # move parameters to check proper partials

        ntol = 5
        dp = 10**(-ntol)

        for par in self.spec.pars:
            # Copy the central parameter values
            dpars = np.array(pars, dtype=np.float64)
            # Choose a parameter to chnage
            ipar = self.spec.ipar(par)

            nll = ROOT.Double(0)  # variable to pass by ref
            grads = dpars*0  # memory in which to store gradients
            # Compute the gradients at the central point
            self.spec._obj.FdF(pars, nll, grads)

            # Shift the parameter slightly down and compute likelihood there
            dpars[ipar] = pars[ipar] - dp;
            nlld = self.spec.nll(dpars)

            # Shift the parameter slightly up and compute likelihood there
            dpars[ipar] = pars[ipar] + dp;
            nllu = self.spec.nll(dpars)

            # Compute the observed gradient for this parameter
            dlldp = (nllu-nlld)/(2*dp)

            # The computed and numeric gradients should be similar, but won't
            # be indentical since the numeric one is an approximation
            self.assertAlmostEqual(dlldp/grads[ipar], 1, ntol-1)

    def test_grad_func(self):
        """Test that the dedicated gradient function agrees with FdF"""
        pars = np.array(self.spec.central, dtype='float64')
        data = np.copy(self.spec(pars))
        data *= 1.1  # move away from centre to ensure non-zero gradients
        self.spec.set_data(data)
        self.move_pars(pars)  # move parameters to check proper partials

        ll = ROOT.Double(0)
        grads1 = pars*0
        grads2 = pars*0

        self.spec._obj.FdF(pars, ll, grads1)
        self.spec._obj.Gradient(pars, grads2)

        np.testing.assert_almost_equal(grads1, grads2)

    def test_ngrads(self):
        """Test the positive likelihood gradients"""
        pars = np.array(self.spec.central, dtype='float')
        data = np.copy(self.spec(pars))
        data *= 1.1  # move away from centre to ensure non-zero gradients
        self.spec.set_data(data)
        self.move_pars(pars)  # move parameters to check proper partials

        grads = pars*0
        ngrads = pars*0

        # Object defaults to NLL for minimization
        self.spec._obj.Gradient(pars, grads)
        self.spec._obj.setNLL(False)
        self.spec._obj.Gradient(pars, ngrads)
        # Reset it
        self.spec._obj.setNLL(True)

        np.testing.assert_almost_equal(grads, -ngrads)

    def test_zero(self):
        builder = SpecBuilder('SpectrumZero')

        sig = [10., 11.]
        src_sig = Source(sig)
        src_sig.use_stats(.5*(2*np.array(sig))**0.5)
        src_sig.set_expression(
            'lumi*xsec_sig',
            ['lumi', 'xsec_sig'],
            ['xsec_sig', 'lumi'])
        builder.add_source(src_sig)
        builder.set_prior('xsec_sig', 1, 0.9, 1.2, 'normal')
        builder.set_prior('lumi', 1, 0.95, 1.05, 'lognormal')

        sig_syst1 = [-5, 0]
        src_sig_syst1_up = Source(sig_syst1, shapeof=src_sig)
        src_sig_syst1_up.set_expression('syst1', polarity='up')
        builder.add_source(src_sig_syst1_up)
        builder.set_prior('syst1', 0, -1, 1, 'normal')

        spec = builder.build()

        pars = list(spec.central)
        data = spec(pars)
        isyst = spec.ipar('syst1')

        pars[isyst] = 2
        # ensure syst made bin go to zero
        self.assertAlmostEqual(spec(pars)[0], 0)
        # ensure not NaN (0 data so bin is ignored)
        self.assertTrue(spec.ll(pars) == spec.ll(pars))

        # try again with negative bin value
        pars[isyst] = 3
        self.assertAlmostEqual(spec(pars)[0], -5)
        self.assertTrue(spec.ll(pars) == spec.ll(pars))

        # now set the data and check that ll goes to -inf
        spec.set_data(data)
        # check also grads, so need memory arrays
        pars = np.array(pars, dtype=np.float64)
        grads = pars*0

        pars[isyst] = 2
        self.assertEqual(spec.ll(pars), float('-inf'))
        spec._obj.Gradient(pars, grads)
        self.assertEqual(grads[isyst], float('inf'))

        pars[isyst] = 3
        self.assertEqual(spec.ll(pars), float('-inf'))
        spec._obj.Gradient(pars, grads)
        self.assertEqual(grads[isyst], float('inf'))



class TestSource(unittest.TestCase):
    def test_except_infer_pars(self):
        """Try to infer bad expression"""
        src = Source([])
        self.assertRaises(RuntimeError, src.set_expression, 'a+a')
        self.assertRaises(RuntimeError, src.set_expression, '2*a')
        self.assertRaises(ValueError, src.set_expression, '2*a', ['a'])
        self.assertRaises(ValueError, src.set_expression, '2*a', grads=['2'])
        self.assertRaises(ValueError, src.set_expression, 'a*b', ['a', 'b'], ['b'])

    def test_except_inherit(self):
        """Don't re-use an inherited parameter"""
        src1 = Source([])
        src1.set_expression('a')
        src2 = Source([], shapeof=src1)
        self.assertRaises(ValueError, src2.set_expression, 'a')
        self.assertRaises(ValueError, src2.set_expression, 'a', ['a'], ['1'])
        self.assertRaises(ValueError, src2.set_expression, 'a*b', ['a', 'b'], ['b', 'a'])

    def test_except_par_name(self):
        """Reject bad parameter names"""
        src = Source([])
        self.assertRaises(ValueError, src.set_expression, '_a', ['_a'], ['1'])
        self.assertRaises(ValueError, src.set_expression, '1a', ['1a'], ['1'])

    def test_except_polarity(self):
        """Reject bad polarity values"""
        src = Source([])
        self.assertRaises(ValueError, src.set_expression, 'a', polarity='invalid')

    def test_except_reset(self):
        """Don't allow re-setting expression"""
        src = Source([])
        src.set_expression('a')
        self.assertRaises(RuntimeError, src.set_expression, 'a')

    def test_data(self):
        """Data is correctly propagated"""
        src = Source([1,2,3])
        np.testing.assert_array_almost_equal([1,2,3], src._data, 15)

    def test_expression(self):
        """Set an expression, parameters and gradients"""
        src = Source([])
        src.set_expression('a*b*b', ['a', 'b'], ['b*b', '2*a*b'])
        self.assertEqual(['a', 'b'], src._pars)
        self.assertEqual(['b*b', '2*a*b'], src._grads)
        # Should convert numerical gradients
        src = Source([])
        src.set_expression('a', ['a'], [1])
        self.assertEqual(['1'], src._grads)

    def test_infer(self):
        """Infer parameters and gradients from expression"""
        src = Source([])
        src.set_expression('a')
        self.assertEqual(['a'], src._pars)
        self.assertEqual(['1'], src._grads)

        src = Source([])
        src.set_expression('a+b')
        self.assertEqual(['a', 'b'], src._pars)
        self.assertEqual(['1', '1'], src._grads)

    def test_inherit(self):
        """Test inheriting from parent sources"""
        # Setup a source which inherits from two others
        src1 = Source([])
        src1.set_expression('a+b')
        src2 = Source([], shapeof=src1)
        src2.set_expression('5*c*c', ['c'], ['10*c'])
        src3 = Source([], shapeof=src2)
        src3.set_expression('d+e')
        # Check the correct compound expression
        self.assertEqual('((a+b) * (5*c*c)) * (d+e)', src3._expr)
        # Ensure paramters correctly ammended
        self.assertEqual(['a', 'b', 'c', 'd', 'e'], src3._pars)
        # Check that the gradients are correctly propagated
        self.assertEqual('((1) * (5*c*c)) * (d+e)', src3._grads[0])
        self.assertEqual('((1) * (5*c*c)) * (d+e)', src3._grads[1])
        self.assertEqual('((10*c) * (a+b)) * (d+e)', src3._grads[2])
        self.assertEqual('(1) * ((a+b) * (5*c*c))', src3._grads[3])
        self.assertEqual('(1) * ((a+b) * (5*c*c))', src3._grads[4])


if __name__ == '__main__':
    unittest.main()

