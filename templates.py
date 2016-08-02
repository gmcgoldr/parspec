import numpy as np
import ROOT

import parspec


def estimate_stats(data):
    # copy into a proper array
    data = np.array(data)
    # copy into a strided array, grouping 3 bins at a time
    data = np.array(np.lib.stride_tricks.as_strided(
        x=data,
        shape=(
            len(data)-3+1,
            3), 
        strides=(
            data.dtype.itemsize,
            data.dtype.itemsize)))
    # fit each set of 3 points, using the same x coodrinates
    x = np.array([-1, 0, 1], dtype=float)
    m, b = np.polyfit(x, data.T, 1)
    # compute the projected values given the fit parameters
    y = m[:, np.newaxis]*x[np.newaxis, :] + b[:, np.newaxis]
    # the standard deviation of the difference should ressemble statistical
    # uncertainty
    return np.std(data-y)


class TemplateSource(object):
    """
    Accumulate information for a source process to the spectrum. Note that by
    convention, low and high refer to absolute values, whereas down and up
    refer to differences relative to the central value.
    """

    def __init__(self, name, data):
        """
        Each source must have a unique name, and a distribution of data it
        contributes to the spectrum.

        :param name: str
            unqiue name for this source
        :param data: [float]
            distribution contributing to spectrum
        """
        self._name = name
        self._data = np.array(data)
        self._lumi = False
        self._stat_errs = list()
        self._xsec = tuple()
        self._systematics = list()
        self._templates = list()

    def set_xsec(self, nominal, low=None, high=None):
        """
        Assign a cross seciton parameter for this source. If the low and high
        uncertainties aren't provided, the parameter won't be regularized.

        :param nominal: float
            nominal cross section value
        :param low: float
            cross section value below nominal which causes 1-sigma penalty
        :param high: float
            cross section value above nominal which causes 1-sigma penalty
        """
        self._xsec = (nominal, low, high)

    def use_lumi(self):
        """
        Allow this source to change with luminosity.
        """
        self._lumi = True

    def use_stats(self, errs):
        """
        The data values in this source are subject to statistical uncertainties
        (i.e. the source is simulated with MC).

        :param err: [float]
            statistical error on each bin
        """
        self._stat_errs = list(errs)

    def add_syst(self, name, data, stats=None, polarity=None):
        """
        Add a systematic variation to this source. This adds a parameter to
        the spectrum (or re-uses the parameter if the systematic name has been
        introduced in another source). This parameter is regularized such that
        the loglikelihood is halved when it reaches +/- 1.

        Note: data is given as absolute values, not realtive to nominal.

        :param name: str
            name of the systematic parameter (can be shared with other sources)
        :param data: [float]
            values of the spectrum when the systematic parameter takes on a
            value of +/- 1 sigma (depends on given polarity)
        :param stats: [float]
            statistical uncertainty on the *difference* of each bin under the
            influence of the systematic shape
        :param polarity: {'up', 'down'}
            this shape applies only if the systematic parameter is positive
            for 'up', or negative for 'down'
        """
        if polarity is not None and polarity not in ['up', 'down']:
            raise ValueError("Unrecognized polarity %s" % polarity)
        # Get the shifts realtive to nominal
        data = np.array(data) - self._data
        if polarity == 'down':
            data *= -1

        self._systematics.append((name, data, polarity, stats))

    def add_template(self, expr, data, pars=None, grads=None):
        """
        Add a template variation to this source. This adds parameters to the
        spectrum (or re-uses them if they are present in other sources). The
        parameters are not regularized, they are allowed to float.

        Note: data is given as absolute values, not realtive to nominal.

        :param expr: str
            C++ expression which yields the normalization for the template
        :param data: [float]
            values of the spectrum when the template expression evaluates to 1
        :param pars: [str]
            names of parameters used in the expression
        :param grads: [str]
            C++ expression which yields the dexpr/dpar for each parameter
        """
        data = np.array(data) - self._data  # relative to nominal
        self._templates.append((expr, data, pars, grads))
            

class TemplateMeasurement(object):
    """
    Accumulate information to build a templated spectrum.
    """

    def __init__(self, name='Templates'):
        """
        Build a templated spectrum with the given name. The name should be 
        unique in a session and work directory.

        :param name: str
            name of parametrized spectrum to generate
        """
        self._name = name
        self._sources = dict()  # mapping source names to TemplateSource
        self._lumi = tuple()  # nominal, low, high prior for luminosity
        self.spec = None  # ParSpec object

    def new_source(self, name, data):
        """
        Add a new source to the measurement, and return its object to configure.

        :param name: str
            TemplateSource name
        :param data: [float]
            TemplateSource data
        """
        if name in self._sources:
            raise RuntimeError("Source name already used: %s" % name)
        source = TemplateSource(name, data)
        self._sources[name] = source
        return source

    def set_lumi(self, nominal, err):
        """
        Set the prior information for the luminosity parameter.

        :param nominal: float
            nominal luminosity value
        :param err: float
            +/- uncertainty on luminosity
        """
        self._lumi = (nominal, nominal-err, nominal+err)

    def prepare(self):
        """
        Prepare and return the builder for the spectrum.
        """
        builder = parspec.SpecBuilder(self._name)
        uses_lumi = False  # determine if lumi is needed from sources

        for temp_src in self._sources.values():
            par_src = parspec.Source(temp_src._data)

            # Remember if any source is subject to lumi uncertainty
            uses_lumi |= temp_src._lumi

            xsec_name = 'xsec_%s' % temp_src._name 
            if temp_src._lumi and temp_src._xsec:
                # Factor is product of lumi and cross section, need to give 
                # parameter names and derivatives
                par_src.set_expression(
                    'lumi*%s' % xsec_name,
                    ['lumi', xsec_name],
                    [xsec_name, 'lumi'])
            elif temp_src._lumi:
                # Factor is just luminosity, derivative is inferred
                par_src.set_expression('lumi')
            elif temp_src._xsec:
                # Factor is just xsec, derivative is inferred
                par_src.set_expression(xsec_name)

            if temp_src._stat_errs:
                par_src.use_stats(temp_src._stat_errs)

            builder.add_source(par_src)

            # Add sources for systematic shifts
            for syst in temp_src._systematics:
                syst_name = 'syst_%s' % syst[0]
                src = parspec.Source(syst[1], shapeof=par_src)
                if syst[3] is not None:
                    src.use_stats(syst[3])
                src.set_expression(syst_name, polarity=syst[2])
                builder.add_source(src)

            # Add sources for templates
            for temp in temp_src._templates:
                src = parspec.Source(temp[1], shapeof=par_src)
                src.set_expression(temp[0], pars=temp[2], grads=temp[3])
                builder.add_source(src)

            # Add regularization for x-section if provided
            if temp_src._xsec:
                builder.set_prior(
                    xsec_name, 
                    *temp_src._xsec, 
                    ptype='lognormal')

            # Add regularization for systematics (overwrite pervious if the
            # same name is in another source, but doesn't matter)
            for syst in temp_src._systematics:
                syst_name = 'syst_%s' % syst[0]
                builder.set_prior(syst_name, 0, -1, 1)

        if uses_lumi:
            if not self._lumi:
                raise RuntimeError("No luminosity uncertainty set")
            builder.set_prior(
                'lumi', 
                *self._lumi,
                ptype='lognormal')

        return builder

    def build(self, builder=None):
        """
        Build the ParSpec object for this templated measurement.
        """
        if builder is None:
            builder = self.prepare()
        self.spec = builder.build()
