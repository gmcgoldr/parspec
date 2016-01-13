import numpy as np
import ROOT

import parspec

class TemplateSource(object):
    """
    Accumulate information for a source process to the spectrum.
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
        self._data = data
        self._lumi = False
        self._stats = False
        self._xsec = tuple()
        self._systematics = list()
        self._templates = list()

    def set_xsec(self, nominal, down=None, up=None):
        """
        Assign a cross seciton parameter for this source. If the up and down
        uncertainties aren't provided, the parameter won't be regularized.

        :param nominal: float
            nominal cross section value
        :param down: float
            cross section value below nominal which causes 1-sigma penalty
        :param up: float
            cross section value above nominal which causes 1-sigma penalty
        """
        self._xsec = (nominal, down, up)

    def use_lumi(self):
        """
        Allow this source to change with luminosity.
        """
        self._lumi = True

    def use_stats(self):
        """
        The data values in this source are subject to statistical uncertainties
        (i.e. the source is simulated with MC).
        """
        self._stats = True

    def add_syst(self, name, data, polarity=None):
        """
        Add a systematic variation to this source. This adds a parameter to
        the spectrum (or re-uses the parameter if the systematic name has been
        introduced in another source). This parameter is regularized such that
        the loglikelihood is halved when it reaches +/- 1.

        :param name: str
            name of the systematic parameter (can be shared with other sources)
        :param data: [float]
            values this systematic adds to nominal data when its value is
            1-sigma above nominal
        :param polarity: {'up', 'down'}
            this shape applies only if the systematic parameter is positive
            for 'up', or negative for 'down'
        """
        if polarity is not None and polarity not in {'up', 'down'}:
            raise ValueError("Unrecognized polarity %s" % polarity)
        # Make a numpy array from the data for manipulation
        data = np.array(data)
        # If the data is for only down values of the systematic, then the
        # shifts are already negative. But they will be multiplied by the
        # systematic value, so change polarity now.
        if polarity == 'down':
            data *= -1
        self._systematics.append((name, data, polarity))

    def add_template(self, expr, data, pars=None, grads=None):
        """
        Add a template variation to this source. This adds parameters to the
        spectrum (or re-uses them if they are present in other sources). The
        parameters are not regularized, they are allowed to float.

        :param expr: str
            C++ expression which yields the normalization for the template
        :param data: [float]
            values which this template add to nominal data (scale with expr)
        :param pars: [str]
            names of parameters used in the expression
        :param grads: [str]
            C++ expression which yields the dexpr/dpar for each parameter
        """
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
        self._lumi = tuple()  # nominal, down, up prior for luminosity
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
        uses_lumi = False

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

            if temp_src._stats:
                par_src.use_stats()

            builder.add_source(par_src)

            # Add sources for systematic shifts
            for syst in temp_src._systematics:
                syst_name = 'syst_%s' % syst[0]
                src = parspec.Source(syst[1], shapeof=par_src)
                src.set_expression(syst_name, polarity=syst[2])
                builder.add_source(src)

            # Add sources for templates
            for temp in temp_src._templates:
                src = parspec.Source(temp[1], shapeof=par_src)
                src.set_expression(temp[0], pars=temp[2], grads=temp[3])
                builder.add_source(src)

            # Add regularization for x-section if provided
            if temp_src._xsec:
                builder.set_prior(xsec_name, *temp_src._xsec)

            # Add regularization for systematics (overwrite pervious if the
            # same name is in another source, but doesn't matter)
            for syst in temp_src._systematics:
                syst_name = 'syst_%s' % syst[0]
                builder.set_prior(syst_name, 0, -1, 1)

        if uses_lumi:
            if not self._lumi:
                raise RuntimeError("No luminosity uncertainty set")
            builder.set_prior('lumi', *self._lumi)

        return builder

    def build(self, builder=None):
        """
        Build the ParSpec object for this templated measurement.
        """
        if builder is None:
            builder = self.prepare()
        self.spec = builder.build()
