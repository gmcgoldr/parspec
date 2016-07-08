import os
import re
import math

import numpy as np
import ROOT

# Check if a given name will be a valid C++ variable name, but don't allow
# starting with underscores
_name_re = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]*$')

_build_path = ''
def set_build_path(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(path)
    ROOT.gSystem.SetBuildDir(path)
    global _build_path
    _build_path = path


def _load_code():
    """Load the C++ template code from the same path as this file"""
    code_path = os.path.dirname(os.path.relpath(__file__))
    with open(os.path.join(code_path, 'parspec.cxx'), 'r') as fin:
        code = fin.read()
    return code
_base_code = _load_code()


class ParSpec(object):
    """
    Wrapper for a constructed parameterized spectrum. The object has a compiled
    C++ object which cannot be modified. Access to internals are not mutable to
    reflect this situation. Can't change a value without realizing it will have
    no impact on the spectrum.
    """

    def __init__(self, name, pars, ncols, central, lows, highs, obj):
        """
        Wrapper for a compile parameterized spectrum object. All interals are
        set at initialization, and are no longer modified.

        :param name: str
            name of the object (library and class use this)
        :param pars: [str]
            list of parameter names, in the order they are used in the object
        :param ncols: int
            number of columns in the spectrum
        :param central: [float]
            central value for each parameter
        :param lows: [float]
            -1 sigma value of each parameter
        :param highs: [float]
            +1 sigma value of each parameter
        :param obj: ROOT.name
            ROOT wrapper for the compiled C++ spectrum object
        """
        self._name = name
        self._pars = tuple(pars)
        self._ipars = dict([(par, i) for i, par in enumerate(pars)])
        self._npars = len(self._pars)
        self._ncols = ncols
        self._obj = obj

        # Note: used internally, don't return as it is mutable
        self._central = tuple(np.array(central, dtype='float64'))

        lows = np.array(lows)
        highs = np.array(highs)

        # Compute the effective stats with default parameters
        istats = [
            i for i in range(self._npars) 
            if self._pars[i].startswith('stat')]
        x = np.array(self._central)
        vals = np.zeros(self._ncols, dtype=np.float64)
        stats = np.zeros(self._ncols, dtype=np.float64)
        self._obj.Compute(x, vals, stats)
        lows[istats] = -stats**0.5
        highs[istats] = stats**0.5

        self._bounds = np.array([lows, highs], dtype='float64').T
        self._scales = tuple(0.5 * (self._bounds[:, 1] - self._bounds[:, 0]))
        self._unconstrained = tuple(
            [p for i, p in enumerate(self._pars) 
            if self._scales[i] == 0])

        # At this point, it's worth ensuring that all scales are posiitve
        assert(not np.any(np.array(self._scales) < 0))

    def _prep_pars(self, x):
        """Use numpy to get an array of parameter values, ensuring ndims"""
        x = np.asarray(x).astype(np.float64)
        if len(x) != self._npars:
            raise ValueError("Incorrect paramter dimensions")
        return x

    def _make_ipar(self, par):
        """Return parameter index, given either parameter string or index"""
        if isinstance(par, int):
            return par
        else:
            return self._ipars[par]

    @property
    def name(self):
        """Return name of spectrum object, used for its class"""
        return self._name

    @property
    def pars(self):
        """Return list of parameter names, in order of indices"""
        return self._pars

    @property
    def npars(self):
        """Return the number of parameters controlling the spectrum"""
        return self._npars

    @property
    def ncols(self):
        """Return the number of columns (bins) in the spectrum"""
        return self._ncols

    @property
    def central(self):
        """Return list of central parameter values"""
        return self._central

    @property
    def scales(self):
        """Return list of parameter scales (symmetrized constraints)"""
        return self._scales

    @property
    def unconstrained(self):
        """Return list of unconstrained parameter names"""
        return self._unconstrained

    def ipar(self, par):
        """
        Get the index for a given parameter name.

        :param par: str
            parameter name
        """
        return self._ipars[par]

    def parinfo(self, par):
        """
        Get the information associated with a parameter.

        :param par: str or int
            parameter name or index
        :return: dict
            index: index of the parameter
            name: name of the parameter
            central: prior central value
            low: prior -1 sigma value
            high: prior +1 sigma value
        """
        ipar = self._make_ipar(par)
        return {
            'index': ipar,
            'name': self.pars[ipar],
            'central': self.central[ipar],
            'low': self._bounds[ipar, 0],
            'high': self._bounds[ipar, 1]}

    def __call__(self, x):
        """
        Compute the spectrum for the given parameters.

        :param x: [float]
            list of parameter values
        """
        x = self._prep_pars(x)
        vals = np.zeros(self._ncols, dtype=np.float64)
        self._obj.Compute(x, vals)
        return vals

    def specstats(self, x):
        """
        Compute the spectrum and statistics for the given parameters.

        :param x: [float]
            list of parameter values
        """
        x = self._prep_pars(x)
        vals = np.zeros(self._ncols, dtype=np.float64)
        stats = np.zeros(self._ncols, dtype=np.float64)
        self._obj.Compute(x, vals, stats)
        return vals, stats**0.5

    def nll(self, x):
        """
        Compute the negative log likelihood for the given parameters.

        :param x: [float]
            list of parameter values
        """
        x = self._prep_pars(x)
        return self._obj(x)

    def ll(self, x):
        """
        Compute the log likelihood for the given parameters.

        :param x: [float]
            list of parameter values
        """
        return -self.nll(x)

    def set_data(self, data):
        """
        Set the data spectrum used as the reference point for likelihood
        evaluation. Note that this copies the given data.

        :param data: [float]
            list of data values for each column
        """
        data = np.asarray(data).astype('float64')
        if len(data) != self._ncols:
            raise ValueError("Incorrect data dimensions")
        # Note that _obj copies the memory
        self._obj.setData(data)

    def build_minimizer(self, central=None, scales=None):
        """
        Build and return a ROOT::TMinimizer object and configure to minimize
        the negative log likleihood.

        :param central: [float]
            override default central values with these
        :param scales: [float]
            override default scales with these ones
        """
        minimizer = ROOT.Math.Factory.CreateMinimizer("Minuit")
        minimizer.SetFunction(self._obj)

        if central is None:
            central = self.central

        if scales is None:
            scales = self.scales

        for par in self.pars:
            ipar = self.ipar(par)
            val = central[ipar]
            scale = scales[ipar]
            if scale == 0:
                scale = 1
            minimizer.SetVariable(ipar, par, val, scale)

        # When the LL is halved, 1 sigma is reached
        minimizer.SetErrorDef(0.5)

        return minimizer


class Source(object):
    """
    A source of data contributing to a spectrum. Typically, sources are physics
    processes such as the signal or backgrounds. Each source gets a separate
    normalization factor, such that their relative contributions to the 
    spectrum can differ.
    """

    def __init__(self, data, shapeof=None):
        """
        Create a new Source object.

        :param data: iterable
            values this source contributes to each spectral bin
        :param shapeof: Source
            indicates that this source applies on top of another source, in
            which case it inherits that source's factor.
        """
        # The source's data is a 1D array 
        self._data = np.array(data, dtype='float64')
        if len(self._data.shape) != 1:
            raise ValueError("Source data must be 1D")

        # Re-setting expression can lead to unexpected behaviour
        self._isset = False
        # C++ expression to compute a scaling for this source
        self._expr = ''
        # Gradient of the factor w.r.t. to each parameter
        self._grads = list()
        # List of variables names used in computing this source factor
        self._pars = list()
        # Relative statistical uncertainty on the count in each bin
        self._stats = np.zeros(len(data), dtype='float64')

        # Inherit the expresion from the shapeof
        if shapeof:
            self._expr = shapeof._expr
            self._grads = shapeof._grads[:]
            self._pars = shapeof._pars[:]

    def set_expression(self, expr, pars=None, grads=None, polarity=''):
        """
        Set an expression whose value will be used to scale this source. The
        expression will be compiled in C++, with the cmath and limits headers.

        Provide the parameters used in the expression, and the gradient of
        the expressiont w.r.t. to each parameter.

        If a single parameter is used, or the parameters are a simple sum,
        then the parameters and gradients can be inferred if not provided.

        :param expr: str
            C++ expression to evaluate the normalization factor
        :param pars: [str]
            list of names of the variables used in the expression
        :param grads: [str]
            list of expressions yielding the gradient of the expression w.r.t.
            each parameter (in the same order)
        :param polarity: str
            turn off contribution if expression evaluates below (above) zero
            and polarity is set to up (down).
        """
        if self._isset:
            raise RuntimeError("Can't re-set expression")

        # Try to infer parameters and gradients
        if not pars and not grads:
            # Get the list of added terms (stripped of spaces)
            pars = map(lambda s: s.strip(), expr.split('+'))
            # Check for term repetition
            if len(set(pars)) != len(pars):
                raise RuntimeError("Can't infer parameters")
            # Check if a term isn't a simple variable name
            for par in pars:
                if not _name_re.match(par):
                    raise RuntimeError("Can't infer parameters")
            # Simple sum, each term has a gradient of 1
            grads = ['1'] * len(pars)

        elif not grads or not pars:
            raise ValueError("Can't povide only one of parameters or gradients")

        elif len(grads) != len(pars):
            raise ValueError("Gradients don't match parameters")

        # Can't re-use parameters from parent source's factor (due to gradients)
        if len(set(pars) & set(self._pars)) != 0:
            raise ValueError("Can't re-use parent source parameter")

        # Ensure the parameter names are correct and won't conflict
        for par in pars:
            if not _name_re.match(par):
                raise ValueError("Invalid parameter name: %s" % par)

        expr = str(expr)
        grads = map(str, grads)

        # Apply polarity bounds
        if polarity == 'down':
            grads = ['((%s)<0) ? %s : 0' % (expr, g) for g in grads]
            expr = '((%s)<0) ? %s : 0' % (expr, expr)
        elif polarity == 'up':
            grads = ['((%s)>0) ? %s : 0' % (expr, g) for g in grads]
            expr = '((%s)>0) ? %s : 0' % (expr, expr)
        elif polarity:
            raise ValueError("Unrecognized polarity value: %s" % polarity)

        # Validate everything before setting, then ensure no more setting
        self._isset = True

        # Combine with parent source's expression and gradients
        if self._expr:
            # The gradients are multiplied by the parent expresssion
            grads = ['(%s) * (%s)' % (g, self._expr) for g in grads]
            # The parent gradients are multplied by the new expression
            self._grads = ['(%s) * (%s)' % (g, expr) for g in self._grads]
            # The factor is multiplied by the parent factor
            self._expr = '(%s) * (%s)' % (self._expr, expr)
            # Append the gradients for the new parameters
            self._grads += grads

        else:
            self._expr = expr
            self._grads = grads

        # Combine parameters for this factor to parent sources
        self._pars += pars

    def use_stats(self, errs):
        """
        Indicate that the contents of this source are subject to statistical
        uncertianties (e.g. Monte-Carlo computed sources).

        :param err: [float]
            statistical error on each bin
        """
        if len(errs) != len(self._stats):
            raise ValueError("Provided errors don't match data")
        self._stats = np.array(errs, dtype='float64')


class SpecBuilder(object):
    """
    Collect information required to build the spectrum code, generate the code
    and compile it with ROOT's ACLIC.
    """

    def __init__(self, name):
        """
        Prepare builder for a new spectrum.

        :param name: str
            name of the spectrum class, should be unique within a process and
            working directory
        """
        if not _name_re.match(name):
            raise ValueError("Invalid spectrum name: %s" % name)

        self.name = name
        # List of all sources
        self._sources = list()
        # List of normalization factor expression for each source
        self._pars = set()
        # Map varible name to the Gaussian constraint parameters
        self._priors = dict()
        # Number of columns determined from first source added
        self._ncols = None

        # Remember the stat parameter names explicitey
        self._stat_pars = list()

        # List of additional regularization expressions and gradients
        self._regularizations = list()

    def add_source(self, source):
        """
        Add a source contributing to the spectrum contents.

        Sets or checks the number of bins. Incorporates parameters used in
        the source factor expression. Counts towards statistical uncertainty
        parameters if toggled.

        :param source: Source
            configured Source object
        """
        if self._ncols is None:
            self._ncols = len(source._data)
        elif len(source._data) != self._ncols:
            raise RuntimeError("Source bins doesn't match spectrum")

        self._sources.append(source)
        # Keep the set of all parameter names
        self._pars |= set(source._pars)

        if len(source._stats) > 0 and not self._stat_pars:
            # Setup parameter names for each bin uncertainty. Make sure
            # they sort alphanumerically, so pad with enough zeros
            nzeros = str(int(math.log10(self._ncols))+1)
            self._stat_pars = [
                ('stat%0'+nzeros+'d') % i for i in range(self._ncols)]
            self._pars |= set(self._stat_pars)

    def set_prior(self, name, central, low=None, high=None):
        """
        Set the prior value for a parameter. If low and high are None, the
        parameter isn't constrained.

        :param name: str
            name of parameter to constrain
        :param central: float
            prior value for the parameter
        :param low: float
            penalize by e^-0.5 at this value (below central)
        :param high: float
            penalize by e^-0.5 at this value (above central)
        """
        if bool(low) != bool(high):
            raise ValueError("Only one prior constraint is provided")
        if low is not None and not low <= central:
            raise ValueError("Invalid lower bound")
        if high is not None and not high >= central:
            raise ValueError("Invalid lower bound")
        self._priors[name] = (central, low, high)

    def add_regularization(self, expr, pars, grads):
        """
        Add an arbitrary regularization expression which will be added to
        the log likelihood computation.

        :param expr: str
            C++ expression to evaluate the regularization contribution to the
            log likelihood
        :param pars: [str]
            list of names of the variables used in the expression
        :param grads: [str]
            list of expressions yielding the gradient of the expression w.r.t.
            each parameter (in the same order)
        """
        if len(grads) != len(pars):
            raise ValueError("Gradients don't match parameters")

        # Ensure the parameter names are correct and won't conflict
        for par in pars:
            if not _name_re.match(par):
                raise ValueError("Invalid parameter name: %s" % par)

        # Add parameter names (in case a new regularization parameter was
        # introduced)
        self._pars |= set(pars)

        expr = str(expr)
        grads = map(str, grads)

        self._regularizations.append((expr, pars, grads))

    def build(self):
        """
        Build the C++ code, compile, and return a spectrum object.
        """

        # Note: this will be the final order of the parameters
        pars = sorted(list(self._pars))
        ipars = dict([(par, i) for i, par in enumerate(pars)])

        code = _base_code

        code_factors = list()
        code_rowpars = list()
        code_rownpars = list()
        code_pargrads = list()

        # Write code to sum each source
        for irow, source in enumerate(self._sources):
            # Expression to evaluate source factor
            code_factors.append(source._expr if source._expr else '1')
            code_rownpars.append(0)
            # Add code to compute gradients of each source factor w.r.t to
            # the parameters in the factor expression for this source
            for par, grad in zip(source._pars, source._grads):
                code_rownpars[-1] += 1
                code_rowpars.append(str(ipars[par]))
                code_pargrads.append(grad)

        code_factors = ',\n'.join(code_factors)
        code_rowpars = ', '.join(code_rowpars)
        code_rownpars = ', '.join(map(str, code_rownpars))
        code_pargrads = ',\n'.join(code_pargrads)

        code_ll = list()   # code computes the log likelihood
        code_gll = list()  # code computes the log likelihood and gradients

        # Add constraint terms for regularizations
        for rexpr, rpars, rgrads in self._regularizations:
            code_ll.append('_f += %s;' % rexpr)
            code_gll.append(code_ll[-1])
            for par, grad in zip(rpars, rgrads):
                code_gll.append('_df[%d] += %s;' % (ipars[par], grad))

        # Substitue generated code into the template
        code = code.replace('__NAME__', self.name)
        code = code.replace('__NROWS__', str(len(self._sources)))
        code = code.replace('__NCOLS__', str(self._ncols))
        code = code.replace('__NDIMS__', str(len(pars)))
        code = code.replace('__ISTATS__', str(ipars[self._stat_pars[0]]))
        code = code.replace('__FACTORS__', '\n%s\n' % code_factors)
        code = code.replace('__PARGRADS__', '\n%s\n' % code_pargrads)
        code = code.replace('__ROWNPARS__', '\n%s\n' % code_rownpars)
        code = code.replace('__ROWPARS__', '\n%s\n' % code_rowpars)
        code = code.replace('__LL__', '\n%s\n' % ('\n'.join(code_ll)))
        code = code.replace('__GLL__', '\n%s\n' % ('\n'.join(code_gll)))

        # replace named variables by parameter id
        for ipar, par in enumerate(pars):
            code = re.sub(r'(?<=[^\w\d])%s(?=\W)'%par, '_x[%d]'%ipar, code)

        binary_data_path = os.path.join(
            _build_path,
            'data_parspec_%s.bin' % self.name)
        binary_data = open(binary_data_path, 'wb')

        code = code.replace('__DATAPATH__', binary_data_path)

        # Piror data to insert in the code
        prior0_data = ['0'] * len(pars)  # central value
        priorDown_data = ['0'] * len(pars)  # down scale
        priorUp_data = ['0'] * len(pars)  # up scale
        priorMask_data = ['0'] * len(pars)  # 1 if the parameter is regularized
        # Find the prior information for each parameter
        for ipar, par in enumerate(pars):
            prior_vals = self._priors.get(par, None)
            if prior_vals is None:
                continue
            # Update data for those that are regularized
            prior0_data[ipar] = prior_vals[0]
            priorDown_data[ipar] = prior_vals[0]-prior_vals[1]
            priorUp_data[ipar] = prior_vals[0]-prior_vals[2]
            priorMask_data[ipar] = '1'
        # Write into the file
        binary_data.write(np.array(prior0_data, dtype='float64'))
        binary_data.write(np.array(priorDown_data, dtype='float64'))
        binary_data.write(np.array(priorUp_data, dtype='float64'))
        binary_data.write(np.array(priorMask_data, dtype='int32'))

        # Source data (spectrum contributions) to insert int he code
        sources_data = [[v for v in s._data] for s in self._sources]
        binary_data.write(np.array(sources_data, dtype='float64'))

        # Statistics for each data bin (statistical uncertainty squared)
        source_stats_data = [[v**2 for v in s._stats] for s in self._sources]
        binary_data.write(np.array(source_stats_data, dtype='float64'))

        binary_data.close()

        # Write out the generated code
        code_file = 'comp_parspec_%s.cxx' % self.name
        code_path = os.path.join(_build_path, code_file)
        code_exists = False

        # First, check if identical file exists, in which case it might already
        # be compiled, and no need to re-compile
        try:
            with open(code_path, 'r') as fin:
                old_code = fin.read()
                if old_code == code:
                    code_exists = True
        except IOError:
            pass

        if not code_exists:
            with open(code_path, 'w') as fout:
                fout.write(code)

        # Ask ROOT to compile and link the code
        prev_level = ROOT.gErrorIgnoreLevel
        ROOT.gErrorIgnoreLevel = ROOT.kWarning
        if ROOT.gROOT.LoadMacro(code_path+'+') != 0:
            raise RuntimeError("Unable to compile macro")
        ROOT.gErrorIgnoreLevel = prev_level
        # Grab the spectrum constructor from the compiled code
        constructor = getattr(ROOT, self.name)

        # Gather prior information for the parameters 
        central = [0] * len(pars)
        lows = [0] * len(pars)
        highs = [0] * len(pars)

        # Tell the spectrum about the central value of constrained parameters
        for par in self._priors:
            ipar = ipars[par]
            central[ipar] = self._priors[par][0]
            # Use scales if prior is constrained
            if self._priors[par][1] is not None:
                lows[ipar] = self._priors[par][1]
                highs[ipar] = self._priors[par][2]
            # Unconstrained prior
            else:
                lows[ipar] = central[ipar]
                highs[ipar] = central[ipar]

        # NOTE: statistical parameter priors computed on construction

        return ParSpec(
            self.name,
            pars,
            self._ncols,
            central,
            lows,
            highs,
            constructor())
