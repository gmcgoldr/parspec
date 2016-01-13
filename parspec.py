import os
import re
import math

import numpy as np
import ROOT

# Check if a given name will be a valid C++ variable name, but don't allow
# starting with underscores
_name_re = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]*$')


def _load_code():
    """Load the C++ template code from the same path as this file"""
    code_path = os.path.dirname(os.path.relpath(__file__))
    with open(os.path.join(code_path, 'parspec.cxx'), 'r') as fin:
        code = fin.read()
    return code
_base_code = _load_code()


class ParSpec(object):
    """
    Wrapper for a constructed parameterized spectrum.
    """

    def __init__(self, name, pars, ncols, constructor):
        """
        Build a new compiled spectrum object, and wrap it.

        :param name: str
            name of the object (library and class use this)
        :param pars: [str]
            list of parameter names, in the order they are used in the object
        :param ncols: int
            number of columns in the spectrum
        :param constructor: function
            constructor for the object
        """
        self.name = name
        self._pars = pars[:]
        self._ipars = {par: i for i, par in enumerate(pars)}
        self._ncols = ncols
        self._npars = len(self._pars)
        self._obj = constructor()
        self._centralx = np.zeros(self._npars, dtype=np.float64)
        self._scales = [[0, 0] for _ in range(self._npars)]

    def _prep_pars(self, x):
        """Use numpy to get an array of parameter values, ensuring ndims"""
        x = np.array(x, dtype=np.float64)
        if len(x) != self._npars:
            raise ValueError("Incorrect paramter dimensions")
        return x

    def _make_ipar(self, par):
        """Return parameter index, given either parameter string or index"""
        if isinstance(par, int):
            return par
        else:
            return self._ipars[par]

    def pars(self):
        """Return a copy of the parameter list"""
        return self._pars[:]

    def npars(self):
        """Quick access to the number of parameters"""
        return self._npars

    def ipar(self, par):
        """
        Get the index for a given parameter name.

        :param par: str
            parameter name
        """
        return self._ipars[par]

    def centralx(self):
        """Return the central parameter values"""
        # Copy so that in-place edits aren't accidentally stored
        return np.copy(self._centralx)

    def scalesx(self):
        """Return the symmetrized scale for each parameter"""
        scales = np.array(self._scales, dtype=float)
        scales = np.fabs(scales[:, 0] - scales[:, 1])/2
        return scales

    def get_scale(self, par, pol=''):
        """
        Get the scale for a parameter. If no scale is set, returns 0.

        :param par: str or int
            parameter name or index
        :param pol: str
            polarity to return one of 'up', 'down' or nothing for symmetric
        """
        if not pol or pol == 'up':
            return self._scales[self._make_ipar(par)][1]
        elif pol == 'down':
            return self._scales[self._make_ipar(par)][0]
        else:
            raise ValueError("Unknown polarity value: %s" % pol)

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
        data = np.array(data, dtype=np.float64)
        if len(data) != self._ncols:
            raise ValueError("Incorrect data dimensions")
        self._obj.setData(data)

    def build_minimizer(self, centre=None, scales=None):
        """
        Build and return a ROOT::TMinimizer object and configure to minimize
        the negative log likleihood.

        :param centre: [float]
            override default central values with these
        :param scales: [float]
            override default scales with these ones
        """
        minimizer = ROOT.Math.Factory.CreateMinimizer("Minuit")
        minimizer.SetFunction(self._obj)

        if centre is None:
            centre = self.centralx()

        if scales is None:
            scales = self.scalesx()

        for par in self.pars():
            ipar = self.ipar(par)
            val = centre[ipar]
            scale = scales[ipar]
            if scale == 0:
                scale = 1
            minimizer.SetVariable(ipar, par, val, scale)

        # When the LL is halved, 1 sigma is reached
        minimizer.SetErrorDef(0.5)
        # Default tolerance is 1e-2, seems a bit generous
        minimizer.SetTolerance(1e-4)

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
        self._data = np.array(data, dtype=np.float64)
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

        # Indicates if the contents of this source are subject to stat. uncert
        self._use_stats = False

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

    def use_stats(self):
        """
        Indicate that the contents of this source are subject to statistical
        uncertianties (e.g. Monte-Carlo computed sources).
        """
        self._use_stats = True


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

        # Remember the total number of events subject to statistical
        # uncertainties in each bin
        self._stat_scales = None
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
            self._stat_scales = np.zeros(self._ncols, dtype=np.float64)

        elif len(source._data) != self._ncols:
            raise RuntimeError("Source bins doesn't match spectrum")

        self._sources.append(source)
        # Keep the set of all parameter names
        self._pars |= set(source._pars)

        if source._use_stats:
            if not self._stat_pars:
                # Setup parameter names for each bin uncertainty. Make sure
                # they sort alphanumerically, so pad with enough zeros
                nzeros = str(int(math.log10(self._ncols))+1)
                self._stat_pars = [
                    ('stat%0'+nzeros+'d') % i for i in range(self._ncols)]
                self._pars |= set(self._stat_pars)
            # Count this source's contents towards stat uncertainty
            self._stat_scales += source._data

    def set_prior(self, name, centre, down=None, up=None):
        """
        Set the prior value for a parameter. If up and down are None, the
        parameter isn't constrained.

        :param name: str
            name of parameter to constrain
        :param centre: float
            prior value for the parameter
        :param down: float
            penalize by e^-0.5 at this value below centre
        :param up: float
            penalize by e^-0.5 at this value above centre
        """
        if bool(down) != bool(up):
            raise ValueError("Only one prior constraint is provided")
        if (down is not None or up is not None) and not (down <= centre <= up):
            raise ValueError("Prior constraints aren't correctly bound")
        self._priors[name] = (centre, down, up)

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

        pars = sorted(list(self._pars))
        ipars = {par: i for i, par in enumerate(pars)}

        code = _base_code

        code_pars = list()  # code assigns parameters to variables
        for ipar, par in enumerate(pars):
            code_pars.append(
                'const double %s = _x[%d];' % (par, ipar))
        # If stattistical parameters are used, they are all sorted together,
        # and occupy a continous block of the parameters. Get the address to
        # the start of the block using the first stat par name.
        if self._stat_pars:
            code_pars.append(
                'const double* _stats = _x + %d;' % 
                ipars[self._stat_pars[0]])
            code_pars.append(
                'const unsigned _istats = %d;' %
                ipars[self._stat_pars[0]])

        code_spec = list()   # code computes the spectrum
        code_gspec = list()  # code computes the spectrum and gradients
        
        # Write code to sum each source
        for irow, source in enumerate(self._sources):
            # Expression to evaluate source factor
            factor = source._expr if source._expr else '1'
            # Sum each source column with the factor into the spectrum
            code_spec.append('for (unsigned i = 0; i < _ncols; i++) {')
            # Source values (array on stack, in the scope of the loop)
            code_spec.append(
                '  const double _cols[] = {%s};' % (
                ', '.join(['%.7e' % v for v in source._data])))
            # The value to add to this spectrum column (scale by stat factor
            # if this source has statistical uncertainty)
            code_spec.append(
                '  const double val = _cols[i] * %s;' % (
                '_stats[i]' if source._use_stats else '1'))
            # Sum into the spetrum, with the source factor
            code_spec.append('  _spec[i] += (%s) * val;' % factor)
            # Use those lines of code in gradient computation as well
            code_gspec += code_spec[-4:]
            # Add code to compute gradients of each spectrum bin w.r.t to
            # the parameters in the factor expression for this source
            for par, grad in zip(source._pars, source._grads):
                # Compute db/dp, b is the bin value, p is the parameter 
                code_gspec.append(
                    '  _grads[%d*_ncols+i] += (%s) * val;' %  (
                    ipars[par], grad))
            # If the source contributes to statistical uncertainty, also
            # compute db/ds, s is the statistical factor
            if source._use_stats:
                code_gspec.append(
                    '  _grads[(_istats+i)*_ncols+i] += (%s) * _cols[i];' %
                    factor)
            # Close the column loop
            code_spec.append('}')
            code_gspec.append(code_spec[-1])
        # When computing just the spectrum, don't allow contents to drop below 0
        code_spec.append(
            'for (unsigned i = 0; i < _ncols; i++) '
            '_spec[i] = std::max(0., _spec[i]);')
        # When computing gradients, cap at 1 to avoid nan (Poisson uncertainty
        # is skewed at low numbers, but it certainly can't be 0)
        code_gspec.append(
            'for (unsigned i = 0; i < _ncols; i++) '
            '_spec[i] = std::max(1., _spec[i]);')

        code_ll = list()   # code computes the log likelihood
        code_gll = list()  # code computes the log likelihood and gradients

        # Compute the log likelihood of the data, given the true values in the
        # spectrum bins (approximation of the Poisson probability)
        code_ll.append(
            'for (unsigned i = 0; i < _ncols; i++) {\n'
            '  _f += -0.5 * std::pow(_spec[i]-_data[i],2) / _spec[i];')
        # Use this also when computing gradients
        code_gll.append(code_ll[-1])
        # Compute the gradient of ll w.r.t. to each parameter using chain rule:
        # dll/dp = dll/db db/dp, where ll is the log likelihood, b is a bin
        # value and p is parameter (b is a function of p). _grads[j*_ncols_i]
        # gives the db/dp for parater j and spectrum bin i
        for par in self._pars:
            code_gll.append(
                '  _df[%d] += '
                '( 0.5*std::pow(_spec[i]-_data[i],2)/std::pow(_spec[i],2) '
                '- (_spec[i]-_data[i])/_spec[i] ) * '
                '_grads[%d*_ncols+i];' % 
                (ipars[par], ipars[par]))
        code_ll.append('}')
        code_gll.append(code_ll[-1])

        # stat_scales is the count of events in each column with a stat.
        # uncertainty. The total in that column can change by n/sqrt(n)
        # with a 1 sigma penalty. Scale for fraction is just 1/n
        for par, scale in zip(self._stat_pars, self._stat_scales):
            scale = 1./max(scale, 1)**0.5
            self._priors[par] = (1, 1-scale, 1+scale)

        # Add constraint terms to the log likelihood
        for prior, vals in sorted(list(self._priors.items())):
            if vals[1] is None:
                continue  # some priors are unconstrained, skip
            scale = 'std::pow(((%s<%.7e) ? %.7e : %.7e), 2)' % (
                prior, vals[0],  # check which side of central value
                vals[0]-vals[1],  # scale by down prior constraint
                vals[0]-vals[2])  # scale by up prior constraint
            code_ll.append(
                '_f += -0.5 * std::pow(%s-%.7e, 2) / %s;' % (
                prior, vals[0], scale))
            code_gll.append(code_ll[-1])
            # Add gradient contribution to ll from each prior
            code_gll.append('_df[%d] += -(%s-%.7e) / %s;' % (
                ipars[prior], prior, vals[0], scale))

        # Add constraint terms for regularizations
        for rexpr, rpars, rgrads in self._regularizations:
            code_ll.append('_f += %s;' % rexpr)
            code_gll.append(code_ll[-1])
            for par, grad in zip(rpars, rgrads):
                code_gll.append('_df[%d] += %s;' % (ipars[par], grad))
    
        # Substitue generated code into the template
        code = code.replace('__NAME__', self.name)
        code = code.replace('__NDIMS__', str(len(pars)))
        code = code.replace('__NCOLS__', str(self._ncols))
        code = code.replace('__PARS__', '\n%s\n' % ('\n'.join(code_pars)))
        code = code.replace('__SPEC__', '\n%s\n' % ('\n'.join(code_spec)))
        code = code.replace('__LL__', '\n%s\n' % ('\n'.join(code_ll)))
        code = code.replace('__GSPEC__', '\n%s\n' % ('\n'.join(code_gspec)))
        code = code.replace('__GLL__', '\n%s\n' % ('\n'.join(code_gll)))

        # Write out the generated code
        code_file = 'comp_parspec_%s.cxx' % self.name
        code_exists = False

        # First, check if identical file exists, in which case it might already
        # be compiled, and no need to re-compile
        try:
            with open(code_file, 'r') as fin:
                old_code = fin.read()
                if old_code == code:
                    code_exists = True
        except IOError:
            pass

        if not code_exists:
            with open(code_file, 'w') as fout:
                fout.write(code)

        # Ask ROOT to compile and link the code
        prev_level = ROOT.gErrorIgnoreLevel
        ROOT.gErrorIgnoreLevel = ROOT.kWarning
        if ROOT.gROOT.LoadMacro(code_file+'+') != 0:
            raise RuntimeError("Unable to compile macro")
        ROOT.gErrorIgnoreLevel = prev_level
        # Grab the spectrum constructor from the compiled code
        constructor = getattr(ROOT, self.name)

        # Build a python spectrum object using that constructor
        spec = ParSpec(self.name, pars, self._ncols, constructor)

        # Tell the spectrum about the central value of constrained parameters
        for par in self._priors:
            spec._centralx[spec._make_ipar(par)] = self._priors[par][0]
            # Use scales if prior is constrained
            if self._priors[par][1] is not None:
                spec._scales[spec._make_ipar(par)] = [
                    self._priors[par][1],
                    self._priors[par][2]]
            # Unconstrained prior
            else:
                spec._scales[spec._make_ipar(par)] = [0, 0]

        return spec

