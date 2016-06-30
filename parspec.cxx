#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>

#include <Math/IFunction.h>

/**
 * Parameterized spectrum: given a set of parmaeters, evaluate the expected
 * spectrum. Given an observed background and priors on parameters, the log
 * likelihood of the parameters can be evaluated.
 *
 * Implements a multi-dimensional gradient function which can be used by
 * ROOT::Math::Minimizer.
 *
 * Note that variables must be prefixed with _ to avoid clashing with varible
 * names introduced by the python genrated code.
 */
class __NAME__ : public ROOT::Math::IGradientFunctionMultiDim {
private:
  static const double _sources[];
  static const unsigned _nrows = __NROWS__;
  static const unsigned _ncols = __NCOLS__;
  static const unsigned _ndims = __NDIMS__;
  bool _negative;
  double _data[_ncols];

public:
  __NAME__() : _negative(true) {
    std::memset(_data, 0, _ncols*sizeof(double));
  }

  void setData(double* data) {
    std::memcpy(_data, data, _ncols*sizeof(double));
  }

  void setNLL() { _negative = true; }
  void setLL() { _negative = false; }

  /** Compute the spectrum for the given parameters */
  void Compute(const double* _x, double* _spec) const {
    std::memset(_spec, 0, _ncols*sizeof(double));
    __PARS__
    // Compute factors for each source
    const double _factors[] = { __FACTORS__ };
    const bool _usestats[] = { __USESTATS__ };
    // Compute spectrum without gradients
    for (unsigned _i = 0; _i < _nrows; _i++)
      for (unsigned _j = 0; _j < _ncols; _j++)
        _spec[_j] += 
            _factors[_i] * 
            (_usestats[_i] ? _stats[_j] : 1) * 
            _sources[_i*_ncols+_j];
    // Don't allow contents to drop below 0
    for (unsigned _j = 0; _j < _ncols; _j++)
      _spec[_j] = std::max(0., _spec[_j]);
  }

  ROOT::Math::IBaseFunctionMultiDim* Clone() const {
    return new __NAME__(*this);
  }

  unsigned int NDim() const { return _ndims; }

  /** Compute the log likelihood and gradient together */
  void FdF(const double* _x, double& _f, double* _df) const {
    // number of entries in each column (the spectrum)
    double _spec[_ncols] = { 0 };
    // gradient of each column content w.r.t. to each parameter
    double _grads[_ncols*_ndims] = { 0 };
    _f = 0;
    std::memset(_df, 0, _ndims*sizeof(double));
    __PARS__
    // Compute factors for each source
    const double _factors[] = { __FACTORS__ };
    // Indicates which rows are affected by statistical uncertainties
    const bool _usestats[] = { __USESTATS__ };
    // The number of parameters affecting each row
    const unsigned _rownpars[] = { __ROWNPARS__ };
    // List of parameter indices for each row
    const unsigned _rowpars[] = { __ROWPARS__ };
    // Gradient of each source factor w.r.t. each paraemeter
    const double _pargrads[] = { __PARGRADS__ };
    unsigned _ipars = 0;
    // Compute spectrum with gradients
    for (unsigned _i = 0; _i < _nrows; _i++) {
      for (unsigned _j = 0; _j < _ncols; _j++) {
        _spec[_j] += 
            _factors[_i] * 
            (_usestats[_i] ? _stats[_j] : 1) * 
            _sources[_i*_ncols+_j];
        if (_usestats[_i])
          _grads[(_istats+_j)*_ncols+_j] += 
              _factors[_i] * 
              _sources[_i*_ncols+_j];
        for (unsigned _k = 0; _k < _rownpars[_i]; _k++)
          _grads[_rowpars[_ipars+_k]*_ncols+_j] += 
              _pargrads[_ipars+_k] *
              (_usestats[_i] ? _stats[_j] : 1) * 
              _sources[_i*_ncols+_j];
      }
      _ipars += _rownpars[_i];
    }
    // Don't allow values below 1, 0 would result in nan when computing
    // Poisson likelihoods
    for (unsigned _j = 0; _j < _ncols; _j++)
      _spec[_j] = std::max(1., _spec[_j]);
    // gradient of ll w.r.t. each parameter
    for (unsigned _j = 0; _j < _ncols; _j++) {
      // log likelihood contributions from each column
      _f += -0.5 * std::pow(_spec[_j]-_data[_j],2) / _spec[_j];
      // graduient of ll due to s, a change in spectrum column _j
      const double dllds = 
          0.5*std::pow(_spec[_j]-_data[_j],2)/std::pow(_spec[_j],2) - 
          (_spec[_j]-_data[_j])/_spec[_j];
      // dll/dp = dll/ds * ds/dp where p is parameter _i
      for (unsigned _i = 0; _i < _ndims; _i++)
        _df[_i] += _grads[_i*_ncols+_j] * dllds;
    }
    // User defined likelihood contributions and gradients
    __GLL__
    if (_negative) {
      _f *= -1;
      for (unsigned i = 0; i < _ndims; i++) _df[i] *= -1;
    }
  }

  /** Wrap FdF for the gradient only to avoid 1-by-1 computation */
  void Gradient(const double* _x, double* _df) const {
    double _f = 0;
    FdF(_x, _f, _df);
  }

private:
  /** No per-dimension implementation, compute full and return one gradient */
  double DoDerivative(const double* _x, unsigned int _icoord) const {
    double _f = 0;
    double _df[_ncols];
    FdF(_x, _f, _df);
    return _df[_icoord];
  }

  double DoEval(const double* _x) const {
    double _f = 0;
    double _spec[_ncols] = { 0 };
    Compute(_x, _spec);
    for (unsigned _j = 0; _j < _ncols; _j++)
      _f += -0.5 * std::pow(_spec[_j]-_data[_j],2) / _spec[_j];
    __LL__
    if (_negative) _f *= -1;
    return _f;
  }
};

const double __NAME__::_sources[] = { 
__SOURCES__
};
