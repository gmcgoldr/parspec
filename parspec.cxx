#include <iostream>
#include <cstring>
#include <algorithm>

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
    // Assign parameter values to nammed variables
    __PARS__
    // Compute spectrum without gradients
    __SPEC__
  }

  ROOT::Math::IBaseFunctionMultiDim* Clone() const {
    return new __NAME__(*this);
  }

  unsigned int NDim() const { return _ndims; }

  /** Compute the log likelihood and gradient together */
  void FdF(const double* _x, double& _f, double* _df) const {
    double _spec[_ncols] = { 0 };
    double _grads[_ncols*_ndims] = { 0 };
    _f = 0;
    std::memset(_df, 0, _ndims*sizeof(double));
    // Assign parameter values to nammed variables
    __PARS__
    // Compute the spectrum and gradient of each bin w.r.t. parameters
    __GSPEC__
    // Compute the log likelihood and gradient of ll w.r.t. parameters
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
    // Assign parameter values to nammed variables
    __PARS__
    // Compute log likelihood without gradients
    __LL__
    if (_negative) _f *= -1;
    return _f;
  }
};
