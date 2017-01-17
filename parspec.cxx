#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cassert>

#include <Math/IFunction.h>

#define R2P 2.5066282746310002
#define RP2 1.2533141373155001

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
  enum PriorTypes { __PRIORTYPES__ };

  // dimensions of the model
  static const unsigned _nrows = __NROWS__;
  static const unsigned _ncols = __NCOLS__;
  static const unsigned _ndims = __NDIMS__;

  // dynamic memory for all the loaded model data
  void* _memory;
  // size of the dynamic memory
  const size_t _nbytes;
  // pointers to relevant parts of the memory
  const double* _prior0;  // prior central values
  const double* _priorDown;  // prior down scale (variance)
  const double* _priorUp;  // prior up scale
  const int* _priorType;  // prior type (e.g. normal)
  const double* _sources;  // array of bin values for each source
  // return negative log likelihood
  bool _negative;
  // the spectral data as computed from the model
  double _data[_ncols];
  // underlying variance for each bin expectation (MC stats)
  double _bin_vars[_ncols];

  void _setMemory() {
    // open a file at a static location with binary data
    FILE* fin = std::fopen("__DATAPATH__", "rb");
    if (!fin) throw std::runtime_error("Binary data not found at __DATAPATH__");
    // allocate the memory for all the dynamic data  
    _memory = std::malloc(_nbytes);
    if (!_memory) throw std::runtime_error("Unable to allocate memory");
    // read all the dynamic data into a single block of memory
    const size_t nread = std::fread(_memory, 1, _nbytes, fin);
    std::fclose(fin);
    if (nread < _nbytes) throw std::runtime_error("Unable to read binary data");
    // set pointers to the various parts of that memory
    size_t i = 0;
    _prior0 = (double*)((char*)_memory+i); i += _ndims*sizeof(double)/sizeof(char);
    _priorDown = (double*)((char*)_memory+i); i += _ndims*sizeof(double)/sizeof(char);
    _priorUp = (double*)((char*)_memory+i); i += _ndims*sizeof(double)/sizeof(char);
    _priorType = (int*)((char*)_memory+i); i += _ndims*sizeof(int)/sizeof(char);
    _sources = (double*)((char*)_memory+i); i += _nrows*_ncols*sizeof(double)/sizeof(char);
    assert(i == _nbytes/sizeof(char) && "Didn't account for all written data");
    // initialize the on stack spectral data
    std::memset(_data, 0, _ncols*sizeof(double));
    std::memset(_bin_vars, 0, _ncols*sizeof(double));
  }

  // disable assignment operator
  __NAME__& operator= (const __NAME__&);

public:
  __NAME__() : 
      _nbytes(
        _ndims*sizeof(double) +  // prior0 (one per parameter)
        _ndims*sizeof(double) +  // priorDown
        _ndims*sizeof(double) +  // priorUp
        _ndims*sizeof(int) +    // priorMask
        _nrows*_ncols*sizeof(double)  // sources
      ),
      _negative(true) {
    // set memory from hard coded path to binary file
    _setMemory();
  }

  /** Copy constructor */
  __NAME__(const __NAME__& other) : 
      // copy state
      _nbytes(other._nbytes), 
      _negative(other._negative) {
    // assign own memory
    _setMemory();
  }

  virtual ~__NAME__() {
    if (_memory) std::free(_memory);
  }

  /** ROOT wants a clone method... so, well, there it is */
  virtual ROOT::Math::IBaseFunctionMultiDim* Clone() const {
    return new __NAME__(*this);
  }

  /**
    * @brief set the data values used to computed the likelihood
    *
    * @param pointer to memory containing bin values.
    */
  virtual void setData(const double* data) {
    std::memcpy(_data, data, _ncols*sizeof(double));
  }

  /**
    * @brief set the bin variances (MC stats.)
    *
    * @param pointer to memory containing bin values.
    */
  virtual void setBinVars(const double* vars) {
    std::memcpy(_bin_vars, vars, _ncols*sizeof(double));
  }

  virtual void setNLL(bool nll) { _negative = nll; }
  virtual bool getNLL() const { return _negative; }

  /** 
    * @brief compute the spectrum for the given parameters 
    *
    * @param _x pointer to input paramter values in memory
    * @param _spec pointer to memory where to store bin values
    */
  virtual void Compute(const double* _x, double* _spec) const {
    // pointer to the chunk of parameters where stat. pars are found
    // prepare the output memory for summing contents into
    std::memset(_spec, 0, _ncols*sizeof(double));
    // Compute factors for each source
    const double _factors[] = { __FACTORS__ };
    // Compute spectrum without gradients
    for (unsigned _i = 0; _i < _nrows; _i++) {
      for (unsigned _j = 0; _j < _ncols; _j++) {
        _spec[_j] += _factors[_i] * _sources[_i*_ncols+_j];
      }
    }
  }

  virtual unsigned int NDim() const { return _ndims; }

  /**
    * @brief compute the log likelihood and gradients in one pass
    *
    * @param _x pointer to input paramter values in memory
    * @param _f reference to variable in which log likelihood is stored
    * @param _df pointer to memory where gradients are stored for each par.
    */
  virtual void FdF(const double* _x, double& _f, double* _df) const {
    // value of each bin (the spectrum)
    double _spec[_ncols] = { 0 };
    // gradient of each bin value w.r.t. to each parameter
    double _spec_grads[_ncols*_ndims] = { 0 };
    _f = 0;
    std::memset(_df, 0, _ndims*sizeof(double));
    // Compute factors for each source
    const double _factors[] = { __FACTORS__ };
    // The number of parameters affecting each row (one entry per row)
    const unsigned _rownpars[] = { __ROWNPARS__ };
    // List of parameter indices for each row (continuous list)
    const unsigned _rowpars[] = { __ROWPARS__ };
    // Gradient of each source factor w.r.t. each paraemeter
    const double _pargrads[] = { __PARGRADS__ };
    unsigned _ipars = 0;  // current position in _rowpars

    // sum contributions to each bin's value and variance from all sources,
    // and the gradient of those w.r.t. to each parameter
    for (unsigned _i = 0; _i < _nrows; _i++) {
      for (unsigned _j = 0; _j < _ncols; _j++) {
        _spec[_j] += _factors[_i] * _sources[_i*_ncols+_j];
        // iterate through the parameters affecting this contribution
        for (unsigned _k = 0; _k < _rownpars[_i]; _k++) {
          const unsigned _ipar = _rowpars[_ipars+_k];
          // _j col w.r.t. to _ipar
          _spec_grads[_ipar*_ncols+_j] += 
              _pargrads[_ipars+_k] * _sources[_i*_ncols+_j];
        }
      }
      // keep track of position in the _rowpars array (each row moves through
      // but by a differing amount---i.e. rownpars)
      _ipars += _rownpars[_i];
    }

    // compute ll penalty due to difference between bin values and _data
    for (unsigned _j = 0; _j < _ncols; _j++) {
      const double _k = _data[_j];  // number observed
      const double _v = _spec[_j];  // number expected (variance)
      if (_v <= 0 && _k <= 0) continue;  // gracefully handle empty bins
      // variance of this bin (sum in quad. of MC stats and data stats)
      const double _var = _bin_vars[_j] + ((_k<1) ? 1 : _k);
      _f += (_v > 0) ?
          -0.5 * std::pow(_k-_v, 2) / _var :
          -std::numeric_limits<double>::infinity();
      // derivative of poisson w.r.t. to expected bin value
      const double dlldv = (_v > 0) ?
          (_k-_v) / _var :
          std::numeric_limits<double>::infinity();
      // keep track of derivative of ll w.r.t. to each par. impacting bin val
      for (unsigned _i = 0; _i < _ndims; _i++)
        _df[_i] += _spec_grads[_i*_ncols+_j] * dlldv;
    }

    // Compute contributions to ll due to prior constraints on parameters
    for (unsigned _i = 0; _i < _ndims; _i++) {
      switch (_priorType[_i]) {
      case _PNONE:
        // unregularized, doesn't contribute to ll
        break;
      case _PNORMAL:
        // ll contribution from prior on parameter _i
        // note: priorUp/Down are squared (variance, not sigma)
        _f += 
            // How far this parameter is from its nominal value
            -0.5 * std::pow(_x[_i]-_prior0[_i],2) / 
            // The width of the prior, conditional whether its below or above 0
            ((_x[_i]<_prior0[_i]) ? _priorDown[_i] : _priorUp[_i]);
        // dll/dp contribution from the prior penalty
        _df[_i] += 
            -(_x[_i]-_prior0[_i]) / 
            ((_x[_i]<_prior0[_i]) ? _priorDown[_i] : _priorUp[_i]);
        break;
      case _PLOGNORMAL:
        const double _lx = std::log(_x[_i]);
        // note: _prior0, Down, Up are already converted to log
        _f += 
            -0.5 * std::pow(_lx-_prior0[_i],2) / 
            ((_lx<_prior0[_i]) ? _priorDown[_i] : _priorUp[_i]);
        _df[_i] += 
            -(_lx-_prior0[_i]) / 
            (_x[_i] * ((_lx<_prior0[_i]) ? _priorDown[_i] : _priorUp[_i]));
        break;
      }
    }

    // User defined likelihood contributions and gradients
    __GLL__

    // invert ll if requested
    if (_negative) {
      _f *= -1;
      for (unsigned _i = 0; _i < _ndims; _i++) _df[_i] *= -1;
    }
  }

  /**
    * @brief compute only the gradients
    *
    * @param _x pointer to input paramter values in memory
    * @param _f reference to variable in which log likelihood is stored
    * @param _df pointer to memory where gradients are stored for each par.
    */
  virtual void Gradient(const double* _x, double* _df) const {
    double _f = 0;
    FdF(_x, _f, _df);
  }

private:
  // No per-dimension implementation, compute full and return one gradient
  virtual double DoDerivative(const double* _x, unsigned int _icoord) const {
    double _f = 0;
    double _df[_ncols];
    FdF(_x, _f, _df);
    return _df[_icoord];
  }

  virtual double DoEval(const double* _x) const {
    double _f = 0;
    double _spec[_ncols] = { 0 };
    Compute(_x, _spec);
    for (unsigned _j = 0; _j < _ncols; _j++) {
      const double _k = _data[_j];
      const double _v = _spec[_j];
      if (_v <= 0 && _k <= 0) continue;
      const double _var = _bin_vars[_j] + ((_k<1) ? 1 : _k);
      _f += (_v > 0) ?
          -0.5 * std::pow(_k-_v, 2) / _var :
          -std::numeric_limits<double>::infinity();
    }
    for (unsigned _i = 0; _i < _ndims; _i++) {
      switch (_priorType[_i]) {
      case _PNONE:
        // unregularized, doesn't contribute to ll
        break;
      case _PNORMAL:
        _f += 
            -0.5 * std::pow(_x[_i]-_prior0[_i],2) / 
            ((_x[_i]<_prior0[_i]) ? _priorDown[_i] : _priorUp[_i]);
        break;
      case _PLOGNORMAL:
        const double _lx = std::log(_x[_i]);
        _f += 
            -0.5 * std::pow(_lx-_prior0[_i],2) / 
            ((_lx<_prior0[_i]) ? _priorDown[_i] : _priorUp[_i]);
        break;
      }
    }
    __LL__
    if (_negative) _f *= -1;
    return _f;
  }
};

