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

#define R2P 2.5066282746310002  // sqrt(2*pi)

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
  // the parameter index at which stat. pars. are found
  static const unsigned _istats = __ISTATS__;

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
  const double* _source_stats;  // array of bin variance for each source
  // if no stat values are given, _istats == -1, disable calculations
  const bool _use_stats;
  // return negative log likelihood
  bool _negative;
  // the spectral data as computed from the model
  double _data[_ncols];

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
    _source_stats = (double*)((char*)_memory+i); i += _nrows*_ncols*sizeof(double)/sizeof(char);
    assert(i == _nbytes/sizeof(char) && "Didn't account for all written data");
    // initialize the on stack spectral data
    std::memset(_data, 0, _ncols*sizeof(double));
  }

  // note: keep in class to keep in namespace
  static double _lfactorial(unsigned long long _i) {
    const double _tab[] = { 
        0.0, 0.0, 0.69314718055994529, 1.791759469228055, 3.1780538303479458, 
        4.7874917427820458, 6.5792512120101012, 8.5251613610654147, 
        10.604602902745251, 12.801827480081469 };
    if (_i >= 10)
      return _i*std::log(_i) - _i + .5*std::log(2*M_PI*_i);
    else
      return _tab[_i];
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
        _nrows*_ncols*sizeof(double) +  // sources
        _nrows*_ncols*sizeof(double) // source_stats
      ),
      _use_stats(_istats != (unsigned)-1),
      _negative(true) {
    // set memory from hard coded path to binary file
    _setMemory();
  }

  /** Copy constructor */
  __NAME__(const __NAME__& other) : 
      // copy state
      _nbytes(other._nbytes), 
      _use_stats(other._use_stats), 
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
    * @param data pointer to memory containing data bin values.
    */
  virtual void setData(const double* data) {
    std::memcpy(_data, data, _ncols*sizeof(double));
  }

  virtual void setNLL() { _negative = true; }
  virtual void setLL() { _negative = false; }

  /** 
    * @brief compute the spectrum for the given parameters 
    *
    * @param _x pointer to input paramter values in memory
    * @param _spec pointer to memory where to store bin values
    * @param _stats pointer to memory where to store bin stat. variance
    */
  virtual void Compute(const double* _x, double* _spec, double* _stats) const {
    // pointer to the chunk of parameters where stat. pars are found
    const double* _stat_pars = _use_stats ? _x+_istats : 0;
    // prepare the output memory for summing contents into
    std::memset(_spec, 0, _ncols*sizeof(double));
    std::memset(_stats, 0, _ncols*sizeof(double));
    // Compute factors for each source
    const double _factors[] = { __FACTORS__ };
    // Compute spectrum without gradients
    for (unsigned _i = 0; _i < _nrows; _i++) {
      for (unsigned _j = 0; _j < _ncols; _j++) {
        _spec[_j] += _factors[_i] * _sources[_i*_ncols+_j];
        _stats[_j] += std::pow(_factors[_i],2) * _source_stats[_i*_ncols+_j];
      }
    }
    // modify computed spectrum with bin-by-bin fluctuations
    if (_use_stats)
      for (unsigned _j = 0; _j < _ncols; _j++)
        _spec[_j] += _stat_pars[_j] * std::pow(_stats[_j], .5);
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
    const double* _stat_pars = _use_stats ? _x+_istats : 0;
    // value of each bin (the spectrum)
    double _spec[_ncols] = { 0 };
    // statistical variance of the value of each bin
    double _stats[_ncols] = { 0 };
    // gradient of each bin value w.r.t. to each parameter
    double _spec_grads[_ncols*_ndims] = { 0 };
    // gradient of each bin variance w.r.t. to each parameter
    double _stat_grads[_ncols*_ndims] = { 0 };
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
        _stats[_j] += std::pow(_factors[_i],2) * _source_stats[_i*_ncols+_j];
        // iterate through the parameters affecting this contribution
        for (unsigned _k = 0; _k < _rownpars[_i]; _k++) {
          const unsigned _ipar = _rowpars[_ipars+_k];
          // _j col w.r.t. to _ipar
          _spec_grads[_ipar*_ncols+_j] += 
              _pargrads[_ipars+_k] * _sources[_i*_ncols+_j];
          _stat_grads[_ipar*_ncols+_j] += 
              _pargrads[_ipars+_k] * 2 * _factors[_i] * _source_stats[_i*_ncols+_j];
        }
      }
      // keep track of position in the _rowpars array (each row moves through
      // but by a differing amount---i.e. rownpars)
      _ipars += _rownpars[_i];
    }
    // Add per column corrections to the spectrum
    if (_use_stats) {
      for (unsigned _j = 0; _j < _ncols; _j++) {
        _spec[_j] += _stat_pars[_j] * std::pow(_stats[_j], .5);
        // take note of how stat par _j affects bin _j
        _spec_grads[(_istats+_j)*_ncols+_j] += std::pow(_stats[_j], .5);
        // further consider how each parameter affects bin _j through _stat[_j]
        for (unsigned _ipar = 0; _ipar < _ndims; _ipar++)
          _spec_grads[_ipar*_ncols+_j] += 
              _stat_pars[_j] * 
              .5*std::pow(_stats[_j], -.5) * 
              _stat_grads[_ipar*_ncols+_j];
      } 
    }

    // compute ll penalty due to difference between bin values and _data
    for (unsigned _j = 0; _j < _ncols; _j++) {
      const double _k = _data[_j];  // number observed
      const double _v = _spec[_j];  // number expected (variance)
      if (_v <= 0 && _k <= 0) continue;  // gracefully handle empty bins
      // log poisson for k given v. The term in brackets in the k! which isn't
      // needed since its constant w.r.t. to pars, but helps keep the ll to
      // some resonnable value (otherwise scale as n*k*ln(v))
      _f += (_v > 0) ?
          _k*std::log(_v) - _v - _lfactorial(_k+1) :
          // zero or smaller bin values are not allowed
          -std::numeric_limits<double>::infinity();
      // derivative of poisson w.r.t. to expected bin value
      const double dlldv = (_v > 0) ?
          _k/_v - 1 :
          std::numeric_limits<double>::infinity();
      // keep track of derivative of ll w.r.t. to each par. impacting bin val
      for (unsigned _i = 0; _i < _ndims; _i++)
        _df[_i] += _spec_grads[_i*_ncols+_j] * dlldv;
    }

    if (_use_stats) {
      // regularize the statistical shifts using the computed bin stat. variances
      for (unsigned _j = 0; _j < _ncols; _j++) {
        const double _s = _stat_pars[_j];  // shift from nominal
        // stat pars are in units of sigma
        _f += -0.5*std::pow(_s,2);
        // change of ll w.r.t. to the actual bin shift
        const double dllds = -_s;
        _df[_istats+_j] += dllds;
      }
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
    const double* _stat_pars = _use_stats ? _x+_istats : 0;
    double _f = 0;
    double _spec[_ncols] = { 0 };
    double _stats[_ncols] = { 0 };
    Compute(_x, _spec, _stats);
    for (unsigned _j = 0; _j < _ncols; _j++) {
      const double _k = _data[_j];
      const double _v = _spec[_j];
      if (_v <= 0 && _k <= 0) continue;
      _f += (_v > 0) ?
          _k*std::log(_v) - _v - _lfactorial(_k+1) :
          -std::numeric_limits<double>::infinity();
    }
    if (_use_stats) {
      for (unsigned _j = 0; _j < _ncols; _j++) {
        const double _s = _stat_pars[_j];
        _f += -0.5*std::pow(_s,2);
      }
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

