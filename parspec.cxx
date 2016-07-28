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
  // dynamic memory for all the loaded model data
  void* _memory;
  // size of the dynamic memory
  const size_t _nbytes;
  // pointers to relevant parts of the memory
  const double* _prior0;
  const double* _priorDown;
  const double* _priorUp;
  const int* _priorMask;
  const double* _sources;
  const double* _source_stats;
  // dimensions of the model
  static const unsigned _nrows = __NROWS__;
  static const unsigned _ncols = __NCOLS__;
  static const unsigned _ndims = __NDIMS__;
  // the parameter index at which stat. pars. are found
  static const unsigned _istats = __ISTATS__;
  // if no stat values are given, _istats == -1, disable calculations
  const bool _use_stats;
  // return negative log likelihood
  bool _negative;
  // the spectral data as computed from the model
  double _data[_ncols];

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
    _priorMask = (int*)((char*)_memory+i); i += _ndims*sizeof(int)/sizeof(char);
    _sources = (double*)((char*)_memory+i); i += _nrows*_ncols*sizeof(double)/sizeof(char);
    _source_stats = (double*)((char*)_memory+i); i += _nrows*_ncols*sizeof(double)/sizeof(char);
    assert(i == _nbytes/sizeof(char) && "Didn't account for all written data");
    // initialize the on stack spectral data
    std::memset(_data, 0, _ncols*sizeof(double));
  }

  ~__NAME__() {
    if (_memory) std::free(_memory);
  }

  /**
    * @brief set the data values used to computed the likelihood
    *
    * @param data pointer to memory containing data bin values.
    */
  void setData(const double* data) {
    std::memcpy(_data, data, _ncols*sizeof(double));
  }

  void setNLL() { _negative = true; }
  void setLL() { _negative = false; }

  /** 
    * @brief compute the spectrum for the given parameters 
    *
    * @param _x pointer to input paramter values in memory
    * @param _spec pointer to memory where to store bin values
    * @param _stats pointer to memory where to store bin stat. variance
    */
  void Compute(const double* _x, double* _spec, double* _stats=0) const {
    // pointer to the chunk of parameters where stat. pars are found
    const double* _stat_pars = _use_stats ? _x+_istats : 0;
    // prepare the output memory for summing contents into
    std::memset(_spec, 0, _ncols*sizeof(double));
    if (_stats) std::memset(_stats, 0, _ncols*sizeof(double));
    // Compute factors for each source
    const double _factors[] = { __FACTORS__ };
    // Compute spectrum without gradients
    for (unsigned _i = 0; _i < _nrows; _i++) {
      for (unsigned _j = 0; _j < _ncols; _j++) {
        _spec[_j] += _factors[_i] * _sources[_i*_ncols+_j];
        if (_stats) _stats[_j] += 
              std::pow(_factors[_i],2) * _source_stats[_i*_ncols+_j];
      }
    }
    // modify computed spectrum with bin-by-bin fluctuations
    if (_use_stats)
      for (unsigned _j = 0; _j < _ncols; _j++)
        _spec[_j] += _stat_pars[_j];
    // Don't allow contents to drop below 0
    for (unsigned _j = 0; _j < _ncols; _j++)
      _spec[_j] = std::max(0., _spec[_j]);
  }

  /** ROOT wants a clone method... so, well, there it is */
  ROOT::Math::IBaseFunctionMultiDim* Clone() const {
    // copy constructor is a good place to start
    __NAME__* cloned = new __NAME__(*this);
    // need to take ownership of the dynamic memory
    cloned->_memory = std::malloc(_nbytes);
    if (!cloned->_memory) throw std::runtime_error("Unable to allocate memory");
    std::memcpy(cloned->_memory, _memory, _nbytes);
    return cloned;
  }

  unsigned int NDim() const { return _ndims; }

  /**
    * @brief compute the log likelihood and gradients in one pass
    *
    * @param _x pointer to input paramter values in memory
    * @param _f reference to variable in which log likelihood is stored
    * @param _df pointer to memory where gradients are stored for each par.
    */
  void FdF(const double* _x, double& _f, double* _df) const {
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
        _spec[_j] += _stat_pars[_j];
        // take note of house stat par _j affects bin _j
        _spec_grads[(_istats+_j)*_ncols+_j] += 1;
      } 
    }
    // enforce lower bound of 1 to bin values, otherwise problems with Poisson
    for (unsigned _j = 0; _j < _ncols; _j++)
      _spec[_j] = std::max(1., _spec[_j]);

    // compute ll penalty due to difference between bin values and _data
    for (unsigned _j = 0; _j < _ncols; _j++) {
      // ll contributions from each bin
      _f += -0.5 * std::pow(_spec[_j]-_data[_j],2) / _spec[_j];
      // dll/ds: change of likelihood when spectrum bin _j changes
      const double dllds = 
          0.5*std::pow(_spec[_j]-_data[_j],2)/std::pow(_spec[_j],2) - 
          (_spec[_j]-_data[_j])/_spec[_j];
      // dll/dp = dll/ds * ds/dp where p is parameter _i
      for (unsigned _i = 0; _i < _ndims; _i++)
        _df[_i] += _spec_grads[_i*_ncols+_j] * dllds;
    }

    if (_use_stats) {
      // regularize the statistical shifts using the computed bin stat. variances
      for (unsigned _j = 0; _j < _ncols; _j++) {
        if (_stats[_j] <= 0) continue;  // protect against bins with no stat. variance
        const double shift = _x[_istats+_j];  // statistical shift to column _j
        // ll contributions from each bin (penaltiy for stat. shift)
        _f += -0.5 * std::pow(shift,2) / _stats[_j];
        // dll/ds: change in likelihood when stat. variance for bin _j changes
        const double dllds = 
            0.5*std::pow(shift,2)/std::pow(_stats[_j],2);
        // parameters influence the computed stat. variance, so do chain rule
        for (unsigned _i = 0; _i < _ndims; _i++)
          _df[_i] += _stat_grads[_i*_ncols+_j] * dllds;
        // dll/dS: change in likelihood when stat. shift _j changes
        const double dlldS = -shift/_stats[_j];
        _df[_istats+_j] += dlldS;
      }
    }

    // Compute contributions to ll due to prior constraints on parameters
    for (unsigned _i = 0; _i < _ndims; _i++) {
      if (!_priorMask[_i]) continue;
      // ll contribution from prior on parameter _i
      _f += 
          // How far this parameter is from its nominal value
          -0.5 * std::pow(_x[_i]-_prior0[_i],2) / 
          // The width of the prior, conditional whether its below or above 0
          std::pow(((_x[_i]<_prior0[_i]) ? _priorDown[_i] : _priorUp[_i]), 2);
      // dll/dp contribution from the prior penalty
      _df[_i] += 
          -(_x[_i]-_prior0[_i]) / 
          std::pow(((_x[_i]<_prior0[_i]) ? _priorDown[_i] : _priorUp[_i]), 2);
    }

    // User defined likelihood contributions and gradients
    __GLL__

    // invert ll if requested
    if (_negative) {
      _f *= -1;
      for (unsigned i = 0; i < _ndims; i++) _df[i] *= -1;
    }
  }

  /**
    * @brief compute only the gradients
    *
    * @param _x pointer to input paramter values in memory
    * @param _f reference to variable in which log likelihood is stored
    * @param _df pointer to memory where gradients are stored for each par.
    */
  void Gradient(const double* _x, double* _df) const {
    double _f = 0;
    FdF(_x, _f, _df);
  }

private:
  // No per-dimension implementation, compute full and return one gradient
  double DoDerivative(const double* _x, unsigned int _icoord) const {
    double _f = 0;
    double _df[_ncols];
    FdF(_x, _f, _df);
    return _df[_icoord];
  }

  double DoEval(const double* _x) const {
    double _f = 0;
    double _spec[_ncols] = { 0 };
    double _stats[_ncols] = { 0 };
    Compute(_x, _spec, _stats);
    for (unsigned _j = 0; _j < _ncols; _j++)
      _f += -0.5 * std::pow(_spec[_j]-_data[_j],2) / _spec[_j];
    if (_use_stats) {
      for (unsigned _j = 0; _j < _ncols; _j++) {
        if (_stats[_j] <= 0) continue;  // protect against bins with no stat. variance
        const double shift = _x[_istats+_j];
        _f += -0.5 * std::pow(shift,2) / _stats[_j];
      }
    }
    for (unsigned _i = 0; _i < _ndims; _i++) {
      if (!_priorMask[_i]) continue;
      _f += 
          -0.5 * std::pow(_x[_i]-_prior0[_i],2) / 
          std::pow(((_x[_i]<_prior0[_i]) ? _priorDown[_i] : _priorUp[_i]), 2);
    }
    __LL__
    if (_negative) _f *= -1;
    return _f;
  }
};

