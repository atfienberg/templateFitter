/*Template fitter class built on top of the Eigen3 linear algebra library

  Aaron Fienberg
  fienberg@uw.edu
*/

#pragma once

#include <vector>
#include <numeric>
#include <cassert>
#include <type_traits>

#include <Eigen/Dense>

#include "TSpline.h"

class TemplateFitter {
 public:
  // output data type
  typedef struct {
    std::vector<double> times;
    std::vector<double> scales;
    double pedestal;
    double chi2;
    bool converged;
  } Output;

  // construct with matrix dimensions (default to 0)
  TemplateFitter(int nPulses = 0, int nSamples = 0);

  // construct w/ template spline, its limits of validity, and matrix dimensions
  TemplateFitter(const TSpline3* tSpline, double tMin, double tMax,
                 int nPulses = 0, int nSamples = 0);

  // give template spline and its limits of validity
  // optionally give number of pts at which to evaluate it
  void setTemplate(const TSpline3* tSpline, double tMin, double tMax,
                   int tPts = 1000);

  // give template as a vector with its time limits (time of first and last
  // sample)
  void setTemplate(const std::vector<double>& temp, double tMin, double tMax);

  // get covariance_ij. don't call this before doing a fit
  // order of parameters is {t1 ... tn, s1 ... sn, pedestal}
  double getCovariance(int i, int j);

  std::vector<double> getResiduals() const {
    return std::vector<double>(deltas_.data(), deltas_.data() + deltas_.size());
  }

  // max number of iterations before giving up
  std::size_t getMaxIterations() const { return maxIterations_; }
  void setMaxIterations(std::size_t maxIters) { maxIterations_ = maxIters; }

  // target accuracy. When max step size is less than this,
  // the numerical minimization stops
  double getAccuracy() const { return accuracy_; }
  void setAccuracy(double newAccuracy) { accuracy_ = newAccuracy; }

  // get number of pts in templates
  std::size_t getNTemplatePoints() const { return template_.size(); }

  // fit() functions
  // n pulses is determined by timeGuesses.size()
  // these assume a contiguous fit region

  // version for single pulses that doesn't require a vector for time guess
  // errortype can be a number or a vector (uncertainty on each sample)
  template <typename sampleType, typename errorType = double>
  Output fit(const std::vector<sampleType>& trace, double timeGuess,
             errorType error = 1.0) {
    return fit(trace, std::vector<double>{timeGuess}, error);
  }

  // this call uses same uncertainty on all data points, defaults to 1.0

  template <typename sampleType, typename noiseType = double>
  Output fit(const std::vector<sampleType>& trace,
             const std::vector<double>& timeGuesses,
             noiseType noiseLevel = 1.0) {
    static_assert(std::is_arithmetic<sampleType>::value,
                  "trace must be vector of numbers!");
    static_assert(std::is_arithmetic<noiseType>::value,
                  "noise level must be a number!");
    assert(noiseLevel != 0);

    bool resized = false;

    if ((trace.size() != static_cast<std::size_t>(pVect_.rows())) ||
        (timeGuesses.size() != static_cast<std::size_t>(D_.rows()))) {
      std::size_t oldSize = sampleTimes_.size();
      resizeMatrices(trace.size(), timeGuesses.size());
      if ((trace.size() > oldSize) || (wasDiscontiguous_)) {
        std::iota(sampleTimes_.begin(), sampleTimes_.end(), 0.0);
      }
      resized = true;
    } else if (wasDiscontiguous_) {
      std::iota(sampleTimes_.begin(), sampleTimes_.end(), 0.0);
    }

    if ((resized) || (!isFlatNoise_) || (lastNoiseLevel_ != noiseLevel)) {
      T_.bottomRows(1).fill(1.0 / noiseLevel);
      lastNoiseLevel_ = noiseLevel;
      isFlatNoise_ = true;
    }

    for (std::size_t i = 0; i < trace.size(); ++i) {
      pVect_(i) = trace[i] * T_.bottomRows(1)(0, i);
    }

    wasDiscontiguous_ = false;
    return doFit(timeGuesses);
  }

  // this call allows for different errors on each data point.
  // trace.size() must equal errors.size()

  // arbitrary error on each sample
  template <typename sampleType, typename noiseType>
  Output fit(const std::vector<sampleType>& trace,
             const std::vector<double>& timeGuesses,
             const std::vector<noiseType>& errors) {
    static_assert(std::is_arithmetic<sampleType>::value,
                  "trace must be vector of numbers!");
    static_assert(std::is_arithmetic<noiseType>::value,
                  "errors must be vector of numbers!");
    assert(errors.size() == trace.size());

    if ((trace.size() != static_cast<std::size_t>(pVect_.rows())) ||
        (timeGuesses.size() != static_cast<std::size_t>(D_.rows()))) {
      std::size_t oldSize = sampleTimes_.size();
      resizeMatrices(trace.size(), timeGuesses.size());
      if ((trace.size() > oldSize) || (wasDiscontiguous_)) {
        std::iota(sampleTimes_.begin(), sampleTimes_.end(), 0.0);
      }
    } else if (wasDiscontiguous_) {
      std::iota(sampleTimes_.begin(), sampleTimes_.end(), 0.0);
    }

    for (std::size_t i = 0; i < trace.size(); ++i) {
      T_.bottomRows(1)(0, i) = 1.0 / errors[i];
      pVect_(i) = trace[i] * T_.bottomRows(1)(0, i);
    }
    isFlatNoise_ = false;

    wasDiscontiguous_ = false;
    return doFit(timeGuesses);
  }

  // discontiguousFit() functions for fitting discontiguous regions
  // these are mainly useful for clipped pulses
  // you must pass in vector of sample times along with vector of sample values

  // single pulse version
  template <typename sampleType, typename timeType, typename errorType = double>
  Output discontiguousFit(const std::vector<sampleType>& trace,
                          const std::vector<timeType>& sampleTimes,
                          double timeGuess, errorType error = 1.0) {
    return discontiguousFit(trace, sampleTimes, std::vector<double>{timeGuess},
                            error);
  }

  // discontiguous fit with flat errors
  template <typename sampleType, typename timeType, typename noiseType = double>
  Output discontiguousFit(const std::vector<sampleType>& trace,
                          const std::vector<timeType>& sampleTimes,
                          const std::vector<double>& timeGuesses,
                          noiseType noiseLevel = 1.0) {
    static_assert(std::is_arithmetic<sampleType>::value,
                  "trace must be vector of numbers!");
    static_assert(std::is_arithmetic<timeType>::value,
                  "sampleTimes must be vector of numbers!");
    static_assert(std::is_arithmetic<noiseType>::value,
                  "noise level must be a number!");
    assert(noiseLevel != 0);
    assert(trace.size() == sampleTimes.size());

    bool resized = false;

    if ((trace.size() != static_cast<std::size_t>(pVect_.rows())) ||
        (timeGuesses.size() != static_cast<std::size_t>(D_.rows()))) {
      resizeMatrices(trace.size(), timeGuesses.size());
      resized = true;
    }

    std::copy(sampleTimes.begin(), sampleTimes.end(), sampleTimes_.begin());

    if ((resized) || (!isFlatNoise_) || (lastNoiseLevel_ != noiseLevel)) {
      T_.bottomRows(1).fill(1.0 / noiseLevel);
      lastNoiseLevel_ = noiseLevel;
      isFlatNoise_ = true;
    }

    for (std::size_t i = 0; i < trace.size(); ++i) {
      pVect_(i) = trace[i] * T_.bottomRows(1)(0, i);
    }

    wasDiscontiguous_ = true;
    return doFit(timeGuesses);
  }

  // discontiguous fit with arbitrary errors
  template <typename sampleType, typename timeType, typename noiseType>
  Output discontiguousFit(const std::vector<sampleType>& trace,
                          const std::vector<timeType>& sampleTimes,
                          const std::vector<double>& timeGuesses,
                          const std::vector<noiseType>& errors) {
    static_assert(std::is_arithmetic<sampleType>::value,
                  "trace must be vector of numbers!");
    static_assert(std::is_arithmetic<timeType>::value,
                  "sampleTimes must be vector of numbers!");
    static_assert(std::is_arithmetic<noiseType>::value,
                  "errors must be vector of numbers!");
    assert(trace.size() == sampleTimes.size());
    assert(errors.size() == trace.size());

    if ((trace.size() != static_cast<std::size_t>(pVect_.rows())) ||
        (timeGuesses.size() != static_cast<std::size_t>(D_.rows()))) {
      resizeMatrices(trace.size(), timeGuesses.size());
    }

    std::copy(sampleTimes.begin(), sampleTimes.end(), sampleTimes_.begin());

    for (std::size_t i = 0; i < trace.size(); ++i) {
      T_.bottomRows(1)(0, i) = 1.0 / errors[i];
      pVect_(i) = trace[i] * T_.bottomRows(1)(0, i);
    }
    isFlatNoise_ = false;

    wasDiscontiguous_ = true;
    return doFit(timeGuesses);
  }

 private:
  Output doFit(const std::vector<double>& timeGuesses);

  void evalTemplates(const std::vector<double>& tGuesses);

  bool hasConverged();

  void calculateCovarianceMatrix();

  std::vector<double> buildDTemplate(const std::vector<double>& temp);

  void resizeMatrices(int nSamples, int nPulses);

  // how small largest time step has to go before stopping minimization
  double accuracy_;
  // max number of iterations
  std::size_t maxIterations_;
  // whether covariance matrix is ready
  bool covReady_;

  double lastNoiseLevel_;
  bool isFlatNoise_;
  bool wasDiscontiguous_;

  // template stuff
  std::vector<double> template_;
  std::vector<double> dTemplate_;
  std::vector<double> d2Template_;
  double tMin_, tMax_;

  // vector of time values corresponding to each sample
  std::vector<double> sampleTimes_;

  // eigen matrices, all kept around to avoid repeated allocation

  // vector of pulse heights weighted by inverse noise at each sample
  Eigen::VectorXd pVect_;
  // T matrix (see document)
  Eigen::MatrixXd T_;
  // linear parameters will be stored in here after each fit
  Eigen::VectorXd b_;
  // vector of pulse minus fit function over noise
  Eigen::VectorXd deltas_;
  // template derivatives matrix
  Eigen::MatrixXd D_;
  // template second derivatives matrix
  Eigen::MatrixXd D2_;
  // hessian
  Eigen::MatrixXd Hess_;
  // covariance matrix
  Eigen::MatrixXd Cov_;
  // proposed time steps to minimum
  Eigen::VectorXd timeSteps_;
};
