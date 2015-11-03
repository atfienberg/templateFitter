/*Template fitter class built on top of the Eigen3 linear algebra library
  Link to document as of now unwritten, but I promise I'll write it
 
  Aaron Fienberg
  fienberg@uw.edu
*/

#pragma once

#include <vector>
#include <memory>
#include <numeric>
#include <cassert>
#include <type_traits>

#include <Eigen/Dense>

#include "TSpline.h"

class TemplateFitter{
public:
  
  //output data type
  typedef struct {
    std::vector<double> times; 
    std::vector<double> scales; 
    double pedestal;
    double chi2;
    bool converged;
  } Output;
    
  //construct with matrix dimensions (default to 0)
  TemplateFitter(int nPulses = 0, int nSamples = 0);
  
  //construct w/ template spline, its limits of validity, and matrix dimensions
  TemplateFitter(const TSpline3* tSpline, double tMin, double tMax, 
		 int nPulses = 0, int nSamples = 0);
  
  //give template spline and its limits of validity
  void setTemplate(const TSpline3* tSpline, double tMin, double tMax);
  
  //get covariance_ij. don't call this before doing a fit
  //order of parameters is {t1 ... tn, s1 ... sn, pedestal}
  double getCovariance(int i, int j);
  
  unsigned int getMaxIterations() const { return maxIterations_; }
  void setMaxIterations(unsigned int maxIters){ maxIterations_ = maxIters; }
  
  double getAccuracy() const { return accuracy_; }
  void setAccuracy(double newAccuracy){ accuracy_ = newAccuracy;}


  //fit() functions
  //n pulses is determined by timeGuesses.size() 
  //this call uses same uncertainty on all data points, defaults to one
  //for now returns parameters in a vector {t1 .. tn, s1 ... sn, pedestal, chi2} 
  //of length 2*nPulses +1
  
  //version for single pulses that doesn't require a vector for time guess
  template<typename T>
  Output fit(const std::vector<T>& trace, 
	     double timeGuess,
	     double noiseLevel = 1.0){
    return fit(trace, std::vector<double> {timeGuess}, noiseLevel);
  }
  
  template<typename T>
  Output fit(const std::vector<T>& trace, 
	     const std::vector<double>& timeGuesses,
	     double noiseLevel = 1.0){
    static_assert(std::is_arithmetic<T>::value, 
		  "trace must be vector of numbers!");
    assert(noiseLevel != 0);

    bool resized = false;

    if((trace.size() != pVect_.rows()) || (timeGuesses.size() != D_.rows())){
      int oldSize = sampleTimes_.size();
      resizeMatrices(trace.size(), timeGuesses.size());
      if((trace.size() > oldSize) || (wasDiscontiguous_)){
	std::iota(sampleTimes_.begin(),sampleTimes_.end(), 0.0);
      }
      resized = true;
    }   

    if((resized) || (!isFlatNoise_) || (lastNoiseLevel_ != noiseLevel)){
      T_.bottomRows(1).fill(1.0/noiseLevel);
      lastNoiseLevel_ = noiseLevel;
      isFlatNoise_ = true;    
    }

    for(int i = 0; i < trace.size(); ++i){
      pVect_(i) = trace[i] * T_.bottomRows(1)(0,i);
    }    
    
    wasDiscontiguous_ = false;
    return doFit(timeGuesses);       
  }  

  //these function allows for different errors on each data point. 
  //trace.size() must equal errors.size()
  
  //arb error version for single pulses 
  template<typename T>
  Output fit(const std::vector<T>& trace, 
	     double timeGuess,
	     const std::vector<double>& errors){
    return fit(trace, std::vector<double> {timeGuess}, errors);
  }

  //arb error for multiple pulses
  template<typename T>
  Output fit(const std::vector<T>& trace, 
	     const std::vector<double>& timeGuesses,
	     const std::vector<double>& errors){
    static_assert(std::is_arithmetic<T>::value, 
		  "trace must be vector of numbers!");
    assert(errors.size() == trace.size());
    
    if((trace.size() != pVect_.rows()) || (timeGuesses.size() != D_.rows())){
      int oldSize = sampleTimes_.size();
      resizeMatrices(trace.size(), timeGuesses.size());
      if((trace.size() > oldSize) || (wasDiscontiguous_)){
	std::iota(sampleTimes_.begin(),sampleTimes_.end(), 0.0);
      }
    }
        
    for(unsigned int i = 0; i < trace.size(); ++i){
      T_.bottomRows(1)(0,i) = 1.0 / errors[i];
      pVect_(i) = trace[i] * T_.bottomRows(1)(0,i);
    }
    isFlatNoise_ = false;  
    
    wasDiscontiguous_ = false;
    return doFit(timeGuesses);
  }

  //calls for fitting discontinous regions
  //these are mainly useful for clipped pulses
  
  //single pulse versions
  template<typename T1, typename T2, typename errorType>
  Output discontiguousFit(const std::vector<T1>& trace, 
			  const std::vector<T2>& sampleTimes, 
			  double timeGuess,
			  errorType error
			  ){
    return discontiguousFit(trace, sampleTimes, std::vector<double> {timeGuess}, error);    
  }

  template <typename T1, typename T2>
  Output discontiguousFit(const std::vector<T1>& trace, 
			  const std::vector<T2>& sampleTimes, 
			  const std::vector<double>& timeGuesses,
			  double noiseLevel = 1.0){ 
    static_assert(std::is_arithmetic<T1>::value, 
		  "trace must be vector of numbers!");
    static_assert(std::is_arithmetic<T2>::value, 
		  "sampleTimes must be vector of numbers!");
    assert(noiseLevel != 0);
    assert(trace.size() == sampleTimes.size());
    
    bool resized = false;

    if((trace.size() != pVect_.rows()) || (timeGuesses.size() != D_.rows())){
      resizeMatrices(trace.size(), timeGuesses.size());
      resized = true;
    }  

    std::copy(sampleTimes.begin(), sampleTimes.end(), sampleTimes_.begin());

    if((resized) || (!isFlatNoise_) || (lastNoiseLevel_ != noiseLevel)){
      T_.bottomRows(1).fill(1.0/noiseLevel);
      lastNoiseLevel_ = noiseLevel;
      isFlatNoise_ = true;    
    }

    for(int i = 0; i < trace.size(); ++i){
      pVect_(i) = trace[i] * T_.bottomRows(1)(0,i);
    }    
    
    wasDiscontiguous_ = true;
    return doFit(timeGuesses);       
  }   

  //discontiguous fit for arbitrary errors    
  template<typename T1, typename T2>
  Output discontiguousFit(const std::vector<T1>& trace, 
			  const std::vector<T2>& sampleTimes, 
			  const std::vector<double>& timeGuesses,
			  const std::vector<double>& errors){ 

    static_assert(std::is_arithmetic<T1>::value, 
		  "trace must be vector of numbers!");
    static_assert(std::is_arithmetic<T2>::value, 
		  "sampleTimes must be vector of numbers!");
    assert(trace.size() == sampleTimes.size());
    assert(errors.size() == trace.size());   

    if((trace.size() != pVect_.rows()) || (timeGuesses.size() != D_.rows())){
      resizeMatrices(trace.size(), timeGuesses.size());
    }  

    std::copy(sampleTimes.begin(), sampleTimes.end(), sampleTimes_.begin());
    
    for(unsigned int i = 0; i < trace.size(); ++i){
      T_.bottomRows(1)(0,i) = 1.0 / errors[i];
      pVect_(i) = trace[i] * T_.bottomRows(1)(0,i);
    }
    
    wasDiscontiguous_ = true;
    return doFit(timeGuesses);       
  }   
  
private:  
  Output doFit(const std::vector<double>& timeGuesses);
  
  void evalTemplates(const std::vector<double>& tGuesses);
  
  bool hasConverged(); 

  void calculateCovarianceMatrix();

  std::unique_ptr<TSpline3> buildDSpline(const TSpline3* s);
  
  void resizeMatrices(int nSamples, int nPulses);

  //how small largest time step has to go before stopping minimization
  double accuracy_;
  //max number of iterations 
  unsigned int maxIterations_;  
  //whether covariance matrix is ready
  bool covReady_;  

  double lastNoiseLevel_;
  bool isFlatNoise_;
  bool wasDiscontiguous_;

  //spline stuff
  const TSpline3* tSpline_;
  std::unique_ptr<TSpline3> dSpline_;
  std::unique_ptr<TSpline3> d2Spline_;
  double tMin_, tMax_;
  double dEvalStep_; //step size for evaluating numerical derivatives
  
  //vector of time values corresponding to each sample
  std::vector<double> sampleTimes_;

  //eigen matrices, all kept around to avoid repeated allocation
  
  //vector of pulse heights weighted by inverse noise at each sample
  Eigen::VectorXd pVect_;
  //T matrix (see document)
  Eigen::MatrixXd T_;
  //b matrix (see document)
  //linear parameters will be stored in here after each fit
  Eigen::VectorXd b_;
  //vector of pulse minus fit function over noise
  Eigen::VectorXd deltas_;
  //template derivatives matrix
  Eigen::MatrixXd D_;
  //template second derivatives matrix
  Eigen::MatrixXd D2_;
  //hessian
  Eigen::MatrixXd Hess_;
  //covariance matrix
  Eigen::MatrixXd Cov_;
  //proposed time steps to minimum
  Eigen::VectorXd timeSteps_;
};
