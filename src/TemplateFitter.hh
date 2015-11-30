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

class TemplateFitter{
public:
  
  typedef struct{
    std::vector<double> scales; 
    double pedestal;
  } LinearParams;

  //output data type
  typedef struct {
    std::vector<double> times; 
    LinearParams evenParams;
    LinearParams oddParams;
    double chi2;
    bool converged;
  } Output;
    
  //construct with matrix dimensions (default to 0)
  TemplateFitter();
  
  //construct w/ template spline, its limits of validity, and matrix dimensions
  TemplateFitter(const TSpline3* tSpline, double tMin, double tMax);
  
  //give template spline and its limits of validity 
  //optionally give number of pts at which to evaluate it
  void setTemplate(const TSpline3* tSpline, double tMin, double tMax, int tPts = 1000);
  
  //give template as a vector with its time limits (time of first and last sample)
  void setTemplate(const std::vector<double>& temp, double tMin, double tMax);

  //get covariance_ij. don't call this before doing a fit
  //order of parameters is {t1 ... tn, s1 ... sn, pedestal}
  double getCovariance(int i, int j);
  
  //max number of iterations before giving up
  unsigned int getMaxIterations() const { return maxIterations_; }
  void setMaxIterations(unsigned int maxIters){ maxIterations_ = maxIters; }
  
  //target accuracy. When max step size is less than this,
  //the numerical minimization stops
  double getAccuracy() const { return accuracy_; }
  void setAccuracy(double newAccuracy){ accuracy_ = newAccuracy;}

  //get number of pts in templates
  unsigned int getNTemplatePoints() const {return template_.size(); }


  //discontiguousFit() functions for fitting discontiguous regions
  //these are mainly useful for clipped pulses
  //you must pass in vector of sample times along with vector of sample values
  
  //single pulse version
  template<typename sampleType, typename timeType, typename errorType = double>
  Output discontiguousFit(const std::vector<sampleType>& trace, 
			  const std::vector<timeType>& sampleTimes, 
			  double timeGuess,
			  errorType error = 1.0
			  ){

    return discontiguousFit(trace, sampleTimes, std::vector<double> {timeGuess}, error);    
    
  }

  //discontiguous fit with flat errors
  template <typename sampleType, typename timeType, typename noiseType = double>
  Output discontiguousFit(const std::vector<sampleType>& trace, 
			  const std::vector<timeType>& sampleTimes, 
			  const std::vector<double>& timeGuesses,
			  noiseType noiseLevel = 1.0){ 
    
    static_assert(std::is_arithmetic<sampleType>::value, 
		  "trace must be vector of numbers!");
    static_assert(std::is_integral<timeType>::value, 
		  "sampleTimes must be vector of integers!");
    static_assert(std::is_arithmetic<noiseType>::value, 
		  "noise level must be a number!");
    assert(noiseLevel != 0);
    assert(trace.size() == sampleTimes.size());
    
    static auto isEven = [](timeType t) { return t % 2 == 0; };
    static auto isOdd = [](timeType t) { return !isEven(t); };
   
    matrices_[0].sampleTimes.resize(std::count_if(sampleTimes.begin(), 
					    sampleTimes.end(),
					    isEven));
    matrices_[1].sampleTimes.resize(sampleTimes.size() - matrices_[0].sampleTimes.size());
       
    bool resized = resizeMatrices(timeGuesses.size());
    
    std::copy_if(sampleTimes.begin(), sampleTimes.end(), 
		 matrices_[0].sampleTimes.begin(), isEven);
    std::copy_if(sampleTimes.begin(), sampleTimes.end(), 
		 matrices_[1].sampleTimes.begin(), isOdd);
    
    if((resized) || (!isFlatNoise_) || (lastNoiseLevel_ != noiseLevel)){
      for(fitMatrices& m : matrices_) m.T.bottomRows(1).fill(1.0/noiseLevel);
      lastNoiseLevel_ = noiseLevel;
      isFlatNoise_ = true;    
    }

    int nextIndex[2] = {0, 0};
    for(int i = 0; i < trace.size(); ++i){
      int evenOrOdd = sampleTimes[i] % 2;
      matrices_[evenOrOdd].pVect(nextIndex[evenOrOdd]) = 
	trace[i] * matrices_[evenOrOdd].T.bottomRows(1)(0, nextIndex[evenOrOdd]);
      nextIndex[evenOrOdd]++;
    }    
        
    wasDiscontiguous_ = true;
    return doFit(timeGuesses);       
  }   

  //discontiguous fit with arbitrary errors    
  template<typename sampleType, typename timeType, typename noiseType>
  Output discontiguousFit(const std::vector<sampleType>& trace, 
			  const std::vector<timeType>& sampleTimes, 
			  const std::vector<double>& timeGuesses,
			  const std::vector<noiseType>& errors){ 

    static_assert(std::is_arithmetic<sampleType>::value, 
		  "trace must be vector of numbers!");
    static_assert(std::is_integral<timeType>::value, 
		  "sampleTimes must be vector of integers!");
    static_assert(std::is_arithmetic<noiseType>::value, 
		  "errors must be vector of numbers!");
    assert(trace.size() == sampleTimes.size());
    
    static auto isEven = [](timeType t) { return t % 2 == 0; };
    static auto isOdd = [](timeType t) { return !isEven(t); };
   
    matrices_[0].sampleTimes.resize(std::count_if(sampleTimes.begin(), 
						  sampleTimes.end(),
						  isEven));
    matrices_[1].sampleTimes.resize(sampleTimes.size() - matrices_[0].sampleTimes.size());
       
    resizeMatrices(timeGuesses.size());
    
    std::copy_if(sampleTimes.begin(), sampleTimes.end(), 
		 matrices_[0].sampleTimes.begin(), isEven);
    std::copy_if(sampleTimes.begin(), sampleTimes.end(), 
		 matrices_[1].sampleTimes.begin(), isOdd);
   
    int nextIndex[2] = {0, 0};
    for(int i = 0; i < trace.size(); ++i){
      int evenOrOdd = sampleTimes[i] % 2;
      matrices_[evenOrOdd].T.bottomRows(1)(0,nextIndex[evenOrOdd]) = 1.0/errors[i];
      matrices_[evenOrOdd].pVect(nextIndex[evenOrOdd]) = 
	trace[i] * matrices_[evenOrOdd].T.bottomRows(1)(0, nextIndex[evenOrOdd]);      
      nextIndex[evenOrOdd]++;
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
  
  bool resizeMatrices(int nPulses);

  //how small largest time step has to go before stopping minimization
  double accuracy_;
  //max number of iterations 
  unsigned int maxIterations_;  
  //whether covariance matrix is ready
  bool covReady_;  

  double lastNoiseLevel_;
  bool isFlatNoise_;
  bool wasDiscontiguous_;

  //template stuff
  std::vector<double> template_;
  std::vector<double> dTemplate_;
  std::vector<double> d2Template_;
  double tMin_, tMax_;
  
  typedef struct{
    //vector of time values corresponding to each sample
    std::vector<double> sampleTimes;
    //eigen matrices, all kept around to avoid repeated allocation
    //vector of pulse heights weighted by inverse noise at each sample
    Eigen::VectorXd pVect;
    //T matrix (see document)
    Eigen::MatrixXd T;
    //linear parameters will be stored in here after each fit
    Eigen::VectorXd b;
    //vector of pulse minus fit function over noise
    Eigen::VectorXd deltas;
    //template derivatives matrix
    Eigen::MatrixXd D;
    //template second derivatives matrix
    Eigen::MatrixXd D2;
    //hessian
    Eigen::MatrixXd Hess;
  } fitMatrices;

  //even matricse in slot 0 and odd in slot 1
  fitMatrices matrices_[2];
  Eigen::MatrixXd totalHess_;

  //covariance matrix
  Eigen::MatrixXd Cov_;
  //proposed time steps to minimum
  Eigen::VectorXd timeSteps_;
};
