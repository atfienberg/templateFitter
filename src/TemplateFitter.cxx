#include "TemplateFitter.hh"

#include "TGraph.h"

#include <cmath>

TemplateFitter::TemplateFitter(int nPulses, int nSamples) :
  TemplateFitter(NULL, 0, 0, nPulses, nSamples)
{}

TemplateFitter::TemplateFitter(const TSpline3* tSpline, double tMin, double tMax, 
			       int nPulses, int nSamples) : 
  accuracy_(1e-4),
  maxIterations_(200),
  covReady_(false),
  lastNoiseLevel_(0),
  isFlatNoise_(false),
  wasDiscontiguous_(false),
  dEvalStep_(0.2),
  sampleTimes_(nSamples),
  pVect_(nSamples),
  T_(nPulses + 1, nSamples),
  b_(nPulses + 1),
  deltas_(nSamples),
  D_(nPulses, nSamples),
  D2_(nPulses, nSamples),
  Hess_(2*nPulses + 1, 2*nPulses + 1),
  Cov_(2*nPulses + 1, 2*nPulses + 1),
  timeSteps_(nPulses)
{  
  setTemplate(tSpline, tMin, tMax);
  std::iota(sampleTimes_.begin(),sampleTimes_.end(), 0.0);
}
  
double TemplateFitter::getCovariance(int i, int j){
  if(covReady_){
    return Cov_(i, j);
  }
  calculateCovarianceMatrix();
  covReady_ = true;
  return Cov_(i, j);
}

void TemplateFitter::setTemplate(const TSpline3* tSpline, double tMin, double tMax){
  if(tSpline){
    tSpline_.reset((TSpline3*)tSpline->Clone());
    tMin_ = tMin;
    tMax_ = tMax;  
    dSpline_ = buildDSpline(tSpline_.get());
    d2Spline_ = buildDSpline(dSpline_.get());
  }
}
 
TemplateFitter::Output TemplateFitter::doFit(const std::vector<double>& timeGuesses){
  
  const int nPulses = D_.rows();
  const int nSamples = D_.cols();

  covReady_ = false;

  unsigned int nIterations = 0;
  
  Output fitOutput = {timeGuesses, std::vector<double>(nPulses), 
		      0, 0, true};

  while(true){
    evalTemplates(fitOutput.times);

    //first solve for linear parameters based on current time guesses
    Hess_.bottomRightCorner(nPulses + 1, nPulses + 1) = T_ * T_.transpose();

    b_ = T_ * pVect_;

    b_ = Hess_.bottomRightCorner(nPulses + 1, nPulses + 1).ldlt().solve(b_);

    //build deltas vector based on current parameters
    deltas_ = pVect_;
    for(int i = 0; i < T_.rows(); ++i){
      deltas_ -= b_(i) * T_.row(i).transpose();
    }

    //build time-time block of Hessian and solve for time steps
    auto diagScales = b_.head(nPulses).asDiagonal();

    Hess_.topLeftCorner(nPulses,nPulses) = D_ * D_.transpose();
    Hess_.topLeftCorner(nPulses,nPulses) = diagScales*Hess_.topLeftCorner(nPulses,nPulses)*diagScales;    
    Hess_.topLeftCorner(nPulses,nPulses) -= (b_.head(nPulses).cwiseProduct(D2_ * deltas_)).asDiagonal();

    //solve set of time steps with Newton's method
    timeSteps_ = -1*Hess_.topLeftCorner(nPulses,nPulses).ldlt().solve(diagScales * D_ * deltas_);   

    //check for convergence, update time guesses
    ++nIterations;
    if((nIterations != maxIterations_) && (!hasConverged())){
      for(unsigned int i = 0; i < timeGuesses.size(); ++i){
	fitOutput.times[i] += timeSteps_(i);
      }    
    }

    else if(nIterations == maxIterations_){
      fitOutput.converged = false;
      break;
    }

    else{
      break;
    }

  }

  //return output
  for(int i = 0; i < nPulses; ++i){
    fitOutput.scales[i] = b_(i);
  }
  fitOutput.pedestal = b_(nPulses);
  fitOutput.chi2 = (deltas_.transpose() * deltas_)(0,0) / (nSamples - 2*nPulses - 1);  
  return fitOutput;
}

void TemplateFitter::evalTemplates(const std::vector<double>& tGuesses){
  assert(tSpline_);
  for(int i = 0; i < D_.rows(); ++i){
    for(int j = 0; j < D_.cols(); ++j){
      if((sampleTimes_[j] - tGuesses[i] > tMin_) && (sampleTimes_[j] - tGuesses[i]  < tMax_)){
	T_(i, j) = tSpline_->Eval(sampleTimes_[j] - tGuesses[i])*T_.bottomRows(1)(0,j);
	D_(i, j) = dSpline_->Eval(sampleTimes_[j] - tGuesses[i])*T_.bottomRows(1)(0,j);
	D2_(i, j) = d2Spline_->Eval(sampleTimes_[j] - tGuesses[i])*T_.bottomRows(1)(0,j);
      }
      else{
	T_(i, j) = 0;
	D_(i, j) = 0;
	D2_(i, j) = 0;
      }
    }
  }
}

bool TemplateFitter::hasConverged(){
  double maxStep = 0;
  for(int i = 0; i < timeSteps_.rows(); ++i){
    double absStep = std::abs(timeSteps_(i));
    maxStep = absStep > maxStep ? absStep : maxStep;
  }

  return maxStep < accuracy_;
}

void TemplateFitter::calculateCovarianceMatrix(){
  const int nPulses = D_.rows();
  
  auto diagScales = b_.head(nPulses).asDiagonal();  

  //assuming a fit was done successfully, the time-time 
  //and scale/ped - scale/ped blocks in hessian 
  //should already be in place
  
  //time - scale/ped derivatives
  Hess_.topRightCorner(nPulses, nPulses + 1) = 
    -1 * diagScales * D_ * T_.transpose();
  
  //fill in symmetric components and invert to get covariance matrix
  Hess_.bottomLeftCorner(nPulses + 1, nPulses) =
    Hess_.topRightCorner(nPulses, nPulses + 1).transpose();
  
  Cov_ = Hess_.inverse();
}

std::unique_ptr<TSpline3> TemplateFitter::buildDSpline(const TSpline3* s){
  TGraph cderivative(0);
  cderivative.SetTitle("cderivative");
  for(double t = tMin_; t < tMax_; t += dEvalStep_){
    cderivative.SetPoint(cderivative.GetN(), t + dEvalStep_/2.0,
			 (s->Eval(t + dEvalStep_) - s->Eval(t))/dEvalStep_);
  }
  std::unique_ptr<TSpline3> dSpline(new TSpline3("dspline", &cderivative));
  return dSpline;
}

void TemplateFitter::resizeMatrices(int nSamples, int nPulses){  
  sampleTimes_.resize(nSamples);
  pVect_.resize(nSamples);
  T_.resize(nPulses + 1, nSamples);
  b_.resize(nPulses + 1);
  deltas_.resize(nSamples);
  D_.resize(nPulses, nSamples);
  D2_.resize(nPulses, nSamples);
  Hess_.resize(2*nPulses + 1, 2*nPulses + 1);
  Cov_.resize(2*nPulses + 1, 2*nPulses + 1);
  timeSteps_.resize(nPulses);
}
