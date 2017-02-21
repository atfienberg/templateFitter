#include "TemplateFitter.hh"

#include <cmath>

TemplateFitter::TemplateFitter(int nPulses, int nSamples)
    : TemplateFitter(NULL, 0, 0, nPulses, nSamples) {}

TemplateFitter::TemplateFitter(const TSpline3* tSpline, double tMin,
                               double tMax, int nPulses, int nSamples)
    : accuracy_(1e-4),
      maxIterations_(200),
      covReady_(false),
      lastNoiseLevel_(0),
      isFlatNoise_(false),
      wasDiscontiguous_(false),
      sampleTimes_(nSamples),
      pVect_(nSamples),
      T_(nPulses + 1, nSamples),
      linearParams_(nPulses + 1),
      deltas_(nSamples),
      D_(nPulses, nSamples),
      D2_(nPulses, nSamples),
      Hess_(2 * nPulses + 1, 2 * nPulses + 1),
      Cov_(2 * nPulses + 1, 2 * nPulses + 1),
      timeSteps_(nPulses) {
  setTemplate(tSpline, tMin, tMax);
  std::iota(sampleTimes_.begin(), sampleTimes_.end(), 0.0);
}

double TemplateFitter::getCovariance(int i, int j) {
  if (covReady_) {
    return Cov_(i, j);
  }
  calculateCovarianceMatrix();
  covReady_ = true;
  return Cov_(i, j);
}

void TemplateFitter::setTemplate(const TSpline3* tSpline, double tMin,
                                 double tMax, int tPts) {
  if (tSpline) {
    tMin_ = tMin;
    tMax_ = tMax;
    template_.resize(tPts);
    double stepSize = (tMax_ - tMin) / (template_.size() - 1);
    for (std::size_t i = 0; i < template_.size(); ++i) {
      template_[i] = tSpline->Eval(tMin_ + i * stepSize);
    }
    dTemplate_ = buildDTemplate(template_);
    d2Template_ = buildDTemplate(dTemplate_);
  }
}

void TemplateFitter::setTemplate(const std::vector<double>& temp, double tMin,
                                 double tMax) {
  assert(temp.size() > 1);
  tMin_ = tMin;
  tMax_ = tMax;
  template_ = temp;
  dTemplate_ = buildDTemplate(template_);
  d2Template_ = buildDTemplate(dTemplate_);
}

TemplateFitter::Output TemplateFitter::doFit(
    const std::vector<double>& timeGuesses) {
  const int nPulses = D_.rows();
  const int nSamples = D_.cols();
  covReady_ = false;

  Output fitOutput = {timeGuesses, std::vector<double>(nPulses, 0), 0, 0,
                      false};
  timeSteps_.setZero(nPulses);

  // break on nIterations <= max, not < max
  // this allows call to fit with maxIterations == 0 to solve
  // for linear params at user provided times, without taking any time steps.
  // iteration j+1 solves for linear params at times found in iteration j
  for (unsigned int nIterations = 0;
       nIterations <= maxIterations_ && !(fitOutput.converged); ++nIterations) {
    // update time guesses
    // it is important to do this at beginning of loop, not end.
    // this ensures times in fit result match times for which linear params
    // were solved and chi2/residuals calculated
    for (std::size_t i = 0; i < timeGuesses.size(); ++i) {
      fitOutput.times[i] += timeSteps_(i);
    }

    evalTemplates(fitOutput.times);

    // first solve for linear parameters based on current time guesses
    Hess_.bottomRightCorner(nPulses + 1, nPulses + 1) = T_ * T_.transpose();

    linearParams_ = T_ * pVect_;

    linearParams_ = Hess_.bottomRightCorner(nPulses + 1, nPulses + 1)
                        .ldlt()
                        .solve(linearParams_);

    // build deltas vector based on current parameters
    deltas_ = pVect_ - T_.transpose() * linearParams_;

    // build time-time block of Hessian and solve for time steps
    auto diagScales = linearParams_.head(nPulses).asDiagonal();

    Hess_.topLeftCorner(nPulses, nPulses) = D_ * D_.transpose();
    Hess_.topLeftCorner(nPulses, nPulses) =
        diagScales * Hess_.topLeftCorner(nPulses, nPulses) * diagScales;
    Hess_.topLeftCorner(nPulses, nPulses) -=
        (linearParams_.head(nPulses).cwiseProduct(D2_ * deltas_)).asDiagonal();

    // solve set of time steps with Newton's method
    timeSteps_ = -1 *
                 Hess_.topLeftCorner(nPulses, nPulses)
                     .ldlt()
                     .solve(diagScales * D_ * deltas_);

    // check for convergence
    fitOutput.converged = hasConverged();
  }

  // return output
  for (int i = 0; i < nPulses; ++i) {
    fitOutput.scales[i] = linearParams_(i);
  }
  fitOutput.pedestal = linearParams_(nPulses);
  fitOutput.chi2 =
      (deltas_.transpose() * deltas_)(0, 0) / (nSamples - 2 * nPulses - 1);
  return fitOutput;
}

void TemplateFitter::evalTemplates(const std::vector<double>& tGuesses) {
  double stepsPerTime = (template_.size() - 1) / (tMax_ - tMin_);

  for (int i = 0; i < D_.rows(); ++i) {
    for (int j = 0; j < D_.cols(); ++j) {
      double t = sampleTimes_[j] - tGuesses[i];
      if ((t > tMin_) && (t < tMax_)) {
        double where = (t - tMin_) * stepsPerTime;
        int low = std::floor(where);
        double dt = where - low;

        T_(i, j) =
            (template_[low] + (template_[low + 1] - template_[low]) * dt) *
            T_.bottomRows(1)(0, j);

        D_(i, j) =
            (dTemplate_[low] + (dTemplate_[low + 1] - dTemplate_[low]) * dt) *
            T_.bottomRows(1)(0, j);

        D2_(i, j) = (d2Template_[low] +
                     (d2Template_[low + 1] - d2Template_[low]) * dt) *
                    T_.bottomRows(1)(0, j);
      }

      else {
        T_(i, j) = 0;

        D_(i, j) = 0;

        D2_(i, j) = 0;
      }
    }
  }
}

bool TemplateFitter::hasConverged() {
  double maxStep = 0;
  for (int i = 0; i < timeSteps_.rows(); ++i) {
    double absStep = std::abs(timeSteps_(i));
    maxStep = absStep > maxStep ? absStep : maxStep;
  }

  return maxStep < accuracy_;
}

void TemplateFitter::calculateCovarianceMatrix() {
  const int nPulses = D_.rows();

  auto diagScales = linearParams_.head(nPulses).asDiagonal();

  // assuming a fit was done successfully, the time-time
  // and scale/ped - scale/ped blocks in hessian
  // should already be in place

  // time - scale/ped derivatives
  Hess_.topRightCorner(nPulses, nPulses + 1) =
      -1 * diagScales * D_ * T_.transpose();

  // fill in symmetric components and invert to get covariance matrix
  Hess_.bottomLeftCorner(nPulses + 1, nPulses) =
      Hess_.topRightCorner(nPulses, nPulses + 1).transpose();

  Cov_ = Hess_.inverse();
}

std::vector<double> TemplateFitter::buildDTemplate(
    const std::vector<double>& temp) {
  assert(temp.size() > 1);

  std::vector<double> dTemplate(temp.size());
  double stepSize = (tMax_ - tMin_) / (temp.size() - 1);

  dTemplate[0] = (temp[1] - temp[0]) / stepSize;
  for (std::size_t i = 1; i < temp.size() - 1; ++i) {
    dTemplate[i] = (temp[i + 1] - temp[i - 1]) / (2 * stepSize);
  }
  dTemplate[temp.size() - 1] =
      (temp[temp.size() - 1] - temp[temp.size() - 2]) / stepSize;

  return dTemplate;
}

void TemplateFitter::resizeMatrices(int nSamples, int nPulses) {
  sampleTimes_.resize(nSamples);
  pVect_.resize(nSamples);
  T_.resize(nPulses + 1, nSamples);
  linearParams_.resize(nPulses + 1);
  deltas_.resize(nSamples);
  D_.resize(nPulses, nSamples);
  D2_.resize(nPulses, nSamples);
  Hess_.resize(2 * nPulses + 1, 2 * nPulses + 1);
  Cov_.resize(2 * nPulses + 1, 2 * nPulses + 1);
  timeSteps_.resize(nPulses);
}
