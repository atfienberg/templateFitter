#include "TemplateFitter.hh"

#include <cmath>

TemplateFitter::TemplateFitter(int nPulses, int nSamples)
    : TemplateFitter(nullptr, nullptr, 0, 0, nPulses, nSamples) {}

TemplateFitter::TemplateFitter(const TSpline3* tSplineA,
                               const TSpline3* tSplineB, double tMin,
                               double tMax, int nPulses, int nSamples)
    : accuracy_(1e-4),
      maxIterations_(200),
      covReady_(false),
      lastNoiseLevel_(0),
      isFlatNoise_(false),
      wasDiscontiguous_(false),
      templates_(2),
      dTemplates_(2),
      d2Templates_(2),
      sampleTimes_(nSamples),
      pVect_(nSamples),
      T_(2 * nPulses + 1, nSamples),
      b_(2 * nPulses + 1),
      deltas_(nSamples),
      D_(nPulses, nSamples),
      D2_(nPulses, nSamples),
      Hess_(3 * nPulses + 1, 3 * nPulses + 1),
      Cov_(23 * nPulses + 1, 3 * nPulses + 1),
      timeSteps_(nPulses) {
  setTemplate(tSplineA, tSplineB, tMin, tMax);
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

void TemplateFitter::setTemplate(const TSpline3* tSplineA,
                                 const TSpline3* tSplineB, double tMin,
                                 double tMax, int tPts) {
  if (tSplineA && tSplineB) {
    tMin_ = tMin;
    tMax_ = tMax;
    for (int tnum = 0; tnum < 2; ++tnum) {
      const TSpline3* tSpline = tnum == 0 ? tSplineA : tSplineB;
      templates_[tnum].resize(tPts);
      double stepSize = (tMax_ - tMin) / (templates_[tnum].size() - 1);
      for (std::size_t i = 0; i < templates_[tnum].size(); ++i) {
        templates_[tnum][i] = tSpline->Eval(tMin_ + i * stepSize);
      }
      dTemplates_[tnum] = buildDTemplate(templates_[tnum]);
      d2Templates_[tnum] = buildDTemplate(dTemplates_[tnum]);
    }
  }
}

void TemplateFitter::setTemplate(const std::vector<double>& tempA,
                                 const std::vector<double>& tempB, double tMin,
                                 double tMax) {
  assert(tempA.size() > 1 && tempB.size() > 2);
  tMin_ = tMin;
  tMax_ = tMax;
  templates_[0] = tempA;
  dTemplates_[0] = buildDTemplate(templates_[0]);
  d2Templates_[0] = buildDTemplate(dTemplates_[0]);
  templates_[1] = tempB;
  dTemplates_[1] = buildDTemplate(templates_[1]);
  d2Templates_[1] = buildDTemplate(dTemplates_[1]);
}

TemplateFitter::Output TemplateFitter::doFit(
    const std::vector<double>& timeGuesses) {
  const int nPulses = D_.rows();
  const int nSamples = D_.cols();

  covReady_ = false;

  std::size_t nIterations = 0;

  Output fitOutput = {timeGuesses, std::vector<double>(nPulses),
                      std::vector<double>(nPulses), 0, 0, true};

  while (true) {
    evalTemplates(fitOutput.times);

    // first solve for linear parameters based on current time guesses
    Hess_.bottomRightCorner(2 * nPulses + 1, 2 * nPulses + 1) =
        T_ * T_.transpose();

    b_ = T_ * pVect_;

    b_ = Hess_.bottomRightCorner(2 * nPulses + 1, 2 * nPulses + 1)
             .ldlt()
             .solve(b_);

    // build deltas vector based on current parameters
    deltas_ = pVect_ - T_.transpose() * b_;

    // evaluate combined derivative templates based on linear parameters
    evalDerivTemplates(fitOutput.times);

    // build time-time block of Hessian and solve for time steps
    Hess_.topLeftCorner(nPulses, nPulses) = D_ * D_.transpose();
    Eigen::VectorXd d2xdeltas = D2_ * deltas_;
    Hess_.topLeftCorner(nPulses, nPulses) -= d2xdeltas.asDiagonal();

    // solve set of time steps with Newton's method
    timeSteps_ =
        -1 * Hess_.topLeftCorner(nPulses, nPulses).ldlt().solve(D_ * deltas_);

    // check for convergence, update time guesses
    ++nIterations;
    if ((nIterations <= maxIterations_) && (!hasConverged())) {
      for (std::size_t i = 0; i < timeGuesses.size(); ++i) {
        fitOutput.times[i] += timeSteps_(i);
      }
    }

    else if (nIterations <= maxIterations_) {
      break;
    }

    else {
      fitOutput.converged = false;
      break;
    }
  }

  // return output
  for (int i = 0; i < nPulses; ++i) {
    fitOutput.scalesA[i] = b_(2 * i);
    fitOutput.scalesB[i] = b_(2 * i + 1);
  }
  fitOutput.pedestal = b_(2 * nPulses);
  fitOutput.chi2 =
      (deltas_.transpose() * deltas_)(0, 0) / (nSamples - 2 * nPulses - 1);
      
  return fitOutput;
}

void TemplateFitter::evalTemplates(const std::vector<double>& tGuesses) {
  double stepsPerTime = (templates_[0].size() - 1) / (tMax_ - tMin_);

  for (int i = 0; i < T_.rows() - 1; ++i) {
    for (int j = 0; j < T_.cols(); ++j) {
      double t = sampleTimes_[j] - tGuesses[i / 2];
      if ((t > tMin_) && (t < tMax_)) {
        double where = (t - tMin_) * stepsPerTime;
        int low = std::floor(where);
        double dt = where - low;

        T_(i, j) =
            (templates_[i % 2][low] +
             (templates_[i % 2][low + 1] - templates_[i % 2][low]) * dt) *
            T_.bottomRows(1)(0, j);

      } else {
        T_(i, j) = 0;
      }
    }
  }
}

void TemplateFitter::evalDerivTemplates(const std::vector<double>& tGuesses) {
  double stepsPerTime = (templates_[0].size() - 1) / (tMax_ - tMin_);

  for (int i = 0; i < D_.rows(); ++i) {
    for (int j = 0; j < D_.cols(); ++j) {
      double t = sampleTimes_[j] - tGuesses[i];
      if ((t > tMin_) && (t < tMax_)) {
        double where = (t - tMin_) * stepsPerTime;
        int low = std::floor(where);
        double dt = where - low;

        double combinedDTemplate = 0;
        double combinedD2Template = 0;
        // combine the A and B template derivitives using current scale guesses
        for (int tnum = 0; tnum < 2; ++tnum) {
          combinedDTemplate +=
              b_[2 * i + tnum] *
              (dTemplates_[tnum][low] +
               (dTemplates_[tnum][low + 1] - dTemplates_[tnum][low]) * dt);

          combinedD2Template +=
              b_[2 * i + tnum] *
              (d2Templates_[tnum][low] +
               (d2Templates_[tnum][low + 1] - d2Templates_[tnum][low]) * dt);
        }

        D_(i, j) = combinedDTemplate * T_.bottomRows(1)(0, j);
        D2_(i, j) = combinedD2Template * T_.bottomRows(1)(0, j);

      }

      else {
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

  // assuming a fit was done successfully, the time-time
  // and scale/ped - scale/ped blocks in hessian
  // should already be in place

  // time - scale/ped derivatives
  Hess_.topRightCorner(nPulses, 2 * nPulses + 1) = -1 * D_ * T_.transpose();

  // fill in symmetric components and invert to get covariance matrix
  Hess_.bottomLeftCorner(2 * nPulses + 1, nPulses) =
      Hess_.topRightCorner(nPulses, 2 * nPulses + 1).transpose();

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
  T_.resize(2 * nPulses + 1, nSamples);
  b_.resize(2 * nPulses + 1);
  deltas_.resize(nSamples);
  D_.resize(nPulses, nSamples);
  D2_.resize(nPulses, nSamples);
  Hess_.resize(3 * nPulses + 1, 3 * nPulses + 1);
  Cov_.resize(3 * nPulses + 1, 3 * nPulses + 1);
  timeSteps_.resize(nPulses);
}
