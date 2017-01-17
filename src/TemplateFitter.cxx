#include "TemplateFitter.hh"

#include <cmath>
#include <limits>

TemplateFitter::TemplateFitter()
    : TemplateFitter(std::vector<const TSpline3*>()) {}

TemplateFitter::TemplateFitter(const TSpline3* spline, int tPts)
    : TemplateFitter(std::vector<const TSpline3*>{spline}, tPts) {}

TemplateFitter::TemplateFitter(const std::vector<const TSpline3*>& splines,
                               int tPts)
    : accuracy_(1e-4),
      maxIterations_(200),
      covReady_(false),
      lastNoiseLevel_(0),
      isFlatNoise_(false),
      wasDiscontiguous_(false) {
  setTemplate(splines, tPts);
  resizeMatrices(0, 0);
}

double TemplateFitter::getCovariance(int i, int j) {
  if (covReady_) {
    return Cov_(i, j);
  } else {
    calculateCovarianceMatrix();
    covReady_ = true;
    return Cov_(i, j);
  }
}

void TemplateFitter::setTemplate(const TSpline3* spline, int tPts) {
  setTemplate(std::vector<const TSpline3*>{spline}, tPts);
}

void TemplateFitter::setTemplate(const std::vector<const TSpline3*>& splines,
                                 int tPts) {
  templates_.resize(splines.size());
  dTemplates_.resize(splines.size());
  d2Templates_.resize(splines.size());
  tMin_ = std::numeric_limits<double>::lowest();
  tMax_ = std::numeric_limits<double>::max();

  // pick largest template xMin for tMin
  // pick smallest template xMax for tMax
  // this ensures all templates are valid in full range
  for (const auto& spline : splines) {
    tMin_ = spline->GetXmin() > tMin_ ? spline->GetXmin() : tMin_;
    tMax_ = spline->GetXmax() < tMax_ ? spline->GetXmax() : tMax_;
  }

  for (unsigned int tnum = 0; tnum < splines.size(); ++tnum) {
    templates_[tnum].resize(tPts);
    double stepSize = (tMax_ - tMin_) / (templates_[tnum].size() - 1);
    for (std::size_t i = 0; i < templates_[tnum].size(); ++i) {
      templates_[tnum][i] = splines[tnum]->Eval(tMin_ + i * stepSize);
    }

    dTemplates_[tnum] = buildDTemplate(templates_[tnum]);
    d2Templates_[tnum] = buildDTemplate(dTemplates_[tnum]);
  }
}

TemplateFitter::Output TemplateFitter::doFit(
    const std::vector<double>& timeGuesses) {
  const int nPulses = D_.rows();
  const int nSamples = D_.cols();
  const int nTemplates = templates_.size();

  covReady_ = false;

  std::size_t nIterations = 0;

  Output fitOutput = {timeGuesses,
                      std::vector<std::vector<double>>(
                          nPulses, std::vector<double>(nTemplates)),
                      0, 0, true};

  while (true) {
    evalTemplates(fitOutput.times);

    // first solve for linear parameters based on current time guesses
    Hess_.bottomRightCorner(nTemplates * nPulses + 1,
                            nTemplates * nPulses + 1) = T_ * T_.transpose();

    linearParams_ = T_ * pVect_;

    linearParams_ = Hess_.bottomRightCorner(nTemplates * nPulses + 1,
                                            nTemplates * nPulses + 1)
                        .ldlt()
                        .solve(linearParams_);

    // build deltas vector based on current parameters
    deltas_ = pVect_ - T_.transpose() * linearParams_;

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
    for (int j = 0; j < nTemplates; ++j) {
      fitOutput.scales[i][j] = linearParams_(nTemplates * i + j);
    }
  }
  fitOutput.pedestal = linearParams_(nTemplates * nPulses);
  fitOutput.chi2 = (deltas_.transpose() * deltas_)(0, 0) /
                   (nSamples - (nTemplates + 1) * nPulses - 1);

  return fitOutput;
}

void TemplateFitter::evalTemplates(const std::vector<double>& tGuesses) {
  double stepsPerTime = (templates_[0].size() - 1) / (tMax_ - tMin_);
  unsigned int nTemplates = templates_.size();

  for (int i = 0; i < T_.rows() - 1; ++i) {
    for (int j = 0; j < T_.cols(); ++j) {
      double t = sampleTimes_[j] - tGuesses[i / nTemplates];
      if ((t > tMin_) && (t < tMax_)) {
        double where = (t - tMin_) * stepsPerTime;
        int low = std::floor(where);
        double dt = where - low;

        T_(i, j) = (templates_[i % nTemplates][low] * (1 - dt) +
                    templates_[i % nTemplates][low + 1] * dt) *
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
        int nTemplates = templates_.size();
        // combine the A and B template derivitives using current scale
        // guesses
        for (int tnum = 0; tnum < nTemplates; ++tnum) {
          combinedDTemplate += linearParams_[nTemplates * i + tnum] *
                               (dTemplates_[tnum][low] * (1 - dt) +
                                dTemplates_[tnum][low + 1] * dt);

          combinedD2Template += linearParams_[nTemplates * i + tnum] *
                                (d2Templates_[tnum][low] * (1 - dt) +
                                 d2Templates_[tnum][low + 1] * dt);
        }

        D_(i, j) = combinedDTemplate * T_.bottomRows(1)(0, j);
        D2_(i, j) = combinedD2Template * T_.bottomRows(1)(0, j);

      } else {
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
  const unsigned int nTemplates = templates_.size();
  // assuming a fit was done successfully, the time-time
  // and scale/ped - scale/ped blocks in hessian
  // should already be in place

  // time - scale/ped derivatives
  Hess_.topRightCorner(nPulses, nTemplates * nPulses + 1) =
      -1 * D_ * T_.transpose();

  // fill in symmetric components and invert to get covariance matrix
  Hess_.bottomLeftCorner(nTemplates * nPulses + 1, nPulses) =
      Hess_.topRightCorner(nPulses, nTemplates * nPulses + 1).transpose();

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
  const int nTemplates = templates_.size();

  sampleTimes_.resize(nSamples);
  pVect_.resize(nSamples);
  T_.resize(nTemplates * nPulses + 1, nSamples);
  linearParams_.resize(nTemplates * nPulses + 1);
  deltas_.resize(nSamples);
  D_.resize(nPulses, nSamples);
  D2_.resize(nPulses, nSamples);
  Hess_.resize((nTemplates + 1) * nPulses + 1, (nTemplates + 1) * nPulses + 1);
  Cov_.resize((nTemplates + 1) * nPulses + 1, (nTemplates + 1) * nPulses + 1);
  timeSteps_.resize(nPulses);
}
