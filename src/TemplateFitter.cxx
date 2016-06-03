#include "TemplateFitter.hh"

TemplateFitter::TemplateFitter() : TemplateFitter(NULL, 0, 0) {}

TemplateFitter::TemplateFitter(const TSpline3* tSpline, double tMin,
                               double tMax)
    : accuracy_(1e-4),
      maxIterations_(200),
      covReady_(false),
      lastNoiseLevel_(0),
      isFlatNoise_(false),
      matrices_() {
  setTemplate(tSpline, tMin, tMax);
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
  const int nPulses = matrices_[0].D.rows();
  const int nSamples = matrices_[0].T.cols() + matrices_[1].T.cols();

  covReady_ = false;

  unsigned int nIterations = 0;

  Output fitOutput = {timeGuesses,
                      {std::vector<double>(nPulses), 0},
                      {std::vector<double>(nPulses), 0},
                      0,
                      true};

  totalHess_.fill(0);

  while (true) {
    evalTemplates(fitOutput.times);
    // solve for derivatives and linear parameters
    // for even and odd samples
    timeSteps_.fill(0);
    for (fitMatrices& m : matrices_) {
      m.Hess.bottomRightCorner(nPulses + 1, nPulses + 1) =
          m.T * m.T.transpose();

      m.b = m.T * m.pVect;

      m.b =
          m.Hess.bottomRightCorner(nPulses + 1, nPulses + 1).ldlt().solve(m.b);

      // build deltas vector based on current parameters
      m.deltas = m.pVect - m.T.transpose() * m.b;

      // build time-time block of Hessian and solve for time steps
      auto diagScales = m.b.head(nPulses).asDiagonal();

      m.Hess.topLeftCorner(nPulses, nPulses) = m.D * m.D.transpose();
      m.Hess.topLeftCorner(nPulses, nPulses) =
          diagScales * m.Hess.topLeftCorner(nPulses, nPulses) * diagScales;
      m.Hess.topLeftCorner(nPulses, nPulses) -=
          (m.b.head(nPulses).cwiseProduct(m.D2 * m.deltas)).asDiagonal();

      timeSteps_ += diagScales * m.D * m.deltas;
    }

    totalHess_.topLeftCorner(nPulses, nPulses) =
        matrices_[0].Hess.topLeftCorner(nPulses, nPulses) +
        matrices_[1].Hess.topLeftCorner(nPulses, nPulses);

    // solve set of time steps with Newton's method
    timeSteps_ =
        -1 *
        totalHess_.topLeftCorner(nPulses, nPulses).ldlt().solve(timeSteps_);

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
    fitOutput.evenParams.scales[i] = matrices_[0].b(i);
    fitOutput.oddParams.scales[i] = matrices_[1].b(i);
  }
  fitOutput.evenParams.pedestal = matrices_[0].b(nPulses);
  fitOutput.oddParams.pedestal = matrices_[1].b(nPulses);

  for (const auto& m : matrices_) {
    fitOutput.chi2 += (m.deltas.transpose() * m.deltas)(0, 0);
  }
  fitOutput.chi2 /= (nSamples - (3 * nPulses + 2));

  return fitOutput;
}

void TemplateFitter::evalTemplates(const std::vector<double>& tGuesses) {
  double stepsPerTime = (template_.size() - 1) / (tMax_ - tMin_);

  for (fitMatrices& m : matrices_) {
    for (int i = 0; i < m.D.rows(); ++i) {
      for (int j = 0; j < m.D.cols(); ++j) {
        double t = m.sampleTimes[j] - tGuesses[i];
        if ((t > tMin_) && (t < tMax_)) {
          double where = (t - tMin_) * stepsPerTime;
          int low = std::floor(where);
          double dt = where - low;

          m.T(i, j) =
              (template_[low] + (template_[low + 1] - template_[low]) * dt) *
              m.T.bottomRows(1)(0, j);

          m.D(i, j) =
              (dTemplate_[low] + (dTemplate_[low + 1] - dTemplate_[low]) * dt) *
              m.T.bottomRows(1)(0, j);

          m.D2(i, j) = (d2Template_[low] +
                        (d2Template_[low + 1] - d2Template_[low]) * dt) *
                       m.T.bottomRows(1)(0, j);
        }

        else {
          m.T(i, j) = 0;

          m.D(i, j) = 0;

          m.D2(i, j) = 0;
        }
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
  const int nPulses = matrices_[0].D.rows();

  // fill in diagonal blocks
  totalHess_.block(nPulses, nPulses, nPulses + 1, nPulses + 1) =
      matrices_[0].Hess.bottomRightCorner(nPulses + 1, nPulses + 1);
  totalHess_.block(2 * nPulses + 1, 2 * nPulses + 1, nPulses + 1, nPulses + 1) =
      matrices_[1].Hess.bottomRightCorner(nPulses + 1, nPulses + 1);

  // fill in all nonzero off diagonal blocks
  for (int i = 0; i < 2; ++i) {
    auto diagScales = matrices_[i].b.head(nPulses).asDiagonal();

    totalHess_.block(0, nPulses + i * (nPulses + 1), nPulses, nPulses + 1) =
        -1 * diagScales * matrices_[i].D * matrices_[i].T.transpose();
  }
  // fill in symmetric components and invert to get covariance matrix
  totalHess_ = totalHess_.selfadjointView<Eigen::Upper>();

  Cov_ = totalHess_.inverse();
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

bool TemplateFitter::resizeMatrices(int nPulses) {
  bool resized = false;
  for (fitMatrices& m : matrices_) {
    int nSamples = m.sampleTimes.size();
    if ((nSamples != m.pVect.rows()) || (m.D.rows() != nPulses)) {
      m.pVect.resize(nSamples);
      m.T.resize(nPulses + 1, nSamples);
      m.b.resize(nPulses + 1);
      m.deltas.resize(nSamples);
      m.D.resize(nPulses, nSamples);
      m.D2.resize(nPulses, nSamples);
      m.Hess.resize(2 * nPulses + 1, 2 * nPulses + 1);
      resized = true;
    }
  }

  if (resized) {
    timeSteps_.resize(nPulses);
    totalHess_.resize(3 * nPulses + 2, 3 * nPulses + 2);
    Cov_.resize(3 * nPulses + 2, 3 * nPulses + 2);
  }

  return resized;
}
