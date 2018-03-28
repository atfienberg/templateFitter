// Template fitter implementation
//
// Aaron Fienberg
// fienberg@uw.edu

#include "TemplateFitter.hh"
#include <cmath>

gm2calo::TemplateFitter::TemplateFitter(int nPulses, int nSamples)
    : accuracy_(1e-4),
      maxIterations_(200),
      covReady_(false),
      lastNoiseLevel_(0),
      isFlatNoise_(false),
      wasDiscontiguous_(false),
      templates_(),
      sampleTimes_(nSamples),
      templatePattern_(),
      pVect_(nSamples),
      T_(nPulses + 1, nSamples),
      linearParams_(nPulses + 1),
      deltas_(nSamples),
      D_(nPulses, nSamples),
      D2_(nPulses, nSamples),
      Hess_(2 * nPulses + 1, 2 * nPulses + 1),
      Cov_(2 * nPulses + 1, 2 * nPulses + 1),
      timeSteps_(nPulses) {
  std::iota(sampleTimes_.begin(), sampleTimes_.end(), 0.0);
}

gm2calo::TemplateFitter::TemplateFitter(const TSpline3& primarySpline,
                                        int nPulses, int nSamples)
    : TemplateFitter(nPulses, nSamples) {
  setPrimaryTemplate(primarySpline);
}

gm2calo::TemplateFitter::TemplateFitter(const TSpline3& primarySpline,
                                        const TSpline3& secondarySpline)
    : TemplateFitter(0, 0) {
  setPrimaryTemplate(primarySpline);
  setSecondaryTemplate(secondarySpline);
}

double gm2calo::TemplateFitter::getCovariance(int i, int j) {
  if (covReady_) {
    return Cov_(i, j);
  }
  calculateCovarianceMatrix();
  covReady_ = true;
  return Cov_(i, j);
}

gm2calo::TemplateFitter::Output gm2calo::TemplateFitter::doFit(
    const std::vector<double>& timeGuesses) {
  const int nPulses = D_.rows();
  const int nSamples = D_.cols();
  covReady_ = false;

  Output fitOutput = {timeGuesses, std::vector<double>(nPulses, 0), 0, 0,
                      false};
  timeSteps_.setZero(nPulses);

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
                     .colPivHouseholderQr()
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

void gm2calo::TemplateFitter::evalTemplates(
    const std::vector<double>& tGuesses) {
  // fill T_, D_, D2_ matrices row by row
  for (int i = 0; i < D_.rows(); ++i) {
    // shift sample times by time guess for this pulse
    auto shiftedTimes = sampleTimes_;
    for (auto& time : shiftedTimes) {
      time -= tGuesses[i];
    }

    // select template based on templatePattern
    const CubicSpline& thisTemplate =
        (templatePattern_.size() > 0) && (templatePattern_[i] == true)
            ? templates_[1]
            : templates_[0];

    // evaluate template at these shifted times
    // first row has the template values
    // second has the first derivatives
    // third has the second derivatives
    const CubicSpline::Matrix valuesAndDerivatives =
        thisTemplate.evalWithDerivatives(shiftedTimes);

    // fill row i of T_, D_, and D2_
    for (unsigned int j = 0; j < valuesAndDerivatives.cols(); ++j) {
      // sample weights (1/uncertainty) are stored in bottom row of T_
      const double thisWeight = T_(T_.rows() - 1, j);
      T_(i, j) = thisWeight * valuesAndDerivatives(0, j);
      D_(i, j) = thisWeight * valuesAndDerivatives(1, j);
      D2_(i, j) = thisWeight * valuesAndDerivatives(2, j);
    }
  }
}

bool gm2calo::TemplateFitter::hasConverged() const {
  double maxStep = 0;
  for (int i = 0; i < timeSteps_.rows(); ++i) {
    double absStep = std::abs(timeSteps_(i));
    maxStep = absStep > maxStep ? absStep : maxStep;
  }

  return maxStep < accuracy_;
}

void gm2calo::TemplateFitter::calculateCovarianceMatrix() {
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

void gm2calo::TemplateFitter::resizeMatrices(int nSamples, int nPulses) {
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

// build gm2calo::CubicSpline from knots stored in TSpline3 object
gm2calo::CubicSpline gm2calo::TemplateFitter::buildCubicSpline(
    const TSpline3& tSpline, CubicSpline::BoundaryType cond) {
  unsigned int nKnots = tSpline.GetNp();

  CubicSpline::Knots knots(nKnots);

  for (unsigned int i = 0; i < nKnots; ++i) {
    tSpline.GetKnot(i, knots.xs[i], knots.ys[i]);
  }

  return CubicSpline(knots, cond);
}
