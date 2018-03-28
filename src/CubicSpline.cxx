// implementation file for the CubicSpline class in CubicSpline.hh

#include "CubicSpline.hh"
#include <array>

double gm2calo::CubicSpline::operator()(double x) const {
  // find the region in which this x resides
  const unsigned int firstNotLessIndex =
      std::distance(xs_.begin(), std::lower_bound(xs_.begin(), xs_.end(), x));

  // 0 means xs_[0] > x, xs.size() means x > xs_.back()
  // in either case, return zero because this is outside spline's region
  if (firstNotLessIndex == 0 || firstNotLessIndex == xs_.size()) {
    return 0;
  }

  // otherwise, the region we want is indexed by firstNotLessIndex - 1
  unsigned int regionIndex = firstNotLessIndex - 1;

  // evaluate polynomial in this region
  x -= xs_[regionIndex];
  double retval = coeffs_(regionIndex, 0);
  double xPowers = 1;
  for (unsigned int power = 1; power < 4; ++power) {
    xPowers *= x;
    retval += xPowers * coeffs_(regionIndex, power);
  }

  return retval;
}

// evaluate spline and derivatives
// times must be sorted
gm2calo::CubicSpline::Matrix gm2calo::CubicSpline::evalWithDerivatives(
    const std::vector<double>& times) const {
  assert(std::is_sorted(times.begin(), times.end()));

  // find the spline time regions associated with each time
  std::vector<unsigned int> regionIndices(times.size());

  auto timesIter = xs_.begin();

  // indices to first and last time inside of the valid spline time window
  unsigned int firstValidIndex = regionIndices.size();
  unsigned int firstInvalidIndex = regionIndices.size();
  for (unsigned int i = 0; i < times.size(); ++i) {
    // find iterator to first knot time >= this time
    // seems that find_if_not ends up being faster than lower_bound in my tests
    // timesIter = std::lower_bound(timesIter, xs_.end(), times[i]);
    const double thisTime = times[i];
    timesIter =
        std::find_if_not(timesIter, xs_.end(),
                         [thisTime](double knotX) { return knotX < thisTime; });

    // stop if we found a time beyond the end of our spline region
    if (timesIter == xs_.end()) {
      firstInvalidIndex = i;
      break;
    }

    const int firstNotLessIndex = std::distance(xs_.begin(), timesIter);
    // the spline time region associated with this time is
    // indexed by one less than the index of first knot w/ time >= this time
    regionIndices[i] = firstNotLessIndex - 1;

    // check if this is the first time inside our spline range
    if (firstValidIndex == regionIndices.size() && firstNotLessIndex > 0) {
      firstValidIndex = i;
    }
  }

  // initialize output matrix with zeros
  Matrix output = Matrix::Zero(3, times.size());

  // evaluate spline and derivatives
  for (unsigned int col = firstValidIndex; col < firstInvalidIndex; ++col) {
    const unsigned int regionIndex = regionIndices[col];
    const double t = times[col] - xs_[regionIndex];

    // calculate powers of t
    std::array<double, 4> tPowers = {{1}};
    for (unsigned int i = 1; i < 4; ++i) {
      tPowers[i] = t * tPowers[i - 1];
    }

    // spline value
    for (unsigned int i = 0; i < 4; ++i) {
      output(0, col) += coeffs_(regionIndex, i) * tPowers[i];
    }

    // first derivative
    for (unsigned int i = 1; i < 4; ++i) {
      tPowers[i - 1] *= i;
      output(1, col) += coeffs_(regionIndex, i) * tPowers[i - 1];
    }

    // second derivative
    for (unsigned int i = 2; i < 4; ++i) {
      tPowers[i - 2] *= i;
      output(2, col) += coeffs_(regionIndex, i) * tPowers[i - 2];
    }
  }

  return output;
}

// generate spline coefficients from set of knots and boundary condition type
// using tridiagonal matrix approach
gm2calo::CubicSpline::Matrix gm2calo::CubicSpline::getCubicSplineCoefficients(
    const Knots& inputKnots, BoundaryType b) {
  const unsigned int nKnots = inputKnots.xs.size();
  std::vector<double> dxvec(inputKnots.xs.size());
  std::adjacent_difference(inputKnots.xs.begin(), inputKnots.xs.end(),
                           dxvec.begin());
  std::vector<double> dyvec(inputKnots.ys.size());
  std::adjacent_difference(inputKnots.ys.begin(), inputKnots.ys.end(),
                           dyvec.begin());

  // first element produced by adjacent difference is just xs[0], ys[0]
  // we don't want that
  auto dxs = dxvec.begin() + 1;
  auto dys = dyvec.begin() + 1;

  // store tridiagonal system as N x 4 matrix containing
  // coefficients for b_i-1, b_i, b_i+1, and RHS of eqn
  constexpr unsigned int nSystemCols = 4;
  Matrix systemOfEqn = Matrix::Zero(nKnots, nSystemCols);
  if (b == BoundaryType::first) {
    // first derivative boundary condition
    systemOfEqn(0, 1) = systemOfEqn(systemOfEqn.rows() - 1, 1) = 1;
  } else {
    // second derivative boundary condition
    std::array<double, nSystemCols> firstRow = {
        {0., 2., 1., 3 * dys[0] / dxs[0]}};
    std::copy(firstRow.begin(), firstRow.end(), systemOfEqn.getRowData(0));

    // dys, dxs are length nKnots - 1
    // last elements are at index [nKnots - 2]
    std::array<double, nSystemCols> lastRow = {
        {1., 2., 0., 3 * dys[nKnots - 2] / dxs[nKnots - 2]}};
    std::copy(lastRow.begin(), lastRow.end(),
              systemOfEqn.getRowData(nKnots - 1));
  }

  // fill internal rows of the system of equations
  for (unsigned int i = 1; i < systemOfEqn.rows() - 1; ++i) {
    systemOfEqn(i, 0) = 1;
    systemOfEqn(i, 1) = 2 * (1 + dxs[i - 1] / dxs[i]);
    systemOfEqn(i, 2) = dxs[i - 1] / dxs[i];
    systemOfEqn(i, 3) =
        3 * (dys[i - 1] / dxs[i - 1] + dxs[i - 1] * dys[i] / (dxs[i] * dxs[i]));
  }

  // solve system of eqn using tridiagonal matrix algorithm
  //
  // first step: modify coefficients
  systemOfEqn(0, 2) /= systemOfEqn(0, 1);
  systemOfEqn(0, 3) /= systemOfEqn(0, 1);
  for (unsigned int i = 1; i < systemOfEqn.rows(); ++i) {
    double denom =
        systemOfEqn(i, 1) - systemOfEqn(i, 0) * systemOfEqn(i - 1, 2);
    systemOfEqn(i, 2) /= denom;
    systemOfEqn(i, 3) =
        (systemOfEqn(i, 3) - systemOfEqn(i, 0) * systemOfEqn(i - 1, 3)) / denom;
  }

  // second step: obtain solutions with back substitution
  for (int i = systemOfEqn.rows() - 2; i >= 0; --i) {
    systemOfEqn(i, 3) -= systemOfEqn(i, 2) * systemOfEqn(i + 1, 3);
  }

  // build and return coefficients matrix
  Matrix coefficients(nKnots - 1, 4);

  // fill a_i, b_i, c_i, and d_i one row at a time
  for (unsigned int i = 0; i < coefficients.rows(); ++i) {
    // a_i == y_i
    coefficients(i, 0) = inputKnots.ys[i];

    // we just solved for b_i
    // b_i is systemOfEqn(i, 3)
    // b_(i+1) is systemOfEqn(i+1, 3)
    coefficients(i, 1) = systemOfEqn(i, 3);

    // solve for c_i
    coefficients(i, 2) =
        1 / dxs[i] *
        (3 * dys[i] / dxs[i] - 2 * systemOfEqn(i, 3) - systemOfEqn(i + 1, 3));

    // solve for d_i
    coefficients(i, 3) = 1.0 / (3 * dxs[i]) *
                         ((systemOfEqn(i + 1, 3) - systemOfEqn(i, 3)) / dxs[i] -
                          2 * coefficients(i, 2));
  }

  return coefficients;
}

void gm2calo::CubicSpline::writeBinaryFile(const std::string& fileName) const {
  std::ofstream binaryOut(fileName, std::ios::binary);

  // write number of knots first
  unsigned int size = xs_.size();
  binaryOut.write((char*)&size, sizeof(unsigned int));
  // write knot x positions
  binaryOut.write((char*)xs_.data(), size * sizeof(double));
  // write matrix of coefficients
  binaryOut.write((char*)coeffs_.data(), coeffs_.size() * sizeof(double));
}

gm2calo::CubicSpline::CubicSpline(const std::string& fileName) {
  std::ifstream binaryIn(fileName, std::ios::binary);

  // get size of x vectors
  unsigned int size;
  binaryIn.read((char*)&size, sizeof(size));

  // read xs
  xs_.resize(size);
  binaryIn.read((char*)xs_.data(), sizeof(double) * size);

  // read coefficients matrix
  coeffs_ = Matrix(size - 1, 4);
  binaryIn.read((char*)coeffs_.data(), sizeof(double) * (size - 1) * 4);
}
