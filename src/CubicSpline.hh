// simple cubic spline class
// can generate coefficients from knots
// or read coefficients out of a binary file
//
// see document doc/cubicSplineDoc.pdf
//
// Aaron Fienberg
// March 2018

#pragma once

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <cassert>

namespace gm2calo {
class CubicSpline {
 public:
  struct Knots {
    explicit Knots(unsigned int nKnots) : xs(nKnots), ys(nKnots) {}

    std::vector<double> xs;
    std::vector<double> ys;
  };

  // simple Matrix class for convenience
  // stored in a vector, row-major order
  class Matrix {
   public:
    Matrix() = default;

    Matrix(unsigned int nRows, std::size_t nCols)
        : nCols_(nCols), data_(nRows * nCols) {}

    Matrix(unsigned int nRows, std::size_t nCols, double fillVal)
        : nCols_(nCols), data_(nRows * nCols, fillVal) {}

    double& operator()(std::size_t rowNum, std::size_t colNum) {
      return data_[rowNum * cols() + colNum];
    }
    double operator()(std::size_t rowNum, std::size_t colNum) const {
      return data_[rowNum * cols() + colNum];
    }

    double* getRowData(std::size_t rowNum) {
      return data_.data() + rowNum * cols();
    }
    const double* getRowData(std::size_t rowNum) const {
      return data_.data() + rowNum * cols();
    };

    double* data() { return getRowData(0); }
    const double* data() const { return getRowData(0); }

    void fill(double fillVal) {
      std::fill(data_.begin(), data_.end(), fillVal);
    }

    std::size_t cols() const { return nCols_; }
    std::size_t rows() const { return data_.size() / cols(); }
    std::size_t size() const { return data_.size(); }

    static Matrix Zero(std::size_t nRows, std::size_t nCols) {
      return Matrix(nRows, nCols, 0.0);
    }

   private:
    unsigned int nCols_;
    std::vector<double> data_;
  };

  // whether the first or second derivatives should be set to zero at
  // boundaries
  enum class BoundaryType { first, second };

  CubicSpline() = default;

  CubicSpline(const Knots& k,
              BoundaryType boundaryCondition = BoundaryType::first)
      : xs_{k.xs}, coeffs_{getCubicSplineCoefficients(k, boundaryCondition)} {}

  // read Knots and coefficients from a binary file
  explicit CubicSpline(const std::string& fileName);

  CubicSpline& operator=(const CubicSpline& other) = default;

  // evaluate the spline at one time
  double operator()(double x) const;

  // evaluate the spline and its first and second derivatives at times
  // return value is a 3xN matrix where N is the length of the passed in vector
  // first row is the spline evaluated at the passed in times
  // second the first derivative at those times
  // third the second derivative at those times
  // output is always 0 for times outside the spline's range of validity
  // note: times must be in increasing order
  Matrix evalWithDerivatives(const std::vector<double>& times) const;

  const Matrix& getCoefficients() const { return coeffs_; }

  // get length of validity region
  double getLength() const {
    return xs_.size() > 1 ? xs_.back() - xs_.front() : 0;
  }

  // get value of first knot
  double getXMin() const { return xs_.size() > 0 ? xs_.front() : 0; }

  // get value of last knot
  double getXMax() const { return xs_.size() > 0 ? xs_.back() : 0; }

  // get knots vector
  const std::vector<double>& getKnots() const { return xs_; }

  // write spline coefficients into a binary file
  void writeBinaryFile(const std::string& fileName) const;

 private:
  static Matrix getCubicSplineCoefficients(const Knots& inputKnots,
                                           BoundaryType b);

  // knot positions in x
  std::vector<double> xs_;
  // polynomial coefficients
  Matrix coeffs_;
};
}
