/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    QcqpProblem.cpp
 * @brief   QCQP cost and constraint implementations.
 * @author  Frank Dellaert
 */

#include <gtsam/constrained/QcqpProblem.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace gtsam {
namespace {

/* ************************************************************************* */
std::vector<Matrix> MatrixValues(const Values& values, const KeyVector& keys,
                                 const SymmetricBlockMatrix& Q,
                                 size_t* columnDim) {
  std::vector<Matrix> matrices;
  matrices.reserve(keys.size());

  size_t D = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    const Matrix Xi = values.at<Matrix>(keys[i]);
    const size_t expectedRows = static_cast<size_t>(Q.getDim(i));
    if (static_cast<size_t>(Xi.rows()) != expectedRows) {
      throw std::invalid_argument(
          "QcqpCost: matrix row count does not match Q block size.");
    }
    if (i == 0) {
      D = static_cast<size_t>(Xi.cols());
      if (D == 0) {
        throw std::invalid_argument(
            "QcqpCost: matrices must have at least one column.");
      }
    } else if (static_cast<size_t>(Xi.cols()) != D) {
      throw std::invalid_argument(
          "QcqpCost: matrices must share one column dimension.");
    }
    matrices.push_back(Xi);
  }

  if (columnDim) {
    *columnDim = D;
  }
  return matrices;
}

/* ************************************************************************* */
Matrix ExpandedBlock(const Matrix& block, size_t D) {
  const DenseIndex rows = block.rows();
  const DenseIndex cols = block.cols();
  Matrix expanded = Matrix::Zero(rows * D, cols * D);
  for (size_t d = 0; d < D; ++d) {
    expanded.block(d * rows, d * cols, rows, cols) = block;
  }
  return expanded;
}

/* ************************************************************************* */
Vector StackedMatrixValues(const std::vector<Matrix>& matrices) {
  size_t totalDim = 0;
  for (const Matrix& X : matrices) {
    totalDim += static_cast<size_t>(X.size());
  }

  Vector x(totalDim);
  size_t row = 0;
  for (const Matrix& X : matrices) {
    const Eigen::Map<const Vector> vectorized(X.data(), X.size());
    x.segment(row, X.size()) = vectorized;
    row += static_cast<size_t>(X.size());
  }
  return x;
}

}  // namespace

/* ************************************************************************* */
QcqpCost::QcqpCost(const KeyVector& keys, const SymmetricBlockMatrix& Q)
    : Base(keys), Q_(Q) {
  if (keys.size() != static_cast<size_t>(Q_.nBlocks())) {
    throw std::invalid_argument(
        "QcqpCost: number of keys must match Q block count.");
  }
}

/* ************************************************************************* */
void QcqpCost::print(const std::string& s,
                     const KeyFormatter& formatter) const {
  Base::print(s + "QcqpCost", formatter);
  gtsam::print(Matrix(Q_.selfadjointView()), "  Q: ");
}

/* ************************************************************************* */
bool QcqpCost::equals(const NonlinearFactor& other, double tol) const {
  const auto* expected = dynamic_cast<const QcqpCost*>(&other);
  return expected != nullptr && Base::equals(other, tol) &&
         equal_with_abs_tol(Matrix(Q_.selfadjointView()),
                            Matrix(expected->Q_.selfadjointView()), tol);
}

/* ************************************************************************* */
double QcqpCost::error(const Values& values) const {
  const std::vector<Matrix> matrices =
      MatrixValues(values, keys(), Q_, nullptr);

  double total = 0.0;
  for (size_t i = 0; i < keys().size(); ++i) {
    for (size_t j = 0; j < keys().size(); ++j) {
      total += (matrices[i].transpose() * Q_.block(i, j) * matrices[j]).trace();
    }
  }
  return 0.5 * total;
}

/* ************************************************************************* */
std::shared_ptr<GaussianFactor> QcqpCost::linearize(
    const Values& values) const {
  size_t D = 0;
  const std::vector<Matrix> matrices = MatrixValues(values, keys(), Q_, &D);
  const Vector x0 = StackedMatrixValues(matrices);

  size_t totalDim = 0;
  for (size_t i = 0; i < keys().size(); ++i) {
    totalDim += static_cast<size_t>(Q_.getDim(i)) * D;
  }

  Matrix expandedQ = Matrix::Zero(totalDim, totalDim);
  size_t row = 0;
  for (size_t i = 0; i < keys().size(); ++i) {
    const size_t rowDim = static_cast<size_t>(Q_.getDim(i)) * D;
    size_t col = 0;
    for (size_t j = 0; j < keys().size(); ++j) {
      const size_t colDim = static_cast<size_t>(Q_.getDim(j)) * D;
      expandedQ.block(row, col, rowDim, colDim) =
          ExpandedBlock(Q_.block(i, j), D);
      col += colDim;
    }
    row += rowDim;
  }

  const Vector Qx0 = expandedQ * x0;
  const Vector g = -Qx0;
  const double f = x0.dot(Qx0);

  std::vector<Matrix> Gs;
  Gs.reserve(keys().size() * (keys().size() + 1) / 2);
  row = 0;
  for (size_t i = 0; i < keys().size(); ++i) {
    const size_t rowDim = static_cast<size_t>(Q_.getDim(i)) * D;
    size_t col = row;
    for (size_t j = i; j < keys().size(); ++j) {
      const size_t colDim = static_cast<size_t>(Q_.getDim(j)) * D;
      Gs.push_back(expandedQ.block(row, col, rowDim, colDim));
      col += colDim;
    }
    row += rowDim;
  }

  std::vector<Vector> gs;
  gs.reserve(keys().size());
  row = 0;
  for (size_t i = 0; i < keys().size(); ++i) {
    const size_t dim = static_cast<size_t>(Q_.getDim(i)) * D;
    gs.push_back(g.segment(row, dim));
    row += dim;
  }

  return std::make_shared<HessianFactor>(keys(), Gs, gs, f);
}

/* ************************************************************************* */
void QcqpProblem::addConstraint(const LinearConstraint& constraint) {
  if (constraint.isEquality()) {
    eqConstraints_.push_back(constraint.createEqualityFactor());
  } else {
    ineqConstraints_.push_back(constraint.createInequalityFactor());
  }
}

/* ************************************************************************* */
void QcqpProblem::addConstraint(const QuadraticConstraint& constraint) {
  if (constraint.isEquality()) {
    eqConstraints_.push_back(constraint.createEqualityFactor());
  } else {
    ineqConstraints_.push_back(constraint.createInequalityFactor());
  }
}

}  // namespace gtsam
