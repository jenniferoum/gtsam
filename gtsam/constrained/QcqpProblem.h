/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    QcqpProblem.h
 * @brief   QCQP represented as a constrained optimization problem.
 * @author  Frank Dellaert
 */

#pragma once

#include <gtsam/base/SymmetricBlockMatrix.h>
#include <gtsam/constrained/ConstrainedOptProblem.h>
#include <gtsam/constrained/LinearConstraint.h>
#include <gtsam/constrained/QuadraticConstraint.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

namespace gtsam {

/**
 * Pure quadratic cost factor evaluating row-space matrix quadratic forms.
 *
 * For matrix values X_i in R^{r_i x d}, this evaluates
 * 0.5 * sum_ij trace(X_i' Q_ij X_j), where Q_ij is the r_i-by-r_j block in
 * Q. The leading 0.5 follows GTSAM's factor-error convention; if a QCQP paper
 * writes sum_ij trace(X_i' Q_ij X_j), pass 2*Q to represent the same objective
 * value in this factor. Vector QCQPs are represented as one-column matrices.
 */
class GTSAM_EXPORT QcqpCost : public NonlinearFactor {
 public:
  using Base = NonlinearFactor;
  using This = QcqpCost;
  using shared_ptr = std::shared_ptr<This>;

  /** Default constructor for I/O. */
  QcqpCost() = default;

  /** Construct a quadratic cost from keys and a symmetric block matrix. */
  QcqpCost(const KeyVector& keys, const SymmetricBlockMatrix& Q);

  /// Symmetric quadratic matrix.
  const SymmetricBlockMatrix& Q() const { return Q_; }

  /** Print the factor for debugging. */
  void print(const std::string& s = "",
             const KeyFormatter& formatter = DefaultKeyFormatter) const override;

  /** Check equality up to a tolerance. */
  bool equals(const NonlinearFactor& other, double tol = 1e-9) const override;

  /** Evaluate 0.5 * sum_ij trace(X_i' Q_ij X_j). */
  double error(const Values& values) const override;

  /** Return scalar cost dimension. */
  size_t dim() const override { return 1; }

  /** Return an exact Hessian factor around the current matrix values. */
  std::shared_ptr<GaussianFactor> linearize(const Values& values) const override;

  /** Return a deep copy of this factor. */
  NonlinearFactor::shared_ptr clone() const override {
    return NonlinearFactor::shared_ptr(new This(*this));
  }

 private:
  SymmetricBlockMatrix Q_;
};

/**
 * Thin constrained optimization problem for QCQPs over matrix variables.
 */
class GTSAM_EXPORT QcqpProblem : public ConstrainedOptProblem {
 public:
  using Base = ConstrainedOptProblem;
  using This = QcqpProblem;
  using shared_ptr = std::shared_ptr<This>;

  /** Default constructor creates an empty QCQP problem. */
  QcqpProblem() = default;

  /** Construct from QCQP cost factors and equality constraints. */
  QcqpProblem(const NonlinearFactorGraph& costs,
              const NonlinearEqualityConstraints& eqConstraints)
      : Base(costs, eqConstraints, NonlinearInequalityConstraints()) {}

  /** Construct from QCQP cost factors and equality/inequality constraints. */
  QcqpProblem(const NonlinearFactorGraph& costs,
              const NonlinearEqualityConstraints& eqConstraints,
              const NonlinearInequalityConstraints& ineqConstraints)
      : Base(costs, eqConstraints, ineqConstraints) {}

  /** Add a QCQP cost. */
  void addCost(const QcqpCost& cost) { costs_.emplace_shared<QcqpCost>(cost); }

  /** Add a linear constraint. */
  void addConstraint(const LinearConstraint& constraint);

  /** Add a quadratic constraint. */
  void addConstraint(const QuadraticConstraint& constraint);
};

}  // namespace gtsam
