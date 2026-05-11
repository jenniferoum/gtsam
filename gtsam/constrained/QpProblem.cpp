/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    QpProblem.cpp
 * @brief   Quadratic programming problem over Vector and Matrix values.
 * @author  Frank Dellaert
 */

#include <gtsam/constrained/ActiveSetSolver.h>
#include <gtsam/constrained/QpProblem.h>

namespace gtsam {
namespace {

/* ************************************************************************* */
ActiveSetSolverParams::shared_ptr DenseQpParams() {
  auto params = std::make_shared<ActiveSetSolverParams>();
  params->qpSubproblemSolver = ActiveSetSolverParams::QpSubproblemSolver::Dense;
  return params;
}

}  // namespace

/* ************************************************************************* */
void QpProblem::addConstraint(const LinearConstraint& constraint) {
  if (constraint.isEquality()) {
    eqConstraints_.push_back(constraint.createEqualityFactor());
  } else {
    ineqConstraints_.push_back(constraint.createInequalityFactor());
  }
}

/* ************************************************************************* */
GaussianFactorGraph QpProblem::linearizedCosts(const Values& values) const {
  GaussianFactorGraph graph;
  for (const auto& cost : costs_) {
    graph.push_back(cost->linearize(values));
  }
  return graph;
}

/* ************************************************************************* */
Values QpProblem::optimize(const Values& initialValues,
                           QpSolverType solverType) const {
  if (solverType == QpSolverType::Dense) {
    ActiveSetSolver solver(*this, DenseQpParams());
    return solver.optimize(initialValues);
  }
  ActiveSetSolver solver(*this);
  return solver.optimize(initialValues);
}

/* ************************************************************************* */
Values QpProblem::optimize(QpSolverType solverType) const {
  if (solverType == QpSolverType::Dense) {
    ActiveSetSolver solver(*this, DenseQpParams());
    return solver.optimize();
  }
  ActiveSetSolver solver(*this);
  return solver.optimize();
}

/* ************************************************************************* */
Values QpProblem::optimizeActiveSet(
    const Values& initialValues,
    std::shared_ptr<ActiveSetSolverParams> params) const {
  ActiveSetSolver solver(*this, params);
  return solver.optimize(initialValues);
}

/* ************************************************************************* */
Values QpProblem::optimizeActiveSet(
    std::shared_ptr<ActiveSetSolverParams> params) const {
  ActiveSetSolver solver(*this, params);
  return solver.optimize();
}

}  // namespace gtsam
