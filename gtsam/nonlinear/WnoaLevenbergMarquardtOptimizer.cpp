/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */
/**
 * @file    WnoaLevenbergMarquardtOptimizer.cpp
 * @brief   A nonlinear optimizer that uses the Levenberg-Marquardt trust-region
 * scheme for WNOAFactorGraphs (derived from standard
 * LevenbergMarquardtOptimizer)
 * @author  Sven Lilge
 * @date    Dec 6, 2012
 */

#include <gtsam/base/Vector.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/WnoaLevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/internal/LevenbergMarquardtState.h>
#if GTSAM_USE_BOOST_FEATURES
#include <gtsam/base/timing.h>
#endif

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

using namespace std;

namespace gtsam {

typedef internal::LevenbergMarquardtState State;

/* ************************************************************************* */
template <typename PoseType>
WNOALevenbergMarquardtOptimizer<PoseType>::WNOALevenbergMarquardtOptimizer(
    const WNOAFactorGraph<PoseType>& graph, const Values& initialValues,
    const LevenbergMarquardtParams& params)
    : NonlinearOptimizer(
          graph, std::unique_ptr<State>(
                     new State(initialValues, graph.error(initialValues),
                               params.lambdaInitial, params.lambdaFactor))),
      wnoa_graph_(graph),
      params_(LevenbergMarquardtParams::EnsureHasOrdering(params, graph)) {}

template <typename PoseType>
WNOALevenbergMarquardtOptimizer<PoseType>::WNOALevenbergMarquardtOptimizer(
    const WNOAFactorGraph<PoseType>& graph, const Values& initialValues,
    const Ordering& ordering, const LevenbergMarquardtParams& params)
    : NonlinearOptimizer(
          graph, std::unique_ptr<State>(
                     new State(initialValues, graph.error(initialValues),
                               params.lambdaInitial, params.lambdaFactor))),
      wnoa_graph_(graph),
      params_(LevenbergMarquardtParams::ReplaceOrdering(params, ordering)) {}

/* ************************************************************************* */
template <typename PoseType>
void WNOALevenbergMarquardtOptimizer<PoseType>::initTime() {
  // use chrono to measure time in microseconds
  startTime_ = std::chrono::high_resolution_clock::now();
}

/* ************************************************************************* */
template <typename PoseType>
double WNOALevenbergMarquardtOptimizer<PoseType>::lambda() const {
  auto currentState = static_cast<const State*>(state_.get());
  return currentState->lambda;
}

/* ************************************************************************* */
template <typename PoseType>
int WNOALevenbergMarquardtOptimizer<PoseType>::getInnerIterations() const {
  auto currentState = static_cast<const State*>(state_.get());
  return currentState->totalNumberInnerIterations;
}

/* ************************************************************************* */
template <typename PoseType>
GaussianFactorGraph::shared_ptr
WNOALevenbergMarquardtOptimizer<PoseType>::linearize() const {
  return wnoa_graph_.linearize(state_->values);
}

/* ************************************************************************* */
template <typename PoseType>
GaussianFactorGraph
WNOALevenbergMarquardtOptimizer<PoseType>::buildDampedSystem(
    const GaussianFactorGraph& linear,
    const VectorValues& sqrtHessianDiagonal) const {
  gttic(damp);
  auto currentState = static_cast<const State*>(state_.get());

  if (params_.verbosityLM >= LevenbergMarquardtParams::DAMPED)
    std::cout << "building damped system with lambda " << currentState->lambda
              << std::endl;

  if (params_.dampingParams.diagonalDamping)
    return currentState->buildDampedSystem(linear, sqrtHessianDiagonal);
  else
    return currentState->buildDampedSystem(linear);
}

/* ************************************************************************* */
// Log current error/lambda to file
template <typename PoseType>
inline void WNOALevenbergMarquardtOptimizer<PoseType>::writeLogFile(
    double currentError) {
  auto currentState = static_cast<const State*>(state_.get());

  if (!params_.logFile.empty()) {
    ofstream os(params_.logFile.c_str(), ios::app);
    // use chrono to measure time in microseconds
    auto currentTime = std::chrono::high_resolution_clock::now();
    // Get the time spent in seconds and print it
    double timeSpent = std::chrono::duration_cast<std::chrono::microseconds>(
                           currentTime - startTime_)
                           .count() /
                       1e6;
    os << /*inner iterations*/ currentState->totalNumberInnerIterations << ","
       << timeSpent << "," << /*current error*/ currentError << ","
       << currentState->lambda << ","
       << /*outer iterations*/ currentState->iterations << endl;
  }
}

/* ************************************************************************* */
template <typename PoseType>
bool WNOALevenbergMarquardtOptimizer<PoseType>::tryLambda(
    const GaussianFactorGraph& linear,
    const VectorValues& sqrtHessianDiagonal) {
  auto currentState = static_cast<const State*>(state_.get());
  bool verbose = (params_.verbosityLM >= LevenbergMarquardtParams::TRYLAMBDA);

#if GTSAM_USE_BOOST_FEATURES
#ifdef GTSAM_USING_NEW_BOOST_TIMERS
  boost::timer::cpu_timer lamda_iteration_timer;
  lamda_iteration_timer.start();
#else
  boost::timer lamda_iteration_timer;
  lamda_iteration_timer.restart();
#endif
#else
  auto start = std::chrono::high_resolution_clock::now();
#endif

  if (verbose) cout << "trying lambda = " << currentState->lambda << endl;

  // Build damped system for this lambda (adds prior factors that make it like
  // gradient descent)
  auto dampedSystem = buildDampedSystem(linear, sqrtHessianDiagonal);

  // Try solving
  double modelFidelity = 0.0;
  bool step_is_successful = false;
  bool stopSearchingLambda = false;
  double newError = numeric_limits<double>::infinity();
  double costChange = 0.0;
  Values newValues;
  VectorValues delta;

  bool systemSolvedSuccessfully;
  try {
    // ============ Solve is where most computation happens !! =================
    delta = solve(dampedSystem, params_);
    systemSolvedSuccessfully = true;
  } catch (const IndeterminantLinearSystemException&) {
    systemSolvedSuccessfully = false;
  }

  if (systemSolvedSuccessfully) {
    if (verbose) cout << "linear delta norm = " << delta.norm() << endl;
    if (params_.verbosityLM >= LevenbergMarquardtParams::TRYDELTA)
      delta.print("delta");

    // Compute the old linearized error as it is not the same
    // as the nonlinear error when robust noise models are used.
    double oldLinearizedError = linear.error(VectorValues::Zero(delta));
    double newlinearizedError = linear.error(delta);

    // cost change in the linearized system (old - new)
    double linearizedCostChange = oldLinearizedError - newlinearizedError;
    if (verbose)
      cout << "newlinearizedError = " << newlinearizedError
           << "  linearizedCostChange = " << linearizedCostChange << endl;

    if (linearizedCostChange >= 0) {  // step is valid
      // update values
      gttic(retract);
      // ============ This is where the solution is updated ====================
      newValues = currentState->values.retract(delta);
      // =======================================================================
      gttoc(retract);

      // compute new error
      gttic(compute_error);
      if (verbose) cout << "calculating error:" << endl;
      newError = wnoa_graph_.error(newValues);
      gttoc(compute_error);

      if (verbose)
        cout << "old error (" << currentState->error
             << ") new (tentative) error (" << newError << ")" << endl;

      // cost change in the original, nonlinear system (old - new)
      costChange = currentState->error - newError;

      if (linearizedCostChange >
          std::numeric_limits<double>::epsilon() * oldLinearizedError) {
        // the (linear) error has to decrease to satisfy this condition
        // fidelity of linearized model VS original system between
        modelFidelity = costChange / linearizedCostChange;
        // if we decrease the error in the nonlinear system and modelFidelity is
        // above threshold
        step_is_successful = modelFidelity > params_.minModelFidelity;
        if (verbose) cout << "modelFidelity: " << modelFidelity << endl;
      }  // else we consider the step non successful and we either increase
         // lambda or stop if error change is small

      double minAbsoluteTolerance =
          params_.relativeErrorTol * currentState->error;
      // if the change is small we terminate
      if (std::abs(costChange) < minAbsoluteTolerance) {
        if (verbose)
          cout << "abs(costChange)=" << std::abs(costChange)
               << "  minAbsoluteTolerance=" << minAbsoluteTolerance
               << " (relativeErrorTol=" << params_.relativeErrorTol << ")"
               << endl;
        stopSearchingLambda = true;
      }
    }
  }  // if (systemSolvedSuccessfully)

  if (params_.verbosityLM == LevenbergMarquardtParams::SUMMARY) {
#if GTSAM_USE_BOOST_FEATURES
// do timing
#ifdef GTSAM_USING_NEW_BOOST_TIMERS
    double iterationTime = 1e-9 * lamda_iteration_timer.elapsed().wall;
#else
    double iterationTime = lamda_iteration_timer.elapsed();
#endif
#else
    auto end = std::chrono::high_resolution_clock::now();
    double iterationTime =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1e6;
#endif
    if (currentState->iterations == 0) {
      cout << "iter      cost      cost_change    lambda  success iter_time"
           << endl;
    }
    cout << setw(4) << currentState->iterations << " " << setw(12) << newError
         << " " << setw(12) << setprecision(2) << costChange << " " << setw(10)
         << setprecision(2) << currentState->lambda << " " << setw(6)
         << systemSolvedSuccessfully << " " << setw(10) << setprecision(2)
         << iterationTime << endl;
  }
  if (step_is_successful) {
    // we have successfully decreased the cost and we have good modelFidelity
    // NOTE(frank): As we return immediately after this, we move the newValues
    // TODO(frank): make Values actually support move. Does not seem to happen
    // now.
    state_ = currentState->decreaseLambda(params_, modelFidelity,
                                          std::move(newValues), newError);
    return true;
  } else if (!stopSearchingLambda) {  // we failed to solved the system or had
                                      // no decrease in cost
    if (verbose) cout << "increasing lambda" << endl;
    State* modifiedState = static_cast<State*>(state_.get());
    modifiedState->increaseLambda(
        params_);  // TODO(frank): make this functional with Values move

    // check if lambda is too big
    if (modifiedState->lambda >= params_.lambdaUpperBound) {
      if (params_.verbosity >= NonlinearOptimizerParams::TERMINATION ||
          params_.verbosityLM == LevenbergMarquardtParams::SUMMARY)
        cout << "Warning:  Levenberg-Marquardt giving up because "
                "cannot decrease error with maximum lambda"
             << endl;
      return true;
    } else {
      return false;  // only case where we will keep trying
    }
  } else {  // the change in the cost is very small and it is not worth trying
            // bigger lambdas
    if (verbose)
      cout
          << "Levenberg-Marquardt: stopping as relative cost reduction is small"
          << endl;
    return true;
  }
}

/* ************************************************************************* */
template <typename PoseType>
GaussianFactorGraph::shared_ptr
WNOALevenbergMarquardtOptimizer<PoseType>::iterate() {
  auto currentState = static_cast<const State*>(state_.get());

  gttic(LM_iterate);

  // Linearize graph
  if (params_.verbosityLM >= LevenbergMarquardtParams::DAMPED)
    cout << "linearizing = " << endl;
  GaussianFactorGraph::shared_ptr linear = linearize();

  if (currentState->totalNumberInnerIterations == 0) {  // write initial error
    writeLogFile(currentState->error);

    if (params_.verbosityLM == LevenbergMarquardtParams::SUMMARY) {
      cout << "Initial error: " << currentState->error
           << ", values: " << currentState->values.size() << std::endl;
    }
  }

  // Only calculate diagonal of Hessian (expensive) once per outer iteration, if
  // we need it
  VectorValues sqrtHessianDiagonal;
  if (params_.dampingParams.diagonalDamping) {
    sqrtHessianDiagonal = linear->hessianDiagonal();
    for (auto& [key, value] : sqrtHessianDiagonal) {
      value = value.cwiseMax(params_.dampingParams.minDiagonal)
                  .cwiseMin(params_.dampingParams.maxDiagonal)
                  .cwiseSqrt();
    }
  }

  // Keep increasing lambda until we make make progress
  while (!tryLambda(*linear, sqrtHessianDiagonal)) {
    auto newState = static_cast<const State*>(state_.get());
    writeLogFile(newState->error);
  }

  return linear;
}

// Explicit template instantiation to ensure the template gets compiled
template class WNOALevenbergMarquardtOptimizer<Point1>;
template class WNOALevenbergMarquardtOptimizer<Point2>;
template class WNOALevenbergMarquardtOptimizer<Point3>;
template class WNOALevenbergMarquardtOptimizer<Pose2>;
template class WNOALevenbergMarquardtOptimizer<Pose3>;

} /* namespace gtsam */
