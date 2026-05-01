/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    WNOALevenbergMarquardtOptimizer.h
 * @brief   A nonlinear optimizer that uses the Levenberg-Marquardt trust-region scheme for WNOAFactorGraphs (derived from standard LevenbergMarquardtOptimizer)
 * @author  Sven Lilge
 * @date    Dec 6, 2012
 */

#pragma once

#include <gtsam/nonlinear/NonlinearOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/nonlinear/WNOAFactorGraph.h>
#include <chrono>

class NonlinearOptimizerMoreOptimizationTest;

namespace gtsam {

/**
 * @brief Levenberg-Marquardt optimizer tuned for WNOA interpolation graphs.
 *
 * `WNOALevenbergMarquardtOptimizer` implements the Levenberg-Marquardt
 * trust-region algorithm and is specialized to operate on
 * `WNOAFactorGraph<PoseType>`. It can leverage interpolation precomputation
 * in the factor graph to reduce redundant work during linearization and
 * error evaluation when many wrapper factors are present.
 *
 * @tparam PoseType Pose group/type used by the underlying WNOA graph
 * (e.g., `Pose2`, `Pose3`).
 */
template<typename PoseType>
class GTSAM_EXPORT WNOALevenbergMarquardtOptimizer: public NonlinearOptimizer {

protected:
  /// Reference to the specialized WNOA factor graph. Must outlive this optimizer.
  const WNOAFactorGraph<PoseType>& wnoa_graph_;

  /// Copied Levenberg-Marquardt parameters used by this optimizer.
  const LevenbergMarquardtParams params_; ///< LM parameters

  /// Time point recording when the optimization started (for logging/timing).
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime_; ///< time when optimization started

  /// Initialize the `startTime_` to the current time.
  void initTime();

public:
  typedef std::shared_ptr<LevenbergMarquardtOptimizer> shared_ptr;

  /// @name Constructors/Destructor
  /// @{

  /**
   * @brief Construct optimizer for a `WNOAFactorGraph` with initial values.
   *
   * @param graph The `WNOAFactorGraph<PoseType>` to optimize.
   * @param initialValues Initial variable assignments for the optimization.
   * @param params Levenberg-Marquardt configuration parameters (optional).
   */
  WNOALevenbergMarquardtOptimizer(const WNOAFactorGraph<PoseType>& graph, const Values& initialValues,
                              const LevenbergMarquardtParams& params = LevenbergMarquardtParams());

  /**
   * @brief Construct optimizer with explicit variable ordering.
   *
   * Variant that accepts an `Ordering` to control elimination/linearization
   * ordering used during factor linearization and solving.
   *
   * @param graph The `WNOAFactorGraph<PoseType>` to optimize.
   * @param initialValues Initial variable assignments.
   * @param ordering Variable ordering to use for elimination/linearization.
   * @param params Levenberg-Marquardt configuration parameters (optional).
   */
  WNOALevenbergMarquardtOptimizer(const WNOAFactorGraph<PoseType>& graph, const Values& initialValues,
                              const Ordering& ordering,
                              const LevenbergMarquardtParams& params = LevenbergMarquardtParams());

  /**
   * @brief Virtual destructor.
   */
  ~WNOALevenbergMarquardtOptimizer() override {
  }

  /// @}

  /// @name Standard interface
  /// @{

  /**
   * @brief Return current LM damping parameter (`lambda`).
   * @return double Current damping value.
   */
  double lambda() const;

  /**
   * @brief Return number of inner LM iterations performed during the last step.
   * @return int Number of inner iterations.
   */
  int getInnerIterations() const;

  /**
   * @brief Print optimizer header and parameters for debugging.
   * @param str Optional prefix string printed before the header.
   */
  void print(const std::string& str = "") const {
    std::cout << str << "LevenbergMarquardtOptimizer" << std::endl;
    this->params_.print("  parameters:\n");
  }

  /// @}

  /// @name Advanced interface
  /// @{

  /**
   * @brief Perform a single LM iteration.
   *
   * Linearizes the current nonlinear graph, builds the damped linear system
   * and attempts a step via the inner LM loop. Returns the linearized
   * `GaussianFactorGraph` produced during this iteration.
   *
   * @return GaussianFactorGraph::shared_ptr Linearized Gaussian factor graph for this iteration.
   */
  GaussianFactorGraph::shared_ptr iterate() override;

  /**
   * @brief Read-only access to the LM parameters used by this optimizer.
   * @return const LevenbergMarquardtParams& Reference to the stored parameters.
   */
  const LevenbergMarquardtParams& params() const {
    return params_;
  }

  /**
   * @brief Append a log entry capturing the current optimizer progress.
   * @param currentError Current scalar error value to write to the log.
   */
  void writeLogFile(double currentError);

  /**
   * @brief Linearize the underlying nonlinear factor graph.
   *
   * This method leverages the efficient linearization routine of WNOAFactor Graph (e.g., leveraging precomputed interpolation batches).
   *
   * @return GaussianFactorGraph::shared_ptr Linearized Gaussian factor graph.
   */
  virtual GaussianFactorGraph::shared_ptr linearize() const;

  /**
   * @brief Build a damped (Levenberg-Marquardt) linear system for testing.
   *
   * Constructs a modified Gaussian factor graph where damping corresponding
   * to a candidate `lambda` is applied (via `sqrtHessianDiagonal`).
   *
   * @param linear Linearized Gaussian factor graph
   * @param sqrtHessianDiagonal Per-variable sqrt of the Hessian diagonal used for damping
   * @return GaussianFactorGraph Damped Gaussian factor graph
   */
  GaussianFactorGraph buildDampedSystem(const GaussianFactorGraph& linear,
                                        const VectorValues& sqrtHessianDiagonal) const;

  /**
   * @brief Inner LM loop: try a candidate lambda, apply or reject the step.
   *
   * Builds and solves the damped system, evaluates the step using the
   * objective, and updates optimizer state based on acceptance criteria.
   *
   * @param linear Linearized Gaussian factor graph
   * @param sqrtHessianDiagonal Per-variable sqrt of the Hessian diagonal
   * @return bool True if the step was accepted (or algorithm terminates), false if rejected.
   */
  bool tryLambda(const GaussianFactorGraph& linear, const VectorValues& sqrtHessianDiagonal);

  /// @}

protected:

  /** Access the parameters (base class version) */
  const NonlinearOptimizerParams& _params() const override {
    return params_;
  }
};

}
