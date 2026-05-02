/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  WnoaFactorGraph.h
 *  @brief Factor graph that handles computation of interpolated states for WNOA
 *  @author Sven Lilge
 */

#pragma once

#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/WnoaInterpolator.h>
#include <gtsam/nonlinear/WnoaStateData.h>

#include <array>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace gtsam {

/**
 * @brief Factor graph specialized for WNOA interpolation-aware computation.
 *
 * `WnoaFactorGraph` wraps standard factor graph functionalities with utilities
 * to compute interpolated pose/velocity states under a
 * White-Noise-on-Acceleration (WNOA) motion prior. It stores the mapping from
 * interpolated query times to their left/right bordering estimated states and
 * precomputes interpolation helpers for efficient repeated evaluation.
 *
 * The graph provides optimized linearization and error computation routines
 * that can exploit precomputed interpolation batches to reduce repeated
 * work when evaluating many wrapper factors using the same query times.
 *
 * @tparam PoseType Pose group/type (e.g. `Pose2`, `Pose3`) used by the
 * interpolator.
 */
template <typename PoseType>
class WnoaFactorGraph : public ExpressionFactorGraph {
 private:
  using This = WnoaFactorGraph<PoseType>;
  using Base = ExpressionFactorGraph;
  using VelocityType = typename gtsam::traits<PoseType>::TangentVector;
  static constexpr int dim = traits<PoseType>::dimension;

  // Convenient matrices
  using Matrix2N = Eigen::Matrix<double, 2 * dim, 2 * dim>;
  using MatrixN = Eigen::Matrix<double, dim, dim>;
  using VectorN = Eigen::Matrix<double, dim, 1>;

  // Interpolator class
  const Interpolator<PoseType> interpolator_;

  using LambdaPsiMats = typename Interpolator<PoseType>::LambdaPsiMats;
  using LocalStateVecs = typename Interpolator<PoseType>::LocalStateVecs;
  using StateJacobians = typename Interpolator<PoseType>::StateJacobians;

  // map interpolated state to border states
  std::unordered_map<StateData, std::pair<StateData, StateData>>
      interp_to_borders_map_;
  std::vector<std::pair<StateData, std::pair<StateData, StateData>>>
      interp_to_borders_vec_;
  std::vector<std::pair<StateData, std::shared_ptr<const LambdaPsiMats>>>
      interp_to_LambdaPsi_vec_;

  // Precomputed batches of borders -> indices in interp_to_borders_vec_
  // Each entry contains the border pair and the list of interp indices that
  // share those borders.
  std::vector<std::pair<std::pair<StateData, StateData>, std::vector<size_t>>>
      border_batches_;

  bool fixed_noise_model_ = false;

  std::unordered_set<Key> border_pose_keys_;
  std::unordered_set<Key> border_vel_keys_;

  // Efficient storage for indices of WnoaInterpFactors
  std::unordered_set<size_t> wnoa_interp_factor_indices_;

  /**
   * @brief Evaluate interpolated states from bordering estimated states.
   *
   * Builds a `Values` container containing interpolated pose and velocity
   * entries for every `StateData` known in `interp_to_borders_map_`. When
   * requested, also fills `InterpJacobians` with flattened Jacobian blocks
   * (ordered [LPose,LVel,RPose,RVel]) and `InterpCondCovs` with the
   * conditional covariance for each interpolated state.
   *
   * @param values Outer `Values` providing the bordering estimated states.
   * @param InterpJacobians Optional output pointer populated with flattened
   * jacobian blocks.
   * @param InterpCondCovs Optional output pointer populated with
   * per-interpolated-state covariances.
   * @return Values Container with interpolated pose and velocity entries.
   */
  Values getInterpolatedValues(
      const Values& values,
      std::unordered_map<Key, std::array<Matrix, 4>>* InterpJacobians,
      std::unordered_map<StateData, Matrix2N>* InterpCondCovs = nullptr) const;

 public:
  /**
   * @brief Linearize the graph into a GaussianFactorGraph.
   *
   * This routine produces a linearized Gaussian factor graph evaluated at
   * `linearizationPoint`. It exploits precomputed interpolation data and
   * batching to reduce duplicated interpolation work across wrapper
   * factors.
   *
   * @param linearizationPoint Values at which to linearize the nonlinear graph.
   * @return std::shared_ptr<GaussianFactorGraph> Linearized Gaussian factor
   * graph.
   */
  std::shared_ptr<GaussianFactorGraph> linearize(
      const Values& linearizationPoint) const override;

  /**
   * @brief Compute the unnormalized graph error (sum of factor losses).
   *
   * Computes the scalar error over all factors in the graph. When the
   * graph contains interpolation wrapper factors this method uses the
   * interpolator to evaluate interpolated states as part of the residual
   * computation and can exploit precomputation to improve throughput.
   *
   * @param values Current `Values` used to evaluate the error.
   * @return double Scalar unnormalized error (sum of factor losses).
   */
  double error(const Values& values) const override;

  /// Clone into a shared pointer while preserving WnoaFactorGraph behavior.
  std::shared_ptr<const NonlinearFactorGraph> cloneShared() const override {
    return std::make_shared<WnoaFactorGraph<PoseType>>(*this);
  }

  /**
   * @brief Construct a `WnoaFactorGraph` with interpolation metadata.
   *
   * @param interp_map Mapping from each interpolated `StateData` to its
   *        left/right bordering estimated `StateData`.
   * @param q_psd_diag Diagonal PSD vector for the WNOA interpolator (size must
   * match PoseType dimension).
   * @param fixed_noise_model If true, the graph will not augment measurement
   * noise for interpolation.
   */
  WnoaFactorGraph(
      std::unordered_map<StateData, std::pair<StateData, StateData>> interp_map,
      const Eigen::Vector<double, dim> q_psd_diag,
      bool fixed_noise_model = false);
};

}  // namespace gtsam
