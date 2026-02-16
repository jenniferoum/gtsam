/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    WNOAFactor.h
 * @brief   White-Noise-On-Acceleration (WNOA) motion prior factor between two
 * states (pose and velocity at times t_k and t_{k+1})
 * @author  Connor Holmes
 * @author  Sven Lilge
 */

#include <gtsam/base/Lie.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/VectorSpace.h>
#include <gtsam/geometry/Point1.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/WNOAStateData.h>

#include <cassert>

#pragma once
using namespace std;

namespace gtsam {

/**
 * @brief WNOA (White Noise on Acceleration) motion prior factor.
 *
 * This factor implements the WNOA motion prior between two
 * states (pose and velocity at times t_k and t_{k+1}). It provides
 * residuals and Jacobians consistent with the WNOA continuous-time
 * prior when discretized over the timestep `delta_t`.
 *
 * Template parameter `Pose` is expected to be a GTSAM pose type (e.g.
 * `Pose2`, `Pose3`) or a vector-space pose; the factor supports both Lie
 * group and vector-space pose representations.
 *
 * The factor's ordering of keys is: pose_k, vel_k, pose_kp1, vel_kp1.
 *
 * @tparam Pose Pose group/type (Pose2, Pose3, etc.)
 */
template <class Pose>
class WNOAMotionFactor
    : public NoiseModelFactorN<Pose, typename traits<Pose>::TangentVector, Pose,
                               typename traits<Pose>::TangentVector> {
  // Check that Pose type is a testable Lie group
  GTSAM_CONCEPT_ASSERT(IsTestable<Pose>);
  // We currently support vector spaces and Lie groups
  static_assert(std::is_same_v<typename traits<Pose>::structure_category,
                               lie_group_tag> ||
                    std::is_same_v<typename traits<Pose>::structure_category,
                                   vector_space_tag>,
                "Pose type must be either a Lie group or vector space");

  GTSAM_CONCEPT_ASSERT(IsLieGroup<Pose>);

 public:
  static constexpr int dim = traits<Pose>::dimension;

 private:
  // Convenient typedefs
  using Velocity = typename gtsam::traits<Pose>::TangentVector;
  using MatrixN = Eigen::Matrix<double, dim, dim>;
  using VectorN = Eigen::Matrix<double, dim, 1>;
  using Matrix2N = Eigen::Matrix<double, 2 * dim, 2 * dim>;
  using Vector2N = Eigen::Matrix<double, 2 * dim, 1>;
  using MatrixNx2N = Eigen::Matrix<double, dim, 2 * dim>;
  typedef NoiseModelFactorN<Pose, Velocity, Pose, Velocity> Base;
  typedef WNOAMotionFactor This;
  // Time between the two states
  double delta_t_;

  inline static const MatrixN Identity = MatrixN::Identity();
  inline static const MatrixN Zero = MatrixN::Zero();

 public:
  // Provide access to the Matrix& version of evaluateError:
  using Base::evaluateError;

  /**
   * @brief Construct a WNOA motion factor from two `StateData` entries.
   *
   * This constructor builds a factor connecting the pose and velocity keys
   * contained in `state_k` and `state_kp1`. The internal noise model is
   * constructed from the provided diagonal PSD vector `Q` and the timestep
   * computed from the two state timestamps.
   *
   * @param state_k StateData for time t_k (provides keys and timestamp).
   * @param state_kp1 StateData for time t_{k+1} (provides keys and timestamp).
   * @param Q Diagonal power spectral density vector used to form the process
   * noise.
   */
  WNOAMotionFactor(const StateData& state_k, const StateData& state_kp1,
                   const VectorN& Q)
      : Base() {
    // define keys
    this->keys_ = {state_k.pose, state_k.vel, state_kp1.pose, state_kp1.vel};
    // define timestep
    this->delta_t_ = state_kp1.time - state_k.time;
    assert(this->delta_t_ > 0.0 &&
           "Time difference between input states must be positive.");
    // define noise model
    this->noiseModel_ = This::buildWNOANoiseModel(this->delta_t_, Q);
  }

  /**
   * @brief Construct a WNOA factor given explicit keys and timestep.
   *
   * @param key1 Pose key at t_k
   * @param key2 Velocity key at t_k
   * @param key3 Pose key at t_{k+1}
   * @param key4 Velocity key at t_{k+1}
   * @param delta_t Time interval t_{k+1} - t_k (must be > 0)
   * @param Q Diagonal PSD vector used to form the process noise.
   */
  WNOAMotionFactor(Key key1, Key key2, Key key3, Key key4, const double delta_t,
                   const VectorN& Q)
      : Base(This::buildWNOANoiseModel(delta_t, Q), key1, key2, key3, key4),
        delta_t_(delta_t) {}

  ~WNOAMotionFactor() override {}
  /** implement functions needed for Testable */
  /** print */
  void print(
      const std::string& s = "",
      const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override {
    std::cout << s << "WNOAMotionFactor(" << keyFormatter(this->key1()) << ","
              << keyFormatter(this->key2()) << "," << keyFormatter(this->key3())
              << "," << keyFormatter(this->key4()) << ")\n";
    this->noiseModel_->print("  noise model: ");
  }

  /** equals */
  bool equals(const NonlinearFactor& expected,
              double tol = 1e-9) const override {
    const This* e = dynamic_cast<const This*>(&expected);
    return e != nullptr && Base::equals(*e, tol);
  }

  /**
   * @brief Evaluate the WNOA factor residual and optional Jacobians.
   *
   * Residual is a 2N vector formed from the difference between the
   * discrete relative pose/velocity and the predicted motion under the
   * WNOA prior. When Jacobian pointers are provided, the function fills
   * the corresponding derivative blocks with respect to the four inputs.
   *
   * @param p1 Pose at time t_k
   * @param v1 Velocity at time t_k
   * @param p2 Pose at time t_{k+1}
   * @param v2 Velocity at time t_{k+1}
   * @param Hp1 Optional output Jacobian w.r.t. p1
   * @param Hv1 Optional output Jacobian w.r.t. v1
   * @param Hp2 Optional output Jacobian w.r.t. p2
   * @param Hv2 Optional output Jacobian w.r.t. v2
   * @return Vector Residual vector of size 2*dim
   */
  Vector evaluateError(const Pose& p1, const Velocity& v1, const Pose& p2,
                       const Velocity& v2, OptionalMatrixType Hp1,
                       OptionalMatrixType Hv1, OptionalMatrixType Hp2,
                       OptionalMatrixType Hv2) const override {
    // Local variable for the relative pose in the tangent space
    // Note that p1 = T(t_k), p2 = T(t_{k+1})
    //  compute xi = log(T_k^-1 T_{k+1})^check
    VectorN xi;
    // Jacobians
    MatrixN dxi_dT1;
    MatrixN dxi_dT2;
    MatrixN right_jac_inv;
    if (Hp1 || Hp2) {
      MatrixN dbetween_p1;
      MatrixN dbetween_p2;
      xi = traits<Pose>::Logmap(
          traits<Pose>::Between(p1, p2, &dbetween_p1, &dbetween_p2),
          &right_jac_inv);
      dxi_dT1 = right_jac_inv * dbetween_p1;
      dxi_dT2 = right_jac_inv * dbetween_p2;
    } else {
      xi = traits<Pose>::Logmap(traits<Pose>::Between(p1, p2), &right_jac_inv);
    }
    // Compute error for local state vector (pose, velocity) in the tangent
    // space
    Vector2N err;
    err << xi - delta_t_ * v1, right_jac_inv * v2 - v1;

    // Derivative of velocity error wrt xi
    MatrixN dvErr_dxi;
    if (Hp1 || Hp2) {
      // Derivative of velocity error wrt xi
      // Zero for vector spaces, use an approximation for Lie groups
      if constexpr (std::is_same_v<typename traits<Pose>::structure_category,
                                   vector_space_tag>) {
        dvErr_dxi.setZero();
      } else {
        // For Lie groups
        dvErr_dxi = -Pose::adjointMap(v2) / 2.0 -
                    (Pose::adjointMap(Pose::adjointMap(xi) * v2) +
                     Pose::adjointMap(xi) * Pose::adjointMap(v2)) /
                        12.0;
      }
    }
    // Compute Final Jacobians
    if (Hp1) {
      // Derivative of error wrt pose p1
      *Hp1 = (Matrix(2 * dim, dim) << dxi_dT1, dvErr_dxi * dxi_dT1).finished();
    }
    if (Hv1) {
      // Derivative of error wrt velocity v1
      *Hv1 =
          (Matrix(2 * dim, dim) << -delta_t_ * Identity, -Identity).finished();
    }
    if (Hp2) {
      // Derivative of error wrt pose p2
      *Hp2 = (Matrix(2 * dim, dim) << dxi_dT2, dvErr_dxi * dxi_dT2).finished();
    }
    if (Hv2) {
      // Derivative of error wrt velocity v2
      *Hv2 = (Matrix(2 * dim, dim) << Zero, right_jac_inv).finished();
    }

    return err;
  }

  /**
   * @brief Build the continuous-time WNOA discretized process covariance.
   *
   * Returns the 2N x 2N covariance matrix for a WNOA prior discretized over
   * `timestep` using the diagonal PSD `Q`.
   * See (11.7) in (Barfoot, 2024) for the derivation of covariance for the WNOA
   * prior
   *
   * @param timestep Time interval over which to compute covariance
   * @param Q Diagonal PSD vector
   * @return Matrix2N Process covariance
   */
  static Matrix2N buildWNOACovariance(double timestep, const VectorN& Q) {
    //
    Matrix2N covariance;
    MatrixN Q_diag = Q.asDiagonal();
    covariance << (1.0 / 3.0 * pow(timestep, 3)) * Q_diag,
        (1.0 / 2.0 * pow(timestep, 2)) * Q_diag,
        (1.0 / 2.0 * pow(timestep, 2)) * Q_diag, timestep * Q_diag;
    return covariance;
  }

  /**
   * @brief Build the inverse of the WNOA discretized process covariance.
   *
   * @param timestep Time interval
   * @param Q Diagonal PSD vector
   * @return Matrix2N Inverse process covariance matrix
   */
  static Matrix2N buildInverseWNOACovariance(double timestep,
                                             const VectorN& Q) {
    // construct the inverse covariance matrix for the WNOA factor
    Matrix2N inverse_covariance;
    MatrixN Q_inv_diag = Q.cwiseInverse().asDiagonal();
    inverse_covariance << (12.0 / (timestep * timestep * timestep)) *
                              Q_inv_diag,
        (-6.0 / (timestep * timestep)) * Q_inv_diag,
        (-6.0 / (timestep * timestep)) * Q_inv_diag,
        (4.0 / timestep) * Q_inv_diag;

    return inverse_covariance;
  }

  /**
   * @brief Convenience helper to construct a Gaussian noise model from `Q`.
   *
   * The noise model uses the covariance produced by `buildWNOACovariance`.
   *
   * @param timestep Time interval
   * @param Q Diagonal PSD vector
   * @return noiseModel::Gaussian::shared_ptr Noise model built from covariance
   */
  static inline noiseModel::Gaussian::shared_ptr buildWNOANoiseModel(
      double timestep, const VectorN& Q) {
    return noiseModel::Gaussian::Covariance(buildWNOACovariance(timestep, Q));
  }

  /**
   * @brief Transition matrix for the WNOA prior.
   *
   * Returns the 2N x 2N transition matrix mapping the concatenated state
   * [pose; vel] over interval `delta_t` using the standard WNOA linearization.
   *
   * @param delta_t Time interval
   * @return Matrix2N Transition matrix
   */
  static Matrix2N transitionFunction(double delta_t) {
    // Construct the transition matrix for the WNOA factor
    Matrix2N F;
    F << Identity, delta_t * Identity, Zero, Identity;
    return F;
  }

  /**
   * @brief Compute interpolation Jacobian with respect to the previous (left)
   * state.
   *
   * Computes the 2N x 2N Jacobian block that maps perturbations in the
   * previous bordering state (pose_k, vel_k) to perturbations in the
   * interpolated discrete residual.
   *
   * @param pv1 Pair (pose_k, vel_k)
   * @param pv2 Pair (pose_kp1, vel_kp1)
   * @param delta_t Time interval
   * @return Matrix2N Jacobian block w.r.t. previous state
   */
  static Matrix2N computeJacobianPrev(const std::pair<Pose, Velocity>& pv1,
                                      const std::pair<Pose, Velocity>& pv2,
                                      double delta_t) {
    // corresponds to F in (11.20) in SER
    auto& [p1, v1] = pv1;
    auto& [p2, v2] = pv2;

    MatrixN dbetween_p1;
    MatrixN dxi_dT1;
    MatrixN right_jac_inv;
    VectorN xi;
    xi = traits<Pose>::Logmap(
        traits<Pose>::Between(p1, p2, &dbetween_p1, nullptr), &right_jac_inv);
    dxi_dT1 = right_jac_inv * dbetween_p1;
    // Derivative of velocity error wrt xi
    MatrixN dvErr_dxi;
    if constexpr (std::is_same_v<typename traits<Pose>::structure_category,
                                 vector_space_tag>) {
      dvErr_dxi.setZero();
    } else {
      dvErr_dxi = -Pose::adjointMap(v2) / 2.0 -
                  (Pose::adjointMap(Pose::adjointMap(xi) * v2) +
                   Pose::adjointMap(xi) * Pose::adjointMap(v2)) /
                      12.0;
    }
    Matrix2N F;
    // first column is pose, second column is velocity
    F << dxi_dT1, -1 * delta_t * Identity, dvErr_dxi * dxi_dT1, -1 * Identity;
    return F;
  }

  /**
   * @brief Compute interpolation Jacobian with respect to the next (right)
   * state.
   *
   * Computes the 2N x 2N Jacobian block that maps perturbations in the
   * next bordering state (pose_kp1, vel_kp1) to perturbations in the
   * interpolated discrete residual.
   *
   * @param pv1 Pair (pose_k, vel_k)
   * @param pv2 Pair (pose_kp1, vel_kp1)
   * @param delta_t Time interval
   * @return Matrix2N Jacobian block w.r.t. next state
   */
  static Matrix2N computeJacobianNext(const std::pair<Pose, Velocity>& pv1,
                                      const std::pair<Pose, Velocity>& pv2,
                                      double delta_t) {
    // corresponds to E in (11.21) in SER
    auto& [p1, v1] = pv1;
    auto& [p2, v2] = pv2;

    MatrixN dxi_dT2;
    MatrixN dbetween_p2;
    MatrixN right_jac_inv;

    VectorN xi;
    xi = traits<Pose>::Logmap(
        traits<Pose>::Between(p1, p2, nullptr, &dbetween_p2), &right_jac_inv);

    dxi_dT2 = right_jac_inv * dbetween_p2;
    // Derivative of velocity error wrt xi
    MatrixN dvErr_dxi;
    if constexpr (std::is_same_v<typename traits<Pose>::structure_category,
                                 vector_space_tag>) {
      dvErr_dxi.setZero();
    } else {
      dvErr_dxi = -Pose::adjointMap(v2) / 2.0 -
                  (Pose::adjointMap(Pose::adjointMap(xi) * v2) +
                   Pose::adjointMap(xi) * Pose::adjointMap(v2)) /
                      12.0;
    }
    Matrix2N E;
    // First column is pose, second column is velocity
    E << dxi_dT2, Zero, dvErr_dxi * dxi_dT2, right_jac_inv;
    return -1 * E;
  }
};

// Make factors testable
template <class Pose>
struct traits<WNOAMotionFactor<Pose>>
    : public Testable<WNOAMotionFactor<Pose>> {};

// Explicit Instantiation.
template class WNOAMotionFactor<Point1>;
template class WNOAMotionFactor<Point2>;
template class WNOAMotionFactor<Point3>;
template class WNOAMotionFactor<Pose2>;
template class WNOAMotionFactor<Pose3>;

}  // namespace gtsam