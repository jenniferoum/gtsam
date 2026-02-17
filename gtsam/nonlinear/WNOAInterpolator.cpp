/* ----------------------------------------------------------------------------
 * @file Interpolator.cpp
 *
 * Note: References to (Barfoot 2024) refer to the following textbook:
 * Barfoot, Timothy D. State estimation for robotics. Cambridge University
 * Press, 2024.
 * -------------------------------------------------------------------------- */

#include "WNOAInterpolator.h"

namespace gtsam {

// ---- Constructors ----
template <typename PoseType>
Interpolator<PoseType>::Interpolator(
    const VectorN& Q_psd,
    std::function<Matrix(double dt)> transitionFunction,
    std::function<Matrix(double dt, const VectorN& Q_psd)> covarianceFunction,
    std::function<Matrix(double dt, const VectorN& Q_psd)>

        inverseCovarianceFunction,
    std::function<Matrix(const std::pair<PoseType, VelocityType>&,
                         const std::pair<PoseType, VelocityType>&, double)>
        computeJacobianPrev,
    std::function<Matrix(const std::pair<PoseType, VelocityType>&,
                         const std::pair<PoseType, VelocityType>&, double)>
        computeJacobianNext)
    : Q_psd_(Q_psd),
      transitionFunction_(transitionFunction),
      covarianceFunction_(covarianceFunction),
      inverseCovarianceFunction_(inverseCovarianceFunction),
      computeJacobianPrev_(computeJacobianPrev),
      computeJacobianNext_(computeJacobianNext) {}

template <typename PoseType>
Interpolator<PoseType>::Interpolator(const VectorN& Q_psd)
    : Interpolator(Q_psd,
                   WNOAMotionFactor<PoseType>::transitionFunction,
                   WNOAMotionFactor<PoseType>::buildWNOACovariance,
                   WNOAMotionFactor<PoseType>::buildInverseWNOACovariance,
                   WNOAMotionFactor<PoseType>::computeJacobianPrev,
                   WNOAMotionFactor<PoseType>::computeJacobianNext) {}

// ---- Member Functions ----
template <typename PoseType>
typename Interpolator<PoseType>::PoseVel
Interpolator<PoseType>::interpolatePoseAndVelocity(
    const std::optional<TimestampedPoseVel>& tPoseVel_k,
    const std::optional<TimestampedPoseVel>& tPoseVel_kp1, double t_tau,
    OptionalMatrixVecType H,
    const std::shared_ptr<Matrix>& mainSolveMarginalMatrix,
    Matrix* covarianceOut,
    const std::shared_ptr<const LambdaPsiMats>& LambdaPsiPreComp,
    const std::shared_ptr<const LocalStateVecs>& localStateVecsPreComp,
    const std::shared_ptr<const LocalGlobalStateJacs>&
        localGlobalStateJacsPreComp) const {
  assert((tPoseVel_k.has_value() || tPoseVel_kp1.has_value()) &&
         "At least one TimestampedPoseVel must be defined");
  // second point not defined, extrap from first
  if (!tPoseVel_kp1.has_value()) {
    const auto& [poseVel, t] = tPoseVel_k.value();
    double t_diff = t_tau - t;
    return extrapolatePoseAndVelocity(poseVel, t_diff, H,
                                      mainSolveMarginalMatrix, covarianceOut);
  }
  // first point not defined, extrapolated from second
  if (!tPoseVel_k.has_value()) {
    const auto& [poseVel, t] = tPoseVel_kp1.value();
    double t_diff = t_tau - t;
    return extrapolatePoseAndVelocity(poseVel, t_diff, H,
                                      mainSolveMarginalMatrix, covarianceOut);
  }

  // extract both values
  const auto& [poseVel_k, t_k] = tPoseVel_k.value();
  const auto& [poseVel_kp1, t_kp1] = tPoseVel_kp1.value();
  // interpolate
  if (t_tau == t_k) {
    // t_tau equal to left boundary time
    return interpolateBoundaryLeft(poseVel_k, H, mainSolveMarginalMatrix,
                                   covarianceOut);

  } else if (t_tau == t_kp1) {
    // t_tau equal to right boundary time
    return interpolateBoundaryRight(poseVel_kp1, H, mainSolveMarginalMatrix,
                                    covarianceOut);

  } else if (t_tau < t_k) {
    // Extrapolation from left boundary
    double t_diff = t_tau - t_k;
    return extrapolatePoseAndVelocity(poseVel_k, t_diff, H,
                                      mainSolveMarginalMatrix, covarianceOut);
  } else if (t_tau > t_kp1) {
    // Extrapolation from right boundary
    double t_diff = t_tau - t_kp1;
    return extrapolatePoseAndVelocity(poseVel_kp1, t_diff, H,
                                      mainSolveMarginalMatrix, covarianceOut);
  } else {
    // only remaining case is that t_tau is within border states
    // call protected overload of interpolate function
    return interpolatePoseAndVelocity_(
          tPoseVel_k.value(), tPoseVel_kp1.value(), t_tau, H,
          mainSolveMarginalMatrix, covarianceOut, LambdaPsiPreComp,
          localStateVecsPreComp, localGlobalStateJacsPreComp);
  }
}

template <typename PoseType>
typename Interpolator<PoseType>::PoseVel
Interpolator<PoseType>::interpolateBoundaryLeft(
    const PoseVelocity<PoseType>& poseVel_k, OptionalMatrixVecType H,
    const std::shared_ptr<Matrix>& mainSolveMarginalMatrix,
    Matrix* covarianceOut) const {
  if (H) {
    // dTtau_dTk
    (*H)[0] = MatrixN::Identity();
    // dTtau_dvarpik
    (*H)[1] = MatrixN::Zero();
    // dTtau_dTkp1
    (*H)[2] = MatrixN::Zero();
    // dTtau_dvarpikp1
    (*H)[3] = MatrixN::Zero();
    // dvarpitau_dTk
    (*H)[4] = MatrixN::Zero();
    // dvarpitau_dvarpik
    (*H)[5] = MatrixN::Identity();
    // dvarpitau_dTkp1
    (*H)[6] = MatrixN::Zero();
    // dvarpitau_dvarpikp1
    (*H)[7] = MatrixN::Zero();
  }
  if (covarianceOut && mainSolveMarginalMatrix) {
    // if t_tau == t_k, then the covariance is the same as that of Tvarpi_k
    *covarianceOut = mainSolveMarginalMatrix->topLeftCorner(dim * 2, dim * 2);
  }
  return poseVel_k;
}

template <typename PoseType>
typename Interpolator<PoseType>::PoseVel
Interpolator<PoseType>::interpolateBoundaryRight(
    const PoseVelocity<PoseType>& poseVel_kp1, OptionalMatrixVecType H,
    const std::shared_ptr<Matrix>& mainSolveMarginalMatrix,
    Matrix* covarianceOut) const {
  if (H) {
    // dTtau_dTk
    (*H)[0] = MatrixN::Zero();
    // dTtau_dvarpik
    (*H)[1] = MatrixN::Zero();
    // dTtau_dTkp1
    (*H)[2] = MatrixN::Identity();
    // dTtau_dvarpikp1
    (*H)[3] = MatrixN::Zero();
    // dvarpitau_dTk
    (*H)[4] = MatrixN::Zero();
    // dvarpitau_dvarpik
    (*H)[5] = MatrixN::Zero();
    // dvarpitau_dTkp1
    (*H)[6] = MatrixN::Zero();
    // dvarpitau_dvarpikp1
    (*H)[7] = MatrixN::Identity();
  }
  if (covarianceOut && mainSolveMarginalMatrix) {
    // if t_tau == t_kp1, then the covariance is the same as that of Tvarpi_kp1
    *covarianceOut =
        mainSolveMarginalMatrix->bottomRightCorner(dim * 2, dim * 2);
  }
  return poseVel_kp1;
}

template <typename PoseType>
typename Interpolator<PoseType>::PoseVel
Interpolator<PoseType>::extrapolatePoseAndVelocity(
    const PoseVelocity<PoseType>& poseVel_ex, double t_diff,
    OptionalMatrixVecType H,
    const std::shared_ptr<Matrix>& mainSolveMarginalMatrix,
    Matrix* covarianceOut) const {
  // follows (11.5) in (Barfoot 2024)
  const auto& [T_ex, varpi_ex] = poseVel_ex;
  // Define local state vector at time k, in the Lie algebra of Pose at time k
  Vector2N gamma_k;
  gamma_k.topRows(dim).setZero();
  gamma_k.bottomRows(dim) = varpi_ex;
  // Extrapolate pose and velocity to time tau, using WNOA extrapolation
  // equations
  auto Psi = transitionFunction_(t_diff);
  auto gamma_ex = Psi * gamma_k;
  // Map back to manifold
  auto T_tau = traits<PoseType>::Compose(
      T_ex, traits<PoseType>::Expmap(gamma_ex.topRows(dim), nullptr));
  auto varpi_tau = gamma_ex.bottomRows(dim);

  // compute covariance of the extrapolated pose and velocity
  // assume that mainSolveMarginalMatrix corresponds to the covariance of
  // Tvarpi_extrapolate_point
  if (mainSolveMarginalMatrix && covarianceOut) {
    assert(mainSolveMarginalMatrix->rows() == 2 * dim &&
           mainSolveMarginalMatrix->cols() == 2 * dim);
    Matrix2N Sigma = covarianceFunction_(t_diff, Q_psd_);
    // (11.5) in (Barfoot 2024)
    *covarianceOut = Sigma + Psi * *mainSolveMarginalMatrix * Psi.transpose();
  }
  return PoseVel{T_tau, varpi_tau};
}

template <typename PoseType>
typename Interpolator<PoseType>::PoseVel
Interpolator<PoseType>::interpolatePoseAndVelocity_(
    const TimestampedPoseVel& tPoseVel_k,
    const TimestampedPoseVel& tPoseVel_kp1, double t_tau,
    OptionalMatrixVecType H,
    const std::shared_ptr<Matrix>& mainSolveMarginalMatrix,
    Matrix* covarianceOut,
    const std::shared_ptr<const LambdaPsiMats>& LambdaPsiPreComp,
    const std::shared_ptr<const LocalStateVecs>& localStateVecsPreComp,
    const std::shared_ptr<const LocalGlobalStateJacs>&
        localGlobalStateJacsPreComp) const {
  // unpack poses and velocities
  const auto& [poseVel_k, t_k] = tPoseVel_k;
  const auto& [poseVel_kp1, t_kp1] = tPoseVel_kp1;
  const auto& [T_k, varpi_k] = poseVel_k;

  // Retrieve interpolation matrices
  Matrix2N Lambda, Psi;
  if (!LambdaPsiPreComp) {
    std::tie(Lambda, Psi) = getLambdaPsi(t_k, t_kp1, t_tau);
  } else {
    std::tie(Lambda, Psi) = *LambdaPsiPreComp;
  }
  // Form local state vectors at time k, in the Lie algebra of Pose at time k
  // Follows Eq. (11.45) in (Barfoot 2024)
  VectorN xi_k, xi_dot_k;
  xi_k.setZero();
  xi_dot_k = varpi_k;

  // Intermediate Jacobians for xi and xi_dot at next time step with respect to
  // the bordering states and velocities
  MatrixN dxi_dTk;
  MatrixN dxi_dTkp1;
  MatrixN dxidot_dTk;
  MatrixN dxidot_dTkp1;
  MatrixN dxidotkp1_dvarpikp1;
  // Get local state vectors at time kp1, in the Lie algebra of Pose at time k
  VectorN xi_kp1, xi_dot_kp1;
  if (localStateVecsPreComp) {
    // If precomputation of local state vectors is provided, use it to save
    // computation
    xi_kp1 = localStateVecsPreComp->first;
    xi_dot_kp1 = localStateVecsPreComp->second;

    if (H && localGlobalStateJacsPreComp) {
      dxi_dTk = localGlobalStateJacsPreComp->at(0);
      dxi_dTkp1 = localGlobalStateJacsPreComp->at(1);
      dxidot_dTk = localGlobalStateJacsPreComp->at(2);
      dxidot_dTkp1 = localGlobalStateJacsPreComp->at(3);
      dxidotkp1_dvarpikp1 = localGlobalStateJacsPreComp->at(4);
    }

  } else {
    LocalStateVecs local_state;
    LocalGlobalStateJacs local_jacs;
    if (H) {
      local_state =
          computeLocalStateVecs(tPoseVel_k, tPoseVel_kp1, &local_jacs);
      dxi_dTk = local_jacs[0];
      dxi_dTkp1 = local_jacs[1];
      dxidot_dTk = local_jacs[2];
      dxidot_dTkp1 = local_jacs[3];
      dxidotkp1_dvarpikp1 = local_jacs[4];
    } else {
      local_state = computeLocalStateVecs(tPoseVel_k, tPoseVel_kp1, nullptr);
    }
    xi_kp1 = local_state.first;
    xi_dot_kp1 = local_state.second;
  }
  // Compute local state vectors at time tau, in the Lie algebra of Pose at time
  // k, using WNOA interpolation equations
  VectorN xi_tau =
      Lambda.block(0, dim, dim, dim) * xi_dot_k +
      Psi.block(0, 0, dim, dim) * xi_kp1 +
      Psi.block(0, dim, dim, dim) * xi_dot_kp1;  // Dropping xi_k term here since it's zero
  VectorN xidot_tau =
      Lambda.block(dim, dim, dim, dim) * xi_dot_k +
      Psi.block(dim, 0, dim, dim) * xi_kp1 +
      Psi.block(dim, dim, dim, dim) * xi_dot_kp1;  // Dropping xi_k term here since it's zero
  // Additional intermediate Jacobians
  MatrixN right_jac_tau;
  MatrixN dTtau_dTk;
  MatrixN dTtau_dxitau;
  // Map the local state vector at time tau back to the manifold to get the
  // interpolated pose Eq. (11.45) in (Barfoot 2024)
  PoseType T_tau;
  if (H) {
    T_tau = traits<PoseType>::Compose(
        T_k, traits<PoseType>::Expmap(xi_tau, &right_jac_tau), &dTtau_dTk,
        &dTtau_dxitau);
    dTtau_dxitau = dTtau_dxitau * right_jac_tau;
  } else {
    T_tau = traits<PoseType>::Compose(
        T_k, traits<PoseType>::Expmap(xi_tau, &right_jac_tau));
  }
  // Compute interpolated velocity at time tau
  auto varpi_tau = right_jac_tau * xidot_tau;

  // Compute complete Jacobians
  if (H) {
    // Zero for vector spaces, use an approximation for Lie groups
    MatrixN dvarpitau_dxitau;
    if constexpr (std::is_same_v<typename traits<PoseType>::structure_category,
                                 vector_space_tag>) {
      dvarpitau_dxitau.setZero();
    } else {
      // For Lie groups
      dvarpitau_dxitau = PoseType::adjointMap(xidot_tau) / 2.0;
    }

    // derivatives of local state position at time tau with respect to bordering
    // states and velocities
    MatrixN dxitau_dTk = Psi(0, 0) * dxi_dTk + Psi(0, dim) * dxidot_dTk;
    MatrixN dxitau_dvarpik =
        Lambda(0, dim) * MatrixN::Identity();  // xi does not depend on varpi_k
                                               // and xi_dot is exactly varpi_k
    MatrixN dxitau_dTkp1 = Psi(0, 0) * dxi_dTkp1 + Psi(0, dim) * dxidot_dTkp1;
    MatrixN dxitau_dvarpikp1 =
        Psi(0, dim) *
        dxidotkp1_dvarpikp1;  // dxikp1 does not depend on varpi_kp1
    // derivatives of local state velocity at time tau with respect to bordering
    // states and velocities
    MatrixN dxidottau_dTk = Psi(dim, 0) * dxi_dTk + Psi(dim, dim) * dxidot_dTk;
    MatrixN dxidottau_dvarpik =
        Lambda(dim, dim) *
        MatrixN::Identity();  // xi does not depend on varpi_k and xi_dot is
                              // exactly varpi_k
    MatrixN dxidottau_dTkp1 =
        Psi(dim, 0) * dxi_dTkp1 + Psi(dim, dim) * dxidot_dTkp1;
    MatrixN dxidottau_dvarpikp1 =
        Psi(dim, dim) *
        dxidotkp1_dvarpikp1;  // dxikp1 does not depend on varpi_kp1
    // Compose final Jacobians using chain rule
    // dTtau_dTk
    (*H)[0] = dTtau_dTk + dTtau_dxitau * dxitau_dTk;
    // dTtau_dvarpik
    (*H)[1] = dTtau_dxitau * dxitau_dvarpik;
    // dTtau_dTkp1
    (*H)[2] = dTtau_dxitau * dxitau_dTkp1;
    // dTtau_dvarpikp1
    (*H)[3] = dTtau_dxitau * dxitau_dvarpikp1;
    // dvarpitau_dTk
    (*H)[4] = right_jac_tau * dxidottau_dTk + dvarpitau_dxitau * dxitau_dTk;
    // dvarpitau_dvarpik
    (*H)[5] =
        right_jac_tau * dxidottau_dvarpik + dvarpitau_dxitau * dxitau_dvarpik;
    // dvarpitau_dTkp1
    (*H)[6] = right_jac_tau * dxidottau_dTkp1 + dvarpitau_dxitau * dxitau_dTkp1;
    // dvarpitau_dvarpikp1
    (*H)[7] = right_jac_tau * dxidottau_dvarpikp1 +
              dvarpitau_dxitau * dxitau_dvarpikp1;
  }

  // Output pair
  auto poseVel_tau = PoseVel{T_tau, varpi_tau};

  // compute covariance of the interpolated pose (and velocity, if required)
  // using Lambda and Psi computed from (11.41) in (Barfoot 2024).
  // This should be equivalent to using (4.23) in the FnT paper.
  if (mainSolveMarginalMatrix && covarianceOut) {
    Eigen::Matrix<double, 2 * dim, 4 * dim> LambdaPsi;
    Matrix2N Sigma = computeConditionalCov(
        tPoseVel_k, tPoseVel_kp1, TimestampedPoseVel{poseVel_tau, t_tau});
    LambdaPsi << Lambda, Psi;
    *covarianceOut =
        Sigma + LambdaPsi * *mainSolveMarginalMatrix * LambdaPsi.transpose();
  }
  return poseVel_tau;
}

template <typename PoseType>
std::map<StateDataInterval, std::shared_ptr<Matrix>>
Interpolator<PoseType>::computeJointMarginals(
    const std::map<StateDataInterval, std::vector<StateData>>& queryBuckets,
    const std::unique_ptr<Marginals>& marginals) {
  std::map<StateDataInterval, std::shared_ptr<Matrix>>
      intervalJointMarginals;  // JointMarginal matrices for each interval
  std::unordered_set<Key> allBoundaryKeys;

  // Lambda function for forming the boundary key vector
  auto formBoundaryKeyVector = [](const StateDataInterval& stateDataBorders) {
    KeyVector boundaryKeyVector;
    if (stateDataBorders.first.has_value()) {
      boundaryKeyVector.push_back(stateDataBorders.first->pose);  // p1
      boundaryKeyVector.push_back(stateDataBorders.first->vel);   // v1
    }
    if (stateDataBorders.second.has_value()) {
      boundaryKeyVector.push_back(stateDataBorders.second->pose);  // p2
      boundaryKeyVector.push_back(stateDataBorders.second->vel);   // v2
    }
    return boundaryKeyVector;
  };

  for (const auto& [stateDataBorders, stateDataInterpVec] : queryBuckets) {
    KeyVector boundaryKeyVector = formBoundaryKeyVector(stateDataBorders);
    allBoundaryKeys.insert(boundaryKeyVector.begin(), boundaryKeyVector.end());
    // Compute JointMarginal for each interval separately
    // This is fast if there are not too many intervals
    JointMarginal mainSolveMarginal =
        marginals->jointMarginalCovariance(boundaryKeyVector);
    // avoid using JointMarginal.fullMatrix() as it returns covariance
    // in alphabetical order of the keys...
    auto mainSolveMarginalMatrix =
        std::make_shared<Matrix>(constructMatrixFromJointMarginal(
            mainSolveMarginal, boundaryKeyVector, dim));
    intervalJointMarginals[stateDataBorders] = mainSolveMarginalMatrix;
  }

  return intervalJointMarginals;
}

template <typename PoseType>
Values Interpolator<PoseType>::interpolatePosesAndVelocities(
    const NonlinearFactorGraph& mainSolveGraph, const Values& mainSolveSolution,
    const StateDataSet& mainSolveStates, const StateDataSet& interpolatedStates,
    std::shared_ptr<CovarianceMap> covarianceMapOut) const {
  // Map from intervals [t1, t2) to query times inside that interval (bucket)
  std::map<StateDataInterval, std::vector<StateData>> queryBuckets;

  // Find all intervals [t_k, t_kp1) that contain at least one query time
  for (const auto& stateDataInterp : interpolatedStates) {
    auto it2 = mainSolveStates.upper_bound(stateDataInterp);
    StateDataInterval interval;
    interval.first = it2 == mainSolveStates.begin()
                         ? std::nullopt
                         : std::optional<StateData>(*std::prev(it2));
    interval.second = it2 == mainSolveStates.end()
                          ? std::nullopt
                          : std::optional<StateData>(*it2);
    queryBuckets[interval].push_back(stateDataInterp);
  }

  Values interpolatedSolution;

  // Only construct a Marginals if requested
  std::unique_ptr<Marginals> fullMarginal;
  // JointMarginal matrices for each interval (every pair of boundary states
  // involved)
  auto intervalJointMarginals =
      std::map<StateDataInterval, std::shared_ptr<Matrix>>();
  if (covarianceMapOut) {
    // Compute all required joint marginals
    fullMarginal =
        std::make_unique<Marginals>(mainSolveGraph, mainSolveSolution);
    intervalJointMarginals = computeJointMarginals(queryBuckets, fullMarginal);
  }

  // Perform interpolation for each bucket
  for (const auto& [stateDataBorder, stateDataInterpVec] : queryBuckets) {
    auto makeTimestampedPV = [&](const std::optional<StateData>& s)
        -> std::optional<TimestampedPoseVel> {
      if (!s) return std::nullopt;
      return TimestampedPoseVel(mainSolveSolution.at<PoseType>(s->pose),
                                mainSolveSolution.at<VelocityType>(s->vel),
                                s->time);
    };

    // Get the poses and velocities at t_k and t_kp1
    std::optional<TimestampedPoseVel> pvk =
        makeTimestampedPV(stateDataBorder.first);
    std::optional<TimestampedPoseVel> pvkp1 =
        makeTimestampedPV(stateDataBorder.second);
    Matrix covarianceOut;  // (2*dim, 2*dim)

    // Interpolate for all query times within this query interval (bucket)
    for (const auto& stateDataInterp : stateDataInterpVec) {
      std::shared_ptr<Matrix> mainSolveMarginalMatrix;  // (4*dim, 4*dim)
      if (covarianceMapOut) {
        mainSolveMarginalMatrix = intervalJointMarginals[stateDataBorder];
      }
      auto pvtau =
          interpolatePoseAndVelocity(pvk, pvkp1, stateDataInterp.time, nullptr,
                                     mainSolveMarginalMatrix, &covarianceOut);
      auto [T_tau, varpi_tau] = pvtau;
      interpolatedSolution.insert(stateDataInterp.pose, T_tau);
      interpolatedSolution.insert(stateDataInterp.vel, varpi_tau);
      if (covarianceMapOut) {
        // upper left covariance block corresponds to pose, lower right block
        // corresponds to velocity
        (*covarianceMapOut)[stateDataInterp.pose] =
            covarianceOut.topLeftCorner(dim, dim);
        (*covarianceMapOut)[stateDataInterp.vel] =
            covarianceOut.bottomRightCorner(dim, dim);
      }
    }
  }
  return interpolatedSolution;
}

// Lambda-Psi interpolator equations for WNOA-specific interpolator
template <typename PoseType>
std::pair<Matrix, Matrix> Interpolator<PoseType>::getLambdaPsi(
    double t_k, double t_kp1, double t_tau) const {
  // See Section 11.1 in (Barfoot 2024) for the derivation of these equations
  // for WNOA
  double dt = t_kp1 - t_k;
  double alpha = (t_tau - t_k) / dt;
  double alpha2 = alpha * alpha;
  double alpha3 = alpha2 * alpha;

  Matrix2N Lambda, Psi;

  double Psi_00 = 3 * alpha2 - 2 * alpha3;
  double Psi_01 = dt * (alpha3 - alpha2);
  double Psi_10 = (6 * alpha / dt) - (6 * alpha2 / dt);
  double Psi_11 = 3 * alpha2 - 2 * alpha;

  double Lambda_00 = 1 - Psi_00;
  double Lambda_01 = dt * (alpha - 2 * alpha2 + alpha3);
  double Lambda_10 = -Psi_10;
  double Lambda_11 = 1 - 4 * alpha + 3 * alpha2;

  Lambda << MatrixN::Identity() * Lambda_00, MatrixN::Identity() * Lambda_01,
      MatrixN::Identity() * Lambda_10, MatrixN::Identity() * Lambda_11;

  Psi << MatrixN::Identity() * Psi_00, MatrixN::Identity() * Psi_01,
      MatrixN::Identity() * Psi_10, MatrixN::Identity() * Psi_11;

  return std::make_pair(Lambda, Psi);
}

template <typename PoseType>
std::pair<Matrix, Matrix> Interpolator<PoseType>::getLambdaPsiGeneral(
    double t_k, double t_kp1, double t_tau) const {
  // Build general interpolation equations for Lambda and Psi using the
  // transition function and See (11.41) in (Barfoot 2024) for the derivation of
  // these equations for general linear time-invariant systems
  double dt = t_kp1 - t_k;
  auto Phi_12 = transitionFunction_(dt);
  auto Phi_1tau = transitionFunction_(t_tau - t_k);
  auto Phi_tau2 = transitionFunction_(t_kp1 - t_tau);

  // Construct Q
  auto Q_12_inv = inverseCovarianceFunction_(dt, Q_psd_);
  auto Q_1tau = covarianceFunction_(t_tau - t_k, Q_psd_);

  // Eq. (11.41) in (Barfoot 2024)
  auto Lambda = Phi_1tau - Q_1tau * Phi_tau2.transpose() * Q_12_inv * Phi_12;
  auto Psi = Q_1tau * Phi_tau2.transpose() * Q_12_inv;

  return std::make_pair(Lambda, Psi);
}

template <typename PoseType>
typename Interpolator<PoseType>::LocalStateVecs
Interpolator<PoseType>::computeLocalStateVecs(
    const TimestampedPoseVel& pvk, const TimestampedPoseVel& pvkp1,
    Interpolator<PoseType>::LocalGlobalStateJacs* jacs) const {
  const auto& [poseVel_k, t_k] = pvk;
  const auto& [poseVel_kp1, t_kp1] = pvkp1;
  const auto& [T_k, varpi_k] = poseVel_k;
  const auto& [T_kp1, varpi_kp1] = poseVel_kp1;

  VectorN xi_kp1, xi_dot_kp1;
  MatrixN right_jac_inv;
  MatrixN dxi_dTk, dxi_dTkp1;
  MatrixN dxidot_dTk, dxidot_dTkp1;
  MatrixN dxidotkp1_dvarpikp1;

  if (jacs) {
    MatrixN dbetween_Tk;
    MatrixN dbetween_Tkp1;
    xi_kp1 = traits<PoseType>::Logmap(
        traits<PoseType>::Between(T_k, T_kp1, &dbetween_Tk, &dbetween_Tkp1),
        &right_jac_inv);
    // Compute derivatives
    dxi_dTk = right_jac_inv * dbetween_Tk;
    dxi_dTkp1 = right_jac_inv * dbetween_Tkp1;
  } else {
    xi_kp1 = traits<PoseType>::Logmap(traits<PoseType>::Between(T_k, T_kp1),
                                      &right_jac_inv);
  }
  xi_dot_kp1 = right_jac_inv * varpi_kp1;
  LocalStateVecs local_state;
  local_state.first = xi_kp1;
  local_state.second = xi_dot_kp1;

  if (jacs) {
    // Zero for vector spaces, use an approximation for Lie groups
    MatrixN dxidot_dxi;
    if constexpr (std::is_same_v<typename traits<PoseType>::structure_category,
                                 vector_space_tag>) {
      dxidot_dxi.setZero();
    } else {
      // For Lie groups
      dxidot_dxi = -PoseType::adjointMap(varpi_kp1) / 2.0;
    }
    dxidot_dTk = dxidot_dxi * dxi_dTk;
    dxidot_dTkp1 = dxidot_dxi * dxi_dTkp1;
    dxidotkp1_dvarpikp1 = right_jac_inv;

    jacs->clear();
    jacs->push_back(dxi_dTk);
    jacs->push_back(dxi_dTkp1);
    jacs->push_back(dxidot_dTk);
    jacs->push_back(dxidot_dTkp1);
    jacs->push_back(dxidotkp1_dvarpikp1);
  }

  return local_state;
}

template <typename PoseType>
typename Interpolator<PoseType>::Matrix2N
Interpolator<PoseType>::computeConditionalCov(
    const TimestampedPoseVel& tPoseVel_k,
    const TimestampedPoseVel& tPoseVel_kp1,
    const TimestampedPoseVel& tPoseVel_tau, OptionalMatrixType Lambda,
    OptionalMatrixType Psi) const {
  auto [poseVel_k, t_k] = tPoseVel_k;
  auto [poseVel_kp1, t_kp1] = tPoseVel_kp1;
  auto [poseVel_tau, t_tau] = tPoseVel_tau;

  // Todo (Daniel): handle the case when t_tau == t_k or t_tau == t_kp1
  assert(t_tau > t_k && t_tau < t_kp1 &&
         "t_tau must be in the interval (t_k, t_kp1)");

  // see Figure 5.4 in the paper
  Matrix2N Q_tau_prev_inv = inverseCovarianceFunction_(t_tau - t_k, Q_psd_);
  Matrix2N Q_tau_next_inv = inverseCovarianceFunction_(t_kp1 - t_tau, Q_psd_);
  Matrix2N E_tau = computeJacobianNext_(poseVel_k.asPair(),
                                        poseVel_tau.asPair(), t_tau - t_k);
  Matrix2N F_k_tau = computeJacobianPrev_(poseVel_tau.asPair(),
                                          poseVel_kp1.asPair(), t_kp1 - t_tau);
  Matrix2N Sigma_inv = E_tau.transpose() * Q_tau_prev_inv * E_tau +
                       F_k_tau.transpose() * Q_tau_next_inv * F_k_tau;
  Matrix2N Sigma = Sigma_inv.inverse();

  // (5.23) in the paper
  if (Lambda) {
    Matrix2N F_tau_km1 = computeJacobianPrev_(
        poseVel_k.asPair(), poseVel_tau.asPair(), t_tau - t_k);
    *Lambda = Sigma * E_tau.transpose() * Q_tau_prev_inv * F_tau_km1;
  }
  if (Psi) {
    Matrix2N E_k = computeJacobianNext_(poseVel_k.asPair(),
                                        poseVel_kp1.asPair(), t_kp1 - t_k);
    *Psi = Sigma * F_k_tau.transpose() * Q_tau_next_inv * E_k;
  }
  return Sigma;
}

template <typename PoseType>
Matrix Interpolator<PoseType>::constructMatrixFromJointMarginal(
    const JointMarginal& blockMatrix, const KeyVector& keyVector,
    size_t blockSize) {
  size_t n = keyVector.size();
  Matrix M(n * blockSize, n * blockSize);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j <= i;
         ++j) {  // symmetric block matrix: j <= i to avoid repeats
      auto block = blockMatrix(keyVector[i], keyVector[j]);  // query block
      // fill in both (i, j) and (j, i) blocks
      M.block(i * blockSize, j * blockSize, blockSize, blockSize) = block;
      if (i != j) {
        M.block(j * blockSize, i * blockSize, blockSize, blockSize) =
            block.transpose();
      }
    }
  }
  return M;
}

// ---- Explicit Instantiations ----
template class Interpolator<Point1>;
template class Interpolator<Point2>;
template class Interpolator<Point3>;
template class Interpolator<Pose2>;
template class Interpolator<Pose3>;

}  // namespace gtsam