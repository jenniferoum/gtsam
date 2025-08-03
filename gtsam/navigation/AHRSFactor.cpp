/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  AHRSFactor.cpp
 *  @author Krunal Chande
 *  @author Luca Carlone
 *  @author Frank Dellaert
 *  @date   July 2014
 **/

#include <gtsam/navigation/AHRSFactor.h>

#include <iostream>

using namespace std;

namespace gtsam {

//------------------------------------------------------------------------------
// Inner class PreintegratedMeasurements
//------------------------------------------------------------------------------
void PreintegratedAhrsMeasurements::print(const string& s) const {
  PreintegratedRotation::print(s);
  cout << "biasHat [" << biasHat_.transpose() << "]" << endl;
  cout << " PreintMeasCov [ " << preintMeasCov_ << " ]" << endl;
}

//------------------------------------------------------------------------------
bool PreintegratedAhrsMeasurements::equals(
    const PreintegratedAhrsMeasurements& other, double tol) const {
  return PreintegratedRotation::equals(other, tol) &&
         equal_with_abs_tol(biasHat_, other.biasHat_, tol);
}

//------------------------------------------------------------------------------
void PreintegratedAhrsMeasurements::resetIntegration() {
  PreintegratedRotation::resetIntegration();
  preintMeasCov_.setZero();
}

//------------------------------------------------------------------------------
void PreintegratedAhrsMeasurements::integrateMeasurement(
    const Vector3& measuredOmega, double deltaT) {
  Matrix3 Fr;
  PreintegratedRotation::integrateGyroMeasurement(measuredOmega, biasHat_,
                                                  deltaT, &Fr);

  // First order uncertainty propagation
  // The deltaT allows to pass from continuous time noise to discrete time
  // noise. Comparing with the IMUFactor.cpp implementation, the latter is an
  // approximation for C * (wCov / dt) * C.transpose(), with C \approx I * dt.
  preintMeasCov_ =
      Fr * preintMeasCov_ * Fr.transpose() + p().gyroscopeCovariance * deltaT;
}

//------------------------------------------------------------------------------
Rot3 PreintegratedAhrsMeasurements::predict(
    const Rot3& Ri, const Vector3& bias, gtsam::OptionalJacobian<3, 3> H1,
    gtsam::OptionalJacobian<3, 3> H2) const {
  // Correct for bias.
  Matrix3 D_biascorrected_bias;
  const Vector3 biasOmegaIncr = bias - biasHat_;
  Rot3 correctedDeltaRij = biascorrectedDeltaRij(
      biasOmegaIncr, H2 ? &D_biascorrected_bias : nullptr);

  // We handle the common case of no Coriolis correction first, in a
  // fast path that can be easily optimized by the compiler.
  if (!p().omegaCoriolis) {
    // Predict final orientation.
    const Rot3 predicted_Rj = Ri.compose(correctedDeltaRij, H1, H2);
    if (H2) *H2 *= D_biascorrected_bias;
    return predicted_Rj;
  } else {
    // 2. Calculate Coriolis effects.
    const Vector3 coriolis = integrateCoriolis(Ri);
    const Rot3 coriolisCorrection = Rot3::Expmap(-coriolis);

    // Compose corrections to get the final relative rotation.
    Matrix3 D_corrected_biascorrected, D_corrected_coriolis;
    correctedDeltaRij = correctedDeltaRij.compose(
        coriolisCorrection, H2 ? &D_corrected_biascorrected : nullptr,
        H1 ? &D_corrected_coriolis : nullptr);

    // Predict final orientation.
    Matrix3 D_predict_Ri, D_predict_delta;
    const Rot3 predicted_Rj =
        Ri.compose(correctedDeltaRij, H1 ? &D_predict_Ri : nullptr,
                   H1 || H2 ? &D_predict_delta : nullptr);

    // Chain rule for Jacobians.
    if (H1) {
      const Matrix3 D_coriolis_Ri =
          skewSymmetric(Ri.transpose() * (*p().omegaCoriolis) * deltaTij_);
      const Matrix3 D_coriolisCorrection_coriolis =
          Rot3::ExpmapDerivative(-coriolis);
      const Matrix3 D_corrected_Ri = D_corrected_coriolis *
                                     D_coriolisCorrection_coriolis *
                                     (-D_coriolis_Ri);

      // Final Jacobian wrt Ri is sum of two paths.
      *H1 = D_predict_Ri + D_predict_delta * D_corrected_Ri;
    }

    if (H2) {
      *H2 = D_predict_delta * D_corrected_biascorrected * D_biascorrected_bias;
    }

    return predicted_Rj;
  }
}

//------------------------------------------------------------------------------
Vector PreintegratedAhrsMeasurements::computeError(
    const Rot3& Ri, const Rot3& Rj, const Vector3& bias,
    gtsam::OptionalJacobian<3, 3> H1, gtsam::OptionalJacobian<3, 3> H2,
    gtsam::OptionalJacobian<3, 3> H3) const {
  // Predict orientation at time j
  Matrix3 D_predict_Ri, D_predict_bias;
  Rot3 predicted_Rj = predict(Ri, bias, H1 ? &D_predict_Ri : nullptr,
                              H3 ? &D_predict_bias : nullptr);

  // Compute the error vector
  Matrix3 D_error_Rj, D_error_predict;
  Vector3 error = Rj.localCoordinates(predicted_Rj, H2 ? &D_error_Rj : nullptr,
                                      H1 || H3 ? &D_error_predict : nullptr);

  // Jacobians using the chain rule
  if (H1) *H1 = D_error_predict * D_predict_Ri;
  if (H2) *H2 = D_error_Rj;
  if (H3) *H3 = D_error_predict * D_predict_bias;

  return error;
}

//------------------------------------------------------------------------------
Vector3 PreintegratedAhrsMeasurements::DeltaAngles(
    const Vector3& msr_gyro_t, double msr_dt, const Vector3& delta_angles) {
  // Note: all delta terms refer to an IMU\sensor system at t0

  // Calculate the corrected measurements using the Bias object
  Vector3 body_t_omega_body = msr_gyro_t;

  Rot3 R_t_to_t0 = Rot3::Expmap(delta_angles);

  R_t_to_t0 = R_t_to_t0 * Rot3::Expmap(body_t_omega_body * msr_dt);
  return Rot3::Logmap(R_t_to_t0);
}

//------------------------------------------------------------------------------
// AHRSFactor methods
//------------------------------------------------------------------------------
AHRSFactor::AHRSFactor(
    Key Ri, Key rot_j, Key bias,
    const PreintegratedAhrsMeasurements& preintegratedMeasurements)
    : Base(noiseModel::Gaussian::Covariance(
               preintegratedMeasurements.preintMeasCov_),
           Ri, rot_j, bias),
      _PIM_(preintegratedMeasurements) {}

gtsam::NonlinearFactor::shared_ptr AHRSFactor::clone() const {
  //------------------------------------------------------------------------------
  return std::static_pointer_cast<gtsam::NonlinearFactor>(
      gtsam::NonlinearFactor::shared_ptr(new This(*this)));
}

//------------------------------------------------------------------------------
void AHRSFactor::print(const string& s,
                       const KeyFormatter& keyFormatter) const {
  cout << s << "AHRSFactor(" << keyFormatter(this->key<1>()) << ","
       << keyFormatter(this->key<2>()) << "," << keyFormatter(this->key<3>())
       << ",";
  _PIM_.print("  preintegrated measurements:");
  noiseModel_->print("  noise model: ");
}

//------------------------------------------------------------------------------
bool AHRSFactor::equals(const NonlinearFactor& other, double tol) const {
  const This* e = dynamic_cast<const This*>(&other);
  return e != nullptr && Base::equals(*e, tol) && _PIM_.equals(e->_PIM_, tol);
}

//------------------------------------------------------------------------------
Vector AHRSFactor::evaluateError(const Rot3& Ri, const Rot3& Rj,
                                 const Vector3& bias, OptionalMatrixType H1,
                                 OptionalMatrixType H2,
                                 OptionalMatrixType H3) const {
  return _PIM_.computeError(Ri, Rj, bias, H1, H2, H3);
}

//------------------------------------------------------------------------------
Rot3 AHRSFactor::Predict(const Rot3& Ri, const Vector3& bias,
                         const PreintegratedAhrsMeasurements& pim) {
  return pim.predict(Ri, bias);
}

//------------------------------------------------------------------------------
AHRSFactor::AHRSFactor(Key Ri, Key rot_j, Key bias,
                       const PreintegratedAhrsMeasurements& pim,
                       const Vector3& omegaCoriolis,
                       const std::optional<Pose3>& body_P_sensor)
    : Base(noiseModel::Gaussian::Covariance(pim.preintMeasCov_), Ri, rot_j,
           bias),
      _PIM_(pim) {
  auto p = std::make_shared<PreintegratedAhrsMeasurements::Params>(pim.p());
  p->body_P_sensor = body_P_sensor;
  _PIM_.p_ = p;
}

//------------------------------------------------------------------------------
Rot3 AHRSFactor::predict(const Rot3& Ri, const Vector3& bias,
                         const PreintegratedAhrsMeasurements& pim,
                         const Vector3& omegaCoriolis,
                         const std::optional<Pose3>& body_P_sensor) {
  auto p = std::make_shared<PreintegratedAhrsMeasurements::Params>(pim.p());
  p->omegaCoriolis = omegaCoriolis;
  p->body_P_sensor = body_P_sensor;
  PreintegratedAhrsMeasurements newPim = pim;
  newPim.p_ = p;
  return Predict(Ri, bias, newPim);
}

}  // namespace gtsam
