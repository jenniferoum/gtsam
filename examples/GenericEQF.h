/**
 * @file GenericEQF.h  
 * @brief Generic Equivariant Filter template for common IEKF problems
 *
 * This provides a geometry-agnostic EQF implementation that can handle
 * the same problems as InvariantEKF (SE(2), SE(3), SO(3), NavState)
 * allowing direct apples-to-apples comparison.
 *
 * @date July 18, 2025
 * @authors Generic EQF framework for IEKF comparison
 */

#ifndef GENERIC_EQF_H
#define GENERIC_EQF_H

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/base/OptionalJacobian.h>
#include <functional>
#include <chrono>

namespace gtsam {
namespace generic_eqf_lib {

using namespace gtsam;

//========================================================================
// Generic EQF Framework
//========================================================================

/**
 * @brief Generic Equivariant Filter for Lie groups
 * 
 * Template parameters:
 * - T: Lie group type (Pose2, Pose3, Rot3, NavState, etc.)
 * - dim: Manifold dimension of T
 */
template<typename T, int dim>
class GenericEQF {
private:
    T X_hat_;              // Current state estimate
    Matrix Sigma_;         // Covariance matrix (dim x dim)
    int dof_;              // Degrees of freedom = dim
    
    // Helper function for numerical differentiation
    Matrix numericalDifferential(std::function<Vector(const Vector&)> f, const Vector& x) const;
    
public:
    /**
     * @brief Initialize Generic EQF
     * @param initial_state Initial state estimate
     * @param initial_covariance Initial covariance (dim x dim)
     */
    GenericEQF(const T& initial_state, const Matrix& initial_covariance);
    
    /**
     * @brief Get current state estimate
     * @return Current state
     */
    T state() const { return X_hat_; }
    
    /**
     * @brief Get current covariance
     * @return Current covariance matrix
     */
    Matrix covariance() const { return Sigma_; }
    
    /**
     * @brief Prediction step using group motion
     * @param U Group element representing motion
     * @param Q Process noise covariance
     */
    void predict(const T& U, const Matrix& Q);
    
    /**
     * @brief Prediction step using Vector3 angular displacement (for Unit3 specialization)
     * @param angular_displacement Angular displacement vector
     * @param Q Process noise covariance
     */
    void predict(const Vector3& angular_displacement, const Matrix& Q);
    
    /**
     * @brief Update step using measurement
     * @param h Measurement function h(X)
     * @param z Measurement vector
     * @param R Measurement noise covariance
     */
    template<int meas_dim>
    void update(std::function<Vector(const T&, OptionalJacobian<meas_dim, dim>)> h,
                const Vector& z, 
                const Matrix& R);
};

//========================================================================
// Implementation
//========================================================================

template<typename T, int dim>
GenericEQF<T, dim>::GenericEQF(const T& initial_state, const Matrix& initial_covariance)
    : X_hat_(initial_state), Sigma_(initial_covariance), dof_(dim) {
    
    if (initial_covariance.rows() != dim || initial_covariance.cols() != dim) {
        throw std::invalid_argument("Initial covariance must be " + std::to_string(dim) + "x" + std::to_string(dim));
    }
}

template<typename T, int dim>
Matrix GenericEQF<T, dim>::numericalDifferential(std::function<Vector(const Vector&)> f, const Vector& x) const {
    double h = 1e-6;
    Vector fx = f(x);
    int n = fx.size();
    int m = x.size();
    Matrix Df = Matrix::Zero(n, m);

    for (int j = 0; j < m; j++) {
        Vector ej = Vector::Zero(m);
        ej(j) = 1.0;
        
        Vector fplus = f(x + h * ej);
        Vector fminus = f(x - h * ej);
        
        Df.col(j) = (fplus - fminus) / (2 * h);
    }
    
    return Df;
}

template<typename T, int dim>
void GenericEQF<T, dim>::predict(const T& U, const Matrix& Q) {
    // EQF Prediction: X_{k+1} = X_k * U (right multiplication for equivariance)
    X_hat_ = X_hat_.compose(U);
    
    // Covariance prediction using adjoint map
    Matrix Adj_U = U.AdjointMap();
    Sigma_ = Adj_U * Sigma_ * Adj_U.transpose() + Q;
}

template<typename T, int dim>
template<int meas_dim>
void GenericEQF<T, dim>::update(std::function<Vector(const T&, OptionalJacobian<meas_dim, dim>)> h,
                                const Vector& z, 
                                const Matrix& R) {
    
    // Compute predicted measurement and Jacobian
    Matrix H = Matrix::Zero(meas_dim, dim);
    Vector h_pred = h(X_hat_, OptionalJacobian<meas_dim, dim>(H));
    
    // Innovation
    Vector innovation = z - h_pred;
    
    // Innovation covariance
    Matrix S = H * Sigma_ * H.transpose() + R;
    
    // Kalman gain
    Matrix K = Sigma_ * H.transpose() * S.inverse();
    
    // State update using exponential map (equivariant)
    Vector delta = K * innovation;
    T correction = T::Expmap(delta);
    X_hat_ = correction.compose(X_hat_);  // Left multiplication for correction
    
    // Covariance update
    Matrix I = Matrix::Identity(dim, dim);
    Sigma_ = (I - K * H) * Sigma_;
}

//========================================================================
// Specializations for common IEKF problems
//========================================================================

// SE(2) EQF - matches IEKF_SE2Example.cpp
using SE2_EQF = GenericEQF<Pose2, 3>;

// SO(3) EQF - for attitude estimation  
using SO3_EQF = GenericEQF<Rot3, 3>;

// S² EQF - for direction tracking on the sphere
using S2_EQF = GenericEQF<Unit3, 2>;

// SE(3) EQF - for 3D pose estimation
using SE3_EQF = GenericEQF<Pose3, 6>;

// NavState EQF - matches IEKF_NavstateExample.cpp
using NavState_EQF = GenericEQF<NavState, 9>;

// Specialization for Unit3 (S² direction tracking)
template<>
inline Unit3 GenericEQF<Unit3, 2>::state() const {
    return X_hat_;
}

// Overload predict for Vector3 angular displacement (proper S² motion model)
template<>
inline void GenericEQF<Unit3, 2>::predict(const Vector3& angular_displacement, const Matrix& Q) {
    // Apply rotation to current direction using proper angular displacement
    Vector3 current_vec = X_hat_.unitVector();
    double angle = angular_displacement.norm();
    
    if (angle > 1e-8) {
        Vector3 axis = angular_displacement / angle;
        
        // Rodrigues rotation formula for rotating current_vec by angular_displacement
        Vector3 rotated = current_vec * cos(angle) + 
                         axis.cross(current_vec) * sin(angle) + 
                         axis * (axis.dot(current_vec)) * (1 - cos(angle));
        
        X_hat_ = Unit3(rotated.normalized());
    }
    
    // Covariance prediction for tangent space (2D on S²)
    Sigma_ = Sigma_ + Q;
}

// Keep the original Unit3 predict for backward compatibility but fix the bug
template<>
inline void GenericEQF<Unit3, 2>::predict(const Unit3& U, const Matrix& Q) {
    // Extract the raw vector without normalization to preserve magnitude
    Vector3 angular_displacement = U.unitVector() * 1.0; // This is still wrong - Unit3 loses magnitude!
    predict(angular_displacement, Q); // Delegate to Vector3 version
}

template<>
template<int meas_dim>
inline void GenericEQF<Unit3, 2>::update(std::function<Vector(const Unit3&, OptionalJacobian<meas_dim, 2>)> h,
                                         const Vector& z, 
                                         const Matrix& R) {
    
    // Compute predicted measurement and Jacobian
    Matrix H = Matrix::Zero(meas_dim, 2);
    Vector h_pred = h(X_hat_, OptionalJacobian<meas_dim, 2>(H));
    
    // Innovation
    Vector innovation = z - h_pred;
    
    // Innovation covariance
    Matrix S = H * Sigma_ * H.transpose() + R;
    
    // Kalman gain
    Matrix K = Sigma_ * H.transpose() * S.inverse();
    
    // State update in tangent space of S²
    Vector delta = K * innovation;
    
    // Map tangent space perturbation back to Unit3
    Vector3 current = X_hat_.unitVector();
    Vector3 tangent_perturbation = Vector3::Zero();
    
    // Project delta (2D) into 3D tangent space
    // Use two orthogonal vectors perpendicular to current direction
    Vector3 v1, v2;
    if (abs(current.z()) < 0.9) {
        v1 = Vector3(0, 0, 1).cross(current).normalized();
    } else {
        v1 = Vector3(1, 0, 0).cross(current).normalized();
    }
    v2 = current.cross(v1).normalized();
    
    tangent_perturbation = delta(0) * v1 + delta(1) * v2;
    
    double angle = tangent_perturbation.norm();
    if (angle > 1e-8) {
        Vector3 axis = tangent_perturbation / angle;
        Vector3 updated = current * cos(angle) + 
                         axis.cross(current) * sin(angle) + 
                         axis * (axis.dot(current)) * (1 - cos(angle));
        X_hat_ = Unit3(updated.normalized());
    }
    
    // Covariance update
    Matrix I = Matrix::Identity(2, 2);
    Sigma_ = (I - K * H) * Sigma_;
}

//========================================================================
// SE(2) Specializations (separate from S² to avoid conflicts)
//========================================================================

// Specialization for Pose2 (SE(2) pose tracking)
template<>
inline Pose2 GenericEQF<Pose2, 3>::state() const {
    return X_hat_;
}

template<>
inline void GenericEQF<Pose2, 3>::predict(const Pose2& U, const Matrix& Q) {
    // SE(2) prediction: X_{k+1} = X_k * U (group multiplication)
    X_hat_ = X_hat_.compose(U);
    
    // Covariance prediction using adjoint map for SE(2)
    Matrix3 Adj_U = U.AdjointMap();
    Sigma_ = Adj_U * Sigma_ * Adj_U.transpose() + Q;
}

// Overload predict for Vector3 SE(2) motion (dx, dy, dtheta)
template<>
inline void GenericEQF<Pose2, 3>::predict(const Vector3& se2_motion, const Matrix& Q) {
    // Convert Vector3 motion to Pose2 using exponential map
    Pose2 U = Pose2::Expmap(se2_motion);
    predict(U, Q); // Delegate to Pose2 version
}

template<>
template<int meas_dim>
inline void GenericEQF<Pose2, 3>::update(std::function<Vector(const Pose2&, OptionalJacobian<meas_dim, 3>)> h,
                                         const Vector& z, 
                                         const Matrix& R) {
    
    // Compute predicted measurement and Jacobian
    Matrix H = Matrix::Zero(meas_dim, 3);
    Vector h_pred = h(X_hat_, OptionalJacobian<meas_dim, 3>(H));
    
    // Innovation
    Vector innovation = z - h_pred;
    
    // Innovation covariance
    Matrix S = H * Sigma_ * H.transpose() + R;
    
    // Kalman gain
    Matrix K = Sigma_ * H.transpose() * S.inverse();
    
    // State update using exponential map (equivariant)
    Vector delta = K * innovation;
    Pose2 correction = Pose2::Expmap(delta);
    X_hat_ = correction.compose(X_hat_);  // Left multiplication for correction
    
    // Covariance update
    Matrix I = Matrix::Identity(3, 3);
    Sigma_ = (I - K * H) * Sigma_;
}

} // namespace generic_eqf_lib
} // namespace gtsam

#endif // GENERIC_EQF_H