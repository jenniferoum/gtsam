//
// Created by darshan on 3/11/25.
//
#include "Measurements.h"
#include <Eigen/Dense>
#include <stdexcept>

Measurement::Measurement(const Vector3& y_vec, const Vector3& d_vec, 
                        const Matrix3& Sigma, int i)
    : y(y_vec), d(d_vec), Sigma(Sigma), cal_idx(i) {
    
    // Check positive semi-definite
    Eigen::SelfAdjointEigenSolver<Matrix3> eigensolver(Sigma);
    if (eigensolver.eigenvalues().minCoeff() < -1e-10) {
        throw std::invalid_argument("Covariance matrix must be semi-positive definite");
    }
}