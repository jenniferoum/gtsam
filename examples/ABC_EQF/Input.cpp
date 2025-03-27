//
// Created by darshan on 3/11/25.
//
#include "Input.h"
#include "utilities.h"
#include <Eigen/Dense>
#include <stdexcept>
#include "gtsam/geometry/Rot3.h"

Input::Input(const Vector3& w, const Matrix& Sigma)
    : w(w), Sigma(Sigma) {
    if (Sigma.rows() != 6 || Sigma.cols() != 6) {
        throw std::invalid_argument("Input measurement noise covariance must be 6x6");
    }
    
    // Check positive semi-definite
    Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(Sigma);
    if (eigensolver.eigenvalues().minCoeff() < -1e-10) {
        throw std::invalid_argument("Covariance matrix must be semi-positive definite");
    }
}

Matrix3 Input::W() const {
    return Rot3::Hat(w);
}

Input Input::random() {
    Vector3 w = Vector3::Random();
    return Input(w, Matrix::Identity(6, 6));
}