//
// Created by darshan on 3/11/25.
//
#include "utilities.h"
#include <cmath>

// Global configuration
const std::string COORDINATE = "EXPONENTIAL";  // Alternative: "NORMAL"

Matrix3 wedge(const Vector3& vec) {
    Matrix3 mat;
    mat << 0.0, -vec(2), vec(1),
           vec(2), 0.0, -vec(0),
           -vec(1), vec(0), 0.0;
    return mat;
}

Vector3 vee(const Matrix3& mat) {
    Vector3 vec;
    vec << mat(2, 1), mat(0, 2), mat(1, 0);
    return vec;
}

Matrix3 Rot3LeftJacobian(const Vector3& arr) {
    double angle = arr.norm();

    // Near |phi|==0, use first order Taylor expansion
    if (angle < 1e-10) {
        return Matrix3::Identity() + 0.5 * wedge(arr);
    }

    Vector3 axis = arr / angle;
    double s = sin(angle);
    double c = cos(angle);

    return (s / angle) * Matrix3::Identity() +
           (1 - (s / angle)) * (axis * axis.transpose()) +
           ((1 - c) / angle) * wedge(axis);
}

bool checkNorm(const Vector3& x, double tol) {
    return abs(x.norm() - 1) < tol || std::isnan(x.norm());
}

Matrix blockDiag(const Matrix& A, const Matrix& B) {
    if (A.size() == 0) {
        return B;
    } else if (B.size() == 0) {
        return A;
    } else {
        Matrix result(A.rows() + B.rows(), A.cols() + B.cols());
        result.setZero();
        result.block(0, 0, A.rows(), A.cols()) = A;
        result.block(A.rows(), A.cols(), B.rows(), B.cols()) = B;
        return result;
    }
}

Matrix repBlock(const Matrix& A, int n) {
    if (n <= 0) return Matrix();

    Matrix result = A;
    for (int i = 1; i < n; i++) {
        result = blockDiag(result, A);
    }
    return result;
}

Matrix numericalDifferential(std::function<Vector(const Vector&)> f, const Vector& x) {
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

        Df.col(j) = (fplus - fminus) / (2*h);
    }

    return Df;
}