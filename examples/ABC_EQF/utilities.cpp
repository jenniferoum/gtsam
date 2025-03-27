//
// Created by darshan on 3/11/25.
//
#include "utilities.h"
#include <cmath>

// Global configuration
const std::string COORDINATE = "EXPONENTIAL";  // Alternative: "NORMAL"


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