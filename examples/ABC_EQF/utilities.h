//
// Created by darshan on 3/11/25.
//

#ifndef UTILITIES_H
#define UTILITIES_H

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot3.h>
#include <Eigen/Dense>
#include <functional>

using namespace gtsam;

// Global configuration
extern const std::string COORDINATE;  // "EXPONENTIAL" or "NORMAL"

/**
 * Utility functions
 */
Matrix3 wedge(const Vector3& vec);
Vector3 vee(const Matrix3& mat);
bool checkNorm(const Vector3& x, double tol = 1e-3);
Matrix blockDiag(const Matrix& A, const Matrix& B);
Matrix repBlock(const Matrix& A, int n);
Matrix numericalDifferential(std::function<Vector(const Vector&)> f, const Vector& x);
#endif //UTILITIES_H
