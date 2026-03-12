/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    GncOptimizer.cpp
 * @brief   Definitions of functions declared in header file
 * @author  Varun Agrawal
 */

#include <gtsam/nonlinear/GncOptimizer.h>

namespace gtsam {

/* ************************************************************************* */
bool isNullType(Type type) { return type == Type::NullPointer; }

/* ************************************************************************* */
bool isNonNoiseModelType(Type type) { return type == Type::NonNoiseModel; }

/* ************************************************************************* */
bool needsWeightUpdate(Type type) { return type == Type::Normal; }

/* ************************************************************************* */
bool hasNoise(Type type) {
  return type == Type::Normal || type == Type::Inlier || type == Type::Outlier;
}

}  // namespace gtsam
