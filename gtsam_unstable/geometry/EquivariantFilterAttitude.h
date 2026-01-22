/**
 * @file EquivariantFilterAttitude.h
 * @brief Attitude-only EqF helper types for Python bindings.
 *
 * Provides the symmetry action used by the Unit3/Rot3 attitude example in
 * tests/testEquivariantFilter.cpp so it can be instantiated in wrappers.
 */

#pragma once

#include <gtsam/base/GroupAction.h>
#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/navigation/EquivariantFilter.h>

namespace gtsam {
namespace attitude_example {

using M = Unit3;
using G = Rot3;

/**
 * Symmetry: right group action on Unit3.
 *   φ_η(Q) = Q^T η.
 */
struct Symmetry : public GroupAction<Symmetry, G, M> {
  static constexpr ActionType type = ActionType::Right;

  M operator()(const M& eta, const G& Q,
               OptionalJacobian<2, 2> H_eta = {},
               OptionalJacobian<2, 3> H_Q = {}) const {
    return Q.unrotate(eta, H_Q, H_eta);
  }
};

}  // namespace attitude_example
}  // namespace gtsam


