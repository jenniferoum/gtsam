/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    LeftLinearEKF.h
 * @brief   EKF on a Lie group with a general left–linear prediction model.
 *
 * See Barrau, Axel, and Silvere Bonnabel. "Linear observed systems on groups."
 * Systems & Control Letters 129 (2019): 36-42.
 * Link: https://www.sciencedirect.com/science/article/pii/S0167691119300805
 *
 * @date    August, 2025
 * @authors Frank Dellaert
 */

#pragma once

#include <gtsam/base/Lie.h>  // For traits (needed for AdjointMap, Expmap)
#include <gtsam/navigation/LieGroupEKF.h>  // Include the base class

#include <type_traits>  // For std::conjunction, std::is_invocable_r, std::is_same

namespace gtsam {

/**
 * @class LeftLinearEKF
 * @brief EKF on a Lie group with a general left–linear prediction model.
 *
 * Discrete step: x⁺ = W · ψ(x) · U, with W,U ∈ G and ψ ∈ Aut(G).
 * For left-invariant error, the state-independent linearization is
 * A = Ad_{U^{-1}} · Φ where Φ := dψ|_e (right-trivialized). The left factor
 * W cancels in A and does not appear there.
 */
template <typename G>
class LeftLinearEKF : public LieGroupEKF<G> {
 public:
  using Base = LieGroupEKF<G>;
  using TangentVector = typename Base::TangentVector;
  using Jacobian = typename Base::Jacobian;
  using Covariance = typename Base::Covariance;

  LeftLinearEKF(const G& X0, const Covariance& P0) : Base(X0, P0) {}

  /**
   * @brief SFINAE template to check if a type satisfies the automorphism
   * concept. Specifically, it checks for the existence of:
   * - G operator()(const G&) const
   * - Jacobian dIdentity() const
   */
  template <typename Psi>
  struct is_automorphism
      : std::conjunction<
            std::is_invocable_r<G, Psi, const G&>,
            std::is_same<decltype(std::declval<Psi>().dIdentity()), Jacobian>> {
  };

  /**
   * General left–linear predict using a ψ functor and its differential at e.
   * Psi must provide: G operator()(const G&) const; and Jacobian dIdentity()
   * const. Update: X⁺ = W · ψ(X) · U;  Covariance: P⁺ = A P Aᵀ + Q with A =
   * Ad_{U^{-1}} Φ.
   */
  template <class Psi, typename = std::enable_if_t<is_automorphism<Psi>::value>>
  void predict(const G& W, const Psi& psi, const G& U, const Covariance& Q) {
    // State update: X ← W · ψ(X) · U
    this->X_ = traits<G>::Compose(traits<G>::Compose(W, psi(this->X_)), U);
    // Linearization: A = Ad_{U^{-1}} · Φ,  Φ := dψ|_e
    const G U_inv = traits<G>::Inverse(U);
    const Jacobian A = traits<G>::AdjointMap(U_inv) * psi.dIdentity();
    this->P_ = A * this->P_ * A.transpose() + Q;
  }
};

}  // namespace gtsam