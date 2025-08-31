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
  static constexpr int Dim = Base::Dim;  ///< Compile-time dimension of G.
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
   * General left–linear dynamics using a ψ functor and its differential at e.
   * Returns W · ψ(X) · U, and optional Jacobian A = Ad_{U^{-1}} Φ,  Φ := dψ|_e.
   */
  template <class Psi, typename = std::enable_if_t<is_automorphism<Psi>::value>>
  static G dynamics(const G& W, const Psi& psi, const G& X, const G& U,
                    OptionalJacobian<Dim, Dim> A = {}) {
    if (A) {
      const G U_inv = traits<G>::Inverse(U);
      *A = traits<G>::AdjointMap(U_inv) * psi.dIdentity();
    }
    return W * psi(X) * U;
  }

  /**
   * General left–linear predict using a ψ functor and its differential at e.
   *   Update: X⁺ = W · ψ(X) · U
   *   Covariance: P⁺ = A P Aᵀ + Q with A = Ad_{U^{-1}} Φ,  Φ := dψ|_e.
   */
  template <class Psi, typename = std::enable_if_t<is_automorphism<Psi>::value>>
  void predict(const G& W, const Psi& psi, const G& U, const Covariance& Q) {
    Jacobian A;
    this->X_ = this->dynamics(W, psi, this->X_, U, A);
    this->P_ = A * this->P_ * A.transpose() + Q;
  }
};

}  // namespace gtsam