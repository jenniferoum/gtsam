/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

 /**
  * @file MatrixLieGroup.h
  * @brief Base class and basic functions for Matrix Lie groups
  * @author Frank Dellaert
  */


#pragma once

#include <gtsam/base/Lie.h>
#include <type_traits>

namespace gtsam {

  namespace internal {
    // Helper to compute product of compile-time dimensions, returning Dynamic if either is Dynamic.
    constexpr int product(int a, int b) {
      return (a == Eigen::Dynamic || b == Eigen::Dynamic) ? Eigen::Dynamic : a * b;
    }

    template<class Class, int D, int N>
    auto computeVectorizedGenerators() {
      // This function is only valid for fixed-size matrices.
      // Dynamic-size groups should provide their own `VectorizedGenerators(size_t)`.
      static_assert(D != Eigen::Dynamic && N != Eigen::Dynamic,
                    "computeVectorizedGenerators is only for fixed-size Lie groups.");

      Eigen::Matrix<double, N * N, D> P_mat;
      for (int i = 0; i < D; ++i) {
        typename Class::TangentVector e_i = Class::TangentVector::Unit(i);
        typename Class::LieAlgebra G_i = Class::Hat(e_i);
        P_mat.col(i) = Eigen::Map<const Eigen::Matrix<double, N * N, 1>>(G_i.data());
      }
      return P_mat;
    }
  } // namespace internal

  /// A CRTP helper class that implements matrix Lie group methods.
  /// To use, derive from MatrixLieGroup<Class,D,N> instead of LieGroup<Class,D>.
  /// Your class must implement a `matrix()` method, static `Hat()/Vee()` methods,
  /// as well as provide a `LieAlgebra` typedef.
  template<class Class, int D, int N>
  struct MatrixLieGroup : public LieGroup<Class, D> {
    using Base = LieGroup<Class, D>;
    using Base::dimension;
    using ChartJacobian = typename Base::ChartJacobian;
    using Jacobian = typename Base::Jacobian;
    using TangentVector = typename Base::TangentVector;

    /// Return vectorized matrix representation.
    Eigen::Matrix<double, internal::product(N,N), 1> vec(OptionalJacobian<internal::product(N,N), D> H = {}) const {
      const auto& derived = static_cast<const Class&>(*this);
      const auto& T = derived.matrix();

      if constexpr (N != Eigen::Dynamic) {  // Fixed-size case
        if (H) {
          const auto& P = VectorizedGenerators();
          // The Jacobian is given by the formula H = (I_N ⊗ T) * P
          // where P is the matrix of vectorized generators.
          // This can be implemented efficiently with block-wise multiplication.
          for (int i = 0; i < N; ++i) {
            H->block(i * N, 0, N, D) = T * P.block(i * N, 0, N, D);
          }
        }
        return Eigen::Map<const Eigen::Matrix<double, N * N, 1>>(T.data());
      } else {  // Dynamic-size case
        const size_t n = T.rows();
        const size_t n2 = n * n;
        Eigen::Matrix<double, Eigen::Dynamic, 1> result(n2);
        result = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>>(T.data(), n2);

        if (H) {
          // For dynamic, VectorizedGenerators must take dimension as argument.
          // It must be a static method on the derived class. SOn has it.
          const size_t d = derived.dim();
          auto P = Class::VectorizedGenerators(n);
          H->resize(n2, d);
          for (size_t i = 0; i < n; i++) {
            H->block(i * n, 0, n, d) = T * P.block(i * n, 0, n, d);
          }
        }
        return result;
      }
    }

    /**
     * A generic implementation of AdjointMap for matrix Lie groups.
     * The Adjoint map is the tangent map of the conjugation `C_g(x) = g*x*g.inverse()`
     * at the identity. For matrix Lie groups, this is `Ad_g(v) = g*v*g.inverse()`
     * where v is an element of the Lie algebra. The columns of the Adjoint matrix
     * are `vee(g * hat(e_i) * g.inverse())` for each basis vector `e_i`.
     * This method can be overridden by derived classes with a more efficient,
     * closed-form solution.
     */
    Jacobian AdjointMap() const {
      const auto& m = static_cast<const Class&>(*this);
      size_t d = D;
      if constexpr (D == Eigen::Dynamic)  d = m.dim();
      Jacobian adj(d, d);
      const auto T_mat = m.matrix();
      const auto T_inv_mat = m.inverse().matrix();
      for (size_t i = 0; i < d; i++) {
        // TangentVector::Unit(d, i) works for both fixed and dynamic size vectors
        const auto G_i = Class::Hat(TangentVector::Unit(d, i));
        adj.col(i) = Class::Vee(T_mat * G_i * T_inv_mat);
      }
      return adj;
    }

  private:
    /// Pre-compute and store vectorized generators.
    inline static const Eigen::Matrix<double, internal::product(N, N), D>& VectorizedGenerators() {
      static_assert(
          N != Eigen::Dynamic && D != Eigen::Dynamic,
          "VectorizedGenerators without arguments is only for fixed-size Lie groups.");
      static const auto P =
          internal::computeVectorizedGenerators<Class, D, N>();
      return P;
    }
  };

  namespace internal {

    /// Adds LieAlgebra, Hat, Vee, and Vec to LieGroupTraits
    template <class Class, int N> struct MatrixLieGroupTraits : LieGroupTraits<Class> {
      using LieAlgebra = typename Class::LieAlgebra;
      using TangentVector = typename LieGroupTraits<Class>::TangentVector;

      static LieAlgebra Hat(const TangentVector& v) {
        return Class::Hat(v);
      }

      static TangentVector Vee(const LieAlgebra& X) {
        return Class::Vee(X);
      }

      /// Vectorize the matrix representation of a Lie group element.
      static Eigen::Matrix<double, product(N, N), 1> Vec(
          const Class& m,
          OptionalJacobian<product(N, N),
                           LieGroupTraits<Class>::dimension> H = {}) {
        return m.vec(H);
      }
    };

    /// Both LieGroupTraits and Testable
    template<class Class, int N> struct MatrixLieGroup : MatrixLieGroupTraits<Class, N>, Testable<Class> {};

  } // \ namespace internal

  /**
   * Matrix Lie Group Concept
   */
  template<typename T>
  class IsMatrixLieGroup : public IsLieGroup<T> {
  public:
    typedef typename traits<T>::LieAlgebra LieAlgebra;
    typedef typename traits<T>::TangentVector TangentVector;

    GTSAM_CONCEPT_USAGE(IsMatrixLieGroup) {
      // hat and vee
      X = traits<T>::Hat(xi);
      xi = traits<T>::Vee(X);
      // vec
      (void)traits<T>::Vec(g);
    }
  private:
    T g;
    LieAlgebra X;
    TangentVector xi;
  };

  /**
   *  Three term approximation of the Baker-Campbell-Hausdorff formula
   *  In non-commutative Lie groups, when composing exp(Z) = exp(X)exp(Y)
   *  it is not true that Z = X+Y. Instead, Z can be calculated using the BCH
   *  formula: Z = X + Y + [X,Y]/2 + [X-Y,[X,Y]]/12 - [Y,[X,[X,Y]]]/24
   *  http://en.wikipedia.org/wiki/Baker-Campbell-Hausdorff_formula
   */
   /// AGC: bracket() only appears in Rot3 tests, should this be used elsewhere?
  template<class T>
  T BCH(const T& X, const T& Y) {
    static const double _2 = 1. / 2., _12 = 1. / 12., _24 = 1. / 24.;
    T X_Y = bracket(X, Y);
    return T(X + Y + _2 * X_Y + _12 * bracket(X - Y, X_Y) - _24 * bracket(Y, bracket(X, X_Y)));
  }

#ifdef GTSAM_ALLOW_DEPRECATED_SINCE_V43
  /// @deprecated: use T::Hat
  template <class T>
  Matrix wedge(const Vector& x) {
    return T::Hat(x);
  }
#endif

  /**
   * Exponential map given exponential coordinates
   * class T needs a constructor from Matrix.
   * @param x exponential coordinates, vector of size n
   * @ return a T
   */
  template <class T>
  T expm(const Vector& x, int K = 7) {
    const Matrix xhat = T::Hat(x);
    return T(expm(xhat, K));
  }

} // namespace gtsam


/**
 * Macros for using the IsMatrixLieGroup
 *  - An instantiation for use inside unit tests
 *  - A typedef for use inside generic algorithms
 *
 * NOTE: intentionally not in the gtsam namespace to allow for classes not in
 * the gtsam namespace to be more easily enforced as testable
 */
#define GTSAM_CONCEPT_MATRIX_LIE_GROUP_INST(T) template class gtsam::IsMatrixLieGroup<T>;
#define GTSAM_CONCEPT_MATRIX_LIE_GROUP_TYPE(T) using _gtsam_IsMatrixLieGroup_##T = gtsam::IsMatrixLieGroup<T>;
