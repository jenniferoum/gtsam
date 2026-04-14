/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    numericalDerivative.h
 * @brief   Numerical derivative helpers for manifold-valued functions
 *
 * This file is organized in three layers:
 * - `internal` contains small traits that select Jacobian matrix types and
 *   deduce callable output types.
 * - `numericalGradient` and `numericalDerivative11` provide the scalar-gradient
 *   and unary central-difference kernels.
 * - The higher-arity `numericalDerivativeXY` wrappers adapt multi-argument
 *   callables onto the unary kernel, while the Hessian helpers at the end build
 *   on the gradient and Jacobian routines.
 * @author  Frank Dellaert
 */

// \callgraph
#pragma once

#include <gtsam/base/Lie.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/nonlinear/Values.h>

#include <functional>
#include <type_traits>
#include <utility>

/**
 * @defgroup numerical_derivatives Numerical Derivative Helpers
 * Finite-difference Jacobian and Hessian utilities for manifold-valued
 * functions.
 */
namespace gtsam {

/** @addtogroup numerical_derivatives */
/// @{

/**
 * The Jacobian helpers accept generic callables. `numericalDerivative11`
 * implements the central-difference kernel for a unary function. The higher
 * arity helpers fix the non-differentiated arguments with a small lambda and
 * forward to that kernel. Thin raw function-reference overloads remain to give
 * overload resolution context for overloaded free functions.
 */

/** @name Internal Type Helpers */
/// @{
namespace internal {
/// Select the fixed-size Jacobian type associated with two manifold types.
template <class Y, class X = double>
struct FixedSizeMatrix {
  typedef Eigen::Matrix<double, traits<Y>::dimension, traits<X>::dimension>
      type;
};

/// Select a fixed-size matrix from explicit row and column counts.
template <int M, int N>
struct MatrixMN {
  typedef Eigen::Matrix<double, M, N> type;
};

/// Marker type used when the callable return type should be deduced.
struct DeducedOutput {};

template <class RequestedY, class F, class... Args>
using OutputType =
    std::conditional_t<std::is_same_v<RequestedY, DeducedOutput>,
                       std::decay_t<std::invoke_result_t<F&, const Args&...>>,
                       RequestedY>;

template <class F, class... Args>
using EnableIfInvocable =
    std::enable_if_t<std::is_invocable_v<F&, const Args&...>, int>;
}  // namespace internal
/// @}

/** @name First-Order Derivative Helpers */
/// @{
/**
 * @par Template requirements
 * - Callable argument: `h` must be invocable with `const` references to the
 *   corresponding argument types (`const X&`, `const X1&`, ...).
 * - Output type: `Y` (or the deduced output type when `Y` is omitted) must be
 *   a manifold type (`traits<Y>::structure_category` derives from
 *   `manifold_tag`) and support `traits<Y>::Local` and
 *   `traits<Y>::GetDimension`.
 * - Differentiated input type: only the argument being differentiated (`X`,
 *   `X1`, `X2`, ...) must be a manifold type and support `traits<X>::Retract`.
 * - Other input types: non-differentiated arguments only need to satisfy
 *   callable invocation of `h`.
 * - Dimension parameter: `N` must be positive; for variable-size manifold
 *   types, `N` must be provided explicitly.
 */
/**
 * @brief Numerically compute gradient of scalar function
 * @return n-dimensional gradient computed via central differencing
 * @tparam X manifold input type
 * @tparam N tangent dimension of `X`; provide explicitly for variable-size
 * manifold types
 */
template <class X, int N = traits<X>::dimension>
typename Eigen::Matrix<double, N, 1> numericalGradient(
    std::function<double(const X&)> h, const X& x, double delta = 1e-5) {
  double factor = 1.0 / (2.0 * delta);

  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X>::structure_category>,
                "Template argument X must be a manifold type.");
  static_assert(
      N > 0,
      "Template argument X must be fixed-size type or N must be specified.");

  // Prepare a tangent vector to perturb x with.
  Eigen::Matrix<double, N, 1> d;
  d.setZero();

  Eigen::Matrix<double, N, 1> g;
  g.setZero();
  for (int j = 0; j < N; j++) {
    d(j) = delta;
    double hxplus = h(traits<X>::Retract(x, d));
    d(j) = -delta;
    double hxmin = h(traits<X>::Retract(x, d));
    d(j) = 0;
    g(j) = (hxplus - hxmin) * factor;
  }
  return g;
}

/**
 * @brief New-style numerical derivatives using manifold_traits
 * @brief Computes numerical derivative in argument 1 of unary function
 * @param h unary function yielding m-vector
 * @param x n-dimensional value at which to evaluate h
 * @param delta increment for numerical derivative
 * @tparam Y output manifold type; defaults to the deduced callable output
 * @tparam X differentiated manifold input type
 * @tparam N tangent dimension of `X`; provide explicitly for variable-size
 * manifold types
 * @return m*n Jacobian computed via central differencing
 */
template <class Y = internal::DeducedOutput, class X,
          int N = traits<X>::dimension, class F,
          internal::EnableIfInvocable<F, X> = 0>
typename internal::MatrixMN<traits<internal::OutputType<Y, F, X>>::dimension,
                            N>::type
numericalDerivative11(F&& h, const X& x, double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X>;
  typedef
      typename internal::MatrixMN<traits<ActualY>::dimension, N>::type Matrix;

  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  typedef traits<ActualY> TraitsY;

  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X>::structure_category>,
                "Template argument X must be a manifold type.");
  static_assert(
      N > 0,
      "Template argument X must be fixed-size type or N must be specified.");
  typedef traits<X> TraitsX;

  // get value at x, and corresponding chart
  const ActualY hx = std::invoke(h, x);

  // Find number of rows m
  const size_t m = TraitsY::GetDimension(hx);

  // Prepare a tangent vector to perturb x with
  Eigen::Matrix<double, N, 1> dx;
  dx.setZero();

  // Fill in Jacobian H
  Matrix H = Matrix::Zero(m, N);
  const double factor = 1.0 / (2.0 * delta);
  for (int j = 0; j < N; j++) {
    dx(j) = delta;
    const auto dy1 =
        TraitsY::Local(hx, std::invoke(h, TraitsX::Retract(x, dx)));
    dx(j) = -delta;
    const auto dy2 =
        TraitsY::Local(hx, std::invoke(h, TraitsX::Retract(x, dx)));
    dx(j) = 0;
    H.col(j) = (dy1 - dy2) * factor;
  }
  return H;
}

template <class Y, class X, int N = traits<X>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative11(Y (&h)(const X&), const X& x, double delta = 1e-5) {
  return numericalDerivative11<Y, X, N>([&](const X& x_) { return h(x_); }, x,
                                        delta);
}

/**
 * Compute numerical derivative in argument 1 of binary function
 * @param h binary function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X1 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2,
          int N = traits<X1>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2>>::dimension, N>::type
numericalDerivative21(F&& h, const X1& x1, const X2& x2, double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X1>::structure_category>,
                "Template argument X1 must be a manifold type.");
  return numericalDerivative11<ActualY, X1, N>(
      [&](const X1& x1_) { return std::invoke(h, x1_, x2); }, x1, delta);
}

template <class Y, class X1, class X2, int N = traits<X1>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative21(Y (&h)(const X1&, const X2&), const X1& x1, const X2& x2,
                      double delta = 1e-5) {
  return numericalDerivative11<Y, X1, N>(
      [&](const X1& x1_) { return h(x1_, x2); }, x1, delta);
}

/**
 * Compute numerical derivative in argument 2 of binary function
 * @param h binary function yielding m-vector
 * @param x1 first argument value
 * @param x2 n-dimensional second argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X2 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2,
          int N = traits<X2>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2>>::dimension, N>::type
numericalDerivative22(F&& h, const X1& x1, const X2& x2, double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2>;
  //  static_assert( (std::is_base_of<gtsam::manifold_tag, typename
  //  traits<X1>::structure_category>::value),
  //       "Template argument X1 must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X2>::structure_category>,
                "Template argument X2 must be a manifold type.");
  return numericalDerivative11<ActualY, X2, N>(
      [&](const X2& x2_) { return std::invoke(h, x1, x2_); }, x2, delta);
}

template <class Y, class X1, class X2, int N = traits<X2>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative22(Y (&h)(const X1&, const X2&), const X1& x1, const X2& x2,
                      double delta = 1e-5) {
  return numericalDerivative11<Y, X2, N>(
      [&](const X2& x2_) { return h(x1, x2_); }, x2, delta);
}

/**
 * Compute numerical derivative in argument 1 of ternary function
 * @param h ternary function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X1 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          int N = traits<X1>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3>>::dimension, N>::type
numericalDerivative31(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X1>::structure_category>,
                "Template argument X1 must be a manifold type.");
  return numericalDerivative11<ActualY, X1, N>(
      [&](const X1& x1_) { return std::invoke(h, x1_, x2, x3); }, x1, delta);
}

template <class Y, class X1, class X2, class X3, int N = traits<X1>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative31(Y (&h)(const X1&, const X2&, const X3&), const X1& x1,
                      const X2& x2, const X3& x3, double delta = 1e-5) {
  return numericalDerivative11<Y, X1, N>(
      [&](const X1& x1_) { return h(x1_, x2, x3); }, x1, delta);
}

/**
 * Compute numerical derivative in argument 2 of ternary function
 * @param h ternary function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X2 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          int N = traits<X2>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3>>::dimension, N>::type
numericalDerivative32(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X2>::structure_category>,
                "Template argument X2 must be a manifold type.");
  return numericalDerivative11<ActualY, X2, N>(
      [&](const X2& x2_) { return std::invoke(h, x1, x2_, x3); }, x2, delta);
}

template <class Y, class X1, class X2, class X3, int N = traits<X2>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative32(Y (&h)(const X1&, const X2&, const X3&), const X1& x1,
                      const X2& x2, const X3& x3, double delta = 1e-5) {
  return numericalDerivative11<Y, X2, N>(
      [&](const X2& x2_) { return h(x1, x2_, x3); }, x2, delta);
}

/**
 * Compute numerical derivative in argument 3 of ternary function
 * @param h ternary function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X3 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          int N = traits<X3>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3>>::dimension, N>::type
numericalDerivative33(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X3>::structure_category>,
                "Template argument X3 must be a manifold type.");
  return numericalDerivative11<ActualY, X3, N>(
      [&](const X3& x3_) { return std::invoke(h, x1, x2, x3_); }, x3, delta);
}

template <class Y, class X1, class X2, class X3, int N = traits<X3>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative33(Y (&h)(const X1&, const X2&, const X3&), const X1& x1,
                      const X2& x2, const X3& x3, double delta = 1e-5) {
  return numericalDerivative11<Y, X3, N>(
      [&](const X3& x3_) { return h(x1, x2, x3_); }, x3, delta);
}

/**
 * Compute numerical derivative in argument 1 of 4-argument function
 * @param h quartic function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param x4 fourth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X1 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, int N = traits<X1>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4>>::dimension, N>::type
numericalDerivative41(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X1>::structure_category>,
                "Template argument X1 must be a manifold type.");
  return numericalDerivative11<ActualY, X1, N>(
      [&](const X1& x1_) { return std::invoke(h, x1_, x2, x3, x4); }, x1,
      delta);
}

template <class Y, class X1, class X2, class X3, class X4,
          int N = traits<X1>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative41(Y (&h)(const X1&, const X2&, const X3&, const X4&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      double delta = 1e-5) {
  return numericalDerivative11<Y, X1, N>(
      [&](const X1& x1_) { return h(x1_, x2, x3, x4); }, x1, delta);
}

/**
 * Compute numerical derivative in argument 2 of 4-argument function
 * @param h quartic function yielding m-vector
 * @param x1 first argument value
 * @param x2 n-dimensional second argument value
 * @param x3 third argument value
 * @param x4 fourth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X2 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, int N = traits<X2>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4>>::dimension, N>::type
numericalDerivative42(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X2>::structure_category>,
                "Template argument X2 must be a manifold type.");
  return numericalDerivative11<ActualY, X2, N>(
      [&](const X2& x2_) { return std::invoke(h, x1, x2_, x3, x4); }, x2,
      delta);
}

template <class Y, class X1, class X2, class X3, class X4,
          int N = traits<X2>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative42(Y (&h)(const X1&, const X2&, const X3&, const X4&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      double delta = 1e-5) {
  return numericalDerivative11<Y, X2, N>(
      [&](const X2& x2_) { return h(x1, x2_, x3, x4); }, x2, delta);
}

/**
 * Compute numerical derivative in argument 3 of 4-argument function
 * @param h quartic function yielding m-vector
 * @param x1 first argument value
 * @param x2 second argument value
 * @param x3 n-dimensional third argument value
 * @param x4 fourth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X3 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, int N = traits<X3>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4>>::dimension, N>::type
numericalDerivative43(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X3>::structure_category>,
                "Template argument X3 must be a manifold type.");
  return numericalDerivative11<ActualY, X3, N>(
      [&](const X3& x3_) { return std::invoke(h, x1, x2, x3_, x4); }, x3,
      delta);
}

template <class Y, class X1, class X2, class X3, class X4,
          int N = traits<X3>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative43(Y (&h)(const X1&, const X2&, const X3&, const X4&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      double delta = 1e-5) {
  return numericalDerivative11<Y, X3, N>(
      [&](const X3& x3_) { return h(x1, x2, x3_, x4); }, x3, delta);
}

/**
 * Compute numerical derivative in argument 4 of 4-argument function
 * @param h quartic function yielding m-vector
 * @param x1 first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param x4 n-dimensional fourth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X4 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, int N = traits<X4>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4>>::dimension, N>::type
numericalDerivative44(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X4>::structure_category>,
                "Template argument X4 must be a manifold type.");
  return numericalDerivative11<ActualY, X4, N>(
      [&](const X4& x4_) { return std::invoke(h, x1, x2, x3, x4_); }, x4,
      delta);
}

template <class Y, class X1, class X2, class X3, class X4,
          int N = traits<X4>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative44(Y (&h)(const X1&, const X2&, const X3&, const X4&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      double delta = 1e-5) {
  return numericalDerivative11<Y, X4, N>(
      [&](const X4& x4_) { return h(x1, x2, x3, x4_); }, x4, delta);
}

/**
 * Compute numerical derivative in argument 1 of 5-argument function
 * @param h quintic function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param x4 fourth argument value
 * @param x5 fifth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X1 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, class X5, int N = traits<X1>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4, X5> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4, X5>>::dimension, N>::type
numericalDerivative51(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, const X5& x5, double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4, X5>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X1>::structure_category>,
                "Template argument X1 must be a manifold type.");
  return numericalDerivative11<ActualY, X1, N>(
      [&](const X1& x1_) { return std::invoke(h, x1_, x2, x3, x4, x5); }, x1,
      delta);
}

template <class Y, class X1, class X2, class X3, class X4, class X5,
          int N = traits<X1>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative51(Y (&h)(const X1&, const X2&, const X3&, const X4&,
                             const X5&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      const X5& x5, double delta = 1e-5) {
  return numericalDerivative11<Y, X1, N>(
      [&](const X1& x1_) { return h(x1_, x2, x3, x4, x5); }, x1, delta);
}

/**
 * Compute numerical derivative in argument 2 of 5-argument function
 * @param h quintic function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param x4 fourth argument value
 * @param x5 fifth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X2 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, class X5, int N = traits<X2>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4, X5> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4, X5>>::dimension, N>::type
numericalDerivative52(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, const X5& x5, double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4, X5>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X2>::structure_category>,
                "Template argument X2 must be a manifold type.");
  return numericalDerivative11<ActualY, X2, N>(
      [&](const X2& x2_) { return std::invoke(h, x1, x2_, x3, x4, x5); }, x2,
      delta);
}

template <class Y, class X1, class X2, class X3, class X4, class X5,
          int N = traits<X2>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative52(Y (&h)(const X1&, const X2&, const X3&, const X4&,
                             const X5&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      const X5& x5, double delta = 1e-5) {
  return numericalDerivative11<Y, X2, N>(
      [&](const X2& x2_) { return h(x1, x2_, x3, x4, x5); }, x2, delta);
}

/**
 * Compute numerical derivative in argument 3 of 5-argument function
 * @param h quintic function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param x4 fourth argument value
 * @param x5 fifth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X3 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, class X5, int N = traits<X3>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4, X5> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4, X5>>::dimension, N>::type
numericalDerivative53(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, const X5& x5, double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4, X5>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X3>::structure_category>,
                "Template argument X3 must be a manifold type.");
  return numericalDerivative11<ActualY, X3, N>(
      [&](const X3& x3_) { return std::invoke(h, x1, x2, x3_, x4, x5); }, x3,
      delta);
}

template <class Y, class X1, class X2, class X3, class X4, class X5,
          int N = traits<X3>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative53(Y (&h)(const X1&, const X2&, const X3&, const X4&,
                             const X5&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      const X5& x5, double delta = 1e-5) {
  return numericalDerivative11<Y, X3, N>(
      [&](const X3& x3_) { return h(x1, x2, x3_, x4, x5); }, x3, delta);
}

/**
 * Compute numerical derivative in argument 4 of 5-argument function
 * @param h quintic function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param x4 fourth argument value
 * @param x5 fifth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X4 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, class X5, int N = traits<X4>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4, X5> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4, X5>>::dimension, N>::type
numericalDerivative54(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, const X5& x5, double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4, X5>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X4>::structure_category>,
                "Template argument X4 must be a manifold type.");
  return numericalDerivative11<ActualY, X4, N>(
      [&](const X4& x4_) { return std::invoke(h, x1, x2, x3, x4_, x5); }, x4,
      delta);
}

template <class Y, class X1, class X2, class X3, class X4, class X5,
          int N = traits<X4>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative54(Y (&h)(const X1&, const X2&, const X3&, const X4&,
                             const X5&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      const X5& x5, double delta = 1e-5) {
  return numericalDerivative11<Y, X4, N>(
      [&](const X4& x4_) { return h(x1, x2, x3, x4_, x5); }, x4, delta);
}

/**
 * Compute numerical derivative in argument 5 of 5-argument function
 * @param h quintic function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param x4 fourth argument value
 * @param x5 fifth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X5 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, class X5, int N = traits<X5>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4, X5> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4, X5>>::dimension, N>::type
numericalDerivative55(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, const X5& x5, double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4, X5>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X5>::structure_category>,
                "Template argument X5 must be a manifold type.");
  return numericalDerivative11<ActualY, X5, N>(
      [&](const X5& x5_) { return std::invoke(h, x1, x2, x3, x4, x5_); }, x5,
      delta);
}

template <class Y, class X1, class X2, class X3, class X4, class X5,
          int N = traits<X5>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative55(Y (&h)(const X1&, const X2&, const X3&, const X4&,
                             const X5&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      const X5& x5, double delta = 1e-5) {
  return numericalDerivative11<Y, X5, N>(
      [&](const X5& x5_) { return h(x1, x2, x3, x4, x5_); }, x5, delta);
}

/**
 * Compute numerical derivative in argument 1 of 6-argument function
 * @param h quintic function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param x4 fourth argument value
 * @param x5 fifth argument value
 * @param x6 sixth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X1 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, class X5, class X6, int N = traits<X1>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4, X5, X6> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4, X5, X6>>::dimension,
    N>::type
numericalDerivative61(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, const X5& x5, const X6& x6,
                      double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4, X5, X6>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X1>::structure_category>,
                "Template argument X1 must be a manifold type.");
  return numericalDerivative11<ActualY, X1, N>(
      [&](const X1& x1_) { return std::invoke(h, x1_, x2, x3, x4, x5, x6); },
      x1, delta);
}

template <class Y, class X1, class X2, class X3, class X4, class X5, class X6,
          int N = traits<X1>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative61(Y (&h)(const X1&, const X2&, const X3&, const X4&,
                             const X5&, const X6&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      const X5& x5, const X6& x6, double delta = 1e-5) {
  return numericalDerivative11<Y, X1, N>(
      [&](const X1& x1_) { return h(x1_, x2, x3, x4, x5, x6); }, x1, delta);
}

/**
 * Compute numerical derivative in argument 2 of 6-argument function
 * @param h quintic function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param x4 fourth argument value
 * @param x5 fifth argument value
 * @param x6 sixth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X2 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, class X5, class X6, int N = traits<X2>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4, X5, X6> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4, X5, X6>>::dimension,
    N>::type
numericalDerivative62(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, const X5& x5, const X6& x6,
                      double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4, X5, X6>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X2>::structure_category>,
                "Template argument X2 must be a manifold type.");
  return numericalDerivative11<ActualY, X2, N>(
      [&](const X2& x2_) { return std::invoke(h, x1, x2_, x3, x4, x5, x6); },
      x2, delta);
}

template <class Y, class X1, class X2, class X3, class X4, class X5, class X6,
          int N = traits<X2>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative62(Y (&h)(const X1&, const X2&, const X3&, const X4&,
                             const X5&, const X6&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      const X5& x5, const X6& x6, double delta = 1e-5) {
  return numericalDerivative11<Y, X2, N>(
      [&](const X2& x2_) { return h(x1, x2_, x3, x4, x5, x6); }, x2, delta);
}

/**
 * Compute numerical derivative in argument 3 of 6-argument function
 * @param h quintic function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param x4 fourth argument value
 * @param x5 fifth argument value
 * @param x6 sixth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X3 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, class X5, class X6, int N = traits<X3>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4, X5, X6> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4, X5, X6>>::dimension,
    N>::type
numericalDerivative63(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, const X5& x5, const X6& x6,
                      double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4, X5, X6>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X3>::structure_category>,
                "Template argument X3 must be a manifold type.");
  return numericalDerivative11<ActualY, X3, N>(
      [&](const X3& x3_) { return std::invoke(h, x1, x2, x3_, x4, x5, x6); },
      x3, delta);
}

template <class Y, class X1, class X2, class X3, class X4, class X5, class X6,
          int N = traits<X3>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative63(Y (&h)(const X1&, const X2&, const X3&, const X4&,
                             const X5&, const X6&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      const X5& x5, const X6& x6, double delta = 1e-5) {
  return numericalDerivative11<Y, X3, N>(
      [&](const X3& x3_) { return h(x1, x2, x3_, x4, x5, x6); }, x3, delta);
}

/**
 * Compute numerical derivative in argument 4 of 6-argument function
 * @param h quintic function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param x4 fourth argument value
 * @param x5 fifth argument value
 * @param x6 sixth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X4 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, class X5, class X6, int N = traits<X4>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4, X5, X6> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4, X5, X6>>::dimension,
    N>::type
numericalDerivative64(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, const X5& x5, const X6& x6,
                      double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4, X5, X6>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X4>::structure_category>,
                "Template argument X4 must be a manifold type.");
  return numericalDerivative11<ActualY, X4, N>(
      [&](const X4& x4_) { return std::invoke(h, x1, x2, x3, x4_, x5, x6); },
      x4, delta);
}

template <class Y, class X1, class X2, class X3, class X4, class X5, class X6,
          int N = traits<X4>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative64(Y (&h)(const X1&, const X2&, const X3&, const X4&,
                             const X5&, const X6&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      const X5& x5, const X6& x6, double delta = 1e-5) {
  return numericalDerivative11<Y, X4, N>(
      [&](const X4& x4_) { return h(x1, x2, x3, x4_, x5, x6); }, x4, delta);
}

/**
 * Compute numerical derivative in argument 5 of 6-argument function
 * @param h quintic function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param x4 fourth argument value
 * @param x5 fifth argument value
 * @param x6 sixth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X5 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, class X5, class X6, int N = traits<X5>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4, X5, X6> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4, X5, X6>>::dimension,
    N>::type
numericalDerivative65(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, const X5& x5, const X6& x6,
                      double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4, X5, X6>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X5>::structure_category>,
                "Template argument X5 must be a manifold type.");
  return numericalDerivative11<ActualY, X5, N>(
      [&](const X5& x5_) { return std::invoke(h, x1, x2, x3, x4, x5_, x6); },
      x5, delta);
}

template <class Y, class X1, class X2, class X3, class X4, class X5, class X6,
          int N = traits<X5>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative65(Y (&h)(const X1&, const X2&, const X3&, const X4&,
                             const X5&, const X6&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      const X5& x5, const X6& x6, double delta = 1e-5) {
  return numericalDerivative11<Y, X5, N>(
      [&](const X5& x5_) { return h(x1, x2, x3, x4, x5_, x6); }, x5, delta);
}

/**
 * Compute numerical derivative in argument 6 of 6-argument function
 * @param h quintic function yielding m-vector
 * @param x1 n-dimensional first argument value
 * @param x2 second argument value
 * @param x3 third argument value
 * @param x4 fourth argument value
 * @param x5 fifth argument value
 * @param x6 sixth argument value
 * @param delta increment for numerical derivative
 * @return m*n Jacobian computed via central differencing
 * @tparam int N is the dimension of the X6 input value if variable dimension
 * type but known at test time
 */
template <class Y = internal::DeducedOutput, class X1, class X2, class X3,
          class X4, class X5, class X6, int N = traits<X6>::dimension, class F,
          internal::EnableIfInvocable<F, X1, X2, X3, X4, X5, X6> = 0>
typename internal::MatrixMN<
    traits<internal::OutputType<Y, F, X1, X2, X3, X4, X5, X6>>::dimension,
    N>::type
numericalDerivative66(F&& h, const X1& x1, const X2& x2, const X3& x3,
                      const X4& x4, const X5& x5, const X6& x6,
                      double delta = 1e-5) {
  using ActualY = internal::OutputType<Y, F, X1, X2, X3, X4, X5, X6>;
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<ActualY>::structure_category>,
                "Template argument Y must be a manifold type.");
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X6>::structure_category>,
                "Template argument X6 must be a manifold type.");
  return numericalDerivative11<ActualY, X6, N>(
      [&](const X6& x6_) { return std::invoke(h, x1, x2, x3, x4, x5, x6_); },
      x6, delta);
}

template <class Y, class X1, class X2, class X3, class X4, class X5, class X6,
          int N = traits<X6>::dimension>
typename internal::MatrixMN<traits<Y>::dimension, N>::type
numericalDerivative66(Y (&h)(const X1&, const X2&, const X3&, const X4&,
                             const X5&, const X6&),
                      const X1& x1, const X2& x2, const X3& x3, const X4& x4,
                      const X5& x5, const X6& x6, double delta = 1e-5) {
  return numericalDerivative11<Y, X6, N>(
      [&](const X6& x6_) { return h(x1, x2, x3, x4, x5, x6_); }, x6, delta);
}
/// @}

/** @name Hessian Helpers */
/// @{
/**
 * Compute numerical Hessian matrix.  Requires a single-argument Lie->scalar
 * function.  This is implemented simply as the derivative of the gradient.
 * @param f A function taking a Lie object as input and returning a scalar
 * @param x The center point for computing the Hessian
 * @param delta The numerical derivative step size
 * @return n*n Hessian matrix computed via central differencing
 */
template <class X, int N = traits<X>::dimension>
inline typename internal::MatrixMN<N, N>::type numericalHessian(
    std::function<double(const X&)> f, const X& x, double delta = 1e-5) {
  static_assert(std::is_base_of_v<gtsam::manifold_tag,
                                  typename traits<X>::structure_category>,
                "Template argument X must be a manifold type.");
  typedef Eigen::Matrix<double, N, 1> VectorD;
  return numericalDerivative11<VectorD, X, N>(
      [&](const X& x) { return numericalGradient<X, N>(f, x, delta); }, x,
      delta);
}

template <class X, int N = traits<X>::dimension>
inline typename internal::MatrixMN<N, N>::type numericalHessian(
    double (*f)(const X&), const X& x, double delta = 1e-5) {
  return numericalHessian<X, N>(std::function<double(const X&)>(f), x, delta);
}

/** Helper class that computes the derivative of f w.r.t. x1, centered about
 * x1_, as a function of x2
 */
template <class X1, class X2, int N1 = traits<X1>::dimension>
class G_x1 {
  const std::function<double(const X1&, const X2&)>& f_;
  X1 x1_;
  double delta_;

 public:
  typedef Eigen::Matrix<double, N1, 1> Vector;

  G_x1(const std::function<double(const X1&, const X2&)>& f, const X1& x1,
       double delta)
      : f_(f), x1_(x1), delta_(delta) {}
  Vector operator()(const X2& x2) {
    return numericalGradient<X1, N1>(
        std::bind(f_, std::placeholders::_1, std::cref(x2)), x1_, delta_);
  }
};

template <class X1, class X2, int N1 = traits<X1>::dimension,
          int N2 = traits<X2>::dimension>
inline typename internal::MatrixMN<N1, N2>::type numericalHessian212(
    std::function<double(const X1&, const X2&)> f, const X1& x1, const X2& x2,
    double delta = 1e-5) {
  typedef Eigen::Matrix<double, N1, 1> Vector;
  G_x1<X1, X2, N1> g_x1(f, x1, delta);
  return numericalDerivative11<Vector, X2, N2>(
      std::function<Vector(const X2&)>(
          std::bind<Vector>(std::ref(g_x1), std::placeholders::_1)),
      x2, delta);
}

template <class X1, class X2, int N1 = traits<X1>::dimension,
          int N2 = traits<X2>::dimension>
inline typename internal::MatrixMN<N1, N2>::type numericalHessian212(
    double (*f)(const X1&, const X2&), const X1& x1, const X2& x2,
    double delta = 1e-5) {
  return numericalHessian212<X1, X2, N1, N2>(
      std::function<double(const X1&, const X2&)>(f), x1, x2, delta);
}

template <class X1, class X2, int N1 = traits<X1>::dimension>
inline typename internal::MatrixMN<N1, N1>::type numericalHessian211(
    std::function<double(const X1&, const X2&)> f, const X1& x1, const X2& x2,
    double delta = 1e-5) {
  typedef Eigen::Matrix<double, N1, 1> Vector;
  std::function<double(const X1&)> f2(
      std::bind(f, std::placeholders::_1, std::cref(x2)));
  return numericalDerivative11<Vector, X1, N1>(
      [&](const X1& x) { return numericalGradient<X1, N1>(f2, x, delta); }, x1,
      delta);
}

template <class X1, class X2, int N1 = traits<X1>::dimension>
inline typename internal::MatrixMN<N1, N1>::type numericalHessian211(
    double (*f)(const X1&, const X2&), const X1& x1, const X2& x2,
    double delta = 1e-5) {
  return numericalHessian211<X1, X2, N1>(
      std::function<double(const X1&, const X2&)>(f), x1, x2, delta);
}

template <class X1, class X2, int N2 = traits<X2>::dimension>
inline typename internal::MatrixMN<N2, N2>::type numericalHessian222(
    std::function<double(const X1&, const X2&)> f, const X1& x1, const X2& x2,
    double delta = 1e-5) {
  typedef Eigen::Matrix<double, N2, 1> Vector;
  std::function<double(const X2&)> f2(
      std::bind(f, std::cref(x1), std::placeholders::_1));
  return numericalDerivative11<Vector, X2, N2>(
      [&](const X2& x) { return numericalGradient<X2, N2>(f2, x, delta); }, x2,
      delta);
}

template <class X1, class X2, int N2 = traits<X2>::dimension>
inline typename internal::MatrixMN<N2, N2>::type numericalHessian222(
    double (*f)(const X1&, const X2&), const X1& x1, const X2& x2,
    double delta = 1e-5) {
  return numericalHessian222<X1, X2, N2>(
      std::function<double(const X1&, const X2&)>(f), x1, x2, delta);
}

/**
 * Numerical Hessian for tenary functions
 */
/* **************************************************************** */
template <class X1, class X2, class X3, int N1 = traits<X1>::dimension>
inline typename internal::MatrixMN<N1, N1>::type numericalHessian311(
    std::function<double(const X1&, const X2&, const X3&)> f, const X1& x1,
    const X2& x2, const X3& x3, double delta = 1e-5) {
  typedef Eigen::Matrix<double, N1, 1> Vector;
  std::function<double(const X1&)> f2(
      std::bind(f, std::placeholders::_1, std::cref(x2), std::cref(x3)));

  return numericalDerivative11<Vector, X1, N1>(
      [&](const X1& x) { return numericalGradient<X1, N1>(f2, x, delta); }, x1,
      delta);
}

template <class X1, class X2, class X3, int N1 = traits<X1>::dimension>
inline typename internal::MatrixMN<N1, N1>::type numericalHessian311(
    double (*f)(const X1&, const X2&, const X3&), const X1& x1, const X2& x2,
    const X3& x3, double delta = 1e-5) {
  return numericalHessian311<X1, X2, X3, N1>(
      std::function<double(const X1&, const X2&, const X3&)>(f), x1, x2, x3,
      delta);
}

/* **************************************************************** */
template <class X1, class X2, class X3, int N2 = traits<X2>::dimension>
inline typename internal::MatrixMN<N2, N2>::type numericalHessian322(
    std::function<double(const X1&, const X2&, const X3&)> f, const X1& x1,
    const X2& x2, const X3& x3, double delta = 1e-5) {
  typedef Eigen::Matrix<double, N2, 1> Vector;
  std::function<double(const X2&)> f2(
      std::bind(f, std::cref(x1), std::placeholders::_1, std::cref(x3)));

  return numericalDerivative11<Vector, X2, N2>(
      [&](const X2& x) { return numericalGradient<X2, N2>(f2, x, delta); }, x2,
      delta);
}

template <class X1, class X2, class X3, int N2 = traits<X2>::dimension>
inline typename internal::MatrixMN<N2, N2>::type numericalHessian322(
    double (*f)(const X1&, const X2&, const X3&), const X1& x1, const X2& x2,
    const X3& x3, double delta = 1e-5) {
  return numericalHessian322<X1, X2, X3, N2>(
      std::function<double(const X1&, const X2&, const X3&)>(f), x1, x2, x3,
      delta);
}

/* **************************************************************** */
template <class X1, class X2, class X3, int N3 = traits<X3>::dimension>
inline typename internal::MatrixMN<N3, N3>::type numericalHessian333(
    std::function<double(const X1&, const X2&, const X3&)> f, const X1& x1,
    const X2& x2, const X3& x3, double delta = 1e-5) {
  typedef Eigen::Matrix<double, N3, 1> Vector;
  std::function<double(const X3&)> f2(
      std::bind(f, std::cref(x1), std::cref(x2), std::placeholders::_1));

  return numericalDerivative11<Vector, X3, N3>(
      [&](const X3& x) { return numericalGradient<X3, N3>(f2, x, delta); }, x3,
      delta);
}

template <class X1, class X2, class X3, int N3 = traits<X3>::dimension>
inline typename internal::MatrixMN<N3, N3>::type numericalHessian333(
    double (*f)(const X1&, const X2&, const X3&), const X1& x1, const X2& x2,
    const X3& x3, double delta = 1e-5) {
  return numericalHessian333<X1, X2, X3, N3>(
      std::function<double(const X1&, const X2&, const X3&)>(f), x1, x2, x3,
      delta);
}

/* **************************************************************** */
template <class X1, class X2, class X3, int N1 = traits<X1>::dimension,
          int N2 = traits<X2>::dimension>
inline typename internal::MatrixMN<N1, N2>::type numericalHessian312(
    std::function<double(const X1&, const X2&, const X3&)> f, const X1& x1,
    const X2& x2, const X3& x3, double delta = 1e-5) {
  return numericalHessian212<X1, X2, N1, N2>(
      std::function<double(const X1&, const X2&)>(std::bind(
          f, std::placeholders::_1, std::placeholders::_2, std::cref(x3))),
      x1, x2, delta);
}

template <class X1, class X2, class X3, int N1 = traits<X1>::dimension,
          int N3 = traits<X3>::dimension>
inline typename internal::MatrixMN<N1, N3>::type numericalHessian313(
    std::function<double(const X1&, const X2&, const X3&)> f, const X1& x1,
    const X2& x2, const X3& x3, double delta = 1e-5) {
  return numericalHessian212<X1, X3, N1, N3>(
      std::function<double(const X1&, const X3&)>(std::bind(
          f, std::placeholders::_1, std::cref(x2), std::placeholders::_2)),
      x1, x3, delta);
}

template <class X1, class X2, class X3, int N2 = traits<X2>::dimension,
          int N3 = traits<X3>::dimension>
inline typename internal::MatrixMN<N2, N3>::type numericalHessian323(
    std::function<double(const X1&, const X2&, const X3&)> f, const X1& x1,
    const X2& x2, const X3& x3, double delta = 1e-5) {
  return numericalHessian212<X2, X3, N2, N3>(
      std::function<double(const X2&, const X3&)>(std::bind(
          f, std::cref(x1), std::placeholders::_1, std::placeholders::_2)),
      x2, x3, delta);
}

/* **************************************************************** */
template <class X1, class X2, class X3, int N1 = traits<X1>::dimension,
          int N2 = traits<X2>::dimension>
inline typename internal::MatrixMN<N1, N2>::type numericalHessian312(
    double (*f)(const X1&, const X2&, const X3&), const X1& x1, const X2& x2,
    const X3& x3, double delta = 1e-5) {
  return numericalHessian312<X1, X2, X3, N1, N2>(
      std::function<double(const X1&, const X2&, const X3&)>(f), x1, x2, x3,
      delta);
}

template <class X1, class X2, class X3, int N1 = traits<X1>::dimension,
          int N3 = traits<X3>::dimension>
inline typename internal::MatrixMN<N1, N3>::type numericalHessian313(
    double (*f)(const X1&, const X2&, const X3&), const X1& x1, const X2& x2,
    const X3& x3, double delta = 1e-5) {
  return numericalHessian313<X1, X2, X3, N1, N3>(
      std::function<double(const X1&, const X2&, const X3&)>(f), x1, x2, x3,
      delta);
}

template <class X1, class X2, class X3, int N2 = traits<X2>::dimension,
          int N3 = traits<X3>::dimension>
inline typename internal::MatrixMN<N2, N3>::type numericalHessian323(
    double (*f)(const X1&, const X2&, const X3&), const X1& x1, const X2& x2,
    const X3& x3, double delta = 1e-5) {
  return numericalHessian323<X1, X2, X3, N2, N3>(
      std::function<double(const X1&, const X2&, const X3&)>(f), x1, x2, x3,
      delta);
}

/// @}
/// @}
}  // namespace gtsam
