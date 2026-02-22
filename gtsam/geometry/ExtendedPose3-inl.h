/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    ExtendedPose3-inl.h
 * @brief   Template implementations for ExtendedPose3<K>
 * @author  Frank Dellaert, et al.
 */

#pragma once

namespace gtsam {

template <int K>
size_t ExtendedPose3<K>::RuntimeK(const TangentVector& xi) {
  if constexpr (K == Eigen::Dynamic) {
    if (xi.size() < 3 || (xi.size() - 3) % 3 != 0) {
      throw std::invalid_argument(
          "ExtendedPose3: tangent vector size must be 3 + 3*k.");
    }
    return static_cast<size_t>((xi.size() - 3) / 3);
  } else {
    return static_cast<size_t>(K);
  }
}

template <int K>
void ExtendedPose3<K>::ZeroJacobian(ChartJacobian H, size_t d) {
  if (!H) return;
  if constexpr (dimension == Eigen::Dynamic) {
    H->setZero(d, d);
  } else {
    (void)d;
    H->setZero();
  }
}

template <int K>
template <int K_, typename>
ExtendedPose3<K>::ExtendedPose3() : R_(Rot3::Identity()), x_(Matrix3K::Zero()) {
}

template <int K>
template <int K_, typename>
ExtendedPose3<K>::ExtendedPose3(size_t k) : R_(Rot3::Identity()), x_(3, k) {
  x_.setZero();
}

template <int K>
ExtendedPose3<K>::ExtendedPose3(const Rot3& R, const Matrix3K& x) : R_(R), x_(x) {}

template <int K>
ExtendedPose3<K>::ExtendedPose3(const LieAlgebra& T) {
  if constexpr (K == Eigen::Dynamic) {
    const auto n = T.rows();
    if (T.cols() != n || n < 3) {
      throw std::invalid_argument("ExtendedPose3: invalid matrix shape.");
    }
    const auto k = n - 3;
    x_.resize(3, k);
    x_.setZero();
  }

  const auto n = T.rows();
  if constexpr (K != Eigen::Dynamic) {
    if (n != matrix_dim || T.cols() != matrix_dim) {
      throw std::invalid_argument("ExtendedPose3: invalid matrix shape.");
    }
  } else {
    if (T.cols() != n || n < 3) {
      throw std::invalid_argument("ExtendedPose3: invalid matrix shape.");
    }
  }

  R_ = Rot3(T.template block<3, 3>(0, 0));
  x_ = T.block(0, 3, 3, n - 3);
}

template <int K>
size_t ExtendedPose3<K>::Dimension(size_t k) {
  return 3 + 3 * k;
}

template <int K>
size_t ExtendedPose3<K>::k() const {
  return static_cast<size_t>(x_.cols());
}

template <int K>
template <int K_, typename>
size_t ExtendedPose3<K>::dim() const {
  return Dimension(k());
}

template <int K>
const Rot3& ExtendedPose3<K>::rotation(ComponentJacobian H) const {
  if (H) {
    if constexpr (dimension == Eigen::Dynamic) {
      H->setZero(3, dim());
    } else {
      H->setZero();
    }
    H->block(0, 0, 3, 3) = I_3x3;
  }
  return R_;
}

template <int K>
Point3 ExtendedPose3<K>::x(size_t i, ComponentJacobian H) const {
  if (i >= k()) throw std::out_of_range("ExtendedPose3: x(i) out of range.");
  if (H) {
    if constexpr (dimension == Eigen::Dynamic) {
      H->setZero(3, dim());
    } else {
      H->setZero();
    }
    H->block(0, 3 + 3 * i, 3, 3) = R_.matrix();
  }
  return x_.col(static_cast<Eigen::Index>(i));
}

template <int K>
const typename ExtendedPose3<K>::Matrix3K& ExtendedPose3<K>::xMatrix() const {
  return x_;
}

template <int K>
typename ExtendedPose3<K>::Matrix3K& ExtendedPose3<K>::xMatrix() {
  return x_;
}

template <int K>
void ExtendedPose3<K>::print(const std::string& s) const {
  std::cout << (s.empty() ? s : s + " ") << *this << std::endl;
}

template <int K>
bool ExtendedPose3<K>::equals(const ExtendedPose3& other, double tol) const {
  return R_.equals(other.R_, tol) && equal_with_abs_tol(x_, other.x_, tol);
}

template <int K>
template <int K_, typename>
ExtendedPose3<K> ExtendedPose3<K>::Identity() {
  return ExtendedPose3();
}

template <int K>
template <int K_, typename>
ExtendedPose3<K> ExtendedPose3<K>::Identity(size_t k) {
  return ExtendedPose3(k);
}

template <int K>
ExtendedPose3<K> ExtendedPose3<K>::inverse() const {
  const Rot3 Rt = R_.inverse();
  const Matrix3K x = -(Rt.matrix() * x_);
  return ExtendedPose3(Rt, x);
}

template <int K>
ExtendedPose3<K> ExtendedPose3<K>::operator*(const ExtendedPose3& other) const {
  if (k() != other.k()) {
    throw std::invalid_argument("ExtendedPose3: compose requires matching k.");
  }
  Matrix3K x = x_ + R_.matrix() * other.x_;
  return ExtendedPose3(R_ * other.R_, x);
}

template <int K>
ExtendedPose3<K> ExtendedPose3<K>::Expmap(const TangentVector& xi,
                                          ChartJacobian Hxi) {
  const size_t k = RuntimeK(xi);

  const Vector3 w = xi.template head<3>();
  const so3::DexpFunctor local(w);
  const Rot3 R(local.expmap());
  const Matrix3 Rt = R.transpose();

  Matrix3K x;
  if constexpr (K == Eigen::Dynamic) x.resize(3, static_cast<Eigen::Index>(k));

  if (Hxi) {
    const size_t d = 3 + 3 * k;
    ZeroJacobian(Hxi, d);
    const Matrix3 Jr = local.Jacobian().right();
    Hxi->block(0, 0, 3, 3) = Jr;
    for (size_t i = 0; i < k; ++i) {
      Matrix3 H_xi_w;
      const Vector3 rho = xi.template segment<3>(3 + 3 * i);
      x.col(static_cast<Eigen::Index>(i)) =
          local.Jacobian().applyLeft(rho, &H_xi_w);
      Hxi->block(3 + 3 * i, 0, 3, 3) = Rt * H_xi_w;
      Hxi->block(3 + 3 * i, 3 + 3 * i, 3, 3) = Jr;
    }
  } else {
    for (size_t i = 0; i < k; ++i) {
      const Vector3 rho = xi.template segment<3>(3 + 3 * i);
      x.col(static_cast<Eigen::Index>(i)) = local.Jacobian().applyLeft(rho);
    }
  }

  return ExtendedPose3(R, x);
}

template <int K>
typename ExtendedPose3<K>::TangentVector ExtendedPose3<K>::Logmap(
    const ExtendedPose3& pose, ChartJacobian Hpose) {
  const Vector3 w = Rot3::Logmap(pose.R_);
  const so3::DexpFunctor local(w);

  TangentVector xi;
  if constexpr (K == Eigen::Dynamic) xi.resize(static_cast<Eigen::Index>(pose.dim()));
  xi.template head<3>() = w;
  for (size_t i = 0; i < pose.k(); ++i) {
    xi.template segment<3>(3 + 3 * i) =
        local.InvJacobian().applyLeft(pose.x_.col(static_cast<Eigen::Index>(i)));
  }

  if (Hpose) *Hpose = LogmapDerivative(xi);
  return xi;
}

template <int K>
typename ExtendedPose3<K>::Jacobian ExtendedPose3<K>::AdjointMap() const {
  const Matrix3 R = R_.matrix();
  Jacobian adj;
  if constexpr (dimension == Eigen::Dynamic) {
    adj.setZero(dim(), dim());
  } else {
    adj.setZero();
  }

  adj.block(0, 0, 3, 3) = R;
  for (size_t i = 0; i < k(); ++i) {
    adj.block(3 + 3 * i, 0, 3, 3) =
        skewSymmetric(x_.col(static_cast<Eigen::Index>(i))) * R;
    adj.block(3 + 3 * i, 3 + 3 * i, 3, 3) = R;
  }
  return adj;
}

template <int K>
typename ExtendedPose3<K>::TangentVector ExtendedPose3<K>::Adjoint(
    const TangentVector& xi_b, ChartJacobian H_this, ChartJacobian H_xib) const {
  const Jacobian Ad = AdjointMap();
  if (H_this) *H_this = -Ad * adjointMap(xi_b);
  if (H_xib) *H_xib = Ad;
  return Ad * xi_b;
}

template <int K>
typename ExtendedPose3<K>::Jacobian ExtendedPose3<K>::adjointMap(
    const TangentVector& xi) {
  const size_t k = RuntimeK(xi);
  const Matrix3 w_hat = skewSymmetric(xi(0), xi(1), xi(2));

  Jacobian adj;
  if constexpr (dimension == Eigen::Dynamic) {
    adj.setZero(3 + 3 * k, 3 + 3 * k);
  } else {
    adj.setZero();
  }

  adj.block(0, 0, 3, 3) = w_hat;
  for (size_t i = 0; i < k; ++i) {
    adj.block(3 + 3 * i, 0, 3, 3) = skewSymmetric(
        xi(3 + 3 * i + 0), xi(3 + 3 * i + 1), xi(3 + 3 * i + 2));
    adj.block(3 + 3 * i, 3 + 3 * i, 3, 3) = w_hat;
  }
  return adj;
}

template <int K>
typename ExtendedPose3<K>::TangentVector ExtendedPose3<K>::adjoint(
    const TangentVector& xi, const TangentVector& y, ChartJacobian Hxi,
    ChartJacobian H_y) {
  const Jacobian ad_xi = adjointMap(xi);
  if (Hxi) {
    if constexpr (dimension == Eigen::Dynamic) {
      Hxi->setZero(xi.size(), xi.size());
    } else {
      Hxi->setZero();
    }
    for (Eigen::Index i = 0; i < xi.size(); ++i) {
      TangentVector dxi;
      if constexpr (dimension == Eigen::Dynamic) {
        dxi = TangentVector::Zero(xi.size());
      } else {
        dxi = TangentVector::Zero();
      }
      dxi(i) = 1.0;
      Hxi->col(i) = adjointMap(dxi) * y;
    }
  }
  if (H_y) *H_y = ad_xi;
  return ad_xi * y;
}

template <int K>
typename ExtendedPose3<K>::Jacobian ExtendedPose3<K>::ExpmapDerivative(
    const TangentVector& xi) {
  Jacobian J;
  Expmap(xi, J);
  return J;
}

template <int K>
typename ExtendedPose3<K>::Jacobian ExtendedPose3<K>::LogmapDerivative(
    const TangentVector& xi) {
  const size_t k = RuntimeK(xi);
  const Vector3 w = xi.template head<3>();
  const so3::DexpFunctor local(w);
  const Matrix3 Rt = local.expmap().transpose();
  const Matrix3 Jw = Rot3::LogmapDerivative(w);

  Jacobian J;
  if constexpr (dimension == Eigen::Dynamic) {
    J.setZero(3 + 3 * k, 3 + 3 * k);
  } else {
    J.setZero();
  }

  J.block(0, 0, 3, 3) = Jw;
  for (size_t i = 0; i < k; ++i) {
    Matrix3 H_xi_w;
    local.Jacobian().applyLeft(xi.template segment<3>(3 + 3 * i), H_xi_w);
    const Matrix3 Q = Rt * H_xi_w;
    J.block(3 + 3 * i, 0, 3, 3) = -Jw * Q * Jw;
    J.block(3 + 3 * i, 3 + 3 * i, 3, 3) = Jw;
  }
  return J;
}

template <int K>
typename ExtendedPose3<K>::Jacobian ExtendedPose3<K>::LogmapDerivative(
    const ExtendedPose3& pose) {
  return LogmapDerivative(Logmap(pose));
}

template <int K>
ExtendedPose3<K> ExtendedPose3<K>::ChartAtOrigin::Retract(
    const TangentVector& xi, ChartJacobian Hxi) {
  return Expmap(xi, Hxi);
}

template <int K>
typename ExtendedPose3<K>::TangentVector ExtendedPose3<K>::ChartAtOrigin::Local(
    const ExtendedPose3& pose, ChartJacobian Hpose) {
  return Logmap(pose, Hpose);
}

template <int K>
typename ExtendedPose3<K>::LieAlgebra ExtendedPose3<K>::matrix() const {
  LieAlgebra M;
  if constexpr (matrix_dim == Eigen::Dynamic) {
    const Eigen::Index n = 3 + static_cast<Eigen::Index>(k());
    M = LieAlgebra::Identity(n, n);
  } else {
    M = LieAlgebra::Identity();
  }
  M.template block<3, 3>(0, 0) = R_.matrix();
  M.block(0, 3, 3, static_cast<Eigen::Index>(k())) = x_;
  return M;
}

template <int K>
typename ExtendedPose3<K>::LieAlgebra ExtendedPose3<K>::Hat(
    const TangentVector& xi) {
  const size_t k = RuntimeK(xi);
  LieAlgebra X;
  if constexpr (matrix_dim == Eigen::Dynamic) {
    X.setZero(3 + k, 3 + k);
  } else {
    X.setZero();
  }
  X.block(0, 0, 3, 3) = skewSymmetric(xi(0), xi(1), xi(2));
  for (size_t i = 0; i < k; ++i) {
    X.block(0, 3 + i, 3, 1) = xi.template segment<3>(3 + 3 * i);
  }
  return X;
}

template <int K>
typename ExtendedPose3<K>::TangentVector ExtendedPose3<K>::Vee(
    const LieAlgebra& X) {
  const Eigen::Index k = X.cols() - 3;
  TangentVector xi;
  if constexpr (dimension == Eigen::Dynamic) {
    xi.resize(3 + 3 * k);
  }
  xi(0) = X(2, 1);
  xi(1) = X(0, 2);
  xi(2) = X(1, 0);
  for (Eigen::Index i = 0; i < k; ++i) {
    xi.template segment<3>(3 + 3 * i) = X.template block<3, 1>(0, 3 + i);
  }
  return xi;
}

}  // namespace gtsam
