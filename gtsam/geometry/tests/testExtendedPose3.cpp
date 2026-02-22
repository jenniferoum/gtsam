/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   testExtendedPose3.cpp
 * @brief  Unit tests for ExtendedPose3<K>
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Lie.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/lieProxies.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/ExtendedPose3.h>

using namespace gtsam;

using ExtendedPose33 = ExtendedPose3<3>;
using ExtendedPose3d = ExtendedPose3<Eigen::Dynamic>;

GTSAM_CONCEPT_TESTABLE_INST(ExtendedPose33)
GTSAM_CONCEPT_MATRIX_LIE_GROUP_INST(ExtendedPose33)
GTSAM_CONCEPT_TESTABLE_INST(ExtendedPose3d)
GTSAM_CONCEPT_MATRIX_LIE_GROUP_INST(ExtendedPose3d)

namespace {

const Rot3 kR1 = Rot3::RzRyRx(0.1, -0.2, 0.3);
const Rot3 kR2 = Rot3::RzRyRx(-0.3, 0.4, -0.2);
const Matrix3 kX1 =
    (Matrix3() << 1.0, 4.0, -1.0, 2.0, 5.0, 0.5, 3.0, 6.0, 2.0).finished();
const Matrix3 kX2 =
    (Matrix3() << -2.0, 1.0, 0.1, 0.3, -1.5, 2.2, 0.7, -0.9, 1.6).finished();
const Vector12 kXi = (Vector12() << 0.11, -0.07, 0.05, 0.3, -0.4, 0.1, -0.2,
                      0.6, -0.5, 0.7, -0.1, 0.2)
                         .finished();

}  // namespace

//******************************************************************************
TEST(ExtendedPose3, Concept) {
  GTSAM_CONCEPT_ASSERT(IsGroup<ExtendedPose33>);
  GTSAM_CONCEPT_ASSERT(IsManifold<ExtendedPose33>);
  GTSAM_CONCEPT_ASSERT(IsMatrixLieGroup<ExtendedPose33>);

  GTSAM_CONCEPT_ASSERT(IsGroup<ExtendedPose3d>);
  GTSAM_CONCEPT_ASSERT(IsManifold<ExtendedPose3d>);
  GTSAM_CONCEPT_ASSERT(IsMatrixLieGroup<ExtendedPose3d>);
}

//******************************************************************************
TEST(ExtendedPose3, Dimensions) {
  const ExtendedPose33 fixed;
  EXPECT_LONGS_EQUAL(12, traits<ExtendedPose33>::GetDimension(fixed));
  EXPECT_LONGS_EQUAL(3, fixed.k());

  const ExtendedPose3d placeholder;
  EXPECT_LONGS_EQUAL(0, placeholder.k());
  EXPECT_LONGS_EQUAL(3, placeholder.dim());

  const ExtendedPose3d dynamic(3);
  EXPECT_LONGS_EQUAL(3, dynamic.k());
  EXPECT_LONGS_EQUAL(12, dynamic.dim());
}

//******************************************************************************
TEST(ExtendedPose3, ConstructorsAndAccess) {
  const ExtendedPose33 fixed(kR1, kX1);
  const ExtendedPose3d dynamic(kR1, ExtendedPose3d::Matrix3K(kX1));

  EXPECT(assert_equal(kR1, fixed.rotation()));
  EXPECT(assert_equal(kR1, dynamic.rotation()));

  for (size_t i = 0; i < 3; ++i) {
    EXPECT(assert_equal(Point3(kX1.col(static_cast<Eigen::Index>(i))),
                        fixed.x(i)));
    EXPECT(assert_equal(Point3(kX1.col(static_cast<Eigen::Index>(i))),
                        dynamic.x(i)));
  }

  EXPECT(assert_equal(fixed.matrix(), dynamic.matrix()));
}

//******************************************************************************
TEST(ExtendedPose3, GroupOperationsMatch) {
  const ExtendedPose33 f1(kR1, kX1), f2(kR2, kX2);
  const ExtendedPose3d d1(kR1, ExtendedPose3d::Matrix3K(kX1));
  const ExtendedPose3d d2(kR2, ExtendedPose3d::Matrix3K(kX2));

  const auto f12 = f1 * f2;
  const auto d12 = d1 * d2;
  EXPECT(assert_equal(f12.matrix(), d12.matrix()));

  const auto f_id = f1 * f1.inverse();
  const auto d_id = d1 * d1.inverse();
  EXPECT(
      assert_equal(ExtendedPose33::Identity().matrix(), f_id.matrix(), 1e-9));
  EXPECT(
      assert_equal(ExtendedPose3d::Identity(3).matrix(), d_id.matrix(), 1e-9));

  const auto f_between = f1.between(f2);
  const auto d_between = d1.between(d2);
  EXPECT(assert_equal(f_between.matrix(), d_between.matrix(), 1e-9));
}

//******************************************************************************
TEST(ExtendedPose3, HatVee) {
  const Vector xi = kXi;
  const auto Xf = ExtendedPose33::Hat(kXi);
  const auto Xd = ExtendedPose3d::Hat(xi);

  EXPECT(assert_equal(xi, Vector(ExtendedPose33::Vee(Xf))));
  EXPECT(assert_equal(xi, ExtendedPose3d::Vee(Xd)));
  EXPECT(assert_equal(Xf, Xd));
}

//******************************************************************************
TEST(ExtendedPose3, ExpmapLogmapRoundTrip) {
  const Vector xi = kXi;
  const ExtendedPose33 f = ExtendedPose33::Expmap(kXi);
  const ExtendedPose3d d = ExtendedPose3d::Expmap(xi);

  EXPECT(assert_equal(f.matrix(), d.matrix(), 1e-9));
  EXPECT(assert_equal(xi, Vector(ExtendedPose33::Logmap(f)), 1e-9));
  EXPECT(assert_equal(xi, ExtendedPose3d::Logmap(d), 1e-9));
}

//******************************************************************************
TEST(ExtendedPose3, AdjointConsistency) {
  const ExtendedPose33 f(kR1, kX1);
  const ExtendedPose3d d(kR1, ExtendedPose3d::Matrix3K(kX1));
  const Vector xi = kXi;

  const auto f_conj = f * ExtendedPose33::Expmap(kXi) * f.inverse();
  const auto d_conj = d * ExtendedPose3d::Expmap(xi) * d.inverse();

  const auto f_adj = ExtendedPose33::Expmap(f.Adjoint(kXi));
  const auto d_adj = ExtendedPose3d::Expmap(d.Adjoint(xi));
  EXPECT(assert_equal(f_conj.matrix(), f_adj.matrix(), 1e-9));
  EXPECT(assert_equal(d_conj.matrix(), d_adj.matrix(), 1e-9));

  const ExtendedPose33::Jacobian f_generic =
      static_cast<const MatrixLieGroup<ExtendedPose33, 12, 6>*>(&f)
          ->AdjointMap();
  const Matrix d_generic =
      static_cast<const MatrixLieGroup<ExtendedPose3d, Eigen::Dynamic,
                                       Eigen::Dynamic>*>(&d)
          ->AdjointMap();
  EXPECT(assert_equal(Matrix(f_generic), Matrix(f.AdjointMap()), 1e-9));
  EXPECT(assert_equal(d_generic, d.AdjointMap(), 1e-9));
}

//******************************************************************************
TEST(ExtendedPose3, Derivatives) {
  const ExtendedPose33 f(kR1, kX1);
  const ExtendedPose3d d(kR1, ExtendedPose3d::Matrix3K(kX1));
  const Vector xi = kXi;

  Matrix Hf;
  ExtendedPose33::Expmap(kXi, Hf);
  const auto f_exp = [](const Vector12& v) {
    return ExtendedPose33::Expmap(v);
  };
  const Matrix Hf_num =
      numericalDerivative11<ExtendedPose33, Vector12, 12>(f_exp, kXi);
  EXPECT(assert_equal(Hf_num, Hf, 1e-6));

  Matrix Hd;
  ExtendedPose3d::Expmap(xi, Hd);
  const auto d_exp = [](const Vector& v) { return ExtendedPose3d::Expmap(v); };
  const Matrix Hd_num =
      numericalDerivative11<ExtendedPose3d, Vector, 12>(d_exp, xi);
  EXPECT(assert_equal(Hd_num, Hd, 1e-6));

  Matrix Lf;
  ExtendedPose33::Logmap(f, Lf);
  const auto f_log = [](const ExtendedPose33& g) {
    return Vector12(ExtendedPose33::Logmap(g));
  };
  const Matrix Lf_num =
      numericalDerivative11<Vector12, ExtendedPose33, 12>(f_log, f);
  EXPECT(assert_equal(Lf_num, Lf, 1e-6));

  Matrix Ld;
  ExtendedPose3d::Logmap(d, Ld);
  const auto d_log = [](const ExtendedPose3d& g) {
    return ExtendedPose3d::Logmap(g);
  };
  const Matrix Ld_num =
      numericalDerivative11<Vector, ExtendedPose3d, 12>(d_log, d);
  EXPECT(assert_equal(Ld_num, Ld, 1e-6));
}

//******************************************************************************
TEST(ExtendedPose3, VecJacobian) {
  const ExtendedPose33 f(kR1, kX1);
  const ExtendedPose3d d(kR1, ExtendedPose3d::Matrix3K(kX1));

  Matrix Hf;
  const Vector vf = f.vec(Hf);
  const auto fv = [](const ExtendedPose33& g) { return Vector(g.vec()); };
  const Matrix Hf_num =
      numericalDerivative11<Vector, ExtendedPose33, 12>(fv, f);
  EXPECT(assert_equal(Hf_num, Hf, 1e-6));
  EXPECT_LONGS_EQUAL(36, vf.size());

  Matrix Hd;
  const Vector vd = d.vec(Hd);
  const auto dv = [](const ExtendedPose3d& g) { return Vector(g.vec()); };
  const Matrix Hd_num =
      numericalDerivative11<Vector, ExtendedPose3d, 12>(dv, d);
  EXPECT(assert_equal(Hd_num, Hd, 1e-6));
  EXPECT_LONGS_EQUAL(36, vd.size());
}

//******************************************************************************
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
//******************************************************************************
