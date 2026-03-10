/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   testEqVIOGroup.cpp
 * @brief  Unit tests for the Eq. (22) VIOGroup
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/navigation/EqVIOGroup.h>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

using namespace gtsam;

namespace {

template <int N, typename G>
void testLieGroupDerivativesN(TestResult& result_, const std::string& name_,
                              const G& t1, const G& t2) {
  Matrix H1, H2;
  using T = traits<G>;
  using OJ = OptionalJacobian<T::dimension, T::dimension>;

  OJ none;
  EXPECT(assert_equal<G>(t1.inverse(), T::Inverse(t1, H1)));
  EXPECT(assert_equal(numericalDerivative21<G, G, OJ, N>(T::Inverse, t1, none),
                      H1));

  EXPECT(assert_equal<G>(t2.inverse(), T::Inverse(t2, H1)));
  EXPECT(assert_equal(numericalDerivative21<G, G, OJ, N>(T::Inverse, t2, none),
                      H1));

  EXPECT(assert_equal<G>(t1 * t2, T::Compose(t1, t2, H1, H2)));
  EXPECT(assert_equal(
      numericalDerivative41<G, G, G, OJ, OJ, N>(T::Compose, t1, t2, none, none),
      H1));
  EXPECT(assert_equal(
      numericalDerivative42<G, G, G, OJ, OJ, N>(T::Compose, t1, t2, none, none),
      H2));

  EXPECT(assert_equal<G>(t1.inverse() * t2, T::Between(t1, t2, H1, H2)));
  EXPECT(assert_equal(
      numericalDerivative41<G, G, G, OJ, OJ, N>(T::Between, t1, t2, none, none),
      H1));
  EXPECT(assert_equal(
      numericalDerivative42<G, G, G, OJ, OJ, N>(T::Between, t1, t2, none, none),
      H2));
}

template <int N, typename G>
void testChartDerivativesN(TestResult& result_, const std::string& name_,
                           const G& t1, const G& t2) {
  Matrix H1, H2;
  using T = traits<G>;
  using V = typename T::TangentVector;
  using OJ = OptionalJacobian<T::dimension, T::dimension>;

  OJ none;
  const V w12 = T::Local(t1, t2);
  EXPECT(assert_equal<G>(t2, T::Retract(t1, w12, H1, H2)));
  EXPECT(assert_equal(
      numericalDerivative41<G, G, V, OJ, OJ, N>(T::Retract, t1, w12, none, none),
      H1));
  EXPECT(assert_equal(
      numericalDerivative42<G, G, V, OJ, OJ, N>(T::Retract, t1, w12, none, none),
      H2));

  EXPECT(assert_equal(w12, T::Local(t1, t2, H1, H2)));
  EXPECT(assert_equal(
      numericalDerivative41<V, G, G, OJ, OJ, N>(T::Local, t1, t2, none, none),
      H1));
  EXPECT(assert_equal(
      numericalDerivative42<V, G, G, OJ, OJ, N>(T::Local, t1, t2, none, none),
      H2));
}

using SE23 = VIOGroup::SE23;
using LandmarkGroup = VIOGroup::LandmarkGroup;

const Rot3 kR1 = Rot3::RzRyRx(0.1, -0.2, 0.3);
const Rot3 kR2 = Rot3::RzRyRx(-0.3, 0.2, -0.1);
const Rot3 kR3 = Rot3::RzRyRx(0.2, 0.1, -0.25);
const Rot3 kR4 = Rot3::RzRyRx(-0.15, -0.1, 0.05);

const SE23::Matrix3K kX1 =
    (SE23::Matrix3K() << 1.0, -0.4, 0.3, 0.9, -0.8, 0.2).finished();
const SE23::Matrix3K kX2 =
    (SE23::Matrix3K() << -0.5, 0.7, 0.1, -0.2, 0.4, 1.1).finished();

const SE23 kA1(kR1, kX1);
const SE23 kA2(kR2, kX2);

const Vector6 kBeta1 =
    (Vector6() << 0.1, -0.2, 0.3, -0.4, 0.2, 0.05).finished();
const Vector6 kBeta2 =
    (Vector6() << -0.05, 0.3, -0.1, 0.2, -0.15, 0.07).finished();

const Pose3 kB1(kR3, Point3(0.2, -0.5, 1.1));
const Pose3 kB2(kR4, Point3(-0.6, 0.7, 0.3));

const SOT3 kQ1(SO3::Expmap((Vector3() << 0.08, -0.04, 0.05).finished()),
               Vector1::Constant(std::log(1.2)));
const SOT3 kQ2(SO3::Expmap((Vector3() << -0.03, 0.06, -0.02).finished()),
               Vector1::Constant(std::log(0.95)));
const SOT3 kQ3(SO3::Expmap((Vector3() << 0.04, 0.07, -0.08).finished()),
               Vector1::Constant(std::log(1.1)));
const SOT3 kQ4(SO3::Expmap((Vector3() << -0.06, -0.02, 0.09).finished()),
               Vector1::Constant(std::log(1.05)));
const SOT3 kQ5(SO3::Expmap((Vector3() << 0.01, 0.02, 0.03).finished()),
               Vector1::Constant(std::log(0.98)));
const SOT3 kQ6(SO3::Expmap((Vector3() << -0.02, 0.03, -0.01).finished()),
               Vector1::Constant(std::log(1.03)));

LandmarkGroup MakeQ0() { return LandmarkGroup(0); }
LandmarkGroup MakeQ1A() { return LandmarkGroup({kQ1}); }
LandmarkGroup MakeQ1B() { return LandmarkGroup({kQ2}); }
LandmarkGroup MakeQ3A() { return LandmarkGroup({kQ1, kQ2, kQ3}); }
LandmarkGroup MakeQ3B() { return LandmarkGroup({kQ4, kQ5, kQ6}); }

const std::vector<int> kIds1{11};
const std::vector<int> kIds3{11, 22, 33};
const std::vector<int> kIds3Other{11, 22, 44};

VIOGroup MakeG0() { return VIOGroup(kA1, kBeta1, kB1, MakeQ0()); }

VIOGroup MakeG0b() { return VIOGroup(kA2, kBeta2, kB2, MakeQ0()); }

VIOGroup MakeG1() { return VIOGroup(kA1, kBeta1, kB1, MakeQ1A(), kIds1); }

VIOGroup MakeG1b() { return VIOGroup(kA2, kBeta2, kB2, MakeQ1B(), kIds1); }

VIOGroup MakeG1NoIds() { return VIOGroup(kA1, kBeta1, kB1, MakeQ1A()); }

VIOGroup MakeG3() { return VIOGroup(kA1, kBeta1, kB1, MakeQ3A(), kIds3); }

VIOGroup MakeG3b() { return VIOGroup(kA2, kBeta2, kB2, MakeQ3B(), kIds3); }

Vector Xi0() {
  return (Vector(21) << 0.05, -0.04, 0.03, 0.2, -0.1, 0.15, -0.05, 0.07,
          -0.09, 0.1, -0.08, 0.06, -0.04, 0.02, 0.03, 0.01, -0.02, 0.04,
          -0.03, 0.05, -0.01)
      .finished();
}

Vector Xi1() {
  Vector xi(25);
  xi << Xi0(), 0.03, -0.02, 0.01, 0.04;
  return xi;
}

Vector Xi3() {
  Vector xi(33);
  xi << Xi0(), 0.03, -0.02, 0.01, 0.04, -0.01, 0.05, -0.03, 0.02, 0.02,
      0.01, -0.04, 0.03;
  return xi;
}

}  // namespace

//******************************************************************************
TEST(VIOGroup, Concept) {
  GTSAM_CONCEPT_ASSERT(IsGroup<VIOGroup>);
  GTSAM_CONCEPT_ASSERT(IsManifold<VIOGroup>);
  GTSAM_CONCEPT_ASSERT(IsLieGroup<VIOGroup>);
  GTSAM_CONCEPT_ASSERT(IsTestable<VIOGroup>);
}

//******************************************************************************
TEST(VIOGroup, ConstructorsAndAccessors) {
  const VIOGroup empty;
  EXPECT_LONGS_EQUAL(0, empty.n());
  EXPECT_LONGS_EQUAL(21, empty.dim());
  EXPECT(empty.ids().empty());

  const VIOGroup identity3 = VIOGroup::Identity(3);
  EXPECT_LONGS_EQUAL(3, identity3.n());
  EXPECT_LONGS_EQUAL(33, identity3.dim());
  EXPECT(identity3.ids().empty());

  const VIOGroup identityIds = VIOGroup::Identity(kIds3);
  EXPECT(identityIds.ids() == kIds3);
  EXPECT_LONGS_EQUAL(3, identityIds.n());

  const VIOGroup g = MakeG3();
  EXPECT(assert_equal(kA1, g.A()));
  EXPECT(assert_equal(Vector(kBeta1), Vector(g.beta())));
  EXPECT(assert_equal(kB1, g.B()));
  EXPECT(assert_equal(MakeQ3A(), g.Q()));
  EXPECT(g.ids() == kIds3);
}

//******************************************************************************
TEST(VIOGroup, GroupOperations) {
  const VIOGroup g1 = MakeG3();
  const VIOGroup g2 = MakeG3b();
  const VIOGroup composed = g1.compose(g2);

  EXPECT(assert_equal(g1.A().compose(g2.A()), composed.A()));
  EXPECT(assert_equal(Vector(g1.beta() + g2.beta()), Vector(composed.beta())));
  EXPECT(assert_equal(g1.B().compose(g2.B()), composed.B()));
  EXPECT(assert_equal(g1.Q().compose(g2.Q()), composed.Q()));
  EXPECT(composed.ids() == kIds3);

  const VIOGroup between = g1.between(g2);
  EXPECT(assert_equal(g1.inverse() * g2, between));

  const VIOGroup inverse = g1.inverse();
  EXPECT(inverse.ids() == kIds3);
  EXPECT(assert_equal(VIOGroup::Identity(kIds3), g1.compose(inverse)));
}

//******************************************************************************
TEST(VIOGroup, ExpmapLogmapAndAdjoint) {
  const Vector xi0 = Xi0();
  const Vector xi1 = Xi1();
  const Vector xi3 = Xi3();

  const VIOGroup g0 = VIOGroup::Expmap(xi0);
  const VIOGroup g1 = VIOGroup::Expmap(xi1);
  const VIOGroup g3 = VIOGroup::Expmap(xi3);

  EXPECT(assert_equal(xi0, VIOGroup::Logmap(g0), 1e-9));
  EXPECT(assert_equal(xi1, VIOGroup::Logmap(g1), 1e-9));
  EXPECT(assert_equal(xi3, VIOGroup::Logmap(g3), 1e-9));
  EXPECT(g3.ids().empty());

  VIOGroup::VIOGroupCore core(
      VIOGroup::SensorCore(g3.A(), g3.beta()),
      VIOGroup::LandmarkCore(g3.B(), g3.Q()));
  EXPECT(assert_equal(core.AdjointMap(), g3.AdjointMap(), 1e-9));
}

//******************************************************************************
TEST(VIOGroup, DerivativesN0) {
  const VIOGroup id = VIOGroup::Identity();
  const VIOGroup g = MakeG0();
  const VIOGroup h = MakeG0b();

  testLieGroupDerivativesN<21>(result_, name_, id, g);
  testLieGroupDerivativesN<21>(result_, name_, g, h);

  testChartDerivativesN<21>(result_, name_, id, g);
  testChartDerivativesN<21>(result_, name_, g, h);
}

//******************************************************************************
TEST(VIOGroup, DerivativesN1) {
  const VIOGroup id = VIOGroup::Identity(kIds1);
  const VIOGroup g = MakeG1();
  const VIOGroup h = MakeG1b();

  testLieGroupDerivativesN<25>(result_, name_, id, g);
  testLieGroupDerivativesN<25>(result_, name_, g, h);

  testChartDerivativesN<25>(result_, name_, id, g);
  testChartDerivativesN<25>(result_, name_, g, h);
}

//******************************************************************************
TEST(VIOGroup, DerivativesN3) {
  const VIOGroup id = VIOGroup::Identity(kIds3);
  const VIOGroup g = MakeG3();
  const VIOGroup h = MakeG3b();

  testLieGroupDerivativesN<33>(result_, name_, id, g);
  testLieGroupDerivativesN<33>(result_, name_, g, h);

  testChartDerivativesN<33>(result_, name_, id, g);
  testChartDerivativesN<33>(result_, name_, g, h);
}

//******************************************************************************
TEST(VIOGroup, IdBehavior) {
  const VIOGroup withIds = MakeG1();
  const VIOGroup withoutIds = MakeG1NoIds();
  const VIOGroup sameCoreOtherIds(kA1, kBeta1, kB1, MakeQ1A(), {99});

  EXPECT(withIds.inverse().ids() == kIds1);
  EXPECT(withIds.compose(withoutIds).ids() == kIds1);
  EXPECT(withoutIds.compose(withIds).ids() == kIds1);
  EXPECT(withIds.equals(sameCoreOtherIds));

}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
