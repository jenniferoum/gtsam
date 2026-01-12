#include <CppUnitLite/TestHarness.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Similarity3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/sfm/TrajectoryAlignerSim3.h>
#include <gtsam/slam/expressions.h>

#include <random>
#include <vector>

using namespace gtsam;
using PoseMeasurements = TrajectoryAlignerSim3::PoseMeasurements;
using ChildrenPoses = TrajectoryAlignerSim3::ChildrenPoses;

namespace {

Pose3 makePose(double x, double y, double z, const Rot3& R = Rot3()) {
  return Pose3(R, Point3(x, y, z));
}

std::vector<Pose3> makeParentPoses() {
  return {
      makePose(0.0, 0.0, 0.0),
      makePose(1.0, 0.1, 0.0, Rot3::RzRyRx(0.15, -0.2, 0.1)),
      makePose(2.0, 0.4, 0.1, Rot3::RzRyRx(0.1, 0.05, -0.03))};
}

PoseMeasurements makeMeasurements(const std::vector<Pose3>& poses,
                                  double noiseSigma = 1e-3) {
  PoseMeasurements m;
  auto noise = noiseModel::Isotropic::Sigma(6, noiseSigma);
  for (size_t i = 0; i < poses.size(); ++i) {
    m.emplace_back(static_cast<Key>(i), poses[i], noise);
  }
  return m;
}

std::vector<Pose3> transformPoses(const Similarity3& sim,
                                  const std::vector<Pose3>& poses) {
  std::vector<Pose3> out;
  out.reserve(poses.size());
  for (const auto& p : poses) out.push_back(sim.transformFrom(p));
  return out;
}


Pose3 perturbPose(const Pose3& p) {
  static thread_local std::mt19937 rng(42);
  std::normal_distribution<double> noise(0.0, 0.01);
  const Rot3 dR = Rot3::RzRyRx(noise(rng), noise(rng), noise(rng));
  const Point3 dt(noise(rng), noise(rng), noise(rng));
  return p.compose(Pose3(dR, dt));
}

std::vector<Pose3> perturbPoses(const std::vector<Pose3>& poses) {
  std::vector<Pose3> out;
  out.reserve(poses.size());
  for (const auto& p : poses) out.push_back(perturbPose(p));
  return out;
}

Similarity3 perturbSim3(const Similarity3& sim) {
  Similarity3 delta(Rot3::RzRyRx(0.2, -0.25, 0.1), Point3(2, 3, -1),
                    2.3);
  return sim * delta;
}

const Similarity3 gtSim1(Rot3::RzRyRx(0.2, 0.1, -0.05), Point3(0.3, -0.1, 0.2),
                         1.5);
const Similarity3 gtSim2(Rot3::RzRyRx(-0.15, 0.05, 0.08),
                         Point3(-0.2, 0.15, 0.25), 1.3);
const Similarity3 gtSim3(Rot3::RzRyRx(0.05, -0.08, 0.12),
                          Point3(0.15, 0.05, -0.1), 1.1);

bool simClose(const Similarity3& expected, const Similarity3& actual,
              double tol) {
  return assert_equal<Similarity3>(expected, actual, tol);
}

}  // namespace

/* ************************************************************************* */
TEST(TrajectoryAlignerSim3, PerfectSingleChild) {
  const auto parent = makeParentPoses();
  const auto child = transformPoses(gtSim1, parent);

  PoseMeasurements aTi = makeMeasurements(parent);
  ChildrenPoses bTi_all{makeMeasurements(child)};
  std::vector<Similarity3> sims{gtSim1};

  TrajectoryAlignerSim3 aligner(aTi, bTi_all, sims);
  Values result = aligner.solve();

  const auto recoveredSim = result.at<Similarity3>(Symbol('S', 0));
  EXPECT(simClose(gtSim1, recoveredSim, 1e-6));

  for (size_t i = 0; i < parent.size(); ++i) {
    EXPECT(assert_equal<Pose3>(parent[i], result.at<Pose3>(i), 1e-6));
  }
}

/* ************************************************************************* */
TEST(TrajectoryAlignerSim3, PerfectSingleChildNoInitialSim) {
  const auto parent = makeParentPoses();
  const Similarity3 gtSim(Rot3::RzRyRx(0.25, -0.05, 0.12),
                          Point3(-0.3, 0.2, -0.15), 1.4);
  const auto child = transformPoses(gtSim, parent);

  PoseMeasurements aTi = makeMeasurements(parent);
  ChildrenPoses bTi_all{makeMeasurements(child)};
  std::vector<Similarity3> sims;  // empty initial Sim3s

  TrajectoryAlignerSim3 aligner(aTi, bTi_all, sims);
  Values result = aligner.solve();

  const auto recoveredSim = result.at<Similarity3>(Symbol('S', 0));
  EXPECT(simClose(gtSim, recoveredSim, 1e-6));
}

/* ************************************************************************* */
TEST(TrajectoryAlignerSim3, NoisySingleChild) {
  const auto parent = makeParentPoses();
  const auto child = transformPoses(gtSim1, parent);

  PoseMeasurements aTi = makeMeasurements(parent, /*noiseSigma=*/1e-2);
  PoseMeasurements bTi = makeMeasurements(perturbPoses(child), 1e-1);
  ChildrenPoses bTi_all{bTi};
  std::vector<Similarity3> sims{perturbSim3(gtSim1)};

  TrajectoryAlignerSim3 aligner(aTi, bTi_all, sims);
  Values result = aligner.solve();

  const auto recoveredSim = result.at<Similarity3>(Symbol('S', 0));
  EXPECT(simClose(gtSim1, recoveredSim, 1e-2));

  for (size_t i = 0; i < parent.size(); ++i) {
    EXPECT(assert_equal<Pose3>(parent[i], result.at<Pose3>(i), 1e-3));
  }
}

/* ************************************************************************* */
TEST(TrajectoryAlignerSim3, SingleChildWithExtraNonOverlap) {
  const auto parent = makeParentPoses();
  auto child = transformPoses(gtSim1, parent);

  PoseMeasurements aTi = makeMeasurements(parent, /*noiseSigma=*/1e-2);
  PoseMeasurements bTi = makeMeasurements(perturbPoses(child), 1e-1);
  // Add a non-overlapping camera in the child frame.
  bTi.emplace_back(static_cast<Key>(10),
                   gtSim1.transformFrom(makePose(4.0, -1.0, 0.5)),
                   noiseModel::Isotropic::Sigma(6, 1e-2));

  ChildrenPoses bTi_all{bTi};
  std::vector<Similarity3> sims{perturbSim3(gtSim1)};

  TrajectoryAlignerSim3 aligner(aTi, bTi_all, sims);
  Values result = aligner.solve();

  const auto recoveredSim = result.at<Similarity3>(Symbol('S', 0));
  EXPECT(simClose(gtSim1, recoveredSim, 2e-2));

  for (size_t i = 0; i < parent.size(); ++i) {
    EXPECT(assert_equal<Pose3>(parent[i], result.at<Pose3>(i), 1e-3));
  }
}

/* ************************************************************************* */
TEST(TrajectoryAlignerSim3, TwoChildrenNoisy) {
  const auto parent = makeParentPoses();

  PoseMeasurements aTi = makeMeasurements(parent, 1e-2);
  PoseMeasurements b1 =
      makeMeasurements(perturbPoses(transformPoses(gtSim1, parent)), 5e-2);
  PoseMeasurements b2 =
      makeMeasurements(perturbPoses(transformPoses(gtSim2, parent)), 5e-2);

  ChildrenPoses bTi_all{b1, b2};
  std::vector<Similarity3> sims{perturbSim3(gtSim1), perturbSim3(gtSim2)};

  TrajectoryAlignerSim3 aligner(aTi, bTi_all, sims);
  Values result = aligner.solve();

  EXPECT(simClose(gtSim1, result.at<Similarity3>(Symbol('S', 0)), 5e-2));
  EXPECT(simClose(gtSim2, result.at<Similarity3>(Symbol('S', 1)), 5e-2));

  for (size_t i = 0; i < parent.size(); ++i) {
    EXPECT(assert_equal<Pose3>(parent[i], result.at<Pose3>(i), 1e-2));
  }
}

/* ************************************************************************* */
TEST(TrajectoryAlignerSim3, ThreeChildrenNoisy) {
  const auto parent = makeParentPoses();

  PoseMeasurements aTi = makeMeasurements(parent, 1e-2);
  PoseMeasurements b1 =
      makeMeasurements(perturbPoses(transformPoses(gtSim1, parent)), 2e-2);
  PoseMeasurements b2 =
      makeMeasurements(perturbPoses(transformPoses(gtSim2, parent)), 2e-2);
  PoseMeasurements b3 =
      makeMeasurements(perturbPoses(transformPoses(gtSim3, parent)), 2e-2);

  ChildrenPoses bTi_all{b1, b2, b3};
  std::vector<Similarity3> sims{perturbSim3(gtSim1), perturbSim3(gtSim2),
                                perturbSim3(gtSim3)};

  TrajectoryAlignerSim3 aligner(aTi, bTi_all, sims);
  Values result = aligner.solve();

  EXPECT(simClose(gtSim1, result.at<Similarity3>(Symbol('S', 0)), 1e-1));
  EXPECT(simClose(gtSim2, result.at<Similarity3>(Symbol('S', 1)), 1e-1));
  EXPECT(simClose(gtSim3, result.at<Similarity3>(Symbol('S', 2)), 1e-1));

  for (size_t i = 0; i < parent.size(); ++i) {
    EXPECT(assert_equal<Pose3>(parent[i], result.at<Pose3>(i), 1e-2));
  }
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
