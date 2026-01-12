/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010-2020
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <gtsam/sfm/TrajectoryAlignerSim3.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/expressions.h>

#include <stdexcept>
#include <unordered_map>

namespace {

using namespace gtsam;

SharedNoiseModel chooseNoise(const SharedNoiseModel &noise, double sigma) {
  if (noise) return noise;
  return noiseModel::Isotropic::Sigma(6, sigma);
}

using PoseMeasurements = std::vector<UnaryMeasurement<Pose3>>;
struct MeasurementPair {
  const UnaryMeasurement<Pose3> &first;
  const UnaryMeasurement<Pose3> &second;
};

std::vector<MeasurementPair> overlappingMeasurementPairs(
    const PoseMeasurements &first, const PoseMeasurements &second) {
  std::unordered_map<Key, size_t> indexLookup;
  indexLookup.reserve(second.size());
  for (size_t i = 0; i < second.size(); ++i) {
    indexLookup.emplace(second[i].key(), i);
  }

  std::vector<MeasurementPair> pairs;
  pairs.reserve(std::min(first.size(), second.size()));
  for (const auto &m : first) {
    auto it = indexLookup.find(m.key());
    if (it != indexLookup.end()) {
      pairs.push_back({m, second[it->second]});
    }
  }
  return pairs;
}

Similarity3 estimateInitialSim3(const std::vector<MeasurementPair> &overlapPairs) {
  if (overlapPairs.size() < 2) return Similarity3();
  Pose3Pairs pairs;
  pairs.reserve(overlapPairs.size());
  for (const auto &[p1, p2] : overlapPairs) {
    // Similarity3::Align expects the pairs to be in the form (aTi, bTi)
    // to estimate aSb, but we want bSa, so we swap the pairs.
    pairs.emplace_back(p2.measured(), p1.measured());
  }
  try {
    return Similarity3::Align(pairs);
  } catch (const std::exception &) {
    return Similarity3();
  }
}

}  // namespace

namespace gtsam {
TrajectoryAlignerSim3::TrajectoryAlignerSim3(
    const PoseMeasurements &aTi, const ChildrenPoses &bTi_all,
    const std::vector<Similarity3> &bSa_all) {
  const size_t childCount = bTi_all.size();
  if (!bSa_all.empty() && bSa_all.size() != childCount) {
    throw std::invalid_argument(
        "TrajectoryAlignerSim3: bSa_all and bTi_all sizes differ");
  }

  // Add priors for all parent poses up front.
  for (const auto &meas : aTi) {
    initial_.insert(meas.key(), meas.measured());
    graph_.addExpressionFactor(Pose3_(meas.key()), meas.measured(),
                               chooseNoise(meas.noiseModel(), 1e-2));
  }

  // Parent-child constraints (only where camera exists in parent).
  for (size_t childIdx = 0; childIdx < childCount; ++childIdx) {
    const auto &bTi = bTi_all[childIdx];

    const Key simKey = Symbol('S', childIdx);
    if (!bSa_all.empty()) {
      initial_.insert(simKey, bSa_all[childIdx]);
    } else {
      const auto overlap = overlappingMeasurementPairs(aTi, bTi);
      initial_.insert(simKey, estimateInitialSim3(overlap));
    }
    
    const Expression<Similarity3> bSa(simKey);
    for (const auto &meas : bTi) {
      Key cameraKey = meas.key();
      if (!initial_.exists(cameraKey)) continue;

      const Pose3_ aPose(cameraKey);
      const Pose3_ expected = Pose3_(bSa, &Similarity3::transformFrom, aPose);
      graph_.addExpressionFactor(expected, meas.measured(), meas.noiseModel());
    }
  }
}

Values TrajectoryAlignerSim3::solve() const {
  if (graph_.empty() || initial_.empty()) {
    return initial_;
  }
  LevenbergMarquardtOptimizer optimizer(graph_, initial_);
  return optimizer.optimize();
}

}  // namespace gtsam

