/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    exportBayesTreeCovarianceVisuals.cpp
 * @brief   Export representative pose-graph query and covariance visuals.
 * @date    March 2026
 * @author  Frank Dellaert
 */

#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <string>
#include <vector>

using namespace gtsam;
using namespace std;

namespace {

/// Return all Pose2 keys in sorted order.
KeyVector poseKeys(const Values& values) {
  KeyVector keys;
  for (const auto& keyValue : values.extract<Pose2>()) {
    keys.push_back(keyValue.first);
  }
  sort(keys.begin(), keys.end());
  return keys;
}

/// Select a contiguous window near the center of the trajectory.
KeyVector centeredWindow(const KeyVector& keys, size_t querySize) {
  if (keys.size() <= querySize) {
    return keys;
  }
  const size_t start = (keys.size() - querySize) / 2;
  return KeyVector(keys.begin() + start, keys.begin() + start + querySize);
}

/// Select approximately evenly spaced keys across the full trajectory.
KeyVector evenlySpaced(const KeyVector& keys, size_t querySize) {
  if (keys.size() <= querySize) {
    return keys;
  }

  KeyVector query;
  query.reserve(querySize);
  for (size_t i = 0; i < querySize; ++i) {
    const double alpha =
        querySize == 1 ? 0.0 : double(i) / double(querySize - 1);
    const size_t index =
        static_cast<size_t>(llround(alpha * double(keys.size() - 1)));
    query.push_back(keys.at(index));
  }
  query.erase(unique(query.begin(), query.end()), query.end());
  return query;
}

/// Write a one-column key list CSV.
void writeKeyListCsv(const filesystem::path& path, const KeyVector& keys) {
  ofstream stream(path);
  stream << "key\n";
  for (Key key : keys) {
    stream << key << '\n';
  }
}

/// Write a dense matrix to CSV with fixed precision.
void writeMatrixCsv(const filesystem::path& path, const Matrix& matrix) {
  ofstream stream(path);
  stream << fixed << setprecision(12);
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
      if (column) {
        stream << ',';
      }
      stream << matrix(row, column);
    }
    stream << '\n';
  }
}

/// Export optimized Pose2 values together with query membership flags.
void writePoseCsv(const filesystem::path& path, const Values& values,
                  const KeyVector& localQuery, const KeyVector& wideQuery) {
  const set<Key> localKeys(localQuery.begin(), localQuery.end());
  const set<Key> wideKeys(wideQuery.begin(), wideQuery.end());

  ofstream stream(path);
  stream << "key,x,y,theta,in_local,in_wide\n";
  stream << fixed << setprecision(9);
  for (Key key : poseKeys(values)) {
    const Pose2 pose = values.at<Pose2>(key);
    stream << key << ',' << pose.x() << ',' << pose.y() << ',' << pose.theta()
           << ',' << (localKeys.count(key) ? 1 : 0) << ','
           << (wideKeys.count(key) ? 1 : 0) << '\n';
  }
}

/// Export the unique Pose2 measurement edges in the factor graph.
void writeEdgeCsv(const filesystem::path& path,
                  const NonlinearFactorGraph& graph) {
  set<pair<Key, Key>> edges;
  for (const auto& factor : graph) {
    auto between = dynamic_pointer_cast<const BetweenFactor<Pose2>>(factor);
    if (!between) {
      continue;
    }
    const Key key1 = between->key1();
    const Key key2 = between->key2();
    const auto edge = minmax(key1, key2);
    edges.insert(edge);
  }

  ofstream stream(path);
  stream << "key1,key2\n";
  for (const auto& [key1, key2] : edges) {
    stream << key1 << ',' << key2 << '\n';
  }
}

/// Read a string argument from argv or return a default value.
string argumentOrDefault(char** begin, char** end, const string& flag,
                         const string& defaultValue) {
  for (auto it = begin; it != end; ++it) {
    if (string(*it) == flag && it + 1 != end) {
      return *(it + 1);
    }
  }
  return defaultValue;
}

}  // namespace

int main(int argc, char** argv) {
  const string datasetName =
      argumentOrDefault(argv, argv + argc, "--dataset", "w100.graph");
  const filesystem::path outputDir =
      argumentOrDefault(argv, argv + argc, "--output-dir",
                        (filesystem::path("timing") / "results" /
                         "bayes_tree_covariance" / "visuals")
                            .string());
  filesystem::create_directories(outputDir);

  const auto [graphPtr, initialPtr] = load2D(findExampleDataFile(datasetName));
  const KeyVector initialPoseKeys = poseKeys(*initialPtr);
  if (!initialPoseKeys.empty()) {
    graphPtr->addPrior(initialPoseKeys.front(),
                       initialPtr->at<Pose2>(initialPoseKeys.front()),
                       noiseModel::Diagonal::Sigmas(
                           (Vector(3) << 1e-6, 1e-6, 1e-6).finished()));
  }

  LevenbergMarquardtOptimizer optimizer(*graphPtr, *initialPtr);
  const Values result = optimizer.optimize();
  const KeyVector queryCandidates = poseKeys(result);
  const KeyVector localQuery = centeredWindow(queryCandidates, 10);
  const KeyVector wideQuery = evenlySpaced(queryCandidates, 10);

  const Marginals marginals(*graphPtr, result, Marginals::CHOLESKY);
  const Matrix localCovariance =
      marginals.jointMarginalCovariance(localQuery).fullMatrix();
  const Matrix wideCovariance =
      marginals.jointMarginalCovariance(wideQuery).fullMatrix();

  writePoseCsv(outputDir / "w100_poses.csv", result, localQuery, wideQuery);
  writeEdgeCsv(outputDir / "w100_edges.csv", *graphPtr);
  writeKeyListCsv(outputDir / "w100_local_keys.csv", localQuery);
  writeKeyListCsv(outputDir / "w100_wide_keys.csv", wideQuery);
  writeMatrixCsv(outputDir / "w100_local_covariance.csv", localCovariance);
  writeMatrixCsv(outputDir / "w100_wide_covariance.csv", wideCovariance);

  return 0;
}
