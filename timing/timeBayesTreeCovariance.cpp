/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    timeBayesTreeCovariance.cpp
 * @brief   Benchmark Bayes-tree covariance recovery variants.
 * @date    March 2026
 * @author  Codex 5.4, prompted by Frank Dellaert
 */

#include <gtsam/config.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/linear/GaussianBayesNet.h>
#include <gtsam/linear/GaussianBayesTree.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/dataset.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace gtsam;
using namespace std;

namespace {

/// Covariance-recovery variant used in the benchmark.
enum class Variant {
  LegacyDense,
  SteinerDense,
  LegacySolve,
  SteinerSolve,
};

/// One benchmark query together with any selected-block split.
struct QueryCase {
  string family;
  size_t querySize;
  KeyVector keys;
  KeyVector left;
  KeyVector right;
  bool crossCovariance = false;
};

/// Structural statistics for the support touched by a query.
struct SupportStats {
  size_t supportCliques = 0;
  size_t compressedCliques = 0;
};

/// Timing result for one query under one variant.
struct RawResult {
  string dataset;
  string ordering;
  string family;
  string mode;
  string variant;
  size_t querySize = 0;
  size_t queryIndex = 0;
  size_t repeatIndex = 0;
  double totalMs = 0.0;
  double reductionMs = 0.0;
  double extractionMs = 0.0;
  size_t supportCliques = 0;
  size_t compressedCliques = 0;
  size_t reducedStateDim = 0;
};

/// Grouping key for per-query aggregation across repeated timings.
struct PerQueryKey {
  string dataset;
  string ordering;
  string family;
  string mode;
  string variant;
  size_t querySize = 0;
  size_t queryIndex = 0;

  bool operator==(const PerQueryKey& other) const {
    return dataset == other.dataset && ordering == other.ordering &&
           family == other.family && mode == other.mode &&
           variant == other.variant && querySize == other.querySize &&
           queryIndex == other.queryIndex;
  }
};

/// Hash function for PerQueryKey.
struct PerQueryKeyHash {
  size_t operator()(const PerQueryKey& key) const {
    size_t seed = std::hash<string>()(key.dataset);
    seed ^= std::hash<string>()(key.ordering) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^= std::hash<string>()(key.family) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^=
        std::hash<string>()(key.mode) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<string>()(key.variant) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^= std::hash<size_t>()(key.querySize) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^= std::hash<size_t>()(key.queryIndex) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    return seed;
  }
};

/// Aggregated result for one fixed query across repeated timing samples.
struct PerQueryResult {
  string dataset;
  string ordering;
  string family;
  string mode;
  string variant;
  size_t querySize = 0;
  size_t queryIndex = 0;
  size_t repeats = 0;
  double meanTotalMs = 0.0;
  double meanReductionMs = 0.0;
  double meanExtractionMs = 0.0;
  double medianTotalMs = 0.0;
  double medianReductionMs = 0.0;
  double medianExtractionMs = 0.0;
  double stddevTotalMs = 0.0;
  double stddevReductionMs = 0.0;
  double stddevExtractionMs = 0.0;
  size_t supportCliques = 0;
  size_t compressedCliques = 0;
  size_t reducedStateDim = 0;
};

/// Grouping key for aggregating repeated benchmark queries.
struct SummaryKey {
  string dataset;
  string ordering;
  string family;
  string mode;
  string variant;
  size_t querySize = 0;

  bool operator==(const SummaryKey& other) const {
    return dataset == other.dataset && ordering == other.ordering &&
           family == other.family && mode == other.mode &&
           variant == other.variant && querySize == other.querySize;
  }
};

/// Hash function for SummaryKey.
struct SummaryKeyHash {
  size_t operator()(const SummaryKey& key) const {
    size_t seed = std::hash<string>()(key.dataset);
    seed ^= std::hash<string>()(key.ordering) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^= std::hash<string>()(key.family) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^=
        std::hash<string>()(key.mode) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<string>()(key.variant) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^= std::hash<size_t>()(key.querySize) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    return seed;
  }
};

/// Aggregated timing samples for one SummaryKey.
struct SummaryValue {
  vector<double> queryMeanTotalMs;
  vector<double> queryMeanReductionMs;
  vector<double> queryMeanExtractionMs;
  vector<double> supportCliques;
  vector<double> compressedCliques;
  vector<double> reducedStateDim;
  size_t repeats = 0;
};

/// Return a deduplicated, sorted key list.
KeyVector uniqueSortedKeys(const KeyVector& keys) {
  KeyVector result = keys;
  sort(result.begin(), result.end());
  result.erase(unique(result.begin(), result.end()), result.end());
  return result;
}

/// Return all Pose2 keys in sorted order.
KeyVector poseKeys(const Values& values) {
  KeyVector keys;
  for (const auto& keyValue : values.extract<Pose2>()) {
    keys.push_back(keyValue.first);
  }
  sort(keys.begin(), keys.end());
  return keys;
}

/// Return the variable dimensions for a key list.
vector<size_t> dimsForKeys(const KeyVector& keys, const Values& values) {
  vector<size_t> dims;
  dims.reserve(keys.size());
  for (Key key : keys) {
    dims.push_back(values.at(key).dim());
  }
  return dims;
}

/// Convert block dimensions to prefix offsets.
vector<size_t> blockOffsets(const vector<size_t>& dims) {
  vector<size_t> offsets(dims.size() + 1, 0);
  for (size_t i = 0; i < dims.size(); ++i) {
    offsets[i + 1] = offsets[i] + dims[i];
  }
  return offsets;
}

/// Sum the dimensions of a key set.
size_t totalDimension(const KeySet& keys, const Values& values) {
  size_t dim = 0;
  for (Key key : keys) {
    dim += values.at(key).dim();
  }
  return dim;
}

/// Compute the lowest common ancestor of two Bayes-tree cliques.
template <class CLIQUE>
shared_ptr<CLIQUE> findLowestCommonAncestor(const shared_ptr<CLIQUE>& lhs,
                                            const shared_ptr<CLIQUE>& rhs) {
  unordered_set<shared_ptr<CLIQUE>> ancestors;
  for (auto current = lhs; current; current = current->parent()) {
    ancestors.insert(current);
  }
  for (auto current = rhs; current; current = current->parent()) {
    if (ancestors.count(current)) {
      return current;
    }
  }
  return nullptr;
}

/// Estimate the support size and compressed size for a query.
SupportStats analyzeSupport(const GaussianBayesTree& bayesTree,
                            const KeyVector& queryKeys) {
  vector<GaussianBayesTree::sharedClique> queryCliques;
  unordered_set<GaussianBayesTree::sharedClique> seen;
  for (Key key : queryKeys) {
    auto clique = bayesTree.clique(key);
    if (seen.insert(clique).second) {
      queryCliques.push_back(clique);
    }
  }
  if (queryCliques.empty()) {
    return {};
  }

  auto root = queryCliques.front();
  for (size_t i = 1; i < queryCliques.size(); ++i) {
    root = findLowestCommonAncestor(root, queryCliques[i]);
  }

  unordered_set<GaussianBayesTree::sharedClique> support;
  support.insert(root);
  for (const auto& clique : queryCliques) {
    for (auto current = clique; current && current != root;
         current = current->parent()) {
      support.insert(current);
    }
  }

  unordered_map<GaussianBayesTree::sharedClique, size_t> supportChildren;
  for (const auto& clique : support) {
    supportChildren[clique] = 0;
  }
  for (const auto& clique : support) {
    if (clique == root) {
      continue;
    }
    auto parent = clique->parent();
    if (parent && support.count(parent)) {
      ++supportChildren[parent];
    }
  }

  unordered_set<GaussianBayesTree::sharedClique> querySet(queryCliques.begin(),
                                                          queryCliques.end());
  size_t compressed = 1;
  for (const auto& clique : support) {
    if (clique == root) {
      continue;
    }
    if (querySet.count(clique) || supportChildren[clique] > 1) {
      ++compressed;
    }
  }

  return {support.size(), compressed};
}

/// Build the legacy reduced factor graph by marginal re-elimination.
GaussianFactorGraph legacyReducedFactorGraph(const GaussianFactorGraph& graph,
                                             const Ordering& fullOrdering,
                                             const KeyVector& queryKeys) {
  (void)fullOrdering;
  return GaussianFactorGraph(
      *graph.marginalMultifrontalBayesTree(queryKeys, EliminatePreferCholesky));
}

/// Build the localized reduced factor graph using Bayes-tree joint extraction.
GaussianFactorGraph steinerReducedFactorGraph(
    const GaussianBayesTree& bayesTree, const KeyVector& queryKeys) {
  return *bayesTree.joint(queryKeys, EliminatePreferCholesky);
}

/// Build the legacy two-key reduced factor graph using the pairwise joint path.
GaussianFactorGraph pairReducedFactorGraph(const GaussianBayesTree& bayesTree,
                                           const KeyVector& queryKeys) {
  return *bayesTree.joint(queryKeys[0], queryKeys[1], EliminatePreferCholesky);
}

/// Eliminate a reduced factor graph to a Bayes net ordered by the query keys.
GaussianBayesNet queryBayesNet(const GaussianFactorGraph& graph,
                               const KeyVector& queryKeys) {
  return *graph.marginalMultifrontalBayesNet(Ordering(queryKeys),
                                             EliminatePreferCholesky);
}

/// Recover selected covariance columns using triangular solves.
Matrix covarianceColumns(const GaussianBayesNet& bayesNet,
                         const KeyVector& orderedKeys,
                         const vector<size_t>& dims,
                         const vector<size_t>& selectedBlocks) {
  const auto [R, rhs] = bayesNet.matrix(Ordering(orderedKeys));
  (void)rhs;
  const vector<size_t> offsets = blockOffsets(dims);
  const size_t totalDim = offsets.back();

  size_t selectedDim = 0;
  for (size_t blockIndex : selectedBlocks) {
    selectedDim += dims.at(blockIndex);
  }

  Matrix selectors = Matrix::Zero(totalDim, selectedDim);
  size_t selectedOffset = 0;
  for (size_t blockIndex : selectedBlocks) {
    const size_t begin = offsets[blockIndex];
    const size_t dim = dims[blockIndex];
    selectors.block(begin, selectedOffset, dim, dim).setIdentity();
    selectedOffset += dim;
  }

  Matrix intermediate =
      R.transpose().triangularView<Eigen::Lower>().solve(selectors);
  return R.triangularView<Eigen::Upper>().solve(intermediate);
}

/// Extract a left-by-right cross-covariance block from packed selected columns.
Matrix extractPackedCrossBlock(const Matrix& selectedColumns,
                               const KeyVector& orderedKeys,
                               const vector<size_t>& dims,
                               const KeyVector& left, const KeyVector& right) {
  const FastMap<Key, size_t> keyIndex = Ordering(orderedKeys).invert();
  const vector<size_t> offsets = blockOffsets(dims);
  size_t leftDim = 0;
  for (Key key : left) {
    leftDim += dims.at(keyIndex.at(key));
  }
  size_t rightDim = 0;
  for (Key key : right) {
    rightDim += dims.at(keyIndex.at(key));
  }

  Matrix result(leftDim, rightDim);
  size_t rowOffset = 0;
  for (Key leftKey : left) {
    const size_t leftBlock = keyIndex.at(leftKey);
    const size_t leftDimBlock = dims[leftBlock];
    const size_t leftBegin = offsets[leftBlock];
    size_t columnOffset = 0;
    size_t selectedOffset = 0;
    for (Key rightKey : right) {
      const size_t rightBlock = keyIndex.at(rightKey);
      const size_t rightDimBlock = dims[rightBlock];
      result.block(rowOffset, columnOffset, leftDimBlock, rightDimBlock) =
          selectedColumns.block(leftBegin, selectedOffset, leftDimBlock,
                                rightDimBlock);
      columnOffset += rightDimBlock;
      selectedOffset += rightDimBlock;
    }
    rowOffset += leftDimBlock;
  }
  return result;
}

/// Extract a left-by-right cross-covariance block from a full covariance
/// matrix.
Matrix extractDenseCrossBlock(const Matrix& fullCovariance,
                              const KeyVector& orderedKeys,
                              const vector<size_t>& dims, const KeyVector& left,
                              const KeyVector& right) {
  const FastMap<Key, size_t> keyIndex = Ordering(orderedKeys).invert();
  const vector<size_t> offsets = blockOffsets(dims);
  size_t leftDim = 0;
  for (Key key : left) {
    leftDim += dims.at(keyIndex.at(key));
  }
  size_t rightDim = 0;
  for (Key key : right) {
    rightDim += dims.at(keyIndex.at(key));
  }

  Matrix result(leftDim, rightDim);
  size_t rowOffset = 0;
  for (Key leftKey : left) {
    const size_t leftBlock = keyIndex.at(leftKey);
    const size_t leftDimBlock = dims[leftBlock];
    const size_t leftBegin = offsets[leftBlock];
    size_t columnOffset = 0;
    for (Key rightKey : right) {
      const size_t rightBlock = keyIndex.at(rightKey);
      const size_t rightDimBlock = dims[rightBlock];
      const size_t rightBegin = offsets[rightBlock];
      result.block(rowOffset, columnOffset, leftDimBlock, rightDimBlock) =
          fullCovariance.block(leftBegin, rightBegin, leftDimBlock,
                               rightDimBlock);
      columnOffset += rightDimBlock;
    }
    rowOffset += leftDimBlock;
  }
  return result;
}

/// Convert a Variant enum to a CSV-friendly name.
string variantName(Variant variant) {
  switch (variant) {
    case Variant::LegacyDense:
      return "legacy_dense";
    case Variant::SteinerDense:
      return "steiner_dense";
    case Variant::LegacySolve:
      return "legacy_solve";
    case Variant::SteinerSolve:
      return "steiner_solve";
  }
  return "unknown";
}

/// Convert an ordering type to a human-readable name.
string orderingName(Ordering::OrderingType orderingType) {
  return orderingType == Ordering::METIS ? "METIS" : "COLAMD";
}

/// Compute the median of a small sample vector.
double median(vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  if (values.size() % 2 == 0) {
    return 0.5 * (values[mid - 1] + values[mid]);
  }
  return values[mid];
}

/// Compute the arithmetic mean of a sample vector.
double mean(const vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  const double sum = accumulate(values.begin(), values.end(), 0.0);
  return sum / static_cast<double>(values.size());
}

/// Compute the population standard deviation of a sample vector.
double stddev(const vector<double>& values) {
  if (values.size() < 2) {
    return 0.0;
  }
  const double sampleMean = mean(values);
  double sumSquares = 0.0;
  for (double value : values) {
    const double delta = value - sampleMean;
    sumSquares += delta * delta;
  }
  return sqrt(sumSquares / static_cast<double>(values.size()));
}

/// Sample representative starting indices across a window range.
vector<size_t> sampledStarts(size_t maxStartInclusive, size_t maxQueries,
                             size_t stride = 1) {
  vector<size_t> starts;
  if (maxStartInclusive == 0) {
    starts.push_back(0);
    return starts;
  }

  for (size_t start = 0;
       start <= maxStartInclusive && starts.size() < maxQueries;
       start += stride) {
    starts.push_back(start);
  }
  if (starts.size() > maxQueries) {
    starts.resize(maxQueries);
  }
  if (starts.size() < maxQueries && starts.back() != maxStartInclusive) {
    starts.push_back(maxStartInclusive);
  }

  if (starts.size() > maxQueries) {
    vector<size_t> reduced;
    reduced.reserve(maxQueries);
    for (size_t i = 0; i < maxQueries; ++i) {
      const size_t index = static_cast<size_t>(
          llround((starts.size() - 1) * (double(i) / double(maxQueries - 1))));
      reduced.push_back(starts[index]);
    }
    starts = reduced;
  }
  return starts;
}

/// Generate contiguous local-window benchmark queries.
vector<QueryCase> generateLocalWindows(const KeyVector& poseKeys,
                                       size_t querySize, size_t maxQueries) {
  vector<QueryCase> queries;
  if (poseKeys.size() < querySize) {
    return queries;
  }
  const size_t maxStart = poseKeys.size() - querySize;
  for (size_t start :
       sampledStarts(maxStart, maxQueries, max<size_t>(1, querySize / 2))) {
    QueryCase query;
    query.family = "local_window";
    query.querySize = querySize;
    query.keys.assign(poseKeys.begin() + start,
                      poseKeys.begin() + start + querySize);
    queries.push_back(query);
  }
  return queries;
}

/// Generate single-pose queries that exercise the legacy marginal path.
vector<QueryCase> generateSinglePoseQueries(const KeyVector& poseKeys,
                                            size_t maxQueries) {
  vector<QueryCase> queries;
  if (poseKeys.empty()) {
    return queries;
  }
  for (size_t start :
       sampledStarts(poseKeys.size() - 1, maxQueries,
                     max<size_t>(1, poseKeys.size() / maxQueries))) {
    QueryCase query;
    query.family = "single_pose";
    query.querySize = 1;
    query.keys = {poseKeys[start]};
    queries.push_back(query);
  }
  return queries;
}

/// Generate adjacent-pair queries that exercise the legacy pairwise path.
vector<QueryCase> generatePairPoseQueries(const KeyVector& poseKeys,
                                          size_t maxQueries) {
  vector<QueryCase> queries;
  if (poseKeys.size() < 2) {
    return queries;
  }
  for (size_t start : sampledStarts(poseKeys.size() - 2, maxQueries, 1)) {
    QueryCase query;
    query.family = "pair_pose";
    query.querySize = 2;
    query.keys = {poseKeys[start], poseKeys[start + 1]};
    queries.push_back(query);
  }
  return queries;
}

/// Generate highly overlapping windows to test repeated local queries.
vector<QueryCase> generateRepeatedOverlap(const KeyVector& poseKeys,
                                          size_t querySize, size_t maxQueries) {
  vector<QueryCase> queries;
  if (poseKeys.size() < querySize) {
    return queries;
  }
  const size_t maxStart = poseKeys.size() - querySize;
  for (size_t start = 0; start <= maxStart && queries.size() < maxQueries;
       ++start) {
    QueryCase query;
    query.family = "overlap_window";
    query.querySize = querySize;
    query.keys.assign(poseKeys.begin() + start,
                      poseKeys.begin() + start + querySize);
    queries.push_back(query);
  }
  return queries;
}

/// Generate wide-separated queries spread across the trajectory.
vector<QueryCase> generateWideSeparated(const KeyVector& poseKeys,
                                        size_t querySize, size_t maxQueries) {
  vector<QueryCase> queries;
  if (poseKeys.size() < querySize) {
    return queries;
  }

  const double gap = querySize == 1
                         ? 0.0
                         : double(poseKeys.size() - 1) / double(querySize - 1);
  for (size_t sample = 0; sample < maxQueries; ++sample) {
    const double offset =
        maxQueries == 0 ? 0.0
                        : (gap / max<double>(1.0, double(maxQueries))) * sample;
    QueryCase query;
    query.family = "wide_separated";
    query.querySize = querySize;
    size_t previous = 0;
    for (size_t j = 0; j < querySize; ++j) {
      size_t index = static_cast<size_t>(llround(offset + gap * j));
      index = min(index, poseKeys.size() - querySize + j);
      if (j > 0) {
        index = max(index, previous + 1);
      }
      previous = index;
      query.keys.push_back(poseKeys[index]);
    }
    queries.push_back(query);
  }
  return queries;
}

/// Generate selected cross-covariance queries from overlapping windows.
vector<QueryCase> generateSelectedCross(const KeyVector& poseKeys,
                                        size_t querySize, size_t maxQueries) {
  vector<QueryCase> queries;
  if (poseKeys.size() < querySize || querySize < 2) {
    return queries;
  }
  const size_t split = querySize / 2;
  for (QueryCase query :
       generateRepeatedOverlap(poseKeys, querySize, maxQueries)) {
    query.family = "selected_cross";
    query.crossCovariance = true;
    query.left.assign(query.keys.begin(), query.keys.begin() + split);
    query.right.assign(query.keys.begin() + split, query.keys.end());
    queries.push_back(query);
  }
  return queries;
}

/// Build the full benchmark workload across all query families.
vector<QueryCase> buildWorkload(const KeyVector& poseKeyList) {
  vector<QueryCase> queries;
  auto singlePose = generateSinglePoseQueries(poseKeyList, 32);
  queries.insert(queries.end(), singlePose.begin(), singlePose.end());

  auto pairPose = generatePairPoseQueries(poseKeyList, 32);
  queries.insert(queries.end(), pairPose.begin(), pairPose.end());

  for (size_t querySize : {size_t(3), size_t(5), size_t(10), size_t(20)}) {
    auto local = generateLocalWindows(poseKeyList, querySize, 16);
    queries.insert(queries.end(), local.begin(), local.end());

    auto wide = generateWideSeparated(poseKeyList, querySize, 8);
    queries.insert(queries.end(), wide.begin(), wide.end());
  }

  auto overlap = generateRepeatedOverlap(poseKeyList, 10, 32);
  queries.insert(queries.end(), overlap.begin(), overlap.end());

  for (size_t querySize : {size_t(10), size_t(20)}) {
    auto selected = generateSelectedCross(poseKeyList, querySize, 24);
    queries.insert(queries.end(), selected.begin(), selected.end());
  }
  return queries;
}

/// Time one query under one covariance-recovery variant.
RawResult benchmarkQuery(const string& datasetName, const string& orderingLabel,
                         Variant variant,
                         const GaussianFactorGraph& linearGraph,
                         const GaussianBayesTree& bayesTree,
                         const Ordering& fullOrdering, const Values& values,
                         const QueryCase& query, size_t queryIndex,
                         size_t repeatIndex) {
  const KeyVector orderedKeys = uniqueSortedKeys(query.keys);
  const SupportStats supportStats = analyzeSupport(bayesTree, orderedKeys);

  if (orderedKeys.size() == 1) {
    const auto reductionStart = chrono::steady_clock::now();
    const auto marginalFactor =
        bayesTree.marginalFactor(orderedKeys.front(), EliminatePreferCholesky);
    const Matrix information = marginalFactor->information();
    const auto reductionEnd = chrono::steady_clock::now();

    const auto extractionStart = chrono::steady_clock::now();
    Matrix recovered;
    if (variant == Variant::LegacyDense || variant == Variant::SteinerDense) {
      recovered = information.inverse();
    } else {
      recovered = information.selfadjointView<Eigen::Upper>().llt().solve(
          Matrix::Identity(information.rows(), information.cols()));
    }
    const auto extractionEnd = chrono::steady_clock::now();

    volatile double checksum = recovered.sum();
    (void)checksum;

    RawResult result;
    result.dataset = datasetName;
    result.ordering = orderingLabel;
    result.family = query.family;
    result.mode = "joint";
    result.variant = variantName(variant);
    result.querySize = query.querySize;
    result.queryIndex = queryIndex;
    result.repeatIndex = repeatIndex;
    result.reductionMs =
        chrono::duration<double, milli>(reductionEnd - reductionStart).count();
    result.extractionMs =
        chrono::duration<double, milli>(extractionEnd - extractionStart)
            .count();
    result.totalMs = result.reductionMs + result.extractionMs;
    result.supportCliques = 1;
    result.compressedCliques = 1;
    result.reducedStateDim = information.rows();
    return result;
  }

  const auto reductionStart = chrono::steady_clock::now();
  GaussianFactorGraph reducedGraph =
      orderedKeys.size() == 2
          ? pairReducedFactorGraph(bayesTree, orderedKeys)
          : ((variant == Variant::LegacyDense ||
              variant == Variant::LegacySolve)
                 ? legacyReducedFactorGraph(linearGraph, fullOrdering,
                                            orderedKeys)
                 : steinerReducedFactorGraph(bayesTree, orderedKeys));
  const auto reductionEnd = chrono::steady_clock::now();

  const auto extractionStart = chrono::steady_clock::now();
  GaussianBayesNet reducedBayesNet = queryBayesNet(reducedGraph, orderedKeys);
  const vector<size_t> dims = dimsForKeys(orderedKeys, values);
  Matrix recovered;
  if (query.crossCovariance) {
    const FastMap<Key, size_t> keyIndex = Ordering(orderedKeys).invert();
    vector<size_t> rightBlocks;
    rightBlocks.reserve(query.right.size());
    for (Key key : query.right) {
      rightBlocks.push_back(keyIndex.at(key));
    }

    Matrix selectedColumns;
    if (variant == Variant::LegacyDense || variant == Variant::SteinerDense) {
      const auto [R, rhs] = reducedBayesNet.matrix(Ordering(orderedKeys));
      (void)rhs;
      const Matrix information = R.transpose() * R;
      const Matrix covariance = information.inverse();
      recovered = extractDenseCrossBlock(covariance, orderedKeys, dims,
                                         query.left, query.right);
    } else {
      selectedColumns =
          covarianceColumns(reducedBayesNet, orderedKeys, dims, rightBlocks);
      recovered = extractPackedCrossBlock(selectedColumns, orderedKeys, dims,
                                          query.left, query.right);
    }
  } else {
    if (variant == Variant::LegacyDense || variant == Variant::SteinerDense) {
      const auto [R, rhs] = reducedBayesNet.matrix(Ordering(orderedKeys));
      (void)rhs;
      const Matrix information = R.transpose() * R;
      recovered = information.inverse();
    } else {
      vector<size_t> allBlocks(orderedKeys.size());
      iota(allBlocks.begin(), allBlocks.end(), 0);
      recovered =
          covarianceColumns(reducedBayesNet, orderedKeys, dims, allBlocks);
    }
  }
  const auto extractionEnd = chrono::steady_clock::now();

  volatile double checksum = recovered.sum();
  (void)checksum;

  RawResult result;
  result.dataset = datasetName;
  result.ordering = orderingLabel;
  result.family = query.family;
  result.mode = query.crossCovariance ? "cross" : "joint";
  result.variant = variantName(variant);
  result.querySize = query.querySize;
  result.queryIndex = queryIndex;
  result.repeatIndex = repeatIndex;
  result.reductionMs =
      chrono::duration<double, milli>(reductionEnd - reductionStart).count();
  result.extractionMs =
      chrono::duration<double, milli>(extractionEnd - extractionStart).count();
  result.totalMs = result.reductionMs + result.extractionMs;
  result.supportCliques = supportStats.supportCliques;
  result.compressedCliques = supportStats.compressedCliques;
  result.reducedStateDim = totalDimension(reducedGraph.keys(), values);
  return result;
}

/// Write per-query timing results to CSV.
void writeRawCsv(const filesystem::path& path,
                 const vector<RawResult>& results) {
  ofstream os(path);
  os << "dataset,ordering,query_family,mode,variant,query_size,query_index,"
        "repeat_index,"
        "total_ms,reduction_ms,extraction_ms,support_cliques,compressed_"
        "cliques,"
        "reduced_state_dim\n";
  os << fixed << setprecision(6);
  for (const auto& result : results) {
    os << result.dataset << ',' << result.ordering << ',' << result.family
       << ',' << result.mode << ',' << result.variant << ',' << result.querySize
       << ',' << result.queryIndex << ',' << result.repeatIndex << ','
       << result.totalMs << ',' << result.reductionMs << ','
       << result.extractionMs << ',' << result.supportCliques << ','
       << result.compressedCliques << ',' << result.reducedStateDim << '\n';
  }
}

/// Aggregate repeated raw timings into one row per distinct query.
vector<PerQueryResult> aggregatePerQueryResults(
    const vector<RawResult>& results) {
  struct PerQuerySamples {
    vector<double> totalMs;
    vector<double> reductionMs;
    vector<double> extractionMs;
    size_t supportCliques = 0;
    size_t compressedCliques = 0;
    size_t reducedStateDim = 0;
  };

  unordered_map<PerQueryKey, PerQuerySamples, PerQueryKeyHash> perQuery;
  for (const auto& result : results) {
    const PerQueryKey key{result.dataset,   result.ordering, result.family,
                          result.mode,      result.variant,  result.querySize,
                          result.queryIndex};
    auto& value = perQuery[key];
    value.totalMs.push_back(result.totalMs);
    value.reductionMs.push_back(result.reductionMs);
    value.extractionMs.push_back(result.extractionMs);
    value.supportCliques = result.supportCliques;
    value.compressedCliques = result.compressedCliques;
    value.reducedStateDim = result.reducedStateDim;
  }

  vector<PerQueryResult> aggregated;
  aggregated.reserve(perQuery.size());
  for (const auto& [key, value] : perQuery) {
    PerQueryResult result;
    result.dataset = key.dataset;
    result.ordering = key.ordering;
    result.family = key.family;
    result.mode = key.mode;
    result.variant = key.variant;
    result.querySize = key.querySize;
    result.queryIndex = key.queryIndex;
    result.repeats = value.totalMs.size();
    result.meanTotalMs = mean(value.totalMs);
    result.meanReductionMs = mean(value.reductionMs);
    result.meanExtractionMs = mean(value.extractionMs);
    result.medianTotalMs = median(value.totalMs);
    result.medianReductionMs = median(value.reductionMs);
    result.medianExtractionMs = median(value.extractionMs);
    result.stddevTotalMs = stddev(value.totalMs);
    result.stddevReductionMs = stddev(value.reductionMs);
    result.stddevExtractionMs = stddev(value.extractionMs);
    result.supportCliques = value.supportCliques;
    result.compressedCliques = value.compressedCliques;
    result.reducedStateDim = value.reducedStateDim;
    aggregated.push_back(result);
  }
  return aggregated;
}

/// Write per-query aggregated timing results to CSV.
void writePerQueryCsv(const filesystem::path& path,
                      const vector<PerQueryResult>& results) {
  ofstream os(path);
  os << "dataset,ordering,query_family,mode,variant,query_size,query_index,"
        "repeats,mean_total_ms,mean_reduction_ms,mean_extraction_ms,"
        "median_total_ms,median_reduction_ms,median_extraction_ms,"
        "stddev_total_ms,stddev_reduction_ms,stddev_extraction_ms,"
        "support_cliques,compressed_cliques,reduced_state_dim\n";
  os << fixed << setprecision(6);
  for (const auto& result : results) {
    os << result.dataset << ',' << result.ordering << ',' << result.family
       << ',' << result.mode << ',' << result.variant << ',' << result.querySize
       << ',' << result.queryIndex << ',' << result.repeats << ','
       << result.meanTotalMs << ',' << result.meanReductionMs << ','
       << result.meanExtractionMs << ',' << result.medianTotalMs << ','
       << result.medianReductionMs << ',' << result.medianExtractionMs << ','
       << result.stddevTotalMs << ',' << result.stddevReductionMs << ','
       << result.stddevExtractionMs << ',' << result.supportCliques << ','
       << result.compressedCliques << ',' << result.reducedStateDim << '\n';
  }
}

/// Write family-level summary statistics aggregated from per-query results.
void writeSummaryCsv(const filesystem::path& path,
                     const vector<PerQueryResult>& results) {
  unordered_map<SummaryKey, SummaryValue, SummaryKeyHash> summary;
  for (const auto& result : results) {
    const SummaryKey key{result.dataset, result.ordering, result.family,
                         result.mode,    result.variant,  result.querySize};
    auto& value = summary[key];
    value.queryMeanTotalMs.push_back(result.meanTotalMs);
    value.queryMeanReductionMs.push_back(result.meanReductionMs);
    value.queryMeanExtractionMs.push_back(result.meanExtractionMs);
    value.supportCliques.push_back(static_cast<double>(result.supportCliques));
    value.compressedCliques.push_back(
        static_cast<double>(result.compressedCliques));
    value.reducedStateDim.push_back(
        static_cast<double>(result.reducedStateDim));
    value.repeats = result.repeats;
  }

  ofstream os(path);
  os << "dataset,ordering,query_family,mode,variant,query_size,queries,repeats,"
        "median_total_ms,sum_query_mean_total_ms,median_reduction_ms,"
        "median_extraction_ms,support_cliques,compressed_cliques,"
        "reduced_state_dim\n";
  os << fixed << setprecision(6);
  for (const auto& [key, value] : summary) {
    const double totalTime = accumulate(value.queryMeanTotalMs.begin(),
                                        value.queryMeanTotalMs.end(), 0.0);
    os << key.dataset << ',' << key.ordering << ',' << key.family << ','
       << key.mode << ',' << key.variant << ',' << key.querySize << ','
       << value.queryMeanTotalMs.size() << ',' << value.repeats << ','
       << median(value.queryMeanTotalMs) << ',' << totalTime << ','
       << median(value.queryMeanReductionMs) << ','
       << median(value.queryMeanExtractionMs) << ','
       << median(value.supportCliques) << ',' << median(value.compressedCliques)
       << ',' << median(value.reducedStateDim) << '\n';
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

/// Read an unsigned integer argument from argv or return a default value.
size_t sizeTArgumentOrDefault(char** begin, char** end, const string& flag,
                              size_t defaultValue) {
  return static_cast<size_t>(
      stoull(argumentOrDefault(begin, end, flag, to_string(defaultValue))));
}

/// Split a comma-separated argument into individual values.
vector<string> splitCommaSeparated(const string& input) {
  vector<string> values;
  size_t start = 0;
  while (start < input.size()) {
    const size_t comma = input.find(',', start);
    if (comma == string::npos) {
      values.push_back(input.substr(start));
      break;
    }
    values.push_back(input.substr(start, comma - start));
    start = comma + 1;
  }
  return values;
}

}  // namespace

int main(int argc, char** argv) {
  const filesystem::path outputDir = argumentOrDefault(
      argv, argv + argc, "--output-dir",
      (filesystem::path("timing") / "results" / "bayes_tree_covariance")
          .string());
  const vector<string> datasets = splitCommaSeparated(argumentOrDefault(
      argv, argv + argc, "--datasets", "w100.graph,w10000.graph,w20000.txt"));
  const size_t repeats =
      sizeTArgumentOrDefault(argv, argv + argc, "--repeats", 10);

  filesystem::create_directories(outputDir);

  vector<RawResult> rawResults;

  for (const string& datasetName : datasets) {
    cout << "Loading " << datasetName << endl;
    const auto [graphPtr, initialPtr] =
        load2D(findExampleDataFile(datasetName));
    const KeyVector anchoredPoseKeys = poseKeys(*initialPtr);
    if (!anchoredPoseKeys.empty()) {
      graphPtr->addPrior(anchoredPoseKeys.front(),
                         initialPtr->at<Pose2>(anchoredPoseKeys.front()),
                         noiseModel::Diagonal::Sigmas(
                             (Vector(3) << 1e-6, 1e-6, 1e-6).finished()));
    }

    cout << "Optimizing " << datasetName << endl;
    LevenbergMarquardtOptimizer optimizer(*graphPtr, *initialPtr);
    Values result = optimizer.optimize();
    GaussianFactorGraph linearGraph = *graphPtr->linearize(result);
    const KeyVector poseKeyList = poseKeys(result);
    const vector<QueryCase> workload = buildWorkload(poseKeyList);

    for (const auto orderingType : {Ordering::COLAMD, Ordering::METIS}) {
#ifndef GTSAM_SUPPORT_NESTED_DISSECTION
      if (orderingType == Ordering::METIS) {
        continue;
      }
#endif
      const Ordering ordering = Ordering::Create(orderingType, linearGraph);
      cout << "  Ordering " << orderingName(orderingType) << " with "
           << workload.size() << " queries" << endl;
      GaussianBayesTree bayesTree =
          *linearGraph.eliminateMultifrontal(ordering, EliminatePreferCholesky);

      size_t queryIndex = 0;
      for (const QueryCase& query : workload) {
        if (query.querySize == 1) {
          (void)bayesTree.marginalFactor(query.keys.front(),
                                         EliminatePreferCholesky);
        } else if (query.querySize == 2) {
          (void)bayesTree.joint(query.keys[0], query.keys[1],
                                EliminatePreferCholesky);
        }
        for (const Variant variant :
             {Variant::LegacyDense, Variant::SteinerDense, Variant::LegacySolve,
              Variant::SteinerSolve}) {
          try {
            (void)benchmarkQuery(datasetName, orderingName(orderingType),
                                 variant, linearGraph, bayesTree, ordering,
                                 result, query, queryIndex, 0);
            for (size_t repeatIndex = 0; repeatIndex < repeats; ++repeatIndex) {
              rawResults.push_back(benchmarkQuery(
                  datasetName, orderingName(orderingType), variant, linearGraph,
                  bayesTree, ordering, result, query, queryIndex, repeatIndex));
            }
          } catch (const std::exception& error) {
            cerr << "Failure for dataset=" << datasetName
                 << " ordering=" << orderingName(orderingType)
                 << " variant=" << variantName(variant)
                 << " family=" << query.family << " query_index=" << queryIndex
                 << ": " << error.what() << endl;
            throw;
          }
        }
        ++queryIndex;
      }
    }
  }

  writeRawCsv(outputDir / "raw.csv", rawResults);
  const vector<PerQueryResult> perQueryResults =
      aggregatePerQueryResults(rawResults);
  writePerQueryCsv(outputDir / "per_query.csv", perQueryResults);
  writeSummaryCsv(outputDir / "summary.csv", perQueryResults);
  cout << "Wrote benchmark results to " << outputDir << endl;
  return 0;
}
