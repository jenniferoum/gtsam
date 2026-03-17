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

enum class Variant {
  LegacyDense,
  SteinerDense,
  LegacySolve,
  SteinerSolve,
};

struct QueryCase {
  string family;
  size_t querySize;
  KeyVector keys;
  KeyVector left;
  KeyVector right;
  bool crossCovariance = false;
};

struct SupportStats {
  size_t supportCliques = 0;
  size_t compressedCliques = 0;
};

struct RawResult {
  string dataset;
  string ordering;
  string family;
  string mode;
  string variant;
  size_t querySize = 0;
  size_t queryIndex = 0;
  double totalMs = 0.0;
  double reductionMs = 0.0;
  double extractionMs = 0.0;
  size_t supportCliques = 0;
  size_t compressedCliques = 0;
  size_t reducedStateDim = 0;
};

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

struct SummaryValue {
  vector<double> totalMs;
  vector<double> reductionMs;
  vector<double> extractionMs;
  size_t supportCliques = 0;
  size_t compressedCliques = 0;
  size_t reducedStateDim = 0;
};

KeyVector uniqueSortedKeys(const KeyVector& keys) {
  KeyVector result = keys;
  sort(result.begin(), result.end());
  result.erase(unique(result.begin(), result.end()), result.end());
  return result;
}

KeyVector poseKeys(const Values& values) {
  KeyVector keys;
  for (const auto& keyValue : values.extract<Pose2>()) {
    keys.push_back(keyValue.first);
  }
  sort(keys.begin(), keys.end());
  return keys;
}

vector<size_t> dimsForKeys(const KeyVector& keys, const Values& values) {
  vector<size_t> dims;
  dims.reserve(keys.size());
  for (Key key : keys) {
    dims.push_back(values.at(key).dim());
  }
  return dims;
}

vector<size_t> blockOffsets(const vector<size_t>& dims) {
  vector<size_t> offsets(dims.size() + 1, 0);
  for (size_t i = 0; i < dims.size(); ++i) {
    offsets[i + 1] = offsets[i] + dims[i];
  }
  return offsets;
}

size_t totalDimension(const KeySet& keys, const Values& values) {
  size_t dim = 0;
  for (Key key : keys) {
    dim += values.at(key).dim();
  }
  return dim;
}

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

GaussianFactorGraph legacyReducedFactorGraph(const GaussianFactorGraph& graph,
                                             const Ordering& fullOrdering,
                                             const KeyVector& queryKeys) {
  (void)fullOrdering;
  return GaussianFactorGraph(
      *graph.marginalMultifrontalBayesTree(queryKeys, EliminatePreferCholesky));
}

GaussianFactorGraph steinerReducedFactorGraph(
    const GaussianBayesTree& bayesTree, const KeyVector& queryKeys) {
  return *bayesTree.joint(queryKeys, EliminatePreferCholesky);
}

GaussianBayesNet queryBayesNet(const GaussianFactorGraph& graph,
                               const KeyVector& queryKeys) {
  return *graph.marginalMultifrontalBayesNet(Ordering(queryKeys),
                                             EliminatePreferCholesky);
}

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

Matrix extractCrossBlock(const Matrix& selectedColumns,
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

string orderingName(Ordering::OrderingType orderingType) {
  return orderingType == Ordering::METIS ? "METIS" : "COLAMD";
}

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

vector<QueryCase> buildWorkload(const KeyVector& poseKeyList) {
  vector<QueryCase> queries;
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

RawResult benchmarkQuery(const string& datasetName, const string& orderingLabel,
                         Variant variant,
                         const GaussianFactorGraph& linearGraph,
                         const GaussianBayesTree& bayesTree,
                         const Ordering& fullOrdering, const Values& values,
                         const QueryCase& query, size_t queryIndex) {
  const KeyVector orderedKeys = uniqueSortedKeys(query.keys);
  const SupportStats supportStats = analyzeSupport(bayesTree, orderedKeys);

  const auto reductionStart = chrono::steady_clock::now();
  GaussianFactorGraph reducedGraph =
      (variant == Variant::LegacyDense || variant == Variant::LegacySolve)
          ? legacyReducedFactorGraph(linearGraph, fullOrdering, orderedKeys)
          : steinerReducedFactorGraph(bayesTree, orderedKeys);
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
      selectedColumns = covariance;
      recovered = extractCrossBlock(selectedColumns, orderedKeys, dims,
                                    query.left, query.right);
    } else {
      selectedColumns =
          covarianceColumns(reducedBayesNet, orderedKeys, dims, rightBlocks);
      recovered = extractCrossBlock(selectedColumns, orderedKeys, dims,
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

void writeRawCsv(const filesystem::path& path,
                 const vector<RawResult>& results) {
  ofstream os(path);
  os << "dataset,ordering,query_family,mode,variant,query_size,query_index,"
        "total_ms,reduction_ms,extraction_ms,support_cliques,compressed_"
        "cliques,"
        "reduced_state_dim\n";
  os << fixed << setprecision(6);
  for (const auto& result : results) {
    os << result.dataset << ',' << result.ordering << ',' << result.family
       << ',' << result.mode << ',' << result.variant << ',' << result.querySize
       << ',' << result.queryIndex << ',' << result.totalMs << ','
       << result.reductionMs << ',' << result.extractionMs << ','
       << result.supportCliques << ',' << result.compressedCliques << ','
       << result.reducedStateDim << '\n';
  }
}

void writeSummaryCsv(const filesystem::path& path,
                     const vector<RawResult>& results) {
  unordered_map<SummaryKey, SummaryValue, SummaryKeyHash> summary;
  for (const auto& result : results) {
    const SummaryKey key{result.dataset, result.ordering, result.family,
                         result.mode,    result.variant,  result.querySize};
    auto& value = summary[key];
    value.totalMs.push_back(result.totalMs);
    value.reductionMs.push_back(result.reductionMs);
    value.extractionMs.push_back(result.extractionMs);
    value.supportCliques = result.supportCliques;
    value.compressedCliques = result.compressedCliques;
    value.reducedStateDim = result.reducedStateDim;
  }

  ofstream os(path);
  os << "dataset,ordering,query_family,mode,variant,query_size,queries,"
        "median_total_ms,total_total_ms,median_reduction_ms,median_extraction_"
        "ms,"
        "support_cliques,compressed_cliques,reduced_state_dim\n";
  os << fixed << setprecision(6);
  for (const auto& [key, value] : summary) {
    const double totalTime =
        accumulate(value.totalMs.begin(), value.totalMs.end(), 0.0);
    os << key.dataset << ',' << key.ordering << ',' << key.family << ','
       << key.mode << ',' << key.variant << ',' << key.querySize << ','
       << value.totalMs.size() << ',' << median(value.totalMs) << ','
       << totalTime << ',' << median(value.reductionMs) << ','
       << median(value.extractionMs) << ',' << value.supportCliques << ','
       << value.compressedCliques << ',' << value.reducedStateDim << '\n';
  }
}

string argumentOrDefault(char** begin, char** end, const string& flag,
                         const string& defaultValue) {
  for (auto it = begin; it != end; ++it) {
    if (string(*it) == flag && it + 1 != end) {
      return *(it + 1);
    }
  }
  return defaultValue;
}

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
        for (const Variant variant :
             {Variant::LegacyDense, Variant::SteinerDense, Variant::LegacySolve,
              Variant::SteinerSolve}) {
          try {
            rawResults.push_back(benchmarkQuery(
                datasetName, orderingName(orderingType), variant, linearGraph,
                bayesTree, ordering, result, query, queryIndex));
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
  writeSummaryCsv(outputDir / "summary.csv", rawResults);
  cout << "Wrote benchmark results to " << outputDir << endl;
  return 0;
}
