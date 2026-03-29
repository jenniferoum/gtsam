/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file Marginals.cpp
 * @brief
 * @author Richard Roberts
 * @author Frank Dellaert
 * @date May 14, 2012
 */

#include <gtsam/base/timing.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/linear/GaussianBayesNet.h>
#include <gtsam/nonlinear/Marginals.h>

#include <numeric>

using namespace std;

namespace gtsam {

namespace {

/* ************************************************************************* */
// Return a sorted key list with duplicates removed.
KeyVector uniqueSortedKeys(const KeyVector& keys) {
  KeyVector result = keys;
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

/* ************************************************************************* */
// Look up the block dimension of each requested variable.
std::vector<size_t> dimsForKeys(const KeyVector& keys, const Values& values) {
  std::vector<size_t> dims;
  dims.reserve(keys.size());
  for (Key key : keys) {
    dims.push_back(values.at(key).dim());
  }
  return dims;
}

/* ************************************************************************* */
// Convert per-variable dimensions into cumulative block offsets.
std::vector<size_t> blockOffsets(const std::vector<size_t>& dims) {
  std::vector<size_t> offsets(dims.size() + 1, 0);
  for (size_t i = 0; i < dims.size(); ++i) {
    offsets[i + 1] = offsets[i] + dims[i];
  }
  return offsets;
}

/* ************************************************************************* */
// Recover a covariance matrix from a symmetric information matrix.
Matrix informationToCovariance(const Matrix& information) {
  if (!information.allFinite()) {
    return Matrix::Zero(information.rows(), information.cols());
  }

  Eigen::LLT<Matrix> llt(information.selfadjointView<Eigen::Upper>());
  Matrix covariance = Matrix::Identity(information.rows(), information.cols());
  llt.solveInPlace(covariance);
  return covariance;
}

// Build the reduced joint factor graph for the requested variable set.
GaussianFactorGraph reducedJointFactorGraph(
    const GaussianBayesTree& bayesTree, Marginals::Factorization factorization,
    const KeyVector& variables) {
  if (factorization == Marginals::CHOLESKY) {
    return *bayesTree.joint(variables, EliminatePreferCholesky);
  } else if (factorization == Marginals::QR) {
    return *bayesTree.joint(variables, EliminateQR);
  }
  throw std::runtime_error(
      "Marginals::reducedJointFactorGraph: Unknown factorization");
}

/* ************************************************************************* */
// Eliminate a reduced factor graph into a Bayes net ordered by the query keys.
GaussianBayesNet queryBayesNet(const GaussianFactorGraph& factorGraph,
                               Marginals::Factorization factorization,
                               const KeyVector& variables) {
  if (factorization == Marginals::CHOLESKY) {
    return *factorGraph.marginalMultifrontalBayesNet(Ordering(variables),
                                                     EliminatePreferCholesky);
  } else if (factorization == Marginals::QR) {
    return *factorGraph.marginalMultifrontalBayesNet(Ordering(variables),
                                                     EliminateQR);
  }
  throw std::runtime_error("Marginals::queryBayesNet: Unknown factorization");
}

/* ************************************************************************* */
// Recover only the covariance columns associated with selected variable blocks.
Matrix covarianceColumns(const GaussianBayesNet& bayesNet,
                         const KeyVector& orderedKeys,
                         const std::vector<size_t>& dims,
                         const std::vector<size_t>& selectedBlocks) {
  const auto [R, rhs] = bayesNet.matrix(Ordering(orderedKeys));
  (void)rhs;

  const std::vector<size_t> offsets = blockOffsets(dims);
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

  R.transpose().triangularView<Eigen::Lower>().solveInPlace(selectors);
  R.triangularView<Eigen::Upper>().solveInPlace(selectors);
  return selectors;
}

}  // namespace

/* ************************************************************************* */
Marginals::Marginals(const NonlinearFactorGraph& graph, const Values& solution, Factorization factorization)
                     : values_(solution), factorization_(factorization) {
  gttic(MarginalsConstructor);
  graph_ = *graph.linearize(solution);
  computeBayesTree();
}

/* ************************************************************************* */
Marginals::Marginals(const NonlinearFactorGraph& graph, const Values& solution, const Ordering& ordering,
                     Factorization factorization)
                     : values_(solution), factorization_(factorization) {
  gttic(MarginalsConstructor);
  graph_ = *graph.linearize(solution);
  computeBayesTree(ordering);
}

/* ************************************************************************* */
Marginals::Marginals(const GaussianFactorGraph& graph, const Values& solution, Factorization factorization)
                     : graph_(graph), values_(solution), factorization_(factorization) {
  gttic(MarginalsConstructor);
  computeBayesTree();
}

/* ************************************************************************* */
Marginals::Marginals(const GaussianFactorGraph& graph, const Values& solution, const Ordering& ordering,
                     Factorization factorization)
                     : graph_(graph), values_(solution), factorization_(factorization) {
  gttic(MarginalsConstructor);
  computeBayesTree(ordering);
}

/* ************************************************************************* */
Marginals::Marginals(const GaussianFactorGraph& graph, const VectorValues& solution, Factorization factorization)
                     : graph_(graph), factorization_(factorization) {
  gttic(MarginalsConstructor);
  for (const auto& keyValue: solution) {
    values_.insert(keyValue.first, keyValue.second);
  }
  computeBayesTree();
}

/* ************************************************************************* */
Marginals::Marginals(const GaussianFactorGraph& graph, const VectorValues& solution, const Ordering& ordering,
                     Factorization factorization)
                     : graph_(graph), factorization_(factorization) {
  gttic(MarginalsConstructor);
  for (const auto& keyValue: solution) {
    values_.insert(keyValue.first, keyValue.second);
  }
  computeBayesTree(ordering);
}

/* ************************************************************************* */
Marginals::Marginals(GaussianBayesTree&& bayesTree,
                     const VectorValues& solution,
                     Factorization factorization)
    : factorization_(factorization), bayesTree_(std::move(bayesTree)) {
  gttic(MarginalsConstructor);
  for (const auto& keyValue: solution) {
    values_.insert(keyValue.first, keyValue.second);
  }
  bayesTree_.addFactorsToGraph(&graph_);
}

/* ************************************************************************* */
void Marginals::computeBayesTree() {
  // The default ordering to use.
  const Ordering::OrderingType defaultOrderingType = Ordering::COLAMD;
  // Compute BayesTree
  if (factorization_ == CHOLESKY)
    bayesTree_ = *graph_.eliminateMultifrontal(defaultOrderingType,
                                               EliminatePreferCholesky);
  else if (factorization_ == QR)
    bayesTree_ =
        *graph_.eliminateMultifrontal(defaultOrderingType, EliminateQR);
}

/* ************************************************************************* */
void Marginals::computeBayesTree(const Ordering& ordering) {
  // Compute BayesTree
  if(factorization_ == CHOLESKY)
    bayesTree_ = *graph_.eliminateMultifrontal(ordering, EliminatePreferCholesky);
  else if(factorization_ == QR)
    bayesTree_ = *graph_.eliminateMultifrontal(ordering, EliminateQR);
}

/* ************************************************************************* */
void Marginals::print(const std::string& str, const KeyFormatter& keyFormatter) const
{
  graph_.print(str+"Graph: ");
  values_.print(str+"Solution: ", keyFormatter);
  bayesTree_.print(str+"Bayes Tree: ");
}

/* ************************************************************************* */
GaussianFactor::shared_ptr Marginals::marginalFactor(Key variable) const {
  gttic(marginalFactor);

  // Compute marginal factor
  if(factorization_ == CHOLESKY)
    return bayesTree_.marginalFactor(variable, EliminatePreferCholesky);
  else if(factorization_ == QR)
    return bayesTree_.marginalFactor(variable, EliminateQR);
  else
    throw std::runtime_error("Marginals::marginalFactor: Unknown factorization");
}

/* ************************************************************************* */
Matrix Marginals::marginalInformation(Key variable) const {

  // Get information matrix (only store upper-right triangle)
  gttic(marginalInformation);
  return marginalFactor(variable)->information();
}

/* ************************************************************************* */
Matrix Marginals::marginalCovariance(Key variable) const {
  Matrix info = marginalInformation(variable);
  return informationToCovariance(info);
}

/* ************************************************************************* */
JointMarginal Marginals::jointMarginalCovariance(
    const KeyVector& variables) const {
  // Normalize the query key set so block order is deterministic.
  const KeyVector variablesSorted = uniqueSortedKeys(variables);
  if (variablesSorted.size() == 1) {
    Matrix covariance = marginalCovariance(variablesSorted.front());
    return JointMarginal(covariance, {static_cast<size_t>(covariance.rows())},
                         variablesSorted);
  }

  // Reduce the Bayes tree query to a Bayes net on just the requested variables.
  const GaussianFactorGraph jointFG =
      reducedJointFactorGraph(bayesTree_, factorization_, variablesSorted);
  const GaussianBayesNet bayesNet =
      queryBayesNet(jointFG, factorization_, variablesSorted);
  const std::vector<size_t> dims = dimsForKeys(variablesSorted, values_);

  // Recover every covariance block by selecting every column block.
  std::vector<size_t> allBlocks(variablesSorted.size());
  std::iota(allBlocks.begin(), allBlocks.end(), 0);
  Matrix covariance =
      covarianceColumns(bayesNet, variablesSorted, dims, allBlocks);
  return JointMarginal(covariance, dims, variablesSorted);
}

/* ************************************************************************* */
JointMarginal Marginals::jointMarginalInformation(
    const KeyVector& variables) const {
  // Normalize the query key set so the returned block layout is stable.
  const KeyVector variablesSorted = uniqueSortedKeys(variables);
  if (variablesSorted.empty()) {
    return JointMarginal(Matrix(), std::vector<size_t>(), variablesSorted);
  }

  // A single-variable query can reuse the existing marginal-factor path.
  if (variablesSorted.size() == 1) {
    Matrix info = marginalInformation(variablesSorted.front());
    std::vector<size_t> dims;
    dims.push_back(info.rows());
    return JointMarginal(info, dims, variablesSorted);
  } else {
    // Reduce the Bayes tree query and then materialize the reduced Hessian.
    GaussianFactorGraph jointFG =
        reducedJointFactorGraph(bayesTree_, factorization_, variablesSorted);
    GaussianBayesNet bayesNet =
        queryBayesNet(jointFG, factorization_, variablesSorted);

    // Get information matrix on the reduced query system.
    const auto [R, rhs] = bayesNet.matrix(Ordering(variablesSorted));
    (void)rhs;
    Matrix info = R.transpose() * R;

    // Record block dimensions in the same key order used for the Hessian.
    std::vector<size_t> dims;
    dims.reserve(variablesSorted.size());
    for (const auto& key : variablesSorted) {
      dims.push_back(values_.at(key).dim());
    }

    return JointMarginal(info, dims, variablesSorted);
  }
}

/* ************************************************************************* */
VectorValues Marginals::optimize() const {
  return bayesTree_.optimize();
}

/* ************************************************************************* */
void Marginals::deleteCachedShortcuts() {
  bayesTree_.deleteCachedShortcuts();
}

/* ************************************************************************* */
void JointMarginal::print(const std::string& s, const KeyFormatter& formatter) const {
  cout << s << "Joint marginal on keys ";
  bool first = true;
  for(const auto& key: keys_) {
    if(!first)
      cout << ", ";
    else
      first = false;
    cout << formatter(key);
  }
  cout << ".  Use 'at' or 'operator()' to query matrix blocks." << endl;
}

} /* namespace gtsam */
