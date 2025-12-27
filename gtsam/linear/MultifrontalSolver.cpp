/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file MultifrontalSolver.cpp
 * @brief
 * @author Frank Dellaert
 * @date   December 2025
 */

#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianFactor.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/MultifrontalClique.h>
#include <gtsam/linear/MultifrontalSolver.h>
#include <gtsam/symbolic/SymbolicEliminationTree.h>
#include <gtsam/symbolic/SymbolicFactorGraph.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <ostream>
#include <set>
#include <stdexcept>
#include <utility>

namespace gtsam {

namespace {

// Compute variable dimensions from the GaussianFactorGraph
std::map<Key, size_t> ComputeDims(const GaussianFactorGraph& graph) {
  std::map<Key, size_t> dims;
  for (const auto& factor : graph) {
    if (!factor) continue;
    if (auto jf = std::dynamic_pointer_cast<JacobianFactor>(factor)) {
      for (auto it = jf->begin(); it != jf->end(); ++it) {
        dims[*it] = jf->getDim(it);
      }
    } else if (auto hf = std::dynamic_pointer_cast<HessianFactor>(factor)) {
      throw std::runtime_error(
          "MultifrontalSolver: HessianFactors not supported.");
    }
  }
  return dims;
}

// Build SymbolicFactorGraph from GaussianFactorGraph
SymbolicFactorGraph BuildSymbolicGraph(const GaussianFactorGraph& graph) {
  SymbolicFactorGraph symbolicGraph;
  symbolicGraph.reserve(graph.size());
  for (size_t i = 0; i < graph.size(); ++i) {
    if (!graph[i]) continue;
    symbolicGraph.emplace_shared<internal::IndexedSymbolicFactor>(*graph[i], i);
  }
  return symbolicGraph;
}

}  // namespace

/* ************************************************************************* */
MultifrontalSolver::MultifrontalSolver(const GaussianFactorGraph& graph,
                                       const Ordering& ordering) {
  // 0. Pre-compute variable dimensions
  dims_ = ComputeDims(graph);

  // 1. Convert to SymbolicFactorGraph to build the elimination tree
  SymbolicFactorGraph symbolicGraph = BuildSymbolicGraph(graph);

  // 2. Build SymbolicEliminationTree and then SymbolicJunctionTree
  SymbolicEliminationTree eliminationTree(symbolicGraph, ordering);
  SymbolicJunctionTree junctionTree(eliminationTree);

  // 3. Recursive function to build Clique hierarchy
  std::function<CliquePtr(const SymbolicJunctionTree::sharedNode&,
                          std::weak_ptr<MultifrontalClique>)>
      buildRecursive =
          [&](const SymbolicJunctionTree::sharedNode& cluster,
              std::weak_ptr<MultifrontalClique> parent) -> CliquePtr {
    if (!cluster) return nullptr;

    // Create Clique
    auto clique = std::make_shared<MultifrontalClique>(cluster);
    auto& c = *clique;
    c.setParent(parent);
    cliques_.push_back(clique);

    // Process children
    for (const auto& childCluster : cluster->children) {
      auto childClique = buildRecursive(childCluster, clique);
      c.addChild(childClique);
    }

    c.calculateSeparatorKeys();

    // Initialize matrices
    std::vector<size_t> blockDims = c.blockDims(dims_);
    size_t vbmRows = c.countRows(graph);
    c.initializeMatrices(blockDims, vbmRows);

    // Initial load
    c.fillAb(graph);

    // Pre-compute parent mapping after separators are finalized.
    c.assignParentIndicesForChildren();

    postOrderCliques_.push_back(clique);
    return clique;
  };

  // 4. Start traversal from roots
  for (const auto& rootCluster : junctionTree.roots()) {
    if (rootCluster) {
      roots_.push_back(
          buildRecursive(rootCluster, std::weak_ptr<MultifrontalClique>()));
    }
  }
}

/* ************************************************************************* */
void MultifrontalSolver::load(const GaussianFactorGraph& graph) {
  for (auto& clique : cliques_) {
    clique->fillAb(graph);
  }
}

/* ************************************************************************* */
void MultifrontalSolver::eliminate() {
  for (auto& clique : postOrderCliques_) {
    clique->eliminateClique();
  }
}

/* ************************************************************************* */
VectorValues MultifrontalSolver::solve() const {
  VectorValues x;
  for (const auto& clique : cliques_) {
    clique->solveClique(dims_, &x);
  }
  return x;
}

/* ************************************************************************* */
std::ostream& operator<<(std::ostream& os, const MultifrontalSolver& solver) {
  os << "MultifrontalSolver(roots=" << solver.roots_.size()
     << ", cliques=" << solver.cliques_.size()
     << ", postOrder=" << solver.postOrderCliques_.size()
     << ", dims=" << solver.dims_.size() << ")\n";

  std::function<void(const MultifrontalSolver::CliquePtr&, int)> dump =
      [&](const MultifrontalSolver::CliquePtr& clique, int depth) {
        if (!clique) return;
        os << std::string(depth * 2, ' ') << *clique << "\n";
        for (const auto& child : clique->children()) {
          dump(child, depth + 1);
        }
      };

  for (const auto& root : solver.roots_) {
    dump(root, 0);
  }
  return os;
}

/* ************************************************************************* */
void MultifrontalSolver::print(const std::string& s,
                               const KeyFormatter& keyFormatter) const {
  if (!s.empty()) std::cout << s;
  std::cout << "MultifrontalSolver matrices (cliques=" << cliques_.size()
            << ")\n";
  for (const auto& clique : cliques_) {
    if (!clique) continue;
    clique->print("", keyFormatter);
  }
}

}  // namespace gtsam
