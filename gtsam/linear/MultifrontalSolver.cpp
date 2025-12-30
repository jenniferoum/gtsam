/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file MultifrontalSolver.cpp
 * @brief Implementation of imperative-style multifrontal solver.
 * @author Frank Dellaert
 * @date   December 2025
 */

#include <gtsam/inference/Key.h>
#include <gtsam/linear/GaussianFactor.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/MultifrontalClique.h>
#include <gtsam/linear/MultifrontalSolver.h>
#include <gtsam/base/types.h>
#include <gtsam/symbolic/SymbolicEliminationTree.h>
#include <gtsam/symbolic/SymbolicFactorGraph.h>
#include <gtsam/base/treeTraversal-inst.h>

#include <functional>
#include <iostream>
#include <ostream>
#include <stdexcept>

namespace gtsam {

namespace {

// Compute variable dimensions from the GaussianFactorGraph
std::map<Key, size_t> computeDims(const GaussianFactorGraph& graph) {
  std::map<Key, size_t> dims;
  for (const auto& factor : graph) {
    if (!factor) continue;
    if (auto jacobianFactor =
            std::dynamic_pointer_cast<JacobianFactor>(factor)) {
      for (auto it = jacobianFactor->begin(); it != jacobianFactor->end();
           ++it) {
        dims[*it] = jacobianFactor->getDim(it);
      }
    } else if (auto hessianFactor =
                   std::dynamic_pointer_cast<HessianFactor>(factor)) {
      throw std::runtime_error(
          "MultifrontalSolver: HessianFactors not supported.");
    }
  }
  return dims;
}

// Build SymbolicFactorGraph from GaussianFactorGraph
SymbolicFactorGraph buildSymbolicGraph(const GaussianFactorGraph& graph) {
  SymbolicFactorGraph symbolicGraph;
  symbolicGraph.reserve(graph.size());
  for (size_t i = 0; i < graph.size(); ++i) {
    if (!graph[i]) continue;
    symbolicGraph.emplace_shared<internal::IndexedSymbolicFactor>(*graph[i], i);
  }
  return symbolicGraph;
}

size_t frontalDimForCluster(const SymbolicJunctionTree::sharedNode& cluster,
                            const std::map<Key, size_t>& dims) {
  size_t dim = 0;
  for (Key key : cluster->orderedFrontalKeys) {
    auto it = dims.find(key);
    if (it != dims.end()) dim += it->second;
  }
  return dim;
}

void mergeSmallClusters(const SymbolicJunctionTree::sharedNode& cluster,
                        const std::map<Key, size_t>& dims,
                        size_t mergeFrontalsBelow) {
  if (!cluster) return;
  for (const auto& child : cluster->children) {
    mergeSmallClusters(child, dims, mergeFrontalsBelow);
  }
  if (cluster->children.empty()) return;

  std::vector<bool> merge(cluster->children.size(), false);
  bool any = false;
  for (size_t i = 0; i < cluster->children.size(); ++i) {
    const auto& child = cluster->children[i];
    if (!child) continue;
    if (frontalDimForCluster(child, dims) < mergeFrontalsBelow) {
      merge[i] = true;
      any = true;
    }
  }
  if (any) {
    cluster->mergeChildren(merge);
  }
}

}  // namespace

/* ************************************************************************* */
MultifrontalSolver::MultifrontalSolver(const GaussianFactorGraph& graph,
                                       const Ordering& ordering,
                                       size_t mergeFrontalsBelow) {
  // 0. Pre-compute variable dimensions
  dims_ = computeDims(graph);
  for (Key key : ordering) {
    solution_.insert(key, Vector::Zero(dims_.at(key)));
  }

  // 1. Convert to SymbolicFactorGraph to build the elimination tree
  SymbolicFactorGraph symbolicGraph = buildSymbolicGraph(graph);

  // 2. Build SymbolicEliminationTree and then SymbolicJunctionTree
  SymbolicEliminationTree eliminationTree(symbolicGraph, ordering);
  SymbolicJunctionTree junctionTree(eliminationTree);

  if (mergeFrontalsBelow > 0) {
    for (const auto& rootCluster : junctionTree.roots()) {
      mergeSmallClusters(rootCluster, dims_, mergeFrontalsBelow);
    }
  }

  // 3. Recursive function to build Clique hierarchy (independent of traversal).
  std::function<CliquePtr(const SymbolicJunctionTree::sharedNode&,
                          std::weak_ptr<MultifrontalClique>)>
      buildRecursive =
          [&](const SymbolicJunctionTree::sharedNode& cluster,
              std::weak_ptr<MultifrontalClique> parent) -> CliquePtr {
    if (!cluster) return nullptr;

    // Create Clique
    auto clique = std::make_shared<MultifrontalClique>(cluster);
    clique->setParent(parent);
    cliques_.push_back(clique);

    // Process children
    for (const auto& childCluster : cluster->children) {
      auto childClique = buildRecursive(childCluster, clique);
      clique->addChild(childClique);
    }

    clique->calculateSeparatorKeys();
    clique->cacheValuePointers(&solution_);

    // Initialize matrices
    std::vector<size_t> blockDims = clique->blockDims(dims_);
    size_t vbmRows = clique->countRows(graph);
    clique->initializeMatrices(blockDims, vbmRows);

    // Initial load
    clique->fillAb(graph);

    // Pre-compute parent mapping after separators are finalized.
    clique->assignParentIndicesForChildren();

    return clique;
  };

  // 4. Start traversal from roots
  for (const auto& rootCluster : junctionTree.roots()) {
    if (rootCluster) {
      roots_.push_back(
          buildRecursive(rootCluster, std::weak_ptr<MultifrontalClique>()));
    }
  }

  // Build a lightweight traversal forest for DepthFirstForestParallel.
  traversalRoots_.clear();
  traversalRoots_.reserve(roots_.size());
  for (const auto& root : roots_) {
    auto node = buildTraversalNode(root, dims_);
    if (node) traversalRoots_.push_back(node);
  }
}

size_t MultifrontalSolver::frontalDimForClique(
    const CliquePtr& clique, const std::map<Key, size_t>& dims) {
  size_t dim = 0;
  for (Key key : clique->frontals()) {
    auto it = dims.find(key);
    if (it != dims.end()) dim += it->second;
  }
  return dim;
}

std::shared_ptr<MultifrontalSolver::CliqueTraversalNode>
MultifrontalSolver::buildTraversalNode(
    const CliquePtr& clique, const std::map<Key, size_t>& dims) {
  if (!clique) return nullptr;
  auto node = std::make_shared<CliqueTraversalNode>();
  node->clique = clique;
  // Use frontal dimension as the traversal "problem size" for scheduling cutoff.
  node->problemSizeValue =
      static_cast<int>(frontalDimForClique(clique, dims));
  node->children.reserve(clique->children().size());
  for (const auto& child : clique->children()) {
    auto childNode = buildTraversalNode(child, dims);
    if (childNode) node->children.push_back(childNode);
  }
  return node;
}

/* ************************************************************************* */
void MultifrontalSolver::load(const GaussianFactorGraph& graph) {
  for (auto& clique : cliques_) {
    clique->fillAb(graph);
  }
}

/* ************************************************************************* */
void MultifrontalSolver::eliminate() {
  // Parallel elimination uses the same traversal as legacy GTSAM (TBB optional).
  struct EliminateTraversalData {};
  struct EliminatePreVisitor {
    EliminateTraversalData operator()(
        const std::shared_ptr<CliqueTraversalNode>&,
        const EliminateTraversalData&) const {
      return EliminateTraversalData();
    }
  };
  struct EliminatePostVisitor {
    void operator()(const std::shared_ptr<CliqueTraversalNode>& node,
                    EliminateTraversalData&) const {
      if (node && node->clique) node->clique->eliminate();
    }
  };
  struct CliqueForestView {
    using Node = CliqueTraversalNode;
    explicit CliqueForestView(
        const std::vector<std::shared_ptr<CliqueTraversalNode>>& roots)
        : roots_(roots) {}
    const std::vector<std::shared_ptr<CliqueTraversalNode>>& roots() const {
      return roots_;
    }
    const std::vector<std::shared_ptr<CliqueTraversalNode>>& roots_;
  };

  CliqueForestView forest(traversalRoots_);
  EliminateTraversalData rootData;
  EliminatePreVisitor visitorPre;
  EliminatePostVisitor visitorPost;
  TbbOpenMPMixedScope threadLimiter;
  treeTraversal::DepthFirstForestParallel(
      forest, rootData, visitorPre, visitorPost, 10);
}

/* ************************************************************************* */
const VectorValues& MultifrontalSolver::solve() const {
  for (const auto& clique : cliques_) {
    clique->solve();
  }
  return solution_;
}

/* ************************************************************************* */
std::ostream& operator<<(std::ostream& os, const MultifrontalSolver& solver) {
  os << "MultifrontalSolver(roots=" << solver.roots_.size()
     << ", cliques=" << solver.cliques_.size()
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
