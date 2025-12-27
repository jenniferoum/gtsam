/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file MultifrontalClique.h
 * @brief  Imperative multifrontal clique data structure
 * @author Frank Dellaert
 * @date   December 2025
 */

#pragma once

#include <gtsam/base/SymmetricBlockMatrix.h>
#include <gtsam/base/VerticalBlockMatrix.h>
#include <gtsam/dllexport.h>
#include <gtsam/inference/Key.h>
#include <gtsam/linear/GaussianFactor.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/symbolic/SymbolicFactor.h>
#include <gtsam/symbolic/SymbolicJunctionTree.h>

#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gtsam {

namespace internal {

// Helper class to track original factor indices.
class IndexedSymbolicFactor : public SymbolicFactor {
 public:
  size_t index_;
  IndexedSymbolicFactor(const GaussianFactor& factor, size_t index)
      : SymbolicFactor(factor), index_(index) {}
};

}  // namespace internal

/**
 * Imperative multifrontal clique structure used by MultifrontalSolver.
 */
class GTSAM_EXPORT MultifrontalClique {
 public:
  using shared_ptr = std::shared_ptr<MultifrontalClique>;

  explicit MultifrontalClique(const SymbolicJunctionTree::sharedNode& cluster);

  /// @name Setup (non-const)
  /// @{
  void setParent(const std::weak_ptr<MultifrontalClique>& parent);
  void addChild(const shared_ptr& child);
  void assignParentIndicesForChildren();
  void calculateSeparatorKeys();
  void initializeMatrices(const std::vector<size_t>& blockDims, size_t vbmRows);
  void fillAb(const GaussianFactorGraph& graph);
  /// @}

  /// @name Read-only accessors
  /// @{
  const KeyVector& frontals() const;
  const KeyVector& separatorKeys() const;
  const std::vector<shared_ptr>& children() const;
  const std::weak_ptr<MultifrontalClique>& parent() const;
  Key key() const;
  size_t factorCount() const;
  const VerticalBlockMatrix& Ab() const { return Ab_; }
  SymmetricBlockMatrix& sbm() { return sbm_; }
  const SymmetricBlockMatrix& sbm() const { return sbm_; }
  const VerticalBlockMatrix& R_Sd() const { return R_Sd_; }
  const std::vector<size_t>& parentIndices() const { return parentIndices_; }
  /// @}

  /// @name Solve (non-const)
  /// @{
  /// Eliminate this clique and propagate to its parent.
  void eliminateClique();

  /// Solve for the variables in this clique and update the solution vector.
  void solveClique(const std::map<Key, size_t>& dims, VectorValues* x) const;
  /// @}

  // Block dimensions exclude RHS; matrices append it via appendOneDimension.
  std::vector<size_t> blockDims(const std::map<Key, size_t>& dims) const;

  size_t countRows(const GaussianFactorGraph& graph) const;
  std::vector<size_t> parentIndicesFor(const MultifrontalClique& parent) const;
  void print(const std::string& s = "",
             const KeyFormatter& keyFormatter = DefaultKeyFormatter) const;

 private:
  void setParentIndices(const std::vector<size_t>& indices) {
    parentIndices_ = indices;
  }
  Key key_;
  std::weak_ptr<MultifrontalClique> parent_;
  std::vector<shared_ptr> children_;

  VerticalBlockMatrix Ab_;
  SymmetricBlockMatrix sbm_;
  VerticalBlockMatrix R_Sd_;

  SymbolicJunctionTree::sharedNode cluster_;
  KeyVector separatorKeys_;
  std::vector<size_t> parentIndices_;
};

std::ostream& operator<<(std::ostream& os, const MultifrontalClique& clique);

}  // namespace gtsam
