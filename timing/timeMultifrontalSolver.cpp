/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    timeMultifrontalSolver.cpp
 * @brief   Compare MultifrontalSolver against standard elimination
 * @author  Frank Dellaert
 * @date    December 2025
 */

#include <gtsam/linear/GaussianBayesTree.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/MultifrontalSolver.h>
#include <tests/smallExample.h>

#include <chrono>
#include <iostream>

using namespace std;
using namespace gtsam;
using namespace example;

int main() {
  const size_t T = 500;
  GaussianFactorGraph smoother = createSmoother(T);
  const Ordering ordering = Ordering::Metis(smoother);

  const size_t iterations = 1000;

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    GaussianBayesTree bt = *smoother.eliminateMultifrontal(ordering);
    VectorValues x = bt.optimize();
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> t_standard = end - start;

  MultifrontalSolver solver(smoother, ordering);
  start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    solver.load(smoother);
    solver.eliminate();
    VectorValues x = solver.solve();
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> t_imperative = end - start;

  cout << "\nBenchmark (T=" << T << ", iterations=" << iterations << "):\n";
  cout << "  Standard GTSAM:     " << t_standard.count() << " s\n";
  cout << "  MultifrontalSolver: " << t_imperative.count() << " s\n";
  cout << "  Speedup:            " << t_standard.count() / t_imperative.count()
       << "x\n";

  return 0;
}
