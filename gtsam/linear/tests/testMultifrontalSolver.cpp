/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file testMultifrontalSolver.cpp
 * @brief
 * @author Frank Dellaert
 * @date   December 2025
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianBayesTree.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/MultifrontalClique.h>
#include <gtsam/linear/MultifrontalSolver.h>
#include <tests/smallExample.h>

#include <chrono>
#include <functional>

using namespace std;
using namespace gtsam;
using symbol_shorthand::X;

namespace {
const Key x1 = 1, x2 = 2, x3 = 3, x4 = 4;
const SharedDiagonal chainNoise = noiseModel::Isotropic::Sigma(1, 0.5);
const GaussianFactorGraph chain = {
    std::make_shared<JacobianFactor>(x2, I_1x1, x1, I_1x1,
                                     (Vector(1) << 1.).finished(), chainNoise),
    std::make_shared<JacobianFactor>(x2, I_1x1, x3, I_1x1,
                                     (Vector(1) << 1.).finished(), chainNoise),
    std::make_shared<JacobianFactor>(x3, I_1x1, x4, I_1x1,
                                     (Vector(1) << 1.).finished(), chainNoise),
    std::make_shared<JacobianFactor>(x4, I_1x1, (Vector(1) << 1.).finished(),
                                     chainNoise)};
const Ordering chainOrdering{x2, x1, x3, x4};
}  // namespace

/* ************************************************************************* */
TEST(MultifrontalSolver, Constructor) {
  MultifrontalSolver solver(chain, chainOrdering);

  // Verify roots
  EXPECT(solver.roots().size() == 1);
  auto root = solver.roots()[0];
  EXPECT(root != nullptr);

  // Root should be {x3, x4} (merged)
  // Frontals: x3, x4
  EXPECT_LONGS_EQUAL(2, root->frontals().size());
  EXPECT_LONGS_EQUAL(x3, root->frontals()[0]);
  EXPECT_LONGS_EQUAL(x4, root->frontals()[1]);

  // Root should have 1 child {x2, x1}
  EXPECT_LONGS_EQUAL(1, root->children().size());
  auto c1 = root->children()[0];

  // Verify matrices in leaf (c1)
  EXPECT_LONGS_EQUAL(4, c1->sbm().nBlocks());
  EXPECT_LONGS_EQUAL(2, c1->Ab().rows());
  EXPECT_LONGS_EQUAL(4, c1->Ab().nBlocks());

  // Verify initial load for c1
  // Block 0 (x2):
  Matrix A0 = c1->Ab()(0);  // 2x1
  EXPECT(assert_equal((Matrix(2, 1) << 1., 1.).finished(), A0));

  // Block 3 (RHS):
  Matrix Ab = c1->Ab()(3);  // 2x1
  EXPECT(assert_equal((Matrix(2, 1) << 1., 1.).finished(), Ab));
}

/* ************************************************************************* */
TEST(MultifrontalSolver, Load) {
  MultifrontalSolver solver(chain, chainOrdering);

  // Create a new graph with doubled values
  GaussianFactorGraph chain2;
  for (const auto& factor : chain) {
    auto jf = std::dynamic_pointer_cast<JacobianFactor>(factor);
    std::map<Key, Matrix> terms;
    for (auto it = jf->begin(); it != jf->end(); ++it) {
      terms[*it] = jf->getA(it) * 2.0;
    }
    chain2.push_back(std::make_shared<JacobianFactor>(terms, jf->getb() * 2.0,
                                                      jf->get_model()));
  }

  solver.load(chain2);

  // Verify values in c1
  auto root = solver.roots()[0];
  auto c1 = root->children()[0];

  // Block 0 (x2) should now be 2.0
  Matrix A0 = c1->Ab()(0);
  EXPECT(assert_equal((Matrix(2, 1) << 2., 2.).finished(), A0));
}

/* ************************************************************************* */
TEST(MultifrontalSolver, Eliminate) {
  MultifrontalSolver solver(chain, chainOrdering);
  solver.eliminate();

  // Solve
  VectorValues actual = solver.solve();

  // Reference elimination and solve
  GaussianBayesTree expectedBT = *chain.eliminateMultifrontal(chainOrdering);
  VectorValues expected = expectedBT.optimize();

  EXPECT(assert_equal(expected, actual, 1e-9));
}

/* ************************************************************************* */
TEST(MultifrontalSolver, BalancedSmoother) {
  // Create smoother with 7 nodes
  auto [nlfg, poses] = example::createNonlinearSmoother(7);
  poses.print("Poses:");
  poses.update(X(1), Point2(1.1, 0.2));
  GaussianFactorGraph smoother = *nlfg.linearize(poses);

  // Create the Bayes tree ordering
  const Ordering ordering{X(1), X(3), X(5), X(7), X(2), X(6), X(4)};

  MultifrontalSolver solver(smoother, ordering);
  std::cout << solver << std::endl;
  solver.print();

  // Verify roots
  EXPECT(solver.roots().size() == 1);
  auto root = solver.roots()[0];

  EXPECT_LONGS_EQUAL(root->frontals().size() + root->separatorKeys().size() + 1,
                     root->sbm().nBlocks());

  // Check a leaf clique (for X(1))
  MultifrontalSolver::CliquePtr cX1 = nullptr;
  std::function<void(MultifrontalSolver::CliquePtr)> findX1 =
      [&](MultifrontalSolver::CliquePtr c) {
        for (Key k : c->frontals())
          if (k == X(1)) cX1 = c;
      for (auto child : c->children()) findX1(child);
      };
  findX1(root);

  EXPECT(cX1 != nullptr);
  EXPECT_LONGS_EQUAL(3, cX1->sbm().nBlocks());

  // Eliminate and solve
  solver.eliminate();
  solver.print();
  VectorValues actual = solver.solve();
  GTSAM_PRINT(actual);

  GaussianBayesTree expectedBT = *smoother.eliminateMultifrontal(ordering);
  VectorValues expected = expectedBT.optimize();
  EXPECT(assert_equal(expected, actual, 1e-9));

  // Eliminate and solve after loading new values
  solver.load(smoother);
  solver.print();
  solver.eliminate();
  VectorValues actual2 = solver.solve();
  EXPECT(assert_equal(expected, actual2, 1e-9));
}

/* *************************************************************************
 */
TEST(MultifrontalSolver, Benchmark) {
  const size_t T = 500;
  GaussianFactorGraph smoother = example::createSmoother(T);
  const Ordering ordering = Ordering::Metis(smoother);

  const size_t iterations = 100;

  // Standard GTSAM
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    GaussianBayesTree bt = *smoother.eliminateMultifrontal(ordering);
    VectorValues x = bt.optimize();
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> t_standard = end - start;

  // MultifrontalSolver
  MultifrontalSolver solver(smoother, ordering);
  start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    solver.load(smoother);
    solver.eliminate();
    VectorValues x = solver.solve();
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> t_imperative = end - start;

  std::cout << "\nBenchmark (T=" << T << ", iterations=" << iterations
            << "):\n";
  std::cout << "  Standard GTSAM:     " << t_standard.count() << " s\n";
  std::cout << "  MultifrontalSolver: " << t_imperative.count() << " s\n";
  std::cout << "  Speedup:            "
            << t_standard.count() / t_imperative.count() << "x\n";
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
