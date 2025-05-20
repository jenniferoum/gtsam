/**
 * @file ABC_Example.h
 * @date March, 2025
 * @author Jennifer & Darshan
 * @brief Example ABC Equivariant Filter
 */

#include "ABC_Example.h"
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

using namespace gtsam;

using G = gtsam::Rot3;
using V = gtsam::Vector3;
using MAT = gtsam::Matrix3;

using M2 = gtsam::abc::Manifold2<G, V, 1>;
using G2 = gtsam::abc::Group2<G, MAT, 1>;

// alias M and G
using M = abc::Manifold<1>;
using Gr = abc::Group<1>;

// Check whether they satisfy the concept
GTSAM_CONCEPT_MANIFOLD_INST(M)
GTSAM_CONCEPT_LIE_INST(G)

// M stateAction2(const G& X, const M& xi) {
//   std::cout << "1 About to return new state from stateAction2" << std::endl;
//   if (xi.S.size() != X.B.size()) {
//     throw std::invalid_argument("Number of calibration states and B elements must match");
//   }
//   std::cout << "2 About to return new state from stateAction2" << std::endl;
//   std::vector<Rot3> new_S;
//   for (size_t i = 0; i < X.B.size(); i++) {
//     new_S.push_back(X.A.inverse() * xi.S[i] * X.B[i]);
//   }
//   std::array<Rot3, 1> new_S_array = { new_S[0] };
//   std::cout << "3 About to return new state from stateAction2" << std::endl;
//   return M(xi.R * X.A,
//               X.A.inverse() * (xi.b - Rot3::Vee(X.a)),
//               new_S_array);
// }

int main(int argc, char* argv[]) {
    M2 xi2(Rot3(), Vector3(1, 2, 3), { Rot3()/*, Rot3()*/ });
    GTSAM_PRINT(xi2);
    M xi(Rot3(), Vector3(1, 2, 3), { Rot3()/*, Rot3()*/ });
    GTSAM_PRINT(xi);
    Gr X(Rot3(), Rot3::Hat(Vector3(4, 5, 6)), { Rot3()/*, Rot3()*/ });
    GTSAM_PRINT(X);

    G2 X2(Rot3(), Rot3::Hat(Vector3(4, 5, 6)), { Rot3()/*, Rot3()*/ });
    GTSAM_PRINT(X2);

    GTSAM_PRINT(xi.R * X.A);
    GTSAM_PRINT(xi2.R * X2.A);

    GTSAM_PRINT(X.A.inverse());
    GTSAM_PRINT(X2.A.inverse());
    // Rot3 R1 = Rot3::RzRyRx(0.1, 0.2, 0.3);
    // Rot3 R2 = Rot3::RzRyRx(0.4, 0.5, 0.6);
    // Vector3 t1(1, 2, 3);
    // Vector3 t2(4, 5, 6);
    //
    // Pose3 SE3_1 = Pose3(R1, t1);
    // Pose3 SE3_2 = Pose3(R2, t2);
    //
    // Pose3 SE3_mult = SE3_1.compose(SE3_2);
    //
    // // SO(3) x R³
    // Rot3 R_mult = R1 * R2;
    // Vector3 t_mult_SO3xR3 = t1 + t2;
    //
    // // Convert to Pose3 for comparison
    // Pose3 SO3xR3_mult(R_mult, t_mult_SO3xR3);
    //
    // // Print the difference in translation
    // Vector3 translation_error = SO3xR3_mult.translation() - SE3_mult.translation();
    // std::cout << "Translation difference: " << translation_error.transpose() << std::endl;
    //
    // if (translation_error.norm() > 1e-6) {
    //     std::cout << "Test passed: SO(3) × R³ is NOT equivalent to SE(3)!" << std::endl;
    // } else {
    //     std::cout << "Test failed: They appear to be the same (unexpected)." << std::endl;
    // }

    return 0;
}
