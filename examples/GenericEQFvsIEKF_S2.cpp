/**
 * @file GenericEQFvsIEKF_S2.cpp
 * @brief S² Generic EqF vs IEKF comparison for direction tracking
 *
 * This demonstrates the Generic EQF template specialized for S² direction tracking
 * compared to IEKF. This uses the enhanced GenericEQF.h template with Unit3 
 * specialization instead of the ABC framework.
 * 
 * Scenario: Tracking the direction to a landmark while the sensor rotates
 * with angular velocity and receives noisy direction measurements.
 *
* @date July 30,2025
 * @authors Darshan Rajasekaran, Jennifer Oum, Frank Dellaert
 */

#include "GenericEQF.h"
#include <gtsam/navigation/InvariantEKF.h>
#include <gtsam/slam/dataset.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace gtsam;
using namespace gtsam::generic_eqf_lib;

/// Direction measurement function for Generic EQF (Unit3 to Vector3)
Vector3 h_direction_generic(const Unit3& direction, OptionalJacobian<3, 2> H = {}) {
    Vector3 result = direction.unitVector();
    
    if (H) {
        // Compute Jacobian: derivative of Unit3::unitVector() w.r.t. tangent space (2D)
        Vector3 current = direction.unitVector();
        
        // Create orthogonal basis for tangent space
        Vector3 v1, v2;
        if (abs(current.z()) < 0.9) {
            v1 = Vector3(0, 0, 1).cross(current).normalized();
        } else {
            v1 = Vector3(1, 0, 0).cross(current).normalized();
        }
        v2 = current.cross(v1).normalized();
        
        // Jacobian maps 2D tangent perturbations to 3D direction changes
        *H = Matrix::Zero(3, 2);
        H->col(0) = v1;  // Partial derivative w.r.t. first tangent direction
        H->col(1) = v2;  // Partial derivative w.r.t. second tangent direction
    }
    
    return result;
}

/// Direction measurement function for IEKF (using Rot3 state)
Vector3 h_direction_iekf(const Rot3& R, OptionalJacobian<3, 3> H = {}) {
    // Measure the direction of the z-axis after rotation
    Vector3 z_axis(0, 0, 1);
    return R.rotate(z_axis, H);
}

int main() {
    cout << "================================================================" << endl;
    cout << "Generic S² EqF vs IEKF Comparison for Direction Tracking" << endl;
    cout << "================================================================" << endl;
    cout << "Tracking direction on sphere S² using Generic EQF Template" << endl;

    // Initial direction (pointing north)
    Unit3 initial_direction(Vector3(0, 0, 1));
    Matrix2 P0_generic = Matrix2::Identity() * 0.1;  // 2D covariance for S²
    Matrix3 P0_iekf = Matrix3::Identity() * 0.1;     // 3D covariance for SO(3)

    // Process and measurement noise
    Matrix2 Q_generic = Matrix2::Identity() * 0.05;  // Tangent space noise for S²
    Matrix3 Q_iekf = Matrix3::Identity() * 0.05;     // Angular velocity noise for SO(3)
    Matrix3 R = Matrix3::Identity() * 0.01;          // Direction measurement noise

    double dt = 1.0;  // Time step

    // Angular velocity inputs (rotating around different axes)
    Vector3 omega1(0.1, 0.2, 0.3);  // rad/s
    Vector3 omega2(0.2, -0.1, 0.1); // rad/s
    Vector3 omega3(-0.1, 0.3, -0.2);// rad/s

    // True direction measurements - simulate actual rotation sequence
    // Start with initial direction and apply rotations
    Unit3 true_direction = initial_direction;
    
    // Apply first rotation
    Rot3 R1 = Rot3::Expmap(omega1 * dt);
    Vector3 dir1_rotated = R1.rotate(true_direction.unitVector());
    Unit3 z1_true(dir1_rotated.normalized());
    
    // Apply second rotation  
    Rot3 R2 = Rot3::Expmap(omega2 * dt);
    Vector3 dir2_rotated = R2.rotate(dir1_rotated);
    Unit3 z2_true(dir2_rotated.normalized());
    
    // Apply third rotation
    Rot3 R3 = Rot3::Expmap(omega3 * dt);
    Vector3 dir3_rotated = R3.rotate(dir2_rotated);
    Unit3 z3_true(dir3_rotated.normalized());

    // Angular displacements for Generic EQF (proper S² motion model)
    Vector3 U1_generic = omega1 * dt;  // Direct angular displacement vector
    Vector3 U2_generic = omega2 * dt; 
    Vector3 U3_generic = omega3 * dt;

    // Convert to Rot3 motions for IEKF
    Rot3 U1_iekf = Rot3::Expmap(omega1 * dt);
    Rot3 U2_iekf = Rot3::Expmap(omega2 * dt);
    Rot3 U3_iekf = Rot3::Expmap(omega3 * dt);

    //========================================================================
    // Generic S² EQF
    //========================================================================

    cout << "\n=== GENERIC S² EQUIVARIANT FILTER ===" << endl;

    auto start_generic = chrono::high_resolution_clock::now();
    Unit3 final_generic;
    double generic_time = 0.0;

    try {
        // Create Generic S² EQF
        S2_EQF generic_eqf(initial_direction, P0_generic);

        cout << "\nGeneric S² EqF Initialization:" << endl;
        cout << "Initial direction: " << generic_eqf.state().unitVector().transpose() << endl;

        // First prediction
        generic_eqf.predict(U1_generic, Q_generic);
        cout << "\nGeneric S² EqF First Prediction:" << endl;
        cout << "Direction: " << generic_eqf.state().unitVector().transpose() << endl;

        // First update
        Vector3 z1_vec = z1_true.unitVector();
        generic_eqf.update<3>(h_direction_generic, z1_vec, R);
        cout << "\nGeneric S² EqF First Update:" << endl;
        cout << "Direction: " << generic_eqf.state().unitVector().transpose() << endl;

        // Second prediction
        generic_eqf.predict(U2_generic, Q_generic);
        cout << "\nGeneric S² EqF Second Prediction:" << endl;
        cout << "Direction: " << generic_eqf.state().unitVector().transpose() << endl;

        // Second update
        Vector3 z2_vec = z2_true.unitVector();
        generic_eqf.update<3>(h_direction_generic, z2_vec, R);
        cout << "\nGeneric S² EqF Second Update:" << endl;
        cout << "Direction: " << generic_eqf.state().unitVector().transpose() << endl;

        // Third prediction
        generic_eqf.predict(U3_generic, Q_generic);
        cout << "\nGeneric S² EqF Third Prediction:" << endl;
        cout << "Direction: " << generic_eqf.state().unitVector().transpose() << endl;

        // Third update
        Vector3 z3_vec = z3_true.unitVector();
        generic_eqf.update<3>(h_direction_generic, z3_vec, R);
        cout << "\nGeneric S² EqF Third Update:" << endl;
        cout << "Direction: " << generic_eqf.state().unitVector().transpose() << endl;

        auto end_generic = chrono::high_resolution_clock::now();
        generic_time = chrono::duration<double, milli>(end_generic - start_generic).count();

        final_generic = generic_eqf.state();
        cout << "\nGeneric S² EqF Final Result: " << final_generic.unitVector().transpose() << endl;
        cout << "Generic S² EqF Computation Time: " << generic_time << " ms" << endl;

    } catch (const exception& e) {
        cout << "Error with Generic S² EqF: " << e.what() << endl;
    }

    //========================================================================
    // IEKF for comparison (using Rot3 to represent orientation)
    //========================================================================

    cout << "\n=== INVARIANT EXTENDED KALMAN FILTER (Rot3) ===" << endl;

    auto start_iekf = chrono::high_resolution_clock::now();

    // IEKF with Rot3 state (representing the rotation from initial direction)
    Rot3 R0 = Rot3::Identity();  // Start with identity rotation
    InvariantEKF<Rot3> iekf(R0, P0_iekf);

    cout << "\nIEKF Initialization:" << endl;
    cout << "Initial rotation: " << iekf.state() << endl;
    cout << "Initial direction: " << h_direction_iekf(iekf.state()).transpose() << endl;

    // First prediction
    iekf.predict(U1_iekf, Q_iekf);
    cout << "\nIEKF First Prediction:" << endl;
    cout << "Rotation: " << iekf.state() << endl;
    cout << "Direction: " << h_direction_iekf(iekf.state()).transpose() << endl;

    // First update
    Vector3 z1_vec = z1_true.unitVector();
    iekf.update(h_direction_iekf, z1_vec, R);
    cout << "\nIEKF First Update:" << endl;
    cout << "Rotation: " << iekf.state() << endl;
    cout << "Direction: " << h_direction_iekf(iekf.state()).transpose() << endl;

    // Second prediction
    iekf.predict(U2_iekf, Q_iekf);
    cout << "\nIEKF Second Prediction:" << endl;
    cout << "Rotation: " << iekf.state() << endl;
    cout << "Direction: " << h_direction_iekf(iekf.state()).transpose() << endl;

    // Second update
    Vector3 z2_vec = z2_true.unitVector();
    iekf.update(h_direction_iekf, z2_vec, R);
    cout << "\nIEKF Second Update:" << endl;
    cout << "Rotation: " << iekf.state() << endl;
    cout << "Direction: " << h_direction_iekf(iekf.state()).transpose() << endl;

    // Third prediction
    iekf.predict(U3_iekf, Q_iekf);
    cout << "\nIEKF Third Prediction:" << endl;
    cout << "Rotation: " << iekf.state() << endl;
    cout << "Direction: " << h_direction_iekf(iekf.state()).transpose() << endl;

    // Third update
    Vector3 z3_vec = z3_true.unitVector();
    iekf.update(h_direction_iekf, z3_vec, R);
    cout << "\nIEKF Third Update:" << endl;
    cout << "Rotation: " << iekf.state() << endl;
    cout << "Direction: " << h_direction_iekf(iekf.state()).transpose() << endl;

    auto end_iekf = chrono::high_resolution_clock::now();
    double iekf_time = chrono::duration<double, milli>(end_iekf - start_iekf).count();

    Vector3 final_iekf_dir = h_direction_iekf(iekf.state());
    cout << "\nIEKF Final Result: " << iekf.state() << endl;
    cout << "IEKF Final Direction: " << final_iekf_dir.transpose() << endl;
    cout << "IEKF Computation Time: " << iekf_time << " ms" << endl;

    //========================================================================
    // Comparison and Analysis
    //========================================================================

    cout << "\n=== COMPARISON ANALYSIS ===" << endl;
    
    // Compare final directions
    cout << "\nTrue final direction: " << z3_true.unitVector().transpose() << endl;
    
    // Calculate angular errors
    try {
        double generic_error = acos(std::max(-1.0, std::min(1.0, abs(final_generic.unitVector().dot(z3_true.unitVector())))));
        double iekf_error = acos(std::max(-1.0, std::min(1.0, abs(final_iekf_dir.dot(z3_true.unitVector())))));
        
        cout << "\nAngular Errors:" << endl;
        cout << "Generic S² EqF error: " << (generic_error * 180.0 / M_PI) << " degrees" << endl;
        cout << "IEKF error: " << (iekf_error * 180.0 / M_PI) << " degrees" << endl;
        
        if (generic_error < iekf_error) {
            cout << "Generic S² EqF is more accurate!" << endl;
        } else {
            cout << "IEKF is more accurate!" << endl;
        }
        
    } catch (const exception& e) {
        cout << "Error calculating angular errors: " << e.what() << endl;
    }

    cout << "\nPerformance:" << endl;
    cout << "Generic S² EqF time: " << generic_time << " ms" << endl;
    cout << "IEKF time: " << iekf_time << " ms" << endl;


    return 0;
}