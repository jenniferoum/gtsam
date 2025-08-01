/**
 * @file GenericEQFvsIEKF_SE2_Corrected.cpp
 * @brief SE(2) Generic EqF vs IEKF comparison for pose tracking
 *
 * This demonstrates the Generic EQF template specialized for SE(2) pose tracking
 * compared to IEKF. This uses the enhanced GenericEQF.h template with Pose2 
 * specialization.
 * 
 * Scenario: Robot moving in 2D with GPS position measurements
 *
 * @date July 21,2025
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

/// GPS measurement function for Generic EQF (Pose2 to Vector2)
Vector2 h_gps_generic(const Pose2& pose, OptionalJacobian<2, 3> H = {}) {
    Vector2 result = pose.translation();
    
    if (H) {
        // Jacobian of translation w.r.t. SE(2) tangent space [dx, dy, dtheta]
        *H = Matrix::Zero(2, 3);
        (*H)(0, 0) = 1.0;  // d(x)/d(dx) = 1
        (*H)(1, 1) = 1.0;  // d(y)/d(dy) = 1
        // d(x)/d(dtheta) = 0, d(y)/d(dtheta) = 0
    }
    
    return result;
}

/// GPS measurement function for IEKF (using Pose2 state)
Vector2 h_gps_iekf(const Pose2& pose, OptionalJacobian<2, 3> H = {}) {
    return pose.translation(H);
}

int main() {
    cout << "Generic SE(2) EqF vs IEKF Comparison for Pose Tracking" << endl;



    // Initial pose (at origin, facing east)
    Pose2 initial_pose = Pose2::Identity();
    Matrix3 P0 = Matrix3::Identity() * 0.1;  // 3x3 covariance for SE(2)

    // Process and measurement noise
    Matrix3 Q = Matrix3::Identity() * 0.05;  // Motion noise
    Matrix2 R = Matrix2::Identity() * 0.01;  // GPS measurement noise

    double dt = 1.0;  // Time step

    // SE(2) motion inputs (dx, dy, dtheta) - robot trajectory
    Vector3 u1(1.0, 0.5, 0.2);   // Move forward-right, turn slightly
    Vector3 u2(0.8, -0.3, -0.1); // Move forward-left, turn back slightly  
    Vector3 u3(1.2, 0.8, 0.3);   // Move forward-right, turn more

    // True pose trajectory - simulate actual robot motion
    Pose2 true_pose = initial_pose;
    
    // Apply first motion
    Pose2 U1 = Pose2::Expmap(u1 * dt);
    true_pose = true_pose.compose(U1);
    Point2 gps1_true = true_pose.translation();
    
    // Apply second motion
    Pose2 U2 = Pose2::Expmap(u2 * dt);
    true_pose = true_pose.compose(U2);
    Point2 gps2_true = true_pose.translation();
    
    // Apply third motion
    Pose2 U3 = Pose2::Expmap(u3 * dt);
    true_pose = true_pose.compose(U3);
    Point2 gps3_true = true_pose.translation();

    // Motion commands for filters
    Vector3 U1_generic = u1 * dt;  // Direct SE(2) motion vector
    Vector3 U2_generic = u2 * dt;
    Vector3 U3_generic = u3 * dt;

    // Convert to Pose2 motions for IEKF
    Pose2 U1_iekf = Pose2::Expmap(u1 * dt);
    Pose2 U2_iekf = Pose2::Expmap(u2 * dt);
    Pose2 U3_iekf = Pose2::Expmap(u3 * dt);

    //========================================================================
    // Generic SE(2) EQF
    //========================================================================

    cout << "\n=== GENERIC SE(2) EQUIVARIANT FILTER ===" << endl;

    auto start_generic = chrono::high_resolution_clock::now();
    Pose2 final_generic;
    double generic_time = 0.0;

    try {
        // Create Generic SE(2) EQF
        SE2_EQF generic_eqf(initial_pose, P0);

        cout << "\nGeneric SE(2) EqF Initialization:" << endl;
        cout << "Initial pose: " << generic_eqf.state() << endl;

        // First prediction
        generic_eqf.predict(U1_generic, Q);
        cout << "\nGeneric SE(2) EqF First Prediction:" << endl;
        cout << "Pose: " << generic_eqf.state() << endl;

        // First GPS update
        Vector2 gps1_vec(gps1_true.x(), gps1_true.y());
        generic_eqf.update<2>(h_gps_generic, gps1_vec, R);
        cout << "\nGeneric SE(2) EqF First Update:" << endl;
        cout << "Pose: " << generic_eqf.state() << endl;

        // Second prediction
        generic_eqf.predict(U2_generic, Q);
        cout << "\nGeneric SE(2) EqF Second Prediction:" << endl;
        cout << "Pose: " << generic_eqf.state() << endl;

        // Second GPS update
        Vector2 gps2_vec(gps2_true.x(), gps2_true.y());
        generic_eqf.update<2>(h_gps_generic, gps2_vec, R);
        cout << "\nGeneric SE(2) EqF Second Update:" << endl;
        cout << "Pose: " << generic_eqf.state() << endl;

        // Third prediction
        generic_eqf.predict(U3_generic, Q);
        cout << "\nGeneric SE(2) EqF Third Prediction:" << endl;
        cout << "Pose: " << generic_eqf.state() << endl;

        // Third GPS update
        Vector2 gps3_vec(gps3_true.x(), gps3_true.y());
        generic_eqf.update<2>(h_gps_generic, gps3_vec, R);
        cout << "\nGeneric SE(2) EqF Third Update:" << endl;
        cout << "Pose: " << generic_eqf.state() << endl;

        auto end_generic = chrono::high_resolution_clock::now();
        generic_time = chrono::duration<double, milli>(end_generic - start_generic).count();

        final_generic = generic_eqf.state();
        cout << "\nGeneric SE(2) EqF Final Result: " << final_generic << endl;
        cout << "Generic SE(2) EqF Computation Time: " << generic_time << " ms" << endl;

    } catch (const exception& e) {
        cout << "Error with Generic SE(2) EqF: " << e.what() << endl;
    }

    //========================================================================
    // IEKF for comparison (using Pose2 state)
    //========================================================================

    cout << "\n=== INVARIANT EXTENDED KALMAN FILTER (Pose2) ===" << endl;

    auto start_iekf = chrono::high_resolution_clock::now();

    // IEKF with Pose2 state
    InvariantEKF<Pose2> iekf(initial_pose, P0);

    cout << "\nIEKF Initialization:" << endl;
    cout << "Initial pose: " << iekf.state() << endl;

    // First prediction
    iekf.predict(U1_iekf, Q);
    cout << "\nIEKF First Prediction:" << endl;
    cout << "Pose: " << iekf.state() << endl;

    // First GPS update
    Vector2 gps1_vec(gps1_true.x(), gps1_true.y());
    iekf.update(h_gps_iekf, gps1_vec, R);
    cout << "\nIEKF First Update:" << endl;
    cout << "Pose: " << iekf.state() << endl;

    // Second prediction
    iekf.predict(U2_iekf, Q);
    cout << "\nIEKF Second Prediction:" << endl;
    cout << "Pose: " << iekf.state() << endl;

    // Second GPS update
    Vector2 gps2_vec(gps2_true.x(), gps2_true.y());
    iekf.update(h_gps_iekf, gps2_vec, R);
    cout << "\nIEKF Second Update:" << endl;
    cout << "Pose: " << iekf.state() << endl;

    // Third prediction
    iekf.predict(U3_iekf, Q);
    cout << "\nIEKF Third Prediction:" << endl;
    cout << "Pose: " << iekf.state() << endl;

    // Third GPS update
    Vector2 gps3_vec(gps3_true.x(), gps3_true.y());
    iekf.update(h_gps_iekf, gps3_vec, R);
    cout << "\nIEKF Third Update:" << endl;
    cout << "Pose: " << iekf.state() << endl;

    auto end_iekf = chrono::high_resolution_clock::now();
    double iekf_time = chrono::duration<double, milli>(end_iekf - start_iekf).count();

    Pose2 final_iekf = iekf.state();
    cout << "\nIEKF Final Result: " << final_iekf << endl;
    cout << "IEKF Computation Time: " << iekf_time << " ms" << endl;

    //========================================================================
    // Comparison and Analysis
    //========================================================================

    cout << "\n=== COMPARISON ANALYSIS ===" << endl;
    
    // Compare final poses
    cout << "\nTrue final pose: " << true_pose << endl;
    
    // Calculate position and orientation errors
    try {
        Vector2 generic_pos = final_generic.translation();
        Vector2 iekf_pos = final_iekf.translation();
        Vector2 true_pos = true_pose.translation();
        
        double generic_pos_error = (generic_pos - true_pos).norm();
        double iekf_pos_error = (iekf_pos - true_pos).norm();
        
        double generic_angle_error = abs(final_generic.theta() - true_pose.theta()) * 180.0 / M_PI;
        double iekf_angle_error = abs(final_iekf.theta() - true_pose.theta()) * 180.0 / M_PI;
        
        cout << "\nPosition Errors:" << endl;
        cout << "Generic SE(2) EqF error: " << generic_pos_error << " meters" << endl;
        cout << "IEKF error: " << iekf_pos_error << " meters" << endl;
        
        cout << "\nOrientation Errors:" << endl;
        cout << "Generic SE(2) EqF error: " << generic_angle_error << " degrees" << endl;
        cout << "IEKF error: " << iekf_angle_error << " degrees" << endl;
        
        if (generic_pos_error < iekf_pos_error && generic_angle_error < iekf_angle_error) {
            cout << "Generic SE(2) EqF is more accurate!" << endl;
        } else if (iekf_pos_error < generic_pos_error && iekf_angle_error < generic_angle_error) {
            cout << "IEKF is more accurate!" << endl;
        } else {
            cout << "Mixed results - both filters have comparable accuracy!" << endl;
        }
        
    } catch (const exception& e) {
        cout << "Error calculating errors: " << e.what() << endl;
    }

    cout << "\nPerformance:" << endl;
    cout << "Generic SE(2) EqF time: " << generic_time << " ms" << endl;
    cout << "IEKF time: " << iekf_time << " ms" << endl;

    return 0;
}