/**
 * @file ABC_EQF.h
 * @brief Header file for the Attitude-Bias-Calibration Equivariant Filter
 *
 * This file contains declarations for the Equivariant Filter (EqF) for attitude estimation
 * with both gyroscope bias and sensor extrinsic calibration, based on the paper:
 * "Overcoming Bias: Equivariant Filter Design for Biased Attitude Estimation
 * with Online Calibration" by Fornasier et al.
 * Authors: Darshan Rajasekaran & Jennifer Oum
 */


#ifndef ABC_EQF_H
#define ABC_EQF_H
#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/navigation/ImuBias.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <functional>
#include <chrono>
#include <numeric>  // For std::accumulate

// All implementations are wrapped in this namespace to avoid conflicts
namespace abc_eqf_lib {

using namespace std;
using namespace gtsam;

// Global configuration
// Define coordinate type: "EXPONENTIAL" or "NORMAL"
extern const std::string COORDINATE;

//========================================================================
// Utility Functions
//========================================================================

/**
 * Check if a vector is a unit vector
 */
bool checkNorm(const Vector3& x, double tol = 1e-3);

/**
 * Check if vector contains NaN values
 */
bool hasNaN(const Vector3& vec);

/**
 * Create a block diagonal matrix from two matrices
 */
Matrix blockDiag(const Matrix& A, const Matrix& B);

/**
 * Repeat a block matrix n times along the diagonal
 */
Matrix repBlock(const Matrix& A, int n);

/**
 * Calculate numerical differential
 */
Matrix numericalDifferential(std::function<Vector(const Vector&)> f, const Vector& x);

//========================================================================
// Core Data Types
//========================================================================

/**
 * Direction class as a S2 element
 */
class Direction {
public:
    Unit3 d;  // Direction (unit vector on S2)

    /**
     * Initialize direction
     * @param d_vec Direction vector (must be unit norm)
     */
    Direction(const Vector3& d_vec);

    // Accessor methods for vector components
    double x() const;
    double y() const;
    double z() const;

    // Check if the direction contains NaN values
    bool hasNaN() const;
};

/**
 * Input class for the Biased Attitude System
 */
class Input {
public:
    Vector3 w;               // Angular velocity
    Matrix Sigma;            // Noise covariance

    /**
     * Initialize Input
     * @param w Angular velocity (3-vector)
     * @param Sigma Noise covariance (6x6 matrix)
     */
    Input(const Vector3& w, const Matrix& Sigma);

    /**
     * Return the Input as a skew-symmetric matrix
     * @return w as a skew-symmetric matrix
     */
    Matrix3 W() const;

    /**
     * Return a random angular velocity
     * @return A random angular velocity as Input element
     */
    static Input random();
};

/**
 * Measurement class
 * cal_idx is an index corresponding to the calibration related to the measurement
 * cal_idx = -1 indicates the measurement is from a calibrated sensor
 */
class Measurement {
public:
    Direction y;         // Measurement direction in sensor frame
    Direction d;         // Known direction in global frame
    Matrix3 Sigma;       // Covariance matrix of the measurement
    int cal_idx = -1;    // Calibration index (-1 for calibrated sensor)

    /**
     * Initialize measurement
     * @param y_vec Direction measurement in sensor frame
     * @param d_vec Known direction in global frame
     * @param Sigma Measurement noise covariance
     * @param i Calibration index (-1 for calibrated sensor)
     */
    Measurement(const Vector3& y_vec, const Vector3& d_vec,
                const Matrix3& Sigma, int i = -1);
};

/**
 * State class representing the state of the Biased Attitude System
 */
class State {
public:
    Rot3 R;                  // Attitude rotation matrix R
    Vector3 b;               // Gyroscope bias b
    std::vector<Rot3> S;     // Sensor calibrations S

    State(const Rot3& R = Rot3::Identity(),
          const Vector3& b = Vector3::Zero(),
          const std::vector<Rot3>& S = std::vector<Rot3>());

    static State identity(int n);
};

/**
 * Data structure for ground-truth, input and output data
 */
struct Data {
    State xi;                         // Ground-truth state
    int n_cal;                        // Number of calibration states
    Input u;                          // Input measurements
    std::vector<Measurement> y;       // Output measurements
    int n_meas;                       // Number of measurements
    double t;                         // Time
    double dt;                        // Time step

    /**
     * Initialize Data
     * @param xi Ground-truth state
     * @param n_cal Number of calibration states
     * @param u Input measurements
     * @param y Output measurements
     * @param n_meas Number of measurements
     * @param t Time
     * @param dt Time step
     */
    Data(const State& xi, int n_cal, const Input& u,
         const std::vector<Measurement>& y, int n_meas,
         double t, double dt);
};

//========================================================================
// Symmetry Group
//========================================================================

/**
 * Symmetry group (SO(3) |x so(3)) x SO(3) x ... x SO(3)
 * Each element of the B list is associated with a calibration state
 */
class G {
public:
    Rot3 A;                 // First SO(3) element
    Matrix3 a;              // so(3) element (skew-symmetric matrix)
    std::vector<Rot3> B;    // List of SO(3) elements for calibration

    /**
     * Initialize the symmetry group G
     * @param A SO3 element
     * @param a so(3) element (skew symmetric matrix)
     * @param B list of SO3 elements
     */
    G(const Rot3& A = Rot3::Identity(),
       const Matrix3& a = Matrix3::Zero(),
       const std::vector<Rot3>& B = std::vector<Rot3>());

    /**
     * Define the group operation (multiplication)
     * @param other Another group element
     * @return The product of this and other
     */
    G operator*(const G& other) const;

    /**
     * Return the inverse element of the symmetry group
     * @return The inverse of this group element
     */
    G inv() const;

    /**
     * Return the identity of the symmetry group
     * @param n Number of calibration elements
     * @return The identity element with n calibration components
     */
    static G identity(int n);

    /**
     * Return a group element X given by X = exp(x)
     * @param x Vector representation of Lie algebra element
     * @return Group element given by the exponential of x
     */
    static G exp(const Vector& x);
};

//========================================================================
// Helper Functions for EqF
//========================================================================

/**
 * Compute the lift of the system (Theorem 3.8, Equation 7)
 * @param xi State
 * @param u Input
 * @return Lift vector
 */
Vector lift(const State& xi, const Input& u);

/**
 * Action of the symmetry group on the state space (Equation 4)
 * @param X Group element
 * @param xi State
 * @return New state after group action
 */
State stateAction(const G& X, const State& xi);

/**
 * Action of the symmetry group on the input space (Equation 5)
 * @param X Group element
 * @param u Input
 * @return New input after group action
 */
Input velocityAction(const G& X, const Input& u);

/**
 * Action of the symmetry group on the output space (Equation 6)
 * @param X Group element
 * @param y Direction measurement
 * @param idx Calibration index
 * @return New direction after group action
 */
Vector3 outputAction(const G& X, const Direction& y, int idx);

/**
 * Local coordinates assuming xi_0 = identity (Equation 9)
 * @param e State representing equivariant error
 * @return Local coordinates
 */
Vector local_coords(const State& e);

/**
 * Local coordinates inverse assuming xi_0 = identity
 * @param eps Local coordinates
 * @return Corresponding state
 */
State local_coords_inv(const Vector& eps);

/**
 * Differential of the phi action at E = Id in local coordinates
 * @param xi State
 * @return Differential matrix
 */
Matrix stateActionDiff(const State& xi);

//========================================================================
// Equivariant Filter (EqF)
//========================================================================

/**
 * Equivariant Filter (EqF) implementation
 */
class EqF {
private:
    int __dof;                 // Degrees of freedom
    int __n_cal;               // Number of calibration states
    G __X_hat;                 // Filter state
    Matrix __Sigma;            // Error covariance
    State __xi_0;              // Origin state
    Matrix __Dphi0;            // Differential of phi at origin
    Matrix __InnovationLift;   // Innovation lift matrix

    /**
     * Return the state matrix A0t (Equation 14a)
     * @param u Input
     * @return State matrix A0t
     */
    Matrix __stateMatrixA(const Input& u) const;

    /**
     * Return the state transition matrix Phi (Equation 17)
     * @param u Input
     * @param dt Time step
     * @return State transition matrix Phi
     */
    Matrix __stateTransitionMatrix(const Input& u, double dt) const;

    /**
     * Return the Input matrix Bt
     * @return Input matrix Bt
     */
    Matrix __inputMatrixBt() const;

    /**
     * Return the measurement matrix C0 (Equation 14b)
     * @param d Known direction
     * @param idx Calibration index
     * @return Measurement matrix C0
     */
    Matrix __measurementMatrixC(const Direction& d, int idx) const;

    /**
     * Return the measurement output matrix Dt
     * @param idx Calibration index
     * @return Measurement output matrix Dt
     */
    Matrix __outputMatrixDt(int idx) const;

public:
    /**
     * Initialize EqF
     * @param Sigma Initial covariance
     * @param n Number of calibration states
     * @param m Number of sensors
     */
    EqF(const Matrix& Sigma, int n, int m);

    /**
     * Return estimated state
     * @return Current state estimate
     */
    State stateEstimate() const;

    /**
     * Propagate the filter state
     * @param u Angular velocity measurement
     * @param dt Time step
     */
    void propagation(const Input& u, double dt);

    /**
     * Update the filter state with a measurement
     * @param y Direction measurement
     */
    void update(const Measurement& y);
};

//========================================================================
// Data Processing Functions
//========================================================================

/**
 * Load data from CSV file into a vector of Data objects for the EqF
 *
 * CSV format:
 * - t: Time
 * - q_w, q_x, q_y, q_z: True attitude quaternion
 * - b_x, b_y, b_z: True bias
 * - cq_w_0, cq_x_0, cq_y_0, cq_z_0: True calibration quaternion
 * - w_x, w_y, w_z: Angular velocity measurements
 * - std_w_x, std_w_y, std_w_z: Angular velocity measurement standard deviations
 * - std_b_x, std_b_y, std_b_z: Bias process noise standard deviations
 * - y_x_0, y_y_0, y_z_0, y_x_1, y_y_1, y_z_1: Direction measurements
 * - std_y_x_0, std_y_y_0, std_y_z_0, std_y_x_1, std_y_y_1, std_y_z_1: Direction measurement standard deviations
 * - d_x_0, d_y_0, d_z_0, d_x_1, d_y_1, d_z_1: Reference directions
 *
 * @param filename Path to the CSV file
 * @param startRow First row to load (default: 0)
 * @param maxRows Maximum number of rows to load (default: all)
 * @param downsample Downsample factor (default: 1, which means no downsampling)
 * @return Vector of Data objects
 */
std::vector<Data> loadDataFromCSV(const std::string& filename,
                                  int startRow = 0,
                                  int maxRows = -1,
                                  int downsample = 1);

/**
 * Process data with EqF and print summary results
 * @param filter Initialized EqF filter
 * @param data_list Vector of Data objects to process
 * @param printInterval Progress indicator interval (used internally)
 */
void processDataWithEqF(EqF& filter, const std::vector<Data>& data_list, int printInterval = 10);

} // namespace abc_eqf_lib

#endif // ABC_EQF_H