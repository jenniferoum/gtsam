//
// Created by darshan on 3/11/25.
//

#ifndef EQF_H
#define EQF_H
#pragma once

#include "State.h"
#include "Input.h"
#include "G.h"
#include "Direction.h"
#include "Measurements.h"
#include <gtsam/base/Matrix.h>

using namespace gtsam;

/**
 * Equivariant Filter (EqF) implementation
 */
class EqF {
private:
    int __dof;                 // Degrees of freedom
    int __n_cal;               // Number of calibration states
    int __n_sensor;            // Number of sensors
    G __X_hat;                 // Filter state
    Matrix __Sigma;            // Error covariance
    State __xi_0;              // Origin state
    Matrix __Dphi0;            // Differential of phi at origin
    Matrix __InnovationLift;   // Innovation lift matrix

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

private:
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
};

// Function declarations for helper functions used by EqF

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
Vector3 outputAction(const G& X, const Direction& y, int idx = -1);

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
#endif //EQF_H
