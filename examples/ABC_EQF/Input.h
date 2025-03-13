//
// Created by darshan on 3/11/25.
//

#ifndef INPUT_H
#define INPUT_H

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>

using namespace gtsam;

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

#endif //INPUT_H
