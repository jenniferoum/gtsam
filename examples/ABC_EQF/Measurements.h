//
// Created by darshan on 3/11/25.
//

#ifndef MEASUREMENTS_H
#define MEASUREMENTS_H


#include "Direction.h"
#include <gtsam/base/Matrix.h>

using namespace gtsam;

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
#endif //MEASUREMENTS_H
