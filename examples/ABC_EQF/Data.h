//
// Created by darshan on 3/11/25.
//

#ifndef DATA_H
#define DATA_H
//#pragma once

#include "State.h"
#include "Input.h"
#include "Measurements.h"
#include <vector>

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
         double t, double dt)
        : xi(xi), n_cal(n_cal), u(u), y(y), n_meas(n_meas), t(t), dt(dt) {}
};
#endif //DATA_H
