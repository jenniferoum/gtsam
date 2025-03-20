//
// Created by darshan on 3/11/25.
//
#include "EqF.h"
#include "State.h"
#include "Input.h"
#include "Direction.h"
#include "Measurements.h"
#include "Data.h"
#include "runEQF_withcsv.h"
#include "utilities.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>

using namespace std;
using namespace gtsam;

// Simplified data loading function - in a real application, implement proper CSV parsing
std::vector<Data> loadSimulatedData() {
    std::vector<Data> data_list;

    double t = 0.0;
    double dt = 0.01;

    // Number of data points
    int num_points = 100;

    // Set up one calibration state
    int n_cal = 1;

    for (int i = 0; i < num_points; i++) {
        t += dt;

        // Create a simple sinusoidal trajectory
        double angle = 0.1 * sin(t);
        Rot3 R = Rot3::Rz(angle);

        // Create a bias
        Vector3 b(0.01, 0.02, 0.03);

        // Create a calibration
        std::vector<Rot3> S;
        S.push_back(Rot3::Ry(0.05));

        // State
        State xi(R, b, S);

        // Input (angular velocity)
        Vector3 w(0.1 * cos(t), 0.05 * sin(t), 0.02);
        Matrix Sigma_u = Matrix::Identity(6, 6) * 0.01;
        Input u(w, Sigma_u);

        // Measurements
        std::vector<Measurement> measurements;

        // Measurement 1 - from uncalibrated sensor
        Vector3 d1_vec = Vector3(1, 0, 0).normalized(); // Known direction in global frame
        Vector3 y1_vec = S[0].inverse().matrix() * R.inverse().matrix() * d1_vec; // Direction in sensor frame
        Matrix3 Sigma1 = Matrix3::Identity() * 0.01;
        measurements.push_back(Measurement(y1_vec, d1_vec, Sigma1, 0)); // cal_idx = 0

        // Measurement 2 - from calibrated sensor
        Vector3 d2_vec = Vector3(0, 1, 0).normalized(); // Known direction in global frame
        Vector3 y2_vec = R.inverse().matrix() * d2_vec; // Direction in sensor frame
        Matrix3 Sigma2 = Matrix3::Identity() * 0.01;
        measurements.push_back(Measurement(y2_vec, d2_vec, Sigma2, -1)); // cal_idx = -1

        // Add to data list
        data_list.push_back(Data(xi, n_cal, u, measurements, 2, t, dt));
    }

    return data_list;
}

void runSimulation(EqF& filter, const std::vector<Data>& data) {
    std::cout << "Starting simulation with " << data.size() << " data points..." << std::endl;

    // Track time for performance measurement
    auto start = std::chrono::high_resolution_clock::now();

    // Store results for analysis
    std::vector<double> times;
    std::vector<Vector3> attitude_errors;
    std::vector<Vector3> bias_errors;
    std::vector<Vector3> calibration_errors;

    for (const auto& d : data) {
        // Propagation
        try {
            filter.propagation(d.u, d.dt);
        } catch (const std::exception& e) {
            std::cerr << "Propagation error at t=" << d.t << ": " << e.what() << std::endl;
            continue;
        }

        // Update with measurements
        for (const auto& y : d.y) {
            try {
                if (!std::isnan(y.y.d.unitVector().norm()) && !std::isnan(y.d.d.unitVector().norm())) {
                    filter.update(y);
                }
            } catch (const std::exception& e) {
                std::cerr << "Update error at t=" << d.t << ": " << e.what() << std::endl;
            }
        }

        // Get state estimate
        State estimate = filter.stateEstimate();

        // Compute errors
        Vector3 att_error = Rot3::Logmap(d.xi.R.between(estimate.R));
        Vector3 bias_error = estimate.b - d.xi.b;
        Vector3 cal_error = Vector3::Zero();
        if (!d.xi.S.empty() && !estimate.S.empty()) {
            cal_error = Rot3::Logmap(d.xi.S[0].between(estimate.S[0]));
        }

        // Store results
        times.push_back(d.t);
        attitude_errors.push_back(att_error);
        bias_errors.push_back(bias_error);
        calibration_errors.push_back(cal_error);

        // Print some info
        if (d.t < 0.1 || fmod(d.t, 1.0) < d.dt) {
            std::cout << "Time: " << d.t
                      << ", Attitude error (deg): " << (att_error.norm() * 180.0/M_PI)
                      << ", Bias error: " << bias_error.norm()
                      << ", Calibration error (deg): " << (cal_error.norm() * 180.0/M_PI)
                      << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Simulation completed in " << elapsed.count() << " seconds" << std::endl;

    // Print summary statistics
    double avg_att_error = 0.0;
    double avg_bias_error = 0.0;
    double avg_cal_error = 0.0;

    for (size_t i = 0; i < times.size(); i++) {
        avg_att_error += attitude_errors[i].norm();
        avg_bias_error += bias_errors[i].norm();
        avg_cal_error += calibration_errors[i].norm();
    }

    avg_att_error /= times.size();
    avg_bias_error /= times.size();
    avg_cal_error /= times.size();

    std::cout << "Average attitude error (deg): " << (avg_att_error * 180.0/M_PI) << std::endl;
    std::cout << "Average bias error: " << avg_bias_error << std::endl;
    std::cout << "Average calibration error (deg): " << (avg_cal_error * 180.0/M_PI) << std::endl;
}

// int main(int argc, char** argv) {
//     std::cout << "ABC-EqF: Attitude-Bias-Calibration Equivariant Filter" << std::endl;
//     std::cout << "========================================================" << std::endl;
//
//     // Initialize filter
//     int n_cal = 1;      // Number of calibration states
//     int n_sensors = 2;  // Number of sensors
//
//     // Initial covariance - larger values for higher uncertainty
//     Matrix initialSigma = Matrix::Identity(6 + 3*n_cal, 6 + 3*n_cal);
//     initialSigma.diagonal().head<3>() = Vector3::Constant(0.5); // Attitude uncertainty
//     initialSigma.diagonal().segment<3>(3) = Vector3::Constant(0.1); // Bias uncertainty
//     initialSigma.diagonal().tail<3>() = Vector3::Constant(0.5); // Calibration uncertainty
//
//     std::cout << "Creating filter with " << n_cal << " calibration states..." << std::endl;
//
//     try {
//         // Create filter
//         EqF filter(initialSigma, n_cal, n_sensors);
//
//         // Generate simulated data
//         std::cout << "Generating simulated data..." << std::endl;
//         std::vector<Data> data = loadSimulatedData();
//
//         // Run simulation
//         runSimulation(filter, data);
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }
//
//     std::cout << "Done." << std::endl;
//     return 0;
int main(int argc, char** argv) {
    std::cout << "ABC-EqF: Attitude-Bias-Calibration Equivariant Filter" << std::endl;
    std::cout << "========================================================" << std::endl;
    
    std::string csvFilePath;
    
    // Check if CSV file path is provided as command line argument
    if (argc > 1) {
        csvFilePath = argv[1];
    } else {
        std::cout << "Please enter the path to your CSV data file: ";
        std::cin >> csvFilePath;
    }
    
    std::cout << "Using CSV data from: " << csvFilePath << std::endl;
    
    try {
        // Run with CSV data
        runEqFWithCSVData(csvFilePath);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Done." << std::endl;
    return 0;

}