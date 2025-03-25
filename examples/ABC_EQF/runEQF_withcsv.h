//
// Created by darshan on 3/17/25.
//

#ifndef RUNEQF_WITHCSV_H
#define RUNEQF_WITHCSV_H

//
// Created by darshan on 3/17/25.
//
#include "Data.h"
#include "State.h"
#include "Input.h"
#include "Direction.h"
#include "Measurements.h"
#include "utilities.h"
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Quaternion.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <chrono>

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
inline std::vector<Data> loadDataFromCSV(const std::string& filename,
                                 int startRow = 0,
                                 int maxRows = -1,
                                 int downsample = 1) {
    std::vector<Data> data_list;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    int lineNumber = 0;
    int rowCount = 0;
    double prevTime = 0.0;

    // Skip header
    std::getline(file, line);
    lineNumber++;

    // Skip to startRow
    while (lineNumber < startRow && std::getline(file, line)) {
        lineNumber++;
    }

    // Read data
    while (std::getline(file, line) && (maxRows == -1 || rowCount < maxRows)) {
        lineNumber++;

        // Apply downsampling
        if ((lineNumber - startRow - 1) % downsample != 0) {
            continue;
        }

        std::istringstream ss(line);
        std::string token;
        std::vector<double> values;

        // Parse line into values
        while (std::getline(ss, token, ',')) {
            try {
                values.push_back(std::stod(token));
            } catch (const std::exception& e) {
                std::cerr << "Error parsing value at line " << lineNumber << ": " << token << std::endl;
                values.push_back(0.0); // Use default value
            }
        }

        // Check if we have enough values
        if (values.size() < 39) {
            std::cerr << "Warning: Line " << lineNumber << " has only " << values.size()
                      << " values, expected 39. Skipping." << std::endl;
            continue;
        }

        // Extract values
        double t = values[0];
        double dt = (rowCount == 0) ? 0.0 : t - prevTime;
        prevTime = t;

        // Create ground truth state
        Quaternion quat(values[1], values[2], values[3], values[4]); // w, x, y, z
        Rot3 R = Rot3(quat);

        Vector3 b(values[5], values[6], values[7]);

        Quaternion calQuat(values[8], values[9], values[10], values[11]); // w, x, y, z
        std::vector<Rot3> S = {Rot3(calQuat)};

        State xi(R, b, S);

        // Create input
        Vector3 w(values[12], values[13], values[14]);

        // Create input covariance matrix (6x6)
        // First 3x3 block for angular velocity, second 3x3 block for bias process noise
        Matrix inputCov = Matrix::Zero(6, 6);
        inputCov(0, 0) = values[15] * values[15]; // std_w_x^2
        inputCov(1, 1) = values[16] * values[16]; // std_w_y^2
        inputCov(2, 2) = values[17] * values[17]; // std_w_z^2
        inputCov(3, 3) = values[18] * values[18]; // std_b_x^2
        inputCov(4, 4) = values[19] * values[19]; // std_b_y^2
        inputCov(5, 5) = values[20] * values[20]; // std_b_z^2

        Input u(w, inputCov);

        // Create measurements
        std::vector<Measurement> measurements;

        // First measurement (calibrated sensor, cal_idx = 0)
        Vector3 y0(values[21], values[22], values[23]);
        Vector3 d0(values[33], values[34], values[35]);

        // Normalize vectors if needed
        if (abs(y0.norm() - 1.0) > 1e-5) y0.normalize();
        if (abs(d0.norm() - 1.0) > 1e-5) d0.normalize();

        // Measurement covariance
        Matrix3 covY0 = Matrix3::Zero();
        covY0(0, 0) = values[27] * values[27]; // std_y_x_0^2
        covY0(1, 1) = values[28] * values[28]; // std_y_y_0^2
        covY0(2, 2) = values[29] * values[29]; // std_y_z_0^2

        // Create measurement
        measurements.push_back(Measurement(y0, d0, covY0, 0));

        // Second measurement (calibrated sensor, cal_idx = -1)
        Vector3 y1(values[24], values[25], values[26]);
        Vector3 d1(values[36], values[37], values[38]);

        // Normalize vectors if needed
        if (abs(y1.norm() - 1.0) > 1e-5) y1.normalize();
        if (abs(d1.norm() - 1.0) > 1e-5) d1.normalize();

        // Measurement covariance
        Matrix3 covY1 = Matrix3::Zero();
        covY1(0, 0) = values[30] * values[30]; // std_y_x_1^2
        covY1(1, 1) = values[31] * values[31]; // std_y_y_1^2
        covY1(2, 2) = values[32] * values[32]; // std_y_z_1^2

        // Create measurement
        measurements.push_back(Measurement(y1, d1, covY1, -1));

        // Create Data object and add to list
        data_list.push_back(Data(xi, 1, u, measurements, 2, t, dt));

        rowCount++;
    }

    std::cout << "Loaded " << data_list.size() << " data points from CSV file." << std::endl;

    return data_list;
}

/**
 * Process Data objects with the EqF filter
 *
 * @param filter EqF filter to use
 * @param data_list Vector of Data objects
 * @param saveResults Whether to save results to a file
 * @param resultFilename Filename to save results to
 */
inline void printDataPoint(const Data& data, int index) {
    std::cout << "Data[" << index << "] @ t=" << data.t << ", dt=" << data.dt << std::endl;

    // Print angular velocity
    std::cout << "  ω = [" << data.u.w[0] << ", " << data.u.w[1] << ", " << data.u.w[2] << "]" << std::endl;

    // Print measurements
    for (size_t i = 0; i < data.y.size(); i++) {
        const Measurement& meas = data.y[i];
        // Use the unitVector() method to get a Vector3 from a Unit3 object
        Vector3 y_vec = meas.y.d.unitVector();
        Vector3 d_vec = meas.d.d.unitVector();
        std::cout << "  y" << i << " = [" << y_vec[0] << ", " << y_vec[1] << ", " << y_vec[2] << "]" << std::endl;
        std::cout << "  d" << i << " = [" << d_vec[0] << ", " << d_vec[1] << ", " << d_vec[2] << "]" << std::endl;
    }

    std::cout << std::endl;
}
   

// Function to print sample data points
inline void printDataSamples(const std::vector<Data>& data_list, int count = 3) {
    int total = data_list.size();

    std::cout << "\n=== First " << count << " Data Points ===" << std::endl;
    for (int i = 0; i < std::min(count, total); i++) {
        printDataPoint(data_list[i], i);
    }

    if (total > 2*count) {
        std::cout << "\n... (" << (total - 2*count) << " points omitted) ...\n" << std::endl;

        std::cout << "=== Last " << count << " Data Points ===" << std::endl;
        for (int i = std::max(count, total - count); i < total; i++) {
            printDataPoint(data_list[i], i);
        }
    }
}

// Function to validate data
inline bool validateData(const std::vector<Data>& data_list) {
    if (data_list.empty()) {
        std::cerr << "ERROR: No data loaded from CSV" << std::endl;
        return false;
    }

    std::cout << "Validating " << data_list.size() << " data points..." << std::endl;

    // Track statistics
    int invalid_count = 0;

    // Open a log file to record detailed issues
    std::ofstream logFile("data_validation.log");
    logFile << "Data Validation Report" << std::endl;
    logFile << "--------------------" << std::endl;

    for (size_t i = 0; i < data_list.size(); ++i) {
        const Data& data = data_list[i];
        bool point_valid = true;

        // Check time and dt
        if (std::isnan(data.t) || std::isnan(data.dt)) {
            logFile << "Point " << i << ": Invalid time values (t=" << data.t
                   << ", dt=" << data.dt << ")" << std::endl;
            point_valid = false;
        }

        // Check ground truth state for NaN - using isnan directly on components
        const auto& R_matrix = data.xi.R.matrix();
        bool R_has_nan = false;
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                if (std::isnan(R_matrix(r, c))) {
                    R_has_nan = true;
                    break;
                }
            }
        }

        if (R_has_nan) {
            logFile << "Point " << i << ": NaN in ground truth attitude matrix" << std::endl;
            point_valid = false;
        }

        // Check bias vector for NaN
        bool b_has_nan = false;
        for (int j = 0; j < 3; j++) {
            if (std::isnan(data.xi.b[j])) {
                b_has_nan = true;
                break;
            }
        }

        if (b_has_nan) {
            logFile << "Point " << i << ": NaN in ground truth bias vector" << std::endl;
            point_valid = false;
        }

        // Check calibration matrices for NaN
        for (size_t j = 0; j < data.xi.S.size(); ++j) {
            const auto& S_matrix = data.xi.S[j].matrix();
            bool S_has_nan = false;
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    if (std::isnan(S_matrix(r, c))) {
                        S_has_nan = true;
                        break;
                    }
                }
            }

            if (S_has_nan) {
                logFile << "Point " << i << ": NaN in ground truth calibration matrix "
                       << j << std::endl;
                point_valid = false;
            }
        }

        // Check input for NaN
        bool w_has_nan = false;
        for (int j = 0; j < 3; j++) {
            if (std::isnan(data.u.w[j])) {
                w_has_nan = true;
                break;
            }
        }

        if (w_has_nan) {
            logFile << "Point " << i << ": NaN in angular velocity" << std::endl;
            point_valid = false;
        }

        // Check measurements
        for (size_t j = 0; j < data.y.size(); ++j) {
            const Measurement& meas = data.y[j];

            // Get the Vector3 representations to check them
            Vector3 y_vec = meas.y.d.unitVector();
            Vector3 d_vec = meas.d.d.unitVector();

            // Check measurement vector for NaN
            bool y_has_nan = false;
            bool d_has_nan = false;

            for (int k = 0; k < 3; k++) {
                if (std::isnan(y_vec[k])) {
                    y_has_nan = true;
                    break;
                }
                if (std::isnan(d_vec[k])) {
                    d_has_nan = true;
                    break;
                }
            }

            if (y_has_nan) {
                logFile << "Point " << i << ", Meas " << j << ": NaN in measurement vector" << std::endl;
                point_valid = false;
            }

            if (d_has_nan) {
                logFile << "Point " << i << ", Meas " << j << ": NaN in reference direction" << std::endl;
                point_valid = false;
            }

            // Calculate norm using Vector3 norms
            double y_norm = y_vec.norm();
            double d_norm = d_vec.norm();

            if (std::abs(y_norm - 1.0) > 1e-5) {
                logFile << "Point " << i << ", Meas " << j
                       << ": Measurement vector not normalized. Norm = " << y_norm << std::endl;
                point_valid = false;
            }

            if (std::abs(d_norm - 1.0) > 1e-5) {
                logFile << "Point " << i << ", Meas " << j
                       << ": Reference direction not normalized. Norm = " << d_norm << std::endl;
                point_valid = false;
            }
        }

        if (!point_valid) {
            invalid_count++;

            // Print first few invalid points to console
            if (invalid_count <= 5) {
                std::cerr << "Invalid data at point " << i << " (t=" << data.t << ")" << std::endl;
            }
        }
    }

    // Close the log
    logFile << std::endl << "Summary: " << invalid_count << " invalid data points out of "
           << data_list.size() << std::endl;
    logFile.close();









    // Print summary
    std::cout << "Data validation complete. " << invalid_count << " invalid points found." << std::endl;
    if (invalid_count > 0) {
        std::cout << "See data_validation.log for details." << std::endl;
    }

    return (invalid_count == 0);
}

inline void processDataWithEqF(EqF& filter,
                       const std::vector<Data>& data_list,
                       bool saveResults = false,
                       const std::string& resultFilename = "eqf_results.csv") {
    std::ofstream resultFile;
    if (saveResults) {
        resultFile.open(resultFilename);
        if (!resultFile.is_open()) {
            throw std::runtime_error("Failed to open result file: " + resultFilename);
        }

        // Write header - now adding roll, pitch, yaw columns for estimated and true values
        resultFile << "t,";
        // Estimated state quaternion
        resultFile << "est_qw,est_qx,est_qy,est_qz,";
        // Estimated bias
        resultFile << "est_bx,est_by,est_bz,";
        // Estimated calibration quaternion
        resultFile << "est_cqw,est_cqx,est_cqy,est_cqz,";
        // True state quaternion
        resultFile << "true_qw,true_qx,true_qy,true_qz,";
        // True bias
        resultFile << "true_bx,true_by,true_bz,";
        // True calibration quaternion
        resultFile << "true_cqw,true_cqx,true_cqy,true_cqz,";
        // Add Euler angles for estimated state
        resultFile << "est_roll,est_pitch,est_yaw,";
        // Add Euler angles for true state
        resultFile << "true_roll,true_pitch,true_yaw,";
        // Add Euler angles for estimated calibration
        resultFile << "est_cal_roll,est_cal_pitch,est_cal_yaw,";
        // Add Euler angles for true calibration
        resultFile << "true_cal_roll,true_cal_pitch,true_cal_yaw";
        resultFile << std::endl;
    }

    std::cout << "Processing data with EqF..." << std::endl;

    // Track time for performance measurement
    auto start = std::chrono::high_resolution_clock::now();

    // Store error metrics
    std::vector<double> att_errors;
    std::vector<double> bias_errors;
    std::vector<double> cal_errors;

    int total_measurements = 0;
    int valid_measurements = 0;
    int invalid_measurements = 0;

    for (size_t i = 0; i < data_list.size(); i++) {
        const Data& data = data_list[i];

        // Propagation
        filter.propagation(data.u, data.dt);

        // Update with measurements
        for (const auto& y : data.y) {
            total_measurements++;

            // Check for NaN values in measurement vectors
            bool has_nan = false;
            Vector3 y_vec = y.y.d.unitVector();
            Vector3 d_vec = y.d.d.unitVector();

            for (int j = 0; j < 3; j++) {
                if (std::isnan(y_vec[j]) || std::isnan(d_vec[j])) {
                    has_nan = true;
                    break;
                }
            }

            if (!has_nan) {
                try {
                    filter.update(y);
                    valid_measurements++;
                } catch (const std::exception& e) {
                    std::cerr << "Error updating at t=" << data.t << ": " << e.what() << std::endl;
                    invalid_measurements++;
                }
            } else {
                invalid_measurements++;
            }
        }

        // Get state estimate
        State estimate = filter.stateEstimate();

        // Compute errors
        Vector3 att_error = Rot3::Logmap(data.xi.R.between(estimate.R));
        Vector3 bias_error = estimate.b - data.xi.b;
        Vector3 cal_error = Vector3::Zero();
        if (!data.xi.S.empty() && !estimate.S.empty()) {
            cal_error = Rot3::Logmap(data.xi.S[0].between(estimate.S[0]));
        }

        // Store errors
        att_errors.push_back(att_error.norm());
        bias_errors.push_back(bias_error.norm());
        cal_errors.push_back(cal_error.norm());

        // Save results
        if (saveResults) {
            // Extract quaternions
            Quaternion est_q = estimate.R.toQuaternion();
            Quaternion true_q = data.xi.R.toQuaternion();

            // Extract Euler angles (roll, pitch, yaw) from estimated rotation
            Vector3 est_rpy = estimate.R.rpy();
            // Convert to degrees for easier comparison
            Vector3 est_rpy_deg = est_rpy * 180.0 / M_PI;

            // Extract Euler angles from true rotation
            Vector3 true_rpy = data.xi.R.rpy();
            // Convert to degrees
            Vector3 true_rpy_deg = true_rpy * 180.0 / M_PI;

            // Get calibration quaternions and Euler angles
            Quaternion est_cal_q, true_cal_q;
            Vector3 est_cal_rpy_deg = Vector3::Zero();
            Vector3 true_cal_rpy_deg = Vector3::Zero();

            if (!estimate.S.empty() && !data.xi.S.empty()) {
                est_cal_q = estimate.S[0].toQuaternion();
                true_cal_q = data.xi.S[0].toQuaternion();

                // Get Euler angles for calibrations
                Vector3 est_cal_rpy = estimate.S[0].rpy();
                est_cal_rpy_deg = est_cal_rpy * 180.0 / M_PI;

                Vector3 true_cal_rpy = data.xi.S[0].rpy();
                true_cal_rpy_deg = true_cal_rpy * 180.0 / M_PI;
            } else {
                est_cal_q = Quaternion(1, 0, 0, 0);  // Identity quaternion
                true_cal_q = Quaternion(1, 0, 0, 0);
            }

            // Write to file
            resultFile << data.t << ",";
            // Estimated quaternion
            resultFile << est_q.w() << "," << est_q.x() << "," << est_q.y() << "," << est_q.z() << ",";
            // Estimated bias
            resultFile << estimate.b[0] << "," << estimate.b[1] << "," << estimate.b[2] << ",";
            // Estimated calibration quaternion
            resultFile << est_cal_q.w() << "," << est_cal_q.x() << "," << est_cal_q.y() << "," << est_cal_q.z() << ",";
            // True quaternion
            resultFile << true_q.w() << "," << true_q.x() << "," << true_q.y() << "," << true_q.z() << ",";
            // True bias
            resultFile << data.xi.b[0] << "," << data.xi.b[1] << "," << data.xi.b[2] << ",";
            // True calibration quaternion
            resultFile << true_cal_q.w() << "," << true_cal_q.x() << "," << true_cal_q.y() << "," << true_cal_q.z() << ",";

            // Add Euler angles (in degrees) for estimated state
            resultFile << est_rpy_deg[0] << "," << est_rpy_deg[1] << "," << est_rpy_deg[2] << ",";
            // Add Euler angles (in degrees) for true state
            resultFile << true_rpy_deg[0] << "," << true_rpy_deg[1] << "," << true_rpy_deg[2] << ",";
            // Add Euler angles (in degrees) for estimated calibration
            resultFile << est_cal_rpy_deg[0] << "," << est_cal_rpy_deg[1] << "," << est_cal_rpy_deg[2] << ",";
            // Add Euler angles (in degrees) for true calibration
            resultFile << true_cal_rpy_deg[0] << "," << true_cal_rpy_deg[1] << "," << true_cal_rpy_deg[2];

            resultFile << std::endl;
        }

        // Print progress
        if (i % 1000 == 0 || i == data_list.size() - 1) {
            std::cout << "Processed " << i+1 << "/" << data_list.size()
                      << " (" << (100.0 * (i+1) / data_list.size()) << "%) ";
            std::cout << "Attitude error: " << (att_error.norm() * 180.0/M_PI) << " deg, ";
            std::cout << "Bias error: " << bias_error.norm() << ", ";
            std::cout << "Calibration error: " << (cal_error.norm() * 180.0/M_PI) << " deg" << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Calculate average errors
    double avg_att_error = 0.0;
    double avg_bias_error = 0.0;
    double avg_cal_error = 0.0;

    if (!att_errors.empty()) {
        avg_att_error = std::accumulate(att_errors.begin(), att_errors.end(), 0.0) / att_errors.size();
        avg_bias_error = std::accumulate(bias_errors.begin(), bias_errors.end(), 0.0) / bias_errors.size();
        avg_cal_error = std::accumulate(cal_errors.begin(), cal_errors.end(), 0.0) / cal_errors.size();
    }

    std::cout << std::endl;
    std::cout << "EqF Processing completed in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Average attitude error: " << (avg_att_error * 180.0/M_PI) << " deg" << std::endl;
    std::cout << "Average bias error: " << avg_bias_error << std::endl;
    std::cout << "Average calibration error: " << (avg_cal_error * 180.0/M_PI) << " deg" << std::endl;
    std::cout << "Total measurements: " << total_measurements << std::endl;
    std::cout << "Valid measurements processed: " << valid_measurements << std::endl;
    std::cout << "Invalid measurements skipped: " << invalid_measurements << std::endl;

    if (saveResults) {
        resultFile.close();
        std::cout << "Results saved to " << resultFilename << std::endl;
    }
}





inline void runEqFWithCSVData(const std::string& filename) {
    try {
        // Load data from CSV file with optional parameters
        int startRow = 0;
        int maxRows = -1;
        int downsample = 1;

        std::vector<Data> data = loadDataFromCSV(filename, startRow, maxRows, downsample);

        if (data.empty()) {
            std::cerr << "No data loaded from CSV file." << std::endl;
            return;
        }

        // Print sample data points to inspect the loaded data
        std::cout << "Data loaded, displaying samples..." << std::endl;
        printDataSamples(data);

        // Validate the data to check for issues
        std::cout << "Validating data integrity..." << std::endl;
        bool dataValid = validateData(data);

        if (!dataValid) {
            std::cout << "Warning: Data validation found issues." << std::endl;
            std::string proceed;
            std::cout << "Do you want to proceed anyway? (y/n): ";
            std::cin >> proceed;
            if (proceed != "y" && proceed != "Y") {
                std::cout << "Operation cancelled by user." << std::endl;
                return;
            }
        }

        // Initialize EqF filter
        int n_cal = 1;      // Number of calibration states (from the data)
        int n_sensors = 2;  // Number of sensors (from the data)

        // Initial covariance
        Matrix initialSigma = Matrix::Identity(6 + 3*n_cal, 6 + 3*n_cal);
        initialSigma.diagonal().head<3>() = Vector3::Constant(0.1); // Reduced attitude uncertainty
        initialSigma.diagonal().segment<3>(3) = Vector3::Constant(0.01); // Reduced bias uncertainty
        initialSigma.diagonal().tail<3>() = Vector3::Constant(0.1); // Reduced calibration uncertainty

        // Create filter
        EqF filter(initialSigma, n_cal, n_sensors);

        // Initialize filter state from the first ground truth if possible
        if (!data.empty()) {
            // You'll need to add a method to your EqF class to set the initial state
            // Something like:
            // filter.setInitialState(data[0].xi);

            // If you don't have such a method, you can print the first ground truth
            // to see if it makes sense
            std::cout << "First ground truth state:" << std::endl;
            std::cout << "Attitude: " << data[0].xi.R.matrix() << std::endl;
            std::cout << "Bias: " << data[0].xi.b.transpose() << std::endl;
            std::cout << "Calibration: " << data[0].xi.S[0].matrix() << std::endl;
        }

        // Process data with the filter and save results
        processDataWithEqF(filter, data, true, "eqf_results.csv");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}
/**
 * Example usage function to demonstrate how to use the data loader with the EqF
 */

#endif //RUNEQF_WITHCSV_H
