/**
 * @file ABC_EQF_Demo.cpp
 * @brief Demonstration of the full Attitude-Bias-Calibration Equivariant Filter
 *
 * This demo shows the Equivariant Filter (EqF) for attitude estimation
 * with both gyroscope bias and sensor extrinsic calibration, based on the paper:
 * "Overcoming Bias: Equivariant Filter Design for Biased Attitude Estimation
 * with Online Calibration" by Fornasier et al.
 * Authors: Darshan Rajasekaran & Jennifer Oum
 */

#include "ABC_EQF.h"

// Use namespace for convenience
using namespace abc_eqf_lib;
using namespace gtsam;

/**
 * Main function for the EqF demo
 * @param argc Number of arguments
 * @param argv Array of arguments
 * @return Exit code
 */


int main(int argc, char* argv[]) {
    std::cout << "ABC-EqF: Attitude-Bias-Calibration Equivariant Filter Demo" << std::endl;
    std::cout << "==============================================================" << std::endl;

    try {
        // Parse command line options
        std::string csvFilePath;
        int maxRows = -1; // Process all rows by default
        int downsample = 1; // No downsampling by default

        if (argc > 1) {
            csvFilePath = argv[1];
        } else {
            // Try to find the EQFdata file in the GTSAM examples directory
            try {
                csvFilePath = findExampleDataFile("EqFdata.csv");
            } catch (const std::exception& e) {
                std::cerr << "Error: Could not find EqFdata.csv" << std::endl;
                std::cerr << "Usage: " << argv[0] << " [csv_file_path] [max_rows] [downsample]" << std::endl;
                return 1;
            }
        }

        // Optional command line parameters
        if (argc > 2) {
            maxRows = std::stoi(argv[2]);
        }

        if (argc > 3) {
            downsample = std::stoi(argv[3]);
        }

        // Load data from CSV file
        std::vector<Data> data = loadDataFromCSV(csvFilePath, 0, maxRows, downsample);

        if (data.empty()) {
            std::cerr << "No data available to process. Exiting." << std::endl;
            return 1;
        }

        // Initialize the EqF filter with one calibration state
        int n_cal = 1;
        int n_sensors = 2;

        // Initial covariance - larger values allow faster convergence
        Matrix initialSigma = Matrix::Identity(6 + 3*n_cal, 6 + 3*n_cal);
        initialSigma.diagonal().head<3>() = Vector3::Constant(0.1); // Attitude uncertainty
        initialSigma.diagonal().segment<3>(3) = Vector3::Constant(0.01); // Bias uncertainty
        initialSigma.diagonal().tail<3>() = Vector3::Constant(0.1); // Calibration uncertainty

        // Create filter
        EqF filter(initialSigma, n_cal, n_sensors);

        // Process data
        processDataWithEqF(filter, data);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nEqF demonstration completed successfully." << std::endl;
    return 0;
}