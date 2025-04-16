/**
 * @file ABC_EQF.cpp
 * @brief Implementation of the Attitude-Bias-Calibration Equivariant Filter
 *
 * This file contains the implementation for the Equivariant Filter (EqF) for attitude estimation
 * with both gyroscope bias and sensor extrinsic calibration, based on the paper:
 * "Overcoming Bias: Equivariant Filter Design for Biased Attitude Estimation
 * with Online Calibration" by Fornasier et al.
 * Authors: Darshan Rajasekaran & Jennifer Oum
 */

#include "ABC_EQF.h"

namespace abc_eqf_lib {

// Global configuration
const std::string COORDINATE = "EXPONENTIAL";  // Denotes how the states are mapped to the vector representations

//========================================================================
// Utility Functions
//========================================================================
/**
 * @brief Verifies if a vector has unit norm within tolerance
 * @param x 3d vector
 * @param tol optional tolerance
 * @return Bool indicating that the vector norm is approximately 1
 * Uses Vector3 norm() method to calculate vector magnitude
 */
bool checkNorm(const Vector3& x, double tol) {
    return abs(x.norm() - 1) < tol || std::isnan(x.norm());
}

/**
 * @brief Checks if the input vector has any NaNs
 * @param vec A 3-D vector
 * @return true if present, false otherwise
 */
bool hasNaN(const Vector3& vec) {
    return std::isnan(vec[0]) || std::isnan(vec[1]) || std::isnan(vec[2]);
}

/**
 * @brief Creates a block diagonal matrix from input matrices
 * @param A Matrix A
 * @param B Matrix B
 * @return A single consolidated matrix with A in the top left and B in the
 * bottom right
 * Uses Matrix's rows(), cols(), setZero(), and block() methods
 */

Matrix blockDiag(const Matrix& A, const Matrix& B) {
    if (A.size() == 0) {
        return B;
    } else if (B.size() == 0) {
        return A;
    } else {
        Matrix result(A.rows() + B.rows(), A.cols() + B.cols());
        result.setZero();
        result.block(0, 0, A.rows(), A.cols()) = A;
        result.block(A.rows(), A.cols(), B.rows(), B.cols()) = B;
        return result;
    }
}
/**
 * @brief Creates a block diagonal matrix by repeating a matrix 'n' times
 * @param A A  matrix
 * @param n Number of times to be repeated
 * @return Block diag matrix with A repeated 'n' times
 * Recursively uses blockDiag() function
 */
Matrix repBlock(const Matrix& A, int n) {
    if (n <= 0) return Matrix();

    Matrix result = A;
    for (int i = 1; i < n; i++) {
        result = blockDiag(result, A);
    }
    return result;
}

/**
 * @brief Calculates the Jacobian matrix using central difference approximation
 * @param f Vector function f
 * @param x The point at which Jacobian is evaluated
 * @return Matrix containing numerical partial derivatives of f at x
 * Uses Vector's size() and Zero(), Matrix's Zero() and col() methods
 */
Matrix numericalDifferential(std::function<Vector(const Vector&)> f, const Vector& x) {
    double h = 1e-6;
    Vector fx = f(x);
    int n = fx.size();
    int m = x.size();
    Matrix Df = Matrix::Zero(n, m);

    for (int j = 0; j < m; j++) {
        Vector ej = Vector::Zero(m);
        ej(j) = 1.0;

        Vector fplus = f(x + h * ej);
        Vector fminus = f(x - h * ej);

        Df.col(j) = (fplus - fminus) / (2*h);
    }

    return Df;
}

//========================================================================
// Direction Class Implementation
//========================================================================

/**
 * @brief Initializes a direction object vector from a provided 3D vector input
 * @param d_vec A 3-D vector that should have a unit norm(This represents a
 * direction in 3D space) Uses Unit3's constructor which normalizes vectors
 */


Direction::Direction(const Vector3& d_vec) : d(d_vec) {
    if (!checkNorm(d_vec)) {
        throw std::invalid_argument("Direction must be a unit vector");
    }
}
 /** Access the individual components of the direction vector defined above using this methods below
  * Uses the Unit3 object from GTSAM to compute the components
  */

double Direction::x() const { 
    return d.unitVector()[0]; 
}

double Direction::y() const { 
    return d.unitVector()[1]; 
}

double Direction::z() const { 
    return d.unitVector()[2]; 
}

bool Direction::hasNaN() const {
    Vector3 vec = d.unitVector();
    return std::isnan(vec[0]) || std::isnan(vec[1]) || std::isnan(vec[2]);
}

//========================================================================
// Input Class Implementation
//========================================================================
/**
 * @brief Constructs an input object using the Angular velocity vector and the
 * covariance matrix
 * @param w Angular vector
 * @param Sigma 6X6 covariance matrix
 * Uses Matrix's rows(), cols() and Eigen's SelfAdjointEigenSolver
 */
Input::Input(const Vector3& w, const Matrix& Sigma) : w(w), Sigma(Sigma) {
    if (Sigma.rows() != 6 || Sigma.cols() != 6) {
        throw std::invalid_argument("Input measurement noise covariance must be 6x6");
    }

    // Check positive semi-definite
    Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(Sigma);
    if (eigensolver.eigenvalues().minCoeff() < -1e-10) {
        throw std::invalid_argument("Covariance matrix must be semi-positive definite");
    }
}
/**
 *
 * @return 3X3 skey symmetric matrix when called
 * Uses Rot3's Hat() to create skew-symmetric matrix
 */
Matrix3 Input::W() const {
    return Rot3::Hat(w);
}


//========================================================================
// Measurement Class Implementation
//========================================================================
/**
 * @brief Constructs measurement with directions and covariance.
 * @param y_vec A 3D vector representing the measured direction in the sensor frame
 * @param d_vec A 3D vector representing the known reference direction in the global frame aka ground truth direction
 * @param Sigma 3x3 positive definite covariance vector representing the uncertainty in the measurements
 * @param i Calibration index - A non-negative integer specifies the element in the calibration vector
 * that corresponds to the sensor of interest. A value of -1 indicates that all the sensors have been calibrated
 *
 * Creates a measurement object that stores the measured direction(y), reference direction(d), measurement noise covariance(Sigma)
 * and Calibration Index cal_idx
 *
 * Uses Eigen's SelfAdjointEigenSolver
 *
 */

Measurement::Measurement(const Vector3& y_vec, const Vector3& d_vec,
                         const Matrix3& Sigma, int i)
    : y(y_vec), d(d_vec), Sigma(Sigma), cal_idx(i) {

    // Check positive semi-definite
    Eigen::SelfAdjointEigenSolver<Matrix3> eigensolver(Sigma);
    if (eigensolver.eigenvalues().minCoeff() < -1e-10) {
        throw std::invalid_argument("Covariance matrix must be semi-positive definite");
    }
}

//========================================================================
// State Class Implementation
//========================================================================
/**
 *
 * @param R Rot3 (Attitude)
 * @param b Vector (Bias)
 * @param S Vector (Rot 3 calibration states)
 * Combines the navigation and the calibration states together and provides a
 * mechanism to represent the complete system
 *
 */
State::State(const Rot3& R, const Vector3& b, const std::vector<Rot3>& S)
    : R(R), b(b), S(S) {}

/**
 *
 * @param n Number of Calibration states
 * @return State object intitialized to identity
 * Creates a default/ initial state
 * Uses GTSAM's Rot3::identity and Vector3 zero function
 */
State State::identity(int n) {
    std::vector<Rot3> calibrations(n, Rot3::Identity());
    return State(Rot3::Identity(), Vector3::Zero(), calibrations);
}

//========================================================================
// Data Struct Implementation
//========================================================================

/**
 *
 * @param xi Ground Truth state
 * @param n_cal Number of calibration states
 * @param u Input measurements
 * @param y Vector of y measurements
 * @param n_meas number of measurements
 * @param t timestamp
 * @param dt time step
 * Used to organize the state, input and measurement data with timestamps for
 * testing Uses Rot3, Vector 3 and Unit3 classes
 */
Data::Data(const State& xi, int n_cal, const Input& u,
           const std::vector<Measurement>& y, int n_meas,
           double t, double dt)
    : xi(xi), n_cal(n_cal), u(u), y(y), n_meas(n_meas), t(t), dt(dt) {}

//========================================================================
// Symmetry Group Implementation - Group Elements and actions
//========================================================================
/**
 *
 * @param A Attitude element of Rot3 type
 * @param a Matrix3 bias element
 * @param B Rot3 vector containing calibration elements
 * Ouptuts a G object using Rot3 for rotation representation
 */
G::G(const Rot3& A, const Matrix3& a, const std::vector<Rot3>& B)
    : A(A), a(a), B(B) {}

/**
 * Defines the group operation (multiplication)
 * @param other Another Group element
 * @return G a product of two group elements
 * Uses Rot3 Hat, Rot3 Vee for multiplication
 *
 */
G G::operator*(const G& other) const {
    if (B.size() != other.B.size()) {
        throw std::invalid_argument("Group elements must have the same number of calibration elements");
    }

    std::vector<Rot3> new_B;
    for (size_t i = 0; i < B.size(); i++) {
        new_B.push_back(B[i] * other.B[i]);
    }

    return G(A * other.A,
            a + Rot3::Hat(A.matrix() * Rot3::Vee(other.a)),
            new_B);
}

/**
 * Used to compute the Group inverse
 * @return The inverse of group element
 * Uses Rot3 inverse, Rot3 matrix, hat and vee functions
 * The left invariant  property of the semi-direct product group structure is implemented here by using the -ve sign
 */
G G::inv() const {
    Matrix3 A_inv = A.inverse().matrix();

    std::vector<Rot3> B_inv;
    for (const auto& b : B) {
        B_inv.push_back(b.inverse());
    }

    return G(A.inverse(),
            -Rot3::Hat(A_inv * Rot3::Vee(a)),
            B_inv);
}

/**
 * Creates the identity element of the group
 * @param n Number of calibration elements
 * @return the identity element
 * Uses Rot3 Identity and Matrix zero
 */
G G::identity(int n) {
    std::vector<Rot3> B(n, Rot3::Identity());
    return G(Rot3::Identity(), Matrix3::Zero(), B);
}
/**
 * Maps the tangent space elements to the group
 * @param x Vector in lie algebra
 * @return the group element G
 * Uses Rot3 expmap and Expmapderivative function
 */
G G::exp(const Vector& x) {
    if (x.size() < 6 || x.size() % 3 != 0) {
        throw std::invalid_argument("Wrong size, a vector with size multiple of 3 and at least 6 must be provided");
    }

    int n = (x.size() - 6) / 3;
    Rot3 A = Rot3::Expmap(x.head<3>());

    Vector3 a_vee = Rot3::ExpmapDerivative(-x.head<3>()) * x.segment<3>(3);
    Matrix3 a = Rot3::Hat(a_vee);

    std::vector<Rot3> B;
    for (int i = 0; i < n; i++) {
        B.push_back(Rot3::Expmap(x.segment<3>(6 + 3*i)));
    }

    return G(A, a, B);
}

//========================================================================
// Helper Functions Implementation
//========================================================================

/**
 * Maps system dynamics to the symmetry group
 * @param xi State
 * @param u Input
 * @return Lifted input in Lie Algebra
 * Uses Vector zero & Rot3 inverse, matrix functions
 */
Vector lift(const State& xi, const Input& u) {
    int n = xi.S.size();
    Vector L = Vector::Zero(6 + 3 * n);

    // First 3 elements
    L.head<3>() = u.w - xi.b;

    // Next 3 elements
    L.segment<3>(3) = -u.W() * xi.b;

    // Remaining elements
    for (int i = 0; i < n; i++) {
        L.segment<3>(6 + 3*i) = xi.S[i].inverse().matrix() * L.head<3>();
    }

    return L;
}
/**
 * Implements group actions on the states
 * @param X A symmetry group element G consisting of the attitude, bias and the
 * calibration components X.a -> Rotation matrix containing the attitude X.b ->
 * A skew-symmetric matrix representing bias X.B -> A vector of Rotation
 * matrices for the calibration components
 * @param xi State object
 * xi.R -> Attitude (Rot3)
 * xi.b -> Gyroscope Bias(Vector 3)
 * xi.S -> Vector of calibration matrices(Rot3)
 * @return Transformed state
 * Uses the Rot3 inverse and Vee functions
 */
State stateAction(const G& X, const State& xi) {
    if (xi.S.size() != X.B.size()) {
        throw std::invalid_argument("Number of calibration states and B elements must match");
    }

    std::vector<Rot3> new_S;
    for (size_t i = 0; i < X.B.size(); i++) {
        new_S.push_back(X.A.inverse() * xi.S[i] * X.B[i]);
    }

    return State(xi.R * X.A,
                X.A.inverse().matrix() * (xi.b - Rot3::Vee(X.a)),
                new_S);
}
/**
 * Transforms the angular velocity measurements b/w frames
 * @param X A symmetry group element X with the components
 * @param u Inputs
 * @return Transformed inputs
 * Uses Rot3 Inverse, matrix and Vee functions and is critical for maintaining
 * the input equivariance
 */
Input velocityAction(const G& X, const Input& u) {
    return Input(X.A.inverse().matrix() * (u.w - Rot3::Vee(X.a)), u.Sigma);
}
/**
 * Transforms the Direction measurements based on the calibration type ( Eqn 6)
 * @param X Group element X
 * @param y Direction measurement y
 * @param idx Calibration index
 * @return Transformed direction
 * Uses Rot3 inverse, matric and Unit3 unitvector functions
 */
Vector3 outputAction(const G& X, const Direction& y, int idx) {
    if (idx == -1) {
        return X.A.inverse().matrix() * y.d.unitVector();
    } else {
        if (idx >= static_cast<int>(X.B.size())) {
            throw std::out_of_range("Calibration index out of range");
        }
        return X.B[idx].inverse().matrix() * y.d.unitVector();
    }
}

/**
 * Maps the error states to vector representations through exponential
 * coordinates
 * @param e error state
 * @return Vector with local coordinates
 * Uses Rot3 logamo for mapping rotations to the tangent space
 */
Vector local_coords(const State& e) {
    if (COORDINATE == "EXPONENTIAL") {
        Vector eps(6 + 3 * e.S.size());

        // First 3 elements
        eps.head<3>() = Rot3::Logmap(e.R);

        // Next 3 elements
        eps.segment<3>(3) = e.b;

        // Remaining elements
        for (size_t i = 0; i < e.S.size(); i++) {
            eps.segment<3>(6 + 3*i) = Rot3::Logmap(e.S[i]);
        }

        return eps;
    } else if (COORDINATE == "NORMAL") {
        throw std::runtime_error("Normal coordinate representation is not implemented yet");
    } else {
        throw std::invalid_argument("Invalid coordinate representation");
    }
}
/**
 * Used to convert the vectorized errors back to state space
 * @param eps Local coordinates in the exponential parameterization
 * @return State object corresponding to these coordinates
 * Uses Rot3 expmap through the G::exp() function
 */
State local_coords_inv(const Vector& eps) {
    G X = G::exp(eps);

    if (COORDINATE == "EXPONENTIAL") {
        std::vector<Rot3> S = X.B;
        return State(X.A, eps.segment<3>(3), S);
    } else if (COORDINATE == "NORMAL") {
        throw std::runtime_error("Normal coordinate representation is not implemented yet");
    } else {
        throw std::invalid_argument("Invalid coordinate representation");
    }
}
/**
 * Computes the differential of a state action at the identity of the symmetry
 * group
 * @param xi State object Xi representing the point at which to evaluate the
 * differential
 * @return A matrix representing the jacobian of the state action
 * Uses numericalDifferential, and Rot3 expmap, logmap
 */
Matrix stateActionDiff(const State& xi) {
    std::function<Vector(const Vector&)> coordsAction =
        [&xi](const Vector& U) {
            return local_coords(stateAction(G::exp(U), xi));
        };

    Vector zeros = Vector::Zero(6 + 3 * xi.S.size());
    Matrix differential = numericalDifferential(coordsAction, zeros);
    return differential;
}

//========================================================================
// Equivariant Filter (EqF) Implementation
//========================================================================
/**
 * Initializes the EqF with state dimension validation  and computes lifted
 * innovation mapping
 * @param Sigma Initial covariance
 * @param n Number of calibration states
 * @param m Number of sensors
 * Uses SelfAdjointSolver, completeOrthoganalDecomposition().pseudoInverse()
 */
EqF::EqF(const Matrix& Sigma, int n, int m)
    : __dof(6 + 3 * n), __n_cal(n), __n_sensor(m), __X_hat(G::identity(n)),
      __Sigma(Sigma), __xi_0(State::identity(n)) {

    if (Sigma.rows() != __dof || Sigma.cols() != __dof) {
        throw std::invalid_argument("Initial covariance dimensions must match the degrees of freedom");
    }

    // Check positive semi-definite
    Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(Sigma);
    if (eigensolver.eigenvalues().minCoeff() < -1e-10) {
        throw std::invalid_argument("Covariance matrix must be semi-positive definite");
    }

    if (n < 0) {
        throw std::invalid_argument("Number of calibration states must be non-negative");
    }

    if (m <= 1) {
        throw std::invalid_argument("Number of direction sensors must be at least 2");
    }

    // Compute differential of phi
    __Dphi0 = stateActionDiff(__xi_0);
    __InnovationLift = __Dphi0.completeOrthogonalDecomposition().pseudoInverse();
}
/**
 * Computes the internal group state to a physical state estimate
 * @return Current state estimate
 */
State EqF::stateEstimate() const {
    return stateAction(__X_hat, __xi_0);
}
/**
 * Implements the prediction step of the EqF using system dynamics and
 * covariance propagation and advances the filter state by symmtery-preserving
 * dynamics.Uses a Lie group integrator scheme for discrete time propagation
 * @param u Angular velocity measurements
 * @param dt time steps
 * Updated internal state and covariance
 */
void EqF::propagation(const Input& u, double dt) {
    State state_est = stateEstimate();
    Vector L = lift(state_est, u);

    Matrix Phi_DT = __stateTransitionMatrix(u, dt);
    Matrix Bt = __inputMatrixBt();

    Matrix tempSigma = blockDiag(u.Sigma,
                                repBlock(1e-9 * Matrix3::Identity(), __n_cal));
    Matrix M_DT = (Bt * tempSigma * Bt.transpose()) * dt;

    __X_hat = __X_hat * G::exp(L * dt);
    __Sigma = Phi_DT * __Sigma * Phi_DT.transpose() + M_DT;
}
/**
 * Implements the correction step of the filter using discrete measurements
 * Computes the measurement residual, Kalman gain and the updates both the state
 * and covariance
 *
 * @param y Measurements
 */
void EqF::update(const Measurement& y) {
  if (y.cal_idx > __n_cal) {
    throw std::invalid_argument("Calibration index out of range");
  }

  // Get vector representations for checking
  Vector3 y_vec = y.y.d.unitVector();
  Vector3 d_vec = y.d.d.unitVector();

  // Skip update if any NaN values are present
  if (std::isnan(y_vec[0]) || std::isnan(y_vec[1]) || std::isnan(y_vec[2]) ||
      std::isnan(d_vec[0]) || std::isnan(d_vec[1]) || std::isnan(d_vec[2])) {
    return;  // Skip this measurement
  }

  Matrix Ct = __measurementMatrixC(y.d, y.cal_idx);
  Vector3 action_result = outputAction(__X_hat.inv(), y.y, y.cal_idx);
  Vector3 delta_vec = Rot3::Hat(y.d.d.unitVector()) * action_result;
  Matrix Dt = __outputMatrixDt(y.cal_idx);
  Matrix S = Ct * __Sigma * Ct.transpose() + Dt * y.Sigma * Dt.transpose();
  Matrix K = __Sigma * Ct.transpose() * S.inverse();
  Vector Delta = __InnovationLift * K * delta_vec;
  __X_hat = G::exp(Delta) * __X_hat;
  __Sigma = (Matrix::Identity(__dof, __dof) - K * Ct) * __Sigma;
}
/**
 * Computes linearized continuous time state matrix
 * @param u Angular velocity
 * @return Linearized state matrix
 * Uses Matrix zero and Identity functions
 */
Matrix EqF::__stateMatrixA(const Input& u) const {
    Matrix3 W0 = velocityAction(__X_hat.inv(), u).W();
    Matrix A1 = Matrix::Zero(6, 6);

    if (COORDINATE == "EXPONENTIAL") {
        A1.block<3, 3>(0, 3) = -Matrix3::Identity();
        A1.block<3, 3>(3, 3) = W0;
        Matrix A2 = repBlock(W0, __n_cal);
        return blockDiag(A1, A2);
    } else if (COORDINATE == "NORMAL") {
        throw std::runtime_error("Normal coordinate representation is not implemented yet");
    } else {
        throw std::invalid_argument("Invalid coordinate representation");
    }
}

/**
 * Computes the discrete time state transition matrix
 * @param u Angular velocity
 * @param dt time step
 * @return State transition matrix in discrete time
 */
Matrix EqF::__stateTransitionMatrix(const Input& u, double dt) const {
    Matrix3 W0 = velocityAction(__X_hat.inv(), u).W();
    Matrix Phi1 = Matrix::Zero(6, 6);

    Matrix3 Phi12 = -dt * (Matrix3::Identity() + (dt / 2) * W0 + ((dt*dt) / 6) * W0 * W0);
    Matrix3 Phi22 = Matrix3::Identity() + dt * W0 + ((dt*dt) / 2) * W0 * W0;

    if (COORDINATE == "EXPONENTIAL") {
        Phi1.block<3, 3>(0, 0) = Matrix3::Identity();
        Phi1.block<3, 3>(0, 3) = Phi12;
        Phi1.block<3, 3>(3, 3) = Phi22;
        Matrix Phi2 = repBlock(Phi22, __n_cal);
        return blockDiag(Phi1, Phi2);
    } else if (COORDINATE == "NORMAL") {
        throw std::runtime_error("Normal coordinate representation is not implemented yet");
    } else {
        throw std::invalid_argument("Invalid coordinate representation");
    }
}
/**
 * Computes the input uncertainty propagation matrix
 * @return
 * Uses the blockdiag matrix
 */
Matrix EqF::__inputMatrixBt() const {
    if (COORDINATE == "EXPONENTIAL") {
        Matrix B1 = blockDiag(__X_hat.A.matrix(), __X_hat.A.matrix());
        Matrix B2;

        for (const auto& B : __X_hat.B) {
            if (B2.size() == 0) {
                B2 = B.matrix();
            } else {
                B2 = blockDiag(B2, B.matrix());
            }
        }

        return blockDiag(B1, B2);
    } else if (COORDINATE == "NORMAL") {
        throw std::runtime_error("Normal coordinate representation is not implemented yet");
    } else {
        throw std::invalid_argument("Invalid coordinate representation");
    }
}
/**
 * Computes the linearized measurement matrix. The structure depends on whether
 * the sensor has a calibration state
 * @param d reference direction
 * @param idx Calibration index
 * @return Measurement matrix
 * Uses the matrix zero, Rot3 hat and the Unitvector functions
 */
Matrix EqF::__measurementMatrixC(const Direction& d, int idx) const {
    Matrix Cc = Matrix::Zero(3, 3 * __n_cal);

    // If the measurement is related to a sensor that has a calibration state
    if (idx >= 0) {
        // Set the correct 3x3 block in Cc
        Cc.block<3, 3>(0, 3 * idx) = Rot3::Hat(d.d.unitVector());
    }

    Matrix3 wedge_d = Rot3::Hat(d.d.unitVector());

    // Create the combined matrix
    Matrix temp(3, 6 + 3 * __n_cal);
    temp.block<3, 3>(0, 0) = wedge_d;
    temp.block<3, 3>(0, 3) = Matrix3::Zero();
    temp.block(0, 6, 3, 3 * __n_cal) = Cc;

    return wedge_d * temp;
}
/**
 * Computes the measurement uncertainty propagation matrix
 * @param idx Calibration index
 * @return Returns B[idx] for calibrated sensors, A for uncalibrated
 */
Matrix EqF::__outputMatrixDt(int idx) const {
    // If the measurement is related to a sensor that has a calibration state
    if (idx >= 0) {
        if (idx >= static_cast<int>(__X_hat.B.size())) {
            throw std::out_of_range("Calibration index out of range");
        }
        return __X_hat.B[idx].matrix();
    } else {
        return __X_hat.A.matrix();
    }
}

//========================================================================
// Data Processing Functions Implementation
//========================================================================

/**
 * @brief Loads the test data from the csv file
 * @param filename path to the csv file is specified
 * @param startRow First row to load based on csv, 0 by default
 * @param maxRows maximum rows to load, defaults to all rows
 * @param downsample Downsample factor, default 1
 * @return A list of data objects
*/



std::vector<Data> loadDataFromCSV(const std::string& filename,
                                  int startRow,
                                  int maxRows,
                                  int downsample) {
    std::vector<Data> data_list;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::cout << "Loading data from " << filename << "..." << std::flush;

    std::string line;
    int lineNumber = 0;
    int rowCount = 0;
    int errorCount = 0;
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
                errorCount++;
                values.push_back(0.0); // Use default value
            }
        }

        // Check if we have enough values
        if (values.size() < 39) {
            errorCount++;
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
        
        // Show loading progress every 1000 rows
        if (rowCount % 1000 == 0) {
            std::cout << "." << std::flush;
        }
    }

    std::cout << " Done!" << std::endl;
    std::cout << "Loaded " << data_list.size() << " data points";
    
    if (errorCount > 0) {
        std::cout << " (" << errorCount << " errors encountered)";
    }
    
    std::cout << std::endl;

    return data_list;
}
/**
 * @brief Takes in the data and runs an EqF on it and reports the results
 * @param filter Initialized EqF filter
 * @param data_list std::vector<Data>
 * @param printInterval Progress indicator
 * Prints the performance statstics like average error etc
 * Uses Rot3 between, logmap and rpy functions
 */
void processDataWithEqF(EqF& filter, const std::vector<Data>& data_list, int printInterval) {
    if (data_list.empty()) {
        std::cerr << "No data to process" << std::endl;
        return;
    }

    std::cout << "Processing " << data_list.size() << " data points with EqF..." << std::endl;

    // Track performance metrics
    std::vector<double> att_errors;
    std::vector<double> bias_errors;
    std::vector<double> cal_errors;

    // Track time for performance measurement
    auto start = std::chrono::high_resolution_clock::now();

    int totalMeasurements = 0;
    int validMeasurements = 0;

    // Define constant for converting radians to degrees
    const double RAD_TO_DEG = 180.0 / M_PI;

    // Print a progress indicator
    int progressStep = data_list.size() / 10; // 10 progress updates
    if (progressStep < 1) progressStep = 1;
    
    std::cout << "Progress: ";
    
    for (size_t i = 0; i < data_list.size(); i++) {
        const Data& data = data_list[i];

        // Propagate filter with current input and time step
        filter.propagation(data.u, data.dt);

        // Process all measurements
        for (const auto& y : data.y) {
            totalMeasurements++;

            // Skip invalid measurements
            if (y.y.hasNaN() || y.d.hasNaN()) {
                continue;
            }

            try {
                filter.update(y);
                validMeasurements++;
            } catch (const std::exception& e) {
                std::cerr << "Error updating at t=" << data.t
                          << ": " << e.what() << std::endl;
            }
        }

        // Get current state estimate
        State estimate = filter.stateEstimate();

        // Calculate errors
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

        // Show progress dots
        if (i % progressStep == 0) {
            std::cout << "." << std::flush;
        }
    }
    
    std::cout << " Done!" << std::endl;

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

    // Calculate final errors from last data point
    const Data& final_data = data_list.back();
    State final_estimate = filter.stateEstimate();
    Vector3 final_att_error = Rot3::Logmap(final_data.xi.R.between(final_estimate.R));
    Vector3 final_bias_error = final_estimate.b - final_data.xi.b;
    Vector3 final_cal_error = Vector3::Zero();
    if (!final_data.xi.S.empty() && !final_estimate.S.empty()) {
        final_cal_error = Rot3::Logmap(final_data.xi.S[0].between(final_estimate.S[0]));
    }

    // Print summary statistics
    std::cout << "\n=== Filter Performance Summary ===" << std::endl;
    std::cout << "Processing time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Processed measurements: " << totalMeasurements << " (valid: " << validMeasurements << ")" << std::endl;
    
    // Average errors
    std::cout << "\n-- Average Errors --" << std::endl;
    std::cout << "Attitude: " << (avg_att_error * RAD_TO_DEG) << "°" << std::endl;
    std::cout << "Bias: " << avg_bias_error << std::endl;
    std::cout << "Calibration: " << (avg_cal_error * RAD_TO_DEG) << "°" << std::endl;
    
    // Final errors
    std::cout << "\n-- Final Errors --" << std::endl;
    std::cout << "Attitude: " << (final_att_error.norm() * RAD_TO_DEG) << "°" << std::endl;
    std::cout << "Bias: " << final_bias_error.norm() << std::endl;
    std::cout << "Calibration: " << (final_cal_error.norm() * RAD_TO_DEG) << "°" << std::endl;
    
    // Print a brief comparison of final estimate vs ground truth
    std::cout << "\n-- Final State vs Ground Truth --" << std::endl;
    std::cout << "Attitude (RPY) - Estimate: " 
              << (final_estimate.R.rpy() * RAD_TO_DEG).transpose() << "° | Truth: " 
              << (final_data.xi.R.rpy() * RAD_TO_DEG).transpose() << "°" << std::endl;
    std::cout << "Bias - Estimate: " << final_estimate.b.transpose() 
              << " | Truth: " << final_data.xi.b.transpose() << std::endl;
    
    if (!final_estimate.S.empty() && !final_data.xi.S.empty()) {
        std::cout << "Calibration (RPY) - Estimate: " 
                 << (final_estimate.S[0].rpy() * RAD_TO_DEG).transpose() << "° | Truth: " 
                 << (final_data.xi.S[0].rpy() * RAD_TO_DEG).transpose() << "°" << std::endl;
    }
}

} // namespace abc_eqf_lib