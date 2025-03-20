//
// Created by darshan on 3/11/25.
//
#include "EqF.h"
#include "utilities.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <functional>

// Implementation of helper functions

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

State stateAction(const G& X, const State& xi) {
    if (xi.S.size() != X.B.size()) {
        throw std::invalid_argument("Number of calibration states and B elements must match");
    }

    std::vector<Rot3> new_S;
    for (size_t i = 0; i < X.B.size(); i++) {
        new_S.push_back(X.A.inverse() * xi.S[i] * X.B[i]);
    }

    return State(xi.R * X.A,
                X.A.inverse().matrix() * (xi.b - vee(X.a)),
                new_S);
}

Input velocityAction(const G& X, const Input& u) {
    return Input(X.A.inverse().matrix() * (u.w - vee(X.a)), u.Sigma);
}

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

Matrix stateActionDiff(const State& xi) {
    std::function<Vector(const Vector&)> coordsAction =
        [&xi](const Vector& U) {
            return local_coords(stateAction(G::exp(U), xi));
        };

    Vector zeros = Vector::Zero(6 + 3 * xi.S.size());
    Matrix differential = numericalDifferential(coordsAction, zeros);
    return differential;
}

// EqF class implementation

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

State EqF::stateEstimate() const {
    return stateAction(__X_hat, __xi_0);
}

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

bool hasNaN(const Vector3& vec) {
    return std::isnan(vec[0]) || std::isnan(vec[1]) || std::isnan(vec[2]);
}

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
  static int update_count = 0;
  if (update_count < 5) {
    std::cout << "Update " << update_count << ":\n";
    std::cout << "y_vec: " << y_vec.transpose() << "\n";
    std::cout << "d_vec: " << d_vec.transpose() << "\n";
    update_count++;
  }



  Matrix Ct = __measurementMatrixC(y.d, y.cal_idx);
  Vector3 action_result = outputAction(__X_hat.inv(), y.y, y.cal_idx);
  Vector3 delta_vec = wedge(y.d.d.unitVector()) * action_result;  // Ensure this is the right operation
  Matrix Dt = __outputMatrixDt(y.cal_idx);
  Matrix S = Ct * __Sigma * Ct.transpose() + Dt * y.Sigma * Dt.transpose();
  Matrix K = __Sigma * Ct.transpose() * S.inverse();
  Vector Delta = __InnovationLift * K * delta_vec;
  __X_hat = G::exp(Delta) * __X_hat;
  __Sigma = (Matrix::Identity(__dof, __dof) - K * Ct) * __Sigma;
}

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

Matrix EqF::__measurementMatrixC(const Direction& d, int idx) const {
  Matrix Cc = Matrix::Zero(3, 3 * __n_cal);

  // If the measurement is related to a sensor that has a calibration state
  if (idx >= 0) {
    // Cc.block<3, 3>(0, 3 * idx) = wedge(d.d.unitVector()); // WRONG
    // Set the correct 3x3 block in Cc
    Cc.block<3, 3>(0, 3 * idx) = wedge(d.d.unitVector());
  }

  Matrix3 wedge_d = wedge(d.d.unitVector());

  // This Matrix concatenation was different from the Python version
  // Create the equivalent of:
  // Rot3.Hat(d.d.unitVector()) @ np.hstack((Rot3.Hat(d.d.unitVector()), np.zeros((3, 3)), Cc))

  Matrix temp(3, 6 + 3 * __n_cal);
  temp.block<3, 3>(0, 0) = wedge_d;
  temp.block<3, 3>(0, 3) = Matrix3::Zero();
  temp.block(0, 6, 3, 3 * __n_cal) = Cc;

  return wedge_d * temp;
}

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