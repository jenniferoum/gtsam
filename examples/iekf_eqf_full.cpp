/**
 * @file iekf_eqf_framework.cpp
 * @brief Imperfect-IEKF via GTSAM's EquivariantFilter<M, Symmetry>
 *
 * Implements the Imperfect-IEKF (Fornasier §5.3 / §5.8.2) using
 * GTSAM's EquivariantFilter template, NOT the InvariantEKF shortcut.
 *
 * This explicitly defines:
 *   - Manifold M = SE₂(3) × R⁶  (NavState + biases)
 *   - Symmetry group G = SE₂(3) × R⁶  (direct product)
 *   - Group action ϕ(X, ξ) = (T·D, b + δ)       [Lemma 5.3.1, eq. 5.11]
 *   - Lift Λ(ξ, u) = (Λ₁, Λ₂)                    [Theorem 5.3.2, eq. 5.12-5.13]
 *   - Error state matrix A                         [eq. 5.35]
 *
 * This structure makes it trivial to later swap to SD-EqF by:
 *   1. Changing the group product to use the adjoint action
 *   2. Changing the lift Λ₂ to eq. 5.29
 *   3. Changing the A matrix to eq. 5.50
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. && make
 *   ./iekf_eqf ../rocket_sim_data_paper.csv
 *
 * References:
 *   [1] Fornasier, "Equivariant Symmetries for INS", Ch. 5
 *   [2] GTSAM EKF-variants / EquivariantFilter documentation
 */

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/base/ProductLieGroup.h>
#include <gtsam/navigation/EquivariantFilter.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/slam/dataset.h>

#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <map>

using namespace gtsam;
using namespace std;

// ═══════════════════════════════════════════════════════════════════════════════
//  Type Definitions
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * State and symmetry group type.
 *
 * For the Imperfect-IEKF, the state space M and the symmetry group G
 * are both the direct product SE₂(3) × R⁶.
 *
 * GTSAM's ProductLieGroup gives us Compose, Inverse, Expmap, Logmap,
 * AdjointMap — all block-diagonal (no coupling between components).
 *
 * To upgrade to SD-EqF, replace this with a SemiDirectProductGroup
 * that uses the adjoint action in Compose and Inverse.
 */
using NavStateBias = ProductLieGroup<NavState, Vector6>;

// Dimensions
static constexpr int DimM = traits<NavStateBias>::dimension;  // 15

// Compile-time tuning profiles. Override with, for example:
//   add_compile_definitions(IEKF_TUNING_MODE=IEKF_TUNING_MEKF_MATCHED)
// or pass -DIEKF_TUNING_MODE=IEKF_TUNING_MEKF_MATCHED in your target flags.
#define IEKF_TUNING_MEKF_MATCHED 0
#define IEKF_TUNING_REDUCED_STATE 1
#ifndef IEKF_TUNING_MODE
#define IEKF_TUNING_MODE IEKF_TUNING_REDUCED_STATE
#endif

static_assert(IEKF_TUNING_MODE == IEKF_TUNING_MEKF_MATCHED ||
              IEKF_TUNING_MODE == IEKF_TUNING_REDUCED_STATE,
              "Unknown IEKF_TUNING_MODE");

double envDouble(const char* name, double fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;

    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    return (end != value && std::isfinite(parsed)) ? parsed : fallback;
}

bool envFlag(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) return false;
    return string(value) != "0" && string(value) != "false" &&
           string(value) != "FALSE";
}

bool envBool(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    const string text(value);
    if (text == "0" || text == "false" || text == "FALSE") return false;
    if (text == "1" || text == "true" || text == "TRUE") return true;
    return fallback;
}

string envString(const char* name, const string& fallback) {
    const char* value = std::getenv(name);
    return (value && *value) ? string(value) : fallback;
}

int envInt(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;

    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return (end != value) ? static_cast<int>(parsed) : fallback;
}

bool envSet(const char* name) {
    const char* value = std::getenv(name);
    return value && *value;
}

Vector3 envVector3(const string& name, const Vector3& fallback,
                   const string& s0, const string& s1, const string& s2) {
    Vector3 values;
    values << envDouble((name + "_" + s0).c_str(), fallback(0)),
              envDouble((name + "_" + s1).c_str(), fallback(1)),
              envDouble((name + "_" + s2).c_str(), fallback(2));
    return values;
}

Vector3 envVector3(const string& name, double fallback,
                   const string& s0, const string& s1, const string& s2) {
    return envVector3(name, Vector3::Constant(fallback), s0, s1, s2);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Symmetry Definition  (Fornasier §5.3, G_ES = SE₂(3) × R⁶)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * IEKFSymmetry encodes the IEKF group action on the state space.
 *
 * Group action ϕ(X, ξ) = (T·D, b + δ)   [Lemma 5.3.1, eq. 5.11]
 *
 * This is just the ProductLieGroup compose, since the action of the
 * direct product on itself is composition. For a semi-direct product,
 * the Orbit would need to apply the adjoint to the bias component.
 */
struct IEKFSymmetry {
    using Group = NavStateBias;

    /**
     * Orbit: ϕ_ξref(X) = ϕ(X, ξ_ref) = ξ_ref · X
     *
     * For the direct product:
     *   ϕ((D, δ), (T_ref, b_ref)) = (T_ref · D, b_ref + δ)
     *
     * This is right composition on the product group.
     */
    struct Orbit {
        NavStateBias xi_ref;

        explicit Orbit(const NavStateBias& ref) : xi_ref(ref) {}

      NavStateBias operator()(const Group& X,
                              OptionalJacobian<DimM, DimM> H = {}) const {
          if (H) {
              Eigen::Matrix<double, 15, 15> H2;
              auto result = traits<NavStateBias>::Compose(xi_ref, X, {}, H2);
              *H = H2;
              return result;
          }
          return traits<NavStateBias>::Compose(xi_ref, X); // left composition, consistent with Jacobian
      }
    };

    using Diffeomorphism = Orbit;
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Lift Functor  (Fornasier Theorem 5.3.2, eq. 5.12-5.13)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * IEKFLift: Λ(ξ, u) → Lie algebra of G
 *
 * Λ₁(ξ, u) = (W - B + N) + T⁻¹(G - N)T   [eq. 5.12]
 *   — This is the se₂(3) tangent vector that drives the NavState forward.
 *   — In body-frame terms: [ω_corrected, 0, a_corrected + R^T g]
 *
 * Λ₂(ξ, u) = τ                              [eq. 5.13]
 *   — Bias dynamics input. Zero for random-walk bias model.
 *
 * The Lift returns a 15D tangent vector of the product group.
 */
struct IEKFLift {
    Vector3 omega_meas;   // raw gyro measurement
    Vector3 accel_meas;   // raw accelerometer measurement
    Vector3 g_ned;        // gravity in NED frame

    IEKFLift(const Vector3& w, const Vector3& a, const Vector3& g)
        : omega_meas(w), accel_meas(a), g_ned(g) {}

    /**
     * Evaluate the lift at the current state estimate.
     *
     * @param xi  Current state estimate (NavState, bias)
     * @param H   Optional Jacobian (unused for explicit-A path)
     * @return    15D tangent vector [Λ₁(9), Λ₂(6)]
     */
    Eigen::Matrix<double, DimM, 1> operator()(
            const NavStateBias& xi,
            OptionalJacobian<DimM, DimM> H = {}) const {

        // Extract state components
        NavState T = xi.first;
        Vector6 bias = xi.second;
        Rot3 R = T.attitude();
        Vector3 bg = bias.head<3>();
        Vector3 ba = bias.tail<3>();

        // Bias-corrected IMU
        Vector3 omega_c = omega_meas - bg;
        Vector3 accel_c = accel_meas - ba;

        // ── Λ₁: navigation part of the lift ──
        //
        // From eq. 5.12: Λ₁ = (W - B + N) + T⁻¹(G - N)T
        //
        // Breaking this down for the tangent vector [ω, δP, δV]:
        //   ω:  omega_corrected (body-frame angular rate)
        //   δV: accel_corrected + R^T g  (body-frame specific force + gravity)
        //   δP: R^T v  (velocity in body frame, from N matrix for p←v coupling)
        //
        // When we do X̂_{k+1} = X̂_k · exp(Λ₁·dt), SE₂(3) composition gives:
        //   R_{k+1} = R_k · exp(ω·dt)
        //   v_{k+1} = v_k + R_k · δV · dt  = v_k + (R_k·a_c + g)·dt  ✓
        //   p_{k+1} = p_k + R_k · δP · dt  = p_k + v_k·dt             ✓
        //
        // Note: GTSAM NavState tangent order is [δR(3), δP(3), δV(3)]
        Vector3 g_body = R.unrotate(g_ned);       // R^T g
        Vector3 v_body = R.unrotate(T.velocity()); // R^T v (for position dynamics)

        Vector9 lambda1;
        lambda1 << omega_c, v_body, accel_c + g_body;

        // ── Λ₂: bias part of the lift ──
        //
        // From eq. 5.13: Λ₂ = τ
        // For random-walk bias model, τ = 0.
        Vector6 lambda2 = Vector6::Zero();

        // Combine into product group tangent vector
        Eigen::Matrix<double, DimM, 1> Lambda;
        Lambda << lambda1, lambda2;

        if (H) {
            H->setZero();
            // For the explicit-A path, this Jacobian is not used
            // (we provide A directly). Left zero.
        }

        return Lambda;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Error State Matrix A  (Fornasier eq. 5.35)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Build the Imperfect-IEKF error state matrix.
 *
 * Error state ordering: [δθ(3), δv(3), δp(3), δbω(3), δba(3)]
 *
 * NOTE: GTSAM NavState tangent order is [δR(3), δP(3), δV(3)].
 * We must account for this when placing blocks. The A matrix from
 * the paper uses [δR, δv, δp] order, so we rearrange accordingly.
 *
 * Paper ordering:  [δR, δv, δp, δbω, δba]   (indices 0-2, 3-5, 6-8, 9-11, 12-14)
 * GTSAM ordering:  [δR, δp, δv, δbω, δba]   (indices 0-2, 3-5, 6-8, 9-11, 12-14)
 *
 * Paper eq. 5.35:
 *          δR      δv      δp     δbω     δba
 *   δR  [  0       0       0     -R̂       0   ]
 *   δv  [ g×       0       0   -v̂×R̂    -R̂   ]
 *   δp  [  0       I       0   -p̂×R̂     0   ]
 *   δbω [  0       0       0      0       0   ]
 *   δba [  0       0       0      0       0   ]
 *
 * Rearranged for GTSAM [δR, δp, δv, δbω, δba]:
 *          δR      δp      δv     δbω     δba
 *   δR  [  0       0       0     -R̂       0   ]
 *   δp  [  0       0       I   -p̂×R̂     0   ]
 *   δv  [ g×       0       0   -v̂×R̂    -R̂   ]
 *   δbω [  0       0       0      0       0   ]
 *   δba [  0       0       0      0       0   ]
 */
// Eigen::Matrix<double, DimM, DimM> buildA_IEKF(
//         const NavStateBias& xi_hat,
//         const Vector3& g_ned) {

//     NavState T_hat = xi_hat.first;
//     Matrix3 Rh = T_hat.attitude().matrix();
//     Vector3 vh = T_hat.velocity();
//     Vector3 ph = T_hat.position();

//     Eigen::Matrix<double, DimM, DimM> A;
//     A.setZero();

//     // In GTSAM ordering: [δR(0:2), δP(3:5), δV(6:8), δbω(9:11), δba(12:14)]

//     // ── δR row (0:2) ──
//     // δR ← δbω:  -R̂
//     A.block<3,3>(0, 9) = -Rh;

//     // ── δP row (3:5)  [this was δp in the paper] ──
//     // δP ← δV:   I    [paper: δp ← δv]
//     A.block<3,3>(3, 6) = Matrix3::Identity();
//     // δP ← δbω: -p̂× R̂
//     A.block<3,3>(3, 9) = -skewSymmetric(ph) * Rh;

//     // ── δV row (6:8)  [this was δv in the paper] ──
//     // δV ← δR:   g×
//     A.block<3,3>(6, 0) = skewSymmetric(g_ned);
//     // δV ← δbω: -v̂× R̂
//     A.block<3,3>(6, 9) = -skewSymmetric(vh) * Rh;
//     // δV ← δba: -R̂
//     A.block<3,3>(6, 12) = -Rh;

//     // ── δbω, δba rows (9:14) ──
//     // All zero (random walk bias model)

//     return A;
// }

// ═══════════════════════════════════════════════════════════════════════════════
//  A MATRIX MODE SELECTOR — change this to test different theories
// ═══════════════════════════════════════════════════════════════════════════════
//
//  MODE 0: Full IEKF (eq. 5.35) — has -v̂×R̂ and -p̂×R̂ terms
//          Expected: DIVERGES on rocket (||A||·dt >> 1)
//
//  MODE 1: IEKF with -v̂×R̂ and -p̂×R̂ zeroed out
//          Expected: WORKS — proves those terms cause divergence
//
//  MODE 2: Full MEKF (eq. 5.32) — -(R̂(a-b̂a))× instead of g×
//          Expected: WORKS — baseline comparison
//
static int A_MATRIX_MODE = 1;  // runtime-overridable with IEKF_A_MODE

Eigen::Matrix<double, DimM, DimM> buildA_IEKF(
        const NavStateBias& xi_hat,
        const Vector3& g_ned,
        const Vector3& accel_meas = Vector3::Zero()) {

    NavState T_hat = xi_hat.first;
    Matrix3 Rh = T_hat.attitude().matrix();
    Vector3 vh = T_hat.velocity();
    Vector3 ph = T_hat.position();
    Vector6 bias = xi_hat.second;
    Vector3 ba_hat = bias.tail<3>();

    Eigen::Matrix<double, DimM, DimM> A;
    A.setZero();

    // GTSAM ordering: [δR(0:2), δP(3:5), δV(6:8), δbω(9:11), δba(12:14)]

    if (A_MATRIX_MODE == 0) {
        // ── MODE 0: Full IEKF (eq. 5.35) ──
        // δR ← δbω:  -R̂
        A.block<3,3>(0, 9) = -Rh;
        // δP ← δV: I
        A.block<3,3>(3, 6) = Matrix3::Identity();
        // δP ← δbω: -p̂× R̂          ← GROWS WITH POSITION
        A.block<3,3>(3, 9) = -skewSymmetric(ph) * Rh;
        // δV ← δR: g×
        A.block<3,3>(6, 0) = skewSymmetric(g_ned);
        // δV ← δbω: -v̂× R̂          ← GROWS WITH VELOCITY
        A.block<3,3>(6, 9) = -skewSymmetric(vh) * Rh;
        // δV ← δba: -R̂
        A.block<3,3>(6, 12) = -Rh;

    } else if (A_MATRIX_MODE == 1) {
        // ── MODE 1: IEKF with dangerous terms zeroed ──
        // Same as Mode 0 but WITHOUT -v̂×R̂ and -p̂×R̂
        // If this works → those terms are the cause of divergence
        A.block<3,3>(0, 9) = -Rh;
        A.block<3,3>(3, 6) = Matrix3::Identity();
        // A.block<3,3>(3, 9) = ZERO   ← removed -p̂×R̂
        A.block<3,3>(6, 0) = skewSymmetric(g_ned);
        // A.block<3,3>(6, 9) = ZERO   ← removed -v̂×R̂
        A.block<3,3>(6, 12) = -Rh;

    } else if (A_MATRIX_MODE == 2) {
        // ── MODE 2: MEKF (eq. 5.32) ──
        // Uses -(R̂(a-b̂a))× instead of g×, no bias-nav coupling
        A.block<3,3>(0, 9) = -Rh;
        A.block<3,3>(3, 6) = Matrix3::Identity();
        // No position-bias coupling
        A.block<3,3>(6, 0) = -skewSymmetric(Rh * (accel_meas - ba_hat));
        // No velocity-bias coupling  
        A.block<3,3>(6, 12) = -Rh;
    }

    return A;
}

/**
 * For reference / future use: SD-EqF A matrix (eq. 5.50)
 *
 * To upgrade to SD-EqF, replace buildA_IEKF with this function
 * and change the symmetry group to use semi-direct product.
 *
 * GTSAM ordering [δR, δP, δV, δbω, δba]:
 *          δR      δP      δV     δbω     δba
 *   δR  [  0       0       0       I       0   ]  ← no R̂!
 *   δP  [  0       0       I     p̂×       0   ]
 *   δV  [ g×       0       0       0       I   ]  ← no R̂!
 *   δbω [              ad(...)                 ]
 *   δba [              ad(...)                 ]
 */
// Eigen::Matrix<double, DimM, DimM> buildA_SDEqF(
//         const NavStateBias& xi_hat,
//         const Vector3& g_ned,
//         const Vector3& omega_origin,
//         const Vector3& accel_origin) {
//     ...
// }

// ═══════════════════════════════════════════════════════════════════════════════
//  Measurement Models
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * GPS position measurement model.
 *
 * h(ξ) = position component of NavState
 *
 * Jacobian w.r.t. NavStateBias local coordinates:
 *   H = [0₃ₓ₃, I₃, 0₃ₓ₃, 0₃ₓ₆]
 *   (GTSAM NavState order: [δR, δP, δV], then [δbias])
 */
// Vector3 h_gps(const NavStateBias& xi, OptionalJacobian<3, DimM> H = {}) {
//     if (H) {
//         H->setZero();
//         // Position is the δP block (indices 3:5 in GTSAM ordering)
//         //H->block<3,3>(0, 3) = Matrix3::Identity();
//         H->block<3,3>(0, 3) = xi.first.attitude().matrix();  // R̂, not I!
//     }
//     return xi.first.position();
// }
Vector3 h_gps(const NavStateBias& xi, OptionalJacobian<3, DimM> H = {}) {
    if (H) {
        H->setZero();
        H->block<3,3>(0, 3) = Matrix3::Identity();  // GTSAM retract: p⁺ = p - δP
        //H->block<3,3>(0, 3) = xi.first.attitude().matrix().transpose(); // Rᵀ maps NED→body
    }
    return xi.first.position();
}

/**
 * Example-local EqF wrapper for the rocket-club GPS comparison.
 *
 * Fornasier's GPS/GNSS model observes position, not attitude. The generic
 * Kalman correction can still rotate the attitude through P(attitude, position)
 * cross-covariance. The rocket-club MEKF does not apply that GPS attitude
 * correction, so this wrapper projects out only the GPS-to-attitude and
 * GPS-to-gyro-bias coupling before the position measurement update.
 *
 * Position and velocity are still corrected through the position and
 * velocity-position covariance blocks. Accelerometer-bias correction is
 * configurable, because the rocket-club MEKF does allow GPS to correct ba.
 */
class RocketIEKF : public EquivariantFilter<NavStateBias, IEKFSymmetry> {
 public:
    using Base = EquivariantFilter<NavStateBias, IEKFSymmetry>;
    using Base::Base;

    void suppressGpsForbiddenCoupling(bool gps_updates_accel_bias) {
        // GPS H reads only the position block, so K(:, gps) depends on P(:, pos).
        // Zero these rows/columns to keep GPS from changing attitude/gyro bias.
        this->P_.template block<3,3>(0, 3).setZero();
        this->P_.template block<3,3>(3, 0).setZero();

        this->P_.template block<3,3>(9, 3).setZero();
        this->P_.template block<3,3>(3, 9).setZero();

        if (!gps_updates_accel_bias) {
            this->P_.template block<3,3>(12, 3).setZero();
            this->P_.template block<3,3>(3, 12).setZero();
        }
    }

    void updateGpsPositionOnly(const Vector3& z_gps, const Matrix3& R_gps,
                               bool gps_updates_accel_bias) {
        suppressGpsForbiddenCoupling(gps_updates_accel_bias);
        this->update(h_gps, z_gps, R_gps);
        suppressGpsForbiddenCoupling(gps_updates_accel_bias);
    }
};
// Vector3 h_gps(const NavStateBias& xi, OptionalJacobian<3, DimM> H = {}) {
//     if (H) {
//         H->setZero();
//         H->block<3,3>(0, 3) = Matrix3::Identity();
//     }
//     return xi.first.position();
// }

// Matrix h_gps(const NavState& state, const Vector6& bias) {
//     Matrix H = Matrix::Zero(3, 15);
    
//     // For local tangent space, the derivative of global position 
//     // with respect to local position error is the Rotation matrix.
//     H.block<3,3>(0,6) = state.attitude().matrix(); 
    
//     return H;
// }

/**
 * Magnetometer measurement model.
 *
 * h(ξ) = R^T · m_NED   (expected body-frame mag reading)
 *
 * Jacobian: only the rotation block contributes
 *   δh = -skew(R^T m_NED) · δR   (body-frame perturbation)
 */
struct MagModel {
    Vector3 mag_ned;

    MagModel(const Vector3& m) : mag_ned(m) {}

    Vector3 operator()(const NavStateBias& xi,
                       OptionalJacobian<3, DimM> H = {}) const {
        Rot3 R = xi.first.attitude();
        Vector3 mag_body = R.unrotate(mag_ned);

        if (H) {
            H->setZero();
            // δh/δR = +skew(mag_body) for right perturbation
            // h(R·Exp(δθ)) = Exp(-δθ)·R^T·m ≈ mag_body + [mag_body]× · δθ
            H->block<3,3>(0, 0) = skewSymmetric(mag_body);
        }
        return mag_body;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  CSV Reader & Coordinate Utilities
// ═══════════════════════════════════════════════════════════════════════════════

struct SensorSample {
    double t;
    Vector3 acc, gyro, gps_lla, magB, ned_true, vel_true;
    Vector3 ba_true, bg_true;
    Vector4 q_true;
};

vector<SensorSample> loadCSV(const string& file,
                              Vector3& lla0, Vector3& magI) {
    ifstream f(file);
    if (!f.is_open()) throw runtime_error("Cannot open: " + file);

    string line;
    getline(f, line);
    istringstream hdr(line);
    string col;
    map<string,int> idx;
    int i = 0;
    while (getline(hdr, col, ',')) {
        col.erase(0, col.find_first_not_of(" \t\r\n"));
        col.erase(col.find_last_not_of(" \t\r\n") + 1);
        idx[col] = i++;
    }

    vector<SensorSample> data;
    bool first = true;
    while (getline(f, line)) {
        if (line.empty()) continue;
        istringstream ss(line);
        vector<double> v;
        string tok;
        while (getline(ss, tok, ',')) {
            tok.erase(0, tok.find_first_not_of(" \t\r\n"));
            tok.erase(tok.find_last_not_of(" \t\r\n") + 1);
            v.push_back(tok.empty() || tok == "nan" || tok == "NaN"
                        ? NAN : stod(tok));
        }
        SensorSample s;
        s.t        = v[idx["time"]];
        s.acc      = Vector3(v[idx["acc_x"]],  v[idx["acc_y"]],  v[idx["acc_z"]]);
        s.gyro     = Vector3(v[idx["gyro_x"]], v[idx["gyro_y"]], v[idx["gyro_z"]]);
        s.gps_lla  = Vector3(v[idx["GPS_x"]],  v[idx["GPS_y"]],  v[idx["GPS_z"]]);
        s.magB     = Vector3(v[idx["magB_x"]], v[idx["magB_y"]], v[idx["magB_z"]]);
        s.ned_true = Vector3(v[idx["NED_x"]],  v[idx["NED_y"]],  v[idx["NED_z"]]);
        s.vel_true = Vector3(v[idx["velI_x"]], v[idx["velI_y"]], v[idx["velI_z"]]);
        s.ba_true  = Vector3(v[idx["ba_x"]],   v[idx["ba_y"]],   v[idx["ba_z"]]);
        s.bg_true  = Vector3(v[idx["bg_x"]],   v[idx["bg_y"]],   v[idx["bg_z"]]);
        s.q_true   = Vector4(v[idx["q_q1"]], v[idx["q_q2"]],
                             v[idx["q_q3"]], v[idx["q_q4"]]);
        if (first) {
            lla0 = Vector3(v[idx["LLA0_x"]], v[idx["LLA0_y"]], v[idx["LLA0_z"]]);
            magI = Vector3(v[idx["magI_x"]], v[idx["magI_y"]], v[idx["magI_z"]]);
            first = false;
        }
        data.push_back(s);
    }
    return data;
}

Vector3 lla2ecef(const Vector3& lla) {
    const double a = 6378137.0, e = 0.08181919;
    double pr = lla(0)*M_PI/180, lr = lla(1)*M_PI/180, h = lla(2);
    double sp = sin(pr), cp = cos(pr), sl = sin(lr), cl = cos(lr);
    double N = a / sqrt(1-(e*sp)*(e*sp));
    return Vector3((N+h)*cp*cl, (N+h)*cp*sl, (N*(1-e*e)+h)*sp);
}

Vector3 lla2ned(const Vector3& lla, const Vector3& lla0) {
    Vector3 e1 = lla2ecef(lla), e0 = lla2ecef(lla0);
    double pr = lla0(0)*M_PI/180, lr = lla0(1)*M_PI/180;
    double s1=sin(pr), c1=cos(pr), s2=sin(lr), c2=cos(lr);
    Matrix3 R;
    R << -s1*c2, -s1*s2, c1, -s2, c2, 0, -c1*c2, -c1*s2, -s1;
    return R * (e1 - e0);
}

Vector3 gravityNED(double lat_deg, double alt) {
    double phi = lat_deg*M_PI/180;
    double s2 = sin(phi)*sin(phi), s2l = sin(2*phi)*sin(2*phi);
    double g = 9.780327*(1+5.3024e-3*s2-5.8e-6*s2l)
               -(3.0877e-6-4.4e-9*s2)*alt+7.2e-14*alt*alt;
    return Vector3(0, 0, g);
}

// Quaternion utilities for error computation
Vector4 rot3ToQuat(const Rot3& R) {
    auto q = R.toQuaternion();
    return Vector4(q.w(), q.x(), q.y(), q.z());
}

double attError(const Vector4& qt, const Vector4& qe) {
    double w1=qt(0),x1=qt(1),y1=qt(2),z1=qt(3);
    double w2=qe(0),x2=-qe(1),y2=-qe(2),z2=-qe(3);  // conjugate of qe
    Vector4 qerr(w1*w2-x1*x2-y1*y2-z1*z2,
                  w1*x2+x1*w2+y1*z2-z1*y2,
                  w1*y2-x1*z2+y1*w2+z1*x2,
                  w1*z2+x1*y2-y1*x2+z1*w2);
    return 2.0*atan2(qerr.tail<3>().norm(), abs(qerr(0)))*180.0/M_PI;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    //string csv = findExampleDataFile("rocket_sim_data_paper.csv");
    string csv = findExampleDataFile("rocket_combined_data_410s.csv");
    //string csv = findExampleDataFile("rocket_sim_data_paper_lower_dyn.csv");
    if (argc > 1) csv = argv[1];

    cout << "================================================" << endl;
    cout << " IEKF via EquivariantFilter<NavStateBias, IEKF>  " << endl;
    cout << " Fornasier §5.3 / §5.8.2 — eq. 5.11, 5.12, 5.35" << endl;
    cout << "================================================" << endl;

    // ── Load data ──
    Vector3 lla0, magI;
    auto data = loadCSV(csv, lla0, magI);
    int N = data.size();
    cout << "Loaded " << N << " samples" << endl;

    // ── Constants ──
    double dt = 1.0 / 100.0;
    // Vector3 g_ned = gravityNED(lla0(0), lla0(2));
    Vector3 g_ned(0, 0, 9.79571); // Hardcoded to match Python mekf result
    cout << "Gravity: " << g_ned.transpose() << endl;

    // Convert the MEKF's latitude/longitude covariance units into local NED
    // meters so R and P0 are comparable to this IEKF state.
    const double lat0r = lla0(0) * M_PI / 180.0;
    const double sin_lat0 = sin(lat0r);
    const double a_e = 6378137.0;
    const double e2 = 0.00669437999014;
    const double denom = 1.0 - e2 * sin_lat0 * sin_lat0;
    const double Rphi = a_e * (1.0 - e2) / pow(denom, 1.5);
    const double Rlam = a_e / sqrt(denom);
    const double mpd_lat = Rphi * M_PI / 180.0;
    const double mpd_lon = Rlam * cos(lat0r) * M_PI / 180.0;

    // ── Calibration ──
    // Data is already in m/s² and rad/s for this dataset.
    // Biases start at zero and converge online (matching MEKF convention).
    // double accel_sf = 1.0;
    // double gyro_sf = 1.0;
    cout << "Accel/gyro scale factor: 1.0 (data in m/s², rad/s)" << endl;
    cout << "Initial biases: zero (converge online)" << endl;

    // ── Noise ──
    // Process noise must account for the scale factor approximation error.
    // double sg = 12e-3*M_PI/180, /*sa = 200e-6*9.81,*/ scale = 10.0;

    // Eigen::Matrix<double, DimM, DimM> Qc;
    // Qc.setZero();
    // Qc.block<3,3>(0,0) = scale*sg*sg * Matrix3::Identity();          // attitude
    // Qc.block<3,3>(3,3) = 0.1 * Matrix3::Identity();                  // position
    // Qc.block<3,3>(6,6) = 10.0 * Matrix3::Identity();                  // velocity
    // Qc.block<3,3>(9,9) = scale*pow(3.0/3600*M_PI/180, 2) * Matrix3::Identity();   // gyro bias
    // Qc.block<3,3>(12,12) = scale*pow(40e-6*9.8, 2) * Matrix3::Identity();         // accel bias

    // ── Noise ──
    // MEKF uses Q = 10 * dt * diag(sigma_gv, sigma_gu, sigma_av, sigma_au)
    // Note: sigma, NOT sigma². This is the MEKF's convention.
    // GTSAM applies Qc as Q_discrete = Qc * dt, so Qc = Q_d / dt = 10 * sigma.
    const double scale = 10.0;
    const double sg  = 12e-3 * M_PI / 180.0;             // gyro velocity noise
    [[maybe_unused]] const double sbg = (3.0 / 3600.0) * M_PI / 180.0; // gyro bias random walk
    const double sa  = 200e-6 * 9.81;                    // accel velocity noise
    const double sba = 40e-6 * 9.8;                      // accel bias random walk

    string tuning_mode_name;
    Eigen::Matrix<double, DimM, DimM> Qc;
    Qc.setZero();
#if IEKF_TUNING_MODE == IEKF_TUNING_MEKF_MATCHED
    tuning_mode_name = "MEKF-matched covariance tuning";
    Qc.block<3,3>(0,0)   = scale * sg  * Matrix3::Identity();
    Qc.block<3,3>(3,3)   = Matrix3::Zero();
    Qc.block<3,3>(6,6)   = scale * sa  * Matrix3::Identity();
    Qc.block<3,3>(9,9)   = scale * sbg * Matrix3::Identity();
    Qc.block<3,3>(12,12) = scale * sba * Matrix3::Identity();
#else
    tuning_mode_name = "reduced-state IEKF tuning";
    // Keep the same GPS sensor R as the MEKF by default, but tune process
    // covariance for this 15-state IEKF, which has no explicit scale-factor states.
    const double q_att_base = envDouble("IEKF_Q_ATT", scale * sg * sg);
    const double q_pos_base = envDouble("IEKF_Q_POS", 1e-8);
    const double q_vel_base = envDouble("IEKF_Q_VEL", scale * sa * sa);
    const double q_bg_base  = envDouble("IEKF_Q_BG", 1e-14);
    const double q_ba_base  = envDouble("IEKF_Q_BA", scale * sba * sba);
    Vector3 q_ba_default = Vector3::Constant(q_ba_base);
    if (!envSet("IEKF_Q_BA")) q_ba_default(2) = 1e-4;

    const Vector3 q_att = envVector3("IEKF_Q_ATT", q_att_base, "X", "Y", "Z");
    const Vector3 q_pos = envVector3("IEKF_Q_POS", q_pos_base, "N", "E", "D");
    const Vector3 q_vel = envVector3("IEKF_Q_VEL", q_vel_base, "N", "E", "D");
    const Vector3 q_bg  = envVector3("IEKF_Q_BG",  q_bg_base,  "X", "Y", "Z");
    const Vector3 q_ba  = envVector3("IEKF_Q_BA",  q_ba_default, "X", "Y", "Z");

    Qc.block<3,3>(0,0)   = q_att.asDiagonal();
    Qc.block<3,3>(3,3)   = q_pos.asDiagonal();
    Qc.block<3,3>(6,6)   = q_vel.asDiagonal();
    Qc.block<3,3>(9,9)   = q_bg.asDiagonal();
    Qc.block<3,3>(12,12) = q_ba.asDiagonal();
#endif


    // GPS noise (LLA → NED conversion)
    // double lat0r = lla0(0)*M_PI/180;
    // double a_e = 6378137.0, e2 = 0.00669437999014;
    // double Rphi = a_e*(1-e2)/pow(1-e2*sin(lat0r)*sin(lat0r), 1.5);
    // double Rlam = a_e/sqrt(1-e2*sin(lat0r)*sin(lat0r));
    // double mpd_lat = Rphi*M_PI/180, mpd_lon = Rlam*cos(lat0r)*M_PI/180;

    Matrix3 R_gps;
    // Default GPS covariance is the MEKF sensor model converted from LLA to NED.
    // IEKF_R_N/E/D and IEKF_R_SCALE let us test IEKF-specific measurement tuning.
    const double default_r_gps_n = 1.35e-5 * mpd_lat * mpd_lat;
    const double default_r_gps_e = 1.65e-5 * mpd_lon * mpd_lon;
    const double default_r_gps_d = 2.0;
    const double r_gps_scale = envDouble("IEKF_R_SCALE", 1.0);
    const double r_gps_n = envDouble("IEKF_R_N", default_r_gps_n * r_gps_scale);
    const double r_gps_e = envDouble("IEKF_R_E", default_r_gps_e * r_gps_scale);
    const double r_gps_d = envDouble("IEKF_R_D", default_r_gps_d * r_gps_scale);
    R_gps << r_gps_n, 0, 0,
             0, r_gps_e, 0,
             0, 0, r_gps_d;


    // Matrix3 R_mag;
    // R_mag << 1e-3, 0, 0, 0, 1e-3, 0, 0, 0, 1e-3;
    Matrix3 R_mag;
    R_mag << 3.2e-7, 0, 0, 0, 4.1e-7, 0, 0, 0, 3.2e-7;

    // ── Initialize EquivariantFilter ──
    //
    // Use true initial attitude from data (the rocket starts on the pad,
    // so R0 aligns body frame with NED). If we use identity, the first
    // gravity rotation is wrong and the filter diverges immediately.
    Vector4 q0 = data[0].q_true;  // [w, x, y, z]
    Rot3 R0 = Rot3::Quaternion(q0(0), q0(1), q0(2), q0(3));
    NavState nav0(R0, Point3(0,0,0), Vector3(0,0,0));
    // Bias initialization modes:
    //   zero  = realistic unknown-bias baseline.
    //   truth = oracle/debug ablation using CSV truth columns.
    const string bias_init_mode = envString("IEKF_BIAS_INIT", "zero");
    Vector6 bias0 = Vector6::Zero();
    if (bias_init_mode == "truth" || bias_init_mode == "TRUE" ||
        bias_init_mode == "1") {
        bias0 << data[0].bg_true, data[0].ba_true;
    } else if (bias_init_mode != "zero") {
        cerr << "Unknown IEKF_BIAS_INIT='" << bias_init_mode
             << "', using zero" << endl;
    }
    cout << "Bias init mode: " << bias_init_mode
         << "  bg0: " << (bias0.head<3>()*180/M_PI).transpose() << " deg/s"
         << "  ba0: " << bias0.tail<3>().transpose() << " m/s²" << endl;

    NavStateBias xi_ref(NavState(), Vector6::Zero());  // origin state (identity)
    NavStateBias X0(nav0, bias0);                       // initial group estimate

    Eigen::Matrix<double, DimM, DimM> P0;
    // P0.setZero();
    // P0.block<3,3>(0,0) = 0.01 * Matrix3::Identity();   // attitude (rad²)
    // P0.block<3,3>(3,3) = 1.0 * Matrix3::Identity();    // position (m²)
    // P0.block<3,3>(6,6) = 0.1 * Matrix3::Identity();    // velocity (m/s)²
    // P0.block<3,3>(9,9) = 1e-4 * Matrix3::Identity();   // gyro bias
    // P0.block<3,3>(12,12) = 1e-2 * Matrix3::Identity(); // accel bias


    P0.setZero();
    // P0.block<3,3>(0,0)   = 0.01  * Matrix3::Identity();  // attitude
    // P0.block<3,3>(3,3)   = 1.0   * Matrix3::Identity();  // try: maybe this is velocity
    // P0.block<3,3>(6,6)   = 10.0  * Matrix3::Identity();  // try: maybe this is position
    // P0.block<3,3>(9,9)   = 1e-4  * Matrix3::Identity();  // gyro bias
    // P0.block<3,3>(12,12) = 1e-2  * Matrix3::Identity();  // accel bias

    // P0.block<3,3>(0,0)   = 1e-4  * Matrix3::Identity();  // attitude
    // P0.block<3,3>(3,3)   = 16.0  * Matrix3::Identity();  // position
    // P0.block<3,3>(6,6)   = 1.0   * Matrix3::Identity();  // velocity
    // P0.block<3,3>(9,9)   = 1e-8  * Matrix3::Identity();  // gyro bias
    // P0.block<3,3>(12,12) = 1e-6  * Matrix3::Identity();  // accel bias

    // P0.block<3,3>(9,9)   = 1e-12 * Matrix3::Identity();
    // P0.block<3,3>(12,12) = 1e-12 * Matrix3::Identity();

#if IEKF_TUNING_MODE == IEKF_TUNING_MEKF_MATCHED
    P0.block<3,3>(0,0)   = 1e-4 * Matrix3::Identity();  // attitude
    P0(3,3) = 1e-4 * mpd_lat * mpd_lat;                  // north position
    P0(4,4) = 1e-4 * mpd_lon * mpd_lon;                  // east position
    P0(5,5) = 1e-4;                                      // down/altitude
    P0.block<3,3>(6,6)   = 1e-4 * Matrix3::Identity();  // velocity
    P0.block<3,3>(9,9)   = 1e-4 * Matrix3::Identity();  // gyro bias
    P0.block<3,3>(12,12) = 1e-4 * Matrix3::Identity();  // accel bias
#else
    // The IEKF starts at the known launch-pad NED origin, so its initial N/E
    // position uncertainty can be much tighter than the MEKF's LLA-degree P0.
    // The default accel-bias Z covariance/noise is intentionally higher for
    // this reduced 15-state model; it absorbs unmodeled IMU scale-factor error
    // without using innovation clipping or leaking GPS directly into attitude.
    const double p_att_base = envDouble("IEKF_P_ATT", 1e-4);
    const double p_pos_base = envDouble("IEKF_P_POS", 16.0);
    const double p_vel_base = envDouble("IEKF_P_VEL", 1.0);
    const double p_bg_base  = envDouble("IEKF_P_BG", 1e-12);
    const double p_ba_base  = envDouble("IEKF_P_BA", 1e-4);
    Vector3 p_ba_default = Vector3::Constant(p_ba_base);
    if (!envSet("IEKF_P_BA")) p_ba_default(2) = 1e-2;

    const Vector3 p_att = envVector3("IEKF_P_ATT", p_att_base, "X", "Y", "Z");
    const Vector3 p_pos = envVector3("IEKF_P_POS", p_pos_base, "N", "E", "D");
    const Vector3 p_vel = envVector3("IEKF_P_VEL", p_vel_base, "N", "E", "D");
    const Vector3 p_bg  = envVector3("IEKF_P_BG",  p_bg_base,  "X", "Y", "Z");
    const Vector3 p_ba  = envVector3("IEKF_P_BA",  p_ba_default, "X", "Y", "Z");

    P0.block<3,3>(0,0)   = p_att.asDiagonal();  // attitude
    P0.block<3,3>(3,3)   = p_pos.asDiagonal();  // position
    P0.block<3,3>(6,6)   = p_vel.asDiagonal();  // velocity
    P0.block<3,3>(9,9)   = p_bg.asDiagonal();   // gyro bias
    P0.block<3,3>(12,12) = p_ba.asDiagonal();   // accel bias
#endif


    // P0.setIdentity();
    // P0 *= 1e-4; // Match Python P0 initialization

    cout << "\nInitial attitude (q0): " << q0.transpose() << endl;
    cout << "Initial R0:\n" << R0.matrix() << endl;

    RocketIEKF ekf(xi_ref, P0, X0);

    MagModel mag_model(magI);

    // ── Storage ──
    vector<double> t_out(N), att_err(N);
    vector<Vector3> pos_out(N), vel_out(N), pos_err(N), vel_err(N);

    // ── Filter Loop ──
    A_MATRIX_MODE = envInt("IEKF_A_MODE", A_MATRIX_MODE);
    cout << "\nRunning EqF (IEKF symmetry) ..." << endl;
    cout << "A_MATRIX_MODE = " << A_MATRIX_MODE;
    if (A_MATRIX_MODE == 0) cout << " (Full IEKF — eq. 5.35, has -v̂×R̂, -p̂×R̂)";
    if (A_MATRIX_MODE == 1) cout << " (IEKF without -v̂×R̂, -p̂×R̂ terms)";
    if (A_MATRIX_MODE == 2) cout << " (MEKF — eq. 5.32)";
    cout << endl;
    cout << "IEKF_TUNING_MODE = " << IEKF_TUNING_MODE
         << " (" << tuning_mode_name << ")" << endl;
#if IEKF_TUNING_MODE == IEKF_TUNING_REDUCED_STATE
    cout << "Reduced IEKF Q diag: att=" << q_att.transpose()
         << " pos=" << q_pos.transpose()
         << " vel=" << q_vel.transpose()
         << " bg=" << q_bg.transpose()
         << " ba=" << q_ba.transpose() << endl;
    cout << "Reduced IEKF P0 diag: att=" << p_att.transpose()
         << " pos=" << p_pos.transpose()
         << " vel=" << p_vel.transpose()
         << " bg=" << p_bg.transpose()
         << " ba=" << p_ba.transpose() << endl;
#endif

    const bool verbose = !envFlag("IEKF_QUIET");

    const bool ENABLE_GPS = envBool("IEKF_ENABLE_GPS", true);
    const bool ENABLE_MAG = envBool("IEKF_ENABLE_MAG", false);
    const bool GPS_POSITION_ONLY_UPDATE = envBool("IEKF_GPS_POSITION_ONLY", true);
    const bool GPS_UPDATE_ACCEL_BIAS = envBool("IEKF_GPS_UPDATE_ACCEL_BIAS", true);
    const bool GPS_LIMIT_INNOVATION = envBool("IEKF_GPS_LIMIT_INNOVATION", false);
    const double MAX_GPS_INNOV = envDouble("IEKF_MAX_GPS_INNOV", 750.0);

    cout << "GPS R diag: N=" << r_gps_n
         << " E=" << r_gps_e
         << " D=" << r_gps_d << endl;
    cout << "GPS update: "
         << (GPS_POSITION_ONLY_UPDATE
                ? "position-only (no GPS attitude/gyro-bias correction)"
                : "full EqF correction")
         << endl;
    cout << "GPS innovation limiting: "
         << (GPS_LIMIT_INNOVATION ? "enabled" : "disabled")
         << endl;
    cout << "GPS accel-bias update: "
         << (GPS_UPDATE_ACCEL_BIAS ? "enabled" : "disabled")
         << endl;

    for (int i = 0; i < N; i++) {
        const auto& d = data[i];

        // Current state estimate
        NavStateBias xi_hat = ekf.state();

        // ── DEBUG (first 5 steps + every 1000) ──
        // if (verbose && (i < 5 || i % 1000 == 0)) {
        //     NavState nav_dbg = xi_hat.first;
        //     Vector6 bias_dbg = xi_hat.second;
        //     Vector3 gyro_s = d.gyro / gyro_sf;
        //     Vector3 accel_s = d.acc / accel_sf;
        //     Vector3 omega_c_dbg = gyro_s - bias_dbg.head<3>();
        // ── DEBUG (first 5 steps + every 1000) ──
        if (verbose && (i < 5 || i % 1000 == 0)) {
            NavState nav_dbg = xi_hat.first;
            Vector6 bias_dbg = xi_hat.second;
            
            // Remove / gyro_sf and / accel_sf here!
            Vector3 gyro_s = d.gyro; 
            Vector3 accel_s = d.acc;
            
            Vector3 omega_c_dbg = gyro_s - bias_dbg.head<3>();
            Vector3 accel_c_dbg = accel_s - bias_dbg.tail<3>();
            Vector3 g_body_dbg = nav_dbg.attitude().unrotate(g_ned);
            cout << "\n--- Step " << i << " t=" << d.t << " ---" << endl;
            cout << "  att_err: " << attError(d.q_true, rot3ToQuat(nav_dbg.attitude())) << " deg" << endl;
            cout << "  pos: " << nav_dbg.position().transpose() << "  truth: " << d.ned_true.transpose() << endl;
            cout << "  vel: " << nav_dbg.velocity().transpose() << "  truth: " << d.vel_true.transpose() << endl;
            cout << "  bias_g: " << (bias_dbg.head<3>()*180/M_PI).transpose() << " deg/s" << endl;
            cout << "  bias_a: " << bias_dbg.tail<3>().transpose() << " m/s²" << endl;
            cout << "  accel_scaled: " << accel_s.transpose() << "  omega_scaled: " << gyro_s.transpose() << endl;
            cout << "  omega_c: " << omega_c_dbg.transpose() << endl;
            cout << "  accel_c: " << accel_c_dbg.transpose() << endl;
            cout << "  g_body:  " << g_body_dbg.transpose() << endl;
            cout << "  accel_c+g_body: " << (accel_c_dbg + g_body_dbg).transpose() << endl;
        }

        // ── BUILD LIFT ──
        // Apply scale factors to raw IMU before passing to lift
        // Vector3 gyro_scaled = d.gyro / gyro_sf;
        // Vector3 accel_scaled = d.acc / accel_sf;
        // IEKFLift lift(gyro_scaled, accel_scaled, g_ned);
        // Do NOT divide by gyro_sf or accel_sf if the CSV is already metric
        Vector3 gyro_scaled = d.gyro; 
        Vector3 accel_scaled = d.acc;
        IEKFLift lift(gyro_scaled, accel_scaled, g_ned);

        // Debug: check what the lift produces
        if (verbose && i < 1) {
            auto Lambda = lift(xi_hat);
            Vector9 nav_tangent = Lambda.head<9>();
            cout << "  Lift nav tangent [ω, δP, δV]: " << nav_tangent.transpose() << endl;
            cout << "  Lift nav tangent * dt:         " << (nav_tangent * dt).transpose() << endl;
            // Check what Expmap does with this
            NavState U_test = NavState::Expmap(nav_tangent * dt);
            cout << "  Expmap(Λ₁*dt) pos: " << U_test.position().transpose()
                 << "  vel: " << U_test.velocity().transpose() << endl;
            // What about Expmap without dt?
            NavState U_noscale = NavState::Expmap(nav_tangent);
            cout << "  Expmap(Λ₁)    pos: " << U_noscale.position().transpose()
                 << "  vel: " << U_noscale.velocity().transpose() << endl;
        }

        // ── BUILD A MATRIX ──
        auto A = buildA_IEKF(xi_hat, g_ned, accel_scaled);

        // Diagnostic: print A matrix norm to track stability
        if (verbose && (i < 5 || i % 1000 == 0)) {
            NavState nav_A = xi_hat.first;
            cout << "  ||A||=" << A.norm()
                 << "  ||A||·dt=" << A.norm()*dt
                 << "  |v̂|=" << nav_A.velocity().norm()
                 << "  |p̂|=" << nav_A.position().norm() << endl;
        }
        // ── PREDICT ──
        ekf.predictWithJacobian(lift, A, Qc, dt);

        // Debug: check state after predict
        if (verbose && i < 1) {
            NavStateBias xi_after = ekf.state();
            cout << "  AFTER predict pos: " << xi_after.first.position().transpose()
                 << "  vel: " << xi_after.first.velocity().transpose() << endl;
        }

        // ── GPS UPDATE ──
        if (ENABLE_GPS && !isnan(d.gps_lla(0))) {
            Vector3 z_ned = lla2ned(d.gps_lla, lla0);
            NavStateBias xi_before = ekf.state();

            Vector3 gps_innov = z_ned - xi_before.first.position();
            double innov_norm = gps_innov.norm();

            Vector3 z_gps_used = z_ned;
            double innov_scale = 1.0;

            if (GPS_LIMIT_INNOVATION && innov_norm > MAX_GPS_INNOV) {
                innov_scale = MAX_GPS_INNOV / innov_norm;
                z_gps_used = xi_before.first.position() + innov_scale * gps_innov;
            }

            if (GPS_POSITION_ONLY_UPDATE) {
                ekf.updateGpsPositionOnly(z_gps_used, R_gps, GPS_UPDATE_ACCEL_BIAS);
            } else {
                ekf.update(h_gps, z_gps_used, R_gps);
            }

            NavStateBias xi_after = ekf.state();

            // if (i < 20) {
            //     cout << "  GPS z_ned: " << z_ned.transpose() << endl;
            //     cout << "  GPS innov: " << (z_ned - xi_before.first.position()).transpose() << endl;
            //     cout << "  GPS pos Δ: " << (xi_after.first.position() - xi_before.first.position()).transpose() << endl;
            //     cout << "  GPS vel Δ: " << (xi_after.first.velocity() - xi_before.first.velocity()).transpose() << endl;
            //     cout << "  GPS att Δ: " << attError(rot3ToQuat(xi_before.first.attitude()), rot3ToQuat(xi_after.first.attitude())) << " deg" << endl;
            // }
            if (verbose && (i < 20 || (i % 1000 == 0))) {
                cout << "  GPS z_ned: " << z_ned.transpose() << endl;
                cout << "  GPS innov: " << gps_innov.transpose()
                    << "  norm=" << innov_norm
                    << "  scale=" << innov_scale << endl;
                if (GPS_LIMIT_INNOVATION && innov_scale < 1.0) {
                    cout << "  GPS used innov: "
                         << (z_gps_used - xi_before.first.position()).transpose()
                         << endl;
                }
                cout << "  GPS pos Δ: " << (xi_after.first.position() - xi_before.first.position()).transpose() << endl;
                cout << "  GPS vel Δ: " << (xi_after.first.velocity() - xi_before.first.velocity()).transpose() << endl;
                cout << "  GPS att Δ: " << attError(rot3ToQuat(xi_before.first.attitude()), rot3ToQuat(xi_after.first.attitude())) << " deg" << endl;
            }

        }

        // ── MAGNETOMETER UPDATE ──
        if (ENABLE_MAG && !isnan(d.magB(0))) {
            NavStateBias xi_before = ekf.state();
            Vector3 mag_pred = xi_before.first.attitude().unrotate(magI);
            ekf.update(mag_model, d.magB, R_mag);
            NavStateBias xi_after = ekf.state();
            if (i < 20) {
                cout << "  MAG meas:  " << d.magB.transpose() << endl;
                cout << "  MAG pred:  " << mag_pred.transpose() << endl;
                cout << "  MAG innov: " << (d.magB - mag_pred).transpose() << endl;
                cout << "  MAG att Δ: " << attError(rot3ToQuat(xi_before.first.attitude()), rot3ToQuat(xi_after.first.attitude())) << " deg" << endl;
                cout << "  MAG bias Δ: " << (xi_after.second - xi_before.second).transpose() << endl;
            }
        }

        // ── Record ──
        NavStateBias xi_est = ekf.state();
        NavState nav_est = xi_est.first;

        t_out[i]   = d.t;
        pos_out[i] = nav_est.position();
        vel_out[i] = nav_est.velocity();
        att_err[i] = attError(d.q_true, rot3ToQuat(nav_est.attitude()));
        pos_err[i] = nav_est.position() - d.ned_true;
        vel_err[i] = nav_est.velocity() - d.vel_true;

        if ((i+1) % (N/10) == 0)
            cout << "  " << 100*(i+1)/N << "%" << endl;
    }
    cout << "Done!" << endl;

    // ── Write results ──
    string out = "iekf_eqf_full_results.csv";
    ofstream fout(out);
    fout << "time,att_err,pos_N,pos_E,pos_D,vel_N,vel_E,vel_D,"
         << "err_N,err_E,err_D,verr_N,verr_E,verr_D" << endl;
    fout << fixed << setprecision(8);
    for (int i = 0; i < N; i++) {
        fout << t_out[i] << "," << att_err[i] << ","
             << pos_out[i](0) << "," << pos_out[i](1) << "," << pos_out[i](2) << ","
             << vel_out[i](0) << "," << vel_out[i](1) << "," << vel_out[i](2) << ","
             << pos_err[i](0) << "," << pos_err[i](1) << "," << pos_err[i](2) << ","
             << vel_err[i](0) << "," << vel_err[i](1) << "," << vel_err[i](2) << endl;
    }
    fout.close();
    cout << "Results: " << out << endl;

    // ── Summary ──
    double a_rmse=0, p_rmse=0, v_rmse=0;
    for (int i = 0; i < N; i++) {
        a_rmse += att_err[i]*att_err[i];
        p_rmse += pos_err[i].squaredNorm();
        v_rmse += vel_err[i].squaredNorm();
    }
    cout << "\n========================================" << endl;
    cout << " EqF (IEKF symmetry) RESULTS" << endl;
    cout << "========================================" << endl;
    cout << "  Att RMSE:  " << sqrt(a_rmse/N) << " deg" << endl;
    cout << "  Pos RMSE:  " << sqrt(p_rmse/N) << " m" << endl;
    cout << "  Vel RMSE:  " << sqrt(v_rmse/N) << " m/s" << endl;
    cout << "  Final att: " << att_err[N-1] << " deg" << endl;
    cout << "  Final pos: " << pos_err[N-1].norm() << " m" << endl;
    cout << "  Final vel: " << vel_err[N-1].norm() << " m/s" << endl;

    // Print final bias estimate
    NavStateBias xi_final = ekf.state();
    Vector6 bias_final = xi_final.second;
    cout << "  Final bg:  " << (bias_final.head<3>()*180/M_PI).transpose() << " deg/s" << endl;
    cout << "  Final ba:  " << bias_final.tail<3>().transpose() << " m/s²" << endl;
    cout << "========================================" << endl;

    return 0;
}
