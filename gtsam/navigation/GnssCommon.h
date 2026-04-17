/**
 *  @file   GnssCommon.h
 *  @brief  Shared constants and utilities for GNSS factors.
 *  @date   April 14, 2026
 **/
#pragma once

#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/dllexport.h>
#include <gtsam/geometry/Point3.h>

namespace gtsam {

/**
 * Base class storing common members for GNSS measurement factors.
 *
 * Shared by pseudorange and carrier phase factors.  The measurement field is
 * in meters (i.e. for carrier phase factors it is already multiplied by the
 * wavelength, `lambda * phi_cycles`).
 */
struct GnssMeasurementBase {
  /// Measurement in meters (pseudorange, or carrier phase * wavelength).
  double measurement_;
  /// Satellite position in WGS84 ECEF meters.
  Point3 satPos_;
  /// Satellite clock bias in seconds.
  double satClkBias_;
};

namespace gnss {

/// Speed of light in a vacuum (m/s).
constexpr double C_LIGHT = 299792458.0;

/// WGS-84 Earth rotation rate (rad/s).
constexpr double OMGE = 7.2921151467e-5;

/**
 * Geometric distance with Sagnac correction.
 *
 * Computes the Euclidean distance between a satellite and a receiver in ECEF,
 * plus the first-order Sagnac correction for the rotating Earth.  When the
 * satellite and receiver positions coincide (within machine epsilon) the unit
 * vector is set to zero and the optional Jacobian is set to zero, which avoids
 * NaNs from dividing by zero.
 *
 * @param[in]  sat   Satellite ECEF position (m).
 * @param[in]  rcv   Receiver ECEF position (m).
 * @param[out] e     Unit vector from receiver to satellite.
 * @param[out] H_rcv Optional Jacobian of the returned range w.r.t. rcv (1x3).
 *                   Includes the contribution from the Sagnac term.
 * @return     Geometric range with Sagnac correction (m).
 */
GTSAM_EXPORT double geodist(const Point3& sat, const Point3& rcv, Point3& e,
                            OptionalJacobian<1, 3> H_rcv = {});

}  // namespace gnss
}  // namespace gtsam
