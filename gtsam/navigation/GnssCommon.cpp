/**
 *  @file   GnssCommon.cpp
 *  @brief  Implementation of shared GNSS utilities.
 *  @date   April 16, 2026
 **/

#include "GnssCommon.h"

#include <limits>

namespace gtsam {
namespace gnss {

double geodist(const Point3& sat, const Point3& rcv, Point3& e,
               OptionalJacobian<1, 3> H_rcv) {
  const Point3 dr = sat - rcv;
  const double r = dr.norm();
  const double sagnac =
      OMGE * (sat.x() * rcv.y() - sat.y() * rcv.x()) / C_LIGHT;
  if (r < std::numeric_limits<double>::epsilon()) {
    e = Point3::Zero();
    if (H_rcv) H_rcv->setZero();
    return sagnac;
  }
  e = dr / r;
  if (H_rcv) {
    // d(||sat - rcv||)/d(rcv) = -e^T (1x3)
    // d(sagnac)/d(rcv)        = OMGE / C_LIGHT * [-sat.y(), sat.x(), 0]
    (*H_rcv)(0, 0) = -e.x() - OMGE * sat.y() / C_LIGHT;
    (*H_rcv)(0, 1) = -e.y() + OMGE * sat.x() / C_LIGHT;
    (*H_rcv)(0, 2) = -e.z();
  }
  return r + sagnac;
}

}  // namespace gnss
}  // namespace gtsam
