/**
 *  @file   PseudorangeFactor.cpp
 *  @author Sammy Guo
 *  @brief  Implementation file for GNSS Pseudorange factor
 *  @date   January 18, 2026
 **/

#include "PseudorangeFactor.h"

namespace {

/// Speed of light in a vacuum (m/s):
constexpr double CLIGHT = 299792458.0;

}  // namespace

namespace gtsam {

PseudorangeFactor::PseudorangeFactor()
    : Base(), pseudorange_(0.0), sat_pos_(Point3::Zero()), sat_clk_bias_(0.0) {}

PseudorangeFactor::PseudorangeFactor(Key receiver_position_key,
                                     Key receiver_clock_bias_key,
                                     const double measured_pseudorange,
                                     const Point3& satellite_position,
                                     const double satellite_clock_bias,
                                     const SharedNoiseModel& model)
    : Base(model, receiver_position_key, receiver_clock_bias_key),
      pseudorange_(measured_pseudorange),
      sat_pos_(satellite_position),
      sat_clk_bias_(satellite_clock_bias) {}

//***************************************************************************
// void PseudorangeFactor::print(const std::string& s, const KeyFormatter&
// keyFormatter) const {
//  cout << (s.empty() ? "" : s + " ") << "PseudorangeFactor on " <<
//  keyFormatter(key())
//       << "\n";
//  cout << "  GPS measurement: " << nT_ << "\n";
//  noiseModel_->print("  noise model: ");
//}

//***************************************************************************
// bool PseudorangeFactor::equals(const NonlinearFactor& expected, double tol)
// const {
//  const This* e = dynamic_cast<const This*>(&expected);
//  return e != nullptr && Base::equals(*e, tol) /*&&
//  traits<Point3>::Equals(nT_, e->nT_, tol)*/;
//}

//***************************************************************************
Vector PseudorangeFactor::evaluateError(
    const Point3& receiver_position, const double& receiver_clock_bias,
    OptionalMatrixType Hreceiver_pos,
    OptionalMatrixType Hreceiver_clock_bias) const {
  // Apply pseudorange equation: rho = range + c*[dt_u - dt^s]
  const Vector3 position_difference = receiver_position - sat_pos_;
  const double range = position_difference.norm();
  const double rho = range + CLIGHT * (receiver_clock_bias - sat_clk_bias_);
  const double error = rho - pseudorange_;

  // Compute associated derivatives:
  if (Hreceiver_pos) {
    if (range < std::numeric_limits<double>::epsilon()) {
      *Hreceiver_pos = Matrix13::Zero();
    } else {
      *Hreceiver_pos = position_difference / range;
    }
  }

  if (Hreceiver_clock_bias) {
    *Hreceiver_clock_bias = Matrix11(CLIGHT);
  }

  return Vector1(error);
}

}  // namespace gtsam
