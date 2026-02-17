#pragma once

#include <gtsam/base/Manifold.h>
#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/base/VectorSpace.h>
#include <gtsam/base/std_optional_serialization.h>
#if GTSAM_ENABLE_BOOST_SERIALIZATION
#include <boost/serialization/nvp.hpp>
#endif

#include <iostream>
#include <optional>
#include <utility>
#include <vector>

namespace gtsam {

typedef Vector1 Point1;

/// Convenience typedefs
using Point1Pair = std::pair<Point1, Point1>;
using Point1Pairs = std::vector<Point1Pair>;

/**
 * @brief Stream insertion operator for Point1Pair.
 *
 * Outputs a human-readable representation of a Point1Pair to the given output
 * stream.
 *
 * @param os The output stream to write to.
 * @param p  The Point1Pair to be serialized to the stream.
 * @return A reference to the output stream \p os after insertion.
 */
GTSAM_EXPORT std::ostream& operator<<(std::ostream& os,
                                      const gtsam::Point1Pair& p);

/**
 * @brief Compute the distance between point and origin.
 *
 * Calculates the absolute distance between two Point1 objects,
 * with optional Jacobians with respect to each point.
 *
 * @param p The first 1D point.
 * @param H Optional Jacobian of the distance with respect to the first point \p
 * p1.
 * @return The scalar distance between \p p and 0.
 */
GTSAM_EXPORT double norm1(const Point1& p, OptionalJacobian<1, 1> H = {});

/**
 * @brief Compute the distance between two 1D points.
 *
 * Calculates the absolute distance between two Point1 objects,
 * with optional Jacobians with respect to each point.
 *
 * @param p1 The first 1D point.
 * @param q  The second 1D point.
 * @param H1 Optional Jacobian of the distance with respect to the first point
 * \p p1.
 * @param H2 Optional Jacobian of the distance with respect to the second point
 * \p q.
 * @return The scalar distance between \p p1 and \p q.
 */
GTSAM_EXPORT double distance1(const Point1& p1, const Point1& q,
                              OptionalJacobian<1, 1> H1 = {},
                              OptionalJacobian<1, 1> H2 = {});

template <typename A1, typename A2>
struct Range;

template <>
struct Range<Point1, Point1> {
  typedef double result_type;
  double operator()(const Point1& p, const Point1& q,
                    OptionalJacobian<1, 1> H1 = {},
                    OptionalJacobian<1, 1> H2 = {}) {
    return distance1(p, q, H1, H2);
  }
};

}  // namespace gtsam
