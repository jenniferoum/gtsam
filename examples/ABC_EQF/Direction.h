//
// Created by darshan on 3/11/25.
//

#ifndef DIRECTION_H
#define DIRECTION_H
//#pragma once
#include <gtsam/geometry/Unit3.h>
#include <gtsam/base/Vector.h>
using namespace gtsam;
/**
 * Direction class as a S2 element
 */
class Direction {
public:
    Unit3 d;  // Direction (unit vector on S2)

    /**
     * Initialize direction
     * @param d_vec Direction vector (must be unit norm)
     */
    Direction(const Vector3& d_vec);

    // Accessor methods for vector components
    double x() const { return d.unitVector()[0]; }
    double y() const { return d.unitVector()[1]; }
    double z() const { return d.unitVector()[2]; }

    // Check if the direction contains NaN values
    bool hasNaN() const {
        Vector3 vec = d.unitVector();
        return std::isnan(vec[0]) || std::isnan(vec[1]) || std::isnan(vec[2]);
    }
};
#endif //DIRECTION_H
