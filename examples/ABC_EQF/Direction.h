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
};
#endif //DIRECTION_H
