//
// Created by darshan on 3/11/25.
//

#include "Direction.h"
#include "utilities.h"
#include <stdexcept>

Direction::Direction(const Vector3& d_vec) : d(d_vec) {
    if (!checkNorm(d_vec)) {
        throw std::invalid_argument("Direction must be a unit vector");
    }
}