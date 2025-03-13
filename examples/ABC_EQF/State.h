//
// Created by darshan on 3/11/25.
//

#ifndef STATE_H
#define STATE_H

#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/Vector.h>
#include <vector>

using namespace gtsam;

/**
 * State class representing the state of the Biased Attitude System
 */
class State {
public:
    Rot3 R;                  // Attitude rotation matrix R
    Vector3 b;               // Gyroscope bias b
    std::vector<Rot3> S;     // Sensor calibrations S
    
    State(const Rot3& R = Rot3::Identity(),
          const Vector3& b = Vector3::Zero(),
          const std::vector<Rot3>& S = std::vector<Rot3>());
    
    static State identity(int n);
};
#endif //STATE_H
