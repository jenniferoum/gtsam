//
// Created by darshan on 3/11/25.
//
#include "State.h"

State::State(const Rot3& R, const Vector3& b, const std::vector<Rot3>& S)
    : R(R), b(b), S(S) {}

State State::identity(int n) {
    std::vector<Rot3> calibrations(n, Rot3::Identity());
    return State(Rot3::Identity(), Vector3::Zero(), calibrations);
}