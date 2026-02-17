/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file WNOAStateData.h
 * @brief Introduces a lightweight struct for identifying states in continuous-time estimation and interpolation.
 * @author Connor Holmes
 * @author Zi Cong Guo
 * @author Sven Lilge
 */

#pragma once

#include <gtsam/inference/Key.h>

#include <functional>
#include <limits>

namespace gtsam {

/**
 * @brief Lightweight container for states used for continuous-time estimation and interpolation.
 *
 * `StateData` stores the keys identifying the pose and velocity variables in
 * a factor graph together with the associated timestamp. It is intentionally
 * minimal and used primarily for ordering, lookup and grouping of states when
 * constructing interpolation queries.
 */
struct StateData {
  /// Key of the pose variable
  Key pose;

  /// Key of the velocity variable
  Key vel;

  /// Timestamp (seconds) associated with this state
  double time;

  /**
   * @brief Default constructor (value-initialized members).
   *
   * Provided for convenience and to enable use in standard containers.
   */
  StateData() = default;

  /**
   * @brief Construct a StateData from explicit keys and timestamp.
   * @param pose_in Pose key
   * @param vel_in Velocity key
   * @param time_in Timestamp in seconds
   */
  StateData(Key pose_in, Key vel_in, double time_in)
      : pose(pose_in), vel(vel_in), time(time_in) {};

  /**
   * @brief Strict-weak ordering: (time, pose, vel) ascending.
   *
   * Primarily orders states chronologically by time. When timestamps are
   * equal, pose and then vel keys are used as tie-breakers to provide a
   * strict-weak ordering suitable for ordered containers such as std::set.
   */
  bool operator<(const StateData& other) const {
    if (this->time < other.time) return true;
    if (this->time > other.time) return false;
    if (this->pose < other.pose) return true;
    if (this->pose > other.pose) return false;
    return this->vel < other.vel;
  }

  /**
   * @brief Compare by timestamp (descending).
   */
  bool operator>(const StateData& other) const { 
    if (this->time > other.time) return true;
    if (this->time < other.time) return false;
    if (this->pose > other.pose) return true;
    if (this->pose < other.pose) return false;
    return this->vel > other.vel;
   }

  /**
   * @brief Compare this state's timestamp with a raw time value.
   * @param time Timestamp to compare against
   * @return true if this->time < time
   */
  bool operator<(double time) const { return this->time < time; }

  /**
   * @brief Compare this state's timestamp with a raw time value.
   * @param time Timestamp to compare against
   * @return true if this->time > time
   */
  bool operator>(double time) const { return this->time > time; }

  /**
   * @brief Equality compares the identifying keys (pose and vel).
   *
   * Two `StateData` objects are considered equal if they refer to the same
   * pose and velocity keys. The timestamp is not used for equality to keep
   * `StateData` usable as a simple identifier in unordered containers.
   */
  bool operator==(const StateData& other) const {
    return (this->pose == other.pose) && (this->vel == other.vel);
  }
};

}  // namespace gtsam

/**
 * @brief Hash `StateData` using the pose and velocity keys.
 *
 * Combines the hash of the pose and velocity keys to produce a stable
 * hash suitable for `unordered_map`/`unordered_set`. The timestamp is not
 * included because `StateData` equality ignores time as well.
 */
namespace std {
template <>
struct hash<gtsam::StateData> {
  std::size_t operator()(const gtsam::StateData& p) const noexcept {
    return std::hash<uint64_t>()(p.pose) ^ (std::hash<uint64_t>()(p.vel) << 1);
  }
};
}  // namespace std