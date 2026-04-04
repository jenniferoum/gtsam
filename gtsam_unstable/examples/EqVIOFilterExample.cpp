/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file EqVIOFilterExample.cpp
 * @brief Run the EqVIOFilter on 10 seconds of the EuRoC MAV Vicon Room 1
 * dataset.
 * @author Rohan Bansal
 */

#include <gtsam/slam/dataset.h>
#include <gtsam_unstable/examples/EqVIOReplayHelper.h>

#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gtsam;
using namespace gtsam::eqvio;
namespace replay = gtsam::eqvio::examples;

namespace {

/**
 * @brief Hardcoded reference values used for a lightweight smoke-style replay
 * check.
 */
struct HardcodedGroundTruth {
  static Vector3 position() { return Vector3(-0.954631, -0.101702, 0.179862); }
  static Vector3 velocity() { return Vector3(-0.120739, -0.314283, 0.119599); }
};

/// Build runtime filter params from replay metadata with fallback defaults.
EqVIOFilterParams paramsFromMetadata(const replay::MetadataMap& metadata) {
  EqVIOFilterParams params;
  const auto setParam = [&metadata](double& field,
                                    std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
      field = replay::metadataFiniteDouble(metadata, key, field);
    }
  };

  setParam(params.initialPointDepth, {"eqf.initial_point_depth"});
  setParam(params.initialPointVariance, {"eqf.initial_point_variance"});
  setParam(params.measurementNoiseVariance,
           {"eqf.measurement_noise_variance_norm"});
  setParam(params.outlierThresholdAbs,
           {"eqf.outlier_threshold_abs", "eqf.outlier_threshold_abs_norm"});
  setParam(params.featureRetention, {"eqf.feature_retention"});
  setParam(params.biasOmegaProcessVariance, {"eqf.process_var_bias_omega"});
  setParam(params.biasAccelProcessVariance, {"eqf.process_var_bias_accel"});
  setParam(params.attitudeProcessVariance, {"eqf.process_var_attitude"});
  setParam(params.positionProcessVariance, {"eqf.process_var_position"});
  setParam(params.velocityProcessVariance, {"eqf.process_var_velocity"});
  setParam(params.cameraAttitudeProcessVariance,
           {"eqf.process_var_cam_attitude"});
  setParam(params.cameraPositionProcessVariance,
           {"eqf.process_var_cam_position"});
  setParam(params.pointProcessVariance, {"eqf.process_var_point"});

  params.inputNoise.setZero();
  for (int idx : {0, 3, 6, 9}) {
    params.inputNoise.block<3, 3>(idx, idx).setIdentity();
  }
  const auto setInputVariance = [&metadata, &params](int idx, const char* key) {
    const double variance = replay::metadataFiniteDouble(
        metadata, key, params.inputNoise(idx, idx));
    params.inputNoise.block<3, 3>(idx, idx) *= variance;
  };
  setInputVariance(0, "eqf.input_var_gyr");
  setInputVariance(3, "eqf.input_var_acc");
  setInputVariance(6, "eqf.input_var_gyr_bias_walk");
  setInputVariance(9, "eqf.input_var_acc_bias_walk");

  return params;
}

/// Construct initial reference state using metadata-provided camera extrinsics.
State initialReferenceState(const replay::MetadataMap& metadata) {
  SensorState sensor;
  sensor.inputBias = Bias::Identity();
  sensor.pose = Pose3::Identity();
  sensor.velocity.setZero();
  sensor.cameraOffset =
      replay::cameraExtrinsicsFromMetadata(metadata).value_or(Pose3::Identity());
  return State(sensor, {});
}

/// Print compact replay statistics and terminal estimate summary.
void printSummary(const EqVIOFilterParams& params, double currentTime,
                  size_t imuCount, size_t visionFrameCount,
                  size_t visionFeatureCount, const State& estimate) {
  std::cout << "CSV replay complete.\n";
  std::cout << "Events: " << (imuCount + visionFrameCount) << ", IMU: "
            << imuCount << ", vision frames: " << visionFrameCount
            << ", vision features: " << visionFeatureCount << "\n";
  std::cout << "Measurement noise variance (normalized): "
            << params.measurementNoiseVariance << "\n";
  std::cout << std::setprecision(17);
  std::cout << "Filter time: " << currentTime << "\n";
  std::cout << std::setprecision(10);
  std::cout << "Landmarks: " << estimate.n() << "\n";
  std::cout << "Pose translation: "
            << estimate.sensor.pose.translation().transpose() << "\n";
  std::cout << "GT pose translation: "
            << HardcodedGroundTruth::position().transpose() << "\n";
  std::cout << "Velocity: " << estimate.sensor.velocity.transpose() << "\n";
  std::cout << "GT velocity: " << HardcodedGroundTruth::velocity().transpose()
            << "\n";
}

}  // namespace

/**
 * @brief Run EqVIO filter replay on bundled EuRoC-derived CSV data.
 */
int main() {
    const std::string csvPath =
      findExampleDataFile("EqVIOdata_eurocmav_room1_10sec.csv");

  try {
    const replay::MetadataMap metadata = replay::readReplayMetadata(csvPath);
    const EqVIOFilterParams params = paramsFromMetadata(metadata);
    const State xi0 = initialReferenceState(metadata);

    std::optional<EqVIOFilter> filter;
    auto camera = std::make_shared<CameraModel>(
        Pose3::Identity(), Cal3_S2(1.0, 1.0, 0.0, 0.0, 0.0));

    size_t imuCount = 0;
    size_t visionFrameCount = 0;
    size_t visionFeatureCount = 0;
    bool gravityInitialized = false;
    double currentTime = -1.0;
    std::vector<IMUInput> imuBuffer;

    replay::ReplayReader reader(csvPath);
    replay::ReplayEvent event;
    while (reader.next(event)) {
      if (event.type == replay::ReplayEvent::Type::Imu) {
        if (!filter) {
          filter.emplace(params);
        }
        if (!gravityInitialized) {
          filter->initializeFromIMU(event.imu);
          currentTime = event.imu.stamp;
          gravityInitialized = true;
        }

        imuBuffer.push_back(event.imu);
        ++imuCount;
        continue;
      }

      ++visionFrameCount;
      visionFeatureCount += event.vision.size();
      if (!gravityInitialized || !filter) {
        continue;
      }

      const replay::BufferedImuPropagation step =
          replay::makeBufferedImuPropagation(imuBuffer, currentTime, event.tAbs);
      if (!step.imuInputs.empty()) {
        filter->predict(step.imuInputs, step.dts);
        currentTime += step.propagatedTime;
      }
      if (step.trimCount > 0) {
        imuBuffer.erase(
            imuBuffer.begin(),
            imuBuffer.begin() +
                static_cast<std::vector<IMUInput>::difference_type>(
                    step.trimCount));
      }

      const Matrix R =
          Matrix::Identity(static_cast<int>(2 * event.vision.size()),
                           static_cast<int>(2 * event.vision.size())) *
          params.measurementNoiseVariance;
      filter->update(event.vision, camera, R);
    }

    const State finalEstimate = filter ? filter->state() : xi0;
    printSummary(params, currentTime, imuCount, visionFrameCount,
                 visionFeatureCount, finalEstimate);
  } catch (const std::exception& e) {
    std::cerr << "EqVIOFilterExample failed: " << e.what() << "\n";
    return 2;
  }

  return 0;
}
