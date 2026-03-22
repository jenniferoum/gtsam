/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/// @file EqVIOFilterCsvReplay.cpp
/// @brief Replay preprocessed EqVIO CSV data without feature tracking.

#include <gtsam_unstable/navigation/EqVIOFilter.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace gtsam;
using namespace gtsam::eqvio;

namespace {

struct HardcodedGroundTruth {
  static Vector3 position() { return Vector3(-0.954631, -0.101702, 0.179862); }
  static Vector3 velocity() { return Vector3(-0.120739, -0.314283, 0.119599); }
};

struct ReplayEvent {
  enum class Type { Imu, VisionFrame };
  Type type = Type::Imu;
  size_t seq = 0;
  double tAbs = 0.0;
  int frameIdx = -1;
  IMUInput imu;
  VisionMeasurement vision;
};

struct ReplayLog {
  std::map<std::string, std::string> metadata;
  std::vector<ReplayEvent> events;
};

std::string trim(const std::string& s) {
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

std::vector<std::string> splitCsvLine(const std::string& line) {
  std::vector<std::string> cols;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) {
    cols.push_back(trim(cell));
  }
  return cols;
}

ReplayLog readReplayCsv(const std::string& csvPath) {
  std::ifstream in(csvPath);
  if (!in.is_open()) {
    throw std::invalid_argument("EqVIOFilterCsvReplay: cannot open file: " +
                                csvPath);
  }

  std::string headerLine;
  if (!std::getline(in, headerLine)) {
    throw std::invalid_argument("EqVIOFilterCsvReplay: empty file: " + csvPath);
  }

  const std::vector<std::string> header = splitCsvLine(headerLine);
  std::unordered_map<std::string, size_t> index;
  for (size_t i = 0; i < header.size(); ++i) index[header[i]] = i;

  const std::vector<std::string> required = {
      "row_type", "t_abs", "seq", "frame_idx", "landmark_id", "gx", "gy",
      "gz",       "ax",    "ay",  "az",        "bgx",         "bgy", "bgz",
      "bax",      "bay",   "baz", "u_norm",    "v_norm",      "key", "value"};
  for (const auto& key : required) {
    if (index.count(key) == 0) {
      throw std::invalid_argument(
          "EqVIOFilterCsvReplay: missing required column: " + key);
    }
  }

  const auto getCell = [&](const std::vector<std::string>& cells,
                           const std::string& key) -> std::string {
    const auto it = index.find(key);
    if (it == index.end()) {
      throw std::invalid_argument(
          "EqVIOFilterCsvReplay: missing required column: " + key);
    }
    return (it->second < cells.size()) ? cells[it->second] : "";
  };

  const auto parseDoubleOr = [](const std::string& s, double fallback = 0.0) {
    return s.empty() ? fallback : std::stod(s);
  };
  const auto parseSizeOr = [](const std::string& s, size_t fallback = 0) {
    return s.empty() ? fallback : static_cast<size_t>(std::stoull(s));
  };
  const auto parseIntOr = [](const std::string& s, int fallback = 0) {
    return s.empty() ? fallback : std::stoi(s);
  };

  ReplayLog log;
  std::unordered_map<size_t, size_t> seqToVisionEvent;

  std::string line;
  while (std::getline(in, line)) {
    if (trim(line).empty()) continue;
    const std::vector<std::string> cells = splitCsvLine(line);
    const std::string rowType = getCell(cells, "row_type");

    if (rowType == "meta") {
      const std::string key = getCell(cells, "key");
      if (!key.empty()) log.metadata[key] = getCell(cells, "value");
      continue;
    }

    if (rowType == "imu") {
      ReplayEvent event;
      event.type = ReplayEvent::Type::Imu;
      event.seq = parseSizeOr(getCell(cells, "seq"));
      event.tAbs = parseDoubleOr(getCell(cells, "t_abs"));
      event.imu.stamp = event.tAbs;
      event.imu.gyr = Vector3(parseDoubleOr(getCell(cells, "gx")),
                              parseDoubleOr(getCell(cells, "gy")),
                              parseDoubleOr(getCell(cells, "gz")));
      event.imu.acc = Vector3(parseDoubleOr(getCell(cells, "ax")),
                              parseDoubleOr(getCell(cells, "ay")),
                              parseDoubleOr(getCell(cells, "az")));
      event.imu.gyrBiasVel = Vector3(parseDoubleOr(getCell(cells, "bgx")),
                                     parseDoubleOr(getCell(cells, "bgy")),
                                     parseDoubleOr(getCell(cells, "bgz")));
      event.imu.accBiasVel = Vector3(parseDoubleOr(getCell(cells, "bax")),
                                     parseDoubleOr(getCell(cells, "bay")),
                                     parseDoubleOr(getCell(cells, "baz")));
      log.events.push_back(std::move(event));
      continue;
    }

    if (rowType == "vision_feature") {
      const size_t seq = parseSizeOr(getCell(cells, "seq"));
      const double tAbs = parseDoubleOr(getCell(cells, "t_abs"));
      const int frameIdx = parseIntOr(getCell(cells, "frame_idx"), -1);
      const int id = parseIntOr(getCell(cells, "landmark_id"));
      const double u = parseDoubleOr(getCell(cells, "u_norm"));
      const double v = parseDoubleOr(getCell(cells, "v_norm"));

      auto it = seqToVisionEvent.find(seq);
      if (it == seqToVisionEvent.end()) {
        ReplayEvent event;
        event.type = ReplayEvent::Type::VisionFrame;
        event.seq = seq;
        event.tAbs = tAbs;
        event.frameIdx = frameIdx;
        event.vision[id] = Point2(u, v);
        seqToVisionEvent.emplace(seq, log.events.size());
        log.events.push_back(std::move(event));
      } else {
        ReplayEvent& event = log.events[it->second];
        if (event.frameIdx != frameIdx) {
          throw std::invalid_argument(
              "EqVIOFilterCsvReplay: inconsistent frame_idx for vision seq=" +
              std::to_string(seq));
        }
        if (std::abs(event.tAbs - tAbs) > 1e-9) {
          throw std::invalid_argument(
              "EqVIOFilterCsvReplay: inconsistent t_abs for vision seq=" +
              std::to_string(seq));
        }
        event.vision[id] = Point2(u, v);
      }
      continue;
    }

    throw std::invalid_argument("EqVIOFilterCsvReplay: unknown row_type: " +
                                rowType);
  }

  std::stable_sort(log.events.begin(), log.events.end(),
                   [](const ReplayEvent& a, const ReplayEvent& b) {
                     return a.seq < b.seq;
                   });
  return log;
}

double metadataDouble(const ReplayLog& log, const std::string& key,
                      double fallback) {
  const auto it = log.metadata.find(key);
  if (it == log.metadata.end() || it->second.empty()) return fallback;
  try {
    return std::stod(it->second);
  } catch (...) {
    return fallback;
  }
}

double metadataFiniteDouble(const ReplayLog& log, const std::string& key,
                            double fallback) {
  const double v = metadataDouble(log, key, fallback);
  return std::isfinite(v) ? v : fallback;
}

std::optional<Pose3> cameraExtrinsicsFromMetadata(const ReplayLog& log) {
  const std::vector<std::string> keys = {"T_ci_tx", "T_ci_ty", "T_ci_tz",
                                         "T_ci_qw", "T_ci_qx", "T_ci_qy",
                                         "T_ci_qz"};
  for (const auto& key : keys) {
    const auto it = log.metadata.find(key);
    if (it == log.metadata.end() || it->second.empty()) return std::nullopt;
  }

  const double tx = std::stod(log.metadata.at("T_ci_tx"));
  const double ty = std::stod(log.metadata.at("T_ci_ty"));
  const double tz = std::stod(log.metadata.at("T_ci_tz"));
  double qw = std::stod(log.metadata.at("T_ci_qw"));
  double qx = std::stod(log.metadata.at("T_ci_qx"));
  double qy = std::stod(log.metadata.at("T_ci_qy"));
  double qz = std::stod(log.metadata.at("T_ci_qz"));

  const double qnorm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  if (qnorm <= 1e-12) return std::nullopt;
  qw /= qnorm;
  qx /= qnorm;
  qy /= qnorm;
  qz /= qnorm;

  return Pose3(Rot3::Quaternion(qw, qx, qy, qz), Point3(tx, ty, tz));
}

void propagateBufferedImu(EqVIOFilter& filter, std::vector<IMUInput>& imuBuffer,
                          double& currentTime, double targetTime) {
  if (!filter.isInitialized() || imuBuffer.empty() ||
      targetTime <= currentTime) {
    return;
  }

  const double tRef = currentTime;
  std::vector<IMUInput> imuInputs;
  std::vector<double> dts;
  imuInputs.reserve(imuBuffer.size());
  dts.reserve(imuBuffer.size());
  double propagatedTime = 0.0;
  for (size_t i = 0; i < imuBuffer.size(); ++i) {
    const double t0 = std::max(imuBuffer[i].stamp, tRef);
    const double t1 =
        i + 1 < imuBuffer.size() ? std::min(imuBuffer[i + 1].stamp, targetTime)
                                 : targetTime;
    const double dt = std::max(t1 - t0, 0.0);
    if (dt <= 0.0) continue;
    imuInputs.push_back(imuBuffer[i]);
    dts.push_back(dt);
    propagatedTime += dt;
  }

  if (!imuInputs.empty()) {
    filter.propagate(imuInputs, dts);
    currentTime += propagatedTime;
  }

  auto it = std::find_if(imuBuffer.begin(), imuBuffer.end(),
                         [targetTime](const IMUInput& imu) {
                           return imu.stamp >= targetTime;
                         });
  if (it != imuBuffer.begin()) {
    --it;
    imuBuffer.erase(imuBuffer.begin(), it);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage:\n  " << argv[0] << " <processed_eqvio_stream.csv>\n";
    return 1;
  }

  const std::string csvPath = argv[1];

  try {
    const ReplayLog log = readReplayCsv(csvPath);

    EqVIOFilterParams params;
    params.initialPointDepth = metadataFiniteDouble(
        log, "eqf.initial_point_depth", params.initialPointDepth);
    params.initialPointVariance = metadataFiniteDouble(
        log, "eqf.initial_point_variance", params.initialPointVariance);
    params.measurementNoiseVariance = metadataFiniteDouble(
        log, "eqf.measurement_noise_variance_norm",
        params.measurementNoiseVariance);
    params.outlierThresholdAbs = metadataFiniteDouble(
        log, "eqf.outlier_threshold_abs", params.outlierThresholdAbs);
    params.outlierThresholdAbs = metadataFiniteDouble(
        log, "eqf.outlier_threshold_abs_norm", params.outlierThresholdAbs);
    params.outlierThresholdProb = metadataFiniteDouble(
        log, "eqf.outlier_threshold_prob", params.outlierThresholdProb);
    params.featureRetention =
        metadataFiniteDouble(log, "eqf.feature_retention", params.featureRetention);

    params.biasOmegaProcessVariance = metadataFiniteDouble(
        log, "eqf.process_var_bias_omega", params.biasOmegaProcessVariance);
    params.biasAccelProcessVariance = metadataFiniteDouble(
        log, "eqf.process_var_bias_accel", params.biasAccelProcessVariance);
    params.attitudeProcessVariance = metadataFiniteDouble(
        log, "eqf.process_var_attitude", params.attitudeProcessVariance);
    params.positionProcessVariance = metadataFiniteDouble(
        log, "eqf.process_var_position", params.positionProcessVariance);
    params.velocityProcessVariance = metadataFiniteDouble(
        log, "eqf.process_var_velocity", params.velocityProcessVariance);
    params.cameraAttitudeProcessVariance = metadataFiniteDouble(
        log, "eqf.process_var_cam_attitude", params.cameraAttitudeProcessVariance);
    params.cameraPositionProcessVariance = metadataFiniteDouble(
        log, "eqf.process_var_cam_position", params.cameraPositionProcessVariance);
    params.pointProcessVariance =
        metadataFiniteDouble(log, "eqf.process_var_point", params.pointProcessVariance);

    params.inputNoise.setZero();
    params.inputNoise.block<3, 3>(0, 0).setIdentity();
    params.inputNoise.block<3, 3>(3, 3).setIdentity();
    params.inputNoise.block<3, 3>(6, 6).setIdentity();
    params.inputNoise.block<3, 3>(9, 9).setIdentity();
    params.inputNoise.block<3, 3>(0, 0) *=
        metadataFiniteDouble(log, "eqf.input_var_gyr", params.inputNoise(0, 0));
    params.inputNoise.block<3, 3>(3, 3) *=
        metadataFiniteDouble(log, "eqf.input_var_acc", params.inputNoise(3, 3));
    params.inputNoise.block<3, 3>(6, 6) *= metadataFiniteDouble(
        log, "eqf.input_var_gyr_bias_walk", params.inputNoise(6, 6));
    params.inputNoise.block<3, 3>(9, 9) *= metadataFiniteDouble(
        log, "eqf.input_var_acc_bias_walk", params.inputNoise(9, 9));

    SensorState sensor;
    sensor.inputBias = Bias::Identity();
    sensor.pose = Pose3::Identity();
    sensor.velocity.setZero();
    sensor.cameraOffset =
        cameraExtrinsicsFromMetadata(log).value_or(Pose3::Identity());
    State xi0(sensor, {});
    Matrix Sigma0 = Matrix::Identity(xi0.dim(), xi0.dim());
    Sigma0.block<3, 3>(0, 0) *=
        metadataFiniteDouble(log, "eqf.initial_var_bias_omega", 0.1);
    Sigma0.block<3, 3>(3, 3) *=
        metadataFiniteDouble(log, "eqf.initial_var_bias_accel", 0.1);
    Sigma0.block<3, 3>(6, 6) *=
        metadataFiniteDouble(log, "eqf.initial_var_attitude", 1e-4);
    Sigma0.block<3, 3>(9, 9) *=
        metadataFiniteDouble(log, "eqf.initial_var_position", 1e-4);
    Sigma0.block<3, 3>(12, 12) *=
        metadataFiniteDouble(log, "eqf.initial_var_velocity", 1e-2);
    Sigma0.block<3, 3>(15, 15) *=
        metadataFiniteDouble(log, "eqf.initial_var_cam_attitude", 1e-5);
    Sigma0.block<3, 3>(18, 18) *=
        metadataFiniteDouble(log, "eqf.initial_var_cam_position", 1e-4);

    EqVIOFilter filter(params);
    filter.setReferenceState(xi0, Sigma0);
    auto camera = std::make_shared<CameraModel>(
        Pose3::Identity(), Cal3_S2(1.0, 1.0, 0.0, 0.0, 0.0));

    size_t imuCount = 0;
    size_t visionFrameCount = 0;
    size_t visionFeatureCount = 0;
    double currentTime = -1.0;
    std::vector<IMUInput> imuBuffer;
    for (const ReplayEvent& event : log.events) {
      if (event.type == ReplayEvent::Type::Imu) {
        if (!filter.isInitialized()) {
          filter.initializeFromIMU(event.imu);
          currentTime = event.imu.stamp;
        }
        if (filter.isInitialized()) imuBuffer.push_back(event.imu);
        ++imuCount;
      } else {
        propagateBufferedImu(filter, imuBuffer, currentTime, event.tAbs);
        const Matrix R =
            Matrix::Identity(static_cast<int>(2 * event.vision.size()),
                             static_cast<int>(2 * event.vision.size())) *
            params.measurementNoiseVariance;
        filter.correct(event.vision, camera, R);
        ++visionFrameCount;
        visionFeatureCount += event.vision.size();
      }
    }

    const State estimate = filter.stateEstimate();
    std::cout << "CSV replay complete.\n";
    std::cout << "Events: " << log.events.size() << ", IMU: " << imuCount
              << ", vision frames: " << visionFrameCount
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
    std::cout << "GT velocity: "
              << HardcodedGroundTruth::velocity().transpose() << "\n";

  } catch (const std::exception& e) {
    std::cerr << "EqVIOFilterCsvReplay failed: " << e.what() << "\n";
    return 2;
  }

  return 0;
}
