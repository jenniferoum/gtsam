/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file EqVIOFilterExample.cpp
 * @brief Run the EqVIOFilter on 10 seconds of the EuRoC MAV Vicon Room 1 dataset.
 * @author Rohan Bansal
 */

#include <gtsam_unstable/navigation/EqVIOFilter.h>
#include <gtsam/slam/dataset.h>

#include <algorithm>
#include <array>
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

constexpr const char* kExampleName = "EqVIOFilterExample";

/**
 * @brief Hardcoded reference values used for a lightweight smoke-style replay check.
 *
 * These correspond to the expected terminal pose/velocity for the bundled
 * 10-second EuRoC replay file.
 */
struct HardcodedGroundTruth {
  static Vector3 position() { return Vector3(-0.954631, -0.101702, 0.179862); }
  static Vector3 velocity() { return Vector3(-0.120739, -0.314283, 0.119599); }
};

/// One replay stream item: either an IMU sample or an aggregated vision frame.
struct ReplayEvent {
  enum class Type { Imu, VisionFrame };
  Type type = Type::Imu;
  size_t seq = 0;
  double tAbs = 0.0;
  int frameIdx = -1;
  IMUInput imu;
  VisionMeasurement vision;
};

/// Parsed CSV replay log including metadata and ordered event sequence.
struct ReplayLog {
  std::map<std::string, std::string> metadata;
  std::vector<ReplayEvent> events;
};

/// Trim leading/trailing ASCII whitespace.
std::string trim(const std::string& s) {
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

/// Split a CSV line by commas and trim each field.
std::vector<std::string> splitCsvLine(const std::string& line) {
  std::vector<std::string> cols;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) {
    cols.push_back(trim(cell));
  }
  return cols;
}

/// Format a consistent exception prefix for this example.
std::string formatError(const std::string& detail) {
  return std::string(kExampleName) + ": " + detail;
}

/// Parse scalar/string helpers used by CSV rows and metadata.
double parseDoubleOr(const std::string& s, double fallback = 0.0) {
  return s.empty() ? fallback : std::stod(s);
}

size_t parseSizeOr(const std::string& s, size_t fallback = 0) {
  return s.empty() ? fallback : static_cast<size_t>(std::stoull(s));
}

int parseIntOr(const std::string& s, int fallback = 0) {
  return s.empty() ? fallback : std::stoi(s);
}

/**
 * @brief Read EqVIO replay CSV and convert to strongly typed event stream.
 *
 * The parser expects rows tagged by `row_type`:
 * - `meta`: key/value metadata.
 * - `imu`: one IMU sample.
 * - `vision_feature`: one feature observation, aggregated by `seq` into a frame.
 *
 * @throws std::invalid_argument if required columns are missing or row data is inconsistent.
 */
ReplayLog readReplayCsv(const std::string& csvPath) {
  std::ifstream in(csvPath);
  if (!in.is_open()) {
    throw std::invalid_argument(formatError("cannot open file: " + csvPath));
  }

  std::string headerLine;
  if (!std::getline(in, headerLine)) {
    throw std::invalid_argument(formatError("empty file: " + csvPath));
  }

  const std::vector<std::string> header = splitCsvLine(headerLine);
  std::unordered_map<std::string, size_t> index;
  for (size_t i = 0; i < header.size(); ++i) index[header[i]] = i;

  const std::array<const char*, 21> required = {
      "row_type", "t_abs", "seq", "frame_idx", "landmark_id", "gx", "gy",
      "gz",       "ax",    "ay",  "az",        "bgx",         "bgy", "bgz",
      "bax",      "bay",   "baz", "u_norm",    "v_norm",      "key", "value"};
  for (const char* key : required) {
    if (index.count(key) == 0) {
      throw std::invalid_argument(
          formatError("missing required column: " + std::string(key)));
    }
  }

  const auto getCell = [&](const std::vector<std::string>& cells,
                           const std::string& key) -> std::string {
    const auto it = index.find(key);
    if (it == index.end()) {
      throw std::invalid_argument(formatError("missing required column: " + key));
    }
    return (it->second < cells.size()) ? cells[it->second] : "";
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
              formatError("inconsistent frame_idx for vision seq=" +
                          std::to_string(seq)));
        }
        if (std::abs(event.tAbs - tAbs) > 1e-9) {
          throw std::invalid_argument(
              formatError("inconsistent t_abs for vision seq=" +
                          std::to_string(seq)));
        }
        event.vision[id] = Point2(u, v);
      }
      continue;
    }

    throw std::invalid_argument(formatError("unknown row_type: " + rowType));
  }

  std::stable_sort(log.events.begin(), log.events.end(),
                   [](const ReplayEvent& a, const ReplayEvent& b) {
                     return a.seq < b.seq;
                   });
  return log;
}

/// Parse one metadata scalar key with fallback on missing/invalid values.
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

/// Parse one metadata scalar key and require finite result, otherwise fallback.
double metadataFiniteDouble(const ReplayLog& log, const std::string& key,
                            double fallback) {
  const double v = metadataDouble(log, key, fallback);
  return std::isfinite(v) ? v : fallback;
}

/**
 * @brief Parse camera extrinsics from metadata keys `T_ci_*` if available.
 * @return `std::nullopt` when any extrinsic field is missing or quaternion is invalid.
 */
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

/**
 * @brief Prepared IMU propagation data extracted from a rolling IMU buffer.
 *
 * `imuInputs[i]` is held for duration `dts[i]`. `trimCount` indicates how many
 * leading IMU entries can be dropped from the caller-owned buffer after use.
 */
struct BufferedImuPropagation {
  std::vector<IMUInput> imuInputs;
  std::vector<double> dts;
  double propagatedTime = 0.0;
  size_t trimCount = 0;
};

/**
 * @brief Convert buffered absolute-timestamp IMU samples into hold segments.
 *
 * The returned segments cover `[currentTime, targetTime]` by clipping each hold
 * interval and preserving the existing replay semantics used by this example.
 */
BufferedImuPropagation makeBufferedImuPropagation(
    const std::vector<IMUInput>& imuBuffer, double currentTime,
    double targetTime) {
  BufferedImuPropagation out;
  if (imuBuffer.empty() || targetTime <= currentTime) {
    return out;
  }

  const double tRef = currentTime;
  out.imuInputs.reserve(imuBuffer.size());
  out.dts.reserve(imuBuffer.size());
  for (size_t i = 0; i < imuBuffer.size(); ++i) {
    const double t0 = std::max(imuBuffer[i].stamp, tRef);
    const double t1 =
        i + 1 < imuBuffer.size() ? std::min(imuBuffer[i + 1].stamp, targetTime)
                                 : targetTime;
    const double dt = std::max(t1 - t0, 0.0);
    if (dt <= 0.0) continue;
    out.imuInputs.push_back(imuBuffer[i]);
    out.dts.push_back(dt);
    out.propagatedTime += dt;
  }

  auto it = std::find_if(imuBuffer.begin(), imuBuffer.end(),
                         [targetTime](const IMUInput& imu) {
                           return imu.stamp >= targetTime;
                         });
  if (it != imuBuffer.begin()) {
    --it;
    out.trimCount = static_cast<size_t>(std::distance(imuBuffer.begin(), it));
  }
  return out;
}

/// Build runtime filter params from replay metadata, falling back to defaults for missing entries.
EqVIOFilterParams paramsFromMetadata(const ReplayLog& log) {
  EqVIOFilterParams params;
  const auto setParam = [&log](double& field,
                               std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
      field = metadataFiniteDouble(log, key, field);
    }
  };

  setParam(params.initialPointDepth, {"eqf.initial_point_depth"});
  setParam(params.initialPointVariance, {"eqf.initial_point_variance"});
  setParam(params.measurementNoiseVariance, {"eqf.measurement_noise_variance_norm"});
  setParam(params.outlierThresholdAbs,
           {"eqf.outlier_threshold_abs", "eqf.outlier_threshold_abs_norm"});
  setParam(params.outlierThresholdProb, {"eqf.outlier_threshold_prob"});
  setParam(params.featureRetention, {"eqf.feature_retention"});
  setParam(params.biasOmegaProcessVariance, {"eqf.process_var_bias_omega"});
  setParam(params.biasAccelProcessVariance, {"eqf.process_var_bias_accel"});
  setParam(params.attitudeProcessVariance, {"eqf.process_var_attitude"});
  setParam(params.positionProcessVariance, {"eqf.process_var_position"});
  setParam(params.velocityProcessVariance, {"eqf.process_var_velocity"});
  setParam(params.cameraAttitudeProcessVariance, {"eqf.process_var_cam_attitude"});
  setParam(params.cameraPositionProcessVariance, {"eqf.process_var_cam_position"});
  setParam(params.pointProcessVariance, {"eqf.process_var_point"});

  params.inputNoise.setZero();
  for (int idx : {0, 3, 6, 9}) {
    params.inputNoise.block<3, 3>(idx, idx).setIdentity();
  }
  const auto setInputVariance = [&log, &params](int idx, const char* key) {
    const double variance = metadataFiniteDouble(log, key, params.inputNoise(idx, idx));
    params.inputNoise.block<3, 3>(idx, idx) *= variance;
  };
  setInputVariance(0, "eqf.input_var_gyr");
  setInputVariance(3, "eqf.input_var_acc");
  setInputVariance(6, "eqf.input_var_gyr_bias_walk");
  setInputVariance(9, "eqf.input_var_acc_bias_walk");

  return params;
}

/// Construct initial reference state using identity sensor state plus optional camera extrinsics metadata.
State initialReferenceState(const ReplayLog& log) {
  SensorState sensor;
  sensor.inputBias = Bias::Identity();
  sensor.pose = Pose3::Identity();
  sensor.velocity.setZero();
  sensor.cameraOffset = cameraExtrinsicsFromMetadata(log).value_or(Pose3::Identity());
  return State(sensor, {});
}

/// Construct initial covariance from metadata overrides with sensible defaults.
Matrix initialCovarianceFromMetadata(const ReplayLog& log, const State& xi0) {
  Matrix Sigma0 = Matrix::Identity(xi0.dim(), xi0.dim());
  const auto setInitialVarianceBlock = [&log, &Sigma0](int idx, const char* key,
                                                       double fallback) {
    Sigma0.block<3, 3>(idx, idx) *= metadataFiniteDouble(log, key, fallback);
  };

  setInitialVarianceBlock(0, "eqf.initial_var_bias_omega", 0.1);
  setInitialVarianceBlock(3, "eqf.initial_var_bias_accel", 0.1);
  setInitialVarianceBlock(6, "eqf.initial_var_attitude", 1e-4);
  setInitialVarianceBlock(9, "eqf.initial_var_position", 1e-4);
  setInitialVarianceBlock(12, "eqf.initial_var_velocity", 1e-2);
  setInitialVarianceBlock(15, "eqf.initial_var_cam_attitude", 1e-5);
  setInitialVarianceBlock(18, "eqf.initial_var_cam_position", 1e-4);
  return Sigma0;
}

/// Print compact replay statistics and terminal estimate summary.
void printSummary(const ReplayLog& log, const EqVIOFilterParams& params,
                  double currentTime, size_t imuCount, size_t visionFrameCount,
                  size_t visionFeatureCount, const State& estimate) {
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
}

}  // namespace

/**
 * @brief Run EqVIO filter replay on bundled EuRoC-derived CSV data.
 *
 * This example loads the dataset using `findExampleDataFile`, initializes the
 * filter from metadata, replays all IMU and vision events, and prints a final
 * summary for quick validation.
 */
int main() {
  // Resolve the bundled replay file from the example data search paths.
  const std::string csvPath =
      findExampleDataFile("EqVIOdata_eurocmav_room1_10sec.csv");

  try {
    // Parse events and metadata, then construct filter configuration and initial conditions.
    const ReplayLog log = readReplayCsv(csvPath);
    const EqVIOFilterParams params = paramsFromMetadata(log);
    const State xi0 = initialReferenceState(log);
    const Matrix Sigma0 = initialCovarianceFromMetadata(log, xi0);

    // Create filter and set reference-state chart origin/covariance.
    EqVIOFilter filter(params);
    filter.setReferenceState(xi0, Sigma0);
    // Replay file stores normalized coordinates, so use identity intrinsics/extrinsics camera.
    auto camera = std::make_shared<CameraModel>(
        Pose3::Identity(), Cal3_S2(1.0, 1.0, 0.0, 0.0, 0.0));

    // Runtime counters and timing for final summary output.
    size_t imuCount = 0;
    size_t visionFrameCount = 0;
    size_t visionFeatureCount = 0;
    // Current propagated filter time; starts once first IMU initializes gravity alignment.
    double currentTime = -1.0;
    // Rolling IMU buffer consumed at each vision timestamp.
    std::vector<IMUInput> imuBuffer;

    // Replay all events in sequence: IMU samples buffer propagation input, vision triggers correction.
    for (const ReplayEvent& event : log.events) {
      if (event.type == ReplayEvent::Type::Imu) {
        if (!filter.isInitialized()) {
          // One-time gravity-based attitude initialization from first IMU sample.
          filter.initializeFromIMU(event.imu);
          currentTime = event.imu.stamp;
        }

        // Keep all IMU events; buffered integration slices these into piecewise holds.
        imuBuffer.push_back(event.imu);
        ++imuCount;
      } else {
        // Propagate to this frame time using buffered IMU holds.
        const BufferedImuPropagation step =
            makeBufferedImuPropagation(imuBuffer, currentTime, event.tAbs);
        if (filter.isInitialized() && !step.imuInputs.empty()) {
          filter.propagate(step.imuInputs, step.dts);
          currentTime += step.propagatedTime;
        }
        // Drop IMU samples already consumed by propagation, keep boundary sample for continuity.
        if (step.trimCount > 0) {
          imuBuffer.erase(
              imuBuffer.begin(),
              imuBuffer.begin() +
                  static_cast<std::vector<IMUInput>::difference_type>(
                      step.trimCount));
        }

        // Build isotropic vision noise for this frame and apply one correction step.
        const Matrix R =
            Matrix::Identity(static_cast<int>(2 * event.vision.size()),
                             static_cast<int>(2 * event.vision.size())) *
            params.measurementNoiseVariance;
        filter.correct(event.vision, camera, R);
        ++visionFrameCount;
        visionFeatureCount += event.vision.size();
      }
    }

    // Report replay statistics and terminal estimate against hardcoded reference values.
    printSummary(log, params, currentTime, imuCount, visionFrameCount,
                 visionFeatureCount, filter.stateEstimate());

  } catch (const std::exception& e) {
    std::cerr << "EqVIOFilterExample failed: " << e.what() << "\n";
    return 2;
  }

  return 0;
}
