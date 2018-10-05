/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details common data structure in VIO
 */
#pragma once
#include <vector>
#include <opencv2/core.hpp>

#define VS_VIO_ENABLE_MAG 1

namespace vs {
/** @brief raw imu data with synchronized acceleromete and gyroscope sensor data */
struct ImuData {
  double ts = -1;              ///< sampling timestamp, second
  double acc[3] = {0, 0, 0};   ///< acceleration, m/s^2
  double gyro[3] = {0, 0, 0};  ///< angular velocity, rad/s
#if VS_VIO_ENABLE_MAG
  double mag[3] = {0, 0, 0};  ///< magnetometer vector, uT
#endif
};

/** @brief camera data */
struct CameraData {
  double ts = -1;             ///< sampling timestamp, second
  std::vector<cv::Mat> imgs;  ///< image list, support one or more camera
};

/** @brief pose data, such as ground truth pose */
struct PoseData {
  double ts = -1;                         ///< sampling timestamp, second
  double qw = 0, qx = 0, qy = 0, qz = 0;  ///< quaternion rotation data
  double tx = 0, ty = 0, tz = 0;          ///< translation/position data
  double vx = 0, vy = 0, vz = 0;          ///< velocity data
  double bw_x = 0, bw_y = 0, bw_z = 0;    ///< gyroscope bias
  double ba_x = 0, ba_y = 0, ba_z = 0;    ///< accelorator bias
};
}  // namespace vs