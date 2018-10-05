/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-06-29 14:56
 * @details IMU numerical integration, pre-integration, ZUPT trigger.
 */
#pragma once
#include <memory>
#include <Eigen/Dense>
#include "vs_vio_type.h"

namespace vs {

struct ImuPreIntegrationResult {
  double sum_dt = 0;                       ///< [seconds]
  Eigen::Vector3d delta_p;                 ///< delta position
  Eigen::Quaterniond delta_q;              ///< delta rotation in quaternion
  Eigen::Vector3d delta_v;                 ///< delata velocity
  Eigen::Vector3d linearized_ba;           ///< acc bias
  Eigen::Vector3d linearized_bg;           ///< gyro bias
  Eigen::Matrix<double, 15, 15> jacobian;  ///< jacobian matrix
};

/** @brief IMU pre-integration, ref VINS-Fusion */
std::shared_ptr<ImuPreIntegrationResult> imuPreIntegrate(const std::vector<ImuData>& imus);

/** @brief integrate attitude quaternion with input angular velocity and delta time
 * @param[in] dt delta time in seconds
 * @param[in] gyro unbiased(substract gyroscope bias) gyroscope data, which is angular velocity
 * @param[in] q0 attitude quaternion at t0, hamilton quaternion of rotaion from body to world
 * @param[out] qt attitude quaternion at t0+dt, hamilton quaternion of rotaion from body to world
 */
void quatIntegral(double dt, const Eigen::Vector3d& gyro, const Eigen::Quaterniond& q0, Eigen::Quaterniond& qt);

/** @brief integrate attitude quaternion with input angular velocity and delta time
 * @param[in] dt delta time in seconds
 * @param[in] gyro unbiased(substract gyroscope bias) gyroscope data, which is body angular velocity
 * @param[in] q0 attitude quaternion at t0, hamilton quaternion of rotaion from body to world
 * @param[out] qt attitude quaternion at t0+dt , hamilton quaternion of rotaion from body to world
 * @param[out] q_dt_2 attitude quaternion at t0+dt/2 , hamilton quaternion of rotaion from body to world
 */
void quatIntegral(double dt, const Eigen::Vector3d& gyro, const Eigen::Quaterniond& q0, Eigen::Quaterniond& q_dt,
                  Eigen::Quaterniond& q_dt_2);

enum IntegralMethod {
  INTEGRAL_METHOD_EULER = 0,     ///< Euler method
  INTEGRAL_METHOD_MIDPOINT = 1,  ///< mid-point method
  INTEGRAL_METHOD_RK4 = 2,       ///< 4th order Runge-Kutta
};

/** @brief IMU integrate with 'un-noised' gyro and acc, using 4th order Runge-Kutta
    @param[in] dt: time duration
    @param[in] gyro: gyro = gyro mata - gyro bias
    @param[in] acc: acc = acc mata - acc bias
    @param[in] g: gravity vector
    @param[in,out] q: quaternion to be integrated, hamilton quaternion of rotaion from body to world
    @param[in,out] p: position to be integrated
    @param[in,out] v: velocity to be integrated
    @param[in] method: 0: euler  1:mid-point  2: 4th order Runge-Kutta
*/
void imuPredict(double dt, const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc, const Eigen::Vector3d& g,
                Eigen::Quaterniond& q, Eigen::Vector3d& p, Eigen::Vector3d& v, int method = INTEGRAL_METHOD_RK4);

/** @brief check whether timed data is invariant */
template <class T>
class DataInvarianceChecker {
 public:
  typedef std::function<bool(const T& diff)> DiffCheckFunc;

  DataInvarianceChecker(DiffCheckFunc diff_func, double smooth_factor = 0.5, int static_cnt_thres = 10)
      : m_diff_check_func(diff_func), m_smooth_factor(smooth_factor), m_static_cnt_thres(static_cnt_thres) {}

  void reset() {
    m_filter_data_valid = false;
    m_static_cnt = 0;
  }

  /** @brief input sensor data, check whether data is invariant
   * @param[in] data input sensor data
   * @return true if data is invariant, false if data is moving
   */
  bool check(const T& data) {
    if (m_filter_data_valid) {
      m_filter_data = m_filter_data * m_smooth_factor + data * (1 - m_smooth_factor);
    } else {
      m_filter_data = data;
      m_filter_data_valid = true;
    }

    T diff = data - m_filter_data;
    if (m_diff_check_func(diff)) {
      m_static_cnt++;
    } else {
      m_static_cnt = 0;
    }
    return m_static_cnt > m_static_cnt_thres;
  }

 private:
  DiffCheckFunc m_diff_check_func;   ///< function to check whether is small enough
  double m_smooth_factor;            ///< smooth factor for data filter
  int m_static_cnt_thres;            ///< minimum consecutive static count for static
  bool m_filter_data_valid = false;  ///< whether filter data is valid
  int m_static_cnt = 0;              ///< current consecutive static count
  T m_filter_data;                   ///< current filter data
};

/** @brief check whether IMU is in stationary with input IMU sensor data flow. Used in ZUPT */
class ImuStaticChecker {
 public:
  ImuStaticChecker();

  /** @brief check whether IMU is static
   * @param[in] ts current IMU timestamp in seconds
   * @param[in] acc current acc meta data
   * @param[in] gyro current gyro meta data
   * @return bool true if IMU is static, else if IMU is moving
   */
  bool check(double ts, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro);

 private:
  double m_ts = 0;  ///< current timestamp in seconds
  std::shared_ptr<DataInvarianceChecker<Eigen::Vector3d>> m_acc_checker;
  std::shared_ptr<DataInvarianceChecker<Eigen::Vector3d>> m_gyro_checker;
  const double m_reset_ts_gap = 0.5;  ///< [seconds] if input ts > m_ts + m_reset_ts_gap, then do reset
};

/** @brief calib magnetometer bias
 * @param[in] mag_data magnetometer meta data list
 * @param[out] mag_bias magnetormeter bias
 * @param[out] mag_norm_mean unbiased magnetometer data norm mean
 * @param[out] mag_norm_std unbiased magnetometer data norm standard deviation
 * @return bool whether calibration ok
 */
bool calibMagnetometer(const std::vector<Eigen::Vector3d>& mag_data, Eigen::Vector3d& mag_bias, double& mag_norm_mean,
                       double& mag_norm_std);

}  // namespace vs