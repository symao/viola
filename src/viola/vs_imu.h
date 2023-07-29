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
#include "vs_geometry3d.h"

namespace vs {

template <typename FloatType>
struct ImuPreIntegrationResult {
  double sum_dt = 0;                          ///< [seconds]
  Vec3_<FloatType> delta_p;                   ///< delta position
  Quat_<FloatType> delta_q;                   ///< delta rotation in quaternion
  Vec3_<FloatType> delta_v;                   ///< delata velocity
  Vec3_<FloatType> linearized_ba;             ///< acc bias
  Vec3_<FloatType> linearized_bg;             ///< gyro bias
  Eigen::Matrix<FloatType, 15, 15> jacobian;  ///< jacobian matrix
};

// code from VINS-Mono(https://github.com/HKUST-Aerial-Robotics/VINS-Mono)
template <typename FloatType>
void midPointIntegration(double _dt, const Vec3_<FloatType>& _acc_0, const Vec3_<FloatType>& _gyr_0,
                         const Vec3_<FloatType>& _acc_1, const Vec3_<FloatType>& _gyr_1,
                         const Vec3_<FloatType>& delta_p, const Quat_<FloatType>& delta_q,
                         const Vec3_<FloatType>& delta_v, const Vec3_<FloatType>& linearized_ba,
                         const Vec3_<FloatType>& linearized_bg, Vec3_<FloatType>& result_delta_p,
                         Quat_<FloatType>& result_delta_q, Vec3_<FloatType>& result_delta_v,
                         Vec3_<FloatType>& result_linearized_ba, Vec3_<FloatType>& result_linearized_bg,
                         Eigen::Matrix<FloatType, 15, 15>& jacobian, bool update_jacobian) {
  Vec3_<FloatType> un_acc_0 = delta_q * (_acc_0 - linearized_ba);
  Vec3_<FloatType> un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
  result_delta_q = delta_q * Quat_<FloatType>(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
  Vec3_<FloatType> un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
  Vec3_<FloatType> un_acc = 0.5 * (un_acc_0 + un_acc_1);
  result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
  result_delta_v = delta_v + un_acc * _dt;
  // ba and bg donot change
  result_linearized_ba = linearized_ba;
  result_linearized_bg = linearized_bg;
  // jacobian to bias, used when the bias changes slightly and no need of repropagation
  if (update_jacobian) {
    // same as un_gyr, gyrometer reference to the local frame bk
    Vec3_<FloatType> w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
    // last acceleration measurement
    Vec3_<FloatType> a_0_x = _acc_0 - linearized_ba;
    // current acceleration measurement
    Vec3_<FloatType> a_1_x = _acc_1 - linearized_ba;
    // used for cross-product pay attention to derivation of matrix product
    Mat33_<FloatType> R_w_x, R_a_0_x, R_a_1_x;
    R_w_x << 0, -w_x(2), w_x(1), w_x(2), 0, -w_x(0), -w_x(1), w_x(0), 0;
    R_a_0_x << 0, -a_0_x(2), a_0_x(1), a_0_x(2), 0, -a_0_x(0), -a_0_x(1), a_0_x(0), 0;
    R_a_1_x << 0, -a_1_x(2), a_1_x(1), a_1_x(2), 0, -a_1_x(0), -a_1_x(1), a_1_x(0), 0;

    // error state model should use discrete format and mid-point approximation
    MatXX_<FloatType> F = MatXX_<FloatType>::Zero(15, 15);
    F.template block<3, 3>(0, 0) = Mat33_<FloatType>::Identity();
    F.template block<3, 3>(0, 3) =
        -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
        -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Mat33_<FloatType>::Identity() - R_w_x * _dt) * _dt * _dt;
    F.template block<3, 3>(0, 6) = MatXX_<FloatType>::Identity(3, 3) * _dt;
    F.template block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
    F.template block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
    F.template block<3, 3>(3, 3) = Mat33_<FloatType>::Identity() - R_w_x * _dt;
    F.template block<3, 3>(3, 12) = -1.0 * MatXX_<FloatType>::Identity(3, 3) * _dt;
    F.template block<3, 3>(6, 3) =
        -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
        -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Mat33_<FloatType>::Identity() - R_w_x * _dt) * _dt;
    F.template block<3, 3>(6, 6) = Mat33_<FloatType>::Identity();
    F.template block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
    F.template block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
    F.template block<3, 3>(9, 9) = Mat33_<FloatType>::Identity();
    F.template block<3, 3>(12, 12) = Mat33_<FloatType>::Identity();
    jacobian = F * jacobian;
  }
}

/** @brief IMU pre-integration, ref VINS-Fusion */
template <typename FloatType>
std::shared_ptr<ImuPreIntegrationResult<FloatType>> imuPreIntegrate(const std::vector<ImuData>& imus) {
  if (imus.empty()) return nullptr;
  std::shared_ptr<ImuPreIntegrationResult<FloatType>> res = std::make_shared<ImuPreIntegrationResult<FloatType>>();
  res->sum_dt = 0.0;
  res->delta_p.setZero();
  res->delta_q.setIdentity();
  res->delta_v.setZero();
  res->jacobian.setIdentity();
  const ImuData& imu_0 = imus[0];
  Vec3_<FloatType> acc_0(imu_0.acc[0], imu_0.acc[1], imu_0.acc[2]);
  Vec3_<FloatType> gyr_0(imu_0.gyro[0], imu_0.gyro[1], imu_0.gyro[2]);
  double t0 = imu_0.ts;
  for (unsigned j = 1; j < imus.size(); ++j) {
    const ImuData& imu_1 = imus[j];
    Vec3_<FloatType> acc_1(imu_1.acc[0], imu_1.acc[1], imu_1.acc[2]);
    Vec3_<FloatType> gyr_1(imu_1.gyro[0], imu_1.gyro[1], imu_1.gyro[2]);
    double t1 = imu_1.ts;
    double dt = t1 - t0;

    Vec3_<FloatType> result_delta_p;
    Quat_<FloatType> result_delta_q;
    Vec3_<FloatType> result_delta_v;
    Vec3_<FloatType> result_linearized_ba;
    Vec3_<FloatType> result_linearized_bg;
    midPointIntegration(dt, acc_0, gyr_0, acc_1, gyr_1, res->delta_p, res->delta_q, res->delta_v, res->linearized_ba,
                        res->linearized_bg, result_delta_p, result_delta_q, result_delta_v, result_linearized_ba,
                        result_linearized_bg, res->jacobian, 1);
    res->delta_p = result_delta_p;
    res->delta_q = result_delta_q;
    res->delta_v = result_delta_v;
    res->linearized_ba = result_linearized_ba;
    res->linearized_bg = result_linearized_bg;
    res->delta_q.normalize();
    res->sum_dt += dt;
    acc_0 = acc_1;
    gyr_0 = gyr_1;
    t0 = t1;
  }
  return res;
}

/** @brief integrate attitude quaternion with input angular velocity and delta time
 * @param[in] dt delta time in seconds
 * @param[in] gyro unbiased(substract gyroscope bias) gyroscope data, which is angular velocity
 * @param[in] q0 attitude quaternion at t0, hamilton quaternion of rotaion from body to world
 * @param[out] qt attitude quaternion at t0+dt, hamilton quaternion of rotaion from body to world
 */
template <typename FloatType>
void quatIntegral(double dt, const Vec3_<FloatType>& gyro, const Quat_<FloatType>& q0, Quat_<FloatType>& qt) {
  double dt_2 = dt / 2;
  double gyro_norm = gyro.norm();
  auto omega = Omega(gyro);
  Mat44_<FloatType> mult;
  if (gyro_norm > 1e-5) {
    mult = (cos(gyro_norm * dt_2) * Mat44_<FloatType>::Identity() + 1 / gyro_norm * sin(gyro_norm * dt_2) * omega);
  } else {
    mult = (Mat44_<FloatType>::Identity() + dt_2 * omega) * cos(gyro_norm * dt_2);
  }
  qt.coeffs() = mult * q0.coeffs();
  qt.normalize();
}

/** @brief integrate attitude quaternion with input angular velocity and delta time
 * @param[in] dt delta time in seconds
 * @param[in] gyro unbiased(substract gyroscope bias) gyroscope data, which is body angular velocity
 * @param[in] q0 attitude quaternion at t0, hamilton quaternion of rotaion from body to world
 * @param[out] qt attitude quaternion at t0+dt , hamilton quaternion of rotaion from body to world
 * @param[out] q_dt_2 attitude quaternion at t0+dt/2 , hamilton quaternion of rotaion from body to world
 */
template <typename FloatType>
void quatIntegral(double dt, const Vec3_<FloatType>& gyro, const Quat_<FloatType>& q0, Quat_<FloatType>& q_dt,
                  Quat_<FloatType>& q_dt_2) {
  double dt_2 = dt / 2;
  double dt_4 = dt / 4;
  double gyro_norm = gyro.norm();
  Mat44_<FloatType> omega = Omega(gyro);
  if (gyro_norm > 1e-5) {
    // q_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() + 1/gyro_norm*sin(gyro_norm*dt*0.5)*omega) * q;
    // q_dt_2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() + 1/gyro_norm*sin(gyro_norm*dt*0.25)*omega) * q;
    Vec4_<FloatType> oq = (1 / gyro_norm) * (omega * q0.coeffs());
    double temp = gyro_norm * dt_2;
    q_dt.coeffs() = cos(temp) * q0.coeffs() + sin(temp) * oq;
    temp /= 2;
    q_dt_2.coeffs() = cos(temp) * q0.coeffs() + sin(temp) * oq;
  } else {
    // q_dt = (Matrix4d::Identity()+0.5*dt*omega) * cos(gyro_norm*dt*0.5) * q;
    // q_dt_2 = (Matrix4d::Identity()+0.25*dt*omega) * cos(gyro_norm*dt*0.25) * q;
    Vec4_<FloatType> oq = omega * q0.coeffs();
    double c1 = cos(gyro_norm * dt_2);
    double c2 = cos(gyro_norm * dt_4);
    q_dt.coeffs() = c1 * q0.coeffs() + dt_2 * c1 * oq;
    q_dt_2.coeffs() = c2 * q0.coeffs() + dt_4 * c2 * oq;
  }
  q_dt.normalize();
  q_dt_2.normalize();
}

enum IntegralMethod {
  INTEGRAL_METHOD_EULER = 0,     ///< Euler method
  INTEGRAL_METHOD_MIDPOINT = 1,  ///< mid-point method
  INTEGRAL_METHOD_RK4 = 2,       ///< 4th order Runge-Kutta
};

template <typename FloatType>
void imuPredictEuler(double dt, const Vec3_<FloatType>& gyro, const Vec3_<FloatType>& acc, const Vec3_<FloatType>& g,
                     Quat_<FloatType>& q, Vec3_<FloatType>& p, Vec3_<FloatType>& v) {
  Quat_<FloatType> q_dt;
  quatIntegral(dt, gyro, q, q_dt);
  Vec3_<FloatType> k1_v_dot = q.toRotationMatrix() * acc + g;
  Vec3_<FloatType> k1_p_dot = v;
  q = q_dt;
  v = v + dt * k1_v_dot;
  p = p + dt * k1_p_dot;
}

template <typename FloatType>
void imuPredictMidPoint(double dt, const Vec3_<FloatType>& gyro, const Vec3_<FloatType>& acc, const Vec3_<FloatType>& g,
                        Quat_<FloatType>& q, Vec3_<FloatType>& p, Vec3_<FloatType>& v) {
  Quat_<FloatType> q_dt, q_dt_2;
  quatIntegral(dt, gyro, q, q_dt, q_dt_2);
  double dt_2 = dt / 2;
  Vec3_<FloatType> k1_v_dot = q.toRotationMatrix() * acc + g;
  Vec3_<FloatType> k2_v_dot = q_dt_2.toRotationMatrix() * acc + g;
  Vec3_<FloatType> k2_p_dot = v + k1_v_dot * dt_2;
  q = q_dt.normalized();
  v = v + dt * k2_v_dot;
  p = p + dt * k2_p_dot;
}

template <typename FloatType>
void imuPredictRK4(double dt, const Vec3_<FloatType>& gyro, const Vec3_<FloatType>& acc, const Vec3_<FloatType>& g,
                   Quat_<FloatType>& q, Vec3_<FloatType>& p, Vec3_<FloatType>& v) {
  Quat_<FloatType> q_dt, q_dt_2;
  quatIntegral(dt, gyro, q, q_dt, q_dt_2);
  double dt_2 = dt / 2;
  // k1 = f(tn, yn)
  Vec3_<FloatType> k1_v_dot = q.toRotationMatrix() * acc + g;
  Vec3_<FloatType> k1_p_dot = v;
  // k2 = f(tn+dt/2, yn+k1*dt/2)
  Vec3_<FloatType> k2_v_dot = q_dt_2.toRotationMatrix() * acc + g;
  Vec3_<FloatType> k2_p_dot = v + k1_v_dot * dt_2;
  // k3 = f(tn+dt/2, yn+k2*dt/2)
  Vec3_<FloatType> k3_v_dot = k2_v_dot;
  Vec3_<FloatType> k3_p_dot = v + k2_v_dot * dt_2;
  // k4 = f(tn+dt, yn+k3*dt)
  Vec3_<FloatType> k4_v_dot = q_dt.toRotationMatrix() * acc + g;
  Vec3_<FloatType> k4_p_dot = v + k3_v_dot * dt;
  // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
  q = q_dt.normalized();
  v = v + (dt / 6) * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot);
  p = p + (dt / 6) * (k1_p_dot + 2 * k2_p_dot + 2 * k3_p_dot + k4_p_dot);
}

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
template <typename FloatType>
void imuPredict(double dt, const Vec3_<FloatType>& gyro, const Vec3_<FloatType>& acc, const Vec3_<FloatType>& g,
                Quat_<FloatType>& q, Vec3_<FloatType>& p, Vec3_<FloatType>& v, int method = INTEGRAL_METHOD_RK4) {
  switch (method) {
    case INTEGRAL_METHOD_EULER:
      imuPredictEuler(dt, gyro, acc, g, q, p, v);
      break;
    case INTEGRAL_METHOD_MIDPOINT:
      imuPredictMidPoint(dt, gyro, acc, g, q, p, v);
      break;
    case INTEGRAL_METHOD_RK4:
    default:
      imuPredictRK4(dt, gyro, acc, g, q, p, v);
  }
}

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
  ImuStaticChecker() {
    const double acc_smooth_factor = 0.5;
    const double gyro_smooth_factor = 0.9;
    const double acc_diff_thres = 0.05;                    ///< [m/s^2]
    const double gyro_diff_thres = 0.5 * 3.1415926 / 180;  ///< [rad/s]
    const double acc_static_cnt_thres = 20;
    const double gyro_static_cnt_thres = 20;
    auto acc_diff_check = [acc_diff_thres](const Eigen::Vector3d& diff) { return diff.norm() < acc_diff_thres; };
    auto gyro_diff_check = [gyro_diff_thres](const Eigen::Vector3d& diff) { return diff.norm() < gyro_diff_thres; };
    m_acc_checker = std::make_shared<DataInvarianceChecker<Eigen::Vector3d>>(acc_diff_check, acc_smooth_factor,
                                                                             acc_static_cnt_thres);
    m_gyro_checker = std::make_shared<DataInvarianceChecker<Eigen::Vector3d>>(gyro_diff_check, gyro_smooth_factor,
                                                                              gyro_static_cnt_thres);
  }

  /** @brief check whether IMU is static
   * @param[in] ts current IMU timestamp in seconds
   * @param[in] acc current acc meta data
   * @param[in] gyro current gyro meta data
   * @return bool true if IMU is static, else if IMU is moving
   */
  template <typename FloatType>
  bool check(double ts, const Vec3_<FloatType>& acc, const Vec3_<FloatType>& gyro) {
    if (ts > m_ts + m_reset_ts_gap) {
      m_acc_checker->reset();
      m_gyro_checker->reset();
    }

    m_ts = ts;
    // Note: don't write 'm_acc_checker->check(acc) && m_gyro_checker->check(gyro)', since the second statement won't be
    // run if the first statement is false
    bool acc_static = m_acc_checker->check(acc);
    bool gyro_static = m_gyro_checker->check(gyro);
    return acc_static && gyro_static;
  }

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
template <typename FloatType>
bool calibMagnetometer(const aligned_vector<Vec3_<FloatType>>& mag_data, Vec3_<FloatType>& mag_bias,
                       FloatType& mag_norm_mean, FloatType& mag_norm_std) {
  int cnt = mag_data.size();
  if (cnt < 100) return false;  // no enough mag data
  // calib bias
  Mat44_<FloatType> A = Mat44_<FloatType>::Zero();
  Vec4_<FloatType> b = Vec4_<FloatType>::Zero();
  for (const auto& m : mag_data) {
    Vec4_<FloatType> tmp(2 * m(0), 2 * m(1), 2 * m(2), 1.f);
    A += tmp * tmp.transpose();
    b += tmp * m.squaredNorm();
  }
  Vec4_<FloatType> sol = A.ldlt().solve(b);
  mag_bias = sol.head(3);
  // calc std variance for unbiased magnetometer norm
  std::vector<FloatType> mag_norm_list;
  for (const auto& m : mag_data) mag_norm_list.push_back((m - mag_bias).norm());
  vecStatistics(mag_norm_list, mag_norm_mean, mag_norm_std);
  return true;
}

template <typename FloatType>
class ImuPredictor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Imu {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double ts;
    Vec3_<FloatType> gyro;
    Vec3_<FloatType> acc;
    using Ptr = std::shared_ptr<Imu>;

    Imu(double _ts, const Vec3_<FloatType>& _gyro, const Vec3_<FloatType>& _acc) : ts(_ts), gyro(_gyro), acc(_acc) {}
  };

  ImuPredictor() { reset(); }

  /** @brief clear all intermediate data */
  void reset() {
    ts_ = -1;
    imu_buf_.clear();
  }

  /** @brief update IMU meta data, integrate IMU pose with new IMU data */
  void updateImu(double ts, const Vec3_<FloatType>& gyro, const Vec3_<FloatType>& acc) {
    typename Imu::Ptr imu(new Imu(ts, gyro, acc));
    imu_buf_.emplace_back(imu);
    if (imu->ts > ts_) propagateOnce(imu);
  }

  /** @brief update IMU state, reset the updated state and integrate IMU pose from update ts to the latest IMU ts. */
  void updateState(double ts, const Quat_<FloatType>& q, const Vec3_<FloatType>& p, const Vec3_<FloatType>& v,
                   const Vec3_<FloatType>& ba, const Vec3_<FloatType>& bg, const Vec3_<FloatType>& gravity) {
    // remove imu buffer earlier than this state
    while (!imu_buf_.empty() && imu_buf_[0]->ts < ts) imu_buf_.pop_front();
    // reset state
    ts_ = ts;
    q_ = q;
    p_ = p;
    v_ = v;
    ba_ = ba;
    bg_ = bg;
    g_ = gravity;
    // imu propagate with imu buffer to newest time
    for (const auto& imu : imu_buf_) propagateOnce(imu);
  }

  /** @brief get the latest IMU pose, vel, bias
   * @param[in] ts timestamp [second]
   * @param[in] p IMU position
   * @param[in] q IMU quaternion
   * @param[in] v IMU velocity
   * @param[in] ba IMU acc bias
   * @param[in] bg IMU gyro bias
   * @return whether latest IMU pose ok
   */
  bool getLatestState(double& ts, Quat_<FloatType>& q, Vec3_<FloatType>& p, Vec3_<FloatType>& v, Vec3_<FloatType>& ba,
                      Vec3_<FloatType>& bg) {
    if (ts_ <= 0) return false;
    ts = ts_;
    q = q_;
    p = p_;
    v = v_;
    ba = ba_;
    bg = bg_;
    return true;
  }

  /** @brief whether latest IMU pose ok */
  bool ok() const { return ts_ > 0; }

  /** @brief get the latest IMU pose */
  Isom3_<FloatType> pose() const {
    Isom3_<FloatType> imu_pose = Isom3_<FloatType>::Identity();
    imu_pose.linear() = quaternionToRotation(q_).transpose();
    imu_pose.translation() = p_;
    return imu_pose;
  }

 private:
  std::deque<typename Imu::Ptr> imu_buf_;        ///< buffer of IMU meta data
  double ts_ = -1;                               ///< timestamp of current IMU state [seconds]
  Quat_<FloatType> q_;                           ///< current IMU quaternion
  Vec3_<FloatType> p_, v_;                       ///< current IMU position, velocity
  Vec3_<FloatType> g_, ba_, bg_;                 ///< current gravity, acc bias, gyro bias
  IntegralMethod method_ = INTEGRAL_METHOD_RK4;  ///< intergration method

  /** @brief integrate IMU pose with one IMU data */
  void propagateOnce(const typename Imu::Ptr& imu) {
    if (ts_ < 0) return;  // not init yet
    double dt = imu->ts - ts_;
    if (dt > 0.5 || dt < 0) {
      printf("ImuPredictor: %f: invalid dt %.3f. This cannot be happened.", imu->ts, dt);
      return;
    }
    imuPredict<FloatType>(dt, imu->gyro - bg_, imu->acc - ba_, g_, q_, p_, v_, method_);
    ts_ = imu->ts;
  }
};

}  // namespace vs