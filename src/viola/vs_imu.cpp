/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-06-29 14:56
 * @details
 */
#include "vs_imu.h"
#include "vs_basic.h"
#include "vs_geometry3d.h"
using namespace Eigen;

namespace vs {

// code from VINS-Mono(https://github.com/HKUST-Aerial-Robotics/VINS-Mono)
static void midPointIntegration(double _dt, const Vector3d& _acc_0, const Vector3d& _gyr_0, const Vector3d& _acc_1,
                                const Vector3d& _gyr_1, const Vector3d& delta_p, const Quaterniond& delta_q,
                                const Vector3d& delta_v, const Vector3d& linearized_ba, const Vector3d& linearized_bg,
                                Vector3d& result_delta_p, Quaterniond& result_delta_q, Vector3d& result_delta_v,
                                Vector3d& result_linearized_ba, Vector3d& result_linearized_bg,
                                Matrix<double, 15, 15>& jacobian, bool update_jacobian) {
  Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
  Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
  result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
  Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
  Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
  result_delta_v = delta_v + un_acc * _dt;
  // ba and bg donot change
  result_linearized_ba = linearized_ba;
  result_linearized_bg = linearized_bg;
  // jacobian to bias, used when the bias changes slightly and no need of repropagation
  if (update_jacobian) {
    // same as un_gyr, gyrometer reference to the local frame bk
    Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
    // last acceleration measurement
    Vector3d a_0_x = _acc_0 - linearized_ba;
    // current acceleration measurement
    Vector3d a_1_x = _acc_1 - linearized_ba;
    // used for cross-product pay attention to derivation of matrix product
    Matrix3d R_w_x, R_a_0_x, R_a_1_x;
    R_w_x << 0, -w_x(2), w_x(1), w_x(2), 0, -w_x(0), -w_x(1), w_x(0), 0;
    R_a_0_x << 0, -a_0_x(2), a_0_x(1), a_0_x(2), 0, -a_0_x(0), -a_0_x(1), a_0_x(0), 0;
    R_a_1_x << 0, -a_1_x(2), a_1_x(1), a_1_x(2), 0, -a_1_x(0), -a_1_x(1), a_1_x(0), 0;

    // error state model should use discrete format and mid-point approximation
    MatrixXd F = MatrixXd::Zero(15, 15);
    F.block<3, 3>(0, 0) = Matrix3d::Identity();
    F.block<3, 3>(0, 3) =
        -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
        -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
    F.block<3, 3>(0, 6) = MatrixXd::Identity(3, 3) * _dt;
    F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
    F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
    F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
    F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3, 3) * _dt;
    F.block<3, 3>(6, 3) =
        -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
        -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
    F.block<3, 3>(6, 6) = Matrix3d::Identity();
    F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
    F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
    F.block<3, 3>(9, 9) = Matrix3d::Identity();
    F.block<3, 3>(12, 12) = Matrix3d::Identity();
    jacobian = F * jacobian;
  }
}

std::shared_ptr<ImuPreIntegrationResult> imuPreIntegrate(const std::vector<ImuData>& imus) {
  if (imus.empty()) return nullptr;
  std::shared_ptr<ImuPreIntegrationResult> res = std::make_shared<ImuPreIntegrationResult>();
  res->sum_dt = 0.0;
  res->delta_p.setZero();
  res->delta_q.setIdentity();
  res->delta_v.setZero();
  res->jacobian.setIdentity();
  const ImuData& imu_0 = imus[0];
  Vector3d acc_0(imu_0.acc[0], imu_0.acc[1], imu_0.acc[2]);
  Vector3d gyr_0(imu_0.gyro[0], imu_0.gyro[1], imu_0.gyro[2]);
  double t0 = imu_0.ts;
  for (unsigned j = 1; j < imus.size(); ++j) {
    const ImuData& imu_1 = imus[j];
    Vector3d acc_1(imu_1.acc[0], imu_1.acc[1], imu_1.acc[2]);
    Vector3d gyr_1(imu_1.gyro[0], imu_1.gyro[1], imu_1.gyro[2]);
    double t1 = imu_1.ts;
    double dt = t1 - t0;

    Vector3d result_delta_p;
    Quaterniond result_delta_q;
    Vector3d result_delta_v;
    Vector3d result_linearized_ba;
    Vector3d result_linearized_bg;
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

void quatIntegral(double dt, const Eigen::Vector3d& gyro, const Eigen::Quaterniond& q0, Eigen::Quaterniond& qt) {
  double dt_2 = dt / 2;
  double gyro_norm = gyro.norm();
  Eigen::Matrix4d omega = Omega(gyro);
  Eigen::Matrix4d mult;
  if (gyro_norm > 1e-5) {
    mult = (cos(gyro_norm * dt_2) * Eigen::Matrix4d::Identity() + 1 / gyro_norm * sin(gyro_norm * dt_2) * omega);
  } else {
    mult = (Eigen::Matrix4d::Identity() + dt_2 * omega) * cos(gyro_norm * dt_2);
  }
  qt.coeffs() = mult * q0.coeffs();
  qt.normalize();
}

void quatIntegral(double dt, const Eigen::Vector3d& gyro, const Eigen::Quaterniond& q0, Eigen::Quaterniond& q_dt,
                  Eigen::Quaterniond& q_dt_2) {
  double dt_2 = dt / 2;
  double dt_4 = dt / 4;
  double gyro_norm = gyro.norm();
  Eigen::Matrix4d omega = Omega(gyro);
  if (gyro_norm > 1e-5) {
    // q_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() + 1/gyro_norm*sin(gyro_norm*dt*0.5)*omega) * q;
    // q_dt_2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() + 1/gyro_norm*sin(gyro_norm*dt*0.25)*omega) * q;
    Eigen::Vector4d oq = (1 / gyro_norm) * (omega * q0.coeffs());
    double temp = gyro_norm * dt_2;
    q_dt.coeffs() = cos(temp) * q0.coeffs() + sin(temp) * oq;
    temp /= 2;
    q_dt_2.coeffs() = cos(temp) * q0.coeffs() + sin(temp) * oq;
  } else {
    // q_dt = (Matrix4d::Identity()+0.5*dt*omega) * cos(gyro_norm*dt*0.5) * q;
    // q_dt_2 = (Matrix4d::Identity()+0.25*dt*omega) * cos(gyro_norm*dt*0.25) * q;
    Eigen::Vector4d oq = omega * q0.coeffs();
    double c1 = cos(gyro_norm * dt_2);
    double c2 = cos(gyro_norm * dt_4);
    q_dt.coeffs() = c1 * q0.coeffs() + dt_2 * c1 * oq;
    q_dt_2.coeffs() = c2 * q0.coeffs() + dt_4 * c2 * oq;
  }
  q_dt.normalize();
  q_dt_2.normalize();
}

static void imuPredictEuler(double dt, const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc,
                            const Eigen::Vector3d& g, Eigen::Quaterniond& q, Eigen::Vector3d& p, Eigen::Vector3d& v) {
  Eigen::Quaterniond q_dt;
  quatIntegral(dt, gyro, q, q_dt);
  Eigen::Vector3d k1_v_dot = q.toRotationMatrix() * acc + g;
  Eigen::Vector3d k1_p_dot = v;
  q = q_dt;
  v = v + dt * k1_v_dot;
  p = p + dt * k1_p_dot;
}

static void imuPredictMidPoint(double dt, const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc,
                               const Eigen::Vector3d& g, Eigen::Quaterniond& q, Eigen::Vector3d& p,
                               Eigen::Vector3d& v) {
  Eigen::Quaterniond q_dt, q_dt_2;
  quatIntegral(dt, gyro, q, q_dt, q_dt_2);
  double dt_2 = dt / 2;
  Eigen::Vector3d k1_v_dot = q.toRotationMatrix() * acc + g;
  Eigen::Vector3d k2_v_dot = q_dt_2.toRotationMatrix() * acc + g;
  Eigen::Vector3d k2_p_dot = v + k1_v_dot * dt_2;
  q = q_dt.normalized();
  v = v + dt * k2_v_dot;
  p = p + dt * k2_p_dot;
}

static void imuPredictRK4(double dt, const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc, const Eigen::Vector3d& g,
                          Eigen::Quaterniond& q, Eigen::Vector3d& p, Eigen::Vector3d& v) {
  Eigen::Quaterniond q_dt, q_dt_2;
  quatIntegral(dt, gyro, q, q_dt, q_dt_2);
  double dt_2 = dt / 2;
  // k1 = f(tn, yn)
  Eigen::Vector3d k1_v_dot = q.toRotationMatrix() * acc + g;
  Eigen::Vector3d k1_p_dot = v;
  // k2 = f(tn+dt/2, yn+k1*dt/2)
  Eigen::Vector3d k2_v_dot = q_dt_2.toRotationMatrix() * acc + g;
  Eigen::Vector3d k2_p_dot = v + k1_v_dot * dt_2;
  // k3 = f(tn+dt/2, yn+k2*dt/2)
  Eigen::Vector3d k3_v_dot = k2_v_dot;
  Eigen::Vector3d k3_p_dot = v + k2_v_dot * dt_2;
  // k4 = f(tn+dt, yn+k3*dt)
  Eigen::Vector3d k4_v_dot = q_dt.toRotationMatrix() * acc + g;
  Eigen::Vector3d k4_p_dot = v + k3_v_dot * dt;
  // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
  q = q_dt.normalized();
  v = v + (dt / 6) * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot);
  p = p + (dt / 6) * (k1_p_dot + 2 * k2_p_dot + 2 * k3_p_dot + k4_p_dot);
}

void imuPredict(double dt, const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc, const Eigen::Vector3d& g,
                Eigen::Quaterniond& q, Eigen::Vector3d& p, Eigen::Vector3d& v, int method) {
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

ImuStaticChecker::ImuStaticChecker() {
  const double acc_smooth_factor = 0.5;
  const double gyro_smooth_factor = 0.9;
  const double acc_diff_thres = 0.05;                    ///< [m/s^2]
  const double gyro_diff_thres = 0.5 * 3.1415926 / 180;  ///< [rad/s]
  const double acc_static_cnt_thres = 20;
  const double gyro_static_cnt_thres = 20;
  auto acc_diff_check = [acc_diff_thres](const Eigen::Vector3d& diff) { return diff.norm() < acc_diff_thres; };
  auto gyro_diff_check = [gyro_diff_thres](const Eigen::Vector3d& diff) { return diff.norm() < gyro_diff_thres; };
  m_acc_checker =
      std::make_shared<DataInvarianceChecker<Eigen::Vector3d>>(acc_diff_check, acc_smooth_factor, acc_static_cnt_thres);
  m_gyro_checker = std::make_shared<DataInvarianceChecker<Eigen::Vector3d>>(gyro_diff_check, gyro_smooth_factor,
                                                                            gyro_static_cnt_thres);
}

bool ImuStaticChecker::check(double ts, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro) {
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

bool calibMagnetometer(const std::vector<Eigen::Vector3d>& mag_data, Eigen::Vector3d& mag_bias, double& mag_norm_mean,
                       double& mag_norm_std) {
  int cnt = mag_data.size();
  if (cnt < 100) return false;  // no enough mag data
  // calib bias
  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  Eigen::Vector4d b = Eigen::Vector4d::Zero();
  for (const auto& m : mag_data) {
    Eigen::Vector4d tmp(2 * m(0), 2 * m(1), 2 * m(2), 1.f);
    A += tmp * tmp.transpose();
    b += tmp * m.squaredNorm();
  }
  Eigen::Vector4d sol = A.ldlt().solve(b);
  mag_bias = sol.head(3);
  // calc std variance for unbiased magnetometer norm
  std::vector<double> mag_norm_list;
  for (const auto& m : mag_data) mag_norm_list.push_back((m - mag_bias).norm());
  vecStatistics(mag_norm_list, mag_norm_mean, mag_norm_std);
  return true;
}
}  // namespace vs