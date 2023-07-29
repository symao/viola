/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details mono/stereo calibration data structure, camera-odometry hand-eye calibration.
 */
#pragma once
#include <opencv2/calib3d.hpp>
#include <vector>
#include "vs_geometry3d.h"

namespace vs {

struct MonoCalib {
  std::string name;
  double fx, fy, cx, cy;
  std::string distort_model;
  cv::Mat K;
  cv::Mat D;
  cv::Mat T_c_b;  // transformation point from camera to body

  bool load(const char* fcalib, const char* cam_name);

  void undistortImg(const cv::Mat& src, cv::Mat& dst, const cv::Mat& K_new);

  void undistortPts(const std::vector<cv::Point2f>& pts_in, std::vector<cv::Point2f>& pts_out,
                    const cv::Matx33d& rectify = cv::Matx33d::eye(),
                    const cv::Vec4d& new_intrin = cv::Vec4d(1, 1, 0, 0));

  std::vector<cv::Point2f> distortPts(const std::vector<cv::Point2f>& pts_in);
};

struct StereoVioCalib {
  cv::Size img_size;
  cv::Mat K0, K1, D0, D1;
  cv::Mat R_c0_c1, t_c0_c1;    // rotation/translation point from camera left to camera right
  cv::Mat R_c0_imu, t_c0_imu;  // rotation/translation point from camera left to body imu
  cv::Mat R_imu_body, t_imu_body;
  cv::Mat R_cam_gps, t_cam_gps;
  std::string distort_model;
  double time_delay;
  void deepCopy(const StereoVioCalib& rhs);
};

std::ostream& operator<<(std::ostream& os, const StereoVioCalib& c);

bool loadStereoVioCalib(const char* calib_file, StereoVioCalib& calib);

bool saveStereoVioCalib(const char* calib_file, const StereoVioCalib& calib);

class StereoRectifier {
 public:
  explicit StereoRectifier(bool enable_cl = false);
  explicit StereoRectifier(const char* calib_file, bool enable_cl = false);
  explicit StereoRectifier(const StereoVioCalib& calib_raw, bool enable_cl = false);

  bool init(const char* calib_file);
  bool init(const StereoVioCalib& calib_raw);

  void rectify(const cv::Mat& img0, const cv::Mat& img1, cv::Mat& out0, cv::Mat& out1);

  StereoVioCalib getCalibRaw() { return m_calib_raw; }

  StereoVioCalib getCalibRectify() { return m_calib_rectify; }

  bool rectified() { return m_rectified; }

  void setOpenCL(bool enable) { m_enable_cl = enable; }

 private:
  bool m_rectified;
  bool m_enable_cl;
  StereoVioCalib m_calib_raw;
  StereoVioCalib m_calib_rectify;
  cv::Mat m_rmap[2][2];
  cv::UMat m_urmap[2][2];

  void calcRectify();
};

namespace impl {
template <typename FloatType>
bool solveRxy(const aligned_vector<Isom3_<FloatType>>& cam_motions,
              const aligned_vector<Isom3_<FloatType>>& odom_motions, Mat33_<FloatType>& R) {
  int n = odom_motions.size();
  MatXX_<FloatType> T(4 * n, 4);
  T.setZero();
  Mat33_<FloatType> Rxy[2];
  for (int i = 0; i < n; i++) {
    Quat_<FloatType> ql(odom_motions[i].linear());
    Quat_<FloatType> qr(cam_motions[i].linear());
    ql.normalize();
    qr.normalize();
    T.template block<4, 4>(i * 4, 0) = quatMultMatLeft(ql) - quatMultMatRight(qr);
  }
  Eigen::JacobiSVD<MatXX_<FloatType>> svd(T, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto t1 = svd.matrixV().template block<4, 1>(0, 2);
  auto t2 = svd.matrixV().template block<4, 1>(0, 3);
  double a = (t1(0) * t1(1) + t1(2) * t1(3));
  double b = t1(1) * t2(0) + t1(0) * t2(1) + t1(3) * t2(2) + t1(2) * t2(3);
  double c = (t2(0) * t2(1) + t2(2) * t2(3));
  double delta = b * b - 4.0 * a * c;
  if (delta < 0) {
    return false;
  }
  double s[2];
  s[0] = (-b + sqrt(delta)) / (2.0 * a);
  s[1] = (-b - sqrt(delta)) / (2.0 * a);

  for (size_t i = 0; i != 2; i++) {
    auto lamda = s[i];
    double t = lamda * lamda * t1.dot(t1) + 2 * lamda * t1.dot(t2) + t2.dot(t2);
    double l2 = sqrt(1.0 / t);
    double l1 = lamda * l2;
    Quat_<FloatType> qxy;
    qxy.coeffs() = (l1 * t1 + l2 * t2);
    qxy.normalize();
    auto& R = Rxy[i];
    R = qxy.toRotationMatrix();
  }
  double yaw0 = atan2(Rxy[0](1, 0), Rxy[0](0, 0));
  double yaw1 = atan2(Rxy[1](1, 0), Rxy[1](0, 0));

  R = (fabs(yaw0) < fabs(yaw1)) ? Rxy[0] : Rxy[1];
  return true;
}
}  // namespace impl
/**
 * @brief Simple hand-eye calibration case for camera odomtry calibration, p_o = T_c_o * p_c
 * where odomtry is on ground plane, and camera poses can be collect by chessboard or marker.
 * Key eqution: $T_{o_i}_{o_{i+1}} * T_c_o = T_c_o * T_{c_i}_{c_{i+1}}$
 * ref:
 * 1. Guo C X, Mirzaei F M, Roumeliotis S I. An analytical least-squares solution to the odometer-camera extrinsic
 *    calibration problem[C]// IEEE International Conference on Robotics & Automation. 2012.
 * 2. Heng L , Li B , Pollefeys M . CamOdoCal: Automatic intrinsic and extrinsic calibration of a rig with multiple
 *    generic cameras and odometry[C]// IEEE/RSJ International Conference on Intelligent Robots & Systems. IEEE, 2013.
 * @note 1. z plane of camera poses must be the ground plane, which is the same as odomter z plane
 *       2. There is NO scale problem in cam poses. So one can put a chessboard or other marker on ground plane to
 *          collect camera poses with pnp.
 *       3. camera poses must be synchronized with odom poses.
 * @param[in]cam_poses: camera poses
 * @param[in]odom_poses: odometry poses
 * @param[out]T_c_o: transformation from camera frame to odometry frame
 * @param[in]method: solve method
 * @param[in]verbose: whether print solve log
 * @return whether calibration ok
 */
template <typename FloatType>
bool camOdomCalib(const aligned_vector<Isom3_<FloatType>>& cam_poses,
                  const aligned_vector<Isom3_<FloatType>>& odom_poses, Isom3_<FloatType>& T_c_o, int method = 1,
                  bool verbose = true) {
  T_c_o.setIdentity();
  int N = cam_poses.size();
  if (N < 3 || N != static_cast<int>(odom_poses.size())) {
    printf("[ERROR]vsCamOdomCalib: cam poses empty or not same size to odom poses\n");
    return false;
  }
  Mat33_<FloatType> Rxy;
  aligned_vector<Isom3_<FloatType>> T_cs, T_os;

  double height = 0.0;
  for (size_t i = 0; i < cam_poses.size(); i++) {
    height += cam_poses[i].translation()(2) / static_cast<double>(cam_poses.size());
  }

  for (int i = 1; i < N; i++) {
    T_cs.push_back(cam_poses[i].inverse() * cam_poses[i - 1]);
    T_os.push_back(odom_poses[i].inverse() * odom_poses[i - 1]);
  }

  switch (method) {
    case 0:
      if (!impl::solveRxy(T_cs, T_os, Rxy)) return false;
      break;
    case 1: {
      aligned_vector<Mat33_<FloatType>> Rxys;
      for (size_t i = 0; i < cam_poses.size(); i++) {
        Vec3_<FloatType> ypr = cam_poses[i].linear().eulerAngles(2, 1, 0);
        Mat33_<FloatType> Rxyi = (AngleAxis_<FloatType>(ypr(1), Vec3_<FloatType>::UnitY()) *
                                  AngleAxis_<FloatType>(ypr(2), Vec3_<FloatType>::UnitX()))
                                     .toRotationMatrix();
        Rxys.push_back(Rxyi);
      }
      Rxy = rotMean(Rxys);
    } break;
    case 2: {
      aligned_vector<Quat_<FloatType>> qxys;
      for (size_t i = 0; i < cam_poses.size(); i++) {
        Vec3_<FloatType> ypr = cam_poses[i].linear().eulerAngles(2, 1, 0);
        Quat_<FloatType> qxyi = (AngleAxis_<FloatType>(ypr(1), Vec3_<FloatType>::UnitY()) *
                                 AngleAxis_<FloatType>(ypr(2), Vec3_<FloatType>::UnitX()));
        qxys.push_back(qxyi);
      }
      Rxy = Quat_<FloatType>(quatMean(qxys)).toRotationMatrix();
    } break;
    default:
      break;
  }

  Mat44_<FloatType> A = Mat44_<FloatType>::Zero();
  Vec4_<FloatType> b = Vec4_<FloatType>::Zero();
  for (int i = 0; i < N - 1; i++) {
    MatXX_<FloatType> JK(2, 4);
    JK.leftCols(2) = T_os[i].matrix().template block<2, 2>(0, 0) - Mat22_<FloatType>::Identity();
    Vec3_<FloatType> p = Rxy * T_cs[i].translation();
    JK.rightCols(2) << p(0), -p(1), p(1), p(0);
    A += JK.transpose() * JK;
    b += -JK.transpose() * T_os[i].translation().topRows(2);
  }
  Vec4_<FloatType> x = A.inverse() * b;

  double tx = x(0);
  double ty = x(1);
  double scale = hypotf(x(2), x(3));
  double yaw = atan2(-x(3), -x(2));

  if (fabs(scale - 1) > 0.1) {
    printf("[ERROR]vsCamOdomCalib: wrong scale %f\n", scale);
    return false;
  }

  T_c_o.linear() = AngleAxis_<FloatType>(yaw, Vec3_<FloatType>::UnitZ()).toRotationMatrix() * Rxy;
  T_c_o.translation() << tx, ty, height;

  if (verbose) {
    auto t = T_c_o.translation();
    Vec3_<FloatType> v = Rbw2rpy<double>(T_c_o.linear() * typicalRot<double>(ROT_FLU2RDF)) * VS_RAD2DEG;
    printf("[INFO]CamOdomCalib t:(%.3f %.3f %.3f) euler-rpy:(%.3f %.3f %.3f)deg scale: %.3f\n", t(0), t(1), t(2), v(0),
           v(1), v(2), scale);
  }
  return true;
}

} /* namespace vs */