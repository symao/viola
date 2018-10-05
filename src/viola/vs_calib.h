/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details mono/stereo calibration data structure, camera-odometry hand-eye calibration.
 */
#pragma once
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <vector>

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
bool camOdomCalib(const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& cam_poses,
                  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& odom_poses,
                  Eigen::Isometry3d& T_c_o, int method = 1, bool verbose = true);

} /* namespace vs */