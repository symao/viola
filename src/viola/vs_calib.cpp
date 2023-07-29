/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_calib.h"
#include <opencv2/imgproc.hpp>
#include "vs_basic.h"
#include "vs_cv_convert.h"
#include "vs_yaml_parser.h"

namespace vs {

static bool checkR(const cv::Mat& R) {
  cv::Mat err = R * R.t() - cv::Mat::eye(3, 3, CV_64FC1);
  double* ptr = (double*)err.data;
  for (int i = 0; i < 9; i++)
    if (*ptr++ > 0.01) return false;
  return true;
}

bool MonoCalib::load(const char* fcalib, const char* cam_name) {
  name = std::string(cam_name);
  YamlParser reader(fcalib);
  if (!reader.isOpened()) return false;
#define MCREAD(a, n) reader.read<a>((name + "/" + std::string(n)).c_str())
  distort_model = MCREAD(std::string, "distortion_model");
  auto intrin = MCREAD(cv::Vec4d, "intrinsics");
  fx = intrin[0];
  fy = intrin[1];
  cx = intrin[2];
  cy = intrin[3];
  K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  D = cv::Mat(MCREAD(cv::Vec4d, "distortion_coeffs"));
  auto v0 = MCREAD(std::vector<double>, "T_body_cam");
  if (v0.size() != 16) return false;
  cv::Mat T_cb(4, 4, CV_64FC1, &v0[0]);
  cv::Mat R = T_cb(cv::Rect(0, 0, 3, 3));
  if (!checkR(R)) {
    printf("[ERROR]Calib read, R invalid.\n");
    return false;
  }
  T_c_b = T_cb.clone();
  return true;
}

void MonoCalib::undistortImg(const cv::Mat& src, cv::Mat& dst, const cv::Mat& K_new) {
  if (distort_model == "none")
    dst = src.clone();
  else if (distort_model == "radtan")
    cv::undistort(src, dst, K, D, K_new);
  else if (distort_model == "equidistant")
    cv::fisheye::undistortImage(src, dst, K, D, K_new);
  else
    printf("[ERROR]Unknown distort model '%s'\n", distort_model.c_str());
}

void MonoCalib::undistortPts(const std::vector<cv::Point2f>& pts_in, std::vector<cv::Point2f>& pts_out,
                             const cv::Matx33d& rectify, const cv::Vec4d& new_intrin) {
  if (pts_in.empty()) return;
  const cv::Matx33d K_new(new_intrin[0], 0.0, new_intrin[2], 0.0, new_intrin[1], new_intrin[3], 0.0, 0.0, 1.0);

  if (distort_model == "none")
    pts_out = pts_in;
  else if (distort_model == "radtan")
    cv::undistortPoints(pts_in, pts_out, K, D, rectify, K_new);
  else if (distort_model == "equidistant")
    cv::fisheye::undistortPoints(pts_in, pts_out, K, D, rectify, K_new);
  else
    printf("[ERROR]Unknown distort model '%s'\n", distort_model.c_str());
}

std::vector<cv::Point2f> MonoCalib::distortPts(const std::vector<cv::Point2f>& pts_in) {
  std::vector<cv::Point2f> pts_out;
  if (distort_model == "radtan") {
    std::vector<cv::Point3f> homogenous_pts;
    cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
    cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K, D, pts_out);
  } else if (distort_model == "equidistant") {
    cv::fisheye::distortPoints(pts_in, pts_out, K, D);
  } else
    printf("[ERROR]Unknown distort model '%s'\n", distort_model.c_str());
  return pts_out;
}

void StereoVioCalib::deepCopy(const StereoVioCalib& rhs) {
  img_size = rhs.img_size;
  K0 = rhs.K0.clone();
  K1 = rhs.K1.clone();
  D0 = rhs.D0.clone();
  D1 = rhs.D1.clone();
  R_c0_c1 = rhs.R_c0_c1.clone();
  t_c0_c1 = rhs.t_c0_c1.clone();
  R_c0_imu = rhs.R_c0_imu.clone();
  t_c0_imu = rhs.t_c0_imu.clone();
  R_imu_body = rhs.R_imu_body.clone();
  t_imu_body = rhs.t_imu_body.clone();
  distort_model = rhs.distort_model;
}

StereoRectifier::StereoRectifier(bool enable_cl) : m_rectified(false), m_enable_cl(enable_cl) {}

StereoRectifier::StereoRectifier(const char* calib_file, bool enable_cl) : m_rectified(false), m_enable_cl(enable_cl) {
  init(calib_file);
}

StereoRectifier::StereoRectifier(const StereoVioCalib& calib_raw, bool enable_cl)
    : m_rectified(false), m_enable_cl(enable_cl) {
  init(calib_raw);
}

bool StereoRectifier::init(const char* calib_file) {
  if (loadStereoVioCalib(calib_file, m_calib_raw)) calcRectify();
  return m_rectified;
}

bool StereoRectifier::init(const StereoVioCalib& calib_raw) {
  m_calib_raw.deepCopy(calib_raw);
  calcRectify();
  return m_rectified;
}

void StereoRectifier::rectify(const cv::Mat& img0, const cv::Mat& img1, cv::Mat& out0, cv::Mat& out1) {
  cv::Size s0 = img0.size();
  cv::Size s1 = img1.size();
  const cv::Size& sc = m_calib_raw.img_size;
  if (s0 != sc || s1 != sc) {
    printf(
        "[ERROR]StereoRectifier: Failed rectify, input size [%dx%d] [%dx%d]"
        " not match calib file [%dx%d]\n",
        s0.width, s0.height, s1.width, s1.height, sc.width, sc.height);
    return;
  }
  if (m_rectified) {
    if (m_enable_cl) {
      cv::UMat uimg0 = img0.getUMat(cv::ACCESS_READ);
      cv::UMat uimg1 = img1.getUMat(cv::ACCESS_READ);
      cv::UMat dst0;
      cv::UMat dst1;
      cv::remap(uimg0, dst0, m_urmap[0][0], m_urmap[0][1], cv::INTER_LINEAR);
      cv::remap(uimg1, dst1, m_urmap[1][0], m_urmap[1][1], cv::INTER_LINEAR);
      dst0.copyTo(out0);
      dst1.copyTo(out1);
    } else {
      cv::remap(img0, out0, m_rmap[0][0], m_rmap[0][1], cv::INTER_LINEAR);
      cv::remap(img1, out1, m_rmap[1][0], m_rmap[1][1], cv::INTER_LINEAR);
    }
  }
}

void StereoRectifier::calcRectify() {
  auto& c = m_calib_raw;
  cv::Mat R0, R1, P0, P1, Q;
  if (c.distort_model == "radtan") {
    cv::stereoRectify(c.K0, c.D0, c.K1, c.D1, c.img_size, c.R_c0_c1, c.t_c0_c1, R0, R1, P0, P1, Q,
                      cv::CALIB_ZERO_DISPARITY, 0);
    cv::initUndistortRectifyMap(c.K0, c.D0, R0, P0, c.img_size, CV_16SC2, m_rmap[0][0], m_rmap[0][1]);
    cv::initUndistortRectifyMap(c.K1, c.D1, R1, P1, c.img_size, CV_16SC2, m_rmap[1][0], m_rmap[1][1]);
  } else if (c.distort_model == "equidistant") {
    cv::fisheye::stereoRectify(c.K0, c.D0, c.K1, c.D1, c.img_size, c.R_c0_c1, c.t_c0_c1, R0, R1, P0, P1, Q,
                               cv::CALIB_ZERO_DISPARITY, c.img_size, 0, 1);
    cv::fisheye::initUndistortRectifyMap(c.K0, c.D0, R0, P0, c.img_size, CV_16SC2, m_rmap[0][0], m_rmap[0][1]);
    cv::fisheye::initUndistortRectifyMap(c.K1, c.D1, R1, P1, c.img_size, CV_16SC2, m_rmap[1][0], m_rmap[1][1]);
  } else {
    printf("[ERROR] Unknown distortion model '%s'.\n", c.distort_model.c_str());
    return;
  }

  auto& cr = m_calib_rectify;
  cr.img_size = c.img_size;
  cr.K0 = P0.colRange(0, 3).clone();
  cr.K1 = P1.colRange(0, 3).clone();
  cr.D0 = cr.D1 = cv::Mat();
  // T_c0r_c1r = T_c1_c1r * T_c0_c1 * T_c0r_c0
  cr.R_c0_c1 = R1 * c.R_c0_c1 * R0.t();
  cr.t_c0_c1 = R1 * c.t_c0_c1;
  // T_c0r_imu = T_c0_imu * T_c0r_c0
  cr.R_c0_imu = c.R_c0_imu * R0.t();
  cr.t_c0_imu = c.t_c0_imu.clone();
  cr.R_imu_body = c.R_imu_body.clone();
  cr.t_imu_body = c.t_imu_body.clone();
  cr.R_cam_gps = c.R_cam_gps.clone();
  cr.t_cam_gps = c.t_cam_gps.clone();
  cr.time_delay = c.time_delay;
#if 0
        WATCH(m_calib_raw);
        WATCH(m_calib_rectify);
        getchar();
#endif

  if (m_enable_cl) {
    m_urmap[0][0] = m_rmap[0][0].getUMat(cv::ACCESS_READ);
    m_urmap[0][1] = m_rmap[0][1].getUMat(cv::ACCESS_READ);
    m_urmap[1][0] = m_rmap[1][0].getUMat(cv::ACCESS_READ);
    m_urmap[1][1] = m_rmap[1][1].getUMat(cv::ACCESS_READ);
  }

  m_rectified = true;
}

bool loadStereoVioCalib(const char* calib_file, StereoVioCalib& calib) {
  YamlParser reader(calib_file, "r");
  if (!reader.isOpened()) {
    return false;
  }
  auto cam0_resolution = reader.read<cv::Vec2i>("cam0/resolution");
  auto cam0_intrinsics = reader.read<cv::Vec4d>("cam0/intrinsics");
  auto cam0_distortion_model = reader.read<std::string>("cam0/distortion_model", "radtan");
  auto cam0_distortion_coeffs = reader.read<cv::Vec4d>("cam0/distortion_coeffs");

  // auto cam1_resolution = reader.read<cv::Vec2i>("cam1/resolution");
  auto cam1_intrinsics = reader.read<cv::Vec4d>("cam1/intrinsics");
  // auto cam1_distortion_model = reader.read<std::string>("cam1/distortion_model", "radtan");
  auto cam1_distortion_coeffs = reader.read<cv::Vec4d>("cam1/distortion_coeffs");

  // NOTE: the name in calib file is opposite to the real transformation
  auto vec_T_imu_cam0 = reader.read<std::vector<double>>("cam0/T_cam_imu");
  auto vec_T_cam0_cam1 = reader.read<std::vector<double>>("cam1/T_cn_cnm1");
  auto vec_T_imu_body = reader.read<std::vector<double>>("T_imu_body");
  auto vec_T_cam_gps = reader.read<std::vector<double>>("T_cam_gps");

  calib.img_size = cv::Size(cam0_resolution[0], cam0_resolution[1]);
  calib.K0 = (cv::Mat_<double>(3, 3) << cam0_intrinsics[0], 0, cam0_intrinsics[2], 0, cam0_intrinsics[1],
              cam0_intrinsics[3], 0, 0, 1);
  calib.D0 = cv::Mat(cam0_distortion_coeffs);
  calib.K1 = (cv::Mat_<double>(3, 3) << cam1_intrinsics[0], 0, cam1_intrinsics[2], 0, cam1_intrinsics[1],
              cam1_intrinsics[3], 0, 0, 1);
  calib.D1 = cv::Mat(cam1_distortion_coeffs);
  calib.distort_model = cam0_distortion_model;

  calib.time_delay = reader.read<double>("cam0/timeshift_cam_imu", 0);

  auto foo_set_Rt = [](const std::vector<double>& vec, cv::Mat& R, cv::Mat& t) {
    if (vec.empty()) {
      R = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
      t = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    } else {
      T2Rt(vec2T(vec), R, t);
    }
  };

  cv::Mat R, t;
  foo_set_Rt(vec_T_imu_cam0, R, t);
  calib.R_c0_imu = R.t();
  calib.t_c0_imu = -R.t() * t;
  foo_set_Rt(vec_T_cam0_cam1, calib.R_c0_c1, calib.t_c0_c1);
  foo_set_Rt(vec_T_imu_body, calib.R_imu_body, calib.t_imu_body);
  foo_set_Rt(vec_T_cam_gps, calib.R_cam_gps, calib.t_cam_gps);

  return true;
}

bool saveStereoVioCalib(const char* calib_file, const StereoVioCalib& calib) {
#if 0
    YamlParser writer(calib_file, "w");
    if(!writer.isOpened())
    {
        return false;
    }
    auto& fs = writer.fs();
    fs<<"resolution"<<calib.img_size;
    fs<<"distort_model"<<calib.distort_model;
    fs<<"K0"<<calib.K0;
    fs<<"D0"<<calib.D0;
    fs<<"K1"<<calib.K1;
    fs<<"D1"<<calib.D1;
    fs<<"T_c0_c1"<<Rt2T(calib.R_c0_c1, calib.t_c0_c1);
    fs<<"T_c0_imu"<<Rt2T(calib.R_c0_imu, calib.t_c0_imu);
    fs<<"T_imu_c0"<<Rt2T(calib.R_c0_imu, calib.t_c0_imu).inv();
    fs<<"T_imu_body"<<Rt2T(calib.R_imu_body, calib.t_imu_body);
    return true;
#else
  FILE* fp = fopen(calib_file, "w");
  if (!fp) return false;

  auto fprint_mat = [&fp](const cv::Mat& m, int tab = 0) {
    double* ptr = (double*)m.data;
    for (int i = 0; i < tab; i++) fprintf(fp, " ");
    fprintf(fp, "[");
    int n = m.rows * m.cols;
    for (int i = 0; i < n - 1; i++) {
      fprintf(fp, "%f", *ptr++);
      if (i % m.cols == m.cols - 1) {
        fprintf(fp, ", \n");
        for (int i = 0; i < tab; i++) fprintf(fp, " ");
      } else
        fprintf(fp, ", ");
    }
    fprintf(fp, "%f]\n", *ptr);
  };

  fprintf(fp, "%%YAML:1.0\n");
  fprintf(fp, "cam0:\n");
  fprintf(fp, "  T_cam_imu:\n");
  fprint_mat(Rt2T(calib.R_c0_imu, calib.t_c0_imu).inv(), 4);
  fprintf(fp, "  camera_model: pinhole\n");
  fprintf(fp, "  distortion_coeffs: ");
  if (calib.D0.empty())
    fprintf(fp, "[0.0, 0.0, 0.0, 0.0]\n");
  else
    fprint_mat(calib.D0);
  fprintf(fp, "  distortion_model: radtan\n");
  fprintf(fp, "  intrinsics: [%f, %f, %f, %f]\n", calib.K0.at<double>(0, 0), calib.K0.at<double>(1, 1),
          calib.K0.at<double>(0, 2), calib.K0.at<double>(1, 2));
  fprintf(fp, "  resolution: [%d, %d]\n", calib.img_size.width, calib.img_size.height);
  fprintf(fp, "  timeshift_cam_imu: %.18f\n", calib.time_delay);

  fprintf(fp, "cam1:\n");
  fprintf(fp, "  T_cn_cnm1:\n");
  fprint_mat(Rt2T(calib.R_c0_c1, calib.t_c0_c1), 4);
  fprintf(fp, "  camera_model: pinhole\n");
  fprintf(fp, "  distortion_coeffs: ");
  if (calib.D0.empty())
    fprintf(fp, "[0.0, 0.0, 0.0, 0.0]\n");
  else
    fprint_mat(calib.D1);
  fprintf(fp, "  distortion_model: radtan\n");
  fprintf(fp, "  intrinsics: [%f, %f, %f, %f]\n", calib.K1.at<double>(0, 0), calib.K1.at<double>(1, 1),
          calib.K1.at<double>(0, 2), calib.K1.at<double>(1, 2));
  fprintf(fp, "  resolution: [%d, %d]\n", calib.img_size.width, calib.img_size.height);
  fprintf(fp, "  timeshift_cam_imu: %.18f\n", calib.time_delay);

  fprintf(fp, "T_imu_body:\n");
  fprint_mat(Rt2T(calib.R_imu_body, calib.t_imu_body), 2);

  fprintf(fp, "T_cam_gps:\n");
  fprint_mat(Rt2T(calib.R_cam_gps, calib.t_cam_gps), 2);

  fclose(fp);
  return true;
#endif
}

std::ostream& operator<<(std::ostream& os, const StereoVioCalib& c) {
  os << "image_size:" << c.img_size << " distort_model:" << c.distort_model << std::endl;
  os << "K0:" << c.K0 << std::endl;
  os << "D0:" << c.D0 << std::endl;
  os << "K1:" << c.K1 << std::endl;
  os << "D1:" << c.D1 << std::endl;
  os << "R_c0_c1:" << c.R_c0_c1 << std::endl;
  os << "t_c0_c1:" << c.t_c0_c1.t() << std::endl;
  os << "R_c0_imu:" << c.R_c0_imu << std::endl;
  os << "t_c0_imu:" << c.t_c0_imu.t() << std::endl;
  os << "R_imu_body:" << c.R_imu_body << std::endl;
  os << "t_imu_body:" << c.t_imu_body.t() << std::endl;
  os << "R_cam_gps:" << c.R_cam_gps << std::endl;
  os << "t_cam_gps:" << c.t_cam_gps.t() << std::endl;
  os << "time_delay:" << c.time_delay << std::endl;
  return os;
}

} /* namespace vs */