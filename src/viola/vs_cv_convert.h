/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details data convertion between Eigen and opencv.
 */
#pragma once
#include <string>
#include <opencv2/calib3d.hpp>
#ifdef EIGEN_MAJOR_VERSION
#include <opencv2/core/eigen.hpp>
#endif  // EIGEN_MAJOR_VERSION
#include <opencv2/core/affine.hpp>
namespace vs {

/** @brief construct transformation matrix from rotation matrix and translation vector
 * @param[in]R: rotation matrix, 3x3
 * @param[in]tvec: translation vector, 3x1
 * @return transformation matrix, 4x4
 */
inline cv::Mat Rt2T(const cv::Mat& R, const cv::Mat& tvec) {
  cv::Mat T = cv::Mat::eye(4, 4, CV_64FC1);
  R.convertTo(T(cv::Rect(0, 0, 3, 3)), CV_32FC1);
  tvec.convertTo(T(cv::Rect(3, 0, 1, 3)), CV_32FC1);
  return T;
}

/** @brief construct transformation matrix from rotation vector and translation vector
 * @param[in]R: rotation vector, 3x1, convert to rotation matrix using Rodrigues
 * @param[in]tvec: translation vector, 3x1
 * @return transformation matrix, 4x4
 */
inline cv::Mat rt2T(const cv::Mat& rvec, const cv::Mat& tvec) {
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  return Rt2T(R, tvec);
}

/** @brief convert rotation vector to rotation matrix */
inline cv::Mat r2R(const cv::Mat& rvec) {
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  return R;
}

/** @brief construct transformation matrix from rotation matrix, set translation vector to zero */
inline cv::Mat R2T(const cv::Mat& R) {
  cv::Mat T = cv::Mat::eye(4, 4, CV_64FC1);
  R.copyTo(T.rowRange(0, 3).colRange(0, 3));
  return T;
}

/** @brief grab rotation matrix and translation vector from transformation matrix
 * @param[in]T: transformation matrix, 4x4
 * @param[in]R: rotation matrix, 3x3
 * @param[in]tvec: translation vector, 3x1
 */
inline void T2Rt(const cv::Mat& T, cv::Mat& R, cv::Mat& t) {
  T.rowRange(0, 3).colRange(0, 3).copyTo(R);
  T.rowRange(0, 3).col(3).copyTo(t);
}

/** @brief convert data vector(size is 16) to 4x4 transformation matrix */
inline cv::Mat vec2T(const std::vector<double>& v) {
  assert(v.size() == 16);
  cv::Mat T(4, 4, CV_64FC1);
  double* Tdata = reinterpret_cast<double*>(T.data);
  for (int i = 0; i < 16; i++) {
    Tdata[i] = v[i];
  }
  return T;
}

/** @brief construct camera matrix with input focal length and optical center */
inline cv::Mat camMat(double fx, double fy, double cx, double cy) {
  return (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

#ifdef EIGEN_MAJOR_VERSION

/** @brief convert 2d point, cv to eigen */
inline Eigen::Vector2d cvt2d(const cv::Point2f& a) { return Eigen::Vector2d(a.x, a.y); }

/** @brief convert 2d point, eigen to cv */
inline cv::Point2f cvt2d(const Eigen::Vector2d& a) { return cv::Point2f(a(0), a(1)); }

/** @brief convert 3d point, cv to eigen */
inline Eigen::Vector3d cvt3d(const cv::Point3f& a) { return Eigen::Vector3d(a.x, a.y, a.z); }

/** @brief convert 3d point, eigen to cv */
inline cv::Point3f cvt3d(const Eigen::Vector3d& a) { return cv::Point3f(a(0), a(1), a(2)); }

/** @brief convert matrix, cv to eigen */
inline Eigen::MatrixXd cvtMat(const cv::Mat& a) {
  Eigen::MatrixXd eig_mat;
  cv2eigen(a, eig_mat);
  return eig_mat;
}

/** @brief convert matrix, eigen to cv */
inline cv::Mat cvtMat(const Eigen::MatrixXd& a) {
  cv::Mat cv_mat;
  eigen2cv(a, cv_mat);
  return cv_mat;
}

/** @brief construct Eigen::Isometry3d transformation with input quaternion and translation */
inline Eigen::Isometry3d isom(double qw, double qx, double qy, double qz, double tx, double ty, double tz) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
  T.translation() << tx, ty, tz;
  return T;
}

/** @brief construct Eigen::Isometry3d 4x4 tramsformation matrix in row major */
inline Eigen::Isometry3d isom(double m00, double m01, double m02, double m03, double m10, double m11, double m12,
                              double m13, double m20, double m21, double m22, double m23, double m30 = 0,
                              double m31 = 0, double m32 = 0, double m33 = 1) {
  Eigen::Isometry3d isom;
  isom.matrix() << m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33;
  return isom;
}

/** @brief construct Eigen::Isometry3d transformation with input rotation matrix and translation vector */
inline Eigen::Isometry3d Rt2isom(const cv::Mat& R, const cv::Mat& tvec) {
  Eigen::Matrix3d eigR;
  Eigen::Vector3d eigt;
  cv2eigen(R, eigR);
  cv2eigen(tvec, eigt);
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = eigR;
  T.translation() = eigt;
  return T;
}

/** @brief construct Eigen::Isometry3d transformation with input rotation vector and translation vector */
inline Eigen::Isometry3d rt2isom(const cv::Mat& rvec, const cv::Mat& tvec) {
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  return Rt2isom(R, tvec);
}

/** @brief grab rotation vector and translation vector from Eigen::Isometry3d transformation */
inline void isom2rt(const Eigen::Isometry3d& T, cv::Mat& rvec, cv::Mat& tvec) {
  cv::Mat R;
  eigen2cv(Eigen::Matrix3d(T.linear()), R);
  eigen2cv(Eigen::Vector3d(T.translation()), tvec);
  cv::Rodrigues(R, rvec);
}

/** @brief convert Eigen::Isometry3d transformation to cv::Affine3f */
inline cv::Affine3f isom2affine(const Eigen::Isometry3d& isom) {
  cv::Mat T;
  eigen2cv(isom.matrix(), T);
  T.convertTo(T, CV_32FC1);
  return cv::Affine3f(T);
}

/** @brief convert data vector(size is 16) to 4x4 transformation */
inline Eigen::Isometry3d vec2isom(const std::vector<double>& v, bool row_major = true) {
  assert(v.size() == 16);
  Eigen::Isometry3d isom;
  if (row_major) {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) isom.matrix()(i, j) = v[i * 4 + j];
  } else {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) isom.matrix()(j, i) = v[i * 4 + j];
  }
  return isom;
}

/** @brief convert mat to string for print */
inline std::string mat2str(const Eigen::MatrixXd& mat, int buf_len = 1024) {
  char* buf = new char[buf_len];
  int idx = 0;
  idx += snprintf(buf + idx, buf_len - idx, "%dx%d [", static_cast<int>(mat.rows()), static_cast<int>(mat.cols()));
  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++) {
      idx += snprintf(buf + idx, buf_len - idx, "%.3f", mat(i, j));
      if (j == mat.cols() - 1) {
        if (i < mat.rows() - 1) idx += snprintf(buf + idx, buf_len - idx, ", ");
      } else {
        idx += snprintf(buf + idx, buf_len - idx, ",");
      }
    }
  }
  idx += snprintf(buf + idx, buf_len - idx, "]");
  std::string str(buf);
  delete[] buf;
  return str;
}

/** @brief convert isometry to string for print
 * @param[in]isom: isometry
 * @param[in]mode: 0-4x4 matrix  1-pose+quat  2-pose+angleaxis
 * @return std::string
 */
inline std::string isom2str(const Eigen::Isometry3d& isom, int mode = 0) {
  if (mode == 0) {
    return mat2str(isom.matrix());
  } else if (mode == 1) {
    const auto& p = isom.translation();
    Eigen::Quaterniond q(isom.linear());
    char* buf = new char[1024];
    snprintf(buf, 1024, "pos:(%.3f, %.3f, %.3f) quat_wxyz:(%.3f, %.3f, %.3f, %.3f)", p.x(), p.y(), p.z(), q.w(), q.x(),
             q.y(), q.z());
    std::string str(buf);
    delete[] buf;
    return str;
  } else if (mode == 2) {
    const auto& p = isom.translation();
    Eigen::AngleAxisd angleaxis(isom.linear());
    auto r = angleaxis.axis() * angleaxis.angle();
    char* buf = new char[1024];
    snprintf(buf, 1024, "pos:(%.3f, %.3f, %.3f) rvec:(%.3f, %.3f, %.3f)", p.x(), p.y(), p.z(), r.x(), r.y(), r.z());
    std::string str(buf);
    delete[] buf;
    return str;
  } else {
    return "";
  }
}

#endif  // EIGEN_MAJOR_VERSION

} /* namespace vs */
