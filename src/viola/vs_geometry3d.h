/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details geometry for 3D transformation and 6 DoF pose.
 */
#pragma once
#include <Eigen/Dense>
#include <vector>
#include "vs_basic.h"
#include "vs_eigen_def.h"

namespace vs {

/** @brief skew conversion */
template <typename T>
Mat33_<T> skew(const Vec3_<T>& w) {
  Mat33_<T> w_hat;
  w_hat << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
  return w_hat;
}

/** @brief build Omega matrix from input gyro, which is used in quaternion intergration
 * @param[in] w input angular velocity vector
 * @return Mat44_<T> Omega matrix
 */
template <typename T>
Mat44_<T> Omega(Vec3_<T> w) {
  Mat44_<T> mat;
  mat.block(0, 0, 3, 3) = -skew(w);
  mat.block(3, 0, 1, 3) = -w.transpose();
  mat.block(0, 3, 3, 1) = w;
  mat(3, 3) = 0;
  return mat;
}

/** @brief calculate rotate delta angle in radians between two rotation matrix
 * @param[in]R1: first rotation matrix
 * @param[in]R2: second rotation matrix
 * @return abstract delta angle in radians between R1 and R2
 */
template <typename T>
T rotDiff(const Mat33_<T>& rot1, const Mat33_<T>& rot2) {
  return fabs(AngleAxis_<T>(rot1.transpose() * rot2).angle());
}

/** @brief check orthogonality for input matrix, check with R*R'=I
 * @return true if R is orthogonal matrix
 */
template <typename T>
bool checkOrthogonality(const Mat33_<T>& R) {
  return (R.transpose() * R - Mat33_<T>::Identity()).norm() < VS_EPS;
}

/** @brief calculate average for list of quaternions */
template <typename T>
Quat_<T> quatMean(const aligned_vector<Quat_<T>>& quats) {
  if (quats.empty()) return Quat_<T>();
  int sum = 1;
  auto mean_quat = quats[0];
  for (size_t i = 1; i < quats.size(); i++) {
    mean_quat = mean_quat.slerp(1.0 / (1.0 + sum), quats[i]);
    sum++;
  }
  return mean_quat;
}

/** @brief calculate average for list of rotation matrix, convert to quaternion to calulate mean rotation */
template <typename T>
Mat33_<T> rotMean(const aligned_vector<Mat33_<T>>& Rs) {
  aligned_vector<Quat_<T>> quats;
  for (const auto& R : Rs) quats.push_back(Quat_<T>(R));
  auto mean_q = quatMean(quats);
  return mean_q.toRotationMatrix();
}

/** @brief calculate average for list of rotation matrix with SO3 */
template <typename T>
Mat33_<T> rotMeanSO3(const aligned_vector<Mat33_<T>>& Rs) {
  if (Rs.empty()) return Mat33_<T>();
  Mat33_<T> R_ref = Rs[0];
  Vec3_<T> se3_sum(0, 0, 0);
  for (const auto& R : Rs) {
    auto v = logSO3(R_ref.transpose() * R);
    se3_sum += v;
  }
  return R_ref * expSO3(se3_sum / Rs.size());
}

template <typename T>
Mat33_<T> expSO3(const Vec3_<T>& w) {
  // get theta
  Mat33_<T> w_x = skew(w);
  T theta = w.norm();
  // Handle small angle values
  T A, B;
  if (theta < VS_EPS) {
    A = 1;
    B = 0.5;
  } else {
    A = sin(theta) / theta;
    B = (1 - cos(theta)) / (theta * theta);
  }
  // compute so(3) rotation
  Mat33_<T> R;
  if (theta == 0) {
    R = Mat33_<T>::Identity();
  } else {
    R = Mat33_<T>::Identity() + A * w_x + B * w_x * w_x;
  }
  return R;
}

template <typename T>
Vec3_<T> logSO3(const Mat33_<T>& R) {
  // magnitude of the skew elements (handle edge case where we sometimes have a>1...)
  T a = 0.5 * (R.trace() - 1);
  T theta = (a > 1) ? acos(1) : ((a < -1) ? acos(-1) : acos(a));
  // Handle small angle values
  T D = (theta < VS_EPS) ? 0.5 : (theta / (2 * sin(theta)));
  // calculate the skew symetric matrix
  Mat33_<T> w_x = D * (R - R.transpose());
  // check if we are near the identity
  if (R != Mat33_<T>::Identity()) {
    return Vec3_<T>(w_x(2, 1), w_x(0, 2), w_x(1, 0));
  } else {
    return Vec3_<T>::Zero();
  }
}

template <typename T>
Mat44_<T> expSE3(const Vec6_<T>& vec) {
  // Precompute our values
  Vec3_<T> w = vec.head(3);
  Vec3_<T> u = vec.tail(3);
  double theta = sqrt(w.dot(w));
  Mat33_<T> wskew = skew(w);

  // Handle small angle values
  T A, B, C;
  if (theta < VS_EPS) {
    A = 1;
    B = 0.5;
    C = 1.0 / 6.0;
  } else {
    A = sin(theta) / theta;
    B = (1 - cos(theta)) / (theta * theta);
    C = (1 - A) / (theta * theta);
  }

  // Matrices we need V and Identity
  Mat33_<T> I_33 = Mat33_<T>::Identity();
  Mat33_<T> V = I_33 + B * wskew + C * wskew * wskew;

  // Get the final matrix to return
  Mat44_<T> mat = Mat44_<T>::Zero();
  mat.block(0, 0, 3, 3) = I_33 + A * wskew + B * wskew * wskew;
  mat.block(0, 3, 3, 1) = V * u;
  mat(3, 3) = 1;
  return mat;
}

template <typename T>
Vec6_<T> logSE3(const Mat44_<T>& mat) {
  // Get sub-matrices
  Mat33_<T> R = mat.block(0, 0, 3, 3);
  Vec3_<T> t = mat.block(0, 3, 3, 1);

  // Get theta (handle edge case where we sometimes have a>1...)
  T a = 0.5 * (R.trace() - 1);
  T theta = (a > 1) ? acos(1) : ((a < -1) ? acos(-1) : acos(a));

  // Handle small angle values
  T A, B, D, E;
  if (theta < VS_EPS) {
    A = 1;
    B = 0.5;
    D = 0.5;
    E = 1.0 / 12.0;
  } else {
    A = sin(theta) / theta;
    B = (1 - cos(theta)) / (theta * theta);
    D = theta / (2 * sin(theta));
    E = 1 / (theta * theta) * (1 - 0.5 * A / B);
  }

  // Get the skew matrix and V inverse
  Mat33_<T> I_33 = Mat33_<T>::Identity();
  Mat33_<T> wskew = D * (R - R.transpose());
  Mat33_<T> Vinv = I_33 - 0.5 * wskew + E * wskew * wskew;

  // Calculate vector
  Vec6_<T> vec;
  vec.head(3) << wskew(2, 1), wskew(0, 2), wskew(1, 0);
  vec.tail(3) = Vinv * t;
  return vec;
}

template <typename T>
Mat44_<T> quatMultMatLeft(const Quat_<T>& q) {
  return (Mat44_<T>() << q.w(), -q.z(), q.y(), q.x(), q.z(), q.w(), -q.x(), q.y(), -q.y(), q.x(), q.w(), q.z(), -q.x(),
          -q.y(), -q.z(), q.w())
      .finished();
}

template <typename T>
Mat44_<T> quatMultMatRight(const Quat_<T>& q) {
  return (Mat44_<T>() << q.w(), q.z(), -q.y(), q.x(), -q.z(), q.w(), q.x(), q.y(), q.y(), -q.x(), q.w(), q.z(), -q.x(),
          -q.y(), -q.z(), q.w())
      .finished();
}

/** @brief convert transformation to vector
 * @return 6x1 vector, (tx,ty,tz,yaw,pitch,roll)
 */
template <typename T>
Vec6_<T> isom2vec(const Isom3_<T>& isom) {
  auto ypr = isom.linear().eulerAngles(2, 1, 0);
  if (ypr(2) < 0) {
    ypr = isom.linear().transpose().eulerAngles(0, 1, 2);
    std::swap(ypr(0), ypr(2));
  }
  auto t = isom.translation();
  return Vec6_<T>(t(0), t(1), t(2), ypr(0), ypr(1), ypr(2));
}

/** @brief calculate conjugate Euler angle
 * @param[in]rpy: input Euler angle in roll-pitch-yaw
 * @return conjugate Euler angle in roll-pitch-yaw
 */
template <typename T>
Vec3_<T> eulerConjugate(const Vec3_<T>& rpy) {
  return Vec3_<T>(normalizeRad(rpy(0) - VS_PI), normalizeRad(VS_PI - rpy(1)), normalizeRad(rpy(2) - VS_PI));
}

/** @brief adjust euler angle roll-pitch-yaw
 * @param[in]rpy: euler angle roll-pitch-yaw
 * @param[in]type: 0-acute roll 1-positive roll
 * @return euler angle after adjustment
 * */
template <typename T>
Vec3_<T> eulerAdjust(const Vec3_<T>& rpy, int type = 0) {
  switch (type) {
    case 0:
      if (fabs(rpy[0]) > VS_PI_2) return eulerConjugate(rpy);
      break;
    case 1:
      if (rpy[0] < 0) return eulerConjugate(rpy);
      break;
    default:
      break;
  }
  return rpy;
}

enum {
  ROT_RDF2FLU = 0,  ///< right-down-front to front-left-up
  ROT_FLU2RDF,      ///< front-left-up to right-down-front
  ROT_RDF2FRD,      ///< right-down-front to front-right-down
  ROT_FRD2RDF,      ///< front-right-down to right-down-front
  ROT_FLU2FRD,      ///< front-left-up to front-right-down
  ROT_FRD2FLU,      ///< front-right-down to front-left-up
  ROT_RBD2FLU,      ///< right-back-down to front-left up
  ROT_FLU2RBD,      ///< front-left up to right-back-down
  ROT_RBD2FRD,      ///< right-back-down to front-right-down
  ROT_FRD2RBD,      ///< front-right-down to right-back-down
};

template <typename T>
Mat33_<T> typicalRot(int type) {
  Mat33_<T> R = Mat33_<T>::Identity();
  switch (type) {
    case ROT_RDF2FLU:
      R << 0, 0, 1, -1, 0, 0, 0, -1, 0;
      break;
    case ROT_FLU2RDF:
      R << 0, -1, 0, 0, 0, -1, 1, 0, 0;
      break;
    case ROT_RDF2FRD:
      R << 0, 0, 1, 1, 0, 0, 0, 1, 0;
      break;
    case ROT_FRD2RDF:
      R << 0, 1, 0, 0, 0, 1, 1, 0, 0;
      break;
    case ROT_FLU2FRD:
      R << 1, 0, 0, 0, -1, 0, 0, 0, -1;
      break;
    case ROT_FRD2FLU:
      R << 1, 0, 0, 0, -1, 0, 0, 0, -1;
      break;
    case ROT_RBD2FLU:
      R << 0, -1, 0, -1, 0, 0, 0, 0, -1;
      break;
    case ROT_FLU2RBD:
      R << 0, -1, 0, -1, 0, 0, 0, 0, -1;
      break;
    case ROT_RBD2FRD:
      R << 0, -1, 0, 1, 0, 0, 0, 0, 1;
      break;
    case ROT_FRD2RBD:
      R << 0, 1, 0, -1, 0, 0, 0, 0, 1;
      break;
  }
  return R;
}

/** @brief calculate Euler angle from body rotation in world coordinate system */
template <typename T>
Vec3_<T> Rbw2rpy(const Mat33_<T>& R_b_w) {
  auto ypr = R_b_w.eulerAngles(2, 1, 0);
  return eulerAdjust(Vec3_<T>(ypr(2), ypr(1), ypr(0)), 0);
}

/** @brief principal component analysis for input data
 * @param[in]data: each row is a data sample
 * @param[out]eig_val: eigenvalue store in a vector
 * @param[out]eig_coef: covariance matrix consist in eigenvectors
 * @param[out]center: data center
 * @return whether success
 */
template <typename T>
bool PCA(const MatXX_<T>& data, VecX_<T>& eig_val, MatXX_<T>& eig_coef, VecX_<T>& center) {
  int N = data.rows();
  if (N <= 1) return false;
  auto c = data.colwise().mean();
  MatXX_<T> d = data.rowwise() - c;
  MatXX_<T> cov = d.transpose() * d;
  cov /= N - 1;
  Eigen::SelfAdjointEigenSolver<MatXX_<T>> es(cov);
  eig_val = es.eigenvalues();
  eig_coef = es.eigenvectors();
  center = c.transpose();
  return true;
}

/** @brief fit 3D line with all input 3D points
 * @param[in]data: input 3D points
 * @param[in]p1: first end-point of 3D line
 * @param[in]p2: second end-point of 3D line
 * @return true if line fit success
 */
template <typename T>
bool lineFit(const MatXX_<T>& data, MatXX_<T>& p1, MatXX_<T>& p2) {
  int n = data.rows();
  int k = data.cols();
  VecX_<T> eig_val, center;
  MatXX_<T> eig_coef;
  PCA(data, eig_val, eig_coef, center);
  int last_idx = k - 1;
  T lambda = eig_val(last_idx);
  if (lambda < 0.1) return false;
  for (int i = 0; i < last_idx; i++)
    if (eig_val(i) / lambda > 0.1) return false;
  auto dir = eig_coef.col(last_idx);
  dir /= dir.norm();
  T dmin = 0, dmax = 0;
  for (int i = 0; i < n; i++) {
    T d = dir.dot(data.row(i).transpose() - center);
    if (d < dmin) {
      dmin = d;
    } else if (d > dmax) {
      dmax = d;
    }
  }
  p1 = center + dmin * dir;
  p2 = center + dmax * dir;
  return true;
}

/**
 * @brief calculate intersection of 3D line to 3D plane
 * @param[in]plane_pt: arbitrary point in 3D plane
 * @param[in]plane_normal: normal vector of plane, must be unit vector
 * @param[in]line_pt: arbitrary point in 3D line
 * @param[in]line_direction: direction vector of 3D line, no necessary to be unit
 * @param[out]intersection: intersection point if exists
 * @return whether intersection exists, return false if and only if 3D line parallel to 3D plane
 */
template <typename T>
bool intersectionLine2Plane(const Vec3_<T>& plane_pt, const Vec3_<T>& plane_normal, const Vec3_<T> line_pt,
                            const Vec3_<T>& line_direction, Vec3_<T>& intersection) {
  float d = plane_normal.dot(plane_pt - line_pt);  // distance from line_pt to plane
  if (fabs(d) < VS_EPS) {
    intersection = line_pt;
    return true;
  }
  float d2 = plane_normal.dot(line_direction);  // distance of ray direction projection on plane normal
  if (fabs(d2) < VS_EPS) return false;
  intersection = line_pt + line_direction * (d / d2);
  return true;
}

/** @brief linear interpolation between two 3D poses
 * @param[in] a the first pose
 * @param[in] b the second pose
 * @param[in] t a factor number range in [0,1], 0 return a, 1 return b, 0~1 return linerpolation result
 * @return Isom3_<T>
 */
template <typename T>
inline Isom3_<T> isomLerp(const Isom3_<T>& a, const Isom3_<T>& b, double t) {
  if (t <= 0) {
    return a;
  } else if (t >= 1) {
    return b;
  } else {
    Isom3_<T> isom = Isom3_<T>::Identity();
    isom.translation() = (1 - t) * a.translation() + t * b.translation();
    Quat_<T> qa(a.linear());
    Quat_<T> qb(b.linear());
    isom.linear() = qa.slerp(t, qb).toRotationMatrix();
    return isom;
  }
}

/** @brief calculate the center of pts, subtract each point with center
 * @param[in,out] pts points to be centralized
 * @param[in] pts_mean center of points
 */
template <typename T>
void ptsCentralize(std::vector<Vec3_<T>>& pts, Vec3_<T>& pts_mean) {
  if (static_cast<int>(pts.size()) < 2) return;
  pts_mean.setZero();
  for (const auto& p : pts) pts_mean += p;
  pts_mean /= pts.size();
  for (auto& p : pts) p -= pts_mean;
}

/** @brief align points correspondences with Umeyama algorithms.
 * definition: target_point = transform * source_point * scale
 * reference: Umeyama, Shinji: Least-squares estimation of transformation parameters
              between two point patterns. IEEE PAMI, 1991
 * @param[in] src_pts source points
 * @param[in] tar_pts tart points, same size as src_pts
 * @param[out] transform transformation which tranform points from source to target
 * @param[in,out] scale if input pointer is not null, scale will be calculated.
 * @return bool whether alignment success
 */
template <typename T>
bool umeyamaAlign(const std::vector<Vec3_<T>>& src_pts, const std::vector<Vec3_<T>>& tar_pts, Isom3_<T>& transform,
                  double* scale = NULL) {
  int pts_cnt = src_pts.size();
  if (pts_cnt < 3 || src_pts.size() != tar_pts.size()) {
    printf("[ERROR]%s: input size not valid, src_pts:%d tar_pts:%d", __func__, pts_cnt,
           static_cast<int>(tar_pts.size()));
    return false;
  }
  auto src_pts_centroid = src_pts;
  auto tar_pts_centroid = tar_pts;
  Vec3_<T> src_mean, tar_mean;
  ptsCentralize(src_pts_centroid, src_mean);
  ptsCentralize(tar_pts_centroid, tar_mean);

  double sigma_x = 0;
  for (const auto& p : src_pts_centroid) sigma_x += p.squaredNorm();
  sigma_x /= pts_cnt;
  if (sigma_x < VS_EPS) {
    printf("[ERROR]%s: all src points seems in the same position. \n", __func__);
    return false;
  }

  Mat33_<T> cov = Mat33_<T>::Zero();
  for (int i = 0; i < pts_cnt; i++) cov += tar_pts_centroid[i] * src_pts_centroid[i].transpose();
  cov /= pts_cnt;

  Eigen::JacobiSVD<Mat33_<T>> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto& d = svd.singularValues();
  const auto& u = svd.matrixU();
  const auto& v = svd.matrixV();
  int count_nonzero = 0;
  for (int i = 0; i < d.rows(); i++)
    if (d[i] > VS_EPS) count_nonzero++;
  if (count_nonzero < d.rows() - 1) {
    printf("[ERROR]%s: Degenerate covariance rank(%.4f,%.4f,%.4f), Umeyama alignment is not possible\n", __func__,
           d.x(), d.y(), d.z());
    return false;
  }

  Mat33_<T> s = Mat33_<T>::Identity();
  if (u.determinant() * v.determinant() < 0) s(2, 2) = -1;
  transform.setIdentity();
  transform.linear() = u * s * v.transpose();
  if (scale) {
    *scale = (d.asDiagonal() * s).trace() / sigma_x;
    transform.translation() = (tar_mean - transform.linear() * src_mean * (*scale)) / (*scale);
  } else {
    transform.translation() = tar_mean - transform.linear() * src_mean;
  }
  return true;
}

template <typename FloatType>
class PoseOdometer {
 public:
  PoseOdometer() : m_dist(0), m_angle(0), m_gap(0), m_prev_pose_valid(false) {}

  void reset() {
    m_dist = m_angle = 0;
    m_gap = 0;
  }

  void update(const Isom3_<FloatType>& pose) {
    if (!m_prev_pose_valid) {
      m_prev_pose = pose;
      m_prev_pose_valid = true;
    } else {
      m_dist += (pose.translation() - m_prev_pose.translation()).norm();
      m_angle += AngleAxis_<FloatType>(pose.linear().transpose() * m_prev_pose.linear()).angle();
      m_gap++;
      m_prev_pose = pose;
    }
  }

  FloatType dist() const { return m_dist; }

  FloatType angle() const { return m_angle; }

  int gap() const { return m_gap; }

 private:
  FloatType m_dist;   ///< move distance [meters]
  FloatType m_angle;  ///< move angle [radians]
  int m_gap;          ///< move frame gap index
  bool m_prev_pose_valid;
  Isom3_<FloatType> m_prev_pose;
};
} /* namespace vs */
