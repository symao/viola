/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_geometry3d.h"

#include "vs_basic.h"

namespace vs {

Eigen::Matrix3d skew(const Eigen::Vector3d& w) {
  Eigen::Matrix3d w_hat;
  w_hat << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
  return w_hat;
}

double rotDiff(const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2) {
  return fabs(Eigen::AngleAxisd(R1.transpose() * R2).angle());
}

bool checkOrthogonality(const Eigen::Matrix3d& R) {
  return (R.transpose() * R - Eigen::Matrix3d::Identity()).norm() < VS_EPS;
}

Eigen::Quaterniond quatMean(
    const std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>>& quats) {
  if (quats.empty()) return Eigen::Quaterniond();
  int sum = 1;
  auto mean_quat = quats[0];
  for (size_t i = 1; i < quats.size(); i++) {
    mean_quat = mean_quat.slerp(1.0 / (1.0 + sum), quats[i]);
    sum++;
  }
  return mean_quat;
}

Eigen::Matrix3d rotMean(const std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& Rs) {
  std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> quats;
  for (const auto& R : Rs) quats.push_back(Eigen::Quaterniond(R));
  auto mean_q = quatMean(quats);
  return mean_q.toRotationMatrix();
}

Eigen::Matrix3d rotMeanSO3(const std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& Rs) {
  if (Rs.empty()) return Eigen::Matrix3d();
  Eigen::Matrix3d R_ref = Rs[0];
  Eigen::Vector3d se3_sum(0, 0, 0);
  for (const auto& R : Rs) {
    auto v = logSO3(R_ref.transpose() * R);
    se3_sum += v;
  }
  return R_ref * expSO3(se3_sum / Rs.size());
}

Eigen::Matrix3d expSO3(const Eigen::Vector3d& w) {
  // get theta
  Eigen::Matrix3d w_x = skew(w);
  double theta = w.norm();
  // Handle small angle values
  double A, B;
  if (theta < VS_EPS) {
    A = 1;
    B = 0.5;
  } else {
    A = sin(theta) / theta;
    B = (1 - cos(theta)) / (theta * theta);
  }
  // compute so(3) rotation
  Eigen::Matrix3d R;
  if (theta == 0)
    R = Eigen::MatrixXd::Identity(3, 3);
  else
    R = Eigen::MatrixXd::Identity(3, 3) + A * w_x + B * w_x * w_x;
  return R;
}

Eigen::Vector3d logSO3(const Eigen::Matrix3d& R) {
  // magnitude of the skew elements (handle edge case where we sometimes have a>1...)
  double a = 0.5 * (R.trace() - 1);
  double theta = (a > 1) ? acos(1) : ((a < -1) ? acos(-1) : acos(a));
  // Handle small angle values
  double D;
  if (theta < VS_EPS)
    D = 0.5;
  else
    D = theta / (2 * sin(theta));

  // calculate the skew symetric matrix
  Eigen::Matrix3d w_x = D * (R - R.transpose());
  // check if we are near the identity
  if (R != Eigen::MatrixXd::Identity(3, 3)) {
    Eigen::Vector3d vec;
    vec << w_x(2, 1), w_x(0, 2), w_x(1, 0);
    return vec;
  } else {
    return Eigen::Vector3d::Zero();
  }
}

Eigen::Matrix4d expSE3(const Eigen::Matrix<double, 6, 1>& vec) {
  // Precompute our values
  Eigen::Vector3d w = vec.head(3);
  Eigen::Vector3d u = vec.tail(3);
  double theta = sqrt(w.dot(w));
  Eigen::Matrix3d wskew;
  wskew << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;

  // Handle small angle values
  double A, B, C;
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
  Eigen::Matrix3d I_33 = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d V = I_33 + B * wskew + C * wskew * wskew;

  // Get the final matrix to return
  Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
  mat.block(0, 0, 3, 3) = I_33 + A * wskew + B * wskew * wskew;
  mat.block(0, 3, 3, 1) = V * u;
  mat(3, 3) = 1;
  return mat;
}

Eigen::Matrix<double, 6, 1> logSE3(const Eigen::Matrix4d& mat) {
  // Get sub-matrices
  Eigen::Matrix3d R = mat.block(0, 0, 3, 3);
  Eigen::Vector3d t = mat.block(0, 3, 3, 1);

  // Get theta (handle edge case where we sometimes have a>1...)
  double a = 0.5 * (R.trace() - 1);
  double theta = (a > 1) ? acos(1) : ((a < -1) ? acos(-1) : acos(a));

  // Handle small angle values
  double A, B, D, E;
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
  Eigen::Matrix3d I_33 = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d wskew = D * (R - R.transpose());
  Eigen::Matrix3d Vinv = I_33 - 0.5 * wskew + E * wskew * wskew;

  // Calculate vector
  Eigen::Matrix<double, 6, 1> vec;
  vec.head(3) << wskew(2, 1), wskew(0, 2), wskew(1, 0);
  vec.tail(3) = Vinv * t;
  return vec;
}

Eigen::Matrix4d Omega(Eigen::Vector3d w) {
  Eigen::Matrix4d mat;
  mat.block(0, 0, 3, 3) = -skew(w);
  mat.block(3, 0, 1, 3) = -w.transpose();
  mat.block(0, 3, 3, 1) = w;
  mat(3, 3) = 0;
  return mat;
}

Eigen::Matrix<double, 6, 1> isom2vec(const Eigen::Isometry3d& T) {
  auto ypr = T.linear().eulerAngles(2, 1, 0);
  if (ypr(2) < 0) {
    ypr = T.linear().transpose().eulerAngles(0, 1, 2);
    std::swap(ypr(0), ypr(2));
  }
  auto t = T.translation();
  Eigen::Matrix<double, 6, 1> res;
  res << t(0), t(1), t(2), ypr(0), ypr(1), ypr(2);
  return res;
}

Eigen::Vector3d eulerConjugate(const Eigen::Vector3d& rpy) {
  return Eigen::Vector3d(normalizeRad(rpy(0) - VS_PI), normalizeRad(VS_PI - rpy(1)), normalizeRad(rpy(2) - VS_PI));
}

Eigen::Vector3d eulerAdjust(const Eigen::Vector3d& rpy, int type) {
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

Eigen::Matrix3d typicalRot(int type) {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
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

Eigen::Vector3d Rbw2rpy(const Eigen::Matrix3d& R_b_w) {
  auto ypr = R_b_w.eulerAngles(2, 1, 0);
  return eulerAdjust(Eigen::Vector3d(ypr(2), ypr(1), ypr(0)), 0);
}

bool PCA(const Eigen::MatrixXd& data, Eigen::MatrixXd& eig_val, Eigen::MatrixXd& eig_coef, Eigen::MatrixXd& center) {
  int N = data.rows();
  if (N <= 1) return false;
  auto c = data.colwise().mean();
  Eigen::MatrixXd d = data.rowwise() - c;
  Eigen::MatrixXd cov = d.transpose() * d;
  cov /= N - 1;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
  eig_val = es.eigenvalues();
  eig_coef = es.eigenvectors();
  center = c.transpose();
  return true;
}

bool lineFit(const Eigen::MatrixXd& data, Eigen::MatrixXd& p1, Eigen::MatrixXd& p2) {
  int n = data.rows();
  int k = data.cols();
  Eigen::MatrixXd eig_val, eig_coef, center;
  PCA(data, eig_val, eig_coef, center);
  int last_idx = k - 1;
  double lambda = eig_val(last_idx);
  if (lambda < 0.1) return false;
  for (int i = 0; i < last_idx; i++)
    if (eig_val(i) / lambda > 0.1) return false;
  auto dir = eig_coef.col(last_idx);
  dir /= dir.norm();
  double dmin = 0, dmax = 0;
  for (int i = 0; i < n; i++) {
    double d = dir.dot(data.row(i).transpose() - center);
    if (d < dmin)
      dmin = d;
    else if (d > dmax)
      dmax = d;
  }
  p1 = center + dmin * dir;
  p2 = center + dmax * dir;
  return true;
}

bool intersectionLine2Plane(const Eigen::Vector3d& plane_pt, const Eigen::Vector3d& plane_normal,
                            const Eigen::Vector3d line_pt, const Eigen::Vector3d& line_direction,
                            Eigen::Vector3d& intersection) {
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

void ptsCentralize(std::vector<Eigen::Vector3d>& pts, Eigen::Vector3d& pts_mean) {
  if (static_cast<int>(pts.size()) < 2) return;
  pts_mean.setZero();
  for (const auto& p : pts) pts_mean += p;
  pts_mean /= pts.size();
  for (auto& p : pts) p -= pts_mean;
}

bool umeyamaAlign(const std::vector<Eigen::Vector3d>& src_pts, const std::vector<Eigen::Vector3d>& tar_pts,
                  Eigen::Isometry3d& transform, double* scale) {
  int pts_cnt = src_pts.size();
  if (pts_cnt < 3 || src_pts.size() != tar_pts.size()) {
    printf("[ERROR]%s: input size not valid, src_pts:%d tar_pts:%d", __func__, pts_cnt,
           static_cast<int>(tar_pts.size()));
    return false;
  }
  auto src_pts_centroid = src_pts;
  auto tar_pts_centroid = tar_pts;
  Eigen::Vector3d src_mean, tar_mean;
  ptsCentralize(src_pts_centroid, src_mean);
  ptsCentralize(tar_pts_centroid, tar_mean);

  double sigma_x = 0;
  for (const auto& p : src_pts_centroid) sigma_x += p.squaredNorm();
  sigma_x /= pts_cnt;
  if (sigma_x < VS_EPS) {
    printf("[ERROR]%s: all src points seems in the same position. \n", __func__);
    return false;
  }

  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (int i = 0; i < pts_cnt; i++) cov += tar_pts_centroid[i] * src_pts_centroid[i].transpose();
  cov /= pts_cnt;

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
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

  Eigen::Matrix3d s = Eigen::Matrix3d::Identity();
  if (u.determinant() * v.determinant() < 0) s(2, 2) = -1;
  transform.setIdentity();
  transform.linear() = u * s * v.transpose();
  if (scale) {
    *scale = (d.asDiagonal() * s).trace() / sigma_x;
    transform.translation() = tar_mean - transform.linear() * src_mean * (*scale);
  } else {
    transform.translation() = tar_mean - transform.linear() * src_mean;
  }
  return true;
}

} /* namespace vs */