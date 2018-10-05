/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details geometry for 3D transformation and 6 DoF pose.
 */
#pragma once
#include <Eigen/Dense>
#include <vector>

namespace vs {

/** @brief skew conversion */
Eigen::Matrix3d skew(const Eigen::Vector3d& w);

/** @brief calculate rotate delta angle in radians between two rotation matrix
 * @param[in]R1: first rotation matrix
 * @param[in]R2: second rotation matrix
 * @return abstract delta angle in radians between R1 and R2
 */
double rotDiff(const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2);

/** @brief check orthogonality for input matrix, check with R*R'=I
 * @return true if R is orthogonal matrix
 */
bool checkOrthogonality(const Eigen::Matrix3d& R);

/** @brief calculate average for list of quaternions */
Eigen::Quaterniond quatMean(const std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>>& quats);

/** @brief calculate average for list of rotation matrix, convert to quaternion to calulate mean rotation */
Eigen::Matrix3d rotMean(const std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& Rs);

/** @brief calculate average for list of rotation matrix with SO3 */
Eigen::Matrix3d rotMeanSO3(const std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& Rs);

Eigen::Matrix3d expSO3(const Eigen::Vector3d& w);

Eigen::Vector3d logSO3(const Eigen::Matrix3d& R);

Eigen::Matrix4d expSE3(const Eigen::Matrix<double, 6, 1>& vec);

Eigen::Matrix<double, 6, 1> logSE3(const Eigen::Matrix4d& mat);

/** @brief build Omega matrix from input gyro, which is used in quaternion intergration
 * @param[in] w input angular velocity vector
 * @return Eigen::Matrix4d Omega matrix
 */
Eigen::Matrix4d Omega(Eigen::Vector3d w);

/** @brief convert transformation to vector
 * @return 6x1 vector, (tx,ty,tz,yaw,pitch,roll)
 */
Eigen::Matrix<double, 6, 1> isom2vec(const Eigen::Isometry3d& T);

/** @brief calculate conjugate Euler angle
 * @param[in]rpy: input Euler angle in roll-pitch-yaw
 * @return conjugate Euler angle in roll-pitch-yaw
 */
Eigen::Vector3d eulerConjugate(const Eigen::Vector3d& rpy);

/** @brief adjust euler angle roll-pitch-yaw
 * @param[in]rpy: euler angle roll-pitch-yaw
 * @param[in]type: 0-acute roll 1-positive roll
 * @return euler angle after adjustment
 * */
Eigen::Vector3d eulerAdjust(const Eigen::Vector3d& rpy, int type = 0);

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
Eigen::Matrix3d typicalRot(int type);

/** @brief calculate Euler angle from body rotation in world coordinate system */
Eigen::Vector3d Rbw2rpy(const Eigen::Matrix3d& R_b_w);

/** @brief principal component analysis for input data
 * @param[in]data: each row is a data sample
 * @param[out]eig_val: eigenvalue store in a vector
 * @param[out]eig_coef: covariance matrix consist in eigenvectors
 * @param[out]center: data center
 * @return whether success
 */
bool PCA(const Eigen::MatrixXd& data, Eigen::MatrixXd& eig_val, Eigen::MatrixXd& eig_coef, Eigen::MatrixXd& center);

/** @brief fit 3D line with all input 3D points
 * @param[in]data: input 3D points
 * @param[in]p1: first end-point of 3D line
 * @param[in]p2: second end-point of 3D line
 * @return true if line fit success
 */
bool lineFit(const Eigen::MatrixXd& data, Eigen::MatrixXd& p1, Eigen::MatrixXd& p2);

/**
 * @brief calculate intersection of 3D line to 3D plane
 * @param[in]plane_pt: arbitrary point in 3D plane
 * @param[in]plane_normal: normal vector of plane, must be unit vector
 * @param[in]line_pt: arbitrary point in 3D line
 * @param[in]line_direction: direction vector of 3D line, no necessary to be unit
 * @param[out]intersection: intersection point if exists
 * @return whether intersection exists, return false if and only if 3D line parallel to 3D plane
 */
bool intersectionLine2Plane(const Eigen::Vector3d& plane_pt, const Eigen::Vector3d& plane_normal,
                            const Eigen::Vector3d line_pt, const Eigen::Vector3d& line_direction,
                            Eigen::Vector3d& intersection);

/** @brief linear interpolation between two 3D poses
 * @param[in] a the first pose
 * @param[in] b the second pose
 * @param[in] t a factor number range in [0,1], 0 return a, 1 return b, 0~1 return linerpolation result
 * @return Eigen::Isometry3d
 */
inline Eigen::Isometry3d isomLerp(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b, double t) {
  if (t <= 0) {
    return a;
  } else if (t >= 1) {
    return b;
  } else {
    Eigen::Isometry3d isom = Eigen::Isometry3d::Identity();
    isom.translation() = (1 - t) * a.translation() + t * b.translation();
    Eigen::Quaterniond qa(a.linear());
    Eigen::Quaterniond qb(b.linear());
    isom.linear() = qa.slerp(t, qb).toRotationMatrix();
    return isom;
  }
}

/** @brief calculate the center of pts, subtract each point with center
 * @param[in,out] pts points to be centralized
 * @param[in] pts_mean center of points
 */
void ptsCentralize(std::vector<Eigen::Vector3d>& pts, Eigen::Vector3d& pts_mean);

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
bool umeyamaAlign(const std::vector<Eigen::Vector3d>& src_pts, const std::vector<Eigen::Vector3d>& tar_pts,
                  Eigen::Isometry3d& transform, double* scale = NULL);

} /* namespace vs */
