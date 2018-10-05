/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details perspective-n-points algorithms, closed form for translation-only pnp, pnl, as well as 4-Dof pnp, pnl.
 */
#pragma once
#include <opencv2/calib3d.hpp>

namespace vs {

/** @brief project 2 points. assert K as eye33, so 2d points are projected into normalized plane
 * @param[in]pt1: the first point in 3D
 * @param[in]uv1: the first projected point in 2D plane
 * @param[in]pt2: the second point in 3D
 * @param[in]uv2: the second projected point in 2D plane
 * @param[out]trans: translation which satisfy: s*uv_i = pt_i + trans for i=1, 2
 */
void p2pTrans(const cv::Point3f& pt1, const cv::Point2f& uv1, const cv::Point3f& pt2, const cv::Point2f& uv2,
              cv::Point3f& trans, bool refine = false);

/** @brief project n points, need at least 2 points. assert K as eye33, so 2d points are projected into normalized plane
 * @param[in]pts3d: points list in 3D
 * @param[in]pts2d: projected points list in 2D plane
 * @param[out]trans: translation which satisfy: s*uv_i = pt_i + trans for i=0, 1, ..., n-1
 */
void pnpTrans(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d, cv::Point3f& trans,
              bool refine = false);

/** @brief project n points, need at least 2 points. assert K as eye33, so 2d points are projected into normalized plane
 * @param[in]pts3d: points list in 3D
 * @param[in]pts2d: projected points list in 2D plane
 * @param[out]trans: translation which satisfy: s*uv_i = pt_i + trans for i=0, 1, ..., n-1
 * @param[out]inliers: inliers flag, the same size as pts3d, 1:inlier, 0:outlier
 * @param[in]max_ite: max iteration count for RANSAC
 * @param[in]reprj_err: the max reprojection error for inliers. Note, this error is in normailzed
 *            image plane, rather than image plane. Thus, reprj_err = pixel_error / focal_length
 *  @return if RANSAC succeed. If RANSAC run to max iterate count, this will return false.
 */
bool pnpTransRansac(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d, cv::Point3f& trans,
                    std::vector<uchar>& inliers, int max_ite = 1000, float reprj_err = 0.1, bool refine = false);

/** @brief pnp to calculate translation and yaw. assert K as eye33, so 2d points are projected into normalized plane,
 *         assert the transformation as: s*uv_i = R * pt_i + trans for i = 0, 1, ..., n-1
 *         where R = c, -s, 0, s, c, 0, 0, 0, 1, c=cos(yaw), s=sin(yaw)
 * @param[in]pts3d: points list in 3D
 * @param[in]pts2d: projected points list in 2D plane
 * @param[out]trans: the result translation
 * @param[out]yaw: the result yaw angle in [deg]
 * @note projection su = Rz(yaw)p + t, where u in pts2d and p in pts3d
 */
void pnpTransYaw(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d, cv::Point3f& trans,
                 float& yaw);

/** @brief project n lines, need at least 3 lines
 *      assert K as eye33, so 2d lines are projected into normalized plane
 *      mathmatics: Support p1,p2 are point of 2D line in normalized plane, $\bar{p1},\bar{p2}$ are
 *      3D point in camera coordinate system. Then $O-\bar{p1}-\bar{p2}$ is a plane, which we want
 *      line3d lie on it, where O is origin. The normal of the plane $n = cross(\bar{p1},\bar{p1})$,
 *      Support one endpoint of 3D line is q, then $dist = n^T(q+t) = 0$, thus $n^T t=-n^T q$
 *      Since the rotation is fix, thus minimizing the dist from two endpoints of 3D line to plane
 *      is equivalent to minimizing the dist from center point of 3D line to plane.
 * @param[in]lns3d: line list in 3D with format (x1, y1, z1, x2, y2, z2)
 * @param[in]lns2d: projected line list in 2D plane with format (u1, v1, u2, v2)
 * @param[out]trans: translation which satisfy: lns2d_i // lns3d_i + trans for i=0, 1, ..., n-1
 */
void pnlTrans(const std::vector<cv::Vec6f>& lns3d, const std::vector<cv::Vec4f>& lns2d, cv::Point3f& trans);

/** @brief project n points and lines, need at least 3 lines, or at least 2 points
 *     assert K as eye33, so 2d lines are projected into normalized plane
 * @param[in]pts3d: points list in 3D
 * @param[in]pts2d: projected points list in 2D plane
 * @param[in]lns3d: line list in 3D with format (x1, y1, z1, x2, y2, z2)
 * @param[in]lns2d: projected line list in 2D plane with format (u1, v1, u2, v2)
 * @param[out]trans: translation which satisfy: lns2d_i//(lns3d_i+trans) and s*uv_i = pt_i+trans
 */
void pnplTrans(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d,
               const std::vector<cv::Vec6f>& lns3d, const std::vector<cv::Vec4f>& lns2d, cv::Point3f& trans,
               const std::vector<float>& line_weights = {});

/** @brief pnp with RANSAC to calculate translation and yaw
 *        assert K as eye33, so 2d points are projected into normalized plane
 *        assert the transformation as: s*uv_i = R * pt_i + trans for i = 0, 1, ..., n-1
 *        where R = c, -s, 0, s, c, 0, 0, 0, 1, c=cos(yaw), s=sin(yaw)
 * @param[in]pts3d: points list in 3D
 * @param[in]pts2d: projected points list in 2D plane
 * @param[out]trans: the result translation
 * @param[out]yaw: the result yaw angle in [rad]
 * @param[out]inliers: inliers flag, the same size as pts3d, 1:inlier, 0:outlier
 * @param[in]ite_cnt: max iterate count
 * @param[in]reprj_err: the max reprojection error for inliers. Note, this error is in normailzed
 *              image plane, rather than image plane. Thus, reprj_err = pixel_error / focal_length
 * @return if RANSAC succeed. If RANSAC run to max iterate count, this will return false.
 * @note projection su = Rz(yaw)p + t, where u in pts2d and p in pts3d
 */
bool pnpRansacTransYaw(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d, cv::Point3f& trans,
                       float& yaw, std::vector<uchar>& inliers, int ite_cnt = 100, float reprj_err = 0.1);

/** @brief pnp with RANSAC using OpenCV
 * @param[in]pts3d: points list in 3D
 * @param[in]pts2d: projected points list in 2D plane
 * @param[in]K: camera intrisin matrix
 * @param[in]D: camera distortion
 * @param[in,out]rvec: rotation vector, which transform pts3d from world to camera coordinate system
 * @param[in,out]tvec: translation vector, which transform pts3d from world to camera coordinate system
 * @param[out]status: inliers flag, the same size as pts3d, >0:inlier, 0:outlier
 * @param[in]use_init_guess: whether use input rvec,tvec as initial guess when solving PnP
 * @param[in]max_ite: max iterate count
 * @param[in]reprj_err: the max reprojection error for inliers. Note, this error is in normailzed
 *                      image plane, rather than image plane. Thus, reprj_err = pixel_error / focal_length
 * @return if RANSAC succeed. If RANSAC run to max iterate count, this will return false.
 * @note projection su = Rz(yaw)p + t, where u in pts2d and p in pts3d
 */
bool pnpSolverOpenCV(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d, const cv::Mat& K,
                     const cv::Mat& D, cv::Mat& rvec, cv::Mat& tvec, std::vector<uchar>& status,
                     bool use_init_guess = false, bool do_ransac = true, int ransac_max_ite = 100,
                     double reproject_err = 3, int refine_cnt = 0, bool centroid = false);

class PointCentralizer {
 public:
  std::vector<cv::Point3f> process(const std::vector<cv::Point3f>& pts3d);

  void forwardRt(const cv::Mat& rvec, cv::Mat& tvec);

  void backwardRt(const cv::Mat& rvec, cv::Mat& tvec);

 private:
  bool mean_vec_valid_ = false;
  cv::Mat mean_vec_;
};

} /* namespace vs */
