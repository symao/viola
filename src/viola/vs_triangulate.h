/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details two-view/multi-view triangulation
 */
#pragma once
#include <Eigen/Dense>
#include <vector>

namespace vs {

/** @brief triangulate point with two views
 * @param[in]cam_pose1: first camera pose
 * @param[in]uv1: normalized plane observation on first camera
 * @param[in]cam_pose2: second camera pose
 * @param[in]uv2: normalized plane observation on second camera
 * @param[out]pos: estimated position in world
 * @return true if estimation ok
 * @note uv in normalized camera plane, thus not need to input camera intrinsic
 */
bool triangulateTwoView(const Eigen::Isometry3d& cam_pose1, const Eigen::Vector2d& uv1,
                        const Eigen::Isometry3d& cam_pose2, const Eigen::Vector2d& uv2, Eigen::Vector3d& pos);

/** @brief triangulate point with mulitple views
 * @param[in]cam_poses: camera pose list
 * @param[in]uvs: normalized plane observations on each camera
 * @param[out]pos: estimated position in world
 * @return true if estimation ok
 * @note uv in normalized camera plane, thus not need to input camera intrinsic
 */
bool triangulateMultiViews(const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& cam_poses,
                           const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& uvs,
                           Eigen::Vector3d& pos);

/** @brief calculate triangulation error which is the average distance from triangulated position to each ray. */
double triangulateError(const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& cam_poses,
                        const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& uvs,
                        const Eigen::Vector3d& pos);

double reprojectionError(const Eigen::Isometry3d& pose, const Eigen::Vector2d& uv, const Eigen::Vector3d& pw);

double reprojectionError(const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& cam_poses,
                         const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& uvs,
                         const Eigen::Vector3d& pos);

bool twoPointRansac(const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& uvs1,
                    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& uvs2,
                    const Eigen::Matrix3d& R_1_to_2, std::vector<uint8_t>& status, double focal_length,
                    double inlier_thres, double success_probability = 0.99);

class LinearTriangulator {
 public:
  LinearTriangulator();

  void update(double u0, double v0, const Eigen::Isometry3d& cam_pose);

  bool good() const { return good_flag_; }

  Eigen::Vector3d pos() const { return pos_; }

 private:
  Eigen::Vector3d pos_;
  Eigen::Matrix3d A_;
  Eigen::Vector3d b_;
  Eigen::Vector3d pt_cam_;
  bool good_flag_ = false;
  int observe_cnt_ = 0;
  double xmin_ = 0, ymin_ = 0, xmax_ = 0, ymax_ = 0;
  double parallex_ = 0;
};

}  // namespace vs