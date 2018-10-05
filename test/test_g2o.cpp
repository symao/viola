/**
 * Copyright (c) 2019-2021 shuyuanmao <maoshuyuan123 at gmail dot com>. All rights reserved.
 * @Author: shuyuanmao
 * @EMail: shuyuanmao123@gmail.com
 * @Date: 2019-05-19 02:07
 * @Description:
 */
#if HAVE_G2O
#include "vs_g2o_solver.h"
#include "vs_random.h"

void testPrior() {
  vs::G2OGraphPnPL graph;
  // graph.
  Eigen::Matrix3d R =
      (Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(-0.1, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX()))
          .toRotationMatrix();

  // graph.addPriorRot(R, Eigen::Matrix3d::Identity());

  Eigen::Isometry3d T;
  T.linear() = R;
  T.translation() << 1, -2, 3;
  graph.addPriorPose(T, g2o::Matrix6d::Identity());

  Eigen::Isometry3d pose;
  graph.solve(pose, 10, 0, 1);

  std::cout << "pose:" << pose.matrix() << std::endl;
  std::cout << "R:" << R << std::endl;
}

void testPnPL() {
  double f = 500;
  double cx = 320;
  double cy = 240;
  cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
  Eigen::Matrix3d R =
      (Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(-0.1, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX()))
          .matrix();
  Eigen::Vector3d t(1, -2, 10);

  std::vector<cv::Point3f> pts3d = {cv::Point3f(0, 0, 0), cv::Point3f(0.5, 0.5, 0), cv::Point3f(0.5, -0.5, 0),
                                    cv::Point3f(-0.5, -0.5, 0), cv::Point3f(-0.5, 0.5, 0)};
  std::vector<cv::Vec6f> lns3d = {cv::Vec6f(0.3, 0.4, 0, 0.5, -0.5, 0), cv::Vec6f(0.5, 0.5, 0, -0.5, 0.5, 0),
                                  cv::Vec6f(0, 0, 0, -0.5, -0.5, 0)};

  auto project = [=](double x, double y, double z) {
    Eigen::Vector3d pc = R * Eigen::Vector3d(x, y, z) + t;
    return cv::Point2f(f * pc(0) / pc(2) + cx, f * pc(1) / pc(2) + cy);
  };

  std::vector<cv::Point2f> pts2d;
  for (const auto& pt : pts3d) {
    pts2d.push_back(project(pt.x, pt.y, pt.z));
  }

  std::vector<cv::Vec4f> lns2d;
  for (const auto& ln : lns3d) {
    auto p1 = project(ln[0], ln[1], ln[2]);
    auto p2 = project(ln[3], ln[4], ln[5]);
    lns2d.push_back(cv::Vec4f(p1.x, p1.y, p2.x, p2.y));
  }

  // add noise
  float noise = 1.0f;
  float noise_line_k = 0.3;
  for (auto& pt : pts2d) {
    pt.x += vs::randf(-noise, noise);
    pt.y += vs::randf(-noise, noise);
  }
  for (auto& ln : lns2d) {
    float dx = ln[2] - ln[0];
    float dy = ln[3] - ln[1];
    float k1 = vs::randf(-noise_line_k, noise_line_k);
    float k2 = vs::randf(-noise_line_k, noise_line_k);
    ln[0] += dx * k1 + vs::randf(-noise, noise);
    ln[1] += dy * k1 + vs::randf(-noise, noise);
    ln[2] += dx * k2 + vs::randf(-noise, noise);
    ln[3] += dy * k2 + vs::randf(-noise, noise);
  }

  vs::G2OGraphPnPL graph;
  graph.setCamera(K);
  graph.addPointPairs(pts3d, pts2d);
  graph.addLinePairs(lns3d, lns2d);
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  // pose.linear() = R;
  // pose.translation() = t;
  // graph.computeGraphError(pose);
  pose.translation() << 1, 2, 3;
  bool ok = graph.solve(pose, 10, 1, 0);
  printf("ok:%d pose:\n", ok);
  std::cout << pose.matrix() << std::endl;
}

int main() {
  // testPrior();
  testPnPL();
}
#else  // HAVE_G2O
#include <stdio.h>

int main() { printf("[WARN]:G2O not build, set BUILD_G2O to 1 in CMakeLists.txt\n"); }
#endif