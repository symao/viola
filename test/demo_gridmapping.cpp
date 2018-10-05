/**
 * Copyright (c) 2019-2021 shuyuanmao <maoshuyuan123 at gmail dot com>. All rights reserved.
 * @Author: shuyuanmao
 * @EMail: shuyuanmao123@gmail.com
 * @Date: 2019-05-19 02:07
 * @Description:
 */
#include <viola/viola.h>

#include <fstream>

cv::Mat mapCvt(const cv::Mat& map, int mid[2]) {
  cv::Mat res;
  if (mid[0] > 0)
    cv::vconcat(map.rowRange(mid[0], map.rows), map.rowRange(0, mid[0]), res);
  else
    res = map;
  cv::Mat res2;
  if (mid[1] > 0)
    cv::hconcat(res.colRange(mid[1], res.cols), res.colRange(0, mid[1]), res2);
  else
    res2 = res;
  return res2;
}

void testGridMapping2D() {
  std::ifstream fin("../data/intelab.map");
  int E = 8;  // map size for each dim = 1<<E
  int map_size = 1 << E;
  int mask = map_size - 1;
  vs::GridMapping2D gmapping(E, 0.05f, 255, 128, 20, 100);
  float pos[2];
  float pts[50000][2] = {0};
  int N;
  while (!fin.eof()) {
    fin >> pos[0] >> pos[1] >> N;
    for (int i = 0; i < N; i++) fin >> pts[i][0] >> pts[i][1];
    gmapping.update(pos, pts, N);
    auto map = gmapping.map();
    auto origin_idx = gmapping.originIdx();

    cv::Mat raw_map(1 << E, 1 << E, CV_8UC1, map);
    int mid[2] = {(origin_idx[0] + (1 << (E - 1))) & mask, (origin_idx[1] + (1 << (E - 1))) & mask};
    cv::Mat center_map = mapCvt(raw_map, mid);

    cv::cvtColor(raw_map, raw_map, cv::COLOR_GRAY2BGR);
    cv::cvtColor(center_map, center_map, cv::COLOR_GRAY2BGR);

    cv::circle(raw_map, cv::Point2i(origin_idx[1] & mask, origin_idx[0] & mask), 2, cv::Scalar(0, 0, 255), 2);
    cv::line(raw_map, cv::Point2i(mid[1], 0), cv::Point2i(mid[1], raw_map.rows), cv::Scalar(128, 128, 0), 1);
    cv::line(raw_map, cv::Point2i(0, mid[0]), cv::Point2i(raw_map.cols, mid[0]), cv::Scalar(128, 128, 0), 1);
    cv::circle(center_map, cv::Point2i((1 << (E - 1)), (1 << (E - 1))), 2, cv::Scalar(0, 0, 255), 2);
    cv::imshow("raw_map", raw_map);
    cv::imshow("center_map", center_map);
    static vs::VideoRecorder recorder("gridmapping.mp4");
    recorder.write(center_map);
    static vs::WaitKeyHandler handler(true);
    handler.waitKey();
  }
}

int depthToCloud(const cv::Mat& depth, const Eigen::Isometry3d pose, float pts[][3], int& N, int gap = 1) {
  double fx = 525;
  double fy = 525;
  double cx = 319.5;
  double cy = 239.5;
  double factor = 5000;
  N = 0;
  for (int i = 0; i < depth.rows; i += gap) {
    const uint16_t* ptr = depth.ptr<uint16_t>(i);
    for (int j = 0; j < depth.cols; j += gap) {
      double z = ptr[j] / factor;
      if (z <= 0.01 || z > 5) continue;
      double x = (j - cx) * z / fx;
      double y = (i - cy) * z / fy;
      auto pw = pose * Eigen::Vector3d(x, y, z);
      pts[N][0] = pw.x();
      pts[N][1] = pw.y();
      pts[N][2] = pw.z();
      N++;
    }
  }
  return N;
}

void testGridMapping3D() {
  const char* tum_rgbd_dir = "/home/symao/data/tum_rgbd/rgbd_dataset_freiburg1_xyz";
  auto f_pose = vs::join(tum_rgbd_dir, "groundtruth.txt");
  auto f_depth = vs::join(tum_rgbd_dir, "depth.txt");
  auto pose_data = vs::loadFileData(f_pose.c_str(), ' ');
  vs::TimeBuffer<Eigen::Isometry3d> pose_buf(vs::isomLerp);
  for (const auto& d : pose_data) {
    double ts = d[0];
    Eigen::Isometry3d pose;
    pose.translation() << d[1], d[2], d[3];
    pose.linear() = Eigen::Quaterniond(d[7], d[4], d[5], d[6]).toRotationMatrix();
    pose_buf.add(ts, pose);
  }

  vs::GridMapping3D gmapping(8, 0.1f, 128, 128, 1, 100);
  vs::Viz3D viz;
  std::vector<cv::Affine3f> cam_traj;

  std::ifstream fin(f_depth);
  std::string line;
  float pts[500000][3] = {0};
  int N = 0;
  while (getline(fin, line)) {
    if (line.length() < 10 || line[0] == '#') continue;
    // prepare data
    double ts;
    char name[128] = {0};
    sscanf(line.c_str(), "%lf %s", &ts, name);
    if(ts < 1305031104.763178) continue;
    cv::Mat depth = cv::imread(vs::join(tum_rgbd_dir, name), cv::IMREAD_UNCHANGED);
    Eigen::Isometry3d pose;
    bool ok = pose_buf.get(ts, pose);
    if (!ok) {
      printf("[ERROR]get pose at %f failed.\n", ts);
      continue;
    }
    depthToCloud(depth, pose, pts, N, 2);
    if (N < 10) {
      printf("[ERROR]no enough points(%d).\n", N);
      continue;
    }

    // call grid mapping
    float origin[3] = {VS_FLOAT(pose.translation().x()), VS_FLOAT(pose.translation().y()),
                       VS_FLOAT(pose.translation().z())};
    gmapping.update(origin, pts, N);

    // viz
    // show cam and traj
    cam_traj.push_back(vs::isom2affine(pose));
    viz.updateWidget("traj", cv::viz::WTrajectory(cam_traj, 2, 1, cv::viz::Color::yellow()));
    viz.updateWidget("frame", cv::viz::WTrajectoryFrustums(std::vector<cv::Affine3f>(1, cam_traj.back()),
                                                           cv::Vec2d(VS_RAD60, VS_RAD45), 0.2, cv::viz::Color::red()));
    // show grid
    int cnt = 0;
    float hr = gmapping.resolution() / 2;
    auto map = gmapping.map();
    for (int i = 0; i < (1 << (gmapping.E + gmapping.E + gmapping.E)); i++) {
      if (gmapping.occupyVal(map[i])) {
        int idx[3] = {(i >> (gmapping.E + gmapping.E)) & gmapping.M, (i >> gmapping.E) & gmapping.M, i & gmapping.M};
        float pos[3];
        gmapping.idx2pos(idx, pos);
        cv::Point3f p_min(pos[0] - hr, pos[1] - hr, pos[2] - hr);
        cv::Point3f p_max(pos[0] + hr, pos[1] + hr, pos[2] + hr);
        char name[128] = {0};
        snprintf(name, 128, "cube%d", cnt++);
        viz.updateWidget(name, cv::viz::WCube(p_min, p_max, true));
      }
    }
    // show cloud
    std::vector<cv::Point3f> cloud;
    for (int i = 0; i < N; i++) cloud.push_back(cv::Point3f(pts[i][0], pts[i][1], pts[i][2]));
    if (!cloud.empty()) viz.updateWidget("cloud", cv::viz::WCloud(cloud, cv::viz::Color::green()));

    cv::Mat depth_rgb;
    depth.convertTo(depth_rgb, CV_8UC1, 1.0 / 5000 * 255 / 3);
    cv::applyColorMap(depth_rgb, depth_rgb, cv::COLORMAP_JET);
    cv::imshow("depth", depth_rgb);
    static vs::WaitKeyHandler handler;
    handler.waitKey();
  }
}

int main(int argc, char** argv) {
  if (argc > 1 && atoi(argv[1]) == 3)
    testGridMapping3D();
  else
    testGridMapping2D();
}