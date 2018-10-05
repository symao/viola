/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2023-02-07 15:07
 * @details 3D mesh data
 */
#pragma once
#include <opencv2/core.hpp>
#if ENABLE_CV_VIZ
#include <opencv2/viz.hpp>
#endif  // ENABLE_CV_VIZ

namespace vs {

struct MeshData {
  std::vector<cv::Point3f> vertices;       ///< vertice coordinate
  std::vector<std::vector<int>> polygons;  ///< face polygons
  std::vector<cv::Vec3b> colors;           ///< vertice color
  std::vector<cv::Vec3f> normals;          ///< vertice normal vector

  bool readObj(const char* file);

  bool writeObj(const char* file);

  bool readPly(const char* file);

  bool writePly(const char* file);

#if ENABLE_CV_VIZ
  cv::viz::WMesh toVizMesh() { return cv::viz::WMesh(vertices, toVizPoly(polygons), colors, normals); }

  cv::viz::WCloud toVizCloud() {
    return colors.empty() ? cv::viz::WCloud(vertices) : cv::viz::WCloud(vertices, colors);
  }
#endif  // ENABLE_CV_VIZ

  /** @brief convert polygons index to cv::viz::WSphere input params */
  static std::vector<int> toVizPoly(const std::vector<std::vector<int>>& polys);
};

}  // namespace vs