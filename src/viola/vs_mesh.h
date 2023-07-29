/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2023-02-07 15:07
 * @details 3D mesh data
 */
#pragma once
#include <vector>
#include <opencv2/core.hpp>
#if ENABLE_CV_VIZ
#include <opencv2/viz.hpp>
#endif  // ENABLE_CV_VIZ

namespace vs {
template <typename T>
std::vector<int> toVizPoly(const std::vector<std::vector<T>>& polys) {
  std::vector<int> res;
  for (const auto& poly : polys) {
    res.push_back(poly.size());
    for (const auto& i : poly) res.push_back(i);
  }
  return res;
}

struct MeshData {
  using Vertex = cv::Point3f;
  using VertexList = std::vector<Vertex>;
  using Color = cv::Vec3b;
  using ColorList = std::vector<Color>;
  using Normal = cv::Vec3f;
  using NormalList = std::vector<Normal>;
  using IndexType = size_t;
  using Polygon = std::vector<IndexType>;
  using PolygonList = std::vector<Polygon>;

  VertexList vertices;   ///< vertice coordinate
  PolygonList polygons;  ///< face polygons
  ColorList colors;      ///< vertice color in B-G-R order
  NormalList normals;    ///< vertice normal vector

  bool readObj(const char* file);

  bool writeObj(const char* file) const;

  bool readPly(const char* file);

  bool writePly(const char* file) const;

#if ENABLE_CV_VIZ
  cv::viz::WMesh toVizMesh() const { return cv::viz::WMesh(vertices, toVizPoly(polygons), colors, normals); }

  cv::viz::WCloud toVizCloud() const {
    return colors.empty() ? cv::viz::WCloud(vertices) : cv::viz::WCloud(vertices, colors);
  }
#endif  // ENABLE_CV_VIZ
};

}  // namespace vs