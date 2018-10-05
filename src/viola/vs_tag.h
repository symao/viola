/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2023-02-19 18:46
 * @details detect 2D tags in image, support chessboard, apriltag, ARUCO.
 */
#pragma once
#include <vector>
#include <memory>
#include <opencv2/core.hpp>

namespace vs {

enum TagType {
  TAG_APRILTAG = 1,
  TAG_ARUCO = 2,
  TAG_CHESSBOARD = 3,
};

/** @brief one tag information */
struct Tag2D {
  int id;                          ///< tag id
  std::vector<cv::Point2f> pts2d;  ///< tag 2D uv in image

  Tag2D(int _id = 0, const std::vector<cv::Point2f>& _pts2d = {}) : id(_id), pts2d(_pts2d) {}
};

struct Tag3D {
  int id = 0;                      ///< tag id
  std::vector<cv::Point3f> pts3d;  ///< tag 3D xyz, calculated by tag config

  Tag3D(int _id = 0, const std::vector<cv::Point3f>& _pts3d = {}) : id(_id), pts3d(_pts3d) {}
};

typedef std::vector<Tag2D> Tag2DList;
typedef std::vector<Tag3D> Tag3DList;

struct TagConfig {
  int tag_type = TAG_APRILTAG;
  int tag_rows;
  int tag_cols;
  double tag_size = 0.01;   ///< tag size, unit meter
  double tag_spacing = 0;   ///< tag spacing, only used in kalibr_tag. For details, please see kalibr tag description.
  double black_border = 2;  ///< if you use kalibr_tag black_boarder = 2; if you use apriltag black_boarder = 1
};

class TagDetector {
 public:
  virtual ~TagDetector() = default;

  virtual Tag2DList detect(const cv::Mat& img) = 0;
};

std::shared_ptr<TagDetector> createTagDetector(const TagConfig& tag_cfg);

Tag3DList generate3DTags(const TagConfig& tag_cfg);

void drawTags(cv::Mat& img, const Tag2DList& tag_list);

}  // namespace vs