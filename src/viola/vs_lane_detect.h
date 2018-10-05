/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details detect lane with specific color.
 */
#pragma once
#include <map>
#include <vector>
#include <opencv2/core.hpp>

namespace vs {

/** @brief struct of lane information */
struct Lane {
  cv::Point2f p1;      ///< first endpoint of lane segment
  cv::Point2f p2;      ///< second endpoint of lane segment
  cv::Point2f center;  ///< center point of lane segment
  cv::Point2f dir;     ///< direction vector of lane segment
  float len;           ///< length
  int lane_type;       ///< land color type, see @LandType

  Lane(const cv::Point2f& _p1, const cv::Point2f& _p2);
};
typedef std::vector<Lane> LaneList;

/** @brief lane type */
enum LaneType {
  LANE_PURE_YELLOW = 3,
};

/** @brief detect lane with traditional method
 * detect lines in image, and reproject lines to ground with camere intrinsic and extrinsic,
 * then find parallel line pair in lane condition
 * @param[in]img: input 3-channel BGR image
 * @param[out]lanes: detect lane result
 * @param[in]K: camera intrinsic
 * @param[in]T_c_b: transformation from camera to body
 * @param[in]lane_type: lane type
 * @param[in]min_lane_w: min lane width in ground
 * @param[in]max_lane_w: max lane width in ground
 * @param[in]draw: whether enable debug draw
 * @return int detect lane count
 */
int laneDetect(const cv::Mat& img, LaneList& lanes, const cv::Mat& K, const cv::Mat& T_c_b,
               int lane_type = LANE_PURE_YELLOW, float min_lane_w = 0.06, float max_lane_w = 0.25, bool draw = false);

} /* namespace vs */
