/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details draw keypoints, feature tracking, mask, camera pose in 2D image with OpenCV. image hstack/vstack/gridstack.
 */
#pragma once
#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "vs_basic.h"
#include "vs_data.h"

namespace vs {

enum ImagePosition {
  POS_TL = 0,  // top-left
  POS_TR = 1,  // top-right
  POS_BR = 2,  // bottom-right
  POS_BL = 3   // bottom-left
};

/** @brief draw float image, such as depth.
 * Convert to CV_8UC1 by input k, and convert to CV_8UC3 using cv::applyColorMap
 * @param[in]img: input float image, CV_32FC1 or CV_64FC1
 * @param[in]k: scale factor when convert data range to [0,255]
 * @return bgr image in CV_8UC3
 */
cv::Mat drawMatf(const cv::Mat& img, float k = 255.0f);

/** @brief draw mask region at image
 * @param[in]img: grayscale or bgr image
 * @param[in]mask: mask image
 * @param[in]color: draw color for mask region
 * @param[in]ratio: [0,1], 0:not draw mask, 1:use mask color, 0~1:weighted sum color
 * @param[in]deep_copy: if true, draw at a new image, if false, draw at input img
 * @return bgr image in CV_8UC3
 */
cv::Mat drawMask(const cv::Mat& img, const cv::Mat& mask, const cv::Scalar& color, float ratio = 1.0f,
                 bool deep_copy = false);

/**
 * @brief draw subimage at image
 * @param[in]img: grayscale or bgr image
 * @param[in]sub_img: sub image
 * @param[in]draw_position: @see ImagePosition
 * @param[in]resize_rate: resize rate of sub image
 * @param[in]deep_copy: if true, draw at a new image, if false, draw at input img
 * @return bgr image in CV_8UC3
 */
cv::Mat drawSubImage(const cv::Mat& img, const cv::Mat& sub_img, int draw_position = POS_BR, float resize_rate = 1,
                     bool deep_copy = false);

/** @brief draw mask region at image
 * @param[in]img: grayscale or bgr image
 * @param[in]pts: draw points
 * @param[in]radius: draw radius for each point, see cv::circle()
 * @param[in]color: draw color for each point, see cv::circle()
 * @param[in]thickness: draw thickness for each point, see cv::circle()
 * @param[in]deep_copy: if true, draw at a new image, if false, draw at input img
 * @return bgr image in CV_8UC3
 */
cv::Mat drawPoints(const cv::Mat& img, const std::vector<cv::Point2f>& pts, int radius = 1,
                   const cv::Scalar& color = cv::Scalar(255, 0, 0), int thickness = 1, bool deep_copy = false);

/** @brief draw mask region at image
 * @param[in]img: grayscale or bgr image
 * @param[in]lines: draw lines, (x1,y1,x2,y2)
 * @param[in]thickness: draw thickness for each line, see cv::line()
 * @param[in]fix_color: whether draw same color for all lines, if false, use random color
 * @param[in]color: draw line color for all lines, only used when fix_color is true
 * @param[in]deep_copy: if true, draw at a new image, if false, draw at input img
 * @return bgr image in CV_8UC3
 */
cv::Mat drawLines(const cv::Mat& img, const std::vector<cv::Vec4f>& lines, int thickness = 1, bool fix_color = false,
                  const cv::Scalar& color = cv::Scalar(255, 0, 0), bool deep_copy = false);

/** @brief draw LK optical flow track result in current image
 * @param[in]img: current grayscale or bgr image
 * @param[in]prev_pts: track points in previous image
 * @param[in]cur_pts: track points in current image
 * @param[in]inliers: inlier status. if set, only draw point when status is not zero. if not set, draw all track points
 * @param[in]deep_copy: if true, draw at a new image, if false, draw at input img
 * @return bgr image in CV_8UC3
 */
cv::Mat drawLKMono(const cv::Mat& img, const std::vector<cv::Point2f>& prev_pts,
                   const std::vector<cv::Point2f>& cur_pts, const std::vector<unsigned char>& inliers = {},
                   bool deep_copy = false);

/** @brief draw LK optical flow track result in the horizontal conbination of previous image and current image
 * @param[in]prev_img: previous grayscale or bgr image
 * @param[in]cur_img: current grayscale or bgr image
 * @param[in]prev_pts: track points in previous image
 * @param[in]cur_pts: track points in current image
 * @param[in]inliers: inlier status. if set, only draw points whose status is not zero. if not set, draw all track
 * points
 * @return bgr image in CV_8UC3
 * @see ::drawMatches
 */
cv::Mat drawLKStereo(const cv::Mat& prev_img, const cv::Mat& cur_img, const std::vector<cv::Point2f>& prev_pts,
                     const std::vector<cv::Point2f>& cur_pts, const std::vector<uchar>& inliers = {});

/** @brief draw match pair in the horizontal conbination of two image
 * @param[in]img1: first grayscale or bgr image
 * @param[in]img2: second grayscale or bgr image
 * @param[in]pts1: points in first image
 * @param[in]pts2: points in second image
 * @param[in]inliers: inlier status. if set, only draw matches whose status is not zero. if not set, draw all matches
 * @return bgr image in CV_8UC3
 * @see ::drawLKStereo
 */
cv::Mat drawMatches(const cv::Mat& img1, const cv::Mat& img2, const std::vector<cv::Point2f>& pts1,
                    const std::vector<cv::Point2f>& pts2, const std::vector<uchar>& inliers = {}, int pt_radius = 4,
                    int pt_thickness = 2, int line_thickness = 2);

/** @brief draw horizontal line in the horizontal conbination of two image, used after stereo rectify
 * @param[in]img1: first grayscale or bgr image
 * @param[in]img2: second grayscale or bgr image
 * @param[in]step: row step between two horizontal line
 * @return bgr image in CV_8UC3
 */
cv::Mat drawRectify(const cv::Mat& imgl, const cv::Mat& imgr, int step = 40);

/** @brief draw 3D coordinate in image with input camera intrinsic and extrinsic
 * call cv::projectPoints(draw_pts, rvec, tvec, K, D, draw_corners), x-y-z axis in b-g-r color
 * @param[in]img: grayscale or bgr image
 * @param[in]rvec: rotation vector which transfrom point from world to camera coordinate system
 * @param[in]tvec: translation vector which transfrom point from world to camera coordinate system
 * @param[in]K: camera intrinsic matrix
 * @param[in]D: camera distortion
 * @param[in]len: coordinate axis length
 * @param[in]deep_copy: if true, draw at a new image, if false, draw at input img
 * @return bgr image in CV_8UC3
 */
cv::Mat drawCoordinate(const cv::Mat& img, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K, const cv::Mat& D,
                       float len = 1, bool deep_copy = false);

/** @brief draw 3D ground arrows in image with input camera intrinsic and extrinsic
 * assert ground plane as z=0, call cv::projectPoints(draw_pts, rvec, tvec, K, D, draw_corners)
 * @param[in]img: grayscale or bgr image
 * @param[in]rvec: rotation vector which transfrom point from world to camera coordinate system
 * @param[in]tvec: translation vector which transfrom point from world to camera coordinate system
 * @param[in]K: camera intrinsic matrix
 * @param[in]D: camera distortion
 * @param[in]max_draw_range: only draw arrow at [-max_draw_range,max_draw_range]
 * @param[in]draw_gap: distance between two arrows
 * @param[in]mask: ground mask
 * @return bgr image in CV_8UC3
 */
cv::Mat drawGroundArrows(const cv::Mat& img, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K,
                         const cv::Mat& D, float max_draw_range = 10, float draw_gap = 0.15, float arrow_size = 0.1,
                         const cv::Mat& mask = cv::Mat(), bool deep_copy = false);

/** @brief draw 3D cube in image with input camera intrinsic and extrinsic
 * call cv::projectPoints(draw_pts, rvec, tvec, K, D, draw_corners)
 * @param[in]img: grayscale or bgr image
 * @param[in]rvec: rotation vector which transfrom point from world to camera coordinate system
 * @param[in]tvec: translation vector which transfrom point from world to camera coordinate system
 * @param[in]K: camera intrinsic matrix
 * @param[in]D: camera distortion
 * @param[in]center: cube center
 * @param[in]cube_size: cube edge length
 * @param[in]color: cube edge color
 * @param[in]thickness: cube edge thickness
 * @param[in]deep_copy: if true, draw at a new image, if false, draw at input img
 * @return bgr image in CV_8UC3
 */
cv::Mat drawCube(const cv::Mat& img, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K, const cv::Mat& D,
                 const cv::Point3f& center = {0, 0, 0}, float cube_size = 1, const cv::Scalar& color = cv::Scalar(0),
                 int thickness = 1, bool deep_copy = false);

/** @brief draw 3D cubes in image with input camera intrinsic and extrinsic
 * call cv::projectPoints(draw_pts, rvec, tvec, K, D, draw_corners)
 * @param[in]img: grayscale or bgr image
 * @param[in]rvec: rotation vector which transfrom point from world to camera coordinate system
 * @param[in]tvec: translation vector which transfrom point from world to camera coordinate system
 * @param[in]K: camera intrinsic matrix
 * @param[in]D: camera distortion
 * @param[in]centers: cube centers
 * @param[in]cube_size: cube edge length
 * @param[in]color: cube edge color
 * @param[in]thickness: cube edge thickness
 * @param[in]deep_copy: if true, draw at a new image, if false, draw at input img
 * @return bgr image in CV_8UC3
 */
cv::Mat drawCubes(const cv::Mat& img, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K, const cv::Mat& D,
                  const std::vector<cv::Point3f>& centers, float cube_size = 1,
                  const std::vector<cv::Scalar>& colors = {}, int thickness = 1, bool deep_copy = false);

cv::Mat drawReprojectionError(const cv::Mat& img, const std::vector<cv::Point3f>& pts3d,
                              const std::vector<cv::Point2f>& pts2d, const cv::Mat& rvec, const cv::Mat& tvec,
                              const cv::Mat& K, const cv::Mat& D, const cv::Scalar& color, int thickness = 1,
                              bool deep_copy = false);

/** @brief draw sparse color depth data in input image
 * @param[in]img: grayscale or bgr image
 * @param[in]depth: sparse depth image
 * @param[in]max_depth: maximum depth, which set to 255 when convert float depth to CV_8UC1
 * @return bgr image in CV_8UC3
 */
cv::Mat drawSparseDepth(const cv::Mat& img, const cv::Mat& depth, float max_depth = 10.0f);

/** @brief horizontal stack list of images
 * @param[in]img: image list, image size can be different.
 * @param[in]border_color: color of border at each image
 * @param[in]bg_color: color of background, when image size not same, the background will be fill with this color
 * @return bgr image in CV_8UC3
 */
cv::Mat hstack(const std::vector<cv::Mat>& imgs, const cv::Scalar& border_color = cv::Scalar(0),
               const cv::Scalar& bg_color = cv::Scalar(0));

/** @brief vertical stack list of images
 * @param[in]img: image list, image size can be different.
 * @param[in]border_color: color of border at each image
 * @param[in]bg_color: color of background, when image size not same, the background will be fill with this color
 * @return bgr image in CV_8UC3
 */
cv::Mat vstack(const std::vector<cv::Mat>& imgs, const cv::Scalar& border_color = cv::Scalar(0),
               const cv::Scalar& bg_color = cv::Scalar(0));

/** @brief stack list of images in grid order, row priority
 * @param[in]img: image list, image size must be same.
 * @param[in]grid_rows: grid row of images
 * @param[in]grid_cols: image count in each grid row, if not set, auto calculated with all image count and grid rows
 * @param[in]force_resize: if not zero, resize to this size before draw into grid
 * @param[in]bg_color: background color, when image count < grid_rows*grid_cols, background will be fill with bg_color
 * @return bgr image in CV_8UC3
 */
cv::Mat gridStack(const std::vector<cv::Mat>& imgs, int grid_rows, int grid_cols = -1,
                  const cv::Size& force_resize = cv::Size(0, 0), const cv::Scalar& bg_color = cv::Scalar(0));

/** @brief convert image to RGB
 * @param[in]img: grayscale or bgr image
 * @param[in]depp_copy: if true, draw at a new image, if false, draw at input img
 * @return RGB image in CV_8UC3
 */
cv::Mat toRgb(const cv::Mat& img, bool depp_copy = false);

/** @brief convert image to grayscale
 * @param[in]img: grayscale or bgr image
 * @param[in]depp_copy: if true, draw at a new image, if false, draw at input img
 * @return grayscale image in CV_8UC1
 */
cv::Mat toGray(const cv::Mat& img, bool depp_copy = false);

/** @brief convert image to RGB, deep copy
 * @param[in]src: grayscale or bgr image
 * @param[out]dst: RGB image in CV_8UC3
 */
void toRgb(const cv::Mat& src, cv::OutputArray dst);

/** @brief convert image to grayscale
 * @param[in]src: grayscale or bgr image
 * @param[out]dst: grayscale image in CV_8UC1
 */
void toGray(const cv::Mat& src, cv::OutputArray dst);

/** @brief get R-G-B color with input rate
 * @note input rate from [0,1], out a color from [red-orange-yellow-green-cyan-blue-purple]
 */
void colorBar(double rate, uchar& R, uchar& G, uchar& B);

/** @brief get R-G-B color with input rate
 * @note input rate from [0,1], out a color from [red-orange-yellow-green-cyan-blue-purple]
 */
cv::Scalar colorBar(double rate);

/** @brief generate random colors */
std::vector<cv::Scalar> randColors(int cnt);

class TimelinePloter {
 public:
  TimelinePloter(double max_store_sec = 5);

  void arrived(int sensor_id, double ts);

  void used(int sensor_id, double ts);

  cv::Mat plot(const cv::Size& img_size = cv::Size(640, 480));

 private:
  struct Timeline {
    vs::TimedQueue<double> arrived_ts_list;
    vs::TimedQueue<double> used_ts_list;

    Timeline() : arrived_ts_list([](const double& a) { return a; }), used_ts_list([](const double& a) { return a; }) {}

    void arrived(double ts) { arrived_ts_list.insert(ts); }

    void used(double ts) { used_ts_list.insert(ts); }

    void removeBefore(double start_ts) {
      arrived_ts_list.removeBefore(start_ts);
      used_ts_list.removeBefore(start_ts);
    }
  };
  std::map<int, Timeline> timelime_map_;
  double max_store_sec_;
  double latest_ts_;
};

/** @brief Handling pause, continue, exit in waitkey
 * @code {.C++}
 * // your show code such as cv::imshow("img", img);
 * // add the two-lines below
 * static WaitKeyHandler handler;
 * handler.waitKey();
 * @endcode
 */
class WaitKeyHandler {
 public:
  explicit WaitKeyHandler(bool init_play = false) : play_(init_play) {}

  void waitKey() {
    auto key = cv::waitKey(play_ ? 10 : 0);
    if (key == VS_ASCII_ESC) {
      std::exit(1);
    } else if (key == 's' || key == 'S' || key == VS_ASCII_SPACE) {
      play_ = !play_;
    }
  }

 private:
  bool play_;
};

} /* namespace vs */