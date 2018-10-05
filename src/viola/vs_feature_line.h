/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-10-28 15:07
 * @details detect and track 2D line feature between consecutive frames
 */
#pragma once
#include <memory>
#include <vector>
#include <opencv2/video/tracking.hpp>

namespace vs {

/** @brief detect lines in gray image with rectangle ROI*/
int detectLine(const cv::Mat& gray, std::vector<cv::Vec4f>& lines, const cv::Rect& roi = cv::Rect());

/** @brief detect lines in gray image with mask*/
int detectLine(const cv::Mat& gray, std::vector<cv::Vec4f>& lines, const cv::Mat& mask);

/**
 * @brief line tracking with LK optical flow
 * sampling discrete points in line for LK tracking, and fit a new line with tracked points
 * @param[in]prev_img: previous image
 * @param[in]cur_img: current image
 * @param[in]prev_lines: previous lines to be tracked
 * @param[out]cur_lines: tracked new lines in current image
 * @param[in]status: track status, 0-lost, 1-tracked
 * @param[in]delta: sampling delta pixels
 * @param[in]cross_check: if true, track current lines back to previous image for cross check
 * @param[in]patch_size: patch for LK optical flow
 * @param[in]pyr_level: pyramid level for LK optical flow
 * @param[in]gradient_filter: whether filter points with small gradient
 * @param[in]draw: whether debug draw, if true, imshow will be called inside
 */
void lineOpticalFlowLK(const cv::Mat& prev_img, const cv::Mat& cur_img, const std::vector<cv::Vec4f>& prev_lines,
                       std::vector<cv::Vec4f>& cur_lines, std::vector<uint8_t>& status, float delta = 5,
                       bool cross_check = false, const cv::Size& patch_size = cv::Size(13, 13), int pyr_level = 3,
                       bool gradient_filter = false, bool draw = false);

/**
 * @brief line tracking with LK optical flow
 * sampling discrete points in line for LK tracking, and fit a new line with tracked points
 * @param[in]prev_img: previous image pyramid
 * @param[in]cur_img: current image pyramid
 * @param[in]prev_lines: previous lines to be tracked
 * @param[out]cur_lines: tracked new lines in current image
 * @param[in]status: track status, 0-lost, 1-tracked
 * @param[in]delta: sampling delta pixels
 * @param[in]cross_check: if true, track current lines back to previous image for cross check
 * @param[in]patch_size: patch for LK optical flow
 * @param[in]pyr_level: pyramid level for LK optical flow
 * @param[in]gradient_filter: whether filter points with small gradient
 * @param[in]draw: whether debug draw, if true, imshow will be called inside
 */
void lineOpticalFlowLK(const std::vector<cv::Mat>& prev_pyr, const std::vector<cv::Mat>& cur_pyr,
                       const std::vector<cv::Vec4f>& prev_lines, std::vector<cv::Vec4f>& cur_lines,
                       std::vector<uint8_t>& status, float delta = 5, bool cross_check = false,
                       const cv::Size& patch_size = cv::Size(13, 13), int pyr_level = 3, bool gradient_filter = false,
                       bool draw = false);

struct Line2D {
  cv::Point2f p1, p2;  ///< endpoint of Line

  Line2D() : p1(0, 0), p2(0, 0) {}

  Line2D(float x1, float y1, float x2, float y2) : p1(x1, y1), p2(x2, y2) {}

  Line2D(const cv::Point2f& _p1, const cv::Point2f& _p2) : p1(_p1), p2(_p2) {}

  Line2D(const cv::Vec4f& l) : p1(l[0], l[1]), p2(l[2], l[3]) {}

  cv::Point2f center() const { return (p1 + p2) / 2; }

  double dir() const { return atan2(p2.y - p1.y, p2.x - p1.x); }

  double length() const { return hypotf(p2.x - p1.x, p2.y - p1.y); }
};

typedef std::vector<Line2D> Line2DList;

enum LineDetectorType {
  LINE_DETECTOR_LSD = 0,  ///< cv::line_descriptors::LSDDetector
  LINE_DETECTOR_LBD = 1,  ///< cv::line_descriptors::BinaryDescriptor
};

class LineDetector {
 public:
  virtual ~LineDetector() = default;

  virtual void detect(const cv::Mat& img, Line2DList& lines, const cv::Mat& mask = cv::Mat()) = 0;
};

typedef std::shared_ptr<LineDetector> LineDetectorPtr;
LineDetectorPtr createLineDetector(int type);

/** @brief Feature tracker, detect feature in image and track with LK optical flow */
class LineFeatureTracker {
 public:
  struct Config {
    float min_line_len = 20.0f;                 ///< minimum line length
    int lk_pyr_level = 3;                       ///< pyramid level for LK optical flow
    cv::Size lk_patch_size = cv::Size(21, 21);  ///< patch size for LK optical flow
    bool hist_equal = false;                    ///< whether do histogram equalization with CLAHE
    double clahe_clip = 3.0;                    ///< clip limit for CLAHE
    cv::Size clahe_grid_size = cv::Size(8, 8);  ///< tile grid size for CLAHE
    bool cross_check = false;                   ///< whether track feature backward to cross check
    float cross_check_thres = 1.0f;             ///< max inlier distance between raw point and back-track point
    float max_move_thres = 0;  ///< max inlier move from previous image to current image, 0 means not do max move check
    int border_thres = 3;      ///< image border for LK optical flow to check outside feature
  };

  typedef uint64_t FeatureIDType;  ///< feature id type

  /** @brief type for tracked feature */
  struct LineFeature : Line2D {
    FeatureIDType id;  ///< feature id
    int track_cnt;     ///< tracked frame count

    LineFeature(const Line2D& l = Line2D(), FeatureIDType _id = 0, int _track_cnt = 0)
        : Line2D(l), id(_id), track_cnt(_track_cnt) {}

    void update(const cv::Vec4f& a) {
      p1.x = a[0];
      p1.y = a[1];
      p2.x = a[2];
      p2.y = a[3];
    }
  };

  typedef std::vector<LineFeature> LineFeatureList;  ///< feature container

  struct TrackResult {
    LineFeatureList line_features;  ///< tracked/detected features in current frame
  };

  /** @brief constructor */
  LineFeatureTracker();

  /** @brief set configuration */
  void setConfig(const Config& cfg) { m_cfg = cfg; }

  /** @brief set current configuration */
  Config getConfig() const { return m_cfg; }

  /** @brief process a frame, track old feature from previous frame, detect new features in current frame
   * @param[in]img: input image
   * @param[in]mask: define region where to detect/track feature
   * @return
   */
  TrackResult process(const cv::Mat& img, const cv::Mat& mask = cv::Mat());

  /**
   * @brief draw tracked line features, draw position and id, color correponding to track count
   * @param[in,out]bgr: input image to draw line features
   * @param[in]features: line features
   */
  void drawLineFeature(cv::Mat& bgr, const LineFeatureList& line_features, bool draw_id = false);

  std::vector<cv::Vec4f> cvt(const LineFeatureList& lines);

 private:
  int m_process_idx;                ///< process frame index
  FeatureIDType m_feature_id;       ///< next feature id, start from 1
  Config m_cfg;                     ///< configuration parameters
  cv::Ptr<cv::CLAHE> m_clahe;       ///< CLAHE handler
  std::vector<cv::Mat> m_prev_pyr;  ///< previous pyramid for optical flow
  LineFeatureList m_prev_lines;     ///< previous feature lines
  LineDetectorPtr m_line_detector;  ///< line detector
};

}  // namespace vs