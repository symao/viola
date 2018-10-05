/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details detect and track 2D point feature between consecutive frames.
 */
#pragma once
#include <opencv2/video/tracking.hpp>

namespace vs {

/** @brief Feature tracker, detect feature in image and track with LK optical flow */
class FeatureTracker {
 public:
  struct Config {
    int feature_type = 0;                       ///< 0: good feature to track 1: FAST
    int max_corner = 150;                       ///< max feature count in a frame
    int min_corner_dist = 15;                   ///< min distance between two features
    int lk_pyr_level = 3;                       ///< pyramid level for LK optical flow
    int thres_fast = 20;                        ///< threshold for FAST
    float thres_gfft = 0.001;                   ///< quality level for good-feature-to-track
    cv::Size lk_patch_size = cv::Size(21, 21);  ///< patch size for LK optical flow

    bool hist_equal = false;                    ///< whether do histogram equalization with CLAHE
    double clahe_clip = 3.0;                    ///< clip limit for CLAHE
    cv::Size clahe_grid_size = cv::Size(8, 8);  ///< tile grid size for CLAHE

    bool cross_check = false;        ///< whether track feature backward to cross check
    float cross_check_thres = 1.0f;  ///< max inlier distance between raw point and back-track point

    float max_move_thres = 0;  ///< max inlier move from previous image to current image, 0 means not do max move check
    int border_thres = 3;      ///< image border for LK optical flow to check outside feature
  };

  typedef uint64_t FeatureIDType;  ///< feature id type

  /** @brief type for tracked feature */
  struct Feature : public cv::Point2f {
    explicit Feature(const cv::Point2f& _p = cv::Point2f(), FeatureIDType _id = -1, int _track_cnt = 0)
        : cv::Point2f(_p), id(_id), track_cnt(_track_cnt) {}

    FeatureIDType id;  ///< feature id
    int track_cnt;     ///< tracked frame count
  };

  typedef std::vector<Feature> FeatureList;  ///< feature container

  struct TrackResult {
    int feature_cnt = 0;     ///< whole tracked/detected features count
    int detectd_cnt = 0;     ///< detected features count
    int tracked_cnt = 0;     ///< tracked features count
    float avg_parallex = 0;  ///< average parallex between current frame and previous track frame
    FeatureList features;    ///< tracked/detected features in current frame
  };

  /** @brief constructor */
  FeatureTracker();

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

  /** @brief detect new feature points
   * @param[in]img: input image
   * @param[in]detect_cnt: max detect feature count
   * @param[in]mask: input mask to define where to detect feature
   */
  std::vector<cv::Point2f> detectFeature(const cv::Mat& img, int detect_cnt, const cv::Mat& mask = cv::Mat());

  /**
   * @brief draw tracked features, draw position and id, color correponding to track count
   * @param[in,out]bgr: input bgr image to draw features
   * @param[in]features: tracked features
   */
  void drawFeature(cv::Mat& bgr, const FeatureList& features);

 private:
  int m_process_idx;                    ///< process frame index
  FeatureIDType m_feature_id;           ///< next feature id, start from 1
  Config m_cfg;                         ///< configuration parameters
  cv::Ptr<cv::CLAHE> m_clahe;           ///< CLAHE handler
  std::vector<cv::Mat> m_prev_pyr;      ///< previous pyramid for optical flow
  std::vector<cv::Point2f> m_prev_pts;  ///< previous feature points
  std::vector<int> m_track_ids;         ///< tracked feature ids
  std::vector<int> m_track_cnt;         ///< tracked feature track count
};

/** @brief Non maximum suppression with image grid, each grid stores up to one feature */
class GridNMS {
 public:
  /** @brief constructor
   * @param[in]:img_size: image size
   * @param[in]:w: grid width
   */
  explicit GridNMS(const cv::Size& img_size, int w = 20) : m_w(w), m_img_size(img_size) {
    m_grid_rows = std::ceil(img_size.height / m_w);
    m_grid_cols = std::ceil(img_size.width / m_w);
    m_grid_size = m_grid_rows * m_grid_cols;
    m_cells.resize(m_grid_size, 0);
  }

  /** @brief reset grid buffer */
  void reset() {
    for (auto it = m_cells.begin(); it <= m_cells.end(); ++it) *it = true;
  }

  /** @brief try to add a feature
   * @param[in]pt: add point in image
   * @return if grid is empty, add sucess and return true, else return false
   */
  bool tryAdd(const cv::Point2f& pt) {
    if (pt.x < 0 || pt.x >= m_img_size.width || pt.y < 0 || pt.y >= m_img_size.height) return false;
    int idx = static_cast<int>(pt.y / m_w) * m_grid_cols + static_cast<int>(pt.x / m_w);
    if (m_cells[idx]) return false;
    m_cells[idx] = true;
    return true;
  }

  /** @brief check whether a feature can be add
   * @param[in]pt: check point in image
   * @return if grid is empty, return true, else return false
   */
  bool query(const cv::Point2f& pt) {
    if (pt.x < 0 || pt.x >= m_img_size.width || pt.y < 0 || pt.y >= m_img_size.height) return false;
    int idx = static_cast<int>(pt.y / m_w) * m_grid_cols + static_cast<int>(pt.x / m_w);
    if (m_cells[idx]) return false;
    return true;
  }

 private:
  int m_w;                                    ///< grid width
  cv::Size m_img_size;                        ///< image size
  int m_grid_rows, m_grid_cols, m_grid_size;  ///< grid rows, grid cols, grid length
  std::vector<bool> m_cells;                  ///< grid buffer cells
};

/** @brief Non maximum suppression with image mask */
class MaskNMS {
 public:
  /** @brief Construct
   * @param[in]img_size: image size
   * @param[in]w: min distance between two features
   * @param[in]resize: resize rate, resize to small mask for acceleration
   */
  explicit MaskNMS(const cv::Size& img_size, int w = 20, float resize = 1)
      : m_w(w), m_img_size(img_size), m_resize(resize) {
    cv::Size mask_size = m_img_size;
    if (m_resize != 1) {
      mask_size.width *= m_resize;
      mask_size.height *= m_resize;
      m_w *= m_resize;
    }
    m_mask = cv::Mat(mask_size, CV_8UC1, cv::Scalar(255));
  }

  /** @brief Set prior mask */
  void setMask(const cv::Mat& mask) { cv::resize(mask, m_mask, m_mask.size()); }

  /** @brief reset mask */
  void reset() {
    if (!m_mask.empty()) {
      uchar* ptr = m_mask.data;
      for (int i = 0; i < m_mask.rows; i++)
        for (int j = 0; j < m_mask.cols; j++) *ptr++ = 255;
    }
  }

  /** @brief try to add a feature, if nms, not add
   * @param[in]pt: add point in image
   * @return if mask value at pt non-zero, return true and draw a circle center at pt and fill with zeros
   */
  bool tryAdd(const cv::Point2f& pt) {
    if (pt.x < 0 || pt.x >= m_img_size.width || pt.y < 0 || pt.y >= m_img_size.height) return false;
    cv::Point2f scale_pt = pt * m_resize;
    if (m_mask.at<uchar>(scale_pt) == 0) return false;
    cv::circle(m_mask, scale_pt, m_w, cv::Scalar(0), -1);
    return true;
  }

  /** @brief add a feature forcely, regardless mask existance
   * @param[in]pt: add point in image
   * @return if mask value at pt non-zero, return true and draw a circle center at pt and fill with zeros
   */
  void add(const cv::Point2f& pt) {
    if (pt.x < 0 || pt.x >= m_img_size.width || pt.y < 0 || pt.y >= m_img_size.height) return;
    cv::Point2f scale_pt = pt * m_resize;
    cv::circle(m_mask, scale_pt, m_w, cv::Scalar(0), -1);
  }

  /** @brief check whether a feature can be add
   * @param[in]pt: check point in image
   * @return if mask value at pt non-zero, return true else return false
   */
  bool query(const cv::Point2f& pt) {
    if (pt.x < 0 || pt.x >= m_img_size.width || pt.y < 0 || pt.y >= m_img_size.height) return false;
    cv::Point2f scale_pt = pt * m_resize;
    if (m_mask.at<uchar>(scale_pt) == 0) return false;
    return true;
  }

  /** @brief get current mask */
  cv::Mat mask() { return m_mask; }

 private:
  int m_w;              ///< min distance between two features
  cv::Size m_img_size;  ///< image size
  cv::Mat m_mask;       ///< mask, 0-can not be added >0-can be added
  float m_resize;       ///< resize rate, default 1 means not resize
};

bool calcOpticalFlowLK(const std::vector<cv::Mat>& prev_pyr, const std::vector<cv::Mat>& cur_pyr,
                       const std::vector<cv::Point2f>& prev_pts, std::vector<cv::Point2f>& cur_pts,
                       std::vector<uchar>& status, const cv::Size& patch_size, int max_level, bool cross_check = false,
                       double cross_check_thres = 1.0, int border_thres = 5, float max_move = 0);

/** @brief calculate Shi-Tomasi score at point(u,v) in image
 * @param[in]img: input image in grayscale
 * @param[in]u: position u in pixels
 * @param[in]v: position v in pixels
 * @return Shi-Tomasi score at point(u,v) in image
 */
float shiTomasiScore(const cv::Mat& img, int u, int v);

/** @brief remove outlier after LK optical flow track
 * @param[in]prev_pts: previous track points
 * @param[in]cur_pts: current track points
 * @param[in/out]status: track status, new outlier's status will be set to 0
 * @param[in]img_size: track image size, use to check border
 * @param[in]border: border check threshold
 * @param[in]max_move: max move distance in pixel
 * @param[in]max_move_rate: max rate = max inlier move dist / middle inlier move dist
 */
void lkOutlierReject(const std::vector<cv::Point2f>& prev_pts, const std::vector<cv::Point2f>& cur_pts,
                     std::vector<uchar>& status, const cv::Size& img_size, int border = 0, float max_move = 1e10,
                     float max_move_rate = 0);

/** @brief detect feature in image
 * @param[in]img: input image in grayscale
 * @param[in]detect_cnt: max detect feature count in whole image
 * @param[in]mask: detect mask
 * @param[in]feature_type: 0:goodFeatureToTrack, 1:FAST
 * @param[in]param1: if feature_type=0, this is quality_level, if feature_type=1, this is FAST thresh
 * @param[in]param2: if feature_type=0, this is min_dist, if feature_type=1, this is FAST thresh low
 * @return detect new features
 * */
std::vector<cv::Point2f> featureDetect(const cv::Mat& img, int detect_cnt, int feature_type = 0, float param1 = 0.001,
                                       float param2 = 10, const cv::Mat& mask = cv::Mat(),
                                       const cv::Rect& roi = cv::Rect());

/** @brief detect FAST feature in image
 * @param[in]img: input image in grayscale
 * @param[in]detect_cnt: max detect feature count in whole image
 * @param[in]mask: detect mask
 * @param[in]fast_thres: FAST thresh
 * @param[in]fast_thres_min: min FAST thresh, if detect less corners than detect_cnt, run FAST again with fast_thres_min
 * @return detect new features
 * */
std::vector<cv::Point2f> featureDetectFast(const cv::Mat& img, int detect_cnt, const cv::Mat& mask = cv::Mat(),
                                           float fast_thres = 20, float fast_thres_min = 7);

/** @brief detect feature in uniform grid sampling
 * @param[in]img: input image in grayscale
 * @param[in]grid_size: split cols x rows grids
 * @param[in]max_feature_cnt: max detect feature count in whole image
 * @param[in]mask: detect mask
 * @param[in]feature_type: 0:goodFeatureToTrack, 1:FAST
 * @param[in]exist_features: if exist features in one grid, then detect count in this grid will subtract the exist
 * count
 * @return detect new features, which not include exist_features
 * */
std::vector<cv::Point2f> featureDetectUniform(const cv::Mat& img, const cv::Size& grid_size, int max_feature_cnt,
                                              const cv::Mat& mask = cv::Mat(), int feature_type = 0,
                                              const std::vector<cv::Point2f>& exist_features = {});

} /* namespace vs */
