/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_feature.h"

#include "vs_basic.h"
#include "vs_geometry2d.h"
#include "vs_random.h"
#include "vs_tictoc.h"
// #include "vs_debug_draw.h"

namespace vs {

FeatureTracker::FeatureTracker() : m_process_idx(0), m_feature_id(1) {}

FeatureTracker::TrackResult FeatureTracker::process(const cv::Mat& img, const cv::Mat& mask) {
  // preprocess
  cv::Mat process_img;
  if (img.channels() == 3) {
    cv::cvtColor(img, process_img, cv::COLOR_BGR2GRAY);
  } else {
    process_img = img;
  }
  if (m_cfg.hist_equal) {
#if (CV_MAJOR_VERSION >= 3)
    if (!m_clahe.get())
#else
    if (!m_clahe.obj)
#endif
      m_clahe = cv::createCLAHE(m_cfg.clahe_clip, m_cfg.clahe_grid_size);
    m_clahe->apply(process_img, process_img);
  }

  // build pyramid
  std::vector<cv::Mat> cur_pyr;
  cv::buildOpticalFlowPyramid(process_img, cur_pyr, m_cfg.lk_patch_size, m_cfg.lk_pyr_level);

  // track
  std::vector<cv::Point2f> cur_pts;
  MaskNMS nms(process_img.size(), m_cfg.min_corner_dist);
  if (!mask.empty()) nms.setMask(mask);
  if (!m_prev_pts.empty()) {
    // calc optical flow
    std::vector<uchar> status;
    calcOpticalFlowLK(m_prev_pyr, cur_pyr, m_prev_pts, cur_pts, status, m_cfg.lk_patch_size, m_cfg.lk_pyr_level,
                      m_cfg.cross_check, m_cfg.cross_check_thres, m_cfg.border_thres, m_cfg.max_move_thres);
    // track feature nms
    for (size_t i = 0; i < status.size(); i++) {
      if (status[i] == 0) continue;
      if (!nms.tryAdd(cur_pts[i])) status[i] = 0;
    }
    // remove lost feature
    vecReduce(cur_pts, status);
    vecReduce(m_prev_pts, status);
    vecReduce(m_track_cnt, status);
    vecReduce(m_track_ids, status);
    // update track count
    for (auto& cnt : m_track_cnt) cnt++;
  }

  int tracked_feature_cnt = cur_pts.size();
  float tracked_parallex = 0;
  if (tracked_feature_cnt > 0) {
    double sum = 0;
    for (size_t i = 0; i < cur_pts.size(); i++) sum += cv::norm(cur_pts[i] - m_prev_pts[i]);
    tracked_parallex = sum / tracked_feature_cnt;
  }

  // add new corners
  bool need_add = (tracked_feature_cnt < m_cfg.max_corner * 0.9);
  if (need_add) {
    int add_cnt = m_cfg.max_corner - cur_pts.size();
    cv::Mat detect_mask = nms.mask();
    if (mask.size() == process_img.size()) cv::bitwise_and(detect_mask, mask, detect_mask);
    std::vector<cv::Point2f> new_pts = detectFeature(process_img, add_cnt, detect_mask);
    for (const auto& p : new_pts) {
      cur_pts.push_back(p);
      m_track_cnt.push_back(1);
      m_track_ids.push_back(m_feature_id++);
    }
  }

  // post process
  m_prev_pts = cur_pts;
  m_prev_pyr = cur_pyr;
  m_process_idx++;

  TrackResult res;
  res.feature_cnt = cur_pts.size();
  res.tracked_cnt = tracked_feature_cnt;
  res.detectd_cnt = res.feature_cnt - res.tracked_cnt;
  res.avg_parallex = tracked_parallex;
  res.features.reserve(cur_pts.size());
  for (size_t i = 0; i < cur_pts.size(); i++) res.features.emplace_back(cur_pts[i], m_track_ids[i], m_track_cnt[i]);
  return res;
}

std::vector<cv::Point2f> FeatureTracker::detectFeature(const cv::Mat& img, int detect_cnt, const cv::Mat& mask) {
  std::vector<cv::Point2f> corners;
  switch (m_cfg.feature_type) {
    case 0:
      corners = featureDetect(img, detect_cnt, 0, m_cfg.thres_gfft, m_cfg.min_corner_dist, mask);
      break;
    case 1:
      corners = featureDetect(img, detect_cnt, 1, m_cfg.thres_fast, 0, mask);
  }
  return corners;
}

void FeatureTracker::drawFeature(cv::Mat& bgr, const FeatureTracker::FeatureList& features) {
  int pt_radius = std::max(2, std::max(bgr.cols, bgr.rows) / 320);
  double text_scale = pt_radius * 0.15;
  cv::Scalar text_color(0, 255, 255);
  int text_thickness = pt_radius / 2;
  for (const auto& f : features) {
    int v = clip(static_cast<float>(f.track_cnt) / 20.0f * 255, 0, 255);
    cv::Scalar color(255 - v, 0, v);
    cv::circle(bgr, cv::Point(f.x, f.y), pt_radius, color, -1);
    char str[128] = {0};
    snprintf(str, 128, "%d", static_cast<int>(f.id));
    cv::putText(bgr, str, cv::Point(f.x + 3, f.y), cv::FONT_HERSHEY_COMPLEX, text_scale, text_color, text_thickness);
  }
}

bool calcOpticalFlowLK(const std::vector<cv::Mat>& prev_pyr, const std::vector<cv::Mat>& cur_pyr,
                       const std::vector<cv::Point2f>& prev_pts, std::vector<cv::Point2f>& cur_pts,
                       std::vector<uchar>& status, const cv::Size& patch_size, int max_level, bool cross_check,
                       double cross_check_thres, int border_thres, float max_move) {
  bool valid_check = !prev_pyr.empty() && !prev_pts.empty() && prev_pyr.size() == cur_pyr.size() &&
                     prev_pyr[0].size() == cur_pyr[0].size();
  if (!valid_check) {
    status.resize(prev_pts.size(), 0);
    return false;
  }
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(prev_pyr, cur_pyr, prev_pts, cur_pts, status, err, patch_size, max_level);
  if (max_move > 0) lkOutlierReject(prev_pts, cur_pts, status, prev_pyr[0].size(), border_thres, max_move);
  if (cross_check) {
    std::vector<cv::Point2f> back_pts = prev_pts;
    std::vector<uchar> status_back;
    cv::calcOpticalFlowPyrLK(cur_pyr, prev_pyr, cur_pts, back_pts, status_back, err, patch_size, max_level,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    for (size_t i = 0; i < status.size(); i++) {
      if (status[i] && (status_back[i] == 0 || cv::norm(back_pts[i] - prev_pts[i]) > cross_check_thres)) {
        status[i] = 0;
      }
    }
  }
  return true;
}

float shiTomasiScore(const cv::Mat& img, int u, int v) {
  assert(img.type() == CV_8UC1);

  float dXX = 0.0;
  float dYY = 0.0;
  float dXY = 0.0;
  const int halfbox_size = 4;
  const int box_size = 2 * halfbox_size;
  const int box_area = box_size * box_size;
  const int x_min = u - halfbox_size;
  const int x_max = u + halfbox_size;
  const int y_min = v - halfbox_size;
  const int y_max = v + halfbox_size;

  if (x_min < 1 || x_max >= img.cols - 1 || y_min < 1 || y_max >= img.rows - 1)
    return 0.0;  // patch is too close to the boundary

  const int stride = img.step.p[0];
  for (int y = y_min; y < y_max; ++y) {
    const uint8_t* ptr_left = img.data + stride * y + x_min - 1;
    const uint8_t* ptr_right = img.data + stride * y + x_min + 1;
    const uint8_t* ptr_top = img.data + stride * (y - 1) + x_min;
    const uint8_t* ptr_bottom = img.data + stride * (y + 1) + x_min;
    for (int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
      float dx = *ptr_right - *ptr_left;
      float dy = *ptr_bottom - *ptr_top;
      dXX += dx * dx;
      dYY += dy * dy;
      dXY += dx * dy;
    }
  }

  // Find and return smaller eigenvalue:
  dXX = dXX / (2.0 * box_area);
  dYY = dYY / (2.0 * box_area);
  dXY = dXY / (2.0 * box_area);
  return 0.5 * (dXX + dYY - sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
}

void lkOutlierReject(const std::vector<cv::Point2f>& prev_pts, const std::vector<cv::Point2f>& cur_pts,
                     std::vector<uchar>& status, const cv::Size& img_size, int border, float max_move,
                     float max_move_rate) {
  if (max_move <= 0) max_move = 1e10;
  for (size_t i = 0; i < status.size(); i++) {
    auto& s = status[i];
    if (s > 0 && (!inside(cur_pts[i], img_size, border) || dist(prev_pts[i], cur_pts[i]) > max_move)) s = 0;
  }
  if (max_move_rate > 1) {
    std::vector<std::pair<int, double>> ids_dists;
    for (size_t i = 0; i < status.size(); i++) {
      if (!status[i]) continue;
      const auto& p0 = prev_pts[i];
      const auto& p1 = cur_pts[i];
      // double dist = fabs(p0.x - p1.x) + fabs(p0.y - p1.y); //Manhatan dist
      double dist = cv::norm(p1 - p0);  // euclidian dist
      ids_dists.push_back(std::make_pair(i, dist));
    }
    if (ids_dists.size() > 10) {
      int mid = ids_dists.size() / 2;
      auto mid_it = ids_dists.begin() + mid;
      auto foo = [](const std::pair<int, double>& a, const std::pair<int, double>& b) { return a.second > b.second; };
      std::nth_element(ids_dists.begin(), mid_it, ids_dists.end(), foo);
      double thres = std::max(mid_it->second * max_move_rate, 10.0);
      for (auto it = ids_dists.begin(); it <= mid_it; ++it) {
        if (it->second > thres) status[it->first] = 0;
      }
    }
  }
}

std::vector<cv::Point2f> featureDetectFast(const cv::Mat& img, int detect_cnt, const cv::Mat& mask, float fast_thres,
                                           float fast_thres_min) {
  std::vector<cv::Point2f> corners;
  if (detect_cnt <= 0) return corners;
  corners.reserve(detect_cnt);
  std::vector<cv::KeyPoint> features;
#if (CV_MAJOR_VERSION >= 3)
  auto detector = cv::FastFeatureDetector::create(fast_thres);
  detector->detect(img, features, mask);
  if (fast_thres_min > 0 && fast_thres_min < fast_thres && static_cast<int>(features.size()) < detect_cnt) {
    features.clear();
    detector = cv::FastFeatureDetector::create(fast_thres_min);
    detector->detect(img, features, mask);
  }
#else
  cv::FastFeatureDetector detector(fast_thres);
  detector.detect(img, features, mask);
  if (fast_thres_min > 0 && fast_thres_min < fast_thres && static_cast<int>(features.size()) < detect_cnt) {
    features.clear();
    cv::FastFeatureDetector detector2(fast_thres_min);
    detector2.detect(img, features, mask);
  }
#endif
  if (static_cast<int>(features.size()) > detect_cnt) {
    // calc shi-tomasi score, select max_corners features into corners
    std::vector<std::pair<float, int>> score_ids;
    for (size_t i = 0; i < features.size(); i++) {
      const auto& k = features[i];
      float score = shiTomasiScore(img, k.pt.x, k.pt.y);
      score_ids.push_back(std::make_pair(score, i));
    }
    std::sort(score_ids.begin(), score_ids.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) { return a.first > b.first; });
    score_ids.resize(detect_cnt);
    for (const auto& it : score_ids) corners.push_back(features[it.second].pt);
  } else {
    for (const auto& f : features) corners.push_back(f.pt);
  }
  return corners;
}

std::vector<cv::Point2f> featureDetect(const cv::Mat& img, int detect_cnt, int feature_type, float param1, float param2,
                                       const cv::Mat& mask, const cv::Rect& roi) {
  assert(img.type() == CV_8UC1);
  cv::Mat img_roi, mask_roi;
  cv::Point base_pt(0, 0);
  if (roi.area() > 0) {
    auto crop_roi = rectCrop(roi, img.size());
    if (crop_roi.area() > 0) {
      img_roi = img(crop_roi);
      base_pt = cv::Point(crop_roi.x, crop_roi.y);
      if (!mask.empty()) mask_roi = mask(crop_roi);
    } else {
      img_roi = img;
      mask_roi = mask;
    }
  } else {
    img_roi = img;
    mask_roi = mask;
  }
  std::vector<cv::Point2f> corners;
  if (detect_cnt <= 0) return corners;
  cv::Size grid_size(16, 12);
  if (img.cols < img.rows) std::swap(grid_size.width, grid_size.height);
  corners.reserve(detect_cnt);
  switch (feature_type) {
    case 0:
      cv::goodFeaturesToTrack(img_roi, corners, detect_cnt, param1, param2, mask_roi);
      break;
    case 1:
      corners = featureDetectUniform(img_roi, grid_size, detect_cnt, mask_roi, feature_type);
    default:
      break;
  }
  if (base_pt.x != 0 || base_pt.y != 0) {
    for (auto& p : corners) {
      p.x += base_pt.x;
      p.y += base_pt.y;
    }
  }
  return corners;
}

std::vector<cv::Point2f> featureDetectUniform(const cv::Mat& img, const cv::Size& grid_size, int max_feature_cnt,
                                              const cv::Mat& mask, int feature_type,
                                              const std::vector<cv::Point2f>& exist_features) {
  // split image region into lot of grids, detect up to K features in each grid, K = max_feature_cnt/grid_cnt
  const int grid_cols = grid_size.width;
  const int grid_rows = grid_size.height;
  const int step_x = img.cols / grid_cols;
  const int step_y = img.rows / grid_rows;
  const int grid_cnt = grid_rows * grid_cols;
  const int cnt_per_grid = std::ceil(static_cast<float>(max_feature_cnt) / grid_cnt);
  std::vector<int> cnt_list(grid_cnt, cnt_per_grid);  ///< grid buffer, store max cnt can be added
  // handle old features, modify grid cnt
  if (!exist_features.empty()) {
    for (const auto& p : exist_features) {
      int ix = p.x / step_x;
      int iy = p.y / step_y;
      if (ix >= 0 && ix < grid_cols && iy >= 0 && iy < grid_rows) cnt_list[iy * grid_cols + ix]--;
    }
  }
  // remove part of grids with mask for acceleration
  if (mask.size() == img.size()) {
    cv::Mat grid_mask;
    cv::resize(mask, grid_mask, grid_size);
    uchar* ptr = grid_mask.data;
    for (int i = 0; i < grid_cnt; i++) {
      if (*ptr++ == 0) cnt_list[i] = 0;
    }
  }
  // detect feature in each grid
  std::vector<cv::Point2f> features;
  features.reserve(max_feature_cnt);
  int idx = 0;
  for (int i = 0; i < grid_rows; i++) {
    int start_row = i * step_y;
    for (int j = 0; j < grid_cols; j++, idx++) {
      int detect_cnt = cnt_list[idx];
      if (detect_cnt <= 0) continue;
      int start_col = j * step_x;
      std::vector<cv::Point2f> grid_features;
      cv::Rect roi(start_col, start_row, step_x, step_y);
      cv::Mat img_roi = img(roi);
      cv::Mat mask_roi;
      if (mask.size() == img.size()) mask_roi = mask(roi);
      if (feature_type == 0) {
        cv::goodFeaturesToTrack(img_roi, grid_features, detect_cnt, 0.001, 15, mask_roi);
      } else {
        grid_features = featureDetectFast(img_roi, detect_cnt, mask_roi);
      }
      for (const auto& p : grid_features) features.push_back(cv::Point2f(p.x + roi.x, p.y + roi.y));
    }
  }
  return features;
}

} /* namespace vs */