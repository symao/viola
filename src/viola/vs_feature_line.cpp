/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-10-28 15:07
 * @details
 */
#include "vs_feature_line.h"
#include "vs_basic.h"
#include "vs_feature.h"
#include "vs_geometry2d.h"

// OpenCV 4.x has remove LSD implementation for license reason, so we move the LSD code into 3rdparty/lsd
// if your OpenCV has LSD impl and you want to use it, set HAVE_CV_CONTRIB_LINE to 1 to use your OpenCV version
#define HAVE_CV_CONTRIB_LINE 0
#if HAVE_CV_CONTRIB_LINE
// have license issue
#include <opencv2/line_descriptor/descriptor.hpp>
#else
#include <lsd/lsd.h>
#endif

namespace vs {

VS_STATIC_FUNC void calcPointGradient(const cv::Mat& gray, int x, int y, float& dx, float& dy) {
  const int gap = 1;
  if (!inside(cv::Point(x, y), gray.size(), gap)) {
    dx = dy = 0;
  } else {
    const uchar* ptr1 = gray.ptr<uchar>(y - gap) + x;
    const uchar* ptr2 = ptr1 + gray.cols * gap;
    const uchar* ptr3 = ptr2 + gray.cols * gap;
    int16_t a[9] = {*(ptr1 - gap), *ptr1,         *(ptr1 + gap), *(ptr2 - gap), *ptr2,
                    *(ptr2 + gap), *(ptr3 - gap), *ptr3,         *(ptr3 + gap)};
    dx = (a[2] - a[0] + 2 * (a[5] - a[3]) + a[8] - a[6]) / 4;
    dy = (a[6] - a[0] + 2 * (a[7] - a[1]) + a[8] - a[2]) / 4;
  }
}

VS_STATIC_FUNC void discreteLines(const cv::Mat& img, const std::vector<cv::Vec4f>& lines,
                                  std::vector<cv::Point2f>& pts, std::vector<std::vector<int>>& line_ids, float delta,
                                  bool gradient_filter) {
  cv::Size img_size = img.size();
  int line_cnt = lines.size();
  pts.clear();
  pts.reserve(line_cnt * 10);
  line_ids.clear();
  line_ids.resize(line_cnt);
  cv::Mat filter_gray;
  float rate = 0.25;
  if (gradient_filter) cv::resize(img, filter_gray, cv::Size(), rate, rate);
  // sample line points for each line segment
  for (int i = 0; i < line_cnt; i++) {
    const auto& l = lines[i];
    cv::Point2f v(l[2] - l[0], l[3] - l[1]);
    v = normalize(v);
    auto& ids = line_ids[i];
    std::vector<cv::Point2f> pts_i;
    int npts = lineDiscreteSample(cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]), pts_i, delta, img_size);
    if (gradient_filter) {
      int k = 0;
      for (int j = 0; j < npts; j++) {
        const auto& p = pts_i[j];
        float dx, dy;
        calcPointGradient(filter_gray, p.x * rate, p.y * rate, dx, dy);
        float mag = hypotf(dx, dy);
        bool flag = mag > 20 && fabs((dx * v.x + dy * v.y) / mag) < 0.2;
        if (flag) pts[k++] = p;
      }
      pts.resize(k);
      npts = k;
    }
    if (npts < 2) continue;
    int size_before = pts.size();
    pts.insert(pts.end(), pts_i.begin(), pts_i.end());
    int size_after = pts.size();
    ids.reserve(npts);
    for (int j = size_before; j < size_after; j++) ids.push_back(j);
  }
}

VS_STATIC_FUNC void fitLinesFromTrackPoints(const std::vector<cv::Point2f>& track_pts,
                                            const std::vector<uchar>& track_status,
                                            const std::vector<std::vector<int>>& line_ids,
                                            std::vector<cv::Vec4f>& track_lines,
                                            std::vector<uint8_t>& track_line_status) {
  int line_cnt = line_ids.size();
  track_lines = std::vector<cv::Vec4f>(line_cnt);
  track_line_status = std::vector<uint8_t>(line_cnt, 0);
  for (int i = 0; i < line_cnt; i++) {
    const auto& ids = line_ids[i];
    if (ids.empty()) continue;
    std::vector<cv::Point2f> pts;
    pts.reserve(ids.size());
    for (auto j : ids) {
      if (track_status[j] > 0) pts.push_back(track_pts[j]);
    }
    cv::Vec4f li;
    bool ok = lineDetectRansac(pts, li, 3, 10, 0.5);
    if (ok) {
      track_lines[i] = li;
      track_line_status[i] = 1;
    }
  }
}

VS_STATIC_FUNC cv::Mat drawLineTrack(const cv::Mat& prev_img, const cv::Mat& cur_img,
                                     const std::vector<cv::Point2f>& prev_pts, const std::vector<cv::Point2f>& cur_pts,
                                     const std::vector<uchar>& track_status,
                                     const std::vector<std::vector<int>>& line_ids,
                                     const std::vector<cv::Vec4f>& prev_lines, const std::vector<cv::Vec4f>& cur_lines,
                                     const std::vector<uchar>& track_line_status) {
  cv::Mat img_show;
  cv::hconcat(prev_img, cur_img, img_show);
  if (img_show.channels() == 1) cv::cvtColor(img_show, img_show, cv::COLOR_GRAY2BGR);
  int w = prev_img.cols;
  for (size_t i = 0; i < line_ids.size(); i++) {
    cv::Scalar color(randi(255), randi(255), randi(255));
    // draw prev line
    const auto& l = prev_lines[i];
    cv::line(img_show, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), color, 2);
    // draw track points
    const auto& ids = line_ids[i];
    for (int idx : ids) {
      if (track_status[idx]) {
        cv::circle(img_show, prev_pts[idx], 2, color, -1);
        cv::circle(img_show, cur_pts[idx] + cv::Point2f(w, 0), 2, color, -1);
        cv::line(img_show, cur_pts[idx] + cv::Point2f(w, 0), prev_pts[idx] + cv::Point2f(w, 0), color, 1);
      } else {
        cv::circle(img_show, prev_pts[idx], 2, cv::Scalar(0, 0, 255), -1);
      }
    }
  }
  return img_show;
}

static cv::Ptr<cv::LineSegmentDetector> plsd() {
  static cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD, 0.8, 0.6, 2, 10);
  return lsd;
}

int detectLine(const cv::Mat& gray, std::vector<cv::Vec4f>& lines, const cv::Rect& roi) {
  if (roi.width > 0 || roi.height > 0) {
    cv::Mat subimg = gray(roi);
    // line segment detect
    plsd()->detect(subimg, lines);
    if (!lines.empty()) {
      for (auto& l : lines) {
        l[0] += roi.x;
        l[1] += roi.y;
        l[2] += roi.x;
        l[3] += roi.y;
      }
    }
  } else {
    plsd()->detect(gray, lines);
  }
  return lines.size();
}

int detectLine(const cv::Mat& gray, std::vector<cv::Vec4f>& lines, const cv::Mat& mask) {
  // cv::LineSegmentDetector not support mask, replace with LSDDetector
  // plsd()->detect(gray, lines, mask);
  plsd()->detect(gray, lines);
  return lines.size();
}

void lineOpticalFlowLK(const std::vector<cv::Mat>& prev_pyr, const std::vector<cv::Mat>& cur_pyr,
                       const std::vector<cv::Vec4f>& prev_lines, std::vector<cv::Vec4f>& cur_lines,
                       std::vector<uint8_t>& status, float delta, bool cross_check, const cv::Size& patch_size,
                       int pyr_level, bool gradient_filter, bool draw) {
  bool input_invalid = prev_lines.empty() || prev_pyr.empty() || prev_pyr.size() != cur_pyr.size() ||
                       prev_pyr[0].size != cur_pyr[0].size;
  if (input_invalid) return;
  // discrete lines into points
  // Timer t1;
  cv::Size img_size = prev_pyr[0].size();
  int prev_line_cnt = prev_lines.size();
  std::vector<cv::Point2f> prev_pts;
  std::vector<std::vector<int>> line_ids;
  discreteLines(prev_pyr[0], prev_lines, prev_pts, line_ids, delta, gradient_filter);
  // t1.stop();

  // track all line points with LK
  // Timer t2;
  if (prev_pts.empty()) return;
  std::vector<cv::Point2f> cur_pts;
  std::vector<uchar> track_status;
  std::vector<float> error;
  cv::calcOpticalFlowPyrLK(prev_pyr, cur_pyr, prev_pts, cur_pts, track_status, error, patch_size, pyr_level);
  lkOutlierReject(prev_pts, cur_pts, track_status, img_size, 5, 50);
  // t2.stop();

  // remove outlier with cross check
  // Timer t3;
  if (cross_check) {
    // track points from cur image to prev image, and check if each point lies in prev line
    std::vector<cv::Point2f> ref_pts;
    std::vector<uchar> back_status;
    cv::calcOpticalFlowPyrLK(cur_pyr, prev_pyr, cur_pts, ref_pts, back_status, error, patch_size, pyr_level);
    for (int i = 0; i < prev_line_cnt; i++) {
      const auto& l = prev_lines[i];
      const auto& ids = line_ids[i];
      cv::Point2f p1(l[0], l[1]);
      cv::Point2f p2(l[2], l[3]);
      auto dir = normalize(p2 - p1);
      for (int j : ids) {
        if (track_status[j] > 0) {
          bool ok = back_status[j] != 0 && inside(ref_pts[j], img_size) && fabs(dir.cross(ref_pts[j] - p1)) < 3;
          if (!ok) track_status[j] = 0;
        }
      }
    }
  }
  // t3.stop();

  // fit line for each line points
  // Timer t4;
  fitLinesFromTrackPoints(cur_pts, track_status, line_ids, cur_lines, status);
  // t4.stop();
  // printf("cost:(%.2f %.2f %.2f %.2f) pts:%d\n", t1.getMsec(), t2.getMsec(), t3.getMsec(),
  //        t4.getMsec(), (int)prev_pts.size());
  if (draw) {
    cv::Mat img_show = drawLineTrack(prev_pyr[0], cur_pyr[0], prev_pts, cur_pts, track_status, line_ids, prev_lines,
                                     cur_lines, status);
    cv::imshow("lineOpticalFlowLK", img_show);
  }
}

void lineOpticalFlowLK(const cv::Mat& prev_img, const cv::Mat& cur_img, const std::vector<cv::Vec4f>& prev_lines,
                       std::vector<cv::Vec4f>& cur_lines, std::vector<uint8_t>& status, float delta, bool cross_check,
                       const cv::Size& patch_size, int pyr_level, bool gradient_filter, bool draw) {
  std::vector<cv::Mat> prev_pyr;
  std::vector<cv::Mat> cur_pyr;
  cv::buildOpticalFlowPyramid(prev_img, prev_pyr, patch_size, pyr_level);
  cv::buildOpticalFlowPyramid(cur_img, cur_pyr, patch_size, pyr_level);
  lineOpticalFlowLK(prev_pyr, cur_pyr, prev_lines, cur_lines, status, delta, cross_check, patch_size, pyr_level,
                    gradient_filter, draw);
}

#if HAVE_CV_CONTRIB_LINE
class LineDetectorLsd : public LineDetector {
 public:
  LineDetectorLsd() { m_lsd = cv::line_descriptor::LSDDetector::createLSDDetector(); }

  virtual void detect(const cv::Mat& img, Line2DList& lines, const cv::Mat& mask = cv::Mat()) {
    std::vector<cv::line_descriptor::KeyLine> out;
    m_lsd->detect(img, out, 2, 1, mask);
    vecMapping(out, lines, [](const cv::line_descriptor::KeyLine& a) {
      return Line2D(a.startPointX, a.sPointInOctaveY, a.endPointX, a.endPointY);
    });
  }

 private:
  cv::Ptr<cv::line_descriptor::LSDDetector> m_lsd;
};

class LineDetectorLbd : public LineDetector {
 public:
  LineDetectorLbd() { m_bd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor(); }

  virtual void detect(const cv::Mat& img, Line2DList& lines, const cv::Mat& mask = cv::Mat()) {
    std::vector<cv::line_descriptor::KeyLine> out;
    m_bd->detect(img, out, mask);
    vecMapping(out, lines, [](const cv::line_descriptor::KeyLine& a) {
      return Line2D(a.startPointX, a.sPointInOctaveY, a.endPointX, a.endPointY);
    });
  }

 private:
  cv::Ptr<cv::line_descriptor::BinaryDescriptor> m_bd;
};

#else   // HAVE_CV_CONTRIB_LINE
class LineDetectorLsd : public LineDetector {
 public:
  LineDetectorLsd() { m_lsd = cv::createLineSegmentDetectorNew(cv::LSD_REFINE_STD); }

  virtual void detect(const cv::Mat& img, Line2DList& lines, const cv::Mat& mask = cv::Mat()) {
    std::vector<cv::Vec4f> out;
    m_lsd->detect(img, out, mask);
    vecMapping(out, lines, [](const cv::Vec4f& a) { return Line2D(a); });
  }

 private:
  cv::Ptr<cv::LineSegmentDetectorNew> m_lsd;
};

class LineDetectorLbd : public LineDetector {
 public:
  virtual void detect(const cv::Mat& img, Line2DList& lines, const cv::Mat& mask = cv::Mat()) {
    printf("[ERROR]LineDetectorLbd::%s: not implemented.", __func__);
  }
};
#endif  // HAVE_CV_CONTRIB_LINE

LineDetectorPtr createLineDetector(int type) {
  switch (type) {
    case LINE_DETECTOR_LSD:
      return std::make_shared<LineDetectorLsd>();
    case LINE_DETECTOR_LBD:
      return std::make_shared<LineDetectorLbd>();
    default:
      return std::make_shared<LineDetectorLsd>();
  }
}

LineFeatureTracker::LineFeatureTracker()
    : m_process_idx(0), m_feature_id(1), m_line_detector(createLineDetector(LINE_DETECTOR_LSD)) {}

LineFeatureTracker::TrackResult LineFeatureTracker::process(const cv::Mat& img, const cv::Mat& mask) {
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
  LineFeatureList cur_lines;
  if (!m_prev_lines.empty()) {
    // calc optical flow
    std::vector<cv::Vec4f> prev_line_vec = cvt(m_prev_lines);
    std::vector<cv::Vec4f> cur_line_vec;
    std::vector<uchar> status;
    lineOpticalFlowLK(m_prev_pyr, cur_pyr, prev_line_vec, cur_line_vec, status, 5.0, m_cfg.cross_check,
                      m_cfg.lk_patch_size, m_cfg.lk_pyr_level, false, true);
    if (!status.empty()) {
      cur_lines = m_prev_lines;
      vecReduce(cur_lines, status);
      vecReduce(cur_line_vec, status);
      for (size_t i = 0; i < cur_lines.size(); i++) {
        auto& l = cur_lines[i];
        l.update(cur_line_vec[i]);
        l.track_cnt++;
      }
    }
  }

  // add new lines
  bool add_new = cur_lines.empty();
  // bool add_new = true;
  if (add_new) {
    cv::Mat line_mask = mask.clone();
    for (const auto& l : cur_lines) cv::line(line_mask, l.p1, l.p2, cv::Scalar(0), 5);
    Line2DList detect_out;
    m_line_detector->detect(process_img, detect_out, mask);
    LineFeatureList new_lines;
    for (const auto& l : detect_out) {
      if (l.length() < m_cfg.min_line_len) continue;
      new_lines.emplace_back(l, m_feature_id++, 0);
    }
    cur_lines.insert(cur_lines.end(), new_lines.begin(), new_lines.end());
  }

  // post process
  m_prev_lines = cur_lines;
  m_prev_pyr = cur_pyr;
  m_process_idx++;

  TrackResult res;
  res.line_features = cur_lines;
  return res;
}

std::vector<cv::Vec4f> LineFeatureTracker::cvt(const LineFeatureList& lines) {
  std::vector<cv::Vec4f> res;
  vecMapping(lines, res, [](const LineFeature& l) { return cv::Vec4f(l.p1.x, l.p1.y, l.p2.x, l.p2.y); });
  return res;
}

void LineFeatureTracker::drawLineFeature(cv::Mat& bgr, const LineFeatureList& line_features, bool draw_id) {
  int pt_radius = std::max(2, std::max(bgr.cols, bgr.rows) / 320);
  double text_scale = pt_radius * 0.15;
  cv::Scalar text_color(0, 255, 255);
  int text_thickness = pt_radius / 2;
  for (const auto& l : line_features) {
    int v = clip(static_cast<float>(l.track_cnt) / 20.0f * 255, 0, 255);
    cv::Scalar color(255 - v, 0, v);
    cv::line(bgr, l.p1, l.p2, color, 1);
    cv::circle(bgr, l.p1, 1, cv::Scalar(0, 255, 0), -1);
    cv::circle(bgr, l.p2, 1, cv::Scalar(0, 255, 0), 1);
    if (draw_id) {
      char str[128] = {0};
      snprintf(str, 128, "%d", static_cast<int>(l.id));
      cv::putText(bgr, str, l.center(), cv::FONT_HERSHEY_COMPLEX, text_scale, text_color, text_thickness);
    }
  }
}
}  // namespace vs