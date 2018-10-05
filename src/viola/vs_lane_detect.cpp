/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_lane_detect.h"
#include <deque>
#include <opencv2/highgui.hpp>
#include "vs_basic.h"
#include "vs_color_filter.h"
#include "vs_geometry2d.h"
#include "vs_improc.h"
#include "vs_random.h"

namespace vs {

#define MIN_LINE_LEN 60
#define MIN_MASK_RATE 0.8f

static void removeInvalid(std::vector<cv::Vec4f>& lines, const cv::Mat& mask);

static void detectLane(std::vector<cv::Vec4f>& lines, const cv::Mat& mask, LaneList& lanes, const cv::Mat& K,
                       const cv::Mat& T_c_b, float min_lane_w, float max_lane_w, float max_lane_x);

static void mergeLane(LaneList& lanes, const cv::Mat& mask);

static cv::Mat calLsdImage(const cv::Mat& img, int method = 1);

Lane::Lane(const cv::Point2f& _p1, const cv::Point2f& _p2) : p1(_p1), p2(_p2) {
  dir = p2 - p1;
  center = (p1 + p2) / 2;
  len = hypotf(dir.x, dir.y);
  if (len > VS_EPS) dir /= len;
}

int laneDetect(const cv::Mat& img, LaneList& lanes, const cv::Mat& K, const cv::Mat& T_c_b, int lane_type,
               float min_lane_w, float max_lane_w, bool draw) {
  if (img.channels() != 3) {
    printf("[ERROR]vsLaneDetect: Need input bgr image.\n");
    return 0;
  }

  // calc color mask
  auto foo_color = [](int type) {
    switch (type) {
      case LANE_PURE_YELLOW:
      default:
        return ColorModelList({ColorModel::yellow()});
    }
  };
  static auto model = foo_color(lane_type);
  cv::Mat color_mask;
  colorFilter(img, color_mask, model, 0.25, 1);

  // line seg detect
  static auto lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD, 0.6, 4, 2, 22.5);
  int lsd_method = 1;
  if (lane_type == 10) lsd_method = 3;
  cv::Mat img_lsd = calLsdImage(img, lsd_method);
  std::vector<cv::Vec4f> lines, raw_lines;
  lsd->detect(img_lsd, lines);
  if (draw) raw_lines = lines;

  // filter with color mask
  removeInvalid(lines, color_mask);

  // merge lsd lines
  mergeLine(lines, 0, 0.996f, 3.0f, 0.0f, 5.0f);

  // detect lane
  detectLane(lines, color_mask, lanes, K, T_c_b, min_lane_w, max_lane_w, 20);
  int nlanes_raw = lanes.size();
  mergeLane(lanes, color_mask);
  int nlanes = lanes.size();

  if (draw) {
    cv::Mat Tbc = T_c_b.inv();
    cv::Mat H = K * Tbc.rowRange(0, 3);
    auto fb2c = [H](const cv::Point2f& p) {
      cv::Mat pc = H * (cv::Mat_<double>(4, 1) << p.x, p.y, 0, 1);
      double s = pc.at<double>(2, 0);
      if (s <= 0) {
        printf("[ERROR]s:%f\n", s);
        return cv::Point2f(0, 0);
      }
      return cv::Point2f(pc.at<double>(0, 0) / s, pc.at<double>(1, 0) / s);
    };
    cv::Mat img_show = img.clone();
    for (auto l : lines) {
      cv::line(img_show, cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]), cv::Scalar(0, 255, 0), 2);
    }
    for (auto lane : lanes) {
      cv::line(img_show, fb2c(lane.p1), fb2c(lane.p2), cv::Scalar(randi(255), randi(255), randi(255)), 3);
    }

    cv::cvtColor(img_lsd, img_lsd, cv::COLOR_GRAY2BGR);
    for (const auto& l : raw_lines) {
      cv::line(img_lsd, cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]), cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("lane_detect", img_show);
    cv::imshow("mask", color_mask);
    cv::imshow("lsd", img_lsd);
    printf("detect lines:%d lanes:%d final:%d\n", static_cast<int>(lines.size()), nlanes_raw, nlanes);
  }
  return nlanes;
}

static inline bool valid(float x1, float y1, float x2, float y2, const cv::Mat& mask, float min_len = 0,
                         float min_rate = MIN_MASK_RATE) {
  // if(x1 < 0 || y1 < 0 || x1 >= mask.cols || y1 >= mask.rows
  //     || x2 < 0 || y2 < 0 || x2 >= mask.cols || y2 >= mask.rows)
  //     return false;

  cv::Point2f step(x2 - x1, y2 - y1);
  float len = hypotf(step.x, step.y);
  if (len < min_len) return false;

  int k = len;
  if (k > 0)
    step /= k;
  else
    step = cv::Point2f(0, 0);

  int cnt = 0;
  int cnt_skip = 0;
  cv::Point2f p(x1, y1);
  for (int i = 0; i <= k; i++, p += step) {
    if (p.x < 0 || p.y < 0 || p.x >= mask.cols || p.y >= mask.rows)
      cnt_skip++;
    else if (mask.at<uchar>(p))
      cnt++;
  }
  int sum = k + 1 - cnt_skip;
  if (sum < 1) return false;
  float rate = static_cast<float>(cnt) / sum;
  return rate > min_rate;
}

static bool valid(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Mat& mask, float min_len = 0,
                  float min_rate = MIN_MASK_RATE) {
  return valid(p1.x, p1.y, p2.x, p2.y, mask, min_len, min_rate);
}

static bool valid(const cv::Vec4f& l, const cv::Mat& mask, float min_len = 0, float min_rate = MIN_MASK_RATE) {
  return valid(l[0], l[1], l[2], l[3], mask, min_len, min_rate);
}

static void removeInvalid(std::vector<cv::Vec4f>& lines, const cv::Mat& mask) {
  int cnt = 0;
  for (size_t i = 0; i < lines.size(); i++) {
    const auto& l = lines[i];
    if (valid(l, mask, MIN_LINE_LEN)) lines[cnt++] = l;
  }
  lines.resize(cnt);
}

static void detectLane(std::vector<cv::Vec4f>& lines, const cv::Mat& mask, LaneList& lanes, const cv::Mat& K,
                       const cv::Mat& T_c_b, float min_lane_w, float max_lane_w, float max_lane_x) {
  if (lines.empty()) return;
  int nlines = lines.size();

  auto foo_valid_d = [min_lane_w, max_lane_w](float d) { return inRange(fabs(d), min_lane_w, max_lane_w); };
  auto foo_valid_cos = [](float cos) { return fabs(cos) > 0.98; };
  auto foo_valid_overlap = [](float r) { return r > 0.6; };

  cv::Mat K_inv = K.inv();
  cv::Mat Rcb = T_c_b(cv::Rect(0, 0, 3, 3));
  cv::Mat tcb = T_c_b(cv::Rect(3, 0, 1, 3));
  cv::Mat Rcb_Kinv = Rcb * K_inv;
  auto fc2b = [Rcb_Kinv, tcb](const cv::Point2f& u, cv::Point2f& pb) {
    cv::Mat pt = Rcb_Kinv * (cv::Mat_<double>(3, 1) << u.x, u.y, 1);
    double ptz = pt.at<double>(2, 0);
    if (ptz >= 0) {
      printf("[ERROR]ptz:%f\n", ptz);
      return false;
    }
    double s = -tcb.at<double>(2, 0) / ptz;
    if (fequal(s, 0)) return false;
    cv::Mat p_b = s * pt + tcb;
    pb.x = p_b.at<double>(0);
    pb.y = p_b.at<double>(1);
    return true;
  };

  struct LinePack {
    void cal() {
      dir = p2 - p1;
      c = (p1 + p2) / 2;
      ucenter = (u1 + u2) / 2;
      len = hypotf(dir.x, dir.y);
      if (len > VS_EPS) dir /= len;
      theta = atan2(dir.y, dir.x);
      if (dir.x < 0) reverse();
    }

    void reverse() {
      std::swap(u1, u2);
      std::swap(p1, p2);
      dir = -dir;
      theta = -theta;
    }

    void print() {
      printf(
          "u1(%.1f %.1f) u2(%.1f %.1f) p1(%.3f %.3f) p2(%.3f %.3f) "
          "dir:(%.3f %.3f) %.1f len:%.2f\n",
          u1.x, u1.y, u2.x, u2.y, p1.x, p1.y, p2.x, p2.y, dir.x, dir.y, rad2deg(theta), len);
    }

    cv::Point2f u1, u2, p1, p2, c, ucenter, dir;
    float theta;
    float len;
  };

  std::vector<LinePack> list;
  list.reserve(nlines);
  for (const auto& l : lines) {
    LinePack lp;
    lp.u1.x = l[0];
    lp.u1.y = l[1];
    lp.u2.x = l[2];
    lp.u2.y = l[3];
    if (fc2b(lp.u1, lp.p1) && fc2b(lp.u2, lp.p2)) {
      lp.cal();
#if 1  // cut line end point at max dist
      double max_x = max_lane_x;
      if (lp.p1.x > max_x && lp.p2.x > max_x) {
        continue;
      } else if (lp.p1.x > max_x) {
        double k = (max_x - lp.p2.x) / (lp.p1.x - lp.p2.x);
        lp.p1 = lp.p2 + k * (lp.p1 - lp.p2);
        lp.cal();
      } else if (lp.p2.x > max_x) {
        double k = (max_x - lp.p1.x) / (lp.p2.x - lp.p1.x);
        lp.p2 = lp.p1 + k * (lp.p2 - lp.p1);
        lp.cal();
      }
#endif
      list.push_back(lp);
      // lp.print();
    }
  }
  std::vector<std::deque<Lane>> lane_buffer;
  for (int i = 0; i < nlines; i++) {
    const auto& li = list[i];
    for (int j = i + 1; j < nlines; j++) {
      auto lj = list[j];

      float cos = li.dir.dot(lj.dir);
      if (cos < 0) lj.reverse();
      cv::Point2f dc = lj.c - li.c;
#if 0
            printf("(%d %d %d %d %d %d) %.2f d:%.3f %.3f overlap:%.2f %.2f "
                    "len:%.2f %.2f dir:%f %f\n",
                foo_valid_cos(cos),
                foo_valid_d(dc.cross(li.dir)) && foo_valid_d(dc.cross(lj.dir)),
                foo_valid_overlap(std::max(overlapRate(li.u1, li.u2, lj.u1, lj.u2, 2),
                                           overlapRate(li.p1, li.p2, lj.p1, lj.p2, 2))),
                valid(li.ucenter, lj.ucenter, mask, 0),
                valid(li.u1, lj.u2, mask, 0),
                valid(li.u2, lj.u1, mask, 0),
                cos, fabs(dc.cross(li.dir)), fabs(dc.cross(lj.dir)),
                overlapRate(li.u1, li.u2, lj.u1, lj.u2, 2),
                overlapRate(li.p1, li.p2, lj.p1, lj.p2, 2),
                li.len, lj.len,
                atan2(li.dir.y, li.dir.x), atan2(lj.dir.y, lj.dir.x));
#endif
      if (foo_valid_cos(cos) && foo_valid_d(dc.cross(li.dir)) && foo_valid_d(dc.cross(lj.dir)) &&
          foo_valid_overlap(
              std::max(overlapRate(li.u1, li.u2, lj.u1, lj.u2, 2), overlapRate(li.p1, li.p2, lj.p1, lj.p2, 2))) &&
          valid(li.ucenter, lj.ucenter, mask, 0) && valid(li.u1, lj.u2, mask, 0) && valid(li.u2, lj.u1, mask, 0)) {
        // construct two edge to centeral line
        Lane lane((li.p1 + lj.p1) / 2, (li.p2 + lj.p2) / 2);
#if 0
                if(cos < 0.996)
                {
                    cv::Point2f intsec = lineIntersect(li.p1, li.dir, lj.p1, lj.dir);
                    cv::Point2f dir_c = li.dir + lj.dir;
                    cv::Point2f cl2 = intsec + dir_c;
                    dir_c /= hypotf(dir_c.x, dir_c.y);
                    lane = Lane((project2line(intsec, cl2, li.p1)
                                + project2line(intsec, cl2, lj.p1)) / 2,
                                (project2line(intsec, cl2, li.p2)
                                + project2line(intsec, cl2, lj.p2)) / 2);
                }
#endif
        lane.lane_type = 3;
        lanes.push_back(lane);
      }
    }
  }
}

static void mergeLane(LaneList& lanes, const cv::Mat& mask) {
  auto foo_need_merge = [mask](const Lane& a, const Lane& b) {
    if (!isCoincidence(a.center, a.dir, b.center, b.dir, 0, 0.996, 0.03)) return false;
    if (overlapLen(a.p1, a.p2, b.p1, b.p2) >= 0) return true;
    float d[4] = {dist(a.p1, b.p1), dist(a.p2, b.p2), dist(a.p1, b.p2), dist(a.p2, b.p1)};
    float dmin = std::min(std::min(d[0], d[1]), std::min(d[2], d[3])) + VS_EPS;
    return (d[0] > dmin || valid(cv::Vec4f(a.p1.x, a.p1.y, b.p1.x, b.p1.y), mask)) &&
           (d[1] > dmin || valid(cv::Vec4f(a.p2.x, a.p2.y, b.p2.x, b.p2.y), mask)) &&
           (d[2] > dmin || valid(cv::Vec4f(a.p1.x, a.p1.y, b.p2.x, b.p2.y), mask)) &&
           (d[3] > dmin || valid(cv::Vec4f(a.p2.x, a.p2.y, b.p1.x, b.p1.y), mask));
  };

  auto foo_merge = [](const Lane& a, const Lane& b) {
    float kmin = 0.0f;
    float kmax = 1.0f;
    float k1 = (b.p1 - a.p1).dot(a.dir) / a.len;
    float k2 = (b.p2 - a.p1).dot(a.dir) / a.len;
    if (kmin > k1) kmin = k1;
    if (kmax < k1) kmax = k1;
    if (kmin > k2) kmin = k2;
    if (kmax < k2) kmax = k2;
    cv::Point2f p1, p2;
    if (fequal(kmin, 0))
      p1 = a.p1;
    else if (fequal(kmin, k1))
      p1 = b.p1;
    else
      p1 = b.p2;
    if (fequal(kmax, 1))
      p2 = a.p2;
    else if (fequal(kmax, k1))
      p2 = b.p1;
    else
      p2 = b.p2;

    Lane c(p1, p2);
    c.lane_type = a.lane_type;
    return c;
  };
  merge<Lane>(lanes, foo_need_merge, foo_merge);
}

static cv::Mat calLsdImage(const cv::Mat& img, int method) {
  cv::Mat img_lsd;
  switch (method) {
    case 0:
      cv::cvtColor(img, img_lsd, cv::COLOR_BGR2GRAY);
      break;
    case 1: {
#if 1
      cv::Mat hsv;
      cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
      std::vector<cv::Mat> split;
      cv::split(hsv, split);
      img_lsd = split[1];
#else
      cv::Mat sat = bgr2saturation(img);
#endif
#if 0
            cv::Mat img_hsv;
            cv::hconcat(split[0], split[1], img_hsv);
            cv::hconcat(img_hsv, split[2], img_hsv);
            cv::imshow("hsv", img_hsv);
#endif
      break;
    }
    case 2: {
      img_lsd = bgr2hue(img);
      break;
    }
    case 3: {
      cv::Mat hsv;
      cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
      std::vector<cv::Mat> split;
      cv::split(hsv, split);
      img_lsd = split[2];
    }
    default:
      cv::cvtColor(img, img_lsd, cv::COLOR_BGR2GRAY);
      break;
  }
  return img_lsd;
}

#if 0
static void adjustLsdResult(std::vector<cv::Vec4f>& lines, const cv::Size& size)
{
    // adjust lsd lines which out of image boundry
    int w = size.width;
    int h = size.height;
    auto f_in = [w,h](const cv::Point2f& p){return p.x >= 0 && p.y >= 0 && p.x < w && p.y < h;};
    for(auto & l : lines)
    {
        cv::Point2f p1(l[0], l[1]);
        cv::Point2f p2(l[2], l[3]);
        bool in1 = f_in(p1);
        bool in2 = f_in(p2);
        if(in1 && in2) continue;
        else if(!in1 && !in2)
        {
            l = cv::Vec4f(0, 0, 0, 0);
            continue;
        }
        // p1 out, p2 in
        if(in1) std::swap(p1, p2);
        auto step = p2 - p1;
        float len = hypotf(step.x, step.y);
        if(len < 1)
        {
            l = cv::Vec4f(0, 0, 0, 0);
            continue;
        }
        step /= len;
        for(int i = 0; i < int(len); i++)
        {
            p1 += step;
            if(f_in(p1)) break;
        }
        l = cv::Vec4f(p1.x, p1.y, p2.x, p2.y);
    }
}
#endif

} /* namespace vs */