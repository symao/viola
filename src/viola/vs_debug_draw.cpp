/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_debug_draw.h"
#include "vs_geometry2d.h"
#include "vs_random.h"

static cv::Scalar COLOR_TRACK_PT(0, 0, 255);
static cv::Scalar COLOR_TRACK_LINE(255, 0, 0);
static cv::Scalar COLOR_LOST_PT(0, 255, 255);
static cv::Scalar COLOR_OUTLIER_LINE(150, 150, 150);
namespace vs {

cv::Mat toRgb(const cv::Mat& img, bool depp_copy) {
  if (img.empty()) return img;
  cv::Mat rgb;
  if (img.channels() == 1)
    cv::cvtColor(img, rgb, cv::COLOR_GRAY2BGR);
  else if (depp_copy)
    img.copyTo(rgb);
  else
    rgb = img;
  return rgb;
}

cv::Mat toGray(const cv::Mat& img, bool depp_copy) {
  if (img.empty()) return img;
  cv::Mat gray;
  if (img.channels() == 3)
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  else if (depp_copy)
    img.copyTo(gray);
  else
    gray = img;
  return gray;
}

void toRgb(const cv::Mat& src, cv::OutputArray dst) {
  if (src.empty())
    return;
  else if (src.channels() == 1)
    cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
  else if (src.channels() == 3)
    src.copyTo(dst);
  else if (src.channels() == 4)
    cv::cvtColor(src, dst, cv::COLOR_BGRA2BGR);
}

void toGray(const cv::Mat& src, cv::OutputArray dst) {
  if (src.empty())
    return;
  else if (src.channels() == 1)
    src.copyTo(dst);
  else if (src.channels() == 3)
    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
  else if (src.channels() == 4)
    cv::cvtColor(src, dst, cv::COLOR_BGRA2GRAY);
}

cv::Mat drawMatf(const cv::Mat& img, float k) {
  cv::Mat img_draw;
  img.convertTo(img_draw, CV_8U, k);
#if (CV_MAJOR_VERSION >= 3)
  cv::applyColorMap(img_draw, img_draw, cv::COLORMAP_JET);
#endif
  return img_draw;
}

cv::Mat drawMask(const cv::Mat& img, const cv::Mat& mask, const cv::Scalar& color, float ratio, bool deep_copy) {
  if (img.size() != mask.size()) return img;
  cv::Mat img_show = toRgb(img, deep_copy);
  if (img_show.type() != CV_8UC3) return img;
  ratio = clip(ratio, 0.0f, 1.0f);
  if (ratio <= 0.0f) return img_show;
  if (mask.type() == CV_8UC1) {
    for (int i = 0; i < img.rows; i++) {
      cv::Vec3b* ptr_img = img_show.ptr<cv::Vec3b>(i);
      const uchar* ptr_mask = mask.ptr<uchar>(i);
      for (int j = 0; j < img.cols; j++) {
        float k = ratio * ((*ptr_mask++) / 255.0f);
        cv::Vec3b& v = *ptr_img++;
        v[0] = v[0] * (1.0f - k) + color[0] * k;
        v[1] = v[1] * (1.0f - k) + color[1] * k;
        v[2] = v[2] * (1.0f - k) + color[2] * k;
      }
    }
  } else if (mask.type() == CV_32FC1) {
    for (int i = 0; i < img.rows; i++) {
      cv::Vec3b* ptr_img = img_show.ptr<cv::Vec3b>(i);
      const float* ptr_mask = mask.ptr<float>(i);
      for (int j = 0; j < img.cols; j++) {
        float k = ratio * clip(*ptr_mask++, 0.0f, 1.0f);
        cv::Vec3b& v = *ptr_img++;
        v[0] = v[0] * (1.0f - k) + color[0] * k;
        v[1] = v[1] * (1.0f - k) + color[1] * k;
        v[2] = v[2] * (1.0f - k) + color[2] * k;
      }
    }
  }
  return img_show;
}

cv::Mat drawSubImage(const cv::Mat& img, const cv::Mat& sub_img, int draw_position, float resize_rate, bool deep_copy) {
  if (img.empty() || sub_img.empty() || img.depth() != sub_img.depth()) return img;

  cv::Mat resize_img;
  if (!fequal(resize_rate, 1)) {
    cv::resize(sub_img, resize_img, cv::Size(), resize_rate, resize_rate);
  } else {
    resize_img = sub_img;
  }

  cv::Mat draw_sub_img;
  if (img.channels() == 1 && resize_img.channels() == 3) {
    cv::cvtColor(resize_img, draw_sub_img, cv::COLOR_BGR2GRAY);
  } else if (img.channels() == 3 && resize_img.channels() == 1) {
    cv::cvtColor(resize_img, draw_sub_img, cv::COLOR_GRAY2BGR);
  } else {
    draw_sub_img = resize_img;
  }

  cv::Mat out_img = deep_copy ? img.clone() : img;
  cv::Rect roi(0, 0, draw_sub_img.cols, draw_sub_img.rows);
  switch (draw_position) {
    case POS_TL:  // top-left
      roi.x = roi.y = 0;
      break;
    case POS_TR:  // top-right
      roi.x = out_img.cols - draw_sub_img.cols;
      roi.y = 0;
      break;
    case POS_BL:  // bottom-left
      roi.x = 0;
      roi.y = out_img.rows - draw_sub_img.rows;
      break;
    case POS_BR:  // bottom-right
      roi.x = out_img.cols - draw_sub_img.cols;
      roi.y = out_img.rows - draw_sub_img.rows;
      break;
    default:
      break;
  }
  draw_sub_img.copyTo(out_img(roi));
  return out_img;
}

cv::Mat drawPoints(const cv::Mat& img, const std::vector<cv::Point2f>& pts, int radius, const cv::Scalar& color,
                   int thickness, bool deep_copy) {
  cv::Mat img_show = toRgb(img, deep_copy);
  for (const auto& p : pts) cv::circle(img_show, p, radius, color, thickness);
  return img_show;
}

cv::Mat drawLines(const cv::Mat& img, const std::vector<cv::Vec4f>& lines, int thickness, bool fix_color,
                  const cv::Scalar& color, bool deep_copy) {
  cv::Mat img_show = toRgb(img, deep_copy);
  for (const auto& l : lines) {
    cv::Scalar line_color = fix_color ? color : cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
    cv::line(img_show, cv::Point(l(0), l(1)), cv::Point(l(2), l(3)), line_color, thickness);
  }
  return img_show;
}

cv::Mat drawLKStereo(const cv::Mat& img1, const cv::Mat& img2, const std::vector<cv::Point2f>& pts1,
                     const std::vector<cv::Point2f>& pts2, const std::vector<uchar>& inliers) {
  cv::Mat img_show(std::max(img1.rows, img2.rows), img1.cols + img2.cols, CV_8UC3);
  cv::Rect roi1(0, 0, img1.cols, img1.rows);
  cv::Rect roi2(img1.cols, 0, img2.cols, img2.rows);
  toRgb(img1, img_show(roi1));
  toRgb(img2, img_show(roi2));
  cv::Point2f dt(img1.cols, 0);
  for (size_t i = 0; i < pts1.size(); i++) {
    const auto& p1 = pts1[i];
    const auto& p2 = pts2[i] + dt;
    if (inliers.empty() || inliers[i] > 0) {
      cv::circle(img_show, p1, 2, COLOR_TRACK_PT, -1);
      cv::circle(img_show, p2, 2, COLOR_TRACK_PT, -1);
      cv::line(img_show, p1, p2, COLOR_TRACK_LINE, 1);
    } else {
      cv::circle(img_show, p1, 2, COLOR_LOST_PT, -1);
    }
  }
  return img_show;
}

cv::Mat drawMatches(const cv::Mat& img1, const cv::Mat& img2, const std::vector<cv::Point2f>& pts1,
                    const std::vector<cv::Point2f>& pts2, const std::vector<uchar>& inliers, int pt_radius,
                    int pt_thickness, int line_thickness) {
  static const std::vector<cv::Scalar> colors = randColors(100);
  cv::Mat img_show(std::max(img1.rows, img2.rows), img1.cols + img2.cols, CV_8UC3);
  cv::Rect roi1(0, 0, img1.cols, img1.rows);
  cv::Rect roi2(img1.cols, 0, img2.cols, img2.rows);
  toRgb(img1, img_show(roi1));
  toRgb(img2, img_show(roi2));
  cv::Point2f dt(img1.cols, 0);
  for (size_t i = 0; i < pts1.size(); i++) {
    const auto& p1 = pts1[i];
    const auto& p2 = pts2[i] + dt;
    const auto& color = colors[i % colors.size()];
    cv::circle(img_show, p1, pt_radius, color, pt_thickness);
    cv::circle(img_show, p2, pt_radius, color, pt_thickness);
    if (inliers.empty() || inliers[i] > 0) {
      cv::line(img_show, p1, p2, color, line_thickness);
    } else {
      cv::line(img_show, p1, p2, COLOR_OUTLIER_LINE, 1);
    }
  }
  return img_show;
}

cv::Mat drawLKMono(const cv::Mat& img, const std::vector<cv::Point2f>& prev_pts,
                   const std::vector<cv::Point2f>& cur_pts, const std::vector<unsigned char>& inliers, bool deep_copy) {
  cv::Mat img_show = toRgb(img, deep_copy);
  if (inliers.empty()) {
    for (size_t i = 0; i < prev_pts.size(); i++) {
      cv::circle(img_show, cur_pts[i], 2, COLOR_TRACK_PT, -1);
      cv::line(img_show, prev_pts[i], cur_pts[i], COLOR_TRACK_LINE, 1);
    }
  } else {
    for (size_t i = 0; i < prev_pts.size(); i++) {
      if (inliers[i]) {
        cv::circle(img_show, cur_pts[i], 2, COLOR_TRACK_PT, -1);
        cv::line(img_show, prev_pts[i], cur_pts[i], COLOR_TRACK_LINE, 1);
      } else {
        cv::circle(img_show, cur_pts[i], 2, COLOR_LOST_PT, -1);
      }
    }
  }
  return img_show;
}

cv::Mat drawRectify(const cv::Mat& imgl, const cv::Mat& imgr, int step) {
  static std::vector<cv::Scalar> colors = randColors(100);
  cv::Mat img_show;
  cv::hconcat(imgl, imgr, img_show);
  if (img_show.channels() == 1) cv::cvtColor(img_show, img_show, cv::COLOR_GRAY2BGR);
  for (int i = step, j = 0; i < img_show.rows; i += step, j++) {
    cv::line(img_show, cv::Point(0, i), cv::Point(img_show.cols, i), colors[j % colors.size()], 1);
  }
  return img_show;
}

cv::Mat drawCoordinate(const cv::Mat& img, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K, const cv::Mat& D,
                       float len, bool deep_copy) {
  cv::Mat img_show = toRgb(img, deep_copy);
  std::vector<cv::Point3f> draw_pts = {{0, 0, 0}, {len, 0, 0}, {0, len, 0}, {0, 0, len}};
  std::vector<cv::Point2f> draw_corners;
  cv::projectPoints(draw_pts, rvec, tvec, K, D, draw_corners);
  cv::line(img_show, draw_corners[0], draw_corners[1], cv::Scalar(255, 0, 0), 2);
  cv::line(img_show, draw_corners[0], draw_corners[2], cv::Scalar(0, 255, 0), 2);
  cv::line(img_show, draw_corners[0], draw_corners[3], cv::Scalar(0, 0, 255), 2);
  return img_show;
}

cv::Mat drawCube(const cv::Mat& img, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K, const cv::Mat& D,
                 const cv::Point3f& center, float cube_size, const cv::Scalar& color, int thickness, bool deep_copy) {
  cv::Mat img_show = toRgb(img, deep_copy);
  float h = cube_size * 0.5f;
  std::vector<cv::Point3f> draw_pts = {
      cv::Point3f(-h, -h, -h), cv::Point3f(-h, h, -h), cv::Point3f(h, h, -h), cv::Point3f(h, -h, -h),
      cv::Point3f(-h, -h, h),  cv::Point3f(-h, h, h),  cv::Point3f(h, h, h),  cv::Point3f(h, -h, h),
  };
  for (auto& p : draw_pts) p += center;
  std::vector<cv::Point2f> draw_corners;
  cv::projectPoints(draw_pts, rvec, tvec, K, D, draw_corners);
  for (int j = 0; j < 4; j++) {
    cv::line(img_show, draw_corners[j], draw_corners[(j + 1) % 4], color, thickness);
    cv::line(img_show, draw_corners[j + 4], draw_corners[(j + 1) % 4 + 4], color, thickness);
    cv::line(img_show, draw_corners[j], draw_corners[j + 4], color, thickness);
  }
  return img_show;
}

cv::Mat drawCubes(const cv::Mat& img, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K, const cv::Mat& D,
                  const std::vector<cv::Point3f>& centers, float cube_size, const std::vector<cv::Scalar>& colors,
                  int thickness, bool deep_copy) {
  std::vector<cv::Scalar> new_colors;
  if (colors.empty()) {
    static auto color_table = randColors(11);
    for (size_t i = 0, j = 0; i < centers.size(); i++, j++) {
      new_colors.push_back(color_table[j++]);
      if (j >= color_table.size()) j = 0;
    }
  } else if (colors.size() < centers.size()) {
    new_colors = colors;
    for (size_t i = colors.size(); i < centers.size(); i++) new_colors.push_back(colors.back());
  }
  const auto& used_colors = new_colors.empty() ? colors : new_colors;
  cv::Mat img_show = toRgb(img, deep_copy);
  float h = cube_size * 0.5f;
  std::vector<cv::Point3f> cube_corners = {
      cv::Point3f(-h, -h, -h), cv::Point3f(-h, h, -h), cv::Point3f(h, h, -h), cv::Point3f(h, -h, -h),
      cv::Point3f(-h, -h, h),  cv::Point3f(-h, h, h),  cv::Point3f(h, h, h),  cv::Point3f(h, -h, h),
  };
  std::vector<cv::Point3f> draw_pts;
  for (const auto& c : centers)
    for (const auto& p : cube_corners) draw_pts.push_back(c + p);
  std::vector<cv::Point2f> draw_corners;
  cv::projectPoints(draw_pts, rvec, tvec, K, D, draw_corners);
  for (size_t i = 0; i < draw_corners.size(); i += 8) {
    const auto& color = used_colors[i / 8];
    for (int j = 0; j < 4; j++) {
      cv::line(img_show, draw_corners[i + j], draw_corners[i + (j + 1) % 4], color, thickness);
      cv::line(img_show, draw_corners[i + j + 4], draw_corners[i + (j + 1) % 4 + 4], color, thickness);
      cv::line(img_show, draw_corners[i + j], draw_corners[i + j + 4], color, thickness);
    }
  }
  return img_show;
}

cv::Mat drawReprojectionError(const cv::Mat& img, const std::vector<cv::Point3f>& pts3d,
                              const std::vector<cv::Point2f>& pts2d, const cv::Mat& rvec, const cv::Mat& tvec,
                              const cv::Mat& K, const cv::Mat& D, const cv::Scalar& color, int thickness,
                              bool deep_copy) {
  cv::Mat img_show = toRgb(img, deep_copy);
  std::vector<cv::Point2f> draw_corners;
  cv::projectPoints(pts3d, rvec, tvec, K, D, draw_corners);
  for (size_t i = 0; i < draw_corners.size(); i++) cv::line(img_show, draw_corners[i], pts2d[i], color, thickness);
  return img_show;
}

cv::Mat drawGroundArrows(const cv::Mat& img, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K,
                         const cv::Mat& D, float max_draw_range, float draw_gap, float arrow_size, const cv::Mat& mask,
                         bool deep_copy) {
  const double ground_z = 0.0;
  static auto color_table = randColors(23);
  cv::Mat img_show = toRgb(img, deep_copy);
  std::vector<cv::Point3f> arrow_contour = {cv::Point3f(0, 0, 0), cv::Point3f(-arrow_size * 0.3, -arrow_size * 0.3, 0),
                                            cv::Point3f(arrow_size * 0.4, 0, 0),
                                            cv::Point3f(-arrow_size * 0.3, arrow_size * 0.3, 0)};
  std::vector<cv::Point3f> centers = meshGrid3D<float>(-max_draw_range, max_draw_range, -max_draw_range, max_draw_range,
                                                       ground_z, ground_z + VS_EPS, draw_gap, draw_gap, 1);

  cv::Mat R;
  cv::Rodrigues(rvec, R);
  R.convertTo(R, CV_32FC1);
  cv::Point3f R_row_3(R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2));
  float tz = tvec.type() == CV_32FC1 ? reinterpret_cast<float*>(tvec.data)[2] : reinterpret_cast<double*>(tvec.data)[2];

  for (size_t i = 0; i < centers.size(); i++) {
    const auto& center = centers[i];
    float zc = R_row_3.dot(center) + tz;
    if (zc < VS_EPS) continue;

    std::vector<cv::Point3f> contour3d;
    contour3d.reserve(arrow_contour.size());
    for (const auto& p : arrow_contour) contour3d.push_back(center + p);
    std::vector<cv::Point2f> contour2d;
    cv::projectPoints(contour3d, rvec, tvec, K, D, contour2d);
    bool valid = true;
    for (const auto& p : contour2d) {
      if (!inside(p, img.size()) || (!mask.empty() && (!inside(p, mask.size()) || mask.at<uchar>(p) == 0))) {
        valid = false;
        break;
      }
    }
    if (valid) {
      std::vector<cv::Point> tmp;
      vecAssign(contour2d, tmp);
      cv::fillPoly(img_show, std::vector<std::vector<cv::Point>>(1, tmp), color_table[i % color_table.size()]);
    }
  }
  return img_show;
}

cv::Mat drawSparseDepth(const cv::Mat& img, const cv::Mat& depth, float max_depth) {
  cv::Mat img_show = toRgb(img);
  cv::Mat depth_color = drawMatf(depth, 255.0f / (max_depth + VS_EPS));
  cv::Mat depth_mask;
  cv::threshold(depth, depth_mask, VS_EPS, 255, cv::THRESH_BINARY);
  depth_mask.convertTo(depth_mask, CV_8U);
  depth_color.copyTo(img_show, depth_mask);
  return img_show;
}

void colorBar(double rate, uchar& R, uchar& G, uchar& B) {
  const static uchar tr[] = {255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0};
  const static uchar tg[] = {0, 125, 255, 255, 255, 0, 0, 125, 255, 255, 255, 0};
  const static uchar tb[] = {0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 255, 255};
  const static int N = sizeof(tb) / sizeof(tb[0]);
  if (rate <= 0) {
    R = tr[0];
    G = tg[0];
    B = tb[0];
  } else if (rate >= 1) {
    R = tr[N - 1];
    G = tg[N - 1];
    B = tb[N - 1];
  } else {
    float k = rate * N;
    int i = k;
    k -= i;
    R = tr[i] * (1 - k) + tr[i + 1] * k;
    G = tg[i] * (1 - k) + tg[i + 1] * k;
    B = tb[i] * (1 - k) + tb[i + 1] * k;
  }
}

cv::Scalar colorBar(double rate) {
  uchar r, g, b;
  colorBar(rate, r, g, b);
  return cv::Scalar(r, g, b);
}

std::vector<cv::Scalar> randColors(int cnt) {
  std::vector<cv::Scalar> colors;
  for (int i = 0; i < cnt; i++) colors.push_back(cv::Scalar(randi(255), randi(255), randi(255)));
  return colors;
}

cv::Mat hstack(const std::vector<cv::Mat>& imgs, const cv::Scalar& border_color, const cv::Scalar& bg_color) {
  int cols = 0;
  int rows = 0;
  int img_type = -1;
  for (const auto& img : imgs) {
    rows = std::max(rows, img.rows);
    cols += img.cols;
    // img type must be same
    if (img_type < 0)
      img_type = img.type();
    else if (img_type != img.type())
      return cv::Mat();
  }
  if (rows <= 0 || cols <= 0) return cv::Mat();
  cv::Mat img_all(rows, cols, img_type, bg_color);
  int col_idx = 0;
  for (const auto& img : imgs) {
    if (img.empty()) continue;
    cv::Rect roi(col_idx, 0, img.cols, img.rows);
    img.copyTo(img_all(roi));
    cv::rectangle(img_all, roi, border_color);
    col_idx += img.cols;
  }
  return img_all;
}

cv::Mat vstack(const std::vector<cv::Mat>& imgs, const cv::Scalar& border_color, const cv::Scalar& bg_color) {
  int cols = 0;
  int rows = 0;
  int img_type = -1;
  for (const auto& img : imgs) {
    rows += img.rows;
    cols = std::max(cols, img.cols);
    // img type must be same
    if (img_type < 0)
      img_type = img.type();
    else if (img_type != img.type())
      return cv::Mat();
  }
  if (rows <= 0 || cols <= 0) return cv::Mat();
  cv::Mat img_all(rows, cols, img_type, bg_color);
  int row_idx = 0;
  for (const auto& img : imgs) {
    if (img.empty()) continue;
    cv::Rect roi(0, row_idx, img.cols, img.rows);
    img.copyTo(img_all(roi));
    cv::rectangle(img_all, roi, border_color);
    row_idx += img.rows;
  }
  return img_all;
}

cv::Mat gridStack(const std::vector<cv::Mat>& imgs, int grid_rows, int grid_cols, const cv::Size& force_resize,
                  const cv::Scalar& bg_color) {
  if (imgs.empty()) return cv::Mat();
  if (grid_rows <= 0 && grid_cols <= 0) return cv::Mat();
  cv::Size img_size = (force_resize.width == 0 || force_resize.height == 0) ? imgs[0].size() : force_resize;
  int N = imgs.size();
  if (grid_rows <= 0) grid_rows = std::ceil(static_cast<float>(N) / grid_cols);
  if (grid_cols <= 0) grid_cols = std::ceil(static_cast<float>(N) / grid_rows);
  grid_rows = std::min(grid_rows, static_cast<int>(std::ceil(static_cast<float>(N) / grid_cols)));
  if (N < grid_cols) grid_cols = N;
  cv::Mat img_draw(grid_rows * img_size.height, grid_cols * img_size.width, imgs[0].type(), bg_color);
  N = std::min(N, grid_rows * grid_cols);
  for (int i = 0; i < N; i++) {
    int x0 = (i % grid_cols) * img_size.width;
    int y0 = (i / grid_cols) * img_size.height;
    cv::Rect roi(x0, y0, img_size.width, img_size.height);
    const auto& img = imgs[i];
    if (img.size() == img_size) {
      img.copyTo(img_draw(roi));
    } else {
      cv::resize(img, img_draw(roi), img_size);
    }
  }
  return img_draw;
}

TimelinePloter::TimelinePloter(double max_store_sec) : max_store_sec_(max_store_sec), latest_ts_(0) {}

void TimelinePloter::arrived(int sensor_id, double ts) {
  auto it = timelime_map_.find(sensor_id);
  if (it == timelime_map_.end()) {
    Timeline timeline;
    timeline.arrived(ts);
    timelime_map_[sensor_id] = timeline;
  } else {
    it->second.arrived(ts);
  }
  if (latest_ts_ < ts) latest_ts_ = ts;
}

void TimelinePloter::used(int sensor_id, double ts) {
  auto it = timelime_map_.find(sensor_id);
  if (it == timelime_map_.end()) {
    Timeline timeline;
    timeline.used(ts);
    timelime_map_[sensor_id] = timeline;
  } else {
    it->second.used(ts);
  }
  if (latest_ts_ < ts) latest_ts_ = ts;
}

cv::Mat TimelinePloter::plot(const cv::Size& img_size) {
  if (timelime_map_.empty()) return cv::Mat(img_size, CV_8UC3, cv::Scalar(255, 255, 255));

  double start_ts = latest_ts_ - max_store_sec_;
  int start_col = img_size.width * 0.03;
  int stop_col = img_size.width * 0.97;
  double step = (stop_col - start_col) / max_store_sec_;
  int subimg_rows = img_size.height / static_cast<int>(timelime_map_.size());

  std::vector<cv::Mat> imgs;
  for (auto& it : timelime_map_) {
    int id = it.first;
    it.second.removeBefore(start_ts);
    cv::Mat img(subimg_rows, img_size.width, CV_8UC3, cv::Scalar(255, 255, 255));
    int row = img.rows * 0.8;
    int top = img.rows * 0.2;
    img.row(row).colRange(start_col, stop_col).setTo(cv::Scalar(0, 0, 0));
    for (const auto& t : it.second.arrived_ts_list.list()) {
      int col = (t - start_ts) * step + start_col;
      img.col(col).rowRange(top, row).setTo(cv::Scalar(128, 128, 128));
    }
    for (const auto& t : it.second.used_ts_list.list()) {
      int col = (t - start_ts) * step + start_col;
      img.col(col).rowRange(top, row).setTo(cv::Scalar(0, 0, 255));
    }
    cv::putText(img, vs::num2str(id), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 0), 1);
    imgs.push_back(img);
  }
  cv::Mat img_show = vs::vstack(imgs);
  cv::putText(img_show, vs::num2str(start_ts), cv::Point(start_col, img_show.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.4,
              cv::Scalar(255, 150, 20), 1);
  cv::putText(img_show, vs::num2str(latest_ts_), cv::Point(stop_col - 20, img_show.rows - 20), cv::FONT_HERSHEY_SIMPLEX,
              0.4, cv::Scalar(255, 150, 20), 1);
  return img_show;
}

} /* namespace vs */