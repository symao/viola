#include "vs_plot.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "vs_basic.h"

namespace vs {
static void plotCross(cv::Mat& img, const cv::Point& p, const cv::Scalar& color, int thickness, int radius = 4) {
  std::vector<cv::Point> pts = {{-radius, -radius}, {radius, radius}, {radius, -radius}, {-radius, radius}};
  cv::line(img, p + pts[0], p + pts[1], color, thickness);
  cv::line(img, p + pts[2], p + pts[3], color, thickness);
}

static void plotStar(cv::Mat& img, const cv::Point& p, const cv::Scalar& color, int thickness, int radius = 5) {
  std::vector<cv::Point> pts = {cv::Point(radius, 0),
                                cv::Point(-radius, 0),
                                cv::Point(radius * VS_COS_DEG60, -radius * VS_SIN_DEG60),
                                cv::Point(-radius * VS_COS_DEG60, radius * VS_SIN_DEG60),
                                cv::Point(radius * VS_COS_DEG60, radius * VS_SIN_DEG60),
                                cv::Point(-radius * VS_COS_DEG60, -radius * VS_SIN_DEG60)};
  cv::line(img, p + pts[0], p + pts[1], color, thickness);
  cv::line(img, p + pts[2], p + pts[3], color, thickness);
  cv::line(img, p + pts[4], p + pts[5], color, thickness);
}

static void plotDashLine(cv::Mat& img, const cv::Point& p1, const cv::Point& p2, const cv::Scalar& color,
                         int thickness) {
  static bool real_part_first = false;
  static const float real = 5;
  static const float dash = 5 + thickness;
  static const float step = dash + real;
  cv::Point diff = p2 - p1;
  float len = hypotf(diff.x, diff.y);
  if (len <= 0) {
    if (real_part_first) cv::line(img, p1, p2, color, thickness);
  } else {
    float vx = diff.x / len;
    float vy = diff.y / len;
    float l = real_part_first ? 0 : dash;
    for (; l < len; l += step) {
      float l2 = std::min(len, l + real);
      if (l < l2)
        cv::line(img, cv::Point(p1.x + vx * l, p1.y + vy * l), cv::Point(p1.x + vx * l2, p1.y + vy * l2), color,
                 thickness);
    }
  }
  real_part_first = !real_part_first;
}

Plot::LineStyle::LineStyle(const std::string& str) : color(255, 0, 0), line_type(STYLE_LINE) {
  if (str.find('r') != str.npos) {
    color = cv::Scalar(0, 0, 255);
  } else if (str.find('g') != str.npos) {
    color = cv::Scalar(0, 255, 0);
  } else if (str.find('b') != str.npos) {
    color = cv::Scalar(255, 0, 0);
  } else if (str.find('y') != str.npos) {
    color = cv::Scalar(0, 255, 255);
  } else if (str.find('c') != str.npos) {
    color = cv::Scalar(255, 255, 0);
  } else if (str.find('p') != str.npos) {
    color = cv::Scalar(255, 0, 255);
  } else if (str.find('k') != str.npos) {
    color = cv::Scalar(10, 10, 10);
  }
  if (str.find("--") != str.npos) {
    line_type = STYLE_DASHED_LINE;
  } else if (str.find('-') != str.npos) {
    line_type = STYLE_LINE;
  } else if (str.find('.') != str.npos) {
    line_type = STYLE_DOT;
  } else if (str.find('*') != str.npos) {
    line_type = STYLE_STAR;
  } else if (str.find('o') != str.npos) {
    line_type = STYLE_CIRCLE;
  } else if (str.find('x') != str.npos) {
    line_type = STYLE_CROSS;
  }
}

Plot::Plot() { reset(); }

void Plot::reset() {
  m_axis_list.clear();
  m_grid_size = cv::Size(1, 1);
  m_cur_axis_id = 0;
}

void Plot::subplot(int val) {
  if (100 < val && val < 1000) {
    m_grid_size.height = val / 100;
    m_grid_size.width = val % 100 / 10;
    m_cur_axis_id = val % 10 - 1;
  } else {
    printf("[ERROR]Plot: wrong subplot input:%d\n", val);
  }
}

void Plot::subplot(int row, int col, int idx) {
  m_grid_size.height = row;
  m_grid_size.width = col;
  if (idx < 0) idx = 0;
  m_cur_axis_id = idx - 1;
}

cv::Mat Plot::render(const cv::Size& render_size) {
  cv::Mat draw_img(render_size.area() > 0 ? render_size : m_render_size, CV_8UC3);
  int grid_cnt = m_grid_size.area();
  int grid_w = draw_img.cols / m_grid_size.width;
  int grid_h = draw_img.rows / m_grid_size.height;
  for (const auto& it : m_axis_list) {
    int idx = it.first;
    const auto& axis = it.second;
    if (idx < grid_cnt) {
      int i = idx / m_grid_size.width;
      int j = idx % m_grid_size.width;
      cv::Mat axis_img = draw_img(cv::Rect(grid_w * j, grid_h * i, grid_w, grid_h));
      axis.draw(axis_img);
    }
  }
  reset();
  return draw_img;
}

void Plot::AxisPack::draw(cv::Mat& img) const {
  cv::Rect roi = axisRoi(img.size());
  drawBorder(img, roi);
  drawText(img);
  auto range = calcDrawRange();
  cv::Mat roi_img = img(roi);
  roi_img.setTo(bg_color);
  drawCurve(roi_img, range);
  drawTicks(img, roi, range);
  if (draw_legend) drawLegend(img);
}

void Plot::AxisPack::drawBorder(cv::Mat& img, const cv::Rect& roi) const {
  if (roi.x > 0) img.colRange(0, roi.x).setTo(border_color);
  if (roi.y > 0) img.rowRange(0, roi.y).setTo(border_color);
  auto br = roi.br();
  if (br.x < img.cols) img.colRange(br.x, img.cols).setTo(border_color);
  if (br.y < img.rows) img.rowRange(br.y, img.rows).setTo(border_color);
}

void Plot::AxisPack::drawCurve(cv::Mat& img, const RangeXY& range) const {
  double xmin = range.xmin;
  double xmax = range.xmax;
  double ymin = range.ymin;
  double ymax = range.ymax;
  if (xmin < xmax && ymin < ymax) {
    double rx = img.cols / (xmax - xmin);
    double ry = img.rows / (ymax - ymin);
    // draw axis
    int idx_x = -xmin * rx;
    int idx_y = img.rows + ymin * ry;
    if (inIntervalL(idx_x, 0, img.cols))
      cv::line(img, cv::Point(idx_x, 0), cv::Point(idx_x, img.rows), cv::Scalar(0, 0, 180));
    if (inIntervalL(idx_y, 0, img.rows))
      cv::line(img, cv::Point(0, idx_y), cv::Point(img.cols, idx_y), cv::Scalar(0, 0, 180));
    // draw curves
    for (const auto& pack : plot_packs) {
      std::vector<cv::Point> pts;
      pts.reserve(pack.x.size());
      for (size_t i = 0; i < pack.x.size(); i++)
        pts.emplace_back((pack.x[i] - xmin) * rx, img.rows - (pack.y[i] - ymin) * ry);
      switch (pack.style.line_type) {
        case STYLE_DOT:
          for (const auto& p : pts) cv::circle(img, p, pack.thickness, pack.style.color, -1);
          break;
        case STYLE_LINE:
          for (size_t i = 1; i < pts.size(); i++) cv::line(img, pts[i - 1], pts[i], pack.style.color, pack.thickness);
          break;
        case STYLE_DASHED_LINE:
          for (size_t i = 1; i < pts.size(); i++)
            plotDashLine(img, pts[i - 1], pts[i], pack.style.color, pack.thickness);
          break;
        case STYLE_STAR:
          for (const auto& p : pts) plotStar(img, p, pack.style.color, pack.thickness);
          break;
        case STYLE_CIRCLE:
          for (const auto& p : pts) cv::circle(img, p, 3, pack.style.color, pack.thickness);
          break;
        case STYLE_CROSS:
          for (const auto& p : pts) plotCross(img, p, pack.style.color, pack.thickness);
          break;
        default:
          break;
      }
    }
  }
}

static double calcTickStep(double range, double k = 1.6) {
  const double k10 = k * 10;
  double scale = 1.0;
  while (true) {
    if (range < k) {
      range *= 10;
      scale /= 10;
    } else if (range < k10) {
      return scale;
    } else {
      range /= 10.0;
      scale *= 10.0;
    }
  }
}

static std::vector<double> calcTicks(double min, double max, double k) {
  if (max <= min) return {};
  double step = calcTickStep(max - min, k);
  double x0 = static_cast<int>(min / step) * step;
  std::vector<double> ticks;
  for (double x = x0; x < max; x += step) ticks.push_back(x);
  return ticks;
}

void Plot::AxisPack::drawTicks(cv::Mat& img, const cv::Rect& roi, const RangeXY& range) const {
  double xmin = range.xmin;
  double xmax = range.xmax;
  double ymin = range.ymin;
  double ymax = range.ymax;
  if (!(xmin < xmax && ymin < ymax)) return;
  auto xticks = calcTicks(xmin, xmax, 1.3);
  double rx = roi.width / (xmax - xmin);
  for (auto x : xticks) {
    int idx_x = (x - xmin) * rx + roi.x;
    if (idx_x < roi.x || idx_x > roi.x + roi.width) continue;
    cv::line(img, cv::Point(idx_x, roi.y + roi.height), cv::Point(idx_x, roi.y + roi.height - 5), cv::Scalar(0), 1);
    cv::putText(img, vs::num2str(x), cv::Point(idx_x - 3, roi.y + roi.height + 15), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(0), 1);
  }
  auto yticks = calcTicks(ymin, ymax, 1.1);
  double ry = roi.height / (ymax - ymin);
  for (auto y : yticks) {
    int idx_y = -(y - ymin) * ry + roi.y + roi.height;
    if (idx_y < roi.y || idx_y > roi.y + roi.height) continue;
    cv::line(img, cv::Point(roi.x, idx_y), cv::Point(roi.x + 5, idx_y), cv::Scalar(0), 1);
    cv::putText(img, vs::num2str(y), cv::Point(roi.x - 25, idx_y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0), 1);
  }
}

void Plot::AxisPack::drawText(cv::Mat& img) const {
  const float text_scale = 0.6f;
  const cv::Scalar text_color(0, 0, 0);
  if (!title.empty())
    cv::putText(img, title, cv::Point(img.cols / 2 - 30, 15), cv::FONT_HERSHEY_SIMPLEX, text_scale, text_color);
  if (!xlabel.empty())
    cv::putText(img, xlabel, cv::Point(img.cols / 2 - 5, img.rows - 3), cv::FONT_HERSHEY_SIMPLEX, text_scale,
                text_color);
  if (!ylabel.empty())
    cv::putText(img, ylabel, cv::Point(2, img.rows / 2), cv::FONT_HERSHEY_SIMPLEX, text_scale, text_color);
}

void Plot::AxisPack::drawLegend(cv::Mat& img) const {
  const int w = 80;
  const int hstep = 20;
  const int h = hstep * plot_packs.size();
  cv::Rect roi(img.cols - w - 5, 5, w, h);
  if (roi.x < 0 || roi.y < 0) return;
  cv::Mat roi_img = img(roi);
  roi_img.setTo(cv::Scalar(240, 240, 240));
  cv::Scalar black(0, 0, 0);
  cv::rectangle(img, roi, black);
  for (size_t i = 0; i < plot_packs.size(); i++) {
    const auto& pack = plot_packs[i];
    int start_x = roi_img.cols * 0.3f;
    int end_x = roi_img.cols * 0.95f;
    int mid_y = hstep * (i + 0.5f);
    if (!pack.label.empty())
      cv::putText(roi_img, pack.label + ":", cv::Point(2, mid_y + 3), cv::FONT_HERSHEY_SIMPLEX, 0.35f, black);
    switch (pack.style.line_type) {
      case STYLE_DOT:
        for (int x = start_x; x < end_x; x += 10) cv::circle(roi_img, cv::Point(x, mid_y), 2, pack.style.color, -1);
        break;
      case STYLE_LINE:
        cv::line(roi_img, cv::Point(start_x, mid_y), cv::Point(end_x, mid_y), pack.style.color, 1);
        break;
      case STYLE_DASHED_LINE:
        plotDashLine(roi_img, cv::Point(start_x, mid_y), cv::Point(end_x, mid_y), pack.style.color, 1);
        break;
      case STYLE_CROSS:
        for (int x = start_x; x < end_x; x += 10) plotCross(roi_img, cv::Point(x, mid_y), pack.style.color, 1, 2);
        break;
      case STYLE_STAR:
        for (int x = start_x; x < end_x; x += 10) plotStar(roi_img, cv::Point(x, mid_y), pack.style.color, 1, 3);
        break;
      case STYLE_CIRCLE:
        for (int x = start_x; x < end_x; x += 10) cv::circle(roi_img, cv::Point(x, mid_y), 2, pack.style.color, 1);
        break;
      default:
        break;
    }
  }
}

Plot::AxisPack::RangeXY Plot::AxisPack::calcDrawRange() const {
  RangeXY range;
  double& xmin = range.xmin;
  double& xmax = range.xmax;
  double& ymin = range.ymin;
  double& ymax = range.ymax;
  xmin = ymin = DBL_MAX;
  xmax = ymax = -1e120;
  if (!(has_xmax && has_xmin && has_ymax && has_ymin)) {
    for (const auto& pack : plot_packs) {
      if (pack.x.empty() || pack.y.empty()) continue;
      double p_xmin, p_xmax, p_ymin, p_ymax;
      vecMinMax(pack.x, p_xmin, p_xmax);
      vecMinMax(pack.y, p_ymin, p_ymax);
      xmax = std::max(p_xmax, xmax);
      xmin = std::min(p_xmin, xmin);
      ymax = std::max(p_ymax, ymax);
      ymin = std::min(p_ymin, ymin);
    }
    if (1) {  // padding range
      float pad_x = (xmax - xmin) * 0.01f;
      float pad_y = (ymax - ymin) * 0.03f;
      if (pad_x <= 0) pad_x = 0.1f;
      if (pad_y <= 0) pad_y = 0.1f;
      xmax += pad_x;
      xmin -= pad_x;
      ymax += pad_y;
      ymin -= pad_y;
    }
  }
  if (has_xmax) xmax = set_xmax;
  if (has_xmin) xmin = set_xmin;
  if (has_ymax) ymax = set_ymax;
  if (has_ymin) ymin = set_ymin;
  return range;
}

cv::Rect Plot::AxisPack::axisRoi(const cv::Size& img_size) const {
  const int up = 25;
  const int down = img_size.height - 20;
  const int left = 30;
  const int right = img_size.width - 8;
  return cv::Rect(left, up, right - left, down - up);
}

}  // namespace vs