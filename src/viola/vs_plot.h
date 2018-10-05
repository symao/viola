/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2023-04-19 14:07
 * @details s simple 2D plot lib implemented by OpenCV, Similar usage to Matlab or matplotlib.pyplot in python.
 */
#pragma once
#include <map>
#include <opencv2/highgui.hpp>

namespace vs {

/** @brief plot by opencv, same use as matlab plot or pyplot
 *  @code
    Plot plt;
    std::vector<double> x(10);
    std::vector<double> y(10);
    for(int i=0;i<10;i++){
      x[i] = 3*i;
      y[i] = sin(i)*5+6;
    }
    plt.plot(x,y);
    cv::imshow("img", plt.render());
    cv::waitKey();
 * @endcode
 */
class Plot {
 public:
  /** @enum Plot line type */
  enum LineType {
    STYLE_LINE = 0,         ///< solid line, "-"
    STYLE_DOT = 1,          ///< dot, "."
    STYLE_DASHED_LINE = 2,  ///< dashed line, "--"
    STYLE_STAR = 3,         ///< star, "*"
    STYLE_CIRCLE = 4,       ///< circle, "o"
    STYLE_CROSS = 5,        ///< cross, "x"
  };

  /** @brief line plot style, which contains plot color and line type */
  struct LineStyle {
    cv::Scalar color;    ///< line plot color in B-G-R order
    LineType line_type;  ///< plot line type, @see LineType

    /** @brief default constructor */
    LineStyle(const cv::Scalar& _color = cv::Scalar(255, 0, 0), LineType _line_type = STYLE_LINE)
        : color(_color), line_type(_line_type) {}

    /** @brief Construct from string. eg: ".b", "g--"
     * @param[in]str: combination of color string and line type string
     *                'r': red
     *                'g': green
     *                'b': blue
     *                'y': yellow
     *                'c': cyan
     *                'p': purple
     *                'k': black
     *                '-': line
     *                '.': dot
     *                '--': dashed line
     *                '*': star
     *                'o': circle
     *                'x': cross
     */
    LineStyle(const std::string& str);
    LineStyle(const char* str) : LineStyle(std::string(str)) {}
  };

  Plot();

  /** @brief Add an Axis to the current figure or retrieve an existing Axis, eg: subplot(221)
   * @param[in]val: same use as matlab, 100<val<999, eg: subplot(221) = subplot(2,2,1)
   */
  void subplot(int val);

  /** @brief Add an Axis to the current figure or retrieve an existing Axis
   * @param[in]row: sub plot axis row
   * @param[in]col: sub plot axis col
   * @param[in]idx: current axis plot index, 1<=idx<=(row*col)
   */
  void subplot(int row, int col, int idx);

  /** @brief plot data from memory
   * @param[in]x: x data pointer
   * @param[in]y: y data pointer
   * @param[in]N: count of data
   * @param[in]style: plot line style
   * @param[in]thickness: plot line thickness
   * @param[in]label: line label, used in legend()
   */
  template <typename Tx, typename Ty>
  void plot(const Tx* x, const Ty* y, int N, const LineStyle& style = LineStyle(), int thickness = 1,
            const std::string& label = "") {
    PlotPack pack;
    pack.x.reserve(N);
    pack.y.reserve(N);
    for (int i = 0; i < N; i++) {
      pack.x.push_back(x[i]);
      pack.y.push_back(y[i]);
    }
    pack.style = style;
    pack.thickness = thickness;
    pack.label = label;
    curAxis().plot_packs.push_back(pack);
  }

  /** @brief plot data from memory with only y data, use index as x in default
   * @param[in]y: y data pointer
   * @param[in]N: count of data
   * @param[in]style: plot line style
   * @param[in]thickness: plot line thickness
   * @param[in]label: line label, used in legend()
   */
  template <typename T>
  void plot(const T* y, int N, const LineStyle& style = LineStyle(), int thickness = 1, const std::string& label = "") {
    std::vector<double> xs(N);
    for (int i = 0; i < N; i++) xs[i] = i;
    plot(&xs[0], y, N, style, thickness, label);
  }

  /** @brief plot data from vector with only y data, use index as x in default
   * @param[in]y: y data
   * @param[in]style: plot line style
   * @param[in]thickness: plot line thickness
   * @param[in]label: line label, used in legend()
   */
  template <typename T>
  void plot(const std::vector<T>& y, const LineStyle& style = LineStyle(), int thickness = 1,
            const std::string& label = "") {
    return plot(&y[0], y.size(), style, thickness, label);
  }

  /** @brief plot data from vector
   * @param[in]x: x data
   * @param[in]y: y data
   * @param[in]style: plot line style
   * @param[in]thickness: plot line thickness
   * @param[in]label: line label, used in legend()
   */
  template <typename Tx, typename Ty>
  void plot(const std::vector<Tx>& x, const std::vector<Ty>& y, const LineStyle& style = LineStyle(), int thickness = 1,
            const std::string& label = "") {
    return plot(&x[0], &y[0], x.size(), style, thickness, label);
  }

  /** @brief set render image size */
  void setRenderSize(const cv::Size& img_size) {
    if (img_size.area() > 0) m_render_size = img_size;
  }

  /** @brief set max x range */
  void setMaxX(double max_x) {
    auto& a = curAxis();
    a.set_xmax = max_x;
    a.has_xmax = true;
  }

  /** @brief set max y range */
  void setMaxY(double max_y) {
    auto& a = curAxis();
    a.set_ymax = max_y;
    a.has_ymax = true;
  }

  /** @brief set min x range */
  void setMinX(double min_x) {
    auto& a = curAxis();
    a.set_xmin = min_x;
    a.has_xmin = true;
  }

  /** @brief set min y range */
  void setMinY(double min_y) {
    auto& a = curAxis();
    a.set_ymin = min_y;
    a.has_ymin = true;
  }

  /** @brief set min/max x range */
  void setXRange(double xmin, double xmax) {
    auto& a = curAxis();
    a.set_xmin = xmin;
    a.set_xmax = xmax;
    a.has_xmin = a.has_xmax = true;
  }

  /** @brief set min/max y range */
  void setYRange(double ymin, double ymax) {
    auto& a = curAxis();
    a.set_ymin = ymin;
    a.set_ymax = ymax;
    a.has_ymin = a.has_ymax = true;
  }

  /** @brief set min/max x,y range */
  void setRange(double xmin, double xmax, double ymin, double ymax) {
    auto& a = curAxis();
    a.set_xmin = xmin;
    a.set_xmax = xmax;
    a.set_ymin = ymin;
    a.set_ymax = ymax;
    a.has_xmin = a.has_xmax = a.has_ymin = a.has_ymax = true;
  }

  /** @brief set x axis label */
  void xlabel(const std::string& label) { curAxis().xlabel = label; }

  /** @brief set y axis label */
  void ylabel(const std::string& label) { curAxis().ylabel = label; }

  /** @brief set title */
  void title(const std::string& title) { curAxis().title = title; }

  /** @brief set background color, default white */
  void setBackgroundColor(const cv::Scalar& color) { curAxis().bg_color = color; }

  /** @brief enable plot legend */
  void legend() {
    for (auto& it : m_axis_list) it.second.draw_legend = true;
  }

  /** @brief reset plot, clear all plot datas */
  void reset();

  /** @brief render all plot data into image and return */
  cv::Mat render(const cv::Size& render_size = cv::Size());

  cv::Mat show(const cv::Size& render_size = cv::Size()) { return render(render_size); }

 private:
  /** @brief data for plotting a curve */
  struct PlotPack {
    std::vector<double> x;
    std::vector<double> y;
    LineStyle style;
    int thickness;
    std::string label;
  };

  /** @brief data to plot a axis in subimage, which contains multiple curves */
  struct AxisPack {
    std::string title, xlabel, ylabel;
    bool draw_legend = false;  ///< whether plot lengen
    // whether range set and set range data
    bool has_xmin = false, has_ymin = false, has_xmax = false, has_ymax = false;
    double set_xmin, set_ymin, set_xmax, set_ymax;
    std::vector<PlotPack> plot_packs;                     ///< curve datas
    cv::Scalar bg_color = cv::Scalar(255, 230, 230);      ///< background color
    cv::Scalar border_color = cv::Scalar(128, 128, 128);  ///< border color

    /** @brief draw axis which contains: text and curve datas. */
    void draw(cv::Mat& img) const;

   private:
    struct RangeXY {
      double xmin, ymin, xmax, ymax;
    };

    /** @brief draw titles and xy labels */
    void drawText(cv::Mat& img) const;

    /** @brief draw all curves */
    void drawCurve(cv::Mat& img, const RangeXY& range) const;

    /** @brief draw legend */
    void drawLegend(cv::Mat& img) const;

    /** @brief draw border */
    void drawBorder(cv::Mat& img, const cv::Rect& roi) const;

    /** @brief draw x,y ticks */
    void drawTicks(cv::Mat& img, const cv::Rect& roi, const RangeXY& range) const;

    /** @brief calculate draw range for input curve datas */
    RangeXY calcDrawRange() const;

    /** @brief calculate roi for input image size */
    cv::Rect axisRoi(const cv::Size& img_size) const;
  };

  cv::Size m_render_size = cv::Size(640, 480);  ///< render image size
  cv::Size m_grid_size;                         ///< subplot grid size
  int m_cur_axis_id;                            ///< current plot axis id
  std::map<int, AxisPack> m_axis_list;          ///< axis package list, each subplot is a AxisPack

  /** @brief return current plot axis */
  AxisPack& curAxis() { return m_axis_list[m_cur_axis_id]; }
};

}  // namespace vs