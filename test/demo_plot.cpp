#include <opencv2/opencv.hpp>
#include <viola/vs_plot.h>
#include <viola/vs_basic.h>
#include <viola/vs_video_saver.h>

void plotCurveFunc(const std::function<double(double)>& func, vs::Plot& plt, double xmax = 10,
                   const vs::Plot::LineStyle& style = vs::Plot::LineStyle(), int thickness = 1,
                   const std::string& label = "") {
  auto xs = vs::vecLinspace<double>(0, xmax, 100);
  std::vector<double> ys;
  ys.reserve(xs.size());
  for (double x : xs) ys.push_back(func(x));
  plt.plot(xs, ys, style, thickness, label);
}

void testStaticPlot() {
  vs::Plot plt;
  plotCurveFunc([](double x) { return x; }, plt, 5, "r", 2, "TrivialLoss");
  plotCurveFunc([](double x) { return x <= 1 ? x : 2 * sqrt(x) - 1; }, plt, 5, "g", 2, "HuberLoss");
  plotCurveFunc([](double x) { return 2 * (sqrt(x + 1) - 1); }, plt, 5, "b", 2, "SoftLOneLoss");
  plotCurveFunc([](double x) { return log(1 + x); }, plt, 5, "c", 2, "CauchyLoss");
  plotCurveFunc([](double x) { return atan(x); }, plt, 5, "y", 2, "AtanLoss");
  plt.legend();
  cv::imshow("img", plt.show());
  cv::waitKey();
}

void testDynamicPlot() {
  vs::Plot plt;
  plt.setRenderSize(cv::Size(320, 240));
  std::vector<double> data_x;
  std::vector<double> data_y1;
  std::vector<double> data_y2;
  std::vector<double> data_y3;
  for (int i = 0; i < 150; i++) {
    data_x.push_back(i * 0.1);
    data_y1.push_back(std::sin(data_x.back()));
    data_y2.push_back(std::cos(data_x.back()));
    data_y3.push_back(std::sin(2 * data_x.back()));
    plt.plot(data_x, data_y1, "b", 2, "sin(x)");
    plt.plot(data_x, data_y2, "*g", 2, "cos(x)");
    plt.plot(data_x, data_y3, ".r", 2, "sin(2x)");
    plt.title("title");
    plt.xlabel("x");
    plt.ylabel("y");
    plt.legend();
    auto img = plt.show();
    cv::imshow("img", img);
    cv::waitKey(10);
#if 0
    static vs::VideoRecorder recorder("plot2d.mp4");
    recorder.write(img);
#endif
  }
  cv::waitKey();
}

int main() {
  testStaticPlot();
  testDynamicPlot();
}