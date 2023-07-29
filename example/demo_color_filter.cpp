#include <viola/vs_color_filter.h>
#include <viola/vs_debug_draw.h>

int main() {
  cv::Mat img = cv::imread(PROJECT_DIR "/data/color_filter.jpg");

  std::map<std::string, vs::ColorModelList> color_table = {
      {"yellow", {vs::ColorModel::yellow()}},
      {"red", {vs::ColorModel::red()}},
      {"green", {vs::ColorModel::green()}},
      {"blue", {vs::ColorModel::blue()}},
      {"black", {vs::ColorModel::black()}},
      {"white", {vs::ColorModel::white()}},
      {"yellow+red", {vs::ColorModel::yellow(), vs::ColorModel::red()}},
      {"blue+red+black", {vs::ColorModel::blue(), vs::ColorModel::red(), vs::ColorModel::black()}},
  };

  for (const auto& it : color_table) {
    const auto& name = it.first;
    const auto& model = it.second;
    cv::Mat mask;
    vs::colorFilter(img, mask, model);
    cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
    cv::Mat img_show = vs::hstack({img, mask}, cv::Scalar(0, 50, 128));
    cv::putText(img_show, name, cv::Point(2, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    cv::imshow("img_show", img_show);
    cv::waitKey();
  }
}