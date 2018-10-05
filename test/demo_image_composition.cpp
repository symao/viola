#include <opencv2/opencv.hpp>
#include <viola/vs_improc.h>
#include <viola/vs_color_adjust.h>

void compareAdjustors(cv::Mat bg_img, const cv::Mat& fg_img, const cv::Mat& fg_mask) {
  std::vector<std::shared_ptr<vs::ColorAdjustor>> adjustors = {
      vs::createColorAdjustor(vs::COLOR_ADJUSTOR_HISTOGRAM_MATCH),
      vs::createColorAdjustor(vs::COLOR_ADJUSTOR_STATISTIC_MATCH),
  };

  std::vector<cv::Mat> fg_list = {fg_img};
  for (auto& adjustor : adjustors) {
    adjustor->init(bg_img);
    cv::Mat adjust_img;
    adjust_img = adjustor->adjust(fg_img, 0.6, fg_mask);
    fg_list.push_back(adjust_img);
  }

  std::vector<cv::Mat> mask_list(fg_list.size(), fg_mask);
  vs::imageComposition(bg_img, fg_list, mask_list);

  cv::imshow("img", bg_img);
  cv::imwrite("img2.jpg", bg_img);
  cv::waitKey();
  cv::destroyAllWindows();
}

int main() {
  cv::Mat fg_img = cv::imread(PROJECT_DIR "/data/girl.png");
  cv::Mat fg_mask = cv::imread(PROJECT_DIR "/data/girl_mask.png", cv::IMREAD_GRAYSCALE);
  std::vector<cv::Mat> bg_img_list = {
      cv::imread(PROJECT_DIR "/data/bg1.jpg"),
      cv::imread(PROJECT_DIR "/data/bg2.jpg"),
      cv::imread(PROJECT_DIR "/data/bg3.jpg"),
      cv::imread(PROJECT_DIR "/data/bg4.jpg"),
  };

  auto adjustor = vs::createColorAdjustor(vs::COLOR_ADJUSTOR_HISTOGRAM_MATCH);
  for (auto bg_img : bg_img_list) {
    adjustor->init(bg_img);
    cv::Mat adjust_fg = adjustor->adjust(fg_img, 0.4, fg_mask);
    vs::imageComposition(bg_img, {adjust_fg}, {fg_mask});
    cv::imshow("image composition", bg_img);
    cv::waitKey();
  }
}