#include <viola/vs_imtrans.h>
#include <viola/vs_debug_draw.h>

int main() {
  const char* fvideo = "/home/symao/workspace/camera_track/rsc/2021_12_08_1424243290_mask.mp4";
  std::vector<vs::ImageTransformHandler> options = {
      // // vs::ImageTransformer::guidedFilter(7, 1e-4),
      // vs::ImageTransformer::trunc(0.3, 0.8),
      vs::ImageTransformer::convertTo(CV_8UC1, 255),
      // vs::ImageTransformer::erode(cv::Size(3, 3)),
      vs::ImageTransformer::connectComponentFilter(20),
  };
  vs::ImageTransformer transformer(options);

  cv::VideoCapture cap(fvideo);
  double resize_k = 0.33333333333;
  cv::Mat img;
  for (int idx = 0; cap.read(img); idx++) {
    cv::resize(img, img, cv::Size(), resize_k, resize_k);
    int w = img.cols / 2;
    cv::Mat bgr = img.colRange(0, w);
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat mask;
    cv::cvtColor(img.colRange(w, w + w), mask, cv::COLOR_BGR2GRAY);
    cv::Mat raw_mask = mask.clone();

    mask.convertTo(mask, CV_32FC1, 1.0 / 255);
    transformer.run(mask, gray);

    cv::imshow("mask", vs::hstack({raw_mask, mask}));
    auto key = cv::waitKey();
    if (key == 27) break;
  }
}