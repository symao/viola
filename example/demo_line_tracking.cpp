#include <opencv2/opencv.hpp>
#include <viola/vs_basic.h>
#include <viola/vs_feature_line.h>
#include <viola/vs_os.h>

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("Usage: ./demo_line_tracking [xxx.mp4 or image dir]\n");
    return -1;
  }
  vs::LineFeatureTracker line_tracker;
  std::string data_path(argv[1]);
  if (vs::isdir(data_path.c_str())) {
    auto files = vs::listdir(data_path.c_str());
    vs::vecReduceIf(files, [](const std::string& path) { return !vs::isimage(path.c_str()); });
    std::sort(files.begin(), files.end());
    for (const auto& file : files) {
      cv::Mat img = cv::imread(vs::join(data_path, file));
      if (img.empty()) continue;
      cv::Mat gray;
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
      auto res = line_tracker.process(gray);
      cv::Mat img_show;
      cv::cvtColor(gray, img_show, cv::COLOR_GRAY2BGR);
      line_tracker.drawLineFeature(img_show, res.line_features);
      cv::imshow("img", img_show);
      auto key = cv::waitKey();
      if (key == 27) break;
    }
  } else if (vs::endswith(data_path, ".mp4") || vs::endswith(data_path, ".avi")) {
    cv::VideoCapture cap(data_path);
    cv::Mat img;
    while (cap.read(img)) {
      if (img.empty()) continue;
      cv::Mat gray;
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
      auto res = line_tracker.process(gray);
      cv::Mat img_show;
      cv::cvtColor(gray, img_show, cv::COLOR_GRAY2BGR);
      line_tracker.drawLineFeature(img_show, res.line_features);
      cv::imshow("img", img_show);
      auto key = cv::waitKey();
      if (key == 27) break;
    }
  } else {
    printf("[ERROR]Unknown datapath:%s\n", data_path.c_str());
  }
  cv::destroyAllWindows();
}