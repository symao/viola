/**
 * Copyright (c) 2019-2021 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @Author: shuyuanmao
 * @EMail: maoshuyuan123@gmail.com
 * @Date: 2022-07-02 16:54
 * @Description:
 */
#include <opencv2/opencv.hpp>
#include <viola/vs_feature.h>

int main(int argc, char** argv) {
  std::string fvideo = argc > 1 ? argv[1] : PROJECT_DIR "/data/feature.mp4";

  vs::FeatureTracker tracker;
  // modify config
  auto cfg = tracker.getConfig();
  cfg.feature_type = 0;
  cfg.min_corner_dist = 50;
  cfg.max_corner = 200;
  tracker.setConfig(cfg);

  // process video stream
  cv::Mat img;
  bool halt = true;
  cv::VideoCapture cap(fvideo);
  while (cap.read(img)) {
    // process image
    auto track_res = tracker.process(img);
    // draw features
    tracker.drawFeature(img, track_res.features);
    cv::imshow("img", img);
    auto key = cv::waitKey(halt ? 0 : 10);
    if (key == 27) {
      break;
    } else if (key == 's') {
      halt = !halt;
    }
  }
}