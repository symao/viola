#include <viola/vs_basic.h>
#include <viola/vs_tag.h>
#include <viola/vs_os.h>
#include <viola/vs_debug_draw.h>
#include <opencv2/opencv.hpp>

void processImg(std::shared_ptr<vs::TagDetector>& detector, cv::Mat img) {
  if (img.empty()) return;
  auto tags = detector->detect(vs::toGray(img));
  vs::drawTags(img, tags);
  // cv::imwrite("tag.jpg", img);
  cv::imshow("tag detect", img);
  static vs::WaitKeyHandler handler;
  handler.waitKey();
}

void testAprilTag() {
  vs::TagConfig tag_cfg;
  tag_cfg.tag_type = vs::TAG_APRILTAG;
  tag_cfg.tag_cols = 6;
  tag_cfg.tag_rows = 6;
  tag_cfg.tag_size = 0.072;
  tag_cfg.tag_spacing = 0.25;
  auto detector = vs::createTagDetector(tag_cfg);
  processImg(detector, cv::imread(PROJECT_DIR "/data/tag_april.png"));
}

void testAruco() {
  vs::TagConfig tag_cfg;
  tag_cfg.tag_type = vs::TAG_ARUCO;
  auto detector = vs::createTagDetector(tag_cfg);
  processImg(detector, cv::imread(PROJECT_DIR "/data/tag_aruco.jpg"));
}

void testChessboard() {
  vs::TagConfig tag_cfg;
  tag_cfg.tag_type = vs::TAG_CHESSBOARD;
  tag_cfg.tag_rows = 7;
  tag_cfg.tag_cols = 7;
  auto detector = vs::createTagDetector(tag_cfg);
  processImg(detector, cv::imread(PROJECT_DIR "/data/tag_chessboard.png"));
}

int main(int argc, char** argv) {
  if (argc < 2) {
    // printf("Usage: ./demo_tag_detect image_path");
    testAprilTag();
    testAruco();
    testChessboard();
    return 0;
  }
  const char* img_path = argv[1];
  vs::TagConfig tag_cfg;
  tag_cfg.tag_type = vs::TAG_APRILTAG;
  tag_cfg.tag_cols = 6;
  tag_cfg.tag_rows = 6;
  tag_cfg.tag_size = 0.072;
  tag_cfg.tag_spacing = 0.25;

  auto detector = vs::createTagDetector(tag_cfg);
  if (vs::isdir(img_path)) {
    auto img_list = vs::listdir(img_path, true);
    std::sort(img_list.begin(), img_list.end());
    for (const auto& fimg : img_list) {
      cv::Mat img = cv::imread(fimg);
      processImg(detector, img);
    }
  } else if (vs::isimage(img_path)) {
    cv::Mat img = cv::imread(img_path);
    processImg(detector, img);
  } else if (vs::isvideo(img_path)) {
    cv::VideoCapture cap(img_path);
    cv::Mat img;
    while (cap.read(img)) {
      processImg(detector, img);
    }
  } else {
    printf("[ERROR]Cannot open image path:%s\n", img_path);
  }
}