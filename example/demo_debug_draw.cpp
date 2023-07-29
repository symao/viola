#include <viola/vs_debug_draw.h>
#include <viola/vs_video_saver.h>

int main(int argc, char** argv) {
  cv::VideoCapture cap(PROJECT_DIR "/data/feature.mp4");
  std::vector<cv::Mat> img_list;
  cv::Mat img;
  for (int idx = 0; cap.read(img); idx++) {
    if (idx % 2 == 0) {
      cv::resize(img, img, cv::Size(160, 120));
      img_list.push_back(img);
    }
  }

  cv::Mat img_hstack = vs::hstack(vs::subvec(img_list, 0, 5));
  cv::putText(img_hstack, "hstack", cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 100, 255), 2);
  cv::Mat img_vstack = vs::vstack(vs::subvec(img_list, 0, 5));
  cv::putText(img_vstack, "vstack", cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 100, 255), 2);
  cv::Mat img_grid_stack = vs::gridStack(img_list, -1, 5);
  cv::putText(img_grid_stack, "grid stack", cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 100, 255),
              2);

  cv::imwrite("hstack.jpg", img_hstack);
  cv::imwrite("vstack.jpg", img_vstack);
  cv::imwrite("gridstack.jpg", img_grid_stack);
}