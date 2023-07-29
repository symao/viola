#include <viola/vs_basic.h>
#include <viola/vs_panorama.h>
#include <viola/vs_debug_draw.h>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
  cv::Mat pano_img = cv::imread(PROJECT_DIR "/data/panorama.jpg");
  double fov = VS_RAD60;
  cv::Size view_size(480, 640);
  for (float phi = 0; phi < 90; phi += 10) {
    for (float theta = -180; theta < 180; theta += 10) {
      cv::Mat view = vs::equirectangular2perspective(pano_img, vs::deg2rad(theta), vs::deg2rad(phi), fov, view_size);
      cv::imshow("view", view);

      // // draw rectangle on pano image
      // cv::Mat debug_img = pano_img.clone();
      // int cx = (vs::normalizeDeg(theta) / 360 + 0.5) * debug_img.cols;
      // int cy = (0.5 - vs::normalizeDeg(phi) / 180) * debug_img.rows;
      // int w = fov / VS_2PI * debug_img.cols;
      // int h = fov * VS_FLOAT(view_size.height) / view_size.width / VS_PI * debug_img.rows;
      // cv::rectangle(debug_img, cv::Rect(cx - w / 2, cy - h / 2, w, h), cv::Scalar(0, 0, 255), 2);
      // cv::resize(view, view, cv::Size(), VS_FLOAT(debug_img.rows) / view.rows, VS_FLOAT(debug_img.rows) / view.rows);
      // cv::imshow("pano", vs::hstack({debug_img, view}));
      // cv::imwrite("../doc/img/pano/pano.jpg", vs::hstack({debug_img, view}));

      auto key = cv::waitKey();
      if (key == 27) break;
    }
  }
}