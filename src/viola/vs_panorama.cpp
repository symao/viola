/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-03-03 16:37
 * @details
 */
#include "vs_panorama.h"
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "vs_basic.h"

namespace vs {

cv::Mat equirectangular2perspective(const cv::Mat& img, const cv::Mat& R, float fov, const cv::Size& out_size) {
  if (img.empty()) return cv::Mat();
  float f = 0.5f * out_size.width / tan(fov * 0.5f);
  float cx = out_size.width * 0.5f;
  float cy = out_size.height * 0.5f;

  cv::Mat K = (cv::Mat_<float>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
  cv::Mat Kinv = K.inv();

  cv::Mat rot_mat;
  R.convertTo(rot_mat, CV_32FC1);
  float* r = reinterpret_cast<float*>(rot_mat.data);

  cv::Mat map_x(out_size, CV_32FC1);
  cv::Mat map_y(out_size, CV_32FC1);
  float* ptr_map_x = reinterpret_cast<float*>(map_x.data);
  float* ptr_map_y = reinterpret_cast<float*>(map_y.data);
  for (int i = 0; i < map_x.rows; i++) {
    for (int j = 0; j < map_x.cols; j++) {
      // pixel to camera
      float x = (j - cx) / f;
      float y = (i - cy) / f;
      float z = 1;

      // camera to world
      float xw = r[0] * x + r[1] * y + r[2] * z;
      float yw = r[3] * x + r[4] * y + r[5] * z;
      float zw = r[6] * x + r[7] * y + r[8] * z;

      // world to lati,longi
      float norm = hypot3(xw, yw, zw);
      xw /= norm;
      yw /= norm;
      zw /= norm;
      float longi = atan2(xw, zw);
      float lati = asin(yw);

      // lati,longi to equirectangle uv
      float u = (longi / VS_2PI + 0.5f) * img.cols;
      float v = (lati / VS_PI + 0.5f) * img.rows;

      *ptr_map_x++ = u;
      *ptr_map_y++ = v;
    }
  }
  cv::Mat out;
  cv::remap(img, out, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_WRAP);
  return out;
}

cv::Mat equirectangular2perspective(const cv::Mat& img, float theta, float phi, float fov, const cv::Size& out_size) {
  cv::Mat Rtheta, Rphi;
  cv::Rodrigues(cv::Vec3f(0, theta, 0), Rtheta);
  cv::Rodrigues(cv::Vec3f(phi, 0, 0), Rphi);
  cv::Mat R = Rtheta * Rphi;
  return equirectangular2perspective(img, R, fov, out_size);
}

} /* namespace vs */