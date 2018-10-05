/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_color_filter.h"

#include <cmath>
#include <fstream>
#include <map>
#include <stack>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace vs {

ColorModel::ColorModel(int type) : type_(type) {}

void ColorModel::filter(const cv::Mat& input, cv::Mat& mask) const {
  mask = cv::Mat(input.size(), CV_8UC1, cv::Scalar(0));
  for (int i = 0; i < input.rows; i++) {
    const cv::Vec3b* p = input.ptr<cv::Vec3b>(i);
    uchar* m = mask.ptr<uchar>(i);
    for (int j = 0; j < input.cols; j++) {
      *m++ = judge(*p++);
    }
  }
}

struct GaussModel : public ColorModel {
  GaussModel(int _type = 0, int _sample_cnt = 0, cv::Mat _mu = cv::Mat(), cv::Mat _cov_inv = cv::Mat(),
             double _sigma = 4.5)
      : ColorModel(_type), sample_cnt(_sample_cnt), mu(_mu), cov_inv(_cov_inv), sigma(_sigma) {}
  int sample_cnt;
  cv::Mat mu;
  cv::Mat cov_inv;
  double sigma;
  virtual uchar judge(const cv::Vec3b& a) const {
    if (sample_cnt <= 0) return false;
    cv::Mat err = a - mu;
    cv::Mat res = err * cov_inv * err.t();
    return (res.at<double>(0, 0) < sigma) ? 255 : 0;
  }
};

struct ColorModelLUT : public ColorModel {
  ColorModelLUT(const char* color_file) : ColorModel(BGR), size(1 << 24), lut(size, 0) {
    std::ifstream fin(color_file, std::ios::binary);
    if (!fin.is_open()) {
      printf("[ERROR] ColorModelLUT cannot open file '%s'\n", color_file);
      return;
    }

    char bit, ds;
    fin.read(&bit, 1);
    fin.read(&ds, 1);

    std::stack<uchar> c;
    auto fget = [&fin, bit, &c]() {
      if (bit == 8) {
        uchar a;
        fin.read((char*)&a, 1);
        return a;
      }
      if (c.empty()) {
        uchar a;
        fin.read((char*)&a, 1);
        int k = 8 / bit;
        int b = 8 - bit;
        int mask = (1 << bit) - 1;
        for (int i = 0; i < k; i++) {
          c.push((a & mask) << b);
          a = a >> bit;
        }
      }
      uchar res = c.top();
      c.pop();
      return res;
    };

    for (int i = 0; i < 256; i += ds)
      for (int j = 0; j < 256; j += ds)
        for (int k = 0; k < 256; k += ds) lut[idx(i, j, k)] = fget();

    interpolation(ds);
  }

  ColorModelLUT(const std::vector<uint64_t>& table, int bit, int ds) : ColorModel(BGR), size(1 << 24), lut(size, 0) {
    std::vector<uint64_t> table_new;
    table_new.reserve(table.size() << 1);
    for (auto it = table.begin(); it != table.end(); it++) {
      uint64_t a = *it;
      if (a != 0)
        table_new.push_back(a);
      else {
        it++;
        for (uint64_t i = 0; i <= *it; i++) table_new.push_back(0);
      }
    }
    int id = 0;
    std::stack<uchar> c;
    int k = 64 / bit;
    int b = 8 - bit;
    int m = (1 << bit) - 1;
    auto fget = [&table_new, &id, &c, bit, k, b, m]() {
      if (c.empty()) {
        uint64_t a = table_new[id++];
        for (int i = 0; i < k; i++) {
          c.push((a & m) << b);
          a = a >> bit;
        }
      }
      uchar res = c.top();
      c.pop();
      return res;
    };

    for (int i = 0; i < 256; i += ds)
      for (int j = 0; j < 256; j += ds)
        for (int k = 0; k < 256; k += ds) lut[idx(i, j, k)] = fget();

    interpolation(ds);
  }

  virtual uchar judge(const cv::Vec3b& a) const {
    uint32_t idx = ((uint32_t)(a[0]) << 16) | ((uint32_t)(a[1]) << 8) | (uint32_t)(a[2]);
    return (lut[idx] > 80) ? 255 : 0;
  }

  const uint32_t size;
  std::vector<uchar> lut;

  int idx(int i, int j, int k) { return (i << 16) | (j << 8) | k; }

  void interpolation(int ds) {
    if (ds != 2) return;
    for (int i = 0; i < 256; i += ds)
      for (int j = 0; j < 256; j += ds) {
        for (int k = 1; k < 255; k += ds)
          lut[idx(i, j, k)] = (static_cast<int>(lut[idx(i, j, k + 1)]) + static_cast<int>(lut[idx(i, j, k - 1)])) >> 1;
        lut[idx(i, j, 255)] = lut[idx(i, j, 254)];
      }

    for (int i = 0; i < 256; i += ds) {
      for (int j = 1; j < 255; j += ds)
        for (int k = 0; k < 256; k++)
          lut[idx(i, j, k)] = (static_cast<int>(lut[idx(i, j + 1, k)]) + static_cast<int>(lut[idx(i, j - 1, k)])) >> 1;
      for (int k = 0; k < 256; k++) lut[idx(i, 255, k)] = lut[idx(i, 254, k)];
    }

    for (int i = 1; i < 255; i += ds) {
      for (int j = 0; j < 256; j++)
        for (int k = 0; k < 256; k++)
          lut[idx(i, j, k)] = (static_cast<int>(lut[idx(i + 1, j, k)]) + static_cast<int>(lut[idx(i - 1, j, k)])) >> 1;
    }
    for (int j = 0; j < 256; j++)
      for (int k = 0; k < 256; k++) lut[idx(255, j, k)] = lut[idx(254, j, k)];
  }
};

ColorModelPtr ColorModel::red() {
  return ColorModelPtr(new ColorModelRange(
      ColorModel::HSV, {ColorModelRange::ColorRange({156, 180}), ColorModelRange::ColorRange({0, 10})},
      {ColorModelRange::ColorRange({43, 255})}, {ColorModelRange::ColorRange({46, 255})}));
}

ColorModelPtr ColorModel::green() {
  return ColorModelPtr(new ColorModelRange(ColorModel::HSV, {ColorModelRange::ColorRange({35, 77})},
                                           {ColorModelRange::ColorRange({43, 255})},
                                           {ColorModelRange::ColorRange({46, 255})}));
}

ColorModelPtr ColorModel::blue() {
  return ColorModelPtr(new ColorModelRange(ColorModel::HSV, {ColorModelRange::ColorRange({100, 124})},
                                           {ColorModelRange::ColorRange({43, 255})},
                                           {ColorModelRange::ColorRange({46, 255})}));
}

ColorModelPtr ColorModel::black() {
  return ColorModelPtr(new ColorModelRange(ColorModel::HSV, {ColorModelRange::ColorRange({0, 180})},
                                           {ColorModelRange::ColorRange({0, 255})},
                                           {ColorModelRange::ColorRange({0, 46})}));
}

ColorModelPtr ColorModel::white() {
  return ColorModelPtr(new ColorModelRange(ColorModel::HSV, {ColorModelRange::ColorRange({0, 180})},
                                           {ColorModelRange::ColorRange({0, 30})},
                                           {ColorModelRange::ColorRange({221, 255})}));
}

ColorModelPtr ColorModel::yellow() {
  return ColorModelPtr(new ColorModelRange(ColorModel::HSV, {ColorModelRange::ColorRange({12, 33})},
                                           {ColorModelRange::ColorRange({44, 255})},
                                           {ColorModelRange::ColorRange({26, 255})}));
}

static auto fkernel = [](int k = 3) { return cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k)); };

void colorFilter(const cv::Mat& img_bgr, cv::Mat& mask, const ColorModelList& model_list, float resize_rate,
                 int post_process) {
  if (img_bgr.channels() != 3) {
    printf("[ERROR]ColorFilter: Need input bgr image.\n");
    return;
  }

  std::map<int, cv::Mat> inputs;
  auto& bgr = inputs[ColorModel::BGR];
  if (resize_rate <= 0)
    bgr = img_bgr;
  else
    cv::resize(img_bgr, bgr, cv::Size(), resize_rate, resize_rate);

  // create inputs
  for (const auto& p : model_list) {
    auto& input = inputs[p->type()];
    if (input.empty()) {
      switch (p->type()) {
        case ColorModel::RGB:
          cv::cvtColor(bgr, input, cv::COLOR_BGR2RGB);
          break;
        case ColorModel::HSV:
          cv::cvtColor(bgr, input, cv::COLOR_BGR2HSV);
          break;
        default:
          printf("[ERROR]Unknown color model type '%d'\n", p->type());
          break;
      }
    }
  }

  mask = cv::Mat(bgr.size(), CV_8UC1, cv::Scalar(0));
  for (const auto& p : model_list) {
    cv::Mat mask_i;
    p->filter(inputs[p->type()], mask_i);
    mask |= mask_i;
  }

  if (post_process) {
    if (post_process & CFPM_MORPHOLOGY) {
      cv::dilate(mask, mask, fkernel(3));
      cv::dilate(mask, mask, fkernel(3));
      cv::erode(mask, mask, fkernel(3));
    }
    if (post_process & CFPM_FLOODFILL) {
      cv::Mat temp = ~mask;
      cv::filterSpeckles(temp, 0, 220, 50);
      mask = ~temp;
    }
    if (post_process & CFPM_SPECKLE) {
      cv::filterSpeckles(mask, 0, 100, 50);
    }
  }

  if (mask.size() != img_bgr.size()) {
    cv::resize(mask, mask, img_bgr.size());
  }
}

} /* namespace vs */