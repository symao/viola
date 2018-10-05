/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_cv_io.h"
#include <fstream>

namespace vs {

cv::Mat readMatBin(std::ifstream& f) {
  if (!f.is_open()) return cv::Mat();
  uint32_t h, w, type;
  f.read(reinterpret_cast<char*>(&h), sizeof(h));
  f.read(reinterpret_cast<char*>(&w), sizeof(w));
  f.read(reinterpret_cast<char*>(&type), sizeof(type));
  cv::Mat mat(h, w, type);
  char* ptr = reinterpret_cast<char*>(mat.data);
  uint32_t size = h * w * mat.elemSize();
  f.read(ptr, size);
  return mat;
}

cv::Mat readMatBin(const char* binfile) {
  std::ifstream fin(binfile, std::ios_base::in | std::ios::binary);
  return readMatBin(fin);
}

void writeMatBin(std::ofstream& f, const cv::Mat& mat) {
  uint32_t h = mat.rows;
  uint32_t w = mat.cols;
  uint32_t type = mat.type();
  f.write(reinterpret_cast<char*>(&h), sizeof(h));
  f.write(reinterpret_cast<char*>(&w), sizeof(w));
  f.write(reinterpret_cast<char*>(&type), sizeof(type));
  uint32_t size = h * w * mat.elemSize();
  f.write(reinterpret_cast<char*>(mat.data), size);
}

bool writeMatBin(const char* binfile, const cv::Mat& mat) {
  std::ofstream f(binfile, std::ios_base::out | std::ios::binary);
  if (!f.is_open()) return false;
  writeMatBin(f, mat);
  return true;
}

}  // namespace vs