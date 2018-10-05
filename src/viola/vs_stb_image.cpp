/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_stb_image.h"
#define VS_ENABLE_STB_IMPL 0

#if VS_ENABLE_STB_IMPL
namespace vs {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#pragma GCC diagnostic pop

#define STB_JPEG_QUALITY_LEVEL 95

// stbi image R-G-B order => cv::imread B-G-R order
static cv::Mat stbToOpenCV(const cv::Mat& stb_img, int flags) {
  if (stb_img.empty()) return cv::Mat();
  cv::Mat cv_img;
  int channel = stb_img.channels();
  if (flags < 0) {  // IMREAD_UNCHANGED
    if (channel == 1)
      stb_img.copyTo(cv_img);
    else if (channel == 3)
      cv::cvtColor(stb_img, cv_img, cv::COLOR_RGB2BGR);
    else if (channel == 4)
      cv::cvtColor(stb_img, cv_img, cv::COLOR_RGBA2BGRA);
  } else if (flags > 0) {  // IMREAD_COLOR
    if (channel == 1)
      cv::cvtColor(stb_img, cv_img, cv::COLOR_GRAY2BGR);
    else if (channel == 3)
      cv::cvtColor(stb_img, cv_img, cv::COLOR_RGB2BGR);
    else if (channel == 4)
      cv::cvtColor(stb_img, cv_img, cv::COLOR_RGBA2BGR);
  } else {  // IMREAD_GRAYSCALE
    if (channel == 1)
      stb_img.copyTo(cv_img);
    else if (channel == 3)
      cv::cvtColor(stb_img, cv_img, cv::COLOR_RGB2GRAY);
    else if (channel == 4)
      cv::cvtColor(stb_img, cv_img, cv::COLOR_RGBA2GRAY);
  }
  return cv_img;
}

// cv::imread B-G-R order => stbi image R-G-B order
static cv::Mat openCvToStb(const cv::Mat& cv_img) {
  if (cv_img.empty()) return cv::Mat();
  cv::Mat stb_img;
  int channel = cv_img.channels();
  if (channel == 1)
    cv::cvtColor(cv_img, stb_img, cv::COLOR_GRAY2RGB);
  else if (channel == 3)
    cv::cvtColor(cv_img, stb_img, cv::COLOR_BGR2RGB);
  else if (channel == 4)
    cv::cvtColor(cv_img, stb_img, cv::COLOR_BGRA2RGBA);
  else
    stb_img = cv_img;
  return stb_img;
}

cv::Mat imread(const char* file, int flags) {
  int w = 0, h = 0, channel = 0;
  int req_cmp = STBI_default;
  unsigned char* data = stbi_load(file, &w, &h, &channel, req_cmp);
  if (!data || w == 0 || h == 0 || channel == 0) return cv::Mat();
  cv::Mat img(h, w, CV_MAKETYPE(CV_8U, channel), data);
  cv::Mat res = stbToOpenCV(img, flags);
  stbi_image_free(data);
  return res;
}

bool imwrite(const char* file, const cv::Mat& img) {
  std::string s(file);
  if (s.length() <= 4) return false;
  cv::Mat img_save = openCvToStb(img);
  if (img_save.empty()) return false;
  int w = img_save.cols;
  int h = img_save.rows;
  int comp = img_save.channels();
  int stride = img_save.cols * img_save.channels();
  const void* data = img_save.data;

  std::string suffix(s.substr(s.length() - 4));
  if (suffix == ".png") {
    stbi_write_png(file, w, h, comp, data, stride);
    return true;
  } else if (suffix == ".jpg") {
    stbi_write_jpg(file, w, h, comp, data, STB_JPEG_QUALITY_LEVEL);
    return true;
  } else if (suffix == ".bmp") {
    stbi_write_bmp(file, w, h, comp, data);
    return true;
  }
  return false;
}

cv::Mat imreadMemory(const char* buffer, int size, int flags) {
  int w = 0, h = 0, channel = 0;
  int req_cmp = STBI_default;
  unsigned char* data =
      stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(buffer), size, &w, &h, &channel, req_cmp);
  if (!data || w == 0 || h == 0 || channel == 0) return cv::Mat();
  cv::Mat img(h, w, CV_MAKETYPE(CV_8U, channel), data);
  cv::Mat res = stbToOpenCV(img, flags);
  stbi_image_free(data);
  return res;
}

static void stbiWriteMemory(void* context, void* data, int size) {
  std::vector<uchar>* ptr = reinterpret_cast<std::vector<uchar>*>(context);
  int raw_size = ptr->size();
  ptr->resize(raw_size + size);
  memcpy(&ptr->at(raw_size), data, size);
}

std::vector<uchar> imwriteMemory(const cv::Mat& img, int format) {
  std::vector<uchar> res;
  cv::Mat img_save = openCvToStb(img);
  if (img_save.empty()) return res;
  int w = img_save.cols;
  int h = img_save.rows;
  int comp = img_save.channels();
  int stride = img_save.cols * img_save.channels();
  const void* data = img_save.data;

  switch (format) {
    case STB_WRITE_PNG:
      stbi_write_png_to_func(stbiWriteMemory, &res, w, h, comp, data, stride);
      break;
    case STB_WRITE_JPG:
      stbi_write_jpg_to_func(stbiWriteMemory, &res, w, h, comp, data, STB_JPEG_QUALITY_LEVEL);
      break;
    case STB_WRITE_BMP:
      stbi_write_bmp_to_func(stbiWriteMemory, &res, w, h, comp, data);
      break;
    default:
      break;
  }
  return res;
}
}  // namespace vs
#else
#include <opencv2/imgcodecs.hpp>

namespace vs {

cv::Mat imread(const char* file, int flags) { return cv::imread(file, flags); }

bool imwrite(const char* file, const cv::Mat& img) { return cv::imwrite(file, img); }

cv::Mat imreadMemory(const char* buffer, int size, int flags) {
  // todo
  return cv::Mat();
}

std::vector<uchar> imwriteMemory(const cv::Mat& img, int format) {
  // todo
  return {};
}

}  // namespace vs
#endif
