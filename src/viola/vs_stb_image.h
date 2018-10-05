/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details Image reading/writing using stb_image api, same use as cv::imread/cv::imwrite.
 */
#pragma once

/** @brief Image read/write api by stb_image, which is the same use as cv::imread/cv::imwrite in OpenCV.
 * How to use:
 * 1. Download stb_image.h, stb_image_write.h from https://github.com/nothings/stb.
 * 2. Make sure vs_stb_image.cpp can include the headers download at step 1.
 * 3. set VS_ENABLE_STB_IMPL to 1 in vs_stb_image.cpp. if you want to use opencv version, set to 0.
 */

#include <opencv2/core.hpp>

namespace vs {

/** @brief same use as cv::imread */
cv::Mat imread(const char* file, int flags = -1);

/** @brief same use as cv::imwrite */
bool imwrite(const char* file, const cv::Mat& img);

/** @brief read image from file memory */
cv::Mat imreadMemory(const char* buffer, int size, int flags = -1);

enum StbWriteFormat { STB_WRITE_PNG = 0, STB_WRITE_JPG = 1, STB_WRITE_BMP = 2 };

/** @brief write image into memory */
std::vector<uchar> imwriteMemory(const cv::Mat& img, int format);

}  // namespace vs
