/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details file I/O for OpenCV mat
 */
#pragma once
#include <fstream>
#include <vector>
#include <opencv2/core.hpp>

namespace vs {

/** @brief write OpenCV mat data into file in inner format.
 * @param[in]f: file path where mat data will be written to.
 * @param[in]mat: mat data to be written.
 * @return where writting ok.
 */
bool writeMatBin(const char* binfile, const cv::Mat& mat);

/** @brief write OpenCV mat data into ofstream in inner format.
 * @param[in/out]f: ostream where mat data will be written to.
 * @param[in]mat: mat data to be written.
 */
void writeMatBin(std::ofstream& f, const cv::Mat& mat);

/** @brief read OpenCV mat data from file which is written by writeMatBin().
 * @param[in]binfile: file path of mat data.
 * @return mat data in input file, return empty mat if read failed.
 */
cv::Mat readMatBin(const char* binfile);

/** @brief read OpenCV mat data from ifstream.
 * @param[in]f: ifstream of file which is written by writeMatBin().
 * @return mat data in ifstream, return empty mat if read failed.
 */
cv::Mat readMatBin(std::ifstream& f);

}  // namespace vs
