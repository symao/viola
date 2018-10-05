/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-03-03 16:37
 * @details convert equirectangular panorama image to perspective local view with camera intrinsic and extrinsic.
 */
#pragma once

#include <opencv2/core.hpp>

namespace vs {

/** @brief convert equirectangular panorama image to perspective local view
 * @param[in]img: input equirectangular panorama image
 * @param[in]R: rotation matrix which transform point from camera coordinate system to world
 * @note: world coordinate system: x-north y-ground z-west
 * @param[in]fov: horizontal field-of-vision of the output perspective image [radians]
 * @param[in]out_size: image size of output perspective image
 * @return cv::Mat output perspective image
 */
cv::Mat equirectangular2perspective(const cv::Mat& img, const cv::Mat& R, float fov, const cv::Size& out_size);

/** @brief convert equirectangular panorama image to perspective local view
 * @param[in]img: input equirectangular panorama image
 * @param[in]theta: z-axis angle (0: forward, pi: backword) [radians]
 * @param[in]phi: y-axis angle (>0: upper, <0: lower) [radians]
 * @param[in]fov: horizontal field-of-vision of the output perspective image [radians]
 * @param[in]out_size: image size of output perspective image
 * @return cv::Mat output perspective image
 */
cv::Mat equirectangular2perspective(const cv::Mat& img, float theta, float phi, float fov, const cv::Size& out_size);

} /* namespace vs */