/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details pixel align using direct method in VSLAM
 */
#pragma once
#include <opencv2/core.hpp>

namespace vs {

/** @brief alignment point with template matching
 * @param[in]src: source image in grayscale
 * @param[in]tar: target image in grayscale
 * @param[in]H: Homography which transform source patch to target image
 * @param[in]src_pt: source point
 * @param[in/out]tar_pt: target point
 * @param[in]patch_size: patch size
 * @param[in]search_size: search size, default is twice patch size
 * @param[in]method: match score type, see cv::matchTemplate(), default TM_SQDIFF.
 * @param[in]subpixel_refine: refine target point in subpixel precision
 * @return whether affine ok
 */
bool affineMatch(const cv::Mat& src, const cv::Mat& tar, const cv::Mat& H, const cv::Point2f& src_pt,
                 cv::Point2f& tar_pt, cv::Size patch_size = cv::Size(11, 11), cv::Size search_size = cv::Size(),
                 int method = 0, bool subpixel_refine = false);

/** @brief find best matched position in image to input patch
 * @param[in]img: search image
 * @param[in]patch: search template patch
 * @param[in/out]pt: input initial position, output refined position
 * @param[in]search_size: search size
 * @param[in]method: match score type, see cv::matchTemplate(), default TM_SQDIFF.
 * @param[in]subpixel_refine: refine target point in subpixel precision
 * @return whether find matched point, if return false then output pt cannot be used.
 */
bool alignTemplateMatching(const cv::Mat& img, const cv::Mat& patch, cv::Point2f& pt, cv::Size search_size,
                           int method = 0, bool subpixel_refine = false);

bool alignTemplateMatching(const cv::Mat& img, const cv::Mat& patch, cv::Point2f& pt, double& match_val,
                           cv::Size search_size, int method = 0, bool subpixel_refine = false);
}  // namespace vs