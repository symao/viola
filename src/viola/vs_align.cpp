/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_align.h"
#include "vs_geometry2d.h"

namespace vs {

bool alignTemplateMatching(const cv::Mat& img, const cv::Mat& patch, cv::Point2f& pt, cv::Size search_size, int method,
                           bool subpixel_refine) {
  double match_val;
  return alignTemplateMatching(img, patch, pt, match_val, search_size, method, subpixel_refine);
}

bool alignTemplateMatching(const cv::Mat& img, const cv::Mat& patch, cv::Point2f& pt, double& match_val,
                           cv::Size search_size, int method, bool subpixel_refine) {
  int half_search_x = search_size.width / 2;
  int half_search_y = search_size.height / 2;

  if (!inside(pt, img.size(), cv::Size(half_search_x, half_search_y))) return false;

  cv::Rect search_rect(static_cast<int>(pt.x - half_search_x), static_cast<int>(pt.y - half_search_y),
                       search_size.width, search_size.height);
  cv::Mat search_img = img(search_rect);
  cv::Mat match_res;
  cv::matchTemplate(search_img, patch, match_res, method);

  double min_val, max_val;
  cv::Point min_loc, max_loc;
  cv::minMaxLoc(match_res, &min_val, &max_val, &min_loc, &max_loc, cv::Mat());
  cv::Point2f match_loc;
  switch (method) {
    case cv::TM_SQDIFF:
    case cv::TM_SQDIFF_NORMED:
      match_loc = cv::Point2f(min_loc.x, min_loc.y);
      match_val = min_val;
      break;
    case cv::TM_CCORR:
    case cv::TM_CCOEFF:
    case cv::TM_CCORR_NORMED:
    case cv::TM_CCOEFF_NORMED:
      match_loc = cv::Point2f(max_loc.x, max_loc.y);
      match_val = max_val;
      break;
    default:
      printf("[ERROR]affineMatch: unknow method %d\n", method);
      return false;
  }
  // do subpixel refinement for match loc
  if (subpixel_refine) {
    if (inInterval(match_loc.x, 0, match_res.cols - 1)) {
      float v_1 = match_res.at<float>(match_loc.y, match_loc.x - 1);
      float v1 = match_res.at<float>(match_loc.y, match_loc.x + 1);
      float dnorm = (v1 + v_1 - 2 * match_val);
      if (fabs(dnorm) > VS_EPS) match_loc.x += (v_1 - v1) / (dnorm * 2);
    }
    if (inInterval(match_loc.y, 0, match_res.rows - 1)) {
      float v_1 = match_res.at<float>(match_loc.y - 1, match_loc.x);
      float v1 = match_res.at<float>(match_loc.y + 1, match_loc.x);
      float dnorm = (v1 + v_1 - 2 * match_val);
      if (fabs(dnorm) > VS_EPS) match_loc.y += (v_1 - v1) / (dnorm * 2);
    }
  }

  // this must do after subpixel refinement since match res not divide rows*cols
  switch (method) {
    case cv::TM_SQDIFF:
    case cv::TM_CCORR:
    case cv::TM_CCOEFF:
      match_val /= patch.rows * patch.cols;
      break;
    default:
      break;
  }

  pt.x = search_rect.x + match_loc.x + patch.cols / 2;
  pt.y = search_rect.y + match_loc.y + patch.rows / 2;
  return true;
}

bool affineMatch(const cv::Mat& src, const cv::Mat& tar, const cv::Mat& H, const cv::Point2f& src_pt,
                 cv::Point2f& tar_pt, cv::Size patch_size, cv::Size search_size, int method, bool subpixel_refine) {
  cv::Mat Hinv = H.inv();
  int half_x = patch_size.width / 2;
  int half_y = patch_size.height / 2;
  if (search_size.width == 0 && search_size.height == 0) {
    search_size.width = half_x * 4 + 1;
    search_size.height = half_y * 4 + 1;
  }
  int half_search_x = search_size.width / 2;
  int half_search_y = search_size.height / 2;

  // transfrom src_pt to target image with Homography
  std::vector<cv::Point2f> proj_temp;
  perspectiveTransform(std::vector<cv::Point2f>({src_pt}), proj_temp, H);
  cv::Point2f proj_pt = proj_temp.front();

  // return false if projection point out of image view
  if (proj_pt.x < half_search_x || proj_pt.x + half_search_x >= tar.cols || proj_pt.y < half_search_y ||
      proj_pt.y + half_search_y >= tar.rows)
    return false;

  // create rect patch in target image center at project point
  std::vector<cv::Point2f> proj_patch_pts;
  proj_patch_pts.reserve(patch_size.width * patch_size.height);
  for (int i = -half_y; i <= half_y; i++)
    for (int j = -half_x; j <= half_x; j++) proj_patch_pts.push_back(cv::Point2f(proj_pt.x + j, proj_pt.y + i));

  // transform patch points from target image back to source image with inverse Homography
  std::vector<cv::Point2f> src_patch_pts;
  perspectiveTransform(proj_patch_pts, src_patch_pts, Hinv);

  // construct warp patch with back project patch points
  cv::Mat warp_patch(1, src_patch_pts.size(), src.type());
  cv::remap(src, warp_patch, src_patch_pts, cv::noArray(), cv::INTER_LINEAR);
  warp_patch = warp_patch.reshape(1, patch_size.height);

  // find best matched patch around target point to warp patch with template matching
  tar_pt = proj_pt;
  bool ok = alignTemplateMatching(tar, warp_patch, tar_pt, search_size, method, subpixel_refine);
#if 0
    cv::imshow("warp_patch", warp_patch);
    printf("%d (%.2f %.2f) (%.2f %.2f)-(%.2f %.2f)=(%.2f %.2f)\n", ok,
        src_pt.x, src_pt.y, proj_pt.x, proj_pt.y, tar_pt.x, tar_pt.y, proj_pt.x - tar_pt.x, proj_pt.y - tar_pt.y);
    cv::waitKey();
#endif
  return ok;
}

}  // namespace vs