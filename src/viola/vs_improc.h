/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details color conversion, region filter, alpha blending, histogram matching
 */
#pragma once
#include <opencv2/imgproc.hpp>
#include <vector>

namespace vs {

/** @brief grab hue channel from bgr image */
cv::Mat bgr2hue(const cv::Mat& bgr);

/** @brief grab saturation channel from bgr image */
cv::Mat bgr2saturation(const cv::Mat& bgr);

/** @brief gradient filter with Sobel */
void sobelFilter(const cv::Mat& img, cv::Mat& grad_bw, int grad_thres = 200);

/** @brief region filter, remove small patch */
void regionFilter(cv::Mat& img, int min_area, int max_area = 1 << 30);

/** @brief connected component filter, remove small area connected component
 * @param[in,out]img: one-channel image to be processed
 * @param[in]bw_thres: threshold to convert to binary image, which used to find connect components
 * @param[in]min_area_ratio: min valid connect component area / whole image area
 * @param[in]max_k: select top-K area connect components
 */
void connectedComponentFilter(cv::Mat& img, double bw_thres, float min_area_ratio = 0.1f, int max_k = 1);

/** @brief resize image so that max(w,h)<=max_length
 * @param[in]img: input image
 * @param[in]max_length: max width/height of resize image
 * @param[in]enlarge: if false, only resize small when max_length<max(w,h); if true, resize so that max(w,h)==max_length
 * @return cv::Mat
 */
cv::Mat resizeMaxLength(const cv::Mat& img, int max_length, bool enlarge = false);

/** @brief find bounding box for non-zero region in input mask
 * @param[in]mask: 1-channel image
 * @return bounding box for non-zero region in input mask
 */
cv::Rect maskRoi(const cv::Mat& mask);

/** @brief merge two same-sized images with alpha mat
 * @param[in]img1: first BGR or grayscale image
 * @param[in]img2: second BGR or grayscale image
 * @param[in]alpha1: single-channel alpha image for first image, which has same size as img1
 *                   if CV_8UC1, multiply with 1.0f/255.0 to float alpha
 * @return merged image, which is img1*alpha1 + img2*(1-alpha1)
 */
cv::Mat immerge(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& alpha1);
cv::Mat immerge(const cv::Mat& img1, const cv::Mat& img2, float alpha1);

/** @brief search the min max non-zero endpoint in input image by the input direction
 * @param[in]img: input one channel image in uint8_t
 * @param[in]search_dir: search direction, if direction is (1,0) then find left-right point, if direction is (-1,0),
 *                        then find right-left point, if direction is (0,1), then find top-down point
 * @param[out]p1: min point in image by search direction
 * @param[out]p2: max point in image by search direction
 * @param[in]max_process_length: resize image max length to max_process_length for acceleration
 * @return whether search success
 */
bool searchEndPointByDir(const cv::Mat& img, const cv::Vec2f& search_dir, cv::Point2f& p1, cv::Point2f& p2,
                         cv::Rect& roi_rect, int max_process_length = -1);

/** @brief search the min max non-zero endpoint in input image by the input direction
 * @param[in]img: input one channel image in uint8_t
 * @param[out]p1: min point in image by search direction
 * @param[out]p2: max point in image by search direction
 * @param[in] search_dir: 0: top->down, 1:left->right, 2: down->top, 3:right->left
 * @return whether search success
 */
bool searchEndPointByDir(const cv::Mat& img, cv::Point2f& p1, cv::Point2f& p2, int search_dir = 0, int search_step = 1);

/** @brief merge two same-sized images with alpha mat in place
 * @param[in]img1: first BGR or grayscale image
 * @param[in,out]img2: second BGR or grayscale image
 * @param[in]alpha1: single-channel alpha image for first image, which has same size as img1
 *                   if CV_8UC1, multiply with 1.0f/255.0 to float alpha
 */
void immergeInPlace(const cv::Mat& img1, cv::Mat& img2, const cv::Mat& alpha1);
void immergeInPlace(const cv::Mat& img1, cv::Mat& img2, float alpha1);

/** @brief image composition
 * @param[in,out]bg_img: background image
 * @param[in]fg_imgs: foreground image list
 * @param[in]fg_masks: foreground mask list
 */
void imageComposition(cv::Mat& bg_img, const std::vector<cv::Mat>& fg_imgs, const std::vector<cv::Mat>& fg_masks);

/** @brief split image with channels into a slice vector */
std::vector<cv::Mat> imsplit(const cv::Mat& img);

/** @brief calc histogram for image, return a 256 size vector, each bin store the frequency
 * @param[in]gray: input 1-channel image, type CV_8UC1
 * @param[in]normalized: if true, then output histgram is normalized sum to 1, else, return histogram count
 * @param[in]mask: mask data, if set, only use pixels from mask area
 * @return std::vector<float> output histogram, which is size 256
 */
std::vector<float> calcHist(const cv::Mat& gray, bool normalized = false, const cv::Mat& mask = cv::Mat());

/** @brief apply look-up table for input image
 * @param[in]src_img: input image to be adjusted
 * @param[in]luts: look-up table, size is [N][256], N must be same as src_img's channels
 * @param[in]roi: region of interest, if set, only adjust roi area
 * @return cv::Mat adjust image whose size is the same as src_img
 */
cv::Mat applyLut(const cv::Mat& src_img, const std::vector<std::vector<uchar>>& luts, const cv::Rect& roi = cv::Rect());

/** @brief crop input image and mask with non-zero region in mask, output ROI image and ROI mask
 * since downsampling image is small, ROI region in downsampling is less. so we crop ROI image before downsampling to
 * keep enough image information
 * @param[in]img: input image
 * @param[in]mask: input mask in 1-channel
 * @param[out]ROI_img: sub-image in input image with ROI rect (bounding box of non-zero region in mask)
 * @param[out]ROI_mask: sub-image in input image with ROI rect (bounding box of non-zero region in mask)
 * @param[in]pad_w: padding width in pixels for ROI rect
 * @param[in]deep_copy: whether deep copy to ROI_img and ROI_mask
 * @return ROI rect
 */
cv::Rect cropMaskRoi(const cv::Mat& img, const cv::Mat& mask, cv::Mat& roi_img, cv::Mat& roi_mask, int pad_w = 0,
                     bool deep_copy = false);

/** @brief image white balance */
cv::Mat whiteBalance(const cv::Mat& img, const cv::Mat& mask = cv::Mat(), float adjust_rate = 1);

/** @brief adjust brightness. level [-100,100] */
cv::Mat adjustBrightness(const cv::Mat& img, int level);

/** @brief Histogram matching, which transfor source color histogram into reference color histogram */
class HistogramMatch {
 public:
  typedef std::vector<std::vector<uchar>> LutType;
  typedef std::vector<std::vector<float>> HistogramType;

  /** @brief Constructor
   * @param[in]ref_img: reference image, which use to calculate reference histogram and CDF
   *                    and reference CDF
   * @param[in]resize_wh: if set, resize small before comput histogram
   * @param[in]prior_hist: if set, src_img histogram will be add with this prior hist
   */
  HistogramMatch(const cv::Mat& ref_img, const cv::Size& resize_wh = cv::Size(), const HistogramType& prior_hist = {});

  /** @brief calculate look-up table for color adjustment
   * @param[in]src_img: source image to be adjusted, channels must be same as ref_img
   * @param[in]adjust_rate: adjust rate range from 0 to 1, 0 means not adjust, 1 means fully adjust
   * @param[in]src_mask: if set, only pixels from mask area will be used to calculate LUT
   * @return look-up table, size [N][256], N is src image channels
   */
  LutType calcAdjustLut(const cv::Mat& src_img, float adjust_rate = 1.0f, const cv::Mat& src_mask = cv::Mat());

  /** @brief adjust image
   * @param[in]src_img: source image to be adjusted, channels must be same as ref_img
   * @param[in]adjust_rate: adjust rate range from 0 to 1, 0 means not adjust, 1 means fully adjust
   * @param[in]src_mask: if set, only pixels from mask area will be used to calculate LUT
   * @return adjusted image, whose size is the same as src_img
   */
  cv::Mat adjust(const cv::Mat& src_img, float adjust_rate = 1.0f, const cv::Mat& src_mask = cv::Mat());

  /** @brief set prior histogram */
  void setPriorHist(const HistogramType& prior_hist) { m_prior_hist = prior_hist; }

  /** @brief get prior histogram */
  HistogramType getPriotHist() const { return m_prior_hist; }

  /** @brief set inner resize */
  void setResizeWh(const cv::Size& resize_wh) { m_resize_wh = resize_wh; }

  /** @brief get inner resize */
  cv::Size getResizeWh() const { return m_resize_wh; }

 private:
  cv::Size m_resize_wh;          ///< downsample source image before calculate histogram for acceleration
  HistogramType m_ref_hist_buf;  ///< histogram of reference image
  HistogramType m_ref_cdf_buf;   ///< CDF(Cumulative Distribution Function) of reference image
  HistogramType m_prior_hist;    ///< prior histogram
};

/** @brief guided filter
 * ref: He K, et. "Guided image filtering" ECCV(2010)
 */
class GuidedFilter {
 public:
  /** @brief constructor
   * @param[in]ref_image: input guided/reference image
   * @param[in]kernel_size: local window size
   * @param[in]eps: regularization parameter
   * */
  GuidedFilter(const cv::Mat& ref_image, int kernel_size, float eps = 1e-4);

  /** @brief filter
   * @param[in]input_img: target image
   * @return filtered image
   * */
  cv::Mat filter(const cv::Mat& input_img) const;

 private:
  cv::Size blur_size_;
  float eps_;
  std::vector<cv::Mat> Is_;
  std::vector<cv::Mat> mean_Is_;
  std::vector<cv::Mat> covs_;

  cv::Mat filterOneChannel(const cv::Mat& P) const;

  cv::Mat blur(const cv::Mat& in) const;

  /** @brief convert to CV_32FC(n)
   * @param[in]in: input image
   * @param[in]out: output image in CV_32FC(n)
   * @return convert alpha
   */
  double convertToFloat(const cv::Mat& in, cv::Mat& out, bool deep_copy = false) const;
};

/** @brief guided filter
 * @ref He K, et. "Guided image filtering" ECCV(2010)
 * @param[in]ref_img: input guided/reference image
 * @param[in]input_img: input mask image
 * @param[in]r: local window size
 * @param[in]eps: regularization parameter
 * @param[in]depth: image channel depth
 * @return filtered image
 * */
cv::Mat guidedFilter(const cv::Mat& ref_img, const cv::Mat& input_img, int r, float eps);

} /* namespace vs */