/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-01-14 10:32
 * @details universal image transformation for image processing
 */
#pragma once
#include <vector>
#include <memory>
#include <opencv2/imgproc.hpp>

namespace vs {

/** @brief Api for image transformation which define the input and output */
class ImageTransformApi {
 public:
  virtual ~ImageTransformApi() = default;

  /** @brief common api for image transformation
   * @param[in,out]mask: input/output mask to be transformation
   * @param[in]reference: reference image if needed, such as guided filter
   */
  virtual void process(cv::Mat& mask, const cv::Mat& reference = cv::Mat()) = 0;
};
typedef std::shared_ptr<ImageTransformApi> ImageTransformHandler;

/** @brief Image post processer
 * provide lots of transformation handler such as blur, guided filter, resize, morph
 * user can implenment own transformation handler which derive from ImageTransformApi
 */
class ImageTransformer {
 public:
  /** @brief Constructor
   * @param[in]options: transformation options in order
   */
  ImageTransformer(const std::vector<ImageTransformHandler>& options = {}) { setOptions(options); }

  /** @brief set transformation options
   * @param[in]options: transformation options in order
   */
  void setOptions(const std::vector<ImageTransformHandler>& options) { options_ = options; }

  /** @brief get transformation options */
  std::vector<ImageTransformHandler> getOptions() const { return options_; }

  /** @brief run all transformation options sequentially
   * @param[in,out]mask: input/output mask to be transformation
   * @param[in]reference: reference image used by some option, such as guided filter
   */
  void run(cv::Mat& mask, const cv::Mat& reference = cv::Mat()) {
    for (auto& opt : options_) opt->process(mask, reference);
  }

  /** @brief run all transformation options sequentially and output time cost
   * @param[in,out]mask: input/output mask to be transformation
   * @param[in]reference: reference image used by some option, such as guided filter
   * @return time cost in miliseconds, same size as options size
   */
  std::vector<float> runAndPerf(cv::Mat& mask, const cv::Mat& reference = cv::Mat());

  /** @brief handler of cv::blur */
  static ImageTransformHandler blur(const cv::Size& kernel_size);

  /** @brief handler of guider filter */
  static ImageTransformHandler guidedFilter(int radius, float eps = 1e-4);

  /** @brief handler of guider filter with previous mask smooth */
  static ImageTransformHandler guidedFilterSmooth(int radius, float eps = 1e-4, cv::Mat init_mask = cv::Mat());

  /** @brief handler of cv::GaussianBlur */
  static ImageTransformHandler gaussianBlur(const cv::Size& kernel_size, double sigma_x = 1, double sigma_y = 0);

  /** @brief handler of cv::dilate */
  static ImageTransformHandler dilate(const cv::Size& kernel_size, int shape = cv::MORPH_RECT);

  /** @brief handler of cv::erode */
  static ImageTransformHandler erode(const cv::Size& kernel_size, int shape = cv::MORPH_RECT);

  /** @brief handler of truncate, if v < thres_min, v = min_value, if v > thres_max, v = max_value */
  static ImageTransformHandler trunc(float thres_min, float thres_max, float min_value = 0, float max_value = 1);

  /** @brief handler of cv::threshold */
  static ImageTransformHandler threshold(double thresh, double maxval, int type = cv::THRESH_BINARY);

  /** @brief handler of cv::convertTo */
  static ImageTransformHandler convertTo(int type, double alpha = 1, double beta = 0);

  /** @brief handler of rotate image, see vs::FastImageRoter */
  static ImageTransformHandler rotate(double rot_rad, float dead_zone = 0.2f);

  /** @brief handler of speckle filter, see cv::filterSpeckles */
  static ImageTransformHandler speckleFilter(double new_val, int max_speckle_size, double max_diff);

  /** @brief handler of connect component filter
   * @param[in]bw_thres: threshold to convert to binary image, which used to find connect components
   * @param[in]min_area_ratio: min valid connect component area / whole image area
   * @param[in]max_k: select top-K area connect components
   */
  static ImageTransformHandler connectComponentFilter(double bw_thres, float min_area_ratio = 0.1f, int max_k = 1);

 private:
  std::vector<ImageTransformHandler> options_;  ///< transformation handles
};

}  // namespace vs