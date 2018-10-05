/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-03-04 11:37
 * @details Adjust image color style to reference image style, such as: histogram matching, color statistic matching.
 */
#pragma once
#include <memory>
#include <opencv2/imgproc.hpp>

namespace vs {

class ColorAdjustor {
 public:
  typedef std::vector<std::vector<uchar>> LutType;  ///< 1x256 or 3x256
  struct Config {
    bool input_rgb = false;                          ///< input image color order, true: R-G-B, false: B-G-R
    cv::Size process_ref_size = cv::Size(100, 100);  ///< max size to compute histogram/statistics from reference image
    cv::Size process_src_size = cv::Size(100, 100);  ///< max size to compute histogram/statistics from source image
    bool adjust_mask_area_only = false;  ///< only adjust mask rect area of ajusted image, other will be fill with 0
    int resize_interpolation_type = cv::INTER_NEAREST;  ///< parameter for resizing reference/source image

    bool use_prior_skin_hist = false;  ///< use prior skin histogram. Only used in COLOR_ADJUSTOR_HISTOGRAM_MATCH
  };

  ColorAdjustor() {}

  explicit ColorAdjustor(const Config& cfg) : m_cfg(cfg) {}

  virtual ~ColorAdjustor() {}

  /** @brief init reference image for color adjustment
   * @param[in]ref_img: reference image
   * @return whether init ok
   */
  virtual bool init(const cv::Mat& ref_img) = 0;

  /** @brief adjust color of input image with reference to init reference image
   * @param[in]src_img: source image to be adjust
   * @param[in]adjust_rate: adjust rate, range [0,1]
   * @param[in]src_mask: source mask
   * @return adjust image which is same size as src_img
   */
  virtual cv::Mat adjust(const cv::Mat& src_img, float adjust_rate = 0.5, const cv::Mat& src_mask = cv::Mat());

  /** @brief adjust color of input image with reference to init reference image
   * @param[in]src_img: source image to be adjust
   * @param[in]adjust_rate: adjust rate, range [0,1]
   * @param[in]src_mask: source mask
   * @return LUT nchannelx256
   */
  virtual LutType calcAdjustLut(const cv::Mat& src_img, float adjust_rate = 0.5,
                                const cv::Mat& src_mask = cv::Mat()) = 0;

  /** @brief set configuration */
  void setConfig(const Config& cfg) { m_cfg = cfg; }

  /** @brief get configuration */
  Config getConfig() const { return m_cfg; }

 protected:
  Config m_cfg;  ///< cofiguration parameters

  /** @brief downsample image into max size */
  cv::Mat downsampleImg(const cv::Mat& img, const cv::Size& max_size);
};

enum ColorAdjustorType {
  COLOR_ADJUSTOR_HISTOGRAM_MATCH = 0,  ///< color adjustment with histogram matching
  COLOR_ADJUSTOR_STATISTIC_MATCH = 1,  ///< color adjustment with image statistic matching
};

std::shared_ptr<ColorAdjustor> createColorAdjustor(ColorAdjustorType type = COLOR_ADJUSTOR_HISTOGRAM_MATCH);

}  // namespace vs