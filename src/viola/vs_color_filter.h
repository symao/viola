/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details label image to mask with specific color such as:red, yellow, white.
 */
#pragma once
#include <memory>
#include <vector>
#include <opencv2/core.hpp>

namespace vs {

class ColorModel;
typedef std::shared_ptr<ColorModel> ColorModelPtr;
typedef std::vector<ColorModelPtr> ColorModelList;
class ColorModel {
 public:
  enum ProcessType {
    BGR = 0,  ///< process color in BGR space
    RGB = 1,  ///< process color in RGB space
    HSV = 2,  ///< process color in HSV space
  };

  /** @brief Constructor
   * @param[in]type: process image type, support BGR,RGB,HSV
   */
  ColorModel(int type = 0);

  virtual ~ColorModel() = default;

  /** @brief get process type */
  int type() const { return type_; }

  /** @brief filter input image into color mask
   * @param[in]input: input CV_8UC3 image, type must be same as type in constructor
   * @param[out]mask: output color mask, CV_8UC1
   */
  void filter(const cv::Mat& input, cv::Mat& mask) const;

  static ColorModelPtr red();

  static ColorModelPtr green();

  static ColorModelPtr blue();

  static ColorModelPtr black();

  static ColorModelPtr white();

  static ColorModelPtr yellow();

 protected:
  int type_;  ///< process type @see ProcessType

  /** @brief implementation of judging color
   * @param[in]a: input pixel data
   * @return output mask value, [0,255]
   */
  virtual uchar judge(const cv::Vec3b& a) const = 0;
};

class ColorModelRange : public ColorModel {
 public:
  typedef std::pair<uchar, uchar> ColorRange;
  typedef std::vector<ColorRange> RangeList;

  ColorModelRange(int type, const RangeList& range_list0, const RangeList& range_list1, const RangeList& range_list2)
      : ColorModel(type) {
    range_lists_[0] = range_list0;
    range_lists_[1] = range_list1;
    range_lists_[2] = range_list2;
  }

  virtual uchar judge(const cv::Vec3b& a) const {
    return check(a[0], range_lists_[0]) && check(a[1], range_lists_[1]) && check(a[2], range_lists_[2]) ? 255 : 0;
  }

 private:
  RangeList range_lists_[3];

  bool check(uchar v, const RangeList& range_list) const {
    for (const auto& it : range_list) {
      if (it.first <= v && v <= it.second) return true;
    }
    return false;
  }
};

enum ColorFilterPostMethod {
  CFPM_MORPHOLOGY = 1 << 0,
  CFPM_FLOODFILL = 1 << 1,
  CFPM_SPECKLE = 1 << 2,
};

/** @brief color filter, input image and output color mask
 * @param[in]img_bgr: input CV_8UC3 BGR image
 * @param[out]mask: output color mask CV_8UC1
 * @param[in]model_list: color model list
 * @param[in]resize_rate: whether resize small for acceleration, if resize_rate<=0, not resize.
 * @param[in]post_process: post process option, CFPM_MORPHOLOGY|CFPM_FLOODFILL|CFPM_SPECKLE
 */
void colorFilter(const cv::Mat& img_bgr, cv::Mat& mask, const ColorModelList& model_list, float resize_rate = -1,
                 int post_process = 0);

} /* namespace vs */