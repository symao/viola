/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-01-14 10:32
 * @details
 */
#include "vs_imtrans.h"
#include "vs_geometry2d.h"
#include "vs_improc.h"
#include "vs_tictoc.h"

namespace vs {

class ImageTransformBlur : public ImageTransformApi {
 public:
  enum BlurType {
    BLUR = 0,
    GAUSSIAN_BLUR = 1,
    MEDIAN_BLUR = 2,
  };

  ImageTransformBlur(int blur_type, const cv::Size& kernel_size, double sigma_x = 1, double sigma_y = 0)
      : blur_type_(blur_type), kernel_size_(kernel_size), sigma_x_(sigma_x), sigma_y_(sigma_y) {}

  virtual void process(cv::Mat& mask, const cv::Mat& reference = cv::Mat()) {
    switch (blur_type_) {
      case BLUR:
        cv::blur(mask, mask, kernel_size_);
        break;
      case GAUSSIAN_BLUR:
        cv::GaussianBlur(mask, mask, kernel_size_, sigma_x_, sigma_y_);
        break;
      case MEDIAN_BLUR:
        cv::medianBlur(mask, mask, kernel_size_.width);
    }
  }

 private:
  int blur_type_;
  cv::Size kernel_size_;
  double sigma_x_, sigma_y_;
};

ImageTransformHandler ImageTransformer::blur(const cv::Size& kernel_size) {
  return std::make_shared<ImageTransformBlur>(ImageTransformBlur::BLUR, kernel_size);
}

ImageTransformHandler ImageTransformer::gaussianBlur(const cv::Size& kernel_size, double sigma_x, double sigma_y) {
  return std::make_shared<ImageTransformBlur>(ImageTransformBlur::GAUSSIAN_BLUR, kernel_size, sigma_x, sigma_y);
}

class ImageTransformGuidedFilter : public ImageTransformApi {
 public:
  ImageTransformGuidedFilter(int radius, double eps) : radius_(radius), eps_(eps) {}

  virtual void process(cv::Mat& mask, const cv::Mat& reference = cv::Mat()) {
    mask = GuidedFilter(reference, radius_, eps_).filter(mask);
  }

 private:
  int radius_;
  float eps_;
};

ImageTransformHandler ImageTransformer::guidedFilter(int radius, float eps) {
  return std::make_shared<ImageTransformGuidedFilter>(radius, eps);
}

class ImageTransformGuidedFilterSmooth : public ImageTransformApi {
 public:
  ImageTransformGuidedFilterSmooth(int radius, double eps, cv::Mat previous_mask)
      : radius_(radius), eps_(eps), previous_mask_(previous_mask) {}

  virtual void process(cv::Mat& mask, const cv::Mat& reference = cv::Mat()) {
    cv::Mat filter_mask =
        previous_mask_.empty() ? mask.clone() : GuidedFilter(previous_mask_, radius_, eps_).filter(mask);
    previous_mask_ = filter_mask;

    if (mask.depth() == CV_8U) {
      cv::Mat m1, m2;
      mask.convertTo(m1, CV_32FC1, 1.0f / 255);
      filter_mask.convertTo(m2, CV_32FC1, 1.0f / 255);
      cv::multiply(m1, m2, m1);
      m1.convertTo(mask, CV_8U, 255);
    } else if (mask.depth() == CV_32F || mask.depth() == CV_64F) {
      cv::multiply(mask, previous_mask_, mask);
    }
  }

 private:
  int radius_;
  float eps_;
  cv::Mat previous_mask_;
};

ImageTransformHandler ImageTransformer::guidedFilterSmooth(int radius, float eps, cv::Mat previous_mask) {
  return std::make_shared<ImageTransformGuidedFilterSmooth>(radius, eps, previous_mask);
}

class ImageTransformMorph : public ImageTransformApi {
 public:
  enum MorphType { MORPH_ERODE = 0, MORPH_DILATE = 1 };
  ImageTransformMorph(int morph_type, const cv::Size& kernel_size, int shape)
      : morph_type_(morph_type), kernel_(cv::getStructuringElement(shape, kernel_size)) {}

  virtual void process(cv::Mat& mask, const cv::Mat& reference = cv::Mat()) {
    switch (morph_type_) {
      case MORPH_ERODE:
        cv::erode(mask, mask, kernel_);
        break;
      case MORPH_DILATE:
        cv::dilate(mask, mask, kernel_);
        break;
    }
  }

 private:
  int morph_type_;
  cv::Mat kernel_;
};

ImageTransformHandler ImageTransformer::dilate(const cv::Size& kernel_size, int shape) {
  return std::make_shared<ImageTransformMorph>(ImageTransformMorph::MORPH_DILATE, kernel_size, shape);
}

ImageTransformHandler ImageTransformer::erode(const cv::Size& kernel_size, int shape) {
  return std::make_shared<ImageTransformMorph>(ImageTransformMorph::MORPH_ERODE, kernel_size, shape);
}

class ImageTransformTrunc : public ImageTransformApi {
 public:
  ImageTransformTrunc(float thres_min, float thres_max, float min_value, float max_value)
      : thres_min_(thres_min), thres_max_(thres_max), min_value_(min_value), max_value_(max_value) {}

  virtual void process(cv::Mat& mask, const cv::Mat& reference = cv::Mat()) {
    if (mask.depth() == CV_8U) {
      run<uchar>(mask);
    } else if (mask.depth() == CV_32F) {
      run<float>(mask);
    } else if (mask.depth() == CV_64F) {
      run<double>(mask);
    }
  }

 private:
  float thres_min_;
  float thres_max_;
  float min_value_;
  float max_value_;

  template <typename T>
  void run(cv::Mat& mask) {
    int col_channel = mask.cols * mask.channels();
    for (int i = 0; i < mask.rows; i++) {
      T* ptr = mask.ptr<T>(i);
      for (int j = 0; j < col_channel; j++) {
        auto& v = *ptr++;
        if (v < thres_min_)
          v = min_value_;
        else if (v > thres_max_)
          v = max_value_;
      }
    }
  };
};

ImageTransformHandler ImageTransformer::trunc(float thres_min, float thres_max, float min_value, float max_value) {
  return std::make_shared<ImageTransformTrunc>(thres_min, thres_max, min_value, max_value);
}

class ImageTransformThreshold : public ImageTransformApi {
 public:
  ImageTransformThreshold(double thresh, double maxval, int type) : thresh_(thresh), maxval_(maxval), type_(type) {}

  virtual void process(cv::Mat& mask, const cv::Mat& reference = cv::Mat()) {
    cv::threshold(mask, mask, thresh_, maxval_, type_);
  }

 private:
  double thresh_;
  double maxval_;
  int type_;
};

ImageTransformHandler ImageTransformer::threshold(double thresh, double maxval, int type) {
  return std::make_shared<ImageTransformThreshold>(thresh, maxval, type);
}

class ImageTransformConvertTo : public ImageTransformApi {
 public:
  ImageTransformConvertTo(int type, double alpha, double beta) : type_(type), alpha_(alpha), beta_(beta) {}

  virtual void process(cv::Mat& mask, const cv::Mat& reference = cv::Mat()) {
    mask.convertTo(mask, type_, alpha_, beta_);
  }

 private:
  int type_;
  double alpha_, beta_;
};

ImageTransformHandler ImageTransformer::convertTo(int type, double alpha, double beta) {
  return std::make_shared<ImageTransformConvertTo>(type, alpha, beta);
}

class ImageTransformRotate : public ImageTransformApi {
 public:
  ImageTransformRotate(float rot_rad, float dead_zone) : rot_rad_(rot_rad), dead_zone_(dead_zone) {}

  virtual void process(cv::Mat& mask, const cv::Mat& reference = cv::Mat()) {
    FastImageRoter roter(mask.size(), rot_rad_, dead_zone_);
    roter.rot(mask, mask);
  }

 private:
  float rot_rad_;
  float dead_zone_;
};

ImageTransformHandler ImageTransformer::rotate(double rot_rad, float dead_zone) {
  return std::make_shared<ImageTransformRotate>(rot_rad, dead_zone);
}

class ImageTransformSpeckleFilter : public ImageTransformApi {
 public:
  ImageTransformSpeckleFilter(double new_val, int max_speckle_size, double max_diff)
      : new_val_(new_val), max_speckle_size_(max_speckle_size), max_diff_(max_diff) {}

  virtual void process(cv::Mat& mask, const cv::Mat& reference = cv::Mat()) {
    cv::filterSpeckles(mask, new_val_, max_speckle_size_, max_diff_);
  }

 private:
  double new_val_;
  int max_speckle_size_;
  double max_diff_;
};

ImageTransformHandler ImageTransformer::speckleFilter(double new_val, int max_speckle_size, double max_diff) {
  return std::make_shared<ImageTransformSpeckleFilter>(new_val, max_speckle_size, max_diff);
}

class ImageTransformConnectComponentFilter : public ImageTransformApi {
 public:
  ImageTransformConnectComponentFilter(double bw_thres, float min_area_ratio, int max_k)
      : bw_thres_(bw_thres), min_area_ratio_(min_area_ratio), max_k_(max_k) {}

  virtual void process(cv::Mat& mask, const cv::Mat& reference = cv::Mat()) {
    connectedComponentFilter(mask, bw_thres_, min_area_ratio_, max_k_);
  }

 private:
  double bw_thres_;
  float min_area_ratio_;
  int max_k_;
};

ImageTransformHandler ImageTransformer::connectComponentFilter(double bw_thres, float min_area_ratio, int max_k) {
  return std::make_shared<ImageTransformConnectComponentFilter>(bw_thres, min_area_ratio, max_k);
}

std::vector<float> ImageTransformer::runAndPerf(cv::Mat& mask, const cv::Mat& reference) {
  std::vector<float> costms;
  for (auto& opt : options_) {
    Timer t1;
    opt->process(mask, reference);
    costms.push_back(t1.stop());
  }
  return costms;
}

}  // namespace vs