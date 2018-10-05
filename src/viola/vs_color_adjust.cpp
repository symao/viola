/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-03-04 11:37
 * @details
 */
#include "vs_color_adjust.h"
#include "vs_basic.h"
#include "vs_geometry2d.h"
#include "vs_debug_draw.h"
#include "vs_improc.h"
namespace vs {

static bool santyCheck(const cv::Mat& img, int ref_channels, const cv::Mat& mask) {
  if (ref_channels == 0) {
    printf("[ERROR]ColorAdjustor: adjust failed, set reference image first.\n");
    return false;
  } else if (img.channels() != ref_channels) {
    printf("[ERROR]ColorAdjustor: adjust image channel(%d) not same as reference image channel(%d)\n", img.channels(),
           ref_channels);
    return false;
  } else if (img.depth() != CV_8U) {
    printf("[ERROR]ColorAdjustor: adjust image depth(%d) is not CV_8U\n", img.depth());
    return false;
  } else if (!mask.empty()) {
    if (mask.size() != img.size()) {
      printf("[ERROR]ColorAdjustor: mask size(%dx%d) not same as image size(%dx%d).\n", mask.cols, mask.rows, img.cols,
             img.rows);
      return false;
    } else if (mask.channels() != 1 || mask.depth() != CV_8U) {
      printf("[ERROR]ColorAdjustor: mask only support CV_8UC1, while input mask channels:%d depth:%d\n",
             mask.channels(), mask.depth());
      return false;
    }
  }
  return true;
}

static ColorAdjustor::LutType defaultLuts(int n) {
  std::vector<uchar> a(256);
  for (int i = 0; i < 256; i++) a[i] = i;
  return ColorAdjustor::LutType(n, a);
}

// 3x256 historgram in R-G-B order, which is the statistic of skin segmentation data
static HistogramMatch::HistogramType p_prior_skin_histogram_rgb = {
    {0.000272, 0.000018, 0.000017, 0.000020, 0.000023, 0.000022, 0.000026, 0.000028, 0.000031, 0.000034, 0.000040,
     0.000045, 0.000050, 0.000055, 0.000062, 0.000068, 0.000073, 0.000078, 0.000084, 0.000090, 0.000099, 0.000108,
     0.000117, 0.000126, 0.000137, 0.000148, 0.000160, 0.000173, 0.000195, 0.000202, 0.000213, 0.000228, 0.000239,
     0.000251, 0.000266, 0.000281, 0.000297, 0.000309, 0.000322, 0.000334, 0.000347, 0.000363, 0.000379, 0.000395,
     0.000412, 0.000427, 0.000443, 0.000458, 0.000469, 0.000484, 0.000500, 0.000515, 0.000535, 0.000557, 0.000577,
     0.000596, 0.000615, 0.000635, 0.000654, 0.000678, 0.000697, 0.000721, 0.000745, 0.000767, 0.000791, 0.000816,
     0.000845, 0.000873, 0.000898, 0.000921, 0.000944, 0.000971, 0.000997, 0.001026, 0.001056, 0.001090, 0.001126,
     0.001163, 0.001196, 0.001225, 0.001258, 0.001287, 0.001307, 0.001339, 0.001369, 0.001398, 0.001447, 0.001492,
     0.001537, 0.001583, 0.001630, 0.001677, 0.001715, 0.001759, 0.001800, 0.001835, 0.001863, 0.001890, 0.001921,
     0.001958, 0.002004, 0.002054, 0.002109, 0.002163, 0.002218, 0.002267, 0.002312, 0.002357, 0.002404, 0.002444,
     0.002478, 0.002504, 0.002535, 0.002572, 0.002614, 0.002663, 0.002722, 0.002784, 0.002841, 0.002898, 0.002950,
     0.003002, 0.003055, 0.003104, 0.003147, 0.003187, 0.003231, 0.003275, 0.003340, 0.003366, 0.003416, 0.003473,
     0.003524, 0.003574, 0.003627, 0.003688, 0.003755, 0.003821, 0.003882, 0.003946, 0.004013, 0.004080, 0.004142,
     0.004203, 0.004265, 0.004332, 0.004398, 0.004457, 0.004522, 0.004587, 0.004655, 0.004719, 0.004772, 0.004835,
     0.004902, 0.004975, 0.005046, 0.005110, 0.005180, 0.005255, 0.005323, 0.005389, 0.005457, 0.005537, 0.005619,
     0.005692, 0.005765, 0.005855, 0.005949, 0.006034, 0.006111, 0.006193, 0.006286, 0.006386, 0.006475, 0.006554,
     0.006637, 0.006724, 0.006830, 0.006919, 0.006987, 0.007070, 0.007166, 0.007264, 0.007342, 0.007428, 0.007524,
     0.007629, 0.007718, 0.007799, 0.007886, 0.007987, 0.008102, 0.008202, 0.008288, 0.008403, 0.008532, 0.008645,
     0.008736, 0.008840, 0.008939, 0.009041, 0.009116, 0.009178, 0.009243, 0.009320, 0.009419, 0.009476, 0.009489,
     0.009526, 0.009591, 0.009633, 0.009642, 0.009663, 0.009702, 0.009748, 0.009762, 0.009753, 0.009785, 0.009812,
     0.009772, 0.009714, 0.009639, 0.009576, 0.009521, 0.009395, 0.009217, 0.009052, 0.008908, 0.008797, 0.008646,
     0.008456, 0.008292, 0.008153, 0.007903, 0.007617, 0.007348, 0.007111, 0.006898, 0.006633, 0.006303, 0.005999,
     0.005728, 0.005452, 0.005143, 0.004827, 0.004548, 0.004243, 0.003968, 0.003725, 0.003534, 0.003469, 0.003554,
     0.004100, 0.005511, 0.014056},
    {0.000253, 0.000076, 0.000069, 0.000068, 0.000078, 0.000091, 0.000109, 0.000127, 0.000133, 0.000147, 0.000166,
     0.000187, 0.000213, 0.000240, 0.000267, 0.000306, 0.000336, 0.000358, 0.000389, 0.000424, 0.000456, 0.000491,
     0.000530, 0.000572, 0.000610, 0.000651, 0.000692, 0.000737, 0.000793, 0.000830, 0.000874, 0.000919, 0.000964,
     0.001008, 0.001050, 0.001092, 0.001134, 0.001180, 0.001227, 0.001274, 0.001321, 0.001369, 0.001414, 0.001462,
     0.001511, 0.001561, 0.001612, 0.001666, 0.001723, 0.001777, 0.001832, 0.001888, 0.001937, 0.001988, 0.002041,
     0.002097, 0.002161, 0.002223, 0.002284, 0.002341, 0.002401, 0.002461, 0.002519, 0.002578, 0.002637, 0.002690,
     0.002745, 0.002800, 0.002849, 0.002907, 0.002964, 0.003027, 0.003090, 0.003159, 0.003231, 0.003289, 0.003352,
     0.003416, 0.003479, 0.003544, 0.003604, 0.003669, 0.003733, 0.003795, 0.003860, 0.003914, 0.003970, 0.004031,
     0.004087, 0.004141, 0.004201, 0.004267, 0.004331, 0.004376, 0.004438, 0.004504, 0.004572, 0.004647, 0.004706,
     0.004773, 0.004827, 0.004881, 0.004943, 0.004993, 0.005053, 0.005114, 0.005179, 0.005257, 0.005316, 0.005383,
     0.005441, 0.005503, 0.005570, 0.005624, 0.005683, 0.005748, 0.005786, 0.005831, 0.005888, 0.005952, 0.006016,
     0.006069, 0.006126, 0.006173, 0.006237, 0.006310, 0.006353, 0.006404, 0.006483, 0.006534, 0.006602, 0.006634,
     0.006671, 0.006711, 0.006757, 0.006808, 0.006835, 0.006883, 0.006952, 0.006982, 0.007031, 0.007081, 0.007126,
     0.007180, 0.007198, 0.007239, 0.007289, 0.007348, 0.007406, 0.007421, 0.007466, 0.007504, 0.007546, 0.007585,
     0.007594, 0.007623, 0.007661, 0.007697, 0.007740, 0.007745, 0.007756, 0.007777, 0.007785, 0.007800, 0.007787,
     0.007790, 0.007812, 0.007788, 0.007795, 0.007812, 0.007829, 0.007844, 0.007811, 0.007785, 0.007761, 0.007738,
     0.007722, 0.007660, 0.007603, 0.007550, 0.007509, 0.007462, 0.007376, 0.007321, 0.007277, 0.007224, 0.007163,
     0.007063, 0.006983, 0.006916, 0.006787, 0.006681, 0.006582, 0.006480, 0.006394, 0.006261, 0.006130, 0.006005,
     0.005889, 0.005770, 0.005600, 0.005470, 0.005328, 0.005182, 0.005049, 0.004869, 0.004713, 0.004556, 0.004403,
     0.004253, 0.004084, 0.003918, 0.003761, 0.003572, 0.003401, 0.003245, 0.003128, 0.003006, 0.002862, 0.002732,
     0.002607, 0.002468, 0.002342, 0.002204, 0.002069, 0.001942, 0.001823, 0.001728, 0.001621, 0.001520, 0.001400,
     0.001309, 0.001221, 0.001133, 0.001062, 0.000992, 0.000919, 0.000858, 0.000799, 0.000738, 0.000691, 0.000642,
     0.000593, 0.000547, 0.000510, 0.000469, 0.000434, 0.000413, 0.000398, 0.000390, 0.000393, 0.000388, 0.000379,
     0.000395, 0.000580, 0.001335},
    {0.000685, 0.000182, 0.000190, 0.000196, 0.000220, 0.000245, 0.000275, 0.000315, 0.000359, 0.000411, 0.000459,
     0.000516, 0.000574, 0.000633, 0.000707, 0.000768, 0.000843, 0.000900, 0.000979, 0.001055, 0.001125, 0.001208,
     0.001269, 0.001342, 0.001414, 0.001490, 0.001567, 0.001642, 0.001720, 0.001778, 0.001842, 0.001907, 0.001976,
     0.002035, 0.002098, 0.002160, 0.002223, 0.002286, 0.002351, 0.002419, 0.002487, 0.002554, 0.002628, 0.002697,
     0.002766, 0.002837, 0.002910, 0.002980, 0.003055, 0.003129, 0.003198, 0.003269, 0.003340, 0.003416, 0.003484,
     0.003548, 0.003615, 0.003685, 0.003763, 0.003842, 0.003919, 0.003984, 0.004055, 0.004123, 0.004188, 0.004255,
     0.004316, 0.004371, 0.004421, 0.004480, 0.004541, 0.004605, 0.004663, 0.004717, 0.004776, 0.004839, 0.004908,
     0.004976, 0.005033, 0.005092, 0.005149, 0.005211, 0.005275, 0.005339, 0.005392, 0.005449, 0.005510, 0.005566,
     0.005626, 0.005672, 0.005712, 0.005763, 0.005819, 0.005883, 0.005933, 0.005977, 0.006023, 0.006071, 0.006125,
     0.006188, 0.006238, 0.006279, 0.006325, 0.006373, 0.006420, 0.006476, 0.006512, 0.006546, 0.006585, 0.006630,
     0.006675, 0.006717, 0.006742, 0.006772, 0.006813, 0.006850, 0.006888, 0.006914, 0.006924, 0.006935, 0.006963,
     0.006996, 0.007027, 0.007047, 0.007053, 0.007073, 0.007088, 0.007119, 0.007151, 0.007141, 0.007146, 0.007133,
     0.007136, 0.007136, 0.007124, 0.007108, 0.007099, 0.007094, 0.007089, 0.007094, 0.007081, 0.007057, 0.007024,
     0.007010, 0.006990, 0.006971, 0.006925, 0.006896, 0.006871, 0.006852, 0.006859, 0.006821, 0.006795, 0.006749,
     0.006727, 0.006703, 0.006691, 0.006674, 0.006633, 0.006597, 0.006566, 0.006533, 0.006482, 0.006424, 0.006368,
     0.006320, 0.006269, 0.006238, 0.006175, 0.006102, 0.006033, 0.005963, 0.005907, 0.005848, 0.005770, 0.005678,
     0.005587, 0.005522, 0.005449, 0.005372, 0.005284, 0.005185, 0.005092, 0.005010, 0.004935, 0.004843, 0.004722,
     0.004616, 0.004515, 0.004411, 0.004321, 0.004201, 0.004093, 0.003969, 0.003856, 0.003760, 0.003650, 0.003557,
     0.003434, 0.003327, 0.003227, 0.003127, 0.003017, 0.002895, 0.002777, 0.002680, 0.002581, 0.002485, 0.002386,
     0.002281, 0.002176, 0.002089, 0.002006, 0.001927, 0.001837, 0.001748, 0.001665, 0.001588, 0.001508, 0.001426,
     0.001342, 0.001266, 0.001201, 0.001146, 0.001090, 0.001025, 0.000964, 0.000921, 0.000857, 0.000814, 0.000763,
     0.000726, 0.000679, 0.000650, 0.000611, 0.000579, 0.000545, 0.000526, 0.000471, 0.000473, 0.000420, 0.000430,
     0.000400, 0.000360, 0.000325, 0.000333, 0.000291, 0.000312, 0.000268, 0.000285, 0.000249, 0.000247, 0.000210,
     0.000305, 0.000223, 0.001000}};
static HistogramMatch::HistogramType p_prior_skin_histogram_bgr = vecReverse(p_prior_skin_histogram_rgb);

cv::Mat ColorAdjustor::adjust(const cv::Mat& src_img, float adjust_rate, const cv::Mat& src_mask) {
  if (adjust_rate <= 0) return src_img;
  if (src_mask.size() == src_img.size()) {
    cv::Mat process_img, process_mask;
    cv::Rect roi = cropMaskRoi(src_img, src_mask, process_img, process_mask);
    auto luts = calcAdjustLut(process_img, adjust_rate, process_mask);
    if (luts.empty()) return src_img;
    return m_cfg.adjust_mask_area_only ? applyLut(src_img, luts, roi) : applyLut(src_img, luts);
  } else {
    auto luts = calcAdjustLut(src_img, adjust_rate, src_mask);
    if (luts.empty()) return src_img;
    return applyLut(src_img, luts);
  }
}

cv::Mat ColorAdjustor::downsampleImg(const cv::Mat& img, const cv::Size& max_size) {
  if (img.empty()) return cv::Mat();
  cv::Mat out;
  if (img.size().area() > max_size.area()) {
    cv::resize(img, out, max_size, m_cfg.resize_interpolation_type);
  } else {
    out = img;
  }
  return out;
}

class ColorAdjustorHistogramMatch : public ColorAdjustor {
 public:
  virtual bool init(const cv::Mat& ref_img);

  virtual LutType calcAdjustLut(const cv::Mat& src_img, float adjust_rate, const cv::Mat& src_mask);

 private:
  std::shared_ptr<HistogramMatch> m_handle;
};

bool ColorAdjustorHistogramMatch::init(const cv::Mat& ref_img) {
  m_handle = std::make_shared<HistogramMatch>(downsampleImg(ref_img, m_cfg.process_ref_size));
  return true;
}

ColorAdjustor::LutType ColorAdjustorHistogramMatch::calcAdjustLut(const cv::Mat& src_img, float adjust_rate,
                                                                  const cv::Mat& src_mask) {
  if (m_handle.get()) {
    auto process_img = downsampleImg(src_img, m_cfg.process_src_size);
    auto process_mask = downsampleImg(src_mask, m_cfg.process_src_size);
    if (m_cfg.use_prior_skin_hist)
      m_handle->setPriorHist(m_cfg.input_rgb ? p_prior_skin_histogram_rgb : p_prior_skin_histogram_bgr);
    return m_handle->calcAdjustLut(process_img, adjust_rate, process_mask);
  }
  return {};
}

class ColorAdjustorStatisticMatch : public ColorAdjustor {
 public:
  virtual bool init(const cv::Mat& ref_img);

  virtual LutType calcAdjustLut(const cv::Mat& src_img, float adjust_rate, const cv::Mat& src_mask);

 private:
  const float m_low_rate = 0.01f;
  const float m_high_rate = 0.99f;
  const float m_bezier_k = 0.5f;
  const float m_max_lut_diff = 64.0f;
  std::vector<cv::Vec3f> m_ref_lmh;

  std::vector<cv::Vec3f> extractFeature(const cv::Mat& img, const cv::Mat& mask = cv::Mat());

  std::vector<cv::Vec3f> extractFeatureNew(const cv::Mat& img, const cv::Mat& mask = cv::Mat());

  std::vector<uchar> bezierLut(const cv::Point2f& p0, const cv::Point2f& p1, const cv::Point2f& p2, float k = 0);

  std::vector<uchar> calcLut(const cv::Vec3f& src, const cv::Vec3f& tar);
};

bool ColorAdjustorStatisticMatch::init(const cv::Mat& ref_img) {
  if (ref_img.empty()) return false;
  cv::Mat process_img = downsampleImg(ref_img, m_cfg.process_ref_size);
  m_ref_lmh = extractFeature(process_img);
  return true;
}

ColorAdjustor::LutType ColorAdjustorStatisticMatch::calcAdjustLut(const cv::Mat& src_img, float adjust_rate,
                                                                  const cv::Mat& src_mask) {
  if (adjust_rate == 0 || !santyCheck(src_img, m_ref_lmh.size(), src_mask)) return defaultLuts(src_img.channels());

  // downsample for speed
  cv::Mat process_img, process_mask;
  cropMaskRoi(src_img, src_mask, process_img, process_mask);
  process_img = downsampleImg(process_img, m_cfg.process_src_size);
  process_mask = downsampleImg(process_mask, m_cfg.process_src_size);

  // extract lmh feature
  auto cur_lmh_list = extractFeature(process_img);

  // calculate LUT for each channel
  ColorAdjustor::LutType luts(src_img.channels());
  for (int i = 0; i < src_img.channels(); i++) {
    const cv::Vec3f& lmh = cur_lmh_list[i];
    auto raw_lut = calcLut(lmh, m_ref_lmh[i]);
    auto& lut = luts[i];
    lut.reserve(raw_lut.size());
    for (size_t j = 0; j < raw_lut.size(); j++)
      lut.push_back(
          clip(j + clip(static_cast<float>(raw_lut[j]) - j, -m_max_lut_diff, m_max_lut_diff) * adjust_rate, 0, 255));
  }
  return luts;
}

std::vector<cv::Vec3f> ColorAdjustorStatisticMatch::extractFeature(const cv::Mat& img, const cv::Mat& mask) {
  // if (img.type() == CV_8UC3) return extractFeatureNew(img, mask);

  // find low,middle,high part in R,G,B channel separately
  std::vector<cv::Vec3f> result;
  auto img_split = imsplit(img);
  if (img_split.empty()) return result;

  // find all rgb values into data_list for later sort
  std::vector<std::vector<uint8_t>> data_list(img_split.size());
  for (size_t ch = 0; ch < img_split.size(); ch++) {
    auto& data = data_list[ch];
    const auto& img_slice = img_split[ch];
    data.reserve(img_slice.cols * img_slice.rows);
    if (mask.empty()) {
      for (int i = 0; i < img_slice.rows; i++) {
        const uchar* ptr = img_slice.ptr<uchar>(i);
        for (int j = 0; j < img_slice.cols; j++) data.push_back(*ptr++);
      }
    } else {
      for (int i = 0; i < img_slice.rows; i++) {
        const uchar* ptr_img = img_slice.ptr<uchar>(i);
        const uchar* ptr_mask = mask.ptr<uchar>(i);
        for (int j = 0; j < img_slice.cols; j++, ptr_img++, ptr_mask++) {
          if (*ptr_mask != 0) data.push_back(*ptr_img);
        }
      }
    }
  }

  // find low, middle, high part in each channel, calculate part average
  result.reserve(img_split.size());
  for (auto& data : data_list) {
    int n = data.size();
    int low_idx = n * m_low_rate;
    int high_idx = n * m_high_rate;
    auto low_it = data.begin() + low_idx;
    auto high_it = data.begin() + high_idx;
    std::nth_element(data.begin(), low_it, data.end());
    std::nth_element(low_it, high_it, data.end());
    float low = std::accumulate(data.begin(), low_it, static_cast<int>(0)) / static_cast<float>(low_idx);
    float middle = std::accumulate(data.begin(), data.end(), static_cast<int>(0)) / static_cast<float>(n);
    float high = std::accumulate(high_it, data.end(), static_cast<int>(0)) / static_cast<float>(n - high_idx);
    result.push_back(cv::Vec3f(low, middle, high));
  }
  return result;
}

std::vector<cv::Vec3f> ColorAdjustorStatisticMatch::extractFeatureNew(const cv::Mat& img, const cv::Mat& mask) {
  // test the result is similar to extractFeature, while this new code is a little bit complicated, thus not used.
  // float coef[3] = {0.212671f, 0.715160f, 0.072169f};  // rgb order
  float coef[3] = {0.072169f, 0.715160f, 0.212671f};  // bgr order
  std::vector<cv::Vec3f> result;
  if (img.type() != CV_8UC3) return result;
  // find low,middle,high part in luminance, extract r,g,b values
  std::vector<std::pair<cv::Point, float>> luminance_list;
  luminance_list.reserve(img.cols * img.rows);
  if (mask.empty()) {
    for (int i = 0; i < img.rows; i++) {
      const cv::Vec3b* ptr_img = img.ptr<cv::Vec3b>(i);
      for (int j = 0; j < img.cols; j++, ptr_img++) {
        const cv::Vec3b& v = *ptr_img;
        float luminance = v[0] * coef[0] + v[1] * coef[1] + v[2] * coef[2];
        luminance_list.push_back(std::make_pair(cv::Point(j, i), luminance));
      }
    }
  } else {
    for (int i = 0; i < img.rows; i++) {
      const cv::Vec3b* ptr_img = img.ptr<cv::Vec3b>(i);
      const uchar* ptr_mask = mask.ptr<uchar>(i);
      for (int j = 0; j < img.cols; j++, ptr_img++, ptr_mask++) {
        if (*ptr_mask != 0) {
          const cv::Vec3b& v = *ptr_img;
          float luminance = v[0] * coef[0] + v[1] * coef[1] + v[2] * coef[2];
          luminance_list.push_back(std::make_pair(cv::Point(j, i), luminance));
        }
      }
    }
  }

  int n = luminance_list.size();
  int low_idx = n * m_low_rate;
  int high_idx = n * m_high_rate;
  auto low_it = luminance_list.begin() + low_idx;
  auto high_it = luminance_list.begin() + high_idx;
  auto foo_cmp = [](const std::pair<cv::Vec2i, float>& a, const std::pair<cv::Vec2i, float>& b) {
    return a.second < b.second;
  };
  std::nth_element(luminance_list.begin(), low_it, luminance_list.end(), foo_cmp);
  std::nth_element(low_it, high_it, luminance_list.end(), foo_cmp);

  result.resize(3, cv::Vec3f(0, 0, 0));
  for (auto it = luminance_list.begin(); it != low_it; it++) {
    const cv::Vec3b& v = img.at<cv::Vec3b>(it->first);
    for (int i = 0; i < 3; i++) result[i][0] += v[i];
  }
  for (auto it = luminance_list.begin(); it != luminance_list.end(); it++) {
    const cv::Vec3b& v = img.at<cv::Vec3b>(it->first);
    for (int i = 0; i < 3; i++) result[i][1] += v[i];
  }
  for (auto it = high_it; it != luminance_list.end(); it++) {
    const cv::Vec3b& v = img.at<cv::Vec3b>(it->first);
    for (int i = 0; i < 3; i++) result[i][2] += v[i];
  }
  for (int i = 0; i < 3; i++) {
    result[i][0] /= static_cast<float>(low_idx);
    result[i][1] /= static_cast<float>(n);
    result[i][2] /= static_cast<float>(n - high_idx);
  }
  return result;
}

std::vector<uchar> ColorAdjustorStatisticMatch::bezierLut(const cv::Point2f& p0, const cv::Point2f& p1,
                                                          const cv::Point2f& p2, float k) {
  std::vector<cv::Point2f> pts;
  if (k <= 0) {
    // naive bezier for p0~p1~p2
    pts = bezier2(p0, p1, p2, vecLinspace<float>(0, 1, 10));
  } else if (k >= 1) {
    // linear segment function for p0-p1-p2
    pts = {p0, p1, p2};
  } else {
    // blend naive bezier and linear function, p0-p01~p1~p12-p2, where p0-p01 and p12-p2 is linear function while
    // p01~p1~p12 is bezier. k range 0~1 control linear rate, the larger k, mapping curve is closer to p1.
    auto p01 = p1 * k + p0 * (1 - k);
    auto p12 = p1 * k + p2 * (1 - k);
    pts = bezier2(p01, p1, p12, vecLinspace<float>(0, 1, 10));
    pts.insert(pts.begin(), p0);
    pts.push_back(p2);
  }
  // linear interpolation with discrete points in mapping curves
  std::vector<uchar> ys;
  size_t idx = 0;
  for (int x = p0.x; x < p2.x; x++) {
    while (idx < pts.size() && pts[idx].x < x) idx++;
    const auto& p = pts[idx];
    if (x == p.x) {
      ys.push_back(p.y);
    } else {
      const auto& q = pts[idx - 1];
      ys.push_back(p.y + (q.y - p.y) / (q.x - p.x) * (x - p.x));
    }
  }
  return ys;
}

std::vector<uchar> ColorAdjustorStatisticMatch::calcLut(const cv::Vec3f& src, const cv::Vec3f& tar) {
  const auto soft_diff = m_max_lut_diff * 2;
  cv::Point2f p0(0, 0);
  cv::Point2f pl(src[0], clip(tar[0], src[0] - soft_diff, src[0] + soft_diff));
  cv::Point2f pm(src[1], clip(tar[1], src[1] - m_max_lut_diff, src[1] + m_max_lut_diff));
  cv::Point2f ph(src[2], clip(tar[2], src[2] - soft_diff, src[2] + soft_diff));
  cv::Point2f p1(255, 255);
  auto lut = bezierLut(p0, pl, pm, m_bezier_k);
  auto lut_high = bezierLut(pm, ph, p1, m_bezier_k);
  lut.insert(lut.end(), lut_high.begin(), lut_high.end());
  for (int i = lut.size(); i < 256; i++) lut.push_back(i);
  return lut;
}

std::shared_ptr<ColorAdjustor> createColorAdjustor(ColorAdjustorType type) {
  switch (type) {
    case COLOR_ADJUSTOR_HISTOGRAM_MATCH:
      return std::make_shared<ColorAdjustorHistogramMatch>();
    case COLOR_ADJUSTOR_STATISTIC_MATCH:
      return std::make_shared<ColorAdjustorStatisticMatch>();
    default:
      return nullptr;
  }
}

}  // namespace vs
