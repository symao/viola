/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_improc.h"

#include <functional>

#include "vs_basic.h"
#include "vs_geometry2d.h"
#include "vs_debug_draw.h"

namespace vs {

static cv::Mat grabBgr(const cv::Mat& bgr, const std::function<uchar(const cv::Vec3b& a)>& foo) {
  cv::Mat res(bgr.size(), CV_8UC1);
  for (int i = 0; i < bgr.rows; i++) {
    const cv::Vec3b* p_bgr = bgr.ptr<cv::Vec3b>(i);
    uchar* p_hue = res.ptr<uchar>(i);
    for (int j = 0; j < bgr.cols; j++) {
      *p_hue++ = foo(*p_bgr++);
    }
  }
  return res;
}

static auto foo_bgr2hue = [](const cv::Vec3b& a) {
  // for (int i = 1; i < 256; i++) hdiv_table[i] = cv::saturate_cast<int>((180 << 12) / (6. * i));
  static std::vector<int> hdiv_table = {
      122880, 61440, 40960, 30720, 24576, 20480, 17554, 15360, 13653, 12288, 11171, 10240, 9452, 8777, 8192, 7680, 7228,
      6827,   6467,  6144,  5851,  5585,  5343,  5120,  4915,  4726,  4551,  4389,  4237,  4096, 3964, 3840, 3724, 3614,
      3511,   3413,  3321,  3234,  3151,  3072,  2997,  2926,  2858,  2793,  2731,  2671,  2614, 2560, 2508, 2458, 2409,
      2363,   2318,  2276,  2234,  2194,  2156,  2119,  2083,  2048,  2014,  1982,  1950,  1920, 1890, 1862, 1834, 1807,
      1781,   1755,  1731,  1707,  1683,  1661,  1638,  1617,  1596,  1575,  1555,  1536,  1517, 1499, 1480, 1463, 1446,
      1429,   1412,  1396,  1381,  1365,  1350,  1336,  1321,  1307,  1293,  1280,  1267,  1254, 1241, 1229, 1217, 1205,
      1193,   1182,  1170,  1159,  1148,  1138,  1127,  1117,  1107,  1097,  1087,  1078,  1069, 1059, 1050, 1041, 1033,
      1024,   1016,  1007,  999,   991,   983,   975,   968,   960,   953,   945,   938,   931,  924,  917,  910,  904,
      897,    890,   884,   878,   871,   865,   859,   853,   847,   842,   836,   830,   825,  819,  814,  808,  803,
      798,    793,   788,   783,   778,   773,   768,   763,   759,   754,   749,   745,   740,  736,  731,  727,  723,
      719,    714,   710,   706,   702,   698,   694,   690,   686,   683,   679,   675,   671,  668,  664,  661,  657,
      654,    650,   647,   643,   640,   637,   633,   630,   627,   624,   621,   617,   614,  611,  608,  605,  602,
      599,    597,   594,   591,   588,   585,   582,   580,   577,   574,   572,   569,   566,  564,  561,  559,  556,
      554,    551,   549,   546,   544,   541,   539,   537,   534,   532,   530,   527,   525,  523,  521,  518,  516,
      514,    512,   510,   508,   506,   504,   502,   500,   497,   495,   493,   492,   490,  488,  486,  484,  482};
  int b = a[0];
  int g = a[1];
  int r = a[2];
  int vmin = min3(b, g, r);
  int v = max3(b, g, r);
  int diff = v - vmin;
  int vr = v == r ? -1 : 0;
  int vg = v == g ? -1 : 0;
  int h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
  h = (h * hdiv_table[diff] + (1 << 11)) >> 12;
  h += h < 0 ? 180 : 0;
  return cv::saturate_cast<uchar>(h);
};

static auto foo_bgr2saturation = [](const cv::Vec3b& a) {
  // for (int i = 1; i < 256; i++) sdiv_table[i] = cv::saturate_cast<int>((255 << 12) / (1. * i));
  static std::vector<int> sdiv_table = {
      1044480, 522240, 348160, 261120, 208896, 174080, 149211, 130560, 116053, 104448, 94953, 87040, 80345, 74606,
      69632,   65280,  61440,  58027,  54973,  52224,  49737,  47476,  45412,  43520,  41779, 40172, 38684, 37303,
      36017,   34816,  33693,  32640,  31651,  30720,  29842,  29013,  28229,  27486,  26782, 26112, 25475, 24869,
      24290,   23738,  23211,  22706,  22223,  21760,  21316,  20890,  20480,  20086,  19707, 19342, 18991, 18651,
      18324,   18008,  17703,  17408,  17123,  16846,  16579,  16320,  16069,  15825,  15589, 15360, 15137, 14921,
      14711,   14507,  14308,  14115,  13926,  13743,  13565,  13391,  13221,  13056,  12895, 12738, 12584, 12434,
      12288,   12145,  12006,  11869,  11736,  11605,  11478,  11353,  11231,  11111,  10995, 10880, 10768, 10658,
      10550,   10445,  10341,  10240,  10141,  10043,  9947,   9854,   9761,   9671,   9582,  9495,  9410,  9326,
      9243,    9162,   9082,   9004,   8927,   8852,   8777,   8704,   8632,   8561,   8492,  8423,  8356,  8290,
      8224,    8160,   8097,   8034,   7973,   7913,   7853,   7795,   7737,   7680,   7624,  7569,  7514,  7461,
      7408,    7355,   7304,   7253,   7203,   7154,   7105,   7057,   7010,   6963,   6917,  6872,  6827,  6782,
      6739,    6695,   6653,   6611,   6569,   6528,   6487,   6447,   6408,   6369,   6330,  6292,  6254,  6217,
      6180,    6144,   6108,   6073,   6037,   6003,   5968,   5935,   5901,   5868,   5835,  5803,  5771,  5739,
      5708,    5677,   5646,   5615,   5585,   5556,   5526,   5497,   5468,   5440,   5412,  5384,  5356,  5329,
      5302,    5275,   5249,   5222,   5196,   5171,   5145,   5120,   5095,   5070,   5046,  5022,  4998,  4974,
      4950,    4927,   4904,   4881,   4858,   4836,   4813,   4791,   4769,   4748,   4726,  4705,  4684,  4663,
      4642,    4622,   4601,   4581,   4561,   4541,   4522,   4502,   4483,   4464,   4445,  4426,  4407,  4389,
      4370,    4352,   4334,   4316,   4298,   4281,   4263,   4246,   4229,   4212,   4195,  4178,  4161,  4145,
      4128,    4112,   4096};
  int b = a[0];
  int g = a[1];
  int r = a[2];
  int vmin = min3(b, g, r);
  int v = max3(b, g, r);
  int diff = v - vmin;
  int s = (diff * sdiv_table[v] + (1 << (11))) >> 12;
  return (uchar)s;
};

cv::Mat bgr2hue(const cv::Mat& bgr) { return grabBgr(bgr, foo_bgr2hue); }

cv::Mat bgr2saturation(const cv::Mat& bgr) { return grabBgr(bgr, foo_bgr2saturation); }

void regionFilter(cv::Mat& img, int min_area, int max_area) {
  if (img.type() != CV_8UC1) return;  // only process CV_8UC1
  typedef cv::Point_<int16_t> Point2s;
  int width = img.cols, height = img.rows, npixels = width * height;
  size_t bufSize = npixels * (int)(sizeof(Point2s) + sizeof(int) + sizeof(uchar));
  cv::Mat _buf;
  _buf.create(1, (int)bufSize, CV_8U);
  uchar* buf = _buf.ptr();
  int i, j, dstep = (int)(img.step / sizeof(uchar));
  int* labels = (int*)buf;
  buf += npixels * sizeof(labels[0]);
  Point2s* wbuf = (Point2s*)buf;
  buf += npixels * sizeof(wbuf[0]);
  uchar* rtype = (uchar*)buf;
  int cur_label = 0;
  // clear out label assignments
  memset(labels, 0, npixels * sizeof(labels[0]));

  for (i = 0; i < height; i++) {
    uchar* ds = img.ptr<uchar>(i);
    int* ls = labels + width * i;
    for (j = 0; j < width; j++) {
      if (ds[j] == 0) continue;
      if (ls[j]) {
        // has a label, check for bad label
        if (rtype[ls[j]])  // bad region label, set to 0
          ds[j] = (uchar)0;
      } else {
        // no label, assign and propagate
        Point2s* ws = wbuf;                 // initialize wavefront
        Point2s p((int16_t)j, (int16_t)i);  // current pixel
        cur_label++;                        // next label
        int cur_area = 0;                   // current region size
        ls[j] = cur_label;
        // wavefront propagation
        while (ws >= wbuf) {  // wavefront not empty
          cur_area++;
          // put neighbors onto wavefront
          uchar* dpp = &img.at<uchar>(p.y, p.x);
          int* lpp = labels + width * p.y + p.x;
          // check down
          if (p.y < height - 1 && !lpp[+width] && dpp[+dstep] != 0) {
            lpp[+width] = cur_label;
            *ws++ = Point2s(p.x, p.y + 1);
          }
          // check up
          if (p.y > 0 && !lpp[-width] && dpp[-dstep] != 0) {
            lpp[-width] = cur_label;
            *ws++ = Point2s(p.x, p.y - 1);
          }
          // check right
          if (p.x < width - 1 && !lpp[+1] && dpp[+1] != 0) {
            lpp[+1] = cur_label;
            *ws++ = Point2s(p.x + 1, p.y);
          }
          // check left
          if (p.x > 0 && !lpp[-1] && dpp[-1] != 0) {
            lpp[-1] = cur_label;
            *ws++ = Point2s(p.x - 1, p.y);
          }
          // pop most recent and propagate
          p = *--ws;
        }

        // assign label type
        if (cur_area < min_area || cur_area > max_area) {
          rtype[ls[j]] = 1;  // bad region label
          ds[j] = (uchar)0;
        } else {
          rtype[ls[j]] = 0;  // good region label
        }
      }
    }
  }
}

void connectedComponentFilter(cv::Mat& img, double bw_thres, float min_area_ratio, int max_k) {
  if (img.channels() != 1) return;
  const int min_area = img.cols * img.rows * min_area_ratio;
  int num = cv::countNonZero(img);
  if (num < min_area) {
    if (num > 0) img.setTo(0);
    return;
  }

  // convert to binary image(CV_8UC1) with input threshold
  cv::Mat bw;
  cv::threshold(img, bw, bw_thres, 255, cv::THRESH_BINARY);
  if (bw.depth() != CV_8U) bw.convertTo(bw, CV_8UC(bw.channels()));

  // detect connected components in binary images
  cv::Mat labels;   // label image, same size as bw, 0 means background, 1-N means component id
  cv::Mat stats;    // statistics output for each label
  cv::Mat centers;  // centroids, not used
  int label_cnt = cv::connectedComponentsWithStats(bw, labels, stats, centers, 4, CV_32S);

  // find components whose area ratio larger input threshold, sort with area descend
  std::vector<std::pair<int, int>> pair;  // <label-id, area>
  for (int i = 1; i < stats.rows; i++) {
    int area = stats.at<int32_t>(i, cv::CC_STAT_AREA);
    if (area > min_area) pair.push_back(std::make_pair(i, area));
  }
  std::sort(pair.begin(), pair.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.second > b.second; });

#if 0
  // select max k area, set label flag true if select
  std::vector<bool> label_flags(label_cnt, false);  // true:select false:not select
  int n = std::min(max_k, static_cast<int>(pair.size()));
  for (int i = 0; i < n; i++) label_flags[pair[i].first] = true;

  // modify img if region label is zero
  int elem_size = img.elemSize();
  for (int i = 0; i < img.rows; i++) {
    uchar* ptr_img = img.ptr<uchar>(i);
    int32_t* ptr_lbl = labels.ptr<int32_t>(i);
    for (int j = 0; j < img.cols; j++, ptr_img += elem_size, ptr_lbl++) {
      int32_t lbl = *ptr_lbl;
      if (label_flags[lbl] == false) memset(ptr_img, 0, elem_size);
    }
  }
#else
  if (pair.empty()) {
    if (label_cnt > 0) img.setTo(0);
  } else if (max_k == 1) {
    img.setTo(0, labels != pair.front().first);
  } else {
    cv::Mat use_mask(img.size(), CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < max_k; i++) {
      use_mask |= (labels == pair[i].first);
    }
    img.setTo(0, ~use_mask);
  }
#endif
}

void sobelFilter(const cv::Mat& img, cv::Mat& grad_bw, int grad_thres) {
  grad_bw = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
  int w = grad_bw.cols;
#if 1
  for (int i = 1; i < img.rows - 1; i++) {
    const uchar* ptr = img.ptr<uchar>(i);
    uchar* ptr_bw = grad_bw.ptr<uchar>(i);
    for (int j = 1; j < img.cols - 1; j++) {
      const uchar* p = ptr + j;
      int grad_x = -*(p - w - 1) + *(p - w + 1) - *(p - 1) * 2 + *(p + 1) * 2 - *(p + w - 1) + *(p + w + 1);
      int grad_y = -*(p - w - 1) - *(p - w) * 2 - *(p - w + 1) + *(p + w - 1) + *(p + w) * 2 + *(p + w + 1);
      if (abs(grad_x) > grad_thres || abs(grad_y) > grad_thres) {
        ptr_bw[j] = 255;
      }
    }
  }
#else
  uchar const* di = (uchar*)img.data;
  uchar* db = (uchar*)grad_bw.data;
  uchar const *p0 = di, *p1 = p0 + w, *p2 = p1 + w;
  uchar* pb = db + w + 1;
  for (int i = 1; i < img.rows - 1; i++) {
    for (int j = 1; j < img.cols - 1; j++) {
      int grad_x = -p0[0] + p0[2] - p1[0] - p1[0] + p1[2] + p1[2] - p2[0] + p2[2];
      int grad_y = -p0[0] - p0[1] - p0[1] - p0[2] + p2[0] + p2[1] + p2[1] + p2[2];
      if (abs(grad_x) > grad_thres || abs(grad_y) > grad_thres) {
        *pb = 255;
      }
      pb++;
      p0++;
      p1++;
      p2++;
    }
    p0 += 2;
    p1 += 2;
    p2 += 2;
    pb += 2;
  }
#endif
}

void immergeInPlace(const cv::Mat& img1, cv::Mat& img2, const cv::Mat& alpha1) {
  bool check = img1.size() == img2.size() && img1.size() == alpha1.size() && img1.type() == img2.type() &&
               alpha1.channels() == 1;
  if (!check) return;

  // convert alpha to float
  cv::Mat float_alpha1;
  if (alpha1.type() == CV_8UC1) {
    alpha1.convertTo(float_alpha1, CV_32FC1, 1.0f / 255.0f);
  } else if (alpha1.type() != CV_32FC1) {
    alpha1.convertTo(float_alpha1, CV_32FC1);
  } else {
    float_alpha1 = alpha1;
  }

  // merge in pixels
  for (int i = 0; i < img1.rows; i++) {
    uchar* ptr_res = img2.data + i * img2.step1();
    const uchar* ptr1 = img1.data + i * img1.step1();
    const float* ptr_alpha = float_alpha1.ptr<float>(i);
    for (int j = 0; j < img1.cols; j++) {
      float k = *ptr_alpha++;
      for (int c = 0; c < img1.channels(); c++, ptr_res++, ptr1++) {
        uchar v_bg = *ptr_res;
        uchar v_fg = *ptr1;
        *ptr_res = clip(v_fg * k + v_bg * (1 - k), 0, 255);
      }
    }
  }
}

cv::Mat immerge(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& alpha1) {
  cv::Mat res = img2.clone();
  immergeInPlace(img1, res, alpha1);
  return res;
}

template <typename T>
cv::Rect maskRoiImpl(const cv::Mat& mask) {
  int imin = mask.rows, imax = 0, jmin = mask.cols, jmax = 0;
  for (int i = 0; i < mask.rows; i++) {
    const T* ptr = mask.ptr<T>(i);
    int tmp_jmin = -1;
    for (int j = 0; j < mask.cols; j++) {
      if (*ptr++ > 0) {
        tmp_jmin = j;
        break;
      }
    }
    if (tmp_jmin < 0) continue;  // whole 0 on current line
    // find non zero on current line
    if (i < imin) imin = i;
    if (i > imax) imax = i;
    if (tmp_jmin < jmin) jmin = tmp_jmin;
    int tmp_jmax = -1;
    ptr = mask.ptr<T>(i) + mask.cols - 1;
    for (int j = mask.cols - 1; j >= tmp_jmin; j--) {
      if (*ptr-- > 0) {
        tmp_jmax = j;
        break;
      }
    }
    if (tmp_jmax < 0) printf("[ERROR]This cannot happen\n");
    if (tmp_jmax > jmax) jmax = tmp_jmax;
  }
  if (jmin > jmax || imin > imax) return cv::Rect();
  return cv::Rect(jmin, imin, jmax - jmin + 1, imax - imin + 1);
}

cv::Rect maskRoi(const cv::Mat& mask) {
  if (mask.empty()) {
    return cv::Rect(0, 0, 0, 0);
  } else if (mask.channels() != 1) {
    printf("[ERROR]maskRoi: input mask channels(%d) is not 1.\n", mask.channels());
    return cv::Rect(0, 0, 0, 0);
  }
  if (mask.depth() == CV_8U || mask.depth() == CV_8S) {
    return maskRoiImpl<uchar>(mask);
  } else if (mask.depth() == CV_16U || mask.depth() == CV_16S) {
    return maskRoiImpl<uint16_t>(mask);
  } else if (mask.depth() == CV_32S) {
    return maskRoiImpl<uint32_t>(mask);
  } else if (mask.depth() == CV_32F) {
    return maskRoiImpl<float>(mask);
  } else {
    return maskRoiImpl<double>(mask);
  }
}

bool searchEndPointByDir(const cv::Mat& img, const cv::Vec2f& search_dir, cv::Point2f& p1, cv::Point2f& p2,
                         cv::Rect& roi_rect, int max_process_length) {
  cv::Mat process_img;
  float k = 1.0f;
  int img_length = std::max(img.cols, img.rows);
  if (max_process_length > 0 && max_process_length < img_length) {
    k = static_cast<float>(max_process_length) / img_length;
    cv::resize(img, process_img, cv::Size(), k, k, cv::INTER_NEAREST);
  } else {
    process_img = img;
  }

  float norm = hypotf(search_dir[0], search_dir[1]);
  if (norm < 1e-4) return false;
  cv::Point2f unit_dir(search_dir[0] / norm, search_dir[1] / norm);

  std::vector<cv::Point> pts;
  if (0) {  // slow, lots of point to be checked
    cv::findNonZero(process_img, pts);
  } else {  // fast, only contour point to be checked
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat bw;
    cv::threshold(process_img, bw, 1, 255, cv::THRESH_BINARY);
    cv::findContours(bw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
      pts.insert(pts.end(), contour.begin(), contour.end());
    }
  }
  if (pts.empty()) return false;
  roi_rect = cv::boundingRect(pts);

  std::vector<float> move_list;
  move_list.reserve(pts.size());
  for (const auto& pt : pts) move_list.push_back(unit_dir.dot(pt));
  int min_idx = 0, max_idx = 0;
  vecArgMinMax(move_list, min_idx, max_idx);
  const auto& pmin = pts[min_idx];
  const auto& pmax = pts[max_idx];
  p1 = cv::Point2f(pmin.x / k, pmin.y / k);
  p2 = cv::Point2f(pmax.x / k, pmax.y / k);
  roi_rect.x /= k;
  roi_rect.y /= k;
  roi_rect.width /= k;
  roi_rect.height /= k;
  roi_rect.width = std::min(roi_rect.width, img.cols - roi_rect.x);
  roi_rect.height = std::min(roi_rect.height, img.rows - roi_rect.y);
  return true;
}

bool searchEndPointByDir(const cv::Mat& img, cv::Point2f& p1, cv::Point2f& p2, int search_dir, int search_step) {
  if (search_dir == 0) {
    auto foo_search_row = [img](int y, cv::Point2f& p, int step) {
      const uchar* ptr = img.ptr<uchar>(y);
      std::vector<int> xlist;
      for (int x = 0; x < img.cols; x += step, ptr += step) {
        if (*ptr > 0) xlist.push_back(x);
      }
      if (xlist.empty()) return false;
      p.x = vecMean(xlist);
      p.y = y;
      return true;
    };

    int y;
    for (y = 0; y < img.rows; y += search_step) {
      if (foo_search_row(y, p1, search_step)) break;
    }
    if (y >= img.rows) return false;
    for (int y = img.rows - 1; y >= 0; y -= search_step) {
      if (foo_search_row(y, p2, search_step)) break;
    }
    return y >= 0;
  } else if (search_dir == 1) {
    auto foo_search_col = [img](int x, cv::Point2f& p, int step) {
      std::vector<int> ylist;
      int stride = img.step * step;
      const uchar* ptr = img.data + x;
      for (int y = 0; y < img.rows; y += step, ptr += stride) {
        if (*ptr > 0) ylist.push_back(y);
      }
      if (ylist.empty()) return false;
      p.x = x;
      p.y = vecMean(ylist);
      return true;
    };
    int x;
    for (x = 0; x < img.cols; x += search_step) {
      if (foo_search_col(x, p1, search_step)) break;
    }
    if (x >= img.cols) return false;
    for (int x = img.cols - 1; x >= 0; x -= search_step) {
      if (foo_search_col(x, p2, search_step)) break;
    }
    return x >= 0;
  } else if (search_dir == 2) {
    if (searchEndPointByDir(img, p1, p2, 0, search_step)) {
      std::swap(p1, p2);
      return true;
    }
  } else if (search_dir == 3) {
    if (searchEndPointByDir(img, p1, p2, 1, search_step)) {
      std::swap(p1, p2);
      return true;
    }
  }
  return false;
}

std::vector<cv::Mat> imsplit(const cv::Mat& img) {
  // split image by channels
  std::vector<cv::Mat> img_split;
  if (img.channels() == 1) {
    img_split = {img};
  } else {
    cv::split(img, img_split);
  }
  return img_split;
}

std::vector<float> calcHist(const cv::Mat& gray, bool normalized, const cv::Mat& mask) {
  const int bin_cnt = 256;
  float range[] = {0, static_cast<float>(bin_cnt)};
  const float* hist_ranges = {range};
  cv::Mat hist;
  cv::calcHist(&gray, 1, 0, mask, hist, 1, &bin_cnt, &hist_ranges, true, false);
  if (normalized) {
    float sum = cv::sum(hist)[0];
    if (sum > 0) hist /= sum;
  }
  std::vector<float> res;
  res.reserve(bin_cnt);
  float* ptr = reinterpret_cast<float*>(hist.data);
  for (int i = 0; i < bin_cnt; i++) res.push_back(*ptr++);
  return res;
}

cv::Mat applyLut(const cv::Mat& src_img, const std::vector<std::vector<uchar>>& luts, const cv::Rect& roi) {
  if (src_img.depth() != CV_8U) {
    printf("[ERROR]ColorAdjustor: apply lut failed, image depth(%d) is not CV_8U.\n", src_img.depth());
    return src_img;
  }
#if 0  // draw lut curves
  std::vector<cv::Mat> imgs;
  for (int i = 0; i < 3; i++) {
    Plot plt(1, 1, 0, 0, 255, 255);
    plt.plot(&luts[i][0], 256, cv::Scalar(255, 0, 0), true, 2);
    auto a = vecArange<float>(256);
    plt.plot(&a[0], 256, cv::Scalar(0, 255, 0), true, 2);
    imgs.push_back(plt.draw());
  }
  cv::imshow("img", hstack(imgs));
  cv::waitKey();
#endif
  cv::Mat lut_mat(1, 256, CV_8UC(luts.size()));
  uchar* lut_ptr = lut_mat.data;
  for (int i = 0; i < 256; i++)
    for (size_t j = 0; j < luts.size(); j++) *lut_ptr++ = luts[j][i];

  if (roi.width > 0 && roi.height > 0) {
    cv::Mat res_img(src_img.size(), src_img.type(), cv::Scalar(0));
    cv::LUT(src_img(roi), lut_mat, res_img(roi));
    return res_img;
  } else {
    cv::Mat res_img;
    cv::LUT(src_img, lut_mat, res_img);
    return res_img;
  }
}

cv::Rect cropMaskRoi(const cv::Mat& img, const cv::Mat& mask, cv::Mat& roi_img, cv::Mat& roi_mask, int pad_w,
                     bool deep_copy) {
  auto rect = maskRoi(mask);
  if (rect.width <= 0 || rect.height <= 0) {
    if (deep_copy) {
      roi_img = img.clone();
      roi_mask = mask.clone();
    } else {
      roi_img = img;
      roi_mask = mask;
    }
    return cv::Rect(0, 0, mask.cols, mask.rows);
  }
  if (pad_w > 0) {
    auto pad_rect = rectCrop(rectPad(rect, pad_w), mask.size());
    rect = pad_rect;
  }
  if (deep_copy) {
    roi_img = img(rect).clone();
    roi_mask = mask(rect).clone();
  } else {
    roi_img = img(rect);
    roi_mask = mask(rect);
  }
  return rect;
}

cv::Mat whiteBalance(const cv::Mat& img, const cv::Mat& mask, float adjust_rate) {
  if (img.type() != CV_8UC3 || adjust_rate <= 0) return img;
  auto img_split = imsplit(img);

  const int bin_cnt = 256;
  const float out_min = 0;
  const float out_max = 255;
  const cv::Size small_size(100, 100);
  cv::Mat small_mask;
  if (!mask.empty()) cv::resize(mask, small_mask, small_size, 0, 0, cv::INTER_NEAREST);

  std::vector<std::vector<uchar>> luts(img_split.size(), std::vector<uchar>(256));
  for (size_t ch = 0; ch < img_split.size(); ch++) {
    const auto& timg = img_split[ch];
    auto& lut = luts[ch];
    int min_value = 0;
    int max_value = bin_cnt;

    cv::Mat hist_img;
    cv::resize(timg, hist_img, small_size, 0, 0, cv::INTER_NEAREST);
    float range[] = {0, static_cast<float>(bin_cnt)};
    const float* hist_ranges = {range};
    cv::Mat hist;
    cv::calcHist(&hist_img, 1, 0, small_mask, hist, 1, &bin_cnt, &hist_ranges, true, false);

    // searching for s1 and s2
    const int n_pixels = hist_img.cols * hist_img.rows;
    const int thres_low = n_pixels * 0.02;
    const int thres_high = n_pixels * 0.98;
    int p1 = 0, p2 = bin_cnt - 1;
    int n1 = 0, n2 = n_pixels;
    while (n1 + hist.at<float>(p1) < thres_low) {
      n1 += cv::saturate_cast<int>(hist.at<float>(p1++));
      min_value++;
    }
    while (n2 - hist.at<float>(p2) > thres_high) {
      n2 -= cv::saturate_cast<int>(hist.at<float>(p2--));
      max_value--;
    }
    min_value = std::min(min_value, 50);
    max_value = std::max(max_value, 205);
    float alpha = 1 + adjust_rate * ((out_max - out_min) / static_cast<float>(max_value - min_value) - 1);
    float beta = (-min_value * (out_max - out_min) / static_cast<float>(max_value - min_value) + out_min) * adjust_rate;
    for (size_t i = 0; i < lut.size(); i++) lut[i] = clip(i * alpha + beta, 0, 255);
  }
  return applyLut(img, luts);
}

cv::Mat adjustBrightness(const cv::Mat& img, int level) {
  static std::vector<float> diff_negative = {
      0.00, 0.01, 0.01, 0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.06, 0.06, 0.07, 0.08, 0.08, 0.09, 0.10, 0.10, 0.11, 0.12,
      0.12, 0.13, 0.13, 0.14, 0.15, 0.15, 0.16, 0.16, 0.17, 0.17, 0.18, 0.19, 0.19, 0.20, 0.20, 0.21, 0.22, 0.22, 0.23,
      0.23, 0.24, 0.24, 0.25, 0.25, 0.26, 0.27, 0.27, 0.28, 0.28, 0.29, 0.30, 0.30, 0.31, 0.31, 0.32, 0.32, 0.33, 0.33,
      0.34, 0.34, 0.35, 0.36, 0.36, 0.37, 0.37, 0.38, 0.38, 0.39, 0.39, 0.40, 0.40, 0.41, 0.41, 0.42, 0.42, 0.43, 0.43,
      0.44, 0.44, 0.45, 0.46, 0.46, 0.47, 0.47, 0.48, 0.48, 0.49, 0.49, 0.50, 0.50, 0.51, 0.51, 0.52, 0.52, 0.53, 0.53,
      0.54, 0.54, 0.55, 0.55, 0.56, 0.56, 0.57, 0.57, 0.58, 0.58, 0.59, 0.59, 0.60, 0.60, 0.61, 0.61, 0.61, 0.62, 0.62,
      0.63, 0.63, 0.64, 0.64, 0.65, 0.65, 0.65, 0.66, 0.66, 0.67, 0.67, 0.68, 0.68, 0.69, 0.69, 0.69, 0.70, 0.70, 0.71,
      0.71, 0.71, 0.72, 0.72, 0.72, 0.73, 0.73, 0.73, 0.73, 0.74, 0.74, 0.74, 0.75, 0.75, 0.75, 0.76, 0.76, 0.76, 0.76,
      0.77, 0.77, 0.77, 0.78, 0.78, 0.78, 0.78, 0.79, 0.79, 0.79, 0.79, 0.80, 0.80, 0.80, 0.80, 0.80, 0.81, 0.81, 0.81,
      0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81,
      0.80, 0.80, 0.80, 0.80, 0.79, 0.79, 0.79, 0.79, 0.78, 0.78, 0.78, 0.78, 0.77, 0.77, 0.77, 0.77, 0.76, 0.76, 0.76,
      0.76, 0.75, 0.75, 0.75, 0.74, 0.74, 0.73, 0.73, 0.73, 0.72, 0.72, 0.71, 0.71, 0.71, 0.70, 0.70, 0.69, 0.69, 0.69,
      0.68, 0.68, 0.68, 0.68, 0.67, 0.67, 0.67, 0.66, 0.66, 0.65, 0.65, 0.65, 0.64, 0.64, 0.64, 0.64, 0.63, 0.63, 0.63,
      0.62, 0.62, 0.62, 0.62, 0.61, 0.61, 0.61, 0.60, 0.60};
  static std::vector<float> diff_positive = {
      0.00, 0.01, 0.03, 0.04, 0.06, 0.07, 0.09, 0.10, 0.12, 0.13, 0.15, 0.16, 0.18, 0.19, 0.21, 0.22, 0.24, 0.25, 0.26,
      0.28, 0.29, 0.31, 0.32, 0.34, 0.35, 0.37, 0.38, 0.40, 0.41, 0.43, 0.44, 0.46, 0.47, 0.49, 0.50, 0.51, 0.52, 0.53,
      0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.62, 0.63, 0.64, 0.65, 0.66, 0.68, 0.69, 0.70, 0.71, 0.71, 0.72, 0.73, 0.74,
      0.74, 0.75, 0.76, 0.76, 0.77, 0.78, 0.78, 0.79, 0.80, 0.81, 0.81, 0.82, 0.83, 0.83, 0.84, 0.84, 0.85, 0.85, 0.86,
      0.86, 0.87, 0.87, 0.88, 0.88, 0.89, 0.89, 0.90, 0.90, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91,
      0.91, 0.92, 0.92, 0.92, 0.92, 0.92, 0.93, 0.93, 0.93, 0.92, 0.92, 0.92, 0.92, 0.91, 0.91, 0.91, 0.90, 0.90, 0.90,
      0.90, 0.90, 0.89, 0.89, 0.89, 0.89, 0.89, 0.88, 0.88, 0.88, 0.87, 0.87, 0.87, 0.86, 0.86, 0.86, 0.86, 0.85, 0.85,
      0.85, 0.84, 0.84, 0.84, 0.83, 0.83, 0.82, 0.82, 0.81, 0.81, 0.80, 0.79, 0.79, 0.78, 0.78, 0.77, 0.76, 0.76, 0.75,
      0.75, 0.74, 0.73, 0.73, 0.72, 0.71, 0.71, 0.70, 0.70, 0.69, 0.68, 0.68, 0.67, 0.66, 0.66, 0.65, 0.64, 0.64, 0.63,
      0.62, 0.62, 0.61, 0.60, 0.60, 0.59, 0.58, 0.58, 0.57, 0.56, 0.56, 0.55, 0.55, 0.54, 0.53, 0.53, 0.52, 0.51, 0.51,
      0.50, 0.49, 0.48, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.37, 0.36, 0.35,
      0.35, 0.34, 0.33, 0.33, 0.32, 0.31, 0.30, 0.30, 0.29, 0.28, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.20,
      0.19, 0.18, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.09, 0.08, 0.08, 0.07, 0.06, 0.06, 0.05,
      0.05, 0.04, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.00};
  level = clip(level, -100, 100);
  if (level == 0) return img;
  std::vector<uint8_t> lut;
  lut.reserve(256);
  if (level > 0) {
    for (int i = 0; i < 256; i++) lut.push_back(i + level * diff_positive[i]);
  } else {
    for (int i = 0; i < 256; i++) lut.push_back(i + level * diff_negative[i]);
  }
  return applyLut(img, {lut});
}

HistogramMatch::HistogramMatch(const cv::Mat& ref_img, const cv::Size& resize_wh, const HistogramType& prior_hist)
    : m_resize_wh(resize_wh), m_prior_hist(prior_hist) {
  cv::Mat process_img;
  if (m_resize_wh.area() > 0 && m_resize_wh.area() < ref_img.size().area()) {
    cv::resize(ref_img, process_img, m_resize_wh, cv::INTER_NEAREST);
  } else {
    process_img = ref_img;
  }
  std::vector<cv::Mat> img_split = imsplit(process_img);
  int n_channels = ref_img.channels();
  m_ref_hist_buf.resize(n_channels);
  m_ref_cdf_buf.resize(n_channels);
  for (int i = 0; i < n_channels; i++) {
    m_ref_hist_buf[i] = calcHist(img_split[i], true);
    m_ref_cdf_buf[i] = vecCumsum(m_ref_hist_buf[i]);
  }
#if 0  // debug
  // Plot plt(0.5,0.0001);
  // plt.plot(reinterpret_cast<float*>(m_ref_hist_b.data), m_bin_cnt, cv::Scalar(255, 0, 0), true, 2);
  // plt.plot(reinterpret_cast<float*>(m_ref_hist_g.data), m_bin_cnt, cv::Scalar(0, 255, 0), true, 2);
  // plt.plot(reinterpret_cast<float*>(m_ref_hist_r.data), m_bin_cnt, cv::Scalar(0, 0, 255), true, 2);
  Plot plt(0.5, 0.002);
  plt.plot(&m_ref_cdf_buf[0][0], m_bin_cnt, cv::Scalar(255, 0, 0), true, 2);
  plt.plot(&m_ref_cdf_buf[1][0], m_bin_cnt, cv::Scalar(0, 255, 0), true, 2);
  plt.plot(&m_ref_cdf_buf[2][0], m_bin_cnt, cv::Scalar(0, 0, 255), true, 2);
  cv::imshow("ref_hist", plt.draw());
  cv::waitKey();
#endif
}

HistogramMatch::LutType HistogramMatch::calcAdjustLut(const cv::Mat& src_img, float adjust_rate,
                                                      const cv::Mat& src_mask) {
  if (adjust_rate <= 0 || src_img.channels() != static_cast<int>(m_ref_cdf_buf.size())) return {};

  cv::Mat process_img, process_mask;
  if (m_resize_wh.area() > 0 && m_resize_wh.area() < process_img.size().area()) {
    cv::resize(src_img, process_img, m_resize_wh, cv::INTER_NEAREST);
    cv::resize(src_mask, process_mask, m_resize_wh, cv::INTER_NEAREST);
  } else {
    process_img = src_img;
    process_mask = src_mask;
  }
  if (process_mask.type() != CV_8UC1) process_mask = cv::Mat();

  // calculate LUT for each channel
  std::vector<cv::Mat> img_split = imsplit(process_img);
  LutType luts(src_img.channels());
  for (int i = 0; i < src_img.channels(); i++) {
    auto hist = calcHist(img_split[i], true, process_mask);
    if (src_img.channels() == static_cast<int>(m_prior_hist.size())) {
      vecElemAdd(hist, m_prior_hist[i]);
      vecNormalize(hist, 1);
    }
    auto cdf = vecCumsum(hist);
    auto index = searchsorted(m_ref_cdf_buf[i], cdf);
    auto& lut = luts[i];
    lut.reserve(index.size());
    for (size_t j = 0; j < index.size(); j++)
      lut.push_back(clip(j + (static_cast<float>(index[j]) - j) * adjust_rate, 0, 255));
  }
  return luts;
}

cv::Mat HistogramMatch::adjust(const cv::Mat& src_img, float adjust_rate, const cv::Mat& src_mask) {
  if (adjust_rate <= 0) return src_img;
  auto luts = calcAdjustLut(src_img, adjust_rate, src_mask);
  if (luts.empty()) return src_img;
  return applyLut(src_img, luts);
}

GuidedFilter::GuidedFilter(const cv::Mat& ref_img, int r, float eps) : blur_size_(r, r), eps_(eps) {
  cv::Mat gray;
  if (ref_img.channels() == 4)
    cv::cvtColor(ref_img, gray, cv::COLOR_BGRA2GRAY);
  else if (ref_img.channels() == 3)
    cv::cvtColor(ref_img, gray, cv::COLOR_BGR2GRAY);
  else if (ref_img.channels() == 1)
    gray = ref_img;

  cv::Mat I;
  convertToFloat(gray, I);
  if (I.channels() == 1) {
    cv::Mat mean_I = blur(I);
    cv::Mat mean_II = blur(I.mul(I));
    cv::Mat var_I = mean_II - mean_I.mul(mean_I);
    Is_.push_back(I);
    mean_Is_.push_back(mean_I);
    covs_.push_back(var_I);
  }
}

cv::Mat GuidedFilter::filter(const cv::Mat& input_img) const {
  cv::Mat float_img;
  double alpha = convertToFloat(input_img, float_img);

  cv::Mat result;
  if (float_img.channels() == 1) {
    result = filterOneChannel(float_img);
  } else {
    std::vector<cv::Mat> channel_imgs;
    cv::split(float_img, channel_imgs);
    for (auto& img : channel_imgs) img = filterOneChannel(img);
    cv::merge(channel_imgs, result);
  }

  if (result.type() != input_img.type()) result.convertTo(result, input_img.type(), 1.0 / alpha);
  return result;
}

cv::Mat GuidedFilter::filterOneChannel(const cv::Mat& P) const {
  if (Is_.size() == 1) {
    const cv::Mat& I = Is_[0];
    const cv::Mat& mean_I = mean_Is_[0];
    const cv::Mat& var_I = covs_[0];
    cv::Mat mean_P = blur(P);
    cv::Mat mean_IP = blur(I.mul(P));
    cv::Mat cov_IP = mean_IP - mean_I.mul(mean_P);
    cv::Mat a = cov_IP / (var_I + eps_);  // Eqn. (5) in the paper;
    cv::Mat b = mean_P - a.mul(mean_I);   // Eqn. (6) in the paper;
    cv::Mat mean_a = blur(a);
    cv::Mat mean_b = blur(b);
    return mean_a.mul(I) + mean_b;
  }
  return cv::Mat();
}

double GuidedFilter::convertToFloat(const cv::Mat& in, cv::Mat& out, bool deep_copy) const {
  double alpha = 1.0;
  if (in.depth() == CV_8U) alpha = 1.0 / 255.0;
  if (in.depth() == CV_32F) {
    out = deep_copy ? in.clone() : in;
  } else {
    in.convertTo(out, CV_32FC(in.channels()), alpha);
  }
  return alpha;
}

cv::Mat GuidedFilter::blur(const cv::Mat& in) const {
  cv::Mat out;
  // cv::blur(in, out, blur_size_, cv::Point(-1, -1), cv::BORDER_REPLICATE);
  cv::blur(in, out, blur_size_);
  return out;
}

cv::Mat guidedFilter(const cv::Mat& ref_img, const cv::Mat& input_img, int r, float eps) {
  GuidedFilter gf(ref_img, r, eps);
  return gf.filter(input_img);
}

cv::Mat resizeMaxLength(const cv::Mat& img, int max_length, bool enlarge) {
  if (img.empty()) return cv::Mat();
  cv::Mat res;
  float k = static_cast<float>(max_length) / std::max(img.cols, img.rows);
  if (enlarge || k < 1) {
    cv::resize(img, res, cv::Size(), k, k);
  } else {
    res = img;
  }
  return res;
}

void imageComposition(cv::Mat& bg_img, const std::vector<cv::Mat>& fg_imgs, const std::vector<cv::Mat>& fg_masks) {
  if (fg_imgs.empty() || fg_imgs.size() != fg_masks.size()) return;

  int fg_cnt = fg_imgs.size();
  int step = bg_img.cols / fg_cnt;
  if (step <= 0) return;
  int min_row = bg_img.rows * 0.3f;
  int max_row = bg_img.rows * 0.95f;

  for (int i = 0; i < fg_cnt; i++) {
    const auto& fg_img = fg_imgs[i];
    const auto& fg_mask = fg_masks[i];
    cv::Rect bg_roi(step * i, min_row, step, max_row - min_row);

    cv::Mat bw;
    cv::threshold(fg_mask, bw, 20, 255, cv::THRESH_BINARY);
    std::vector<cv::Point> mask_pts;
    cv::findNonZero(bw, mask_pts);
    cv::Rect fg_roi = cv::boundingRect(mask_pts);

    float k =
        std::min(static_cast<float>(bg_roi.width) / fg_roi.width, static_cast<float>(bg_roi.height) / fg_roi.height);
    cv::Mat fg_roi_img, fg_roi_mask;
    cv::resize(fg_img(fg_roi), fg_roi_img, cv::Size(), k, k);
    cv::resize(fg_mask(fg_roi), fg_roi_mask, fg_roi_img.size());
    cv::Rect merge_roi(bg_roi.x + (bg_roi.width - fg_roi_img.cols) / 2, bg_roi.y + bg_roi.height - fg_roi_img.rows,
                       fg_roi_img.cols, fg_roi_img.rows);
    cv::Mat bg_roi_img = bg_img(merge_roi);
    immergeInPlace(fg_roi_img, bg_roi_img, fg_roi_mask);
  }
}

} /* namespace vs */