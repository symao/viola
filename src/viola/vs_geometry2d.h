/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details geometry for 2D point, line, rect.
 */
#pragma once
#include <functional>
#include <set>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "vs_basic.h"
#include "vs_random.h"

namespace vs {

/** @brief check whether a point inside image with border threshold */
template <typename T>
bool inside(const cv::Point_<T>& p, const cv::Size& img_size, int border = 0) {
  return border <= p.x && p.x + border < img_size.width && border <= p.y && p.y + border < img_size.height;
}

/** @brief check whether a point inside image with border threshold */
template <typename T>
bool inside(const cv::Point_<T>& p, const cv::Size& img_size, const cv::Size& border) {
  return border.width <= p.x && p.x + border.width < img_size.width && border.height <= p.y &&
         p.y + border.height < img_size.height;
}

/** @brief check whether a point inside rect with border threshold */
template <typename T1, typename T2>
bool inside(const cv::Point_<T1>& p, const cv::Rect_<T2>& rect, int border = 0) {
  return rect.x + border <= p.x && p.x + border < rect.x + rect.width && rect.y + border <= p.y &&
         p.y + border < rect.y + rect.height;
}

/** @brief normalize a point to mod 1 */
inline cv::Point2f normalize(const cv::Point2f& a) { return a * (1.0f / (cv::norm(a) + VS_EPS)); }

/** @brief euclidian distance between two points */
inline float dist(const cv::Point2f& p1, const cv::Point2f& p2) { return hypotf(p1.x - p2.x, p1.y - p2.y); }

/** @brief distance from point to line
 * @param[in]p1: first line point
 * @param[in]p2: second line point
 * @param[in]p: check point
 * @return distance from point p to line p1-p2
 */
inline float dist2line(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p) {
  cv::Point2f d = p2 - p1;
  float len = hypotf(d.x, d.y);
  if (len < VS_EPS) return dist(p, p1);
  return fabs(d.cross(p - p1) / len);
}

/** @brief projection from point to line
 * @param[in]p1: first line point
 * @param[in]p2: second line point
 * @param[in]p: check point
 * @return projection of point p on line p1-p2
 */
inline cv::Point2f project2line(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p) {
  cv::Point2f d = p2 - p1;
  float len = hypotf(d.x, d.y);
  if (len < VS_EPS) return p1;
  d *= (1.0f / len);
  return p1 + (p - p1).dot(d) * d;
}

/** @brief distance from point to line segment
 * @param[in]p1: first end-point of line segment
 * @param[in]p2: second end-point of line segment
 * @param[in]p: check point
 * @return distance from point p to line segment p1-p2
 */
inline float dist2lineseg(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p) {
  cv::Point2f m = project2line(p1, p2, p);
  auto v1 = m - p1;
  auto v2 = m - p2;
  if (v1.dot(v2) < 0)
    return dist(p, m);
  else if (fabs(v1.x) + fabs(v1.y) < fabs(v2.x) + fabs(v2.y))
    return dist(p, p1);
  else
    return dist(p, p2);
}

/** @brief angle of two vector
 * @param[in]a: first vector, no need to be normalized
 * @param[in]b: second vector, no need to be normalized
 * @return angle between two vector in radians
 */
inline float vecAngle(const cv::Point2f& a, const cv::Point2f& b) {
  return acos(a.dot(b) / (cv::norm(a) * cv::norm(b)));
}

/** @brief check if two line direction are parallel
 * @param[in] dir1: UNIT direction vector
 * @param[in] dir2: UNIT direction vector
 * @param[in] thres_parallel: cosine between two direction. cos(5deg)=0.996
 * @param[in] mode 0: not concern direction   '-- --'
 *                 1: same direction          '-> ->'
 *                 2: opposite direction      '<- ->'
 */
inline bool isParallel(const cv::Point2f& dir1, const cv::Point2f& dir2, int mode = 0, float thres_parallel = 0.996f) {
  float c = dir1.dot(dir2);
  switch (mode) {
    case 0:
      return fabs(c) > thres_parallel;
    case 1:
      return c > thres_parallel;
    case 2:
      return -c > thres_parallel;
  }
  return false;
}

/** @brief check if two line are coincidence
 * @param[in] p1: point in one line
 * @param[in] dir1: UNIT direction vector
 * @param[in] p2: point in another line
 * @param[in] dir2: UNIT direction vector
 * @param[in] thres_parallel: cosine between two direction. cos(5deg)=0.996
 * @param[in] mode 0: not concern direction   '-- --'
 *                 1: same direction          '-> ->'
 *                 2: opposite direction      '<- ->'
 */
inline bool isCoincidence(const cv::Point2f& p1, const cv::Point2f& dir1, const cv::Point2f& p2,
                          const cv::Point2f& dir2, int mode = 0, float thres_parallel = 0.996f,
                          float thres_coincidence = 3.0f) {
  return isParallel(dir1, dir2, mode, thres_parallel) && (fabs((p2 - p1).cross(dir1)) < thres_coincidence ||
                                                          /*&&*/ fabs((p2 - p1).cross(dir2)) < thres_coincidence);
}

/** @brief calculate overlap length between two lines
 * @param[in]a1: first end-point of line A
 * @param[in]a2: second end-point of line A
 * @param[in]b1: first end-point of line B
 * @param[in]b2: second end-point of line B
 * @return overlap length between line A and line B
 */
inline float overlapLen(const cv::Point2f& a1, const cv::Point2f& a2, const cv::Point2f& b1, const cv::Point2f& b2) {
  cv::Point2f dir_a = a2 - a1;
  float len_a = hypotf(dir_a.x, dir_a.y);
  if (len_a <= 0) return 0;
  dir_a *= (1.0f / len_a);
  float k1 = dir_a.dot(b1 - a1);
  float k2 = dir_a.dot(b2 - a1);
  return fabs(clip(k1, 0.0f, len_a) - clip(k2, 0.0f, len_a));
}

/** @brief calculate overlap rate between two lines
 * @param[in]a1: first end-point of line A
 * @param[in]a2: second end-point of line A
 * @param[in]b1: first end-point of line B
 * @param[in]b2: second end-point of line B
 * @param[in]mode: rate calculation method, 0:len/len_a, 1:len/lenb, 2:len/min_len 3:len/max_len
 * @return overlap rate
 */
inline float overlapRate(const cv::Point2f& a1, const cv::Point2f& a2, const cv::Point2f& b1, const cv::Point2f& b2,
                         int mode = 0) {
  cv::Point2f dir_a = a2 - a1;
  float len_a = hypotf(dir_a.x, dir_a.y);
  if (len_a <= 0) return 0;
  dir_a *= (1.0f / len_a);
  float len_b = hypotf(b1.x - b2.x, b1.y - b2.y);
  if (len_b <= 0) return 0;
  float k1 = dir_a.dot(b1 - a1);
  float k2 = dir_a.dot(b2 - a1);
  float olen = fabs(clip(k1, 0.0f, len_a) - clip(k2, 0.0f, len_a));
  switch (mode) {
    case 0:
      return olen / len_a;
    case 1:
      return olen / len_b;
    case 2:
      return olen / std::min(len_a, len_b);
    case 3:
      return olen / std::max(len_a, len_b);
    default:
      return olen / (max3(k1, k2, len_a) - min3(k1, k2, 0.0f));  // iou
  }
}

/** @brief check whether two unit direction vector is perpendicular
 * @param[in]dir1: unit vector of first direction
 * @param[in]dir2: unit vector of second direction
 * @param[in]thres_perpendi: dot threshold
 * @return whether perpendicular
 */
inline bool isPerpendicular(const cv::Point2f& dir1, const cv::Point2f& dir2, float thres_perpendi = VS_COS_DEG80) {
  return fabs(dir1.dot(dir2)) < thres_perpendi;
}

/** @brief check whether angle between two unit direction vector in cosine range
 * @param[in]dir1: unit vector of first direction
 * @param[in]dir2: unit vector of second direction
 * @param[in]cos_min: min cosine angle
 * @param[in]cos_max: max cosine angle
 * @return whether angle between two unit direction vector in cosine range
 */
inline bool isAngleInRange(const cv::Point2f& dir1, const cv::Point2f& dir2, float cos_min, float cos_max) {
  return inRange(dir1.dot(dir2), cos_min, cos_max);
}

/** @brief get the intersection of two lines
 * intersection of line A (x1,y1)-(x2,y2) and line B (x3,y3)-(x4,y4)
 * @param[in]x1: x of p1
 * @param[in]y1: y of p1
 * @param[in]x2: x of p2
 * @param[in]y2: y of p2
 * @param[in]x3: x of p3
 * @param[in]y3: y of p3
 * @param[in]x4: x of p4
 * @param[in]y4: y of p4
 * @param[in]x: x of intersection point
 * @param[in]y: y of intersection point
 * @note: two line must not be paralell, if parallel, x,y will be set to -1
 */
inline void lineIntersect(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4, float& x,
                          float& y) {
  float d = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4));
  if (fabs(d) > VS_EPS) {
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
  } else {
    x = y = -1;
  }
}

/** @brief get the intersection of two lines
 * @note: two line must not be paralell, if parallel, x,y will be set to -1
 * @return intersection point
 */
inline cv::Point2f lineIntersect(const cv::Vec4f& a, const cv::Vec4f& b) {
  cv::Point2f p;
  lineIntersect(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], p.x, p.y);
  return p;
}

/** @brief get the intersection of two lines
 * @param[in]p1: arbitrary point on first line
 * @param[in]dir1: direction vector of first line
 * @param[in]p2: arbitrary point on second line
 * @param[in]dir2: direction vector of second line
 * @note: two line must not be paralell, if parallel, x,y will be set to -1
 * @return intersection point
 */
inline cv::Point2f lineIntersect(const cv::Point2f& p1, const cv::Point2f& dir1, const cv::Point2f& p2,
                                 const cv::Point2f& dir2) {
  cv::Point2f p;
  lineIntersect(p1.x, p1.y, p1.x + dir1.x, p1.y + dir1.y, p2.x, p2.y, p2.x + dir2.x, p2.y + dir2.y, p.x, p.y);
  return p;
}

/** @brief quadratic bezier curve in order 2 */
inline std::vector<cv::Point2f> bezier2(const cv::Point2f& p0, const cv::Point2f& p1, const cv::Point2f& p2,
                                        const std::vector<float>& ts) {
  std::vector<cv::Point2f> pts;
  for (float t : ts) pts.push_back(sq(1 - t) * p0 + (2 * (1 - t) * t) * p1 + sq(t) * p2);
  return pts;
}

/** @brief cubic bezier curve in order 3*/
inline std::vector<cv::Point2f> bezier3(const cv::Point2f& p0, const cv::Point2f& p1, const cv::Point2f& p2,
                                        const cv::Point2f& p3, const std::vector<float>& ts) {
  std::vector<cv::Point2f> pts;
  for (float t : ts) {
    float t2 = t * t;
    float t3 = t2 * t;
    float i = 1 - t;
    float i2 = i * i;
    float i3 = i2 * i;
    pts.push_back(i3 * p0 + (3 * i2 * t) * p1 + (3 * i * t2) * p2 + t3 * p3);
  }
  return pts;
}

/** @brief 2D arrow shape, consist of three point A-O-B, which O is arrow center top and OA,OB are arrow edges */
struct Arrow {
  Arrow() {}
  Arrow(const cv::Point2f& _top, const cv::Point2f& _dir1, const cv::Point2f& _dir2, int _id1 = 0, int _id2 = 0)
      : top(_top), dir1(_dir1), dir2(_dir2), id1(_id1), id2(_id2) {
    ray = normalize(dir1 + dir2);
    if (dir1.cross(dir2) < 0) {
      std::swap(dir1, dir2);
      std::swap(id1, id2);
    }
  }

  cv::Point2f top;   ///< top point
  cv::Point2f ray;   ///< unit direction vector of arrow center
  cv::Point2f dir1;  ///< unit direction vector of first edge
  cv::Point2f dir2;  ///< unit direction vector of second edge
  int id1;           ///< line id of first edge
  int id2;           ///< line id of second edge
};

/** @brief calculate average for angle list, which consider the angle normalization */
template <class T>
T angleMean(const std::vector<T>& angles) {
  T m = 0;
  if (angles.empty()) return m;
  T neg_min = 0.0;
  T neg_max = -9999.0;
  T pos_min = 9999.0;
  T pos_max = 0.0;

  for (auto a : angles) {
    if (a < 0) {
      if (a < neg_min) neg_min = a;
      if (a > neg_max) neg_max = a;
    } else {
      if (a < pos_min) pos_min = a;
      if (a > pos_max) pos_max = a;
    }
  }
  // has positive angles and negative angles
  //         |-PI --- neg_min --- neg_max ------ 0 ---- pos_min ----- pos_max --- PI|
  // normal: |        *******************************************************       |
  // cross:  |***************************               ****************************|
  if (neg_min <= neg_max && pos_min <= pos_max && VS_2PI - (pos_min - neg_max) < pos_max - neg_min) {
    T sum = 0;
    for (auto& a : angles) {
      sum += a < 0 ? a + VS_2PI : a;
    }
    return sum / angles.size();
  } else {
    return vecMean(angles);
  }
}

/** @brief merge list elements
 * @param[in]list: data list
 * @param[in]foo_need_merge: function to check whether two elements in list can be merge
 * @param[in]foo_merge: function to merge two elements and output a new merged element
 */
template <class T>
void merge(std::vector<T>& list, std::function<bool(const T& a, const T& b)> foo_need_merge,
           std::function<T(const T& a, const T& b)> foo_merge) {
  int cnt = list.size();
  if (cnt < 2) return;
  std::vector<T> new_list;
  new_list.reserve(cnt);

  std::set<int> ids;
  for (int i = 0; i < cnt; i++) ids.insert(i);
  while (!ids.empty()) {
    auto it = ids.begin();
    T a = list[*it];
    ids.erase(it);
    bool no_merge = false;
    while (!no_merge) {
      no_merge = true;
      for (auto i : ids) {
        T li = list[i];
        if (foo_need_merge(a, li)) {
          a = foo_merge(a, li);
          ids.erase(i);
          no_merge = false;
          break;
        }
      }
    }
    new_list.push_back(a);
  }
  list = new_list;
}

/** @brief merge list of 2D lines
 * @param[in]lines: list of input 2D lines
 * @param[in]mode: mode to check parallel, see isCoincidence()
 * @param[in]thres_parallel: cosine threshold to check parallel between two lines
 * @param[in]thres_coincidence: distance threshold to check coincidence between two lines
 * @param[in]thres_overlap: overlap rate threshold between two lines
 * @param[in]thres_connect: distance threshold to check connection between two lines
 */
inline void mergeLine(std::vector<cv::Vec4f>& lines, int mode = 0, float thres_parallel = VS_COS_DEG5,
                      float thres_coincidence = 3.0f, float thres_overlap = 0.0f, float thres_connect = 1.0f) {
  auto foo_need_merge = [mode, thres_parallel, thres_coincidence, thres_overlap, thres_connect](const cv::Vec4f& a,
                                                                                                const cv::Vec4f& b) {
    cv::Point2f a1(a[0], a[1]), a2(a[2], a[3]);
    cv::Point2f b1(b[0], b[1]), b2(b[2], b[3]);
    cv::Point2f da = a2 - a1;
    cv::Point2f db = b2 - b1;
    float lena = hypotf(da.x, da.y);
    if (lena <= VS_EPS) return false;
    float lenb = hypotf(db.x, db.y);
    if (lenb <= VS_EPS) return false;
    da *= (1.0f / lena);
    db *= (1.0f / lenb);
    cv::Point2f ca = (a1 + a2) * 0.5f;
    cv::Point2f cb = (b1 + b2) * 0.5f;
    return isCoincidence(ca, da, cb, db, mode, thres_parallel, thres_coincidence) &&
           (overlapLen(a1, a2, b1, b2) > thres_overlap || dist(a1, b1) < thres_connect ||
            dist(a1, b2) < thres_connect || dist(a2, b1) < thres_connect || dist(a2, b2) < thres_connect);
  };

  auto foo_merge = [](const cv::Vec4f& a, const cv::Vec4f& b) {
    cv::Point2f a1(a[0], a[1]), a2(a[2], a[3]);
    cv::Point2f b1(b[0], b[1]), b2(b[2], b[3]);
    cv::Point2f da = a2 - a1;
    cv::Point2f db = b2 - b1;
    float lena = hypotf(da.x, da.y);
    float lenb = hypotf(db.x, db.y);
    da *= (1.0f / lena);
    db *= (1.0f / lenb);

    float kmin = 0.0f;
    float kmax = 1.0f;
    float k1 = (b1 - a1).dot(da) / lena;
    float k2 = (b2 - a1).dot(da) / lena;
    if (kmin > k1) kmin = k1;
    if (kmax < k1) kmax = k1;
    if (kmin > k2) kmin = k2;
    if (kmax < k2) kmax = k2;
    cv::Point2f p1, p2;
    if (fequal(kmin, 0))
      p1 = a1;
    else if (fequal(kmin, k1))
      p1 = b1;
    else
      p1 = b2;
    if (fequal(kmax, 1))
      p2 = a2;
    else if (fequal(kmax, k1))
      p2 = b1;
    else
      p2 = b2;
    return cv::Vec4f(p1.x, p1.y, p2.x, p2.y);
  };
  merge<cv::Vec4f>(lines, foo_need_merge, foo_merge);
}

/** @brief calculate feature reprojection error in RMSE(Root Mean Square Error)
 * project 3D feature into 2D image plane with camera intrinsic and extrinsic, calculare root mean square error
 * @param[in]pts3d: 3D feature points
 * @param[in]pts2d: 2D feature points correponding to pts3d
 * @param[in]K: camera intrinsic matrix
 * @param[in]D: camera distortion
 * @param[in]rvec: camera extrinsic rotation vector
 * @param[in]tvec: camera extrinsic translation vector
 * @return reprojection error in RMSE
 */
inline float reprojectError(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d,
                            const cv::Mat& K, const cv::Mat& D, const cv::Mat& rvec, const cv::Mat& tvec) {
  int N = pts3d.size();
  if (N < 1 || static_cast<int>(pts2d.size()) != N) return FLT_MAX;
  std::vector<cv::Point2f> project_pts;
  cv::projectPoints(pts3d, rvec, tvec, K, D, project_pts);
  float reproject_error = 0;
  for (int i = 0; i < N; i++) {
    auto d = project_pts[i] - pts2d[i];
    reproject_error += sqsum2(d.x, d.y);
  }
  return std::sqrt(reproject_error / N);
}

/** @brief calculate camera intrinsic matrix from fov and image size
 * @param[in]fov_rad: horizontal FoV(Field of Vision) in radians
 * @param[in]img_size: image size
 * @return camera intrinsic (fx,fy,cx,cy)
 */
inline cv::Vec4f fov2intrinsic(float fov_rad, const cv::Size& img_size) {
  float u0 = img_size.width * 0.5f;
  float v0 = img_size.height * 0.5f;
  float f = img_size.width * 0.5f / tan(fov_rad * 0.5f);
  return cv::Vec4f(f, f, u0, v0);
}

/** @brief compute fov from camera intrinsic
 * @param[in]fx: focal length x
 * @param[in]fy: focal length y
 * @param[in]img_size: image size
 * @return x,y fov in radians
 */
inline cv::Vec2f intrinsic2fov(float fx, float fy, const cv::Size& img_size) {
  float fov_x = atan2(img_size.width * 0.5f, fx) * 2;
  float fov_y = atan2(img_size.height * 0.5f, fy) * 2;
  return cv::Vec2f(fov_x, fov_y);
}

/** @brief calculate camera intrinsic matrix from fov and image size
 * @param[in]fov_rad: horizontal FoV(Field of Vision) in radians
 * @param[in]img_size: image size
 * @return camera intrinsic matrix
 */
template <typename T>
inline cv::Mat fov2K(double fov_rad, const cv::Size& img_size) {
  auto intrin = fov2intrinsic(fov_rad, img_size);
  return (cv::Mat_<T>(3, 3) << intrin[0], 0, intrin[2], 0, intrin[1], intrin[3], 0, 0, 1);
}

/** @brief Image rotation, rotate image, keep all pixels so image size may be larger than raw size */
class ImageRoter {
 public:
  /** @brief Construct a new Image Roter object
   * @param[in]raw_size: raw image size before rotation
   * @param[in]rot_rad: rotate angle in radians, positive values mean counter-clockwise rotation
   */
  ImageRoter(const cv::Size& raw_size, double rot_rad) {
    const int w = raw_size.width;
    const int h = raw_size.height;
    double sa = sin(rot_rad);
    double ca = cos(rot_rad);
    double hnew = w * fabs(sa) + h * fabs(ca);
    double wnew = h * fabs(sa) + w * fabs(ca);
    m_rot_mat = cv::getRotationMatrix2D(cv::Point2f(w / 2.0f, h / 2.0f), rad2deg(rot_rad), 1);
    m_rot_mat.at<double>(0, 2) += (wnew - w) * 0.5;
    m_rot_mat.at<double>(1, 2) += (hnew - h) * 0.5;
    m_rot_mat_inv = cv::Mat(m_rot_mat.size(), m_rot_mat.type());
    m_rot_mat_inv.colRange(0, 2) = m_rot_mat.colRange(0, 2).t();
    m_rot_mat_inv.col(2) = -m_rot_mat_inv.colRange(0, 2) * m_rot_mat.col(2);
    m_raw_size = raw_size;
    m_new_size = cv::Size(wnew, hnew);
  }

  /** @brief forward rotate image
   * @param[in]img: raw image to be rotated
   * @param[in]rot: image after rotate
   */
  void rot(const cv::Mat& img, cv::Mat& rot) { cv::warpAffine(img, rot, m_rot_mat, m_new_size); }

  /** @brief backward rotate image
   * @param[in]img: raw image to be rotated back
   * @param[in]rot: image after rotate back
   */
  void rotBack(const cv::Mat& img, cv::Mat& rot) { cv::warpAffine(img, rot, m_rot_mat_inv, m_raw_size); }

  /** @brief forward rotate 2D point in image */
  cv::Point2f rot(float x, float y) {
    std::vector<cv::Point2f> out;
    cv::transform(std::vector<cv::Point2f>(1, cv::Point2f(x, y)), out, m_rot_mat);
    return out[0];
  }

  /** @brief backward rotate 2D point in image */
  cv::Point2f rotBack(float x, float y) {
    std::vector<cv::Point2f> out;
    cv::transform(std::vector<cv::Point2f>(1, cv::Point2f(x, y)), out, m_rot_mat_inv);
    return out[0];
  }

  /** @brief image size before rotation */
  cv::Size rawSize() const { return m_raw_size; }

  /** @brief image size after rotation */
  cv::Size rotSize() const { return m_new_size; }

 private:
  cv::Mat m_rot_mat, m_rot_mat_inv;  ///< rotation affine matrix for rotate and rotate back
  cv::Size m_raw_size, m_new_size;   ///< image size before rotation and after rotation
};

/** @brief Fast image rotation
 * use matrix operation to quickly rotate image when rotate angle near at 0,90,180,270 degrees
 */
class FastImageRoter {
 public:
  /** @brief Construct
   * @param[in]raw_size: raw image size before rotation
   * @param[in]rot_rad: rotate angle in radians, positive values mean counter-clockwise rotation
   * @param[in]dead_zone: dead zone in radians to use matrix operation
   */
  FastImageRoter(const cv::Size& raw_size, float rot_rad, float dead_zone = VS_RAD10) {
    m_raw_size = raw_size;
    // calc rotate type: 0:general rot 1-4:fast rot
    m_rot_type = checkRotType(rot_rad, dead_zone);
    // calc final rot
    switch (m_rot_type) {
      case 1:  // rot 0 deg
        m_final_rot_rad = 0;
      case 2:  // rot 90 deg
        m_final_rot_rad = VS_PI_2;
      case 3:  // rot -90 deg
        m_final_rot_rad = -VS_PI_2;
      case 4:  // rot 180 deg
        m_final_rot_rad = VS_PI;
      default:
        m_final_rot_rad = rot_rad;
    }
    // calc image size after rotation and calc rot mat for general rot
    const int w = raw_size.width;
    const int h = raw_size.height;
    if (m_rot_type == 1 || m_rot_type == 4) {
      m_new_size = raw_size;
    } else if (m_rot_type == 2 || m_rot_type == 3) {
      m_new_size = cv::Size(raw_size.height, raw_size.width);
    } else {
      double sa = sin(rot_rad);
      double ca = cos(rot_rad);
      double hnew = w * fabs(sa) + h * fabs(ca);
      double wnew = h * fabs(sa) + w * fabs(ca);
      m_rot_mat = cv::getRotationMatrix2D(cv::Point2f(w / 2.0f, h / 2.0f), rad2deg(rot_rad), 1);
      m_rot_mat.at<double>(0, 2) += (wnew - w) * 0.5;
      m_rot_mat.at<double>(1, 2) += (hnew - h) * 0.5;
      m_rot_mat_inv = cv::Mat(m_rot_mat.size(), m_rot_mat.type());
      m_rot_mat_inv.colRange(0, 2) = m_rot_mat.colRange(0, 2).t();
      m_rot_mat_inv.col(2) = -m_rot_mat_inv.colRange(0, 2) * m_rot_mat.col(2);
      m_new_size = cv::Size(wnew, hnew);
    }
  }

  /** @brief get rotate angle in radians */
  float getRotAngle() { return m_final_rot_rad; }

  /** @brief forward rotate image
   * @param[in]img: raw image to be rotated
   * @param[in]rot: image after rotate
   */
  void rot(const cv::Mat& img, cv::Mat& rot) {
    switch (m_rot_type) {
      case 1:  // rot 0 deg
        rot = img;
        break;
      case 2:  // rot 90 deg
        cv::transpose(img, rot);
        cv::flip(rot, rot, 0);
        break;
      case 3:  // rot -90 deg
        cv::transpose(img, rot);
        cv::flip(rot, rot, 1);
        break;
      case 4:  // rot 180 deg
        flip(img, rot, -1);
        break;
      default:
        cv::warpAffine(img, rot, m_rot_mat, m_new_size);
        break;
    }
  }

  /** @brief backward rotate image
   * @param[in]img: raw image to be rotated back
   * @param[in]rot: image after rotate back
   */
  void rotBack(const cv::Mat& img, cv::Mat& rot) {
    switch (m_rot_type) {
      case 1:  // rot 0 deg
        rot = img;
        break;
      case 2:
        cv::transpose(img, rot);
        cv::flip(rot, rot, 1);
        break;
      case 3:
        cv::transpose(img, rot);
        cv::flip(rot, rot, 0);
        break;
      case 4:
        cv::flip(img, rot, -1);
        break;
      default:
        cv::warpAffine(img, rot, m_rot_mat_inv, m_raw_size);
        break;
    }
  }

  /** @brief forward rotate 2D point in image */
  cv::Point2f rot(float x, float y) {
    switch (m_rot_type) {
      case 1:  // rot 0 deg
        return cv::Point2f(x, y);
      case 2:  // rot 90 deg
        return cv::Point2f(y, m_new_size.height - x);
      case 3:  // rot -90 deg
        return cv::Point2f(m_new_size.width - y, x);
      case 4:  // rot 180 deg
        return cv::Point2f(m_new_size.width - x, m_new_size.height - y);
      default:
        std::vector<cv::Point2f> out;
        cv::transform(std::vector<cv::Point2f>(1, cv::Point2f(x, y)), out, m_rot_mat);
        return out[0];
    }
  }

  /** @brief backward rotate 2D point in image */
  cv::Point2f rotBack(float x, float y) {
    switch (m_rot_type) {
      case 1:  // rot 0 deg
        return cv::Point2f(x, y);
      case 2:  // rot 90 deg
        return cv::Point2f(m_raw_size.width - y, x);
      case 3:  // rot -90 deg
        return cv::Point2f(y, m_raw_size.height - x);
      case 4:  // rot 180 deg
        return cv::Point2f(m_raw_size.width - x, m_raw_size.height - y);
      default:
        std::vector<cv::Point2f> out;
        cv::transform(std::vector<cv::Point2f>(1, cv::Point2f(x, y)), out, m_rot_mat_inv);
        return out[0];
    }
  }

  /** @brief forward rotate 2D rect in image */
  template <typename T>
  cv::Rect_<T> rot(const cv::Rect_<T>& a) {
    switch (m_rot_type) {
      case 1:  // rot 0 deg
        return a;
      case 2:  // rot 90 deg
        return cv::Rect_<T>(a.y, m_new_size.height - a.x - a.width, a.height, a.width);
      case 3:  // rot -90 deg
        return cv::Rect_<T>(m_new_size.width - a.y - a.height, a.x, a.height, a.width);
      case 4:  // rot 180 deg
        return cv::Rect_<T>(m_new_size.width - a.x - a.width, m_new_size.height - a.y - a.height, a.width, a.height);
      default:
        return rotBox(a, m_rot_mat);
    }
  }

  /** @brief backward rotate 2D rect in image */
  template <typename T>
  cv::Rect_<T> rotBack(const cv::Rect_<T>& a) {
    switch (m_rot_type) {
      case 1:  // rot 0 deg
        return a;
      case 2:  // rot 90 deg
        return cv::Rect_<T>(m_raw_size.width - a.y - a.height, a.x, a.height, a.width);
      case 3:  // rot -90 deg
        return cv::Rect_<T>(a.y, m_raw_size.height - a.x - a.width, a.height, a.width);
      case 4:  // rot 180 deg
        return cv::Rect_<T>(m_raw_size.width - a.x - a.width, m_raw_size.height - a.y - a.height, a.width, a.height);
      default:
        return rotBox(a, m_rot_mat_inv);
    }
  }

  /** @brief forward rotate 2D direction vector in image */
  cv::Point2f rotDir(float x, float y) {
    switch (m_rot_type) {
      case 1:  // rot 0 deg
        return cv::Point2f(x, y);
      case 2:  // rot 90 deg
        return cv::Point2f(y, -x);
      case 3:  // rot -90 deg
        return cv::Point2f(-y, x);
      case 4:  // rot 180 deg
        return cv::Point2f(-x, -y);
    }
    float c = cos(m_final_rot_rad);
    float s = sin(m_final_rot_rad);
    return cv::Point2f(c * x + s * y, c * y - s * x);
  }

  /** @brief backward rotate 2D direction vector in image */
  cv::Point2f rotDirBack(float x, float y) {
    switch (m_rot_type) {
      case 1:  // rot 0 deg
        return cv::Point2f(x, y);
      case 2:  // rot 90 deg
        return cv::Point2f(-y, x);
      case 3:  // rot -90 deg
        return cv::Point2f(y, -x);
      case 4:  // rot 180 deg
        return cv::Point2f(-x, -y);
    }
    float c = cos(m_final_rot_rad);
    float s = sin(m_final_rot_rad);
    return cv::Point2f(c * x - s * y, c * y + s * x);
  }

  /** @brief image size before rotation */
  cv::Size rawSize() const { return m_raw_size; }

  /** @brief image size after rotation */
  cv::Size rotSize() const { return m_new_size; }

 private:
  int m_rot_type;                    ///< 0:general 1:0deg 2:90deg 3:-90deg 4:180deg
  float m_final_rot_rad;             ///< final rotate in radian, positive values mean counter-clockwise rotation
  cv::Mat m_rot_mat, m_rot_mat_inv;  ///< rotation affine matrix for rotate and rotate back
  cv::Size m_raw_size, m_new_size;   ///< image size before rotation and after rotation

  /** @brief check rotation type. 0:general 1:0deg 2:90deg 3:-90deg 4:180deg */
  int checkRotType(float rot_rad, float dead_zone) {
    rot_rad = normalizeRad(rot_rad);
    if (fabs(rot_rad) <= dead_zone)
      return 1;
    else if (fabs(rot_rad - VS_PI_2) <= dead_zone)
      return 2;
    else if (fabs(rot_rad + VS_PI_2) <= dead_zone)
      return 3;
    else if (fabs(rot_rad - VS_PI) <= dead_zone || fabs(rot_rad + VS_PI) <= dead_zone)
      return 4;
    return 0;
  }

  /** @brief rotate bounding box */
  template <typename T>
  cv::Rect_<T> rotBox(const cv::Rect_<T>& a, const cv::Mat& rot_mat) {
    std::vector<cv::Point2f> in = {cv::Point2f(a.x, a.y), cv::Point2f(a.x + a.width, a.y),
                                   cv::Point2f(a.x, a.y + a.height), cv::Point2f(a.x + a.width, a.y + a.height)};
    std::vector<cv::Point2f> out;
    cv::transform(in, out, rot_mat);
    float xmin = FLT_MAX, ymin = FLT_MAX, xmax = 0, ymax = 0;
    for (const auto& p : out) {
      if (p.x < xmin) xmin = p.x;
      if (p.x > xmax) xmax = p.x;
      if (p.y < ymin) ymin = p.y;
      if (p.y > ymax) ymax = p.y;
    }
    return cv::Rect_<T>(xmin, ymin, xmax - xmin, ymax - ymin);
  }
};

/** @brief crop image with rate
 * @param[in]img: input image
 * @param[in]crop_rate: crop rate range from (0, 1)
 * @param[in]deep_copy: whether do deep copy
 * @return crop subimage center at image center
 */
inline cv::Mat imcrop(const cv::Mat& img, float crop_rate, bool deep_copy = true) {
  if (crop_rate <= 0) return cv::Mat();
  int w = img.cols;
  int h = img.rows;
  int w2 = std::min(crop_rate, 1.0f) * w;
  int h2 = std::min(crop_rate, 1.0f) * h;
  cv::Mat sub_img = img(cv::Rect((w - w2) / 2, (h - h2) / 2, w2, h2));
  return deep_copy ? sub_img.clone() : sub_img;
}

/** @brief find bounding box with input point list.
 * same as cv::boundingRect, but return rect same type as input point
 */
template <typename T>
cv::Rect_<T> findBoundingRect(const std::vector<cv::Point_<T>>& pts) {
  if (pts.empty()) return cv::Rect_<T>();
  T xmin, xmax, ymin, ymax;
  xmin = xmax = pts[0].x;
  ymin = ymax = pts[0].y;
  for (const auto& p : pts) {
    if (p.x < xmin) {
      xmin = p.x;
    } else if (p.x > xmax) {
      xmax = p.x;
    }
    if (p.y < ymin) {
      ymin = p.y;
    } else if (p.y > ymax) {
      ymax = p.y;
    }
  }
  return cv::Rect_<T>(xmin, ymin, xmax - xmin, ymax - ymin);
}

/** @brief create contour with input rect, order:top-left, top-right, bottom-right, bottom-left
 * @param[in]r: rect
 * @return rect contour points
 */
template <typename T>
std::vector<cv::Point_<T>> rectContour(const cv::Rect_<T>& r) {
  return std::vector<cv::Point_<T>>({
      cv::Point_<T>(r.x, r.y),
      cv::Point_<T>(r.x + r.width, r.y),
      cv::Point_<T>(r.x + r.width, r.y + r.height),
      cv::Point_<T>(r.x, r.y + r.height),
  });
}

/** @brief crop rect inside image size region
 * @param[in]rect: input rect, maybe outside image region
 * @param[in]img_size: image size
 * @return crop rect which all corners inside image region
 */
template <typename T>
cv::Rect_<T> rectCrop(const cv::Rect_<T>& rect, const cv::Size& img_size) {
  T xmin = std::max(rect.x, (T)(0));
  T ymin = std::max(rect.y, (T)(0));
  T xmax = std::min(rect.x + rect.width, (T)(img_size.width));
  T ymax = std::min(rect.y + rect.height, (T)(img_size.height));
  return cv::Rect_<T>(xmin, ymin, xmax - xmin, ymax - ymin);
}

/** @brief padding rect. pad rect width/height = raw width/height + pad.
 * @param[in]rect: input rect
 * @param[in]pad: padding value for width and height
 * @return padding rect
 */
template <typename T>
cv::Rect_<T> rectPad(const cv::Rect_<T>& rect, T pad) {
  float cx = rect.x + rect.width * 0.5f;
  float cy = rect.y + rect.height * 0.5f;
  T new_w = rect.width + pad * 2;
  T new_h = rect.height + pad * 2;
  T x0 = (T)(cx - new_w * 0.5f);
  T y0 = (T)(cy - new_h * 0.5f);
  return cv::Rect_<T>(x0, y0, new_w, new_h);
}

/** @brief padding rect rate. pad rect width/height = raw width/height * (1 + 2 * pad_rate).
 * @param[in]rect: input rect
 * @param[in]pad_rate: pad_rate
 * @return padding rect
 */
template <typename T>
cv::Rect_<T> rectPadRate(const cv::Rect_<T>& rect, float pad_rate) {
  float k = 1 + pad_rate * 2;
  T new_w = rect.width * k;
  T new_h = rect.height * k;
  float cx = rect.x + rect.width * 0.5f;
  float cy = rect.y + rect.height * 0.5f;
  T x0 = (T)(cx - new_w * 0.5f);
  T y0 = (T)(cy - new_h * 0.5f);
  return cv::Rect_<T>(x0, y0, new_w, new_h);
}

/** @brief shift/move rect */
template <typename T>
cv::Rect_<T> rectShift(const cv::Rect_<T>& rect, T shift_x, T shift_y) {
  return cv::Rect_<T>(rect.x + shift_x, rect.y + shift_y, rect.width, rect.height);
}

/** @brief scale rect */
template <typename T>
cv::Rect_<T> rectScale(const cv::Rect_<T>& rect, float scale) {
  float cx = rect.x + rect.width * 0.5f;
  float cy = rect.y + rect.height * 0.5f;
  float w = rect.width * scale;
  float h = rect.height * scale;
  return cv::Rect_<T>(cx - w * 0.5, cy - h * 0.5, w, h);
}

/** @brief shift and scale rect */
template <typename T>
cv::Rect_<T> rectShiftScale(const cv::Rect_<T>& rect, T shift_x, T shift_y, float scale) {
  float cx = rect.x + rect.width * 0.5f;
  float cy = rect.y + rect.height * 0.5f;
  float w = rect.width * scale;
  float h = rect.height * scale;
  return cv::Rect_<T>(cx + shift_x - w * 0.5f, cy + shift_y - h * 0.5f, w, h);
}

/** @brief calculate IOU(Intersection over Union) between two rect */
template <typename T1, typename T2>
float rectIOU(const cv::Rect_<T1>& a, const cv::Rect_<T2>& b) {
  float xmin = std::max(a.x, b.x);
  float ymin = std::max(a.y, b.y);
  float xmax = std::min(a.x + a.width, b.x + b.width);
  float ymax = std::min(a.y + a.height, b.y + b.height);
  float inner_area = std::max(0.0f, xmax - xmin) * std::max(0.0f, ymax - ymin);
  float iou = inner_area / (a.width * a.height + b.width * b.height - inner_area);
  return iou;
}

/** @brief padding contour points in rate
 * new distance to contour center = (1 + pad_rate) * old distance to contour center
 */
template <typename T>
std::vector<cv::Point_<T>> contourPadRate(const std::vector<cv::Point_<T>>& contour, float pad_rate) {
  int n = contour.size();
  if (n < 2) return contour;
  cv::Point2f c(0, 0);
  for (const auto& p : contour) {
    c.x += p.x;
    c.y += p.y;
  }
  c.x /= n;
  c.y /= n;
  std::vector<cv::Point_<T>> res;
  for (const auto& p : contour) {
    res.push_back(cv::Point_<T>(p.x + (p.x - c.x) * pad_rate, p.y + (p.y - c.y) * pad_rate));
  }
  return res;
}

/** @brief sample discrete points in a line
 * @param[in]p1: first end-point of line
 * @param[in]p2: second end-point of line
 * @param[out]pts: sample points
 * @param[in]delta: sample distance step
 * @param[in]img_size: image size, if set, only sample point inside image, if not set, sample all points in line
 * @return int sample points count
 */
template <typename T>
int lineDiscreteSample(const cv::Point_<T>& p1, const cv::Point_<T>& p2, std::vector<cv::Point_<T>>& pts,
                       float delta = 1, const cv::Size& img_size = cv::Size()) {
  auto dp = p2 - p1;
  float len = hypotf(dp.x, dp.y);
  int n = len / delta;
  pts.clear();
  pts.reserve(n + 2);  // need to add two end-points p1 and p2
  if (n <= 0) {
    pts.push_back(p1);
    pts.push_back(p2);
  } else {
    float x = p1.x;
    float y = p1.y;
    float dx = dp.x / len * delta;
    float dy = dp.y / len * delta;
    if (img_size.area() > 0) {
      for (int i = 0; i <= n; i++) {
        cv::Point_<T> p(x, y);
        if (inside(p, img_size)) pts.push_back(p);
        x += dx;
        y += dy;
      }
      if (len - n * delta > 0.1 && inside(p2, img_size)) pts.push_back(p2);
    } else {
      for (int i = 0; i <= n; i++) {
        pts.push_back(cv::Point_<T>(x, y));
        x += dx;
        y += dy;
      }
      if (len - n * delta > 0.1) pts.push_back(p2);
    }
  }
  return pts.size();
}

/** @brief fitting 2D line segment from input points
 * @param[in]pts: input 2D discrete points
 * @param[in]dist_type: distance line type, see cv::fitLine
 * @return two end-point output fit line, (x1,y1,x2,y2)
 */
template <typename T>
cv::Vec4f lineSegFit(const std::vector<cv::Point_<T>>& pts, int dist_type = 2) {
  cv::Vec4f line_param;
  cv::fitLine(pts, line_param, dist_type, 0, 0.01, 0.01);
  float vx = line_param[0];
  float vy = line_param[1];
  float x0 = line_param[2];
  float y0 = line_param[3];
  float k1 = 0, k2 = 0;
  for (const auto& p : pts) {
    float a = (p.x - x0) * vx + (p.y - y0) * vy;
    if (a < k1)
      k1 = a;
    else if (a > k2)
      k2 = a;
  }
  return cv::Vec4f(x0 + k1 * vx, y0 + k1 * vy, x0 + k2 * vx, y0 + k2 * vy);
}

/** @brief find point index ids in pts which distance to line p1-p2 less than threshold
 * @param[in]pts: input points to check
 * @param[in]p1: first end-point of line
 * @param[in]p2: second end-point of line
 * @param[in]thres_dist: distance threshold, point which distance to line less than this value will be seen as inlier
 * @return index ids of inlier in pts
 */
template <typename T1, typename T2>
std::vector<int> lineInlierIds(const std::vector<cv::Point_<T1>>& pts, const cv::Point_<T2>& p1,
                               const cv::Point_<T2>& p2, float thres_dist) {
  std::vector<int> inlier_ids;
  cv::Point2f dir(p2.x - p1.x, p2.y - p1.y);
  float d = hypotf(dir.x, dir.y);
  if (d < VS_EPS) return inlier_ids;
  dir.x /= d;
  dir.y /= d;
  int n_pts = pts.size();
  for (int i = 0; i < n_pts; i++) {
    const auto& p = pts[i];
    cv::Point2f a(p.x - p1.x, p.y - p1.y);
    if (fabs(dir.cross(a)) < thres_dist) {
      inlier_ids.push_back(i);
    }
  }
  return inlier_ids;
}

/** @brief detect line in input points using RANSAC
 * @param[in]pts: input 2D points
 * @param[out]line: output detect line, valid only when return true
 * @param[in]thres_dist: distance threshold for inlier point to line
 * @param[in]max_ite: max RANSAC iteration
 * @param[in]confidence: confidence inlier rate to terminate iteration
 * @return true if line find
 */
template <typename T>
bool lineDetectRansac(const std::vector<cv::Point_<T>>& pts, cv::Vec4f& line, float thres_dist = 2, int max_ite = 100,
                      float confidence = 0.6) {
  int n_pts = pts.size();
  if (n_pts < 2) return false;
  int half_npt = n_pts / 2;
  float npt_thres = n_pts * confidence;

  // ransac
  std::vector<int> best_inlier_idx;
  for (int ite = 0; ite < max_ite; ite++) {
    const auto& p1 = pts[randi(0, half_npt)];
    const auto& p2 = pts[randi(half_npt, n_pts)];
    auto inlier_idx = lineInlierIds(pts, p1, p2, thres_dist);
    if (inlier_idx.size() > best_inlier_idx.size()) {
      best_inlier_idx = inlier_idx;
      if (static_cast<int>(best_inlier_idx.size()) > npt_thres) break;
    }
  }
  if (best_inlier_idx.size() < 2) return false;

  // fit line with all inlier
  std::vector<cv::Point_<T>> inlier_pts;
  inlier_pts.reserve(best_inlier_idx.size());
  for (auto i : best_inlier_idx) inlier_pts.push_back(pts[i]);
  line = lineSegFit(inlier_pts);
  return true;
}

/** @brief detect line in input points using RANSAC
 * @param[in]pts: input 2D points
 * @param[out]line: output detect line, valid only when return true
 * @param[out]status: inlier status
 * @param[in]thres_dist: distance threshold for inlier point to line
 * @param[in]max_ite: max RANSAC iteration
 * @param[in]confidence: confidence inlier rate to terminate iteration
 * @return true if line find
 */
template <typename T>
bool lineDetectRansac(const std::vector<cv::Point_<T>>& pts, cv::Vec4f& line, std::vector<uint8_t>& status,
                      float thres_dist = 2, int max_ite = 100, float confidence = 0.6) {
  int n_pts = pts.size();
  if (n_pts < 2) return false;
  int half_npt = n_pts / 2;
  float npt_thres = n_pts * confidence;

  // ransac
  std::vector<int> best_inlier_idx;
  for (int ite = 0; ite < max_ite; ite++) {
    const auto& p1 = pts[randi(0, half_npt)];
    const auto& p2 = pts[randi(half_npt, n_pts)];
    auto inlier_idx = lineInlierIds(pts, p1, p2, thres_dist);
    if (inlier_idx.size() > best_inlier_idx.size()) {
      best_inlier_idx = inlier_idx;
      if (static_cast<int>(best_inlier_idx.size()) > npt_thres) break;
    }
  }
  if (best_inlier_idx.size() < 2) return false;
  // fit line with all inlier
  std::vector<cv::Point_<T>> inlier_pts;
  inlier_pts.reserve(best_inlier_idx.size());
  for (auto i : best_inlier_idx) inlier_pts.push_back(pts[i]);
  line = lineSegFit(inlier_pts);

  // set status
  best_inlier_idx = lineInlierIds(pts, cv::Point2f(line[0], line[1]), cv::Point2f(line[2], line[3]), thres_dist);
  status = std::vector<uint8_t>(n_pts, 0);
  for (auto i : best_inlier_idx) status[i] = 255;

  // refine twice
  if (0) {
    inlier_pts.clear();
    inlier_pts.reserve(best_inlier_idx.size());
    for (auto i : best_inlier_idx) inlier_pts.push_back(pts[i]);
    line = lineSegFit(inlier_pts);
  }
  return true;
}

/** @brief multiply points */
template <typename T>
void resizePointList(std::vector<cv::Point_<T>>& polygon, float kx, float ky = 0) {
  if (ky <= 0) ky = kx;
  for (auto& p : polygon) {
    p.x *= kx;
    p.y *= ky;
  }
}

/** @brief multiply points with source image size and target image size */
template <typename T>
void resizePointList(std::vector<cv::Point_<T>>& polygon, const cv::Size& src_size, const cv::Size& tar_size) {
  float ratio_x = static_cast<float>(tar_size.width) / src_size.width;
  float ratio_y = static_cast<float>(tar_size.height) / src_size.height;
  for (auto& p : polygon) {
    p.x *= ratio_x;
    p.y *= ratio_y;
  }
}

/** @brief calculate bounding box for input x,y */
template <typename T>
class BoundingRect {
 public:
  void add(T x, T y) {
    if (first_data_) {
      first_data_ = false;
      xmin_ = xmax_ = x;
      ymin_ = ymax_ = y;
    } else {
      if (x < xmin_)
        xmin_ = x;
      else if (x > xmax_)
        xmax_ = x;
      if (y < ymin_)
        ymin_ = y;
      else if (y > ymax_)
        ymax_ = y;
    }
  }

  T xmin() const { return xmin_; }

  T xmax() const { return xmax_; }

  T ymin() const { return ymin_; }

  T ymax() const { return ymax_; }

  T width() const { return xmax_ - xmin_; }

  T height() const { return ymax_ - ymin_; }

  cv::Rect_<T> rect() const { return cv::Rect_<T>(xmin_, ymin_, xmax_ - xmin_, ymax_ - ymin_); }

 private:
  bool first_data_ = true;
  T xmin_ = 0, xmax_ = 0, ymin_ = 0, ymax_ = 0;
};

/** @brief Generate mesh grid 2D point list
 * @param[in]xmin: min x, include
 * @param[in]xmax: max x, exclude
 * @param[in]ymin: min y, include
 * @param[in]ymax: max y, exclude
 * @param[in]xstep: incremental step x
 * @param[in]ystep: incremantal step y
 * @return point list, order: (x0,y0) (x0,y1) ... (x0,yn) (x1,y0) (x1,y1) ... (x1,yn) ......(xm, yn)
 */
template <typename T>
std::vector<cv::Point_<T>> meshGrid2D(T xmin, T xmax, T ymin, T ymax, T xstep = 1, T ystep = 1) {
  std::vector<cv::Point_<T>> pts;
  pts.reserve(((xmax - xmin) / xstep) * ((ymax - ymin) / ystep));
  for (T x = xmin; x < xmax; x += xstep)
    for (T y = ymin; y < ymax; y += ystep) pts.push_back(cv::Point_<T>(x, y));
  return pts;
}

/** @brief Generate mesh grid 3D point list
 * @param[in]xmin: min x, include
 * @param[in]xmax: max x, exclude
 * @param[in]ymin: min y, include
 * @param[in]ymax: max y, exclude
 * @param[in]zmin: min z, include
 * @param[in]zmax: max z, exclude
 * @param[in]xstep: incremental step x
 * @param[in]ystep: incremantal step y
 * @param[in]zstep: incremantal step z
 * @return point list, order: (x0,y0,z0) (x0,y0,z1) ... (x0,y0,zn) ...... (x0,ym,zn) ...... (xl,ym,zn)
 */
template <typename T>
std::vector<cv::Point3_<T>> meshGrid3D(T xmin, T xmax, T ymin, T ymax, T zmin, T zmax, T xstep = 1, T ystep = 1,
                                       T zstep = 1) {
  std::vector<cv::Point3_<T>> pts;
  pts.reserve(((xmax - xmin) / xstep) * ((ymax - ymin) / ystep) * ((zmax - zmin) / zstep));
  for (T x = xmin; x < xmax; x += xstep)
    for (T y = ymin; y < ymax; y += ystep)
      for (T z = zmin; z < zmax; z += zstep) pts.push_back(cv::Point3_<T>(x, y, z));
  return pts;
}

/**
 * @brief convert 2D convex quadrilateral to ellipse. see: https://static.laszlokorte.de/quad/
 * @param[in]q1: first corner of convex quadrilateral
 * @param[in]q2: second corner of convex quadrilateral
 * @param[in]q3: third corner of convex quadrilateral
 * @param[in]q4: fourth corner of convex quadrilateral
 * @param[out]center: ellipse center
 * @param[out]radius: ellipse radius
 * @param[out]angle: angle from x axis to ellipse first radius, [radians]
 * @return true if conversion ok
 */
template <typename T>
bool quadToEllipse(const cv::Point_<T>& q1, const cv::Point_<T>& q2, const cv::Point_<T>& q3, const cv::Point_<T>& q4,
                   cv::Point2f& center, cv::Vec2f& radius, float& angle) {
  // Reconstruct matrix that transforms the unit square ((-1,-1), (1,1)) into quad (W,X,Y,Z)
  double m00 = q1.x * q2.x * q3.y - q4.x * q2.x * q3.y - q1.x * q2.y * q3.x + q4.x * q2.y * q3.x - q4.x * q1.y * q3.x +
               q4.y * q1.x * q3.x + q4.x * q1.y * q2.x - q4.y * q1.x * q2.x;
  double m01 = q4.x * q2.x * q3.y - q4.x * q1.x * q3.y - q1.x * q2.y * q3.x + q1.y * q2.x * q3.x - q4.y * q2.x * q3.x +
               q4.y * q1.x * q3.x + q4.x * q1.x * q2.y - q4.x * q1.y * q2.x;
  double m02 = q1.x * q2.x * q3.y - q4.x * q1.x * q3.y - q4.x * q2.y * q3.x - q1.y * q2.x * q3.x + q4.y * q2.x * q3.x +
               q4.x * q1.y * q3.x + q4.x * q1.x * q2.y - q4.y * q1.x * q2.x;
  double m10 = q1.y * q2.x * q3.y - q4.y * q2.x * q3.y - q4.x * q1.y * q3.y + q4.y * q1.x * q3.y - q1.y * q2.y * q3.x +
               q4.y * q2.y * q3.x + q4.x * q1.y * q2.y - q4.y * q1.x * q2.y;
  double m11 = -q1.x * q2.y * q3.y + q4.x * q2.y * q3.y + q1.y * q2.x * q3.y - q4.x * q1.y * q3.y - q4.y * q2.y * q3.x +
               q4.y * q1.y * q3.x + q4.y * q1.x * q2.y - q4.y * q1.y * q2.x;
  double m12 = q1.x * q2.y * q3.y - q4.x * q2.y * q3.y + q4.y * q2.x * q3.y - q4.y * q1.x * q3.y - q1.y * q2.y * q3.x +
               q4.y * q1.y * q3.x + q4.x * q1.y * q2.y - q4.y * q1.y * q2.x;
  double m20 =
      q1.x * q3.y - q4.x * q3.y - q1.y * q3.x + q4.y * q3.x - q1.x * q2.y + q4.x * q2.y + q1.y * q2.x - q4.y * q2.x;
  double m21 =
      q2.x * q3.y - q1.x * q3.y - q2.y * q3.x + q1.y * q3.x + q4.x * q2.y - q4.y * q2.x - q4.x * q1.y + q4.y * q1.x;
  double m22 =
      q2.x * q3.y - q4.x * q3.y - q2.y * q3.x + q4.y * q3.x + q1.x * q2.y - q1.y * q2.x + q4.x * q1.y - q4.y * q1.x;

  // invert matrix
  double determinant = +m00 * (m11 * m22 - m21 * m12) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20);
  if (determinant == 0) return false;
  double invdet = 1 / determinant;
  double J = (m11 * m22 - m21 * m12) * invdet;
  double K = -(m01 * m22 - m02 * m21) * invdet;
  double L = (m01 * m12 - m02 * m11) * invdet;
  double M = -(m10 * m22 - m12 * m20) * invdet;
  double N = (m00 * m22 - m02 * m20) * invdet;
  double O = -(m00 * m12 - m10 * m02) * invdet;
  double P = (m10 * m21 - m20 * m11) * invdet;
  double Q = -(m00 * m21 - m20 * m01) * invdet;
  double R = (m00 * m11 - m10 * m01) * invdet;

  // extract ellipse coefficients from matrix
  double a = J * J + M * M - P * P;
  double b = J * K + M * N - P * Q;
  double c = K * K + N * N - Q * Q;
  double d = J * L + M * O - P * R;
  double f = K * L + N * O - Q * R;
  double g = L * L + O * O - R * R;

  // deduce ellipse center from coefficients
  double bbac = b * b - a * c;
  if (bbac == 0) return false;
  center.x = (c * d - b * f) / bbac;
  center.y = (a * f - b * d) / bbac;

  // deduce ellipse radius from coefficients
  double tmp_sqrt = std::sqrt((a - c) * (a - c) + 4 * b * b);
  double tmp_deno0 = (bbac * (tmp_sqrt - (a + c)));
  double tmp_deno1 = (bbac * (-tmp_sqrt - (a + c)));
  double tmp_num = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g);
  if (tmp_deno0 == 0 || tmp_deno1 == 0) return false;
  double tmp0 = tmp_num / tmp_deno0;
  double tmp1 = tmp_num / tmp_deno1;
  if (tmp0 <= 0 || tmp1 <= 0) return false;
  radius[0] = std::sqrt(tmp0);
  radius[1] = std::sqrt(tmp1);

  // deduce ellipse rotation from coefficients
  angle = 0;
  if (b == 0 && a <= c) {
    angle = 0;
  } else if (b == 0 && a >= c) {
    angle = VS_PI_2;
  } else if (b != 0 && a > c) {
    angle = VS_PI_2 + 0.5 * (VS_PI_2 - std::atan2((a - c), (2 * b)));
  } else if (b != 0 && a <= c) {
    angle = VS_PI_2 + 0.5 * (VS_PI_2 - std::atan2((a - c), (2 * b)));
  }
  return true;
}

} /* namespace vs */
