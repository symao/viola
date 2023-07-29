/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details macros, numerics utils, vector utils, string utils
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <numeric>
#include <fstream>

#define VS_WNO_UNUSED_FUNCTION 1
#if (VS_WNO_UNUSED_FUNCTION && !defined(WIN32))
#define VS_STATIC_FUNC __attribute__((unused)) static
#else
#define VS_STATIC_FUNC static
#endif

#define VS_EPS 0.000001                   ///< epsilon, use to check zero
#define VS_E 2.7182818284590452354        ///< e, natural logarithm
#define VS_LOG2E 1.4426950408889634074    ///< log 2e
#define VS_LOG10E 0.43429448190325182765  ///< log 10e
#define VS_LN2 0.69314718055994530942     ///< log e2
#define VS_LN10 2.30258509299404568402    ///< log e10
#define VS_SQRT2 1.41421356237309504880   ///< sqrt(2)
#define VS_SQRT3 1.7320508075688771932    ///< sqrt(3)

#define VS_PI_2 1.5707963267948965580  ///< pi/2
#define VS_PI_4 0.7853981633974482790  ///< pi/4
#define VS_PI_8 0.3926990816987241395  ///< pi/8
#define VS_PI 3.1415926535897931160    ///< pi
#define VS_2PI 6.2831853071795862320   ///< 2*pi
#define VS_1_PI 0.3183098861837906912  ///< 1/pi
#define VS_2_PI 0.6366197723675813824  ///< 2/pi

#define VS_RAD2DEG 57.2957795130823228646  ///< 180/pi
#define VS_DEG2RAD 0.0174532925199432955   ///< pi/180

#define VS_RAD1 0.0174532925199432955    ///< 1*pi/180
#define VS_RAD5 0.0872664625997164739    ///< 5*pi/180
#define VS_RAD10 0.1745329251994329478   ///< 10*pi/180
#define VS_RAD20 0.3490658503988658956   ///< 20*pi/180
#define VS_RAD30 0.5235987755982988157   ///< 30*pi/180
#define VS_RAD45 0.7853981633974482790   ///< 45*pi/180
#define VS_RAD60 1.0471975511965976313   ///< 60*pi/180
#define VS_RAD90 1.5707963267948965580   ///< 90*pi/180
#define VS_RAD120 2.0943951023931952626  ///< 120*pi/180

#define VS_COS_DEG1 0.9998476951563912696   ///< cos(1*pi/180)
#define VS_COS_DEG5 0.9961946980917455452   ///< cos(5*pi/180)
#define VS_COS_DEG10 0.9848077530122080203  ///< cos(10*pi/180)
#define VS_COS_DEG20 0.9396926207859084279  ///< cos(20*pi/180)
#define VS_COS_DEG30 0.8660254037844387076  ///< cos(30*pi/180)
#define VS_COS_DEG40 0.7660444431189780135  ///< cos(40*pi/180)
#define VS_COS_DEG50 0.6427876096865393629  ///< cos(50*pi/180)
#define VS_COS_DEG60 0.5                    ///< cos(60*pi/180)
#define VS_COS_DEG70 0.3420201433256688239  ///< cos(70*pi/180)
#define VS_COS_DEG80 0.1736481776669304145  ///< cos(80*pi/180)
#define VS_COS_DEG85 0.0871557427476581381  ///< cos(85*pi/180)
#define VS_COS_DEG89 0.0174524064372835984  ///< cos(89*pi/180)

#define VS_SIN_DEG1 VS_COS_DEG89   ///< sin(1*pi/180)
#define VS_SIN_DEG5 VS_COS_DEG85   ///< sin(5*pi/180)
#define VS_SIN_DEG10 VS_COS_DEG80  ///< sin(10*pi/180)
#define VS_SIN_DEG20 VS_COS_DEG70  ///< sin(20*pi/180)
#define VS_SIN_DEG30 VS_COS_DEG60  ///< sin(30*pi/180)
#define VS_SIN_DEG40 VS_COS_DEG50  ///< sin(40*pi/180)
#define VS_SIN_DEG50 VS_COS_DEG40  ///< sin(50*pi/180)
#define VS_SIN_DEG60 VS_COS_DEG30  ///< sin(60*pi/180)
#define VS_SIN_DEG70 VS_COS_DEG20  ///< sin(70*pi/180)
#define VS_SIN_DEG80 VS_COS_DEG10  ///< sin(80*pi/180)
#define VS_SIN_DEG85 VS_COS_DEG5   ///< sin(85*pi/180)
#define VS_SIN_DEG89 VS_COS_DEG1   ///< sin(89*pi/180)

#define VS_ASCII_ESC 27    ///< ASCII code of 'ESC'
#define VS_ASCII_SPACE 32  ///< ASCII code of 'SPACE'

#define VS_INT(x) static_cast<int>(x)
#define VS_FLOAT(x) static_cast<float>(x)
#define VS_DOUBLE(x) static_cast<double>(x)

#define VS_MS_TO_SEC 1e-3
#define VS_US_TO_SEC 1e-6
#define VS_NS_TO_SEC 1e-9
#define VS_SEC_TO_MS 1e3
#define VS_SEC_TO_US 1e6
#define VS_SEC_TO_NS 1e9

namespace vs {

/*======================== Numeric Math ===============================*/
/** @brief check equal between two float numbers. */
inline bool fequal(double a, double b, double eta = VS_EPS) { return fabs(a - b) < eta; }

/** @brief check sign of number. 1 means non-negative -1 means negative */
inline int sign(double d) { return (d >= 0) ? 1 : -1; }

/** @brief calc square */
template <typename T>
T sq(T a) {
  return a * a;
}

/** @brief check number in range */
template <class T1, class T2, class T3>
bool inRange(T1 a, T2 amin, T3 amax) {
  return amin <= a && a <= amax;
}

/** @brief check number in interval [amin, amax] */
template <class T1, class T2, class T3>
bool inIntervalLR(T1 a, T2 amin, T3 amax) {
  return amin <= a && a <= amax;
}

/** @brief check number in interval (amin, amax] */
template <class T1, class T2, class T3>
bool inIntervalR(T1 a, T2 amin, T3 amax) {
  return amin < a && a <= amax;
}

/** @brief check number in interval [amin, amax) */
template <class T1, class T2, class T3>
bool inIntervalL(T1 a, T2 amin, T3 amax) {
  return amin <= a && a < amax;
}

/** @brief check number in interval (amin, amax) */
template <class T1, class T2, class T3>
bool inInterval(T1 a, T2 amin, T3 amax) {
  return amin < a && a < amax;
}

/** @brief minimum number */
template <typename T>
T min3(T a, T b, T c) {
  T t = a < b ? a : b;
  return t < c ? t : c;
}

/** @brief maximum number */
template <typename T>
T max3(T a, T b, T c) {
  T t = a > b ? a : b;
  return t > c ? t : c;
}

/** @brief minimum number */
template <typename T>
T min4(T a, T b, T c, T d) {
  T x = a < b ? a : b;
  T y = c < d ? c : d;
  return x < y ? x : y;
}

/** @brief maximum number */
template <typename T>
T max4(T a, T b, T c, T d) {
  T x = a > b ? a : b;
  T y = c > d ? c : d;
  return x > y ? x : y;
}

/** @brief square sum, a^2+b^2 */
template <typename T>
T sqsum(T a, T b) {
  return a * a + b * b;
}

/** @brief square sum, a^2+b^2 */
template <typename T>
T sqsum2(T a, T b) {
  return a * a + b * b;
}

/** @brief square sum, a^2+b^2+c^2 */
template <typename T>
T sqsum3(T a, T b, T c) {
  return a * a + b * b + c * c;
}

/** @brief square sum, a^2+b^2+c^2+d^2 */
template <typename T>
T sqsum4(T a, T b, T c, T d) {
  return a * a + b * b + c * c + d * d;
}

/** @brief root of square sum, sqrt(a^2+b^2+c^2) */
template <typename T>
double hypot3(T a, T b, T c) {
  return std::sqrt(sqsum3(a, b, c));
}

/** @brief root of square sum, sqrt(a^2+b^2+c^2+d^2) */
template <typename T>
double hypot4(T a, T b, T c, T d) {
  return std::sqrt(sqsum4(a, b, c, d));
}

/** @brief clip number to min-max range */
template <class T1, class T2, class T3>
T1 clip(T1 val, T2 min_val, T3 max_val) {
  return val < min_val ? min_val : (val > max_val ? max_val : val);
}

/** @brief Normalize angle in rad to range (-pi, pi]
 * @param[in] angle input angle in rad
 * @return normalized angle in (-pi, pi] equal to input angle
 */
inline double normalizeRad(double angle) {
  angle = fmod(angle, VS_2PI);
  if (angle <= -VS_PI)
    angle += VS_2PI;
  else if (angle > VS_PI)
    angle -= VS_2PI;
  return angle;
}

/** @brief Normalize angle in deg to range (-180, 180]
 * @param[in] angle input angle in deg
 * @return normalized angle in (-180, 180] equal to input angle
 */
inline double normalizeDeg(double theta) {
  theta = fmod(theta, 360);
  if (theta <= -180)
    theta += 360;
  else if (theta > 180)
    theta -= 360;
  return theta;
}

/** @brief calculate norm for list */
template <typename T>
double listnorm(T* a, int n) {
  double res = 0;
  for (int i = 0; i < n; i++) res += sq(a[i]);
  return std::sqrt(res);
}

/** @brief normalize list value */
template <typename T>
void normalizeList(T* a, int n) {
  double norm = listnorm(a, n);
  if (norm > VS_EPS) {
    for (int i = 0; i < n; i++) a[i] /= norm;
  }
}

/** @brief convert angle from unit deg to unit rad
 * @param[in] angle input angle in deg
 * @return angle in rad equivalent to input angle
 */
inline double deg2rad(double angle) { return angle * VS_DEG2RAD; }

/** @brief convert angle from unit rad to unit deg
 * @param[in] angle input angle in rad
 * @return angle in deg equivalent to input angle
 */
inline double rad2deg(double angle) { return angle * VS_RAD2DEG; }

/** @brief linear interpolation for timestamped data
 * @param[in]t1: timestamp for first data
 * @param[in]p1: value for first data
 * @param[in]t2: timestamp for second data
 * @param[in]p2: value for second data
 * @param[in]t: timetamp in range [t1,t2]
 * @param[out]p: interpolation value at timestamp t
 * @return true if t is in [t1,t2] and success
 */
template <class T>
bool lerp(double t1, const T& p1, double t2, const T& p2, double t, T& p) {
  if (t1 > t2) {
    return lerp(t2, p2, t1, p1, t, p);
  } else if (t1 <= t && t <= t2) {
    double dt = t2 - t1;
    double k = dt < VS_EPS ? 0.5 : (t - t1) / dt;
    p = (1 - k) * p1 + k * p2;
    return true;
  } else {
    printf("[ERROR]lerp: bad input, t must range [t1,t2].\n");
    return false;
  }
}

/** @brief Solve linear equation y=ax+b.
 * Given 2 points (x1, y1), (x2, y2) in line and x, find the y in line corresponding to x.
 * @param[in] x1 x of first point(x1, y1)
 * @param[in] y1 y of first point(x1, y1)
 * @param[in] x2 x of second point(x2, y2)
 * @param[in] y2 y of second point(x2, y2)
 * @param[in] x the x to be solved
 * @return y y coresponding to x. (y-y1)/(x-x1) = (y2-y1)/(x2-x1) = (y-y2)/(x-x2);
 */
inline double linearInter(double x1, double y1, double x2, double y2, double x) {
  if (fequal(x1, x2)) {
    printf("[ERROR] linearInter: assert x1 != x2\n");
    return 0;
  }
  return (y2 - y1) / (x2 - x1) * (x - x1) + y1;
}

/** @brief Solve linear equation y=ax+b, clip to [yl,yr]
 * Given 2 points (x1, y1), (x2, y2) in line and x, find the y in line corresponding to x.
 * @param[in] x1 x of first point(x1, y1)
 * @param[in] y1 y of first point(x1, y1)
 * @param[in] x2 x of second point(x2, y2)
 * @param[in] y2 y of second point(x2, y2)
 * @param[in] x the x to be solved
 * @return y y coresponding to x, clip to [yl,yr]. (y-y1)/(x-x1) = (y2-y1)/(x2-x1) = (y-y2)/(x-x2);
 */
inline double linearClip(double xl, double yl, double xr, double yr, double x) {
  if (xl <= xr) {
    if (x <= xl)
      return yl;
    else if (x >= xr)
      return yr;
  } else {
    if (x <= xr)
      return yr;
    else if (x >= xl)
      return yl;
  }
  return linearInter(xl, yl, xr, yr, x);
}

/** @brief Solve piecewise function with multiple linear equations
 * @param[in] xs list of x with ascend order
 * @param[in] ys list of x corresponding to xs
 * @param[in] x to interpolate
 * @return y y coresponding to x
 */
inline double linearClipMulti(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
  if (xs.size() < 2 || xs.size() != ys.size()) return 0;
  if (x <= xs[0]) return ys[0];
  for (size_t i = 1; i < xs.size(); i++) {
    const auto& xi = xs[i];
    if (x <= xi) return linearInter(xs[i - 1], ys[i - 1], xi, ys[i], x);
  }
  return ys.back();
}

/** @brief substract and normalization for radians */
inline double radSub(double rad_a, double rad_b) { return normalizeRad(rad_a - rad_b); }

/** @brief normalized weighted sum for two radians */
inline double radAddW(double k1, double rad_a, double k2, double rad_b) {
  double r2 = rad_a + normalizeRad(rad_b - rad_a);
  return normalizeRad(k1 * rad_a + k2 * r2);
}

/** @return acute angle between two radians
 * @return range from(0, pi/2]
 */
inline double radDiffAcute(double rad_a, double rad_b) {
  double a = fabs(normalizeRad(rad_a - rad_b));
  return a > VS_PI_2 ? VS_PI - a : a;
}

template <typename T>
std::vector<std::pair<int, int>> matRowMax(const T* data, int rows, int cols, T thres, bool cross_check = false) {
  std::vector<int> row_ids(rows, -1);
  std::vector<int> col_ids(cols, -1);
  const T* ptr = data;
  for (int i = 0; i < rows; i++) {
    int max_j = -1;
    T max_v = thres;
    for (int j = 0; j < cols; j++) {
      T v = *ptr++;
      if (v > max_v) {
        max_v = v;
        max_j = j;
      }
    }
    if (max_j >= 0) row_ids[i] = max_j;
  }
  if (cross_check) {
    for (int i = 0; i < rows; i++) {
      int& max_j = row_ids[i];
      if (max_j < 0) continue;
      int max_i = col_ids[max_j];
      if (max_i < 0) {
        max_i = i;
        T max_v = data[max_i * cols + max_j];
        const T* ptr = data + max_j;
        for (int i = 0; i < rows; i++, ptr += cols) {
          if (*ptr > max_v) {
            max_v = *ptr;
            max_i = i;
          }
        }
        col_ids[max_j] = max_i;
      }
      if (max_i != i) max_j = -1;
    }
  }
  std::vector<std::pair<int, int>> res;
  res.reserve(rows);
  for (int i = 0; i < rows; i++) {
    int j = row_ids[i];
    if (j >= 0) res.push_back(std::make_pair(i, j));
  }
  return res;
}

template <typename T>
std::vector<std::pair<int, int>> matRowMin(const T* data, int rows, int cols, T thres, bool cross_check = false) {
  std::vector<int> row_ids(rows, -1);
  std::vector<int> col_ids(cols, -1);
  const T* ptr = data;
  for (int i = 0; i < rows; i++) {
    int min_j = -1;
    T min_v = thres;
    for (int j = 0; j < cols; j++) {
      T v = *ptr++;
      if (v < min_v) {
        min_v = v;
        min_j = j;
      }
    }
    if (min_j >= 0) row_ids[i] = min_j;
  }
  if (cross_check) {
    for (int i = 0; i < rows; i++) {
      int& min_j = row_ids[i];
      if (min_j < 0) continue;
      int min_i = col_ids[min_j];
      if (min_i < 0) {
        min_i = i;
        T min_v = data[min_i * cols + min_j];
        const T* ptr = data + min_j;
        for (int i = 0; i < rows; i++, ptr += cols) {
          if (*ptr < min_v) {
            min_v = *ptr;
            min_i = i;
          }
        }
        col_ids[min_j] = min_i;
      }
      if (min_i != i) min_j = -1;
    }
  }
  std::vector<std::pair<int, int>> res;
  res.reserve(rows);
  for (int i = 0; i < rows; i++) {
    int j = row_ids[i];
    if (j >= 0) res.push_back(std::make_pair(i, j));
  }
  return res;
}

/*======================== End of Numeric Math ========================*/

/*======================== Vector Utils ===============================*/

/** @brief minimum value in vector */
template <class T>
T vecMin(const std::vector<T>& a) {
  T m = 0;
  if (a.empty()) return m;
  m = a[0];
  for (auto i : a)
    if (m > i) m = i;
  return m;
}

/** @brief maximum value in vector */
template <class T>
T vecMax(const std::vector<T>& a) {
  T m = 0;
  if (a.empty()) return m;
  m = a[0];
  for (auto i : a)
    if (m < i) m = i;
  return m;
}

/** @brief index of minimum value in vector */
template <class T>
int vecArgMin(const std::vector<T>& a) {
  if (a.empty()) return -1;
  T min_v = a[0];
  int min_idx = 0;
  for (size_t i = 1; i < a.size(); i++) {
    float v = a[i];
    if (v < min_v) {
      min_v = v;
      min_idx = i;
    }
  }
  return min_idx;
}

/** @brief index of maximum value in vector */
template <class T>
int vecArgMax(const std::vector<T>& a) {
  if (a.empty()) return -1;
  T max_v = a[0];
  int max_idx = 0;
  for (size_t i = 1; i < a.size(); i++) {
    float v = a[i];
    if (v > max_v) {
      max_v = v;
      max_idx = i;
    }
  }
  return max_idx;
}

/** @brief index of minimum function value in vector */
template <class T, class Func>
int vecArgMin(const std::vector<T>& a, Func func) {
  if (a.empty()) return -1;
  float min_v = func(a[0]);
  int min_idx = 0;
  for (size_t i = 1; i < a.size(); i++) {
    float v = func(a[i]);
    if (v < min_v) {
      min_v = a[i];
      min_idx = i;
    }
  }
  return min_idx;
}

/** @brief index of maximum function value in vector */
template <class T, class Func>
int vecArgMax(const std::vector<T>& a, Func func) {
  if (a.empty()) return -1;
  float max_v = func(a[0]);
  int max_idx = 0;
  for (size_t i = 1; i < a.size(); i++) {
    float v = func(a[i]);
    if (v > max_v) {
      max_v = v;
      max_idx = i;
    }
  }
  return max_idx;
}

/** @brief minimum and maximum value in vector */
template <class T>
void vecMinMax(const std::vector<T>& a, T& amin, T& amax) {
  if (a.empty()) return;
  amin = amax = a[0];
  for (auto i : a) {
    if (amin > i)
      amin = i;
    else if (amax < i)
      amax = i;
  }
}

/** @brief index of minimum and maximum value in vector */
template <class T>
void vecArgMinMax(const std::vector<T>& a, int& min_idx, int& max_idx) {
  if (a.empty()) return;
  T amin, amax;
  amin = amax = a[0];
  min_idx = max_idx = 0;
  for (size_t i = 0; i < a.size(); i++) {
    const T& a_i = a[i];
    if (a_i < amin) {
      amin = a_i;
      min_idx = i;
    } else if (a_i > amax) {
      amax = a_i;
      max_idx = i;
    }
  }
}

/** @brief is all value in vecor zero. */
template <class T>
bool vecAllZero(const std::vector<T>& a) {
  for (size_t i = 0; i < a.size(); i++)
    if (fabs(static_cast<double>(a[i])) > VS_EPS) return false;
  return true;
}

/** @brief find the kth largest number in vector. */
template <class T>
T findKth(std::vector<T> vec, int k) {
  if (vec.empty()) return T();
  int n = vec.size();
  if (k >= n || k <= -n) return T();
  if (k < 0) k = n + k;
  std::nth_element(vec.begin(), vec.begin() + k, vec.end());
  return vec[k];
}

/** @brief find the kth largest number in vector. this will modify vec. */
template <class T>
T findKthInPlace(std::vector<T>& vec, int k) {
  std::nth_element(vec.begin(), vec.begin() + k, vec.end());
  return vec[k];
}

/** @brief find medium value in vector. */
template <class T>
T vecMedian(const std::vector<T>& vec) {
  return findKth(vec, vec.size() / 2);
}

/** @brief find medium value in vector. this will modify vec. */
template <class T>
T vecMedianInPlace(std::vector<T>& vec) {
  return findKth(vec, vec.size() / 2);
}

/** @brief calculate norm of vector, 0:L0 norm 1:L1 norm 2:L2 norm -1:L_oo norm */
template <class T>
double vecNorm(const std::vector<T>& a, int l = 2) {
  double res = 0;
  switch (l) {
    case 0:
      for (auto i : a)
        if (i) res++;
      break;
    case 1:
      for (auto i : a) res += fabs(i);
      break;
    case 2:
      for (auto i : a) res += i * i;
      res = std::sqrt(res);
      break;
    case -1:
      res = vecMax(a);
      break;
    default:
      printf("[ERROR]vecNorm: invalid l %d, set(0,1,2,-1)\n", l);
  }
  return res;
}

/** @brief convert vector to normalize vector */
template <class T>
void vecNormalize(std::vector<T>& vec, int l = 2) {
  if (vec.empty()) return;
  double norm = vecNorm(vec, l);
  if (fabs(norm) < VS_EPS) return;
  for (auto& v : vec) {
    v /= norm;
  }
}

/** @brief the sum of all values in vector */
template <class T>
T vecSum(const std::vector<T>& vec) {
  T sum = T(0);
  for (const auto& i : vec) sum += i;
  return sum;
}

/** @brief the mean of all values in vector */
template <class T>
T vecMean(const std::vector<T>& vec) {
  if (vec.empty()) return T(0);
  T s = vecSum(vec);
  return s * (1.0f / vec.size());
}

/** @brief the root mean square of all values in vector */
template <class T>
double vecRms(const std::vector<T>& vec) {
  if (vec.empty()) return 0;
  double sum = 0;
  for (const auto& i : vec) sum += i * i;
  return std::sqrt(sum / vec.size());
}

/** @brief the product of all values in vector */
template <class T>
T vecProduct(const std::vector<T>& vec) {
  T res = T(1);
  for (const auto& i : vec) res *= i;
  return res;
}

/** @brief the count of all non-zero values in vector */
template <class T>
int vecCount(const std::vector<T>& vec) {
  return std::count_if(vec.begin(), vec.end(), [](const T& a) { return static_cast<bool>(a); });
}

/** @brief any values is no zero */
template <class T>
bool vecAny(const std::vector<T>& vec) {
  for (const auto& i : vec)
    if (i) return true;
  return false;
}

/** @brief all values is no zero */
template <class T>
bool vecAll(const std::vector<T>& vec) {
  for (const auto& i : vec)
    if (i == 0) return false;
  return true;
}

/** @brief compute the mean and variance of vector. */
template <class T>
void vecStatistics(const std::vector<T>& vec, T& mean, T& stdvar) {
  mean = 0;
  stdvar = 0;
  if (vec.empty()) return;
  mean = vecMean(vec);
  if (vec.size() == 1) return;
  for (auto v : vec) stdvar += sq(v - mean);
  stdvar = std::sqrt(stdvar / (vec.size() - 1));
}

/** @brief the cumsum of all values in vector */
template <class T>
std::vector<T> vecCumsum(const std::vector<T>& vec) {
  if (vec.empty()) return {};
  std::vector<T> res;
  res.reserve(vec.size());
  float sum = 0;
  for (const auto& i : vec) {
    sum += i;
    res.push_back(sum);
  }
  return res;
}

/** @brief Get subset of a vector with begin index and length.
    the similar use to substr() in std::string */
template <class T>
std::vector<T> subvec(const std::vector<T>& vec, int beg, int len = -1) {
  std::vector<T> res;
  if (len == -1)
    res.insert(res.begin(), vec.begin() + beg, vec.end());
  else
    res.insert(res.begin(), vec.begin() + beg, vec.begin() + beg + len);
  return res;
}

/** @brief Get subset of a vector with indice */
template <class T>
std::vector<T> subvec(const std::vector<T>& vec, const std::vector<int>& ids) {
  std::vector<T> res;
  res.reserve(ids.size());
  for (auto i : ids) res.push_back(vec[i]);
  return res;
}

/** @brief generate vector with arange data from start to stop with step */
template <class T>
std::vector<T> vecArange(T start, T stop, T step = 1) {
  std::vector<T> res;
  res.reserve((stop - start) / step);
  for (T a = start; a < stop; a += step) res.push_back(a);
  return res;
}

/** @brief generate vector with arange data from 0 to stop with step 1 */
template <class T>
std::vector<T> vecArange(T stop) {
  return vecArange<T>(0, stop);
}

template <class T>
std::vector<T> range(T start, T stop, T step = 1) {
  return vecArange(start, stop, step);
}

inline std::vector<int> range(int stop) { return vecArange<int>(0, stop, 1); }

/** @brief generate vector with N datas range from start to stop */
template <class T>
std::vector<T> vecLinspace(T start, T stop, int N) {
  if (N <= 0) {
    return {};
  } else if (N == 1) {
    return {start};
  } else if (N == 2) {
    return {start, stop};
  }
  std::vector<T> res;
  res.reserve(N);
  float step = static_cast<float>(stop - start) / (N - 1);
  float v = 0;
  for (int i = 1; i < N; i++) {
    res.push_back(v);
    v += step;
  }
  res.push_back(stop);
  return res;
}

/** @brief dot/inner product between a and b */
template <class T>
T vecDot(const std::vector<T>& a, const std::vector<T>& b) {
  size_t n = a.size();
  if (n <= 0) return T(0);
  if (b.size() != n) {
    printf("[ERROR]vecDot: size not match(%d, %d).\n", static_cast<int>(n), static_cast<int>(b.size()));
    return T(0);
  }
  T res = 0;
  for (size_t i = 0; i < n; i++) res += a[i] * b[i];
  return res;
}

/** @brief a[i] = a[i]*b[i] */
template <class T>
void vecElemMul(std::vector<T>& a, const std::vector<T>& b) {
  size_t n = a.size();
  if (n <= 0) return;
  if (b.size() != n) {
    printf("[ERROR]vecElemMul: size not match(%d, %d).\n", static_cast<int>(n), static_cast<int>(b.size()));
    return;
  }
  for (size_t i = 0; i < n; i++) a[i] *= b[i];
}

/** @brief a[i] = a[i]+b[i] */
template <class T>
void vecElemAdd(std::vector<T>& a, const std::vector<T>& b) {
  size_t n = a.size();
  if (n <= 0) return;
  if (b.size() != n) {
    printf("[ERROR]vecElemAdd: size not match(%d, %d).\n", static_cast<int>(n), static_cast<int>(b.size()));
    return;
  }
  for (size_t i = 0; i < n; i++) a[i] += b[i];
}

/** @brief a[i] = a[i]-b[i] */
template <class T>
void vecElemSub(std::vector<T>& a, const std::vector<T>& b) {
  size_t n = a.size();
  if (n <= 0) return;
  if (b.size() != n) {
    printf("[ERROR]vecElemMul: size not match(%d, %d).\n", static_cast<int>(n), static_cast<int>(b.size()));
    return;
  }
  for (size_t i = 0; i < n; i++) a[i] -= b[i];
}

/** @brief a[i] = a[i]/b[i] */
template <class T>
void vecElemDiv(std::vector<T>& a, const std::vector<T>& b) {
  size_t n = a.size();
  if (n <= 0) return;
  if (b.size() != n) {
    printf("[ERROR]vecElemAdd: size not match(%d, %d).\n", static_cast<int>(n), static_cast<int>(b.size()));
    return;
  }
  for (size_t i = 0; i < n; i++) a[i] /= b[i];
}

/** @brief a[i] = a[i]+b */
template <class T, class T2>
void vecAdd(std::vector<T>& a, T2 b) {
  if (a.empty()) return;
  for (size_t i = 0; i < a.size(); i++) a[i] += b;
}

/** @brief a[i] = a[i]-b */
template <class T, class T2>
void vecSub(std::vector<T>& a, T2 b) {
  if (a.empty()) return;
  for (size_t i = 0; i < a.size(); i++) a[i] -= b;
}

/** @brief a[i] = a[i]*b */
template <class T, class T2>
void vecMul(std::vector<T>& a, T2 b) {
  if (a.empty()) return;
  for (size_t i = 0; i < a.size(); i++) a[i] *= b;
}

/** @brief a[i] = a[i]/b */
template <class T, class T2>
void vecDiv(std::vector<T>& a, T2 b) {
  if (a.empty()) return;
  for (size_t i = 0; i < a.size(); i++) a[i] /= b;
}

/** @brief count element which larger than threshold */
template <class T, class T2>
int vecCmpCnt(const std::vector<T>& a, T2 thres) {
  int cnt = 0;
  for (auto i : a)
    if (i > thres) cnt++;
  return cnt;
}

/** @brief reduce vector with status, remove elements which's status is 0 */
template <class T>
void vecReduce(std::vector<T>& v, std::vector<uint8_t> status) {
  size_t j = 0;
  for (size_t i = 0; i < v.size(); i++)
    if (status[i]) v[j++] = v[i];
  if (j == 0)
    v.clear();
  else
    v.resize(j);
}

/** @brief reduce vector with input function check reduce, remove element if foo_reduce return true
 * @param[in,out]v: vector to be reduce
 * @param[in]foo_reduce: <bool(const T&)> check need reduce
 */
template <class T, class FooType>
void vecReduceIf(std::vector<T>& v, FooType foo_reduce) {
  size_t j = 0;
  for (size_t i = 0; i < v.size(); i++)
    if (!foo_reduce(v[i])) v[j++] = v[i];
  if (j == 0)
    v.clear();
  else
    v.resize(j);
}

/** @brief check if element in vector */
template <class T>
bool vecInside(const std::vector<T>& v, const T& a) {
  auto it = std::find(v.begin(), v.end(), a);
  return it != v.end();
}

/** @brief vector assign, often used to change element type */
template <class T1, class T2>
void vecAssign(const std::vector<T1>& src, std::vector<T2>& tar) {
  tar.resize(src.size());
  for (size_t i = 0; i < src.size(); i++) tar[i] = src[i];
}

/** @brief set whole vector with input value */
template <class T>
void vecSet(std::vector<T>& v, const T& value) {
  for (auto& i : v) i = value;
}

/** @brief vector mapping, use func to mapping each elements from src to tar */
template <class T1, class T2, class Func>
void vecMapping(const std::vector<T1>& src, std::vector<T2>& tar, Func func) {
  tar.resize(src.size());
  for (size_t i = 0; i < src.size(); i++) tar[i] = func(src[i]);
}

/** @brief find indices of valid elements in vector */
template <class T>
std::vector<int> vecValidIds(const std::vector<T>& a) {
  std::vector<int> ids;
  ids.reserve(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i]) ids.push_back(i);
  }
  return ids;
}

/** @brief find indices of valid elements in vector */
template <class T, class Func>
std::vector<int> vecValidIds(const std::vector<T>& a, Func func) {
  std::vector<int> ids;
  ids.reserve(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    if (func(a[i])) ids.push_back(i);
  }
  return ids;
}

/** @brief reverse vector elements */
template <class T>
std::vector<T> vecReverse(const std::vector<T>& a) {
  if (a.empty()) return a;
  std::vector<T> res;
  res.reserve(a.size());
  auto it_begin = a.begin();
  auto it = a.end();
  while (--it != it_begin) res.push_back(*it);
  res.push_back(*it_begin);
  return res;
}

/** @brief print vector to string */
template <class T>
std::string vecStr(const std::vector<T>& a, const char* format = "%d", int buf_len = 2048) {
  char* buf = new char[buf_len];
  int idx = 0;
  idx += snprintf(buf + idx, buf_len - idx, "[");
  if (!a.empty()) {
    idx += snprintf(buf + idx, buf_len - idx, format, a[0]);
    for (size_t i = 1; i < a.size(); i++) {
      idx += snprintf(buf + idx, buf_len - idx, ", ");
      idx += snprintf(buf + idx, buf_len - idx, format, a[i]);
    }
  }
  idx += snprintf(buf + idx, buf_len - idx, "]");
  std::string str(buf);
  delete[] buf;
  return str;
}

/** @brief minimum value of array */
template <typename T>
T arrayMin(const T* a, int N) {
  if (N <= 0) return T(0);
  T m = a[0];
  for (int i = 1; i < N; i++)
    if (a[i] < m) m = a[i];
  return m;
}

/** @brief maximum value of array */
template <typename T>
T arrayMax(const T* a, int N) {
  if (N <= 0) return T(0);
  T m = a[0];
  for (int i = 1; i < N; i++)
    if (a[i] > m) m = a[i];
  return m;
}

/** @brief try insert each element of search_data into sorted_array, and return the insert index
 * @param[in]sorted_array: a sorted array to be inserted.
 * @param[in]search_data: insert elements, must be sorted too.
 * @return insert index of elements, same size as search_data.
 */
template <typename T>
std::vector<int> searchsorted(const std::vector<T>& sorted_array, const std::vector<T>& search_data) {
  std::vector<int> index;
  index.reserve(search_data.size());
  int idx = 0;
  int N = sorted_array.size();
  for (auto v : search_data) {
    while (idx < N && v > sorted_array[idx]) idx++;
    index.push_back(idx);
  }
  return index;
}

/*======================== End of Vector Utils ========================*/

/*======================== String Utils ===============================*/
/** @brief returns True if the string starts with the specified value, otherwise False. */
inline bool startswith(const std::string& line, const std::string& target) {
  return line.substr(0, target.length()) == target;
}

/** @brief returns True if the string ends with the specified value, otherwise False. */
inline bool endswith(const std::string& line, const std::string& target) {
  return line.substr(line.length() - target.length()) == target;
}

/** @brief Case insensitive string comparision */
inline bool iequals(const std::string& str1, const std::string& str2) {
  return ((str1.size() == str2.size()) &&
          std::equal(str1.begin(), str1.end(), str2.begin(), [](const char& c1, const char& c2) {
            return (c1 == c2 || std::toupper(c1) == std::toupper(c2));
          }));
}

/** @brief judge if line contains substr */
inline bool has(const std::string& line, const std::string& substr) { return line.find(substr.c_str()) != line.npos; }

/** @brief cut the front part of line until the first occur of target end */
inline std::string cut(const std::string& line, const std::string& target) {
  size_t i = line.find(target.c_str());
  return i == line.npos ? "" : line.substr(i + target.length());
}

/** @brief cut line to two part at the first occur of target end */
inline void cut(const std::string& line, const std::string& target, std::string& front, std::string& back) {
  size_t i = line.find(target.c_str());
  if (i == line.npos) {
    front = line;
    back = "";
  } else {
    front = line.substr(0, i);
    back = line.substr(i + target.length());
  }
}

/** @brief delete the space in head/tail of str */
inline void strim(std::string& str) {
  if (str.empty()) return;
  str.erase(0, str.find_first_not_of(' '));
  str.erase(str.find_last_not_of("\r ") + 1);
}

/** @brief Convert string to vector of double numbers */
inline std::vector<double> str2vec(const std::string& s, char spliter = ' ') {
  std::string a = s;
  strim(a);
  std::vector<double> data;
  data.reserve(500);
  std::stringstream ss(a.c_str());
  std::string t;
  while (getline(ss, t, spliter)) {
    if (t.length() == 0) continue;
    data.push_back(atof(t.c_str()));
  }
  return data;
}

/** @brief Split string to vector of string */
inline std::vector<std::string> split(const std::string& s, const std::string& delimiters = "") {
  std::vector<std::string> tokens;
  std::string::size_type last_pos = s.find_first_not_of(delimiters, 0);
  std::string::size_type pos = s.find_first_of(delimiters, last_pos);
  while (pos != std::string::npos || last_pos != std::string::npos) {
    tokens.push_back(s.substr(last_pos, pos - last_pos));
    last_pos = s.find_first_not_of(delimiters, pos);
    pos = s.find_first_of(delimiters, last_pos);
  }
  return tokens;
}

/** @brief convert a number or a object which overload operate << to string type */
template <typename T>
std::string num2str(T a) {
  std::stringstream ss;
  ss << a;
  return ss.str();
}

/** @brief convert a string to number of object which overload operate >> */
template <typename T>
T str2num(const std::string& a) {
  T res;
  std::stringstream ss;
  ss << a;
  ss >> res;
  return res;
}

/** @brief get file suffix of input path, which include '.' */
inline std::string suffix(const std::string& s) {
  auto n = s.find_last_of('.');
  return n == s.npos ? std::string() : s.substr(n);
}

/*======================== End of String Utils ========================*/

#define VS_PATH_SPLITER '/'

/** @brief join two path */
inline std::string join(const std::string& f1, const std::string& f2) {
  if (f1.back() == VS_PATH_SPLITER) return f1 + f2;
  return f1 + VS_PATH_SPLITER + f2;
}

/** @brief join three path */
inline std::string join(const std::string& f1, const std::string& f2, const std::string& f3) {
  return join(join(f1, f2), f3);
}

/** @brief split path to dirname and basename */
inline void split(const std::string& file, std::string& dirname, std::string& basename) {
  auto n = file.find_last_of(VS_PATH_SPLITER);
  if (n == file.npos) {
    dirname = file;
    basename = std::string();
  } else {
    dirname = file.substr(0, n);
    basename = file.substr(n + 1);
  }
}

/** @brief split path to name and suffix */
inline void splitext(const std::string& file, std::string& filename, std::string& suffix) {
  auto n = file.find_last_of('.');
  if (n == file.npos) {
    filename = file;
    suffix = std::string();
  } else {
    filename = file.substr(0, n);
    suffix = file.substr(n);
  }
}

/** @brief split path to name and suffix */
inline std::vector<std::string> splitext(const std::string& file) {
  std::string filename, suffix;
  splitext(file, filename, suffix);
  return {filename, suffix};
}

/** @brief base name. eg: /a/b/c/d/ef.gh => ef.gh */
inline std::string basename(const char* path) {
  std::string a, b;
  split(std::string(path), a, b);
  return b;
}

/** @brief dir name. eg: /a/b/c/d/ef.gh => /a/b/c/d/ */
inline std::string dirname(const char* path) {
  std::string a, b;
  split(std::string(path), a, b);
  return a;
}

/** @brief get filename of path */
inline std::string filename(const char* path) {
  std::string a, b;
  splitext(std::string(path), a, b);
  return a;
}

/** @brief get suffix of path */
inline std::string suffix(const char* path) {
  std::string a, b;
  splitext(std::string(path), a, b);
  return b;
}

/** @brief check whether path is a video file */
inline bool isvideo(const char* path) {
  static std::set<std::string> suffix_set = {".mp4", ".avi"};
  return suffix_set.find(suffix(path)) != suffix_set.end();
}

/** @brief check whether path is a image file */
inline bool isimage(const char* path) {
  static std::set<std::string> suffix_set = {".jpg", ".png", ".pgm", ".bmp", ".jpeg"};
  return suffix_set.find(suffix(path)) != suffix_set.end();
}

/** @brief load data table from file
 * comment line starts with '#', each column split with ' ' or ',', each line is a data frame
 * @param[in] file data file
 * @param[in] spliter spliter charactor, if set to 0, it will deduce spliter with the first un-comment line
 * @param[in] min_valid_length min valid length, line length less than this value will be discarded
 * @return std::vector<std::vector<double>> data table
 */
inline std::vector<std::vector<double>> loadFileData(const char* file, char spliter = 0, int min_valid_length = 10) {
  std::ifstream fin(file);
  if (!fin.is_open()) {
    printf("[ERROR]%s:Cannot open file '%s'\n", __func__, file);
    return {};
  }
  std::vector<std::vector<double>> datas;
  std::string line;
  while (getline(fin, line)) {
    strim(line);
    if (line[0] == '#' || static_cast<int>(line.length()) < min_valid_length) continue;
    if (spliter == 0) {
      // init splier by first line
      if (line.find(',') != line.npos) {
        spliter = ',';
      } else if (line.find(' ') != line.npos) {
        spliter = ' ';
      } else {
        printf("[ERROR]%s: Cannot init spliter, line:%s\n", __func__, line.c_str());
        return {};
      }
    }
    datas.push_back(str2vec(line, spliter));
  }
  return datas;
}

} /* namespace vs */