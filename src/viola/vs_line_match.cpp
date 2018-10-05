/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_line_match.h"

#include <Eigen/Dense>

#include "vs_basic.h"

namespace vs {

static double diff(double p1, double p2) {
  double dp = p1 - p2;
  while (dp <= -VS_PI_2) dp += VS_PI;
  while (dp > VS_PI_2) dp -= VS_PI;
  return dp;
}

bool near(const LineSeg2D& l1, const LineSeg2D& l2, double max_angle, double max_dist, double min_overlap) {
  if (l1.length <= 0 || l2.length <= 0) return false;
  if (fabs(diff(l1.theta, l2.theta)) > max_angle) return false;
  auto k = (l1.p1 + l1.p2 - l2.p1 - l2.p2) / 2;
  double dist = fabs(k.cross(((l1.length > l2.length) ? l1 : l2).dir));
  if (dist > max_dist) return false;
  if (min_overlap > 0) {
    double len2 = l1.length * l1.length;
    double k1 = (l1.p2 - l1.p1).dot(l2.p1 - l1.p1) / len2;
    double k2 = (l1.p2 - l1.p1).dot(l2.p2 - l1.p1) / len2;
    double overlap = fabs(clip(k1, 0.0, 1.0) - clip(k2, 0.0, 1.0));
    if (overlap < min_overlap) return false;
  }
  return true;
}

double distance(const LineSeg2D& l1, const LineSeg2D& l2, double max_angle, double max_dist, double min_overlap,
                double w_angle, double w_dist) {
  const static double MAX_DIST = 1e100;
  if (l1.length <= 0 || l2.length <= 0) return MAX_DIST;
  double angle = fabs(diff(l1.theta, l2.theta));
  if (angle > max_angle) {
    // printf("angle:%f\n",angle);
    return MAX_DIST;
  }
  auto k = (l1.p1 + l1.p2 - l2.p1 - l2.p2) / 2;
  double dist = fabs(k.cross(((l1.length > l2.length) ? l1 : l2).dir));
  if (dist > max_dist) {
    // printf("dist:%f\n",dist);
    return MAX_DIST;
  }
  if (min_overlap > 0) {
    double k1 = (l1.p2 - l1.p1).dot(l2.p1 - l1.p1) / l1.length;
    double k2 = (l1.p2 - l1.p1).dot(l2.p2 - l1.p1) / l1.length;
    double overlap = (fabs(clip(k1, 0.0, l1.length) - clip(k2, 0.0, l1.length))) / std::min(l1.length, l2.length);
    if (overlap < min_overlap) {
      // printf("overlap:%f\n",overlap);
      return MAX_DIST;
    }
  }
  return w_angle * angle + w_dist * dist;
}

bool solveTransform(const LineSeg2DList& model, const LineSeg2DList& target, cv::Point3f& T) {
  int N = model.size();
  if (N <= 0 || (int)target.size() != N) return false;

  double sum_angle = 0;
  double sum_weight = 0;
  for (int i = 0; i < N; i++) {
    const auto& l1 = model[i];
    const auto& l2 = target[i];
    double dtheta = diff(l1.theta, l2.theta);
    double w = l1.length * l2.length;
    sum_angle += dtheta * w;
    sum_weight += w;
  }
  if (sum_weight <= 0) return false;
  double angle = sum_angle / sum_weight;
  double ca = cos(angle);
  double sa = sin(angle);
  double A[4] = {0.001, 0, 0, 0.001};
  double b[2] = {0};
  for (int i = 0; i < N; i++) {
    const auto& l1 = model[i];
    const auto& l2 = target[i];
    const auto& dir1 = l1.dir;
    double w = l1.length * l2.length;
    double xx = dir1.y * dir1.y * w;
    double yy = dir1.x * dir1.x * w;
    double xy = -dir1.x * dir1.y * w;
    A[0] += xx;
    A[1] += xy;
    A[2] += xy;
    A[3] += yy;
    cv::Point2f c = (l2.p1 + l2.p2) / 2;
    double k1 = ca * c.x - sa * c.y - l1.p1.x;
    double k2 = sa * c.x + ca * c.y - l1.p1.y;
    b[0] -= xx * k1 + xy * k2;
    b[1] -= xy * k1 + yy * k2;
  }
  cv::Mat t = cv::Mat(2, 2, CV_64FC1, A).inv() * cv::Mat(2, 1, CV_64FC1, b);
  T.x = t.at<double>(0, 0);
  T.y = t.at<double>(1, 0);
  T.z = angle;
  return true;
}

// use covariance estimate
bool solveTransform(const LineSeg2DList& model, const LineSeg2DList& target, cv::Point3f& T, cv::Mat& info) {
  int N = model.size();
  if (N <= 0 || (int)target.size() != N) return false;

  double sum_angle = 0;
  double sum_weight = 0;
  for (int i = 0; i < N; i++) {
    const auto& l1 = model[i];
    const auto& l2 = target[i];
    double dtheta = diff(l1.theta, l2.theta);
    double w = l1.length * l2.length;
    sum_angle += dtheta * w;
    sum_weight += w;
  }
  if (sum_weight <= 0) return false;
  double angle = sum_angle / sum_weight;
  double ca = cos(angle);
  double sa = sin(angle);

  double A[4] = {0, 0, 0, 0};
  double b[2] = {0};
  for (int i = 0; i < N; i++) {
    const auto& l1 = model[i];
    const auto& l2 = target[i];
    const auto& dir1 = l1.dir;
    double w = l1.length * l2.length;
    double xx = dir1.y * dir1.y * w;
    double yy = dir1.x * dir1.x * w;
    double xy = -dir1.x * dir1.y * w;
    A[0] += xx;
    A[1] += xy;
    A[3] += yy;
    cv::Point2f c = (l2.p1 + l2.p2) / 2;
    double k1 = ca * c.x - sa * c.y - l1.p1.x;
    double k2 = sa * c.x + ca * c.y - l1.p1.y;
    b[0] -= xx * k1 + xy * k2;
    b[1] -= xy * k1 + yy * k2;
  }
  A[2] = A[1];
  info = cv::Mat(2, 2, CV_64FC1, A).clone();

  // info*eigvecs.row(i).t() = eigvals.at<double>(i)*eigvecs.row(i).t()
  // NOTE: eigen vector in eigvecs are stored rowwise, so eigvecs = P^T
  cv::Mat eigvals, eigvecs;
  cv::eigen(info, eigvals, eigvecs);
  double temp[4] = {1.0 / std::max(eigvals.at<double>(0, 0), VS_EPS), 0, 0,
                    1.0 / std::max(eigvals.at<double>(1, 0), VS_EPS)};
  cv::Mat cov = eigvecs.t() * cv::Mat(2, 2, CV_64FC1, temp) * eigvecs;
  cv::Mat t = cov * cv::Mat(2, 1, CV_64FC1, b);
  T.x = t.at<double>(0, 0);
  T.y = t.at<double>(1, 0);
  T.z = angle;
  return true;
}

static Eigen::Matrix3d cvtIsom(const cv::Point3f& T) {
  double s = sin(T.z);
  double c = cos(T.z);
  Eigen::Matrix3d A;
  A << c, -s, T.x, s, c, T.y, 0, 0, 1;
  return A;
}

static void transformLines(LineSeg2DList& lines, const Eigen::Matrix3d& T) {
  for (auto& l : lines) {
    Eigen::Vector3d p1 = T * Eigen::Vector3d(l.p1.x, l.p1.y, 1);
    Eigen::Vector3d p2 = T * Eigen::Vector3d(l.p2.x, l.p2.y, 1);
    l.p1 = cv::Point2f(p1(0), p1(1));
    l.p2 = cv::Point2f(p2(0), p2(1));
    l.cal();
  }
}

static void projAffine(cv::Point3f& pose, cv::Point3f& raw_pose, const LineSeg2DList& lines) {
  for (size_t i = 0; i < lines.size() - 1; i++) {
    if (fabs(diff(lines[i].theta, lines[i + 1].theta)) > 0.05) return;
  }
  // TODO: Will it be more accurate to use average dir?
  cv::Point2f dir = lines[0].dir;
  cv::Point2f delta = cv::Point2f(raw_pose.x - pose.x, raw_pose.y - pose.y).dot(dir) * dir;
  pose.x += delta.x;
  pose.y += delta.y;
}

bool ICL(const LineSeg2DList& model, const LineSeg2DList& target, cv::Point3f& transform, bool use_init_guess,
         double thres_angle, double thres_dist, bool proj_affine) {
#if 0
    std::map<int, int> match_ids;
    return ICL(model, target, transform, match_ids, use_init_guess,
               thres_angle, thres_dist, proj_affine);
#else
  ICLSolver solver(model);
  solver.setProjectAffine(proj_affine);
  return solver.match(target, transform, use_init_guess, thres_angle, thres_dist);
#endif
}

bool ICL(const LineSeg2DList& model, const LineSeg2DList& target, cv::Point3f& transform, std::map<int, int>& match_ids,
         cv::Mat& info, bool use_init_guess, double thres_angle, double thres_dist, bool proj_affine) {
  cv::Point3f raw_transform = transform;
  LineSeg2DList lines = target;
  Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
  if (use_init_guess) {
    T = cvtIsom(transform);
    transformLines(lines, T);
  }
  bool solve_once = false;
  int max_ite = 3;
  LineSeg2DList list1, list2;
  while (max_ite-- > 0) {
    list1.clear();
    list2.clear();
    match_ids.clear();
    // data association
    for (size_t i2 = 0; i2 < lines.size(); i2++) {
      const auto& l2 = lines[i2];
      double min_dist = 1e10;
      int min_idx = -1;
      for (size_t i = 0; i < model.size(); i++) {
        const auto& l1 = model[i];
        double dist = distance(l1, l2, thres_angle, thres_dist, 0.1);
        if (dist < min_dist) {
          min_dist = dist;
          min_idx = i;
        }
      }
      if (min_idx >= 0) {
        list1.push_back(model[min_idx]);
        list2.push_back(l2);
        match_ids[i2] = min_idx;
      }
    }
    // solve
    cv::Point3f temp_transform;
    if (!solveTransform(list1, list2, temp_transform, info)) return false;
    solve_once = true;

    Eigen::Matrix3d temp_T = cvtIsom(temp_transform);
    T = temp_T * T;
    transformLines(lines, temp_T);
    if (fabs(temp_transform.z) < VS_EPS && fabs(temp_transform.x) < VS_EPS && fabs(temp_transform.y) < VS_EPS) break;

    thres_angle = std::max(thres_angle / 2, 0.03);
    thres_dist = std::max(thres_dist / 2, 0.05);
  }
  if (solve_once) {
    transform.x = T(0, 2);
    transform.y = T(1, 2);
    transform.z = atan2(T(1, 0), T(0, 0));
    if (proj_affine) {
      projAffine(transform, raw_transform, list1);
    }
    return true;
  }
  return false;
}

bool ICL(const LineSeg2DList& model, const LineSeg2DList& target, cv::Point3f& transform, std::map<int, int>& match_ids,
         bool use_init_guess, double thres_angle, double thres_dist, bool proj_affine) {
  static cv::Mat info;
  return ICL(model, target, transform, match_ids, info, use_init_guess, thres_angle, thres_dist, proj_affine);
}

class LineSegStoreList : public LineSeg2DNN {
 public:
  virtual void build(const LineSeg2DList& linelist) { m_list = linelist; }

  virtual bool nearest(const LineSeg2D& lquery, LineSeg2D& lnearest, double max_angle, double max_dist);

 private:
  LineSeg2DList m_list;
};

class LineNNGrid : public LineSeg2DNN {
 public:
  virtual void build(const LineSeg2DList& linelist);

  virtual bool nearest(const LineSeg2D& lquery, LineSeg2D& lnearest, double max_angle, double max_dist);

 private:
  std::vector<LineSeg2DList> m_grid;
  double m_angle_min, m_angle_max, m_angle_step;
  int m_angle_n;

  int index(double theta) {
    int i = (diff(theta, 0) - m_angle_min) / m_angle_step;
    while (i < 0) i += m_angle_n;
    while (i >= m_angle_n) i -= m_angle_n;
    return i;
  }
};

class LineNNThetaD : public LineSeg2DNN {
  /* 尝试过对theta和d都做离散化，构造二维grid，d是指原点到直线的距离，
  但发现如果直线离原点很远，一点点theta的变化，会导致d差很大，从而暴力搜索能搜到的直线
  利用grid搜索却搜索不到，所以不对d做离散化了，只对theta做离散化，这样虽然提速少了很多，
  但至少保证grid搜出的结果和暴搜的结果一致。
  如果只对theta做离散化，那grid做加速的意义还大吗??? */
 public:
  virtual void build(const LineSeg2DList& linelist);

  virtual bool nearest(const LineSeg2D& lquery, LineSeg2D& lnearest, double max_angle, double max_dist);

 private:
  std::vector<std::vector<LineSeg2DList>> m_grid;
  double m_angle_min, m_angle_max, m_angle_step;
  double m_d_min, m_d_max, m_d_step;
  int m_angle_n, m_d_n;

  bool index(double theta, double d, int& i, int& j) {
    i = (diff(theta, 0) - m_angle_min) / m_angle_step;
    while (i < 0) i += m_angle_n;
    while (i >= m_angle_n) i -= m_angle_n;
    if (i < 0 || i >= m_angle_n) return false;
    j = (fabs(d) - m_d_min) / m_d_step;
    if (j < 0 || j >= m_d_n) return false;
    return true;
  }
};

void ICLSolver::setModel(const LineSeg2DList& model, bool store_grid) {
  if (model.empty()) return;
  if (store_grid)
    m_map.reset(new LineNNGrid);
  else
    m_map.reset(new LineSegStoreList);
  m_map->build(model);
}

bool ICLSolver::match(const LineSeg2DList& target, cv::Point3f& transform, bool use_init_guess, double thres_angle,
                      double thres_dist) {
  cv::Point3f raw_transform = transform;
  LineSeg2DList lines = target;
  Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
  if (use_init_guess) {
    T = cvtIsom(transform);
    transformLines(lines, T);
  }

  bool solve_once = false;
  int max_ite = 3;
  LineSeg2DList list1, list2;
  while (max_ite-- > 0) {
    list1.clear();
    list2.clear();
    // data association
    for (const auto& l2 : lines) {
      LineSeg2D l1 = l2;
      if (m_map->nearest(l2, l1, thres_angle, thres_dist)) {
        list1.push_back(l1);
        list2.push_back(l2);
      }
    }
    // solve
    cv::Point3f temp_transform;
    if (!solveTransform(list1, list2, temp_transform, m_fisher_info)) return false;
    solve_once = true;

    Eigen::Matrix3d temp_T = cvtIsom(temp_transform);
    T = temp_T * T;
    transformLines(lines, temp_T);

    if (fabs(temp_transform.z) < VS_EPS && fabs(temp_transform.x) < VS_EPS && fabs(temp_transform.y) < VS_EPS) break;

    thres_angle = std::max(thres_angle / 2, 0.03);
    thres_dist = std::max(thres_dist / 2, 0.05);
  }

  if (solve_once) {
    transform.x = T(0, 2);
    transform.y = T(1, 2);
    transform.z = atan2(T(1, 0), T(0, 0));
    if (m_proj_affine) {
      projAffine(transform, raw_transform, list1);
    }
    return true;
  }
  return false;
}

bool LineSegStoreList::nearest(const LineSeg2D& lquery, LineSeg2D& lnearest, double max_angle, double max_dist) {
  double nearest_dist = 1e10;
  const LineSeg2D* nearest_ptr = NULL;
  for (const auto& l : m_list) {
    double dist = distance(l, lquery, max_angle, max_dist, 0.1);
    if (dist < nearest_dist) {
      nearest_dist = dist;
      nearest_ptr = &l;
    }
  }
  if (nearest_ptr) {
    lnearest = *nearest_ptr;
    return true;
  }
  return false;
}

void LineNNGrid::build(const LineSeg2DList& model) {
  if (model.empty()) return;
  m_angle_min = -VS_PI_2;
  m_angle_max = VS_PI_2;
  m_angle_n = 36;
  m_angle_step = (m_angle_max - m_angle_min) / m_angle_n;
  m_grid.clear();  // clear old data
  m_grid.resize(m_angle_n);
  for (const auto& l : model) {
    m_grid[index(l.theta)].push_back(l);
  }
}

bool LineNNGrid::nearest(const LineSeg2D& lquery, LineSeg2D& lnearest, double max_angle, double max_dist) {
  double nearest_dist = 1e10;
  const LineSeg2D* nearest_ptr = NULL;
  double theta_min = lquery.theta - max_angle;
  double theta_max = lquery.theta + max_angle;
  int imin = index(theta_min);
  int imax = index(theta_max);
  if (imin > imax) imax += m_angle_n;
  for (int i = imin; i <= imax; i++) {
    int i2 = i;
    while (i2 < 0) i2 += m_angle_n;
    while (i2 >= m_angle_n) i2 -= m_angle_n;
    for (const auto& l : m_grid[i2]) {
      double dist = distance(l, lquery, max_angle, max_dist, 0.1);
      if (dist < nearest_dist) {
        nearest_dist = dist;
        nearest_ptr = &l;
      }
    }
  }
  if (nearest_ptr) {
    lnearest = *nearest_ptr;
    return true;
  }
  return false;
}

void LineNNThetaD::build(const LineSeg2DList& model) {
  if (model.empty()) return;

  m_angle_min = -VS_PI_2;
  m_angle_max = VS_PI_2;
  m_angle_n = 18;
  m_angle_step = (m_angle_max - m_angle_min) / m_angle_n;

  m_d_step = 1.0;
  m_d_min = 0;
  m_d_max = fabs(model[0].d);
  for (const auto& l : model) {
    auto d = fabs(l.d);
    if (d > m_d_max) m_d_max = d;
  }
  m_d_n = (m_d_max - m_d_min) / m_d_step + 1;

  m_grid.resize(m_angle_n);
  for (auto& r : m_grid) {
    r.clear();  // clear old data
    r.resize(m_d_n);
  }
  for (const auto& l : model) {
    int i, j;
    if (index(l.theta, l.d, i, j)) {
      m_grid[i][j].push_back(l);
    }
  }
}

bool LineNNThetaD::nearest(const LineSeg2D& lquery, LineSeg2D& lnearest, double max_angle, double max_dist) {
  double nearest_dist = 1e10;
  const LineSeg2D* nearest_ptr = NULL;
  double theta_min = lquery.theta - max_angle;
  double theta_max = lquery.theta + max_angle;

  double d = fabs(lquery.d);
  double d_min = std::max(d - max_dist, m_d_min);
  double d_max = std::min(d + max_dist, m_d_max);

  int imin, jmin, imax, jmax;
  if (!index(theta_min, d_min, imin, jmin) || !index(theta_max, d_max, imax, jmax)) return false;

  if (imin > imax) imax += m_angle_n;
  for (int i = imin; i <= imax; i++) {
    int i2 = i;
    while (i2 < 0) i2 += m_angle_n;
    while (i2 >= m_angle_n) i2 -= m_angle_n;
    for (int j = jmin; j <= jmax; j++) {
      if (j < 0 || j >= m_d_n) continue;
      for (const auto& l : m_grid[i2][j]) {
        double dist = distance(l, lquery, max_angle, max_dist, 0.1);
        if (dist < nearest_dist) {
          nearest_dist = dist;
          nearest_ptr = &l;
        }
      }
    }
  }
  if (nearest_ptr) {
    lnearest = *nearest_ptr;
    return true;
  }
  return false;
}

} /* namespace vs */