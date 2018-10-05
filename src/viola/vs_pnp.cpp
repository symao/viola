/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_pnp.h"

#include "vs_basic.h"
#include "vs_random.h"

namespace vs {

void p2pTrans(const cv::Point3f& pt1, const cv::Point2f& uv1, const cv::Point3f& pt2, const cv::Point2f& uv2,
              cv::Point3f& trans, bool refine) {
  float a[9] = {0};
  float b[3] = {0};
  a[0] = a[4] = 2;
  a[2] = a[6] = -uv1.x - uv2.x;
  a[5] = a[7] = -uv1.y - uv2.y;
  a[8] = (uv1.x * uv1.x + uv1.y * uv1.y) + (uv2.x * uv2.x + uv2.y * uv2.y);
  a[1] = a[3] = 0;
  b[0] = (uv1.x * pt1.z - pt1.x) + (uv2.x * pt2.z - pt2.x);
  b[1] = (uv1.y * pt1.z - pt1.y) + (uv2.y * pt2.z - pt2.y);
  b[2] = (uv1.x * (pt1.x - uv1.x * pt1.z) + uv1.y * (pt1.y - uv1.y * pt1.z)) +
         (uv2.x * (pt2.x - uv2.x * pt2.z) + uv2.y * (pt2.y - uv2.y * pt2.z));
  cv::Mat t = cv::Mat(3, 3, CV_32FC1, a).inv() * cv::Mat(3, 1, CV_32FC1, b);
  trans = cv::Point3f(t);

  if (refine) {
    // divice (tz+pz) to minimize reprojection error on uv space
    float w1 = 1.0f / sq(pt1.z + trans.z);  // weights for the first point equitions
    float w2 = 1.0f / sq(pt2.z + trans.z);  // weights for the second point equitions
    a[0] = a[4] = w1 + w2;
    a[2] = a[6] = -w1 * uv1.x - w2 * uv2.x;
    a[5] = a[7] = -w1 * uv1.y - w2 * uv2.y;
    a[8] = w1 * (uv1.x * uv1.x + uv1.y * uv1.y) + w2 * (uv2.x * uv2.x + uv2.y * uv2.y);
    a[1] = a[3] = 0;
    b[0] = w1 * (uv1.x * pt1.z - pt1.x) + w2 * (uv2.x * pt2.z - pt2.x);
    b[1] = w1 * (uv1.y * pt1.z - pt1.y) + w2 * (uv2.y * pt2.z - pt2.y);
    b[2] = w1 * (uv1.x * (pt1.x - uv1.x * pt1.z) + uv1.y * (pt1.y - uv1.y * pt1.z)) +
           w2 * (uv2.x * (pt2.x - uv2.x * pt2.z) + uv2.y * (pt2.y - uv2.y * pt2.z));
    cv::Mat t = cv::Mat(3, 3, CV_32FC1, a).inv() * cv::Mat(3, 1, CV_32FC1, b);
    trans = cv::Point3f(t);
  }
}

void pnpTrans(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d, cv::Point3f& trans,
              bool refine) {
  if (pts3d.size() < 2) return;
  float a[9] = {0};
  float b[3] = {0};
  int N = pts3d.size();
  for (int i = 0; i < N; i++) {
    const auto& pti = pts3d[i];
    const auto& uvi = pts2d[i];
    a[2] -= uvi.x;
    a[5] -= uvi.y;
    a[8] += (uvi.x * uvi.x + uvi.y * uvi.y);
    float b0 = uvi.x * pti.z - pti.x;
    float b1 = uvi.y * pti.z - pti.y;
    b[0] += b0;
    b[1] += b1;
    b[2] += -uvi.x * b0 - uvi.y * b1;
  }
  a[0] = a[4] = N;
  a[6] = a[2];
  a[7] = a[5];
  cv::Mat t = cv::Mat(3, 3, CV_32FC1, a).inv() * cv::Mat(3, 1, CV_32FC1, b);
  trans = cv::Point3f(t);

  if (refine) {
    float a[9] = {0};
    float b[3] = {0};
    for (int i = 0; i < N; i++) {
      const auto& pti = pts3d[i];
      const auto& uvi = pts2d[i];
      float w = 1.0f / sq(pti.z + trans.z);
      a[0] += w;
      a[2] -= w * uvi.x;
      a[5] -= w * uvi.y;
      a[8] += w * (uvi.x * uvi.x + uvi.y * uvi.y);
      float b0 = uvi.x * pti.z - pti.x;
      float b1 = uvi.y * pti.z - pti.y;
      b[0] += w * b0;
      b[1] += w * b1;
      b[2] += w * (-uvi.x * b0 - uvi.y * b1);
    }
    a[4] = a[0];
    a[6] = a[2];
    a[7] = a[5];
    a[1] = a[3] = 0;
    cv::Mat t = cv::Mat(3, 3, CV_32FC1, a).inv() * cv::Mat(3, 1, CV_32FC1, b);
    trans = cv::Point3f(t);
  }
}

bool pnpTransRansac(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d, cv::Point3f& trans,
                    std::vector<uchar>& inliers, int max_ite, float reprj_err, bool refine) {
  int n_pts = pts3d.size();
  int half_npt = n_pts / 2;
  // ransac
  std::vector<int> best_inlier_idx;
  for (int ite = 0; ite < max_ite; ite++) {
    int i = randi(0, half_npt);
    int j = randi(half_npt, n_pts);
    const auto& pt1 = pts3d[i];
    const auto& pt2 = pts3d[j];
    const auto& uv1 = pts2d[i];
    const auto& uv2 = pts2d[j];
    if (fabs(pt1.x - pt2.x) + fabs(pt1.y - pt2.y) + fabs(pt1.z - pt2.z) < 0.01 ||
        fabs(uv1.x - uv2.x) + fabs(uv1.y - uv2.y) < 0.01)
      continue;
    cv::Point3f temp_trans;
    p2pTrans(pt1, uv1, pt2, uv2, temp_trans, refine);
    std::vector<int> inlier_idx;
    for (int i = 0; i < n_pts; i++) {
      auto p = pts3d[i] + temp_trans;
      const auto& u = pts2d[i];
      if (p.z < 0.01) continue;
      float error = hypotf(p.x / p.z - u.x, p.y / p.z - u.y);
      if (error < reprj_err) inlier_idx.push_back(i);
    }
    // printf("\nite %d, inliers:%d trans:(%.2f %.2f %.2f)\n", ite, (int)inlier_idx.size(),
    // temp_trans.x, temp_trans.y, temp_trans.z);
    if (inlier_idx.size() > best_inlier_idx.size()) {
      best_inlier_idx = inlier_idx;
      if (static_cast<int>(best_inlier_idx.size()) > half_npt) break;
    }
  }
  if (best_inlier_idx.size() < 2) return false;

  inliers.clear();
  inliers.resize(n_pts, 0);
  for (auto i : best_inlier_idx) inliers[i] = 1;

  if (best_inlier_idx.size() == pts3d.size()) {
    pnpTrans(pts3d, pts2d, trans, refine);
    return true;
  }

  std::vector<cv::Point3f> best_pts3d;
  std::vector<cv::Point2f> best_pts2d;
  best_pts3d.reserve(n_pts);
  best_pts2d.reserve(n_pts);
  int refine_cnt = 2;
  for (int ite = 0; ite < refine_cnt; ite++) {
    best_pts3d.clear();
    best_pts2d.clear();
    for (int i = 0; i < n_pts; i++) {
      if (inliers[i]) {
        best_pts3d.push_back(pts3d[i]);
        best_pts2d.push_back(pts2d[i]);
      }
    }
    if (best_pts3d.size() < 2) return false;
    pnpTrans(best_pts3d, best_pts2d, trans, refine);
    if (ite + 1 < refine_cnt) {
      for (int i = 0; i < n_pts; i++) {
        auto p = pts3d[i] + trans;
        const auto& u = pts2d[i];
        inliers[i] = (p.z > 0.01 && hypotf(p.x / p.z - u.x, p.y / p.z - u.y) < reprj_err) ? 1 : 0;
      }
    }
  }
  return true;
}

void pnpTransYaw(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d, cv::Point3f& trans,
                 float& yaw) {
  int N = pts3d.size();
  if (N < 2) return;

  std::vector<double> yaws;
  yaws.reserve(N * (N - 1));
  for (int i = 0; i < N; i++) {
    const auto& p2i = pts2d[i];
    const auto& p3i = pts3d[i];
    for (int j = i + 1; j < N; j++) {
      auto dp2 = pts2d[j] - p2i;
      auto dp3 = pts3d[j] - p3i;
      yaws.push_back(normalizeRad(atan2(dp2.y, dp2.x) - atan2(dp3.y, dp3.x)));
    }
  }

  double min_yaw, max_yaw;
  vecMinMax(yaws, min_yaw, max_yaw);
  if (max_yaw - min_yaw > VS_PI) {
    for (auto& i : yaws) {
      if (i < 0) i += VS_2PI;
    }
  }
  yaw = vecMean(yaws);

  float c = cos(yaw);
  float s = sin(yaw);
  std::vector<cv::Point3f> pts3d_r;
  pts3d_r.reserve(N);
  for (const auto& p : pts3d) {
    pts3d_r.push_back(cv::Point3f(p.x * c - p.y * s, p.x * s + p.y * c, p.z));
  }
  pnpTrans(pts3d_r, pts2d, trans);
}

void pnlTrans(const std::vector<cv::Vec6f>& lns3d, const std::vector<cv::Vec4f>& lns2d, cv::Point3f& trans) {
  int N = lns3d.size();
  if (N < 3) return;
  float a[9] = {0};
  float b[3] = {0};
  for (int i = 0; i < N; i++) {
    const auto ln3d = lns3d[i];
    const auto ln2d = lns2d[i];

    float u1 = ln2d[0];
    float v1 = ln2d[1];
    float u2 = ln2d[2];
    float v2 = ln2d[3];
    float n[3] = {v1 - v2, -u1 + u2, u1 * v2 - u2 * v1};
    if (n[2] == 0) continue;
    n[0] /= n[2];
    n[1] /= n[2];
    n[2] = 1;

    float ai[9] = {n[0] * n[0], n[0] * n[1], n[0] * n[2], n[1] * n[0], n[1] * n[1],
                   n[1] * n[2], n[2] * n[0], n[2] * n[1], n[2] * n[2]};
    float x = (ln3d[0] + ln3d[3]) / 2.0;
    float y = (ln3d[1] + ln3d[4]) / 2.0;
    float z = (ln3d[2] + ln3d[5]) / 2.0;
    for (int j = 0; j < 9; j++) {
      a[j] += ai[j];
    }

    b[0] -= ai[0] * x + ai[1] * y + ai[2] * z;
    b[1] -= ai[3] * x + ai[4] * y + ai[5] * z;
    b[2] -= ai[6] * x + ai[7] * y + ai[8] * z;
  }
  cv::Mat t = cv::Mat(3, 3, CV_32FC1, a).inv() * cv::Mat(3, 1, CV_32FC1, b);
  trans = cv::Point3f(t);
}

void pnplTrans(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d,
               const std::vector<cv::Vec6f>& lns3d, const std::vector<cv::Vec4f>& lns2d, cv::Point3f& trans,
               const std::vector<float>& line_weights) {
  int Np = pts3d.size();
  int Nl = lns3d.size();
  if (Np < 2 || Nl < 3) {
    return;
  }
  float a[9] = {0};
  float b[3] = {0};
  // pnp
  for (int i = 0; i < Np; i++) {
    const auto& pti = pts3d[i];
    const auto& uvi = pts2d[i];
    a[2] -= uvi.x;
    a[5] -= uvi.y;
    a[8] += (uvi.x * uvi.x + uvi.y * uvi.y);
    float b0 = uvi.x * pti.z - pti.x;
    float b1 = uvi.y * pti.z - pti.y;
    b[0] += b0;
    b[1] += b1;
    b[2] += -uvi.x * b0 - uvi.y * b1;
  }
  a[0] = a[4] = Np;
  a[6] = a[2];
  a[7] = a[5];
  // pnl
  for (int i = 0; i < Nl; i++) {
    const auto ln3d = lns3d[i];
    const auto ln2d = lns2d[i];
    float w = (i < static_cast<int>(line_weights.size())) ? line_weights[i] : 1;

    float u1 = ln2d[0];
    float v1 = ln2d[1];
    float u2 = ln2d[2];
    float v2 = ln2d[3];
    float n[3] = {v1 - v2, -u1 + u2, u1 * v2 - u2 * v1};
    if (n[2] == 0) continue;
    n[0] *= w / n[2];
    n[1] *= w / n[2];
    n[2] = w;

    float ai[9] = {n[0] * n[0], n[0] * n[1], n[0] * n[2], n[1] * n[0], n[1] * n[1],
                   n[1] * n[2], n[2] * n[0], n[2] * n[1], n[2] * n[2]};
    float x = (ln3d[0] + ln3d[3]) / 2.0;
    float y = (ln3d[1] + ln3d[4]) / 2.0;
    float z = (ln3d[2] + ln3d[5]) / 2.0;
    for (int j = 0; j < 9; j++) {
      a[j] += ai[j];
    }

    b[0] -= ai[0] * x + ai[1] * y + ai[2] * z;
    b[1] -= ai[3] * x + ai[4] * y + ai[5] * z;
    b[2] -= ai[6] * x + ai[7] * y + ai[8] * z;
  }
  cv::Mat t = cv::Mat(3, 3, CV_32FC1, a).inv() * cv::Mat(3, 1, CV_32FC1, b);
  trans = cv::Point3f(t);
}

bool pnpRansacTransYaw(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d, cv::Point3f& trans,
                       float& yaw, std::vector<uchar>& inliers, int ite_cnt, float reprj_err) {
  // ransac
  int n_pts = pts3d.size();
  int half_npt = n_pts / 2;
  int max_inlier_cnt = 0;
  cv::Point3f max_t;
  std::vector<int> max_inlier_idx;
  float max_yaw;
  for (int ite = 0; ite < ite_cnt && max_inlier_cnt < half_npt;) {
    int i = randi(0, half_npt);
    int j = randi(half_npt, n_pts);
    if (j == i || cv::norm(pts2d[i] - pts2d[j]) < 0.01) {
      ite++;
      continue;
    }

    const auto& p1 = pts3d[i];
    const auto& p2 = pts3d[j];
    const auto& u1 = pts2d[i];
    const auto& u2 = pts2d[j];

    float tyaw = normalizeRad(atan2(u2.y - u1.y, u2.x - u1.x) - atan2(p2.y - p1.y, p2.x - p1.x));
    float c = cos(tyaw);
    float s = sin(tyaw);
    cv::Point3f pt1(p1.x * c - p1.y * s, p1.x * s + p1.y * c, p1.z);
    cv::Point3f pt2(p2.x * c - p2.y * s, p2.x * s + p2.y * c, p2.z);
    cv::Point3f trans;
    p2pTrans(pt1, u1, pt2, u2, trans);

    // reproj
    int inlier_cnt = 0;
    std::vector<int> temp_inlier_idx;
    for (int k = 0; k < n_pts; k++) {
      const auto& pt = pts3d[k];
      const auto& uv = pts2d[k];
      float x = pt.x * c - pt.y * s + trans.x;
      float y = pt.x * s + pt.y * c + trans.y;
      float z = pt.z + trans.z;
      float dx = (x / z) - uv.x;
      float dy = (y / z) - uv.y;
      if (hypotf(dx, dy) < reprj_err) {
        inlier_cnt++;
        temp_inlier_idx.push_back(k);
      }
    }
    if (inlier_cnt > max_inlier_cnt) {
      max_inlier_cnt = inlier_cnt;
      max_inlier_idx = temp_inlier_idx;
      max_yaw = tyaw;
    }
    ite++;
    // printf("yaw:%f trans:%f %f %f inlier:%d\n", tyaw, trans.x, trans.y, trans.z, inlier_cnt);
  }
  if (max_inlier_cnt <= 2) {
    return false;
  }

  // refine yaw
  yaw = max_yaw;
  {
    float sum_diff = 0;
    int sum_cnt = 0;
    for (int i = 0; i < max_inlier_cnt; i++) {
      for (int j = 0; j < max_inlier_cnt; j++) {
        int ii = max_inlier_idx[i];
        int jj = max_inlier_idx[j];
        if (cv::norm(pts2d[ii] - pts2d[jj]) < 0.01) continue;
        sum_diff += normalizeRad(atan2(pts2d[jj].y - pts2d[ii].y, pts2d[jj].x - pts2d[ii].x) -
                                 atan2(pts3d[jj].y - pts3d[ii].y, pts3d[jj].x - pts3d[ii].x) - yaw);
        sum_cnt++;
      }
    }
    if (sum_cnt > 0) {
      yaw += sum_diff / sum_cnt;
    }
  }
  float s = sin(yaw);
  float c = cos(yaw);
  std::vector<cv::Point3f> pts3d_rot;
  pts3d_rot.reserve(n_pts);
  for (const auto& pt : pts3d) {
    float x = pt.x * c - pt.y * s;
    float y = pt.x * s + pt.y * c;
    pts3d_rot.push_back(cv::Point3f(x, y, pt.z));
  }
  // printf("max_cnt:%d max_yaw:%.3f refine_yaw:%.3f\n", max_inlier_cnt, max_yaw, yaw);

  // refine pos
  inliers.resize(n_pts);
  std::fill(inliers.begin(), inliers.end(), 0);
  for (auto i : max_inlier_idx) inliers[i] = 1;
  for (int ite = 0; ite < 2; ite++) {
    std::vector<cv::Point3f> pts3d_inlier;
    std::vector<cv::Point2f> pts2d_inlier;
    pts3d_inlier.reserve(n_pts);
    pts2d_inlier.reserve(n_pts);
    for (int i = 0; i < n_pts; i++) {
      if (inliers[i]) {
        pts3d_inlier.push_back(pts3d_rot[i]);
        pts2d_inlier.push_back(pts2d[i]);
      }
    }
    pnpTrans(pts3d_inlier, pts2d_inlier, trans);
    // recalculate inlier
    for (int i = 0; i < n_pts; i++) {
      const auto& pt = pts3d_rot[i];
      const auto& uv = pts2d[i];
      float x = pt.x + trans.x;
      float y = pt.y + trans.y;
      float z = pt.z + trans.z;
      float dx = (x / z) - uv.x;
      float dy = (y / z) - uv.y;
      inliers[i] = hypotf(dx, dy) < reprj_err ? 1 : 0;
    }
  }
  return true;
}

static bool cvPnPRansac(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d, const cv::Mat& K,
                        const cv::Mat& D, cv::Mat& rvec, cv::Mat& tvec, std::vector<uchar>& status, bool use_init_guess,
                        int max_ite, double reproject_err) {
  std::vector<int> inliers;
#if (CV_MAJOR_VERSION >= 3)
  cv::solvePnPRansac(pts3d, pts2d, K, D, rvec, tvec, use_init_guess, max_ite, reproject_err, 0.95, inliers,
                     cv::SOLVEPNP_EPNP);
#else
  cv::solvePnPRansac(pts3d, pts2d, K, D, rvec, tvec, use_init_guess, max_ite, reproject_err,
                     static_cast<int>(pts2d.size() * 0.7f), inliers, cv::EPNP);
#endif
  if (inliers.size() < 3) return false;
  status = std::vector<uchar>(pts2d.size(), 0);
  for (auto i : inliers) status[i] = 255;
  return true;
}

static bool cvPnPRefine(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d, const cv::Mat& K,
                        const cv::Mat& D, cv::Mat& rvec, cv::Mat& tvec, bool use_init_guess) {
  if (pts3d.size() < 3 || pts3d.size() != pts2d.size()) return false;
#if (CV_MAJOR_VERSION >= 4)
  // use EPnP to compute init transformation
  if (!use_init_guess) cv::solvePnP(pts3d, pts2d, K, D, rvec, tvec, false, cv::SOLVEPNP_EPNP);
  cv::solvePnPRefineLM(pts3d, pts2d, K, D, rvec, tvec);
  return true;
#elif (CV_MAJOR_VERSION == 3)
  return cv::solvePnP(pts3d, pts2d, K, D, rvec, tvec, use_init_guess, cv::SOLVEPNP_ITERATIVE);
#else
  return cv::solvePnP(pts3d, pts2d, K, D, rvec, tvec, use_init_guess, cv::ITERATIVE);
#endif
}

bool pnpSolverOpenCV(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d, const cv::Mat& K,
                     const cv::Mat& D, cv::Mat& rvec, cv::Mat& tvec, std::vector<uchar>& status, bool use_init_guess,
                     bool do_ransac, int ransac_max_ite, double reproject_err, int refine_cnt, bool centroid) {
  PointCentralizer centralizer;
  std::vector<cv::Point3f> pts3d_centeroid;
  if (centroid) {
    pts3d_centeroid = centralizer.process(pts3d);
    centralizer.forwardRt(rvec, tvec);
  }
  bool ok;
  const auto& process_pts3d = centroid ? pts3d_centeroid : pts3d;
  if (do_ransac) {
    ok = cvPnPRansac(process_pts3d, pts2d, K, D, rvec, tvec, status, use_init_guess, ransac_max_ite, reproject_err);
  } else {
    ok = cvPnPRefine(process_pts3d, pts2d, K, D, rvec, tvec, use_init_guess);
    status.clear();
    status.resize(pts3d.size(), 255);
  }
  if (!ok) return false;

  if (refine_cnt > 0) {
    for (int ite = 0; ite < refine_cnt; ite++) {
      std::vector<cv::Point2f> project_pts;
      cv::projectPoints(process_pts3d, rvec, tvec, K, D, project_pts);
      for (size_t i = 0; i < status.size(); i++) {
        status[i] = cv::norm(pts2d[i] - project_pts[i]) < reproject_err ? 255 : 0;
      }
      auto ids = vecValidIds(status);
      ok = cvPnPRefine(subvec(process_pts3d, ids), subvec(pts2d, ids), K, D, rvec, tvec, true);
      if (!ok) return false;
    }
  }

  if (centroid) centralizer.backwardRt(rvec, tvec);
  return vecCount(status) > 5;
}

std::vector<cv::Point3f> PointCentralizer::process(const std::vector<cv::Point3f>& pts3d) {
  if (pts3d.empty()) return {};
  cv::Point3f pts_mean = vecMean(pts3d);
  mean_vec_ = cv::Mat(pts_mean);
  mean_vec_valid_ = true;
  std::vector<cv::Point3f> pts3d_centeroid;
  pts3d_centeroid.reserve(pts3d.size());
  for (auto& p : pts3d) pts3d_centeroid.push_back(p - pts_mean);
  return pts3d_centeroid;
}

void PointCentralizer::forwardRt(const cv::Mat& rvec, cv::Mat& tvec) {
  if (mean_vec_valid_) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    mean_vec_.convertTo(mean_vec_, R.type());
    tvec += R * mean_vec_;
  }
}

void PointCentralizer::backwardRt(const cv::Mat& rvec, cv::Mat& tvec) {
  if (mean_vec_valid_) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    mean_vec_.convertTo(mean_vec_, R.type());
    tvec -= R * mean_vec_;
  }
}

} /* namespace vs */