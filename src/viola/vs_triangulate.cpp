#include "vs_triangulate.h"
#include "vs_basic.h"
namespace vs {

bool triangulateTwoView(const Eigen::Isometry3d& cam_pose1, const Eigen::Vector2d& uv1,
                        const Eigen::Isometry3d& cam_pose2, const Eigen::Vector2d& uv2, Eigen::Vector3d& pos) {
  // d*R*v1 + t // v2 => (d*R*v1 + t).cross(v2) = 0
  Eigen::Isometry3d T1to2 = cam_pose2.inverse() * cam_pose1;
  Eigen::Vector3d t = T1to2.translation();
  if (t.norm() < 1e-4 || (uv1 - uv2).norm() < 1e-4) return false;  // no camera move or no parallex

  Eigen::Vector3d Rv1 = T1to2.linear() * Eigen::Vector3d(uv1.x(), uv1.y(), 1.0);
  Eigen::Vector2d A(Rv1.x() - uv2.x() * Rv1.z(), Rv1.y() - uv2.y() * Rv1.z());
  Eigen::Vector2d b(uv2.x() * t.z() - t.x(), uv2.y() * t.z() - t.y());
  double depth = (A.transpose() * A).inverse() * A.transpose() * b;
  pos = cam_pose1 * (depth * Eigen::Vector3d(uv1.x(), uv1.y(), 1));
  return depth > 0;
}

bool triangulateTwoViewNew(const Eigen::Isometry3d& cam_pose1, const Eigen::Vector2d& uv1,
                           const Eigen::Isometry3d& cam_pose2, const Eigen::Vector2d& uv2, Eigen::Vector3d& pos) {
  // calc the midpoint of skew lines common perpendicular, which sum cost is half to triangulateTwoView()
  Eigen::Vector3d p1 = cam_pose1.translation();
  Eigen::Vector3d dir1 = cam_pose1.linear() * Eigen::Vector3d(uv1.x(), uv1.y(), 1).normalized();
  Eigen::Vector3d p2 = cam_pose2.translation();
  Eigen::Vector3d dir2 = cam_pose2.linear() * Eigen::Vector3d(uv2.x(), uv2.y(), 1).normalized();
  auto dp = p2 - p1;
  double dir_dot = dir1.dot(dir2);
  double den = (1 - dir_dot * dir_dot);
  if (fabs(den) < VS_EPS) return false;
  double t1 = (dp.dot(dir1) - dp.dot(dir2) * dir_dot) / den;
  double t2 = (dp.dot(dir1) * dir_dot - dp.dot(dir2)) / den;
  if (t1 < VS_EPS || t2 < VS_EPS) return false;
  auto m1 = p1 + t1 * dir1;
  auto m2 = p2 + t2 * dir2;
  pos = (m1 + m2) / 2;
  return true;
}

static void cost(const Eigen::Isometry3d& T_c0_ci, const Eigen::Vector3d& x, const Eigen::Vector2d& z, double& e) {
  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);
  Eigen::Vector3d h = T_c0_ci.linear() * Eigen::Vector3d(alpha, beta, 1.0) + rho * T_c0_ci.translation();
  // Predict the feature observation in ci frame.
  Eigen::Vector2d z_hat(h(0) / h(2), h(1) / h(2));
  // Compute the residual.
  e = (z_hat - z).squaredNorm();
}

static void jacobian(const Eigen::Isometry3d& T_c0_ci, const Eigen::Vector3d& x, const Eigen::Vector2d& z,
                     Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r, double& w) {
  const double huber_epsilon = 0.01;
  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);

  Eigen::Vector3d h = T_c0_ci.linear() * Eigen::Vector3d(alpha, beta, 1.0) + rho * T_c0_ci.translation();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Compute the Jacobian.
  Eigen::Matrix3d W;
  W.leftCols<2>() = T_c0_ci.linear().leftCols<2>();
  W.rightCols<1>() = T_c0_ci.translation();
  J.row(0) = 1 / h3 * W.row(0) - h1 / (h3 * h3) * W.row(2);
  J.row(1) = 1 / h3 * W.row(1) - h2 / (h3 * h3) * W.row(2);

  // Compute the residual.
  Eigen::Vector2d z_hat(h1 / h3, h2 / h3);
  r = z_hat - z;

  // Compute the weight based on the residual.
  double e = r.norm();
  w = e <= huber_epsilon ? 1.0 : huber_epsilon / (2 * e);
}

static bool solveNls(const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& cam_poses,
                     const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& uvs,
                     Eigen::Vector3d& pos) {
  const double initial_damping = 1e-3;
  const double estimation_precision = 5e-7;
  const int inner_loop_max_iter = 10;
  const int outer_loop_max_iter = 10;
  Eigen::Vector3d solution(pos(0) / pos(2), pos(1) / pos(2), 1.0 / pos(2));
  // Apply Levenberg-Marquart method to solve for the 3d position.
  double lambda = initial_damping;
  int inner_loop_cntr = 0;
  int outer_loop_cntr = 0;
  bool is_cost_reduced = false;
  double delta_norm = 0;

  int N = cam_poses.size();
  // Compute the initial cost
  double total_cost = 0.0;
  for (int i = 0; i < N; ++i) {
    double this_cost = 0.0;
    cost(cam_poses[i], solution, uvs[i], this_cost);
    total_cost += this_cost;
  }

  // Outer loop
  do {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    for (int i = 0; i < N; ++i) {
      Eigen::Matrix<double, 2, 3> J;
      Eigen::Vector2d r;
      double w;
      jacobian(cam_poses[i], solution, uvs[i], J, r, w);
      if (w == 1) {
        A += J.transpose() * J;
        b += J.transpose() * r;
      } else {
        double w_square = w * w;
        A += w_square * J.transpose() * J;
        b += w_square * J.transpose() * r;
      }
    }
    A /= N;
    b /= N;

    // Inner loop: Solve for the delta that can reduce the total cost.
    do {
      Eigen::Matrix3d damper = lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d A_inv = (A + damper).inverse();
      Eigen::Vector3d delta = A_inv * b;
      // Eigen::Vector3d delta = (A+damper).ldlt().solve(b);
      Eigen::Vector3d new_solution = solution - delta;
      delta_norm = delta.norm();
      // check if cost reduce, if reduce, then use new solution
      // else, reduce the optimizer step by increase damper lambda
      double new_cost = 0.0;
      for (int i = 0; i < N; ++i) {
        double this_cost = 0.0;
        cost(cam_poses[i], new_solution, uvs[i], this_cost);
        new_cost += this_cost;
      }
      if (new_cost < total_cost) {
        is_cost_reduced = true;
        solution = new_solution;
        total_cost = new_cost;
        lambda = std::max(lambda / 10, 1e-10);
      } else {
        is_cost_reduced = false;
        lambda = std::min(lambda * 10, 1e12);
      }
    } while (inner_loop_cntr++ < inner_loop_max_iter && !is_cost_reduced);
    inner_loop_cntr = 0;
  } while (outer_loop_cntr++ < outer_loop_max_iter && delta_norm > estimation_precision);

  // Covert the feature position from inverse depth representation to its 3d coordinate.
  pos(0) = solution(0) / solution(2);
  pos(1) = solution(1) / solution(2);
  pos(2) = 1.0 / solution(2);

  // Check if the solution is valid. Make sure the feature is in front of every camera frame observing it.
  for (int i = 0; i < N; i++) {
    if ((cam_poses[i] * pos)(2) <= 0) return false;
  }
  return true;
}

static int findMaxDistance(const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& uvs,
                           const Eigen::Vector2d& ref) {
  int idx = -1;
  double max_d = 0;
  for (size_t i = 0; i < uvs.size(); i++) {
    double d = (uvs[i] - ref).norm();
    if (d > max_d) {
      max_d = d;
      idx = i;
    }
  }
  return idx;
}

bool triangulateMultiViews(const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& cam_poses,
                           const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& uvs,
                           Eigen::Vector3d& pos) {
  // check input
  if (cam_poses.size() < 2 || cam_poses.size() != uvs.size()) return false;
  int ref_idx = 0;
  // triangulate two view for initial guess
  int idx1 = findMaxDistance(uvs, uvs[ref_idx]);
  if (!triangulateTwoViewNew(cam_poses[ref_idx], uvs[ref_idx], cam_poses[idx1], uvs[idx1], pos)) return false;
  // refine with multiview using non-linear least square optimization
  if (cam_poses.size() > 2) {
    // convert camera poses to first camera
    Eigen::Isometry3d ref_pose = cam_poses[ref_idx];
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> Ts_ref_to_ci;
    Ts_ref_to_ci.reserve(cam_poses.size());
    for (const auto& pose : cam_poses) {
      Ts_ref_to_ci.push_back(pose.inverse() * ref_pose);
    }
    Eigen::Vector3d refine_pos = ref_pose.inverse() * pos;
    if (solveNls(Ts_ref_to_ci, uvs, refine_pos)) {
      pos = ref_pose * refine_pos;
    }
  }
  return true;
}

double triangulateError(const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& cam_poses,
                        const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& uvs,
                        const Eigen::Vector3d& pos) {
  int cnt = cam_poses.size();
  if (cnt < 1) return 0;
  double err = 0;
  for (int i = 0; i < cnt; i++) {
    const auto& cam_pose = cam_poses[i];
    const auto& uv = uvs[i];
    const auto& p0 = cam_pose.translation();
    auto p1 = cam_pose * Eigen::Vector3d(uv.x(), uv.y(), 1);
    auto dir = (p1 - p0).normalized();
    err += (pos - p0).cross(dir).norm();
  }
  return err / cnt;
}

double reprojectionError(const Eigen::Isometry3d& pose, const Eigen::Vector2d& uv, const Eigen::Vector3d& pw) {
  auto pc = pose.inverse() * pw;
  return hypot(pc.x() / pc.z() - uv.x(), pc.y() / pc.z() - uv.y());
}

double reprojectionError(const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& cam_poses,
                         const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& uvs,
                         const Eigen::Vector3d& pw) {
  double reproject_error = 0;
  int N = cam_poses.size();
  for (int i = 0; i < N; i++) {
    auto pc = cam_poses[i].inverse() * pw;
    const auto& uv = uvs[i];
    reproject_error += sqsum2(pc.x() / pc.z() - uv.x(), pc.y() / pc.z() - uv.y());
  }
  return std::sqrt(reproject_error / N);
}

bool twoPointRansacImpl(const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& pts1,
                        const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& pts2,
                        std::vector<uint8_t>& status, double norm_pixel_unit, int max_ite, double inlier_thres) {
  int cnt = pts1.size();
  status.clear();
  status.resize(cnt, 255);

  // calculate point diff and distance
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> diff_vec_list;
  std::vector<double> dist_list;  // distance between Rp1 and p2
  diff_vec_list.reserve(cnt);
  dist_list.reserve(cnt);
  for (int i = 0; i < cnt; ++i) {
    auto diff = pts1[i] - pts2[i];
    diff_vec_list.push_back(diff);
    dist_list.push_back(diff.norm());
  }

  // remove large move, compute average inlier distance.
  const double max_inlier_diff = norm_pixel_unit * 50;  // 50 pixels
  double avg_inlier_dist = 0.0;
  int raw_inlier_cnt = 0;
  for (int i = 0; i < cnt; ++i) {
    double dist = dist_list[i];
    if (dist > max_inlier_diff) {
      status[i] = 0;
    } else {
      avg_inlier_dist += dist;
      raw_inlier_cnt++;
    }
  }
  avg_inlier_dist /= raw_inlier_cnt;
  if (raw_inlier_cnt < 3) {
    for (auto& s : status) s = 0;
    return true;
  }

  // pure rotation case: no translation between the frames, RANSAC not work.
  if (avg_inlier_dist < norm_pixel_unit) {
    for (int i = 0; i < cnt; ++i) {
      if (status[i] != 0 && dist_list[i] > inlier_thres) status[i] = 0;
    }
    return true;
  }

  // general motion case: RANSAC works. The three column corresponds to tx, ty, and tz respectively.
  Eigen::MatrixXd coeff_t(cnt, 3);
  for (size_t i = 0; i < diff_vec_list.size(); ++i) {
    coeff_t(i, 0) = diff_vec_list[i].y();
    coeff_t(i, 1) = -diff_vec_list[i].x();
    const auto& p1 = pts1[i];
    const auto& p2 = pts2[i];
    coeff_t(i, 2) = p1.x() * p2.y() - p1.y() * p2.x();
  }

  std::vector<int> raw_inlier_idx = vecValidIds(status);
  raw_inlier_cnt = raw_inlier_idx.size();
  std::vector<int> best_inlier_ids;
  double best_error = 1e10;

  for (int ite = 0; ite < max_ite; ite++) {
    int select_idx1 = rand() % raw_inlier_cnt;
    int select_idx2 = select_idx1 + 1 + rand() % (raw_inlier_cnt - 1);
    if (select_idx2 >= raw_inlier_cnt) select_idx2 -= raw_inlier_cnt;
    int pair_idx1 = raw_inlier_idx[select_idx1];
    int pair_idx2 = raw_inlier_idx[select_idx2];

    // Construct the model;
    Eigen::Vector2d coeff_tx(coeff_t(pair_idx1, 0), coeff_t(pair_idx2, 0));
    Eigen::Vector2d coeff_ty(coeff_t(pair_idx1, 1), coeff_t(pair_idx2, 1));
    Eigen::Vector2d coeff_tz(coeff_t(pair_idx1, 2), coeff_t(pair_idx2, 2));
    std::vector<double> coeff_l1_norm = {coeff_tx.lpNorm<1>(), coeff_ty.lpNorm<1>(), coeff_tz.lpNorm<1>()};
    int base_indicator = std::min_element(coeff_l1_norm.begin(), coeff_l1_norm.end()) - coeff_l1_norm.begin();
    Eigen::Vector3d model(0.0, 0.0, 0.0);
    if (base_indicator == 0) {
      Eigen::Matrix2d A;
      A << coeff_ty, coeff_tz;
      Eigen::Vector2d solution = A.inverse() * (-coeff_tx);
      model(0) = 1.0;
      model(1) = solution(0);
      model(2) = solution(1);
    } else if (base_indicator == 1) {
      Eigen::Matrix2d A;
      A << coeff_tx, coeff_tz;
      Eigen::Vector2d solution = A.inverse() * (-coeff_ty);
      model(0) = solution(0);
      model(1) = 1.0;
      model(2) = solution(1);
    } else {
      Eigen::Matrix2d A;
      A << coeff_tx, coeff_ty;
      Eigen::Vector2d solution = A.inverse() * (-coeff_tz);
      model(0) = solution(0);
      model(1) = solution(1);
      model(2) = 1.0;
    }
    // Find all the inliers among point pairs.
    Eigen::VectorXd error = coeff_t * model;

    std::vector<int> inlier_ids;
    for (int i = 0; i < error.rows(); i++) {
      if (status[i] && fabs(error(i)) < inlier_thres) inlier_ids.push_back(i);
    }

    // If the number of inliers is small, the current model is probably wrong.
    int n_inlier = inlier_ids.size();
    if (n_inlier < 0.2 * cnt || n_inlier <= static_cast<int>(best_inlier_ids.size())) continue;

    // Refit the model using all of the possible inliers.
    Eigen::VectorXd coeff_tx_better(n_inlier);
    Eigen::VectorXd coeff_ty_better(n_inlier);
    Eigen::VectorXd coeff_tz_better(n_inlier);
    for (int i = 0; i < n_inlier; ++i) {
      coeff_tx_better(i) = coeff_t(inlier_ids[i], 0);
      coeff_ty_better(i) = coeff_t(inlier_ids[i], 1);
      coeff_tz_better(i) = coeff_t(inlier_ids[i], 2);
    }
    Eigen::Vector3d model_better(0.0, 0.0, 0.0);
    if (base_indicator == 0) {
      Eigen::MatrixXd A(n_inlier, 2);
      A << coeff_ty_better, coeff_tz_better;
      Eigen::Vector2d solution = -(A.transpose() * A).inverse() * A.transpose() * coeff_tx_better;
      model_better(0) = 1.0;
      model_better(1) = solution(0);
      model_better(2) = solution(1);
    } else if (base_indicator == 1) {
      Eigen::MatrixXd A(n_inlier, 2);
      A << coeff_tx_better, coeff_tz_better;
      Eigen::Vector2d solution = -(A.transpose() * A).inverse() * A.transpose() * coeff_ty_better;
      model_better(0) = solution(0);
      model_better(1) = 1.0;
      model_better(2) = solution(1);
    } else {
      Eigen::MatrixXd A(n_inlier, 2);
      A << coeff_tx_better, coeff_ty_better;
      Eigen::Vector2d solution = -(A.transpose() * A).inverse() * A.transpose() * coeff_tz_better;
      model_better(0) = solution(0);
      model_better(1) = solution(1);
      model_better(2) = 1.0;
    }

    // Compute the error and upate the best model if possible.
    Eigen::VectorXd new_error = coeff_t * model_better;
    double this_error = 0.0;
    for (auto i : inlier_ids) this_error += std::abs(new_error(i));
    this_error /= n_inlier;
    best_error = this_error;
    if (best_error < 1e10) {
      best_inlier_ids = inlier_ids;
    }
  }
  // Fill in the markers.
  vecSet<uint8_t>(status, 0);
  for (auto i : best_inlier_ids) status[i] = 255;
  return true;
}

bool twoPointRansac(const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& uvs1,
                    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& uvs2,
                    const Eigen::Matrix3d& R_1_to_2, std::vector<uint8_t>& status, double focal_length,
                    double inlier_thres, double success_probability) {
  int cnt = uvs1.size();
  if (cnt < 2 || uvs1.size() != uvs2.size()) return false;

  const double norm_pixel_unit = (1.0 / focal_length);
  const int max_ite = static_cast<int>(std::ceil(log(1 - success_probability) / log(1 - sq(0.7))));

  // rotate uvs1 with rotation matrix from 1 to 2
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pts1;
  pts1.reserve(cnt);
  for (const auto& uv : uvs1) {
    Eigen::Vector3d v = R_1_to_2 * Eigen::Vector3d(uv.x(), uv.y(), 1);
    pts1.push_back(Eigen::Vector2d(v.x() / v.z(), v.y() / v.z()));
  }

  // normalize scale for numerical stability
  const bool do_scale_normalization = false;
  if (do_scale_normalization) {
    double sum_norm = 0;
    for (int i = 0; i < cnt; i++) sum_norm += pts1[i].norm() + uvs2[i].norm();
    double scale_factor = cnt / sum_norm * VS_SQRT2;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pts2 = uvs2;
    for (int i = 0; i < cnt; i++) {
      pts1[i] *= scale_factor;
      pts2[i] *= scale_factor;
    };
    return twoPointRansacImpl(pts1, pts2, status, norm_pixel_unit * scale_factor, max_ite, inlier_thres);
  }
  return twoPointRansacImpl(pts1, uvs2, status, norm_pixel_unit, max_ite, inlier_thres);
}

LinearTriangulator::LinearTriangulator() : pos_(0, 0, 0), A_(Eigen::Matrix3d::Zero()), b_(0, 0, 0) {}

void LinearTriangulator::update(double u0, double v0, const Eigen::Isometry3d& cam_pose) {
  Eigen::Vector3d u(u0, v0, 1);
  u = cam_pose.linear() * u / u.norm();
  Eigen::Matrix3d tA = Eigen::Matrix3d::Identity() - u * u.transpose();
  Eigen::Vector3d tb = tA * cam_pose.translation();
  A_ += tA;
  b_ += tb;
  observe_cnt_++;
  if (observe_cnt_ == 1) {
    xmin_ = xmax_ = u0;
    ymin_ = ymax_ = v0;
  } else {
    if (u0 < xmin_) xmin_ = u0;
    if (u0 > xmax_) xmax_ = u0;
    if (v0 < ymin_) ymin_ = v0;
    if (v0 > ymax_) ymax_ = v0;
  }
  parallex_ = hypotf(xmax_ - xmin_, ymax_ - ymin_);
  if (parallex_ > 0.02) {
    Eigen::Vector3cd eigvals = A_.eigenvalues();
    double e[3] = {eigvals(0).real(), eigvals(1).real(), eigvals(2).real()};
    if (e[0] < e[1]) std::swap(e[0], e[1]);
    if (e[0] < e[2]) std::swap(e[0], e[2]);
    if (e[1] < e[2]) std::swap(e[1], e[2]);
    if (e[2] > 0.01 && e[2] / e[0] > 0.0001) {
      pos_ = A_.inverse() * b_;
      pt_cam_ = cam_pose.inverse() * pos_;  // z>0
      good_flag_ = pt_cam_.z() > 1e-3;
    } else {
      good_flag_ = false;
    }
  }
}
}  // namespace vs
