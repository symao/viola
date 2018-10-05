/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details g2o wrapper for PnP, Graph-SLAM.
 */
#pragma once
#if HAVE_G2O
#include <Eigen/Dense>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/linear_solver_dense.h>
#include <g2o/types/types_six_dof_expmap.h>

#include <opencv2/core.hpp>
#include "vs_cv_convert.h"

namespace g2o {

typedef Eigen::Matrix<double, 6, 1, Eigen::ColMajor> Vector6D;
typedef Eigen::Matrix<double, 6, 6> Matrix6D;

class VertexSBALine : public BaseVertex<6, Vector6D> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexSBALine() : BaseVertex<6, Vector6D>() {}

  virtual bool read(std::istream& is) {
    Vector6D lv;
    for (int i = 0; i < 6; i++) is >> _estimate[i];
    return true;
  }

  virtual bool write(std::ostream& os) const {
    Vector6D lv = estimate();
    for (int i = 0; i < 6; i++) {
      os << lv[i] << " ";
    }
    return os.good();
  }

  virtual void setToOriginImpl() { _estimate.fill(0.); }

  virtual void oplusImpl(const double* update) {
    Eigen::Map<const Vector6D> v(update);
    _estimate += v;
  }
};

class EdgeSE3ProjectLineOnlyPose : public BaseUnaryEdge<2, Vector4d, VertexSE3Expmap> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectLineOnlyPose() {}

  bool read(std::istream& is) { return false; }

  bool write(std::ostream& os) const { return false; }

  void computeError() {
    const VertexSE3Expmap* v = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    auto pc1 = cam_project(v->estimate().map(pw1));
    auto pc2 = cam_project(v->estimate().map(pw2));
    _error(0) = ln.dot(pc1 - lp);
    _error(1) = ln.dot(pc2 - lp);
  }

  virtual void setMeasurement(const Eigen::Vector4d& m) {
    _measurement = m;
    lp = m.head(2);
    ln = Eigen::Vector2d(m(1) - m(3), m(2) - m(0));
    ln.normalize();
  }

  Eigen::MatrixXd jacobianPt(const SE3Quat& se3, const Eigen::Vector3d& pw) {
    Vector3d pc = se3.map(pw);
    double x = pc[0];
    double y = pc[1];
    double invz = 1.0 / pc[2];
    double invz_2 = invz * invz;
    Eigen::MatrixXd J(2, 6);
    J(0, 0) = x * y * invz_2 * fx;
    J(0, 1) = -(1 + (x * x * invz_2)) * fx;
    J(0, 2) = y * invz * fx;
    J(0, 3) = -invz * fx;
    J(0, 4) = 0;
    J(0, 5) = x * invz_2 * fx;
    J(1, 0) = (1 + y * y * invz_2) * fy;
    J(1, 1) = -x * y * invz_2 * fy;
    J(1, 2) = -x * invz * fy;
    J(1, 3) = 0;
    J(1, 4) = -invz * fy;
    J(1, 5) = y * invz_2 * fy;
    return J;
  }

  virtual void linearizeOplus() {
    auto se3 = static_cast<VertexSE3Expmap*>(_vertices[0])->estimate();
    auto J1 = jacobianPt(se3, pw1);
    auto J2 = jacobianPt(se3, pw2);
    _jacobianOplusXi.row(0) = -ln.transpose() * J1;
    _jacobianOplusXi.row(1) = -ln.transpose() * J2;
  }

  bool isDepthPositive() {
    const auto& T = static_cast<const VertexSE3Expmap*>(_vertices[0])->estimate();
    return (T.map(pw1))(2) > 0.0 && (T.map(pw2))(2) > 0.0;
  }

  Vector2d cam_project(const Vector3d& trans_xyz) const {
    return Vector2d(trans_xyz.x() / trans_xyz.z() * fx + cx, trans_xyz.y() / trans_xyz.z() * fy + cy);
  }

  Vector3d pw1, pw2;
  double fx, fy, cx, cy;

 private:
  Eigen::Vector2d lp, ln;  // line point, line normal
};

class EdgeSe3PriorRot : public BaseUnaryEdge<3, Eigen::Matrix3d, VertexSE3Expmap> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeSe3PriorRot() {}

  bool read(std::istream& is) { return false; }

  bool write(std::ostream& os) const { return false; }

  void computeError() {
    const VertexSE3Expmap* v = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    Eigen::Quaterniond q = v->estimate().rotation() * _inverseQ;
    q.normalize();
    _error = q.coeffs().head<3>();
  }

  // virtual void linearizeOplus() {}

  virtual void setMeasurement(const Eigen::Matrix3d& m) {
    _measurement = m;
    _inverseQ = Eigen::Quaterniond(m.transpose());
  }

 private:
  Eigen::Quaterniond _inverseQ;
};

class EdgeSe3PriorPose : public BaseUnaryEdge<6, Eigen::Isometry3d, VertexSE3Expmap> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeSe3PriorPose() {}

  bool read(std::istream& is) { return false; }

  bool write(std::ostream& os) const { return false; }

  void computeError() {
    Eigen::Isometry3d T = static_cast<const VertexSE3Expmap*>(_vertices[0])->estimate();
    Eigen::Isometry3d delta = T * _inverseT;
    Eigen::Quaterniond q(delta.linear());
    q.normalize();
    _error.head(3) = delta.translation();
    _error.tail(3) = q.coeffs().head<3>();
  }

  // virtual void linearizeOplus() {}

  virtual void setMeasurement(const Eigen::Isometry3d& m) {
    _measurement = m;
    _inverseT = m.inverse();
  }

 private:
  Eigen::Isometry3d _inverseT;
};

}  // namespace g2o

namespace vs {

class G2OGraphPnPL {
 public:
  G2OGraphPnPL() {
    g2o::BlockSolver_6_3::LinearSolverType* linear_solver;
    linear_solver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linear_solver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    m_optimizer.setAlgorithm(solver);
    reset();
  }

  void setCamera(const cv::Mat& K) {
    cv::Mat a;
    K.convertTo(a, CV_64F);
    double* p = reinterpret_cast<double*>(a.data);
    m_fx = p[0];
    m_fy = p[4];
    m_cx = p[2];
    m_cy = p[5];
  }

  void reset() {
    m_optimizer.clear();
    g2o::VertexSE3Expmap* v_pose = new g2o::VertexSE3Expmap();
    Eigen::Isometry3d Twc = Eigen::Isometry3d::Identity();
    v_pose->setEstimate(g2o::SE3Quat(Twc.linear(), Twc.translation()));
    v_pose->setId(m_vertex_id);
    v_pose->setFixed(false);
    m_optimizer.addVertex(v_pose);
  }

  void addPriorRot(const Eigen::Matrix3d& rot, const g2o::Matrix3d& info) {
    auto* v = getPoseVertex();
    g2o::EdgeSe3PriorRot* e = new g2o::EdgeSe3PriorRot();
    e->setVertex(0, v);
    e->setMeasurement(rot);
    e->setInformation(info);
    m_optimizer.addEdge(e);
  }

  void addPriorPose(const Eigen::Isometry3d& T, const g2o::Matrix6d& info) {
    auto* v = getPoseVertex();
    g2o::EdgeSe3PriorPose* e = new g2o::EdgeSe3PriorPose();
    e->setVertex(0, v);
    e->setMeasurement(T);
    e->setInformation(info);
    m_optimizer.addEdge(e);
  }

  void addPointPair(const cv::Point3f& p3d, const cv::Point2f& p2d, float weight = 1.0f, float huber_param = 0.0f) {
    auto* v = getPoseVertex();
    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
    e->setVertex(0, v);
    e->setMeasurement(Eigen::Vector2d(p2d.x, p2d.y));
    e->setInformation(Eigen::Matrix2d::Identity() * weight * weight);
    setHuber(e, huber_param);
    e->fx = m_fx;
    e->fy = m_fy;
    e->cx = m_cx;
    e->cy = m_cy;
    e->Xw << p3d.x, p3d.y, p3d.z;
    m_optimizer.addEdge(e);
  }

  void addPointPairs(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d,
                     const std::vector<float>& weights = {}, float huber_param = 0.0f) {
    auto* v = getPoseVertex();
    int N = std::min(pts3d.size(), pts2d.size());
    int wN = weights.size();
    for (int i = 0; i < N; i++) {
      const auto& p3d = pts3d[i];
      const auto& p2d = pts2d[i];
      g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
      e->setVertex(0, v);
      e->setMeasurement(Eigen::Vector2d(p2d.x, p2d.y));
      if (i < wN) {
        float w = weights[i];
        e->setInformation(Eigen::Matrix2d::Identity() * (w * w));
      } else {
        e->setInformation(Eigen::Matrix2d::Identity());
      }
      setHuber(e, huber_param);
      e->fx = m_fx;
      e->fy = m_fy;
      e->cx = m_cx;
      e->cy = m_cy;
      e->Xw << p3d.x, p3d.y, p3d.z;
      m_optimizer.addEdge(e);
    }
  }

  void addLinePair(const cv::Vec6f& l3d, const cv::Vec4f& l2d, float weight = 1.0f, float huber_param = 0.0f) {
    auto* v = getPoseVertex();
    g2o::EdgeSE3ProjectLineOnlyPose* e = new g2o::EdgeSE3ProjectLineOnlyPose();
    e->setVertex(0, v);
    e->setMeasurement(Eigen::Vector4d(l2d[0], l2d[1], l2d[2], l2d[3]));
    e->setInformation(Eigen::Matrix2d::Identity() * (weight * weight));
    setHuber(e, huber_param);
    e->fx = m_fx;
    e->fy = m_fy;
    e->cx = m_cx;
    e->cy = m_cy;
    e->pw1 = Eigen::Vector3d(l3d[0], l3d[1], l3d[2]);
    e->pw2 = Eigen::Vector3d(l3d[3], l3d[4], l3d[5]);
    m_optimizer.addEdge(e);
  }

  void addLinePairs(const std::vector<cv::Vec6f>& lines3d, const std::vector<cv::Vec4f>& lines2d,
                    const std::vector<float>& weights = {}, float huber_param = 0.0f) {
    auto* v = getPoseVertex();
    int N = std::min(lines3d.size(), lines2d.size());
    int wN = weights.size();
    for (int i = 0; i < N; i++) {
      const auto& l3d = lines3d[i];
      const auto& l2d = lines2d[i];
      g2o::EdgeSE3ProjectLineOnlyPose* e = new g2o::EdgeSE3ProjectLineOnlyPose();
      e->setVertex(0, v);
      e->setMeasurement(Eigen::Vector4d(l2d[0], l2d[1], l2d[2], l2d[3]));
      if (i < wN) {
        float w = weights[i];
        e->setInformation(Eigen::Matrix2d::Identity() * (w * w));
      } else {
        e->setInformation(Eigen::Matrix2d::Identity());
      }
      setHuber(e, huber_param);
      e->fx = m_fx;
      e->fy = m_fy;
      e->cx = m_cx;
      e->cy = m_cy;
      e->pw1 = Eigen::Vector3d(l3d[0], l3d[1], l3d[2]);
      e->pw2 = Eigen::Vector3d(l3d[3], l3d[4], l3d[5]);
      m_optimizer.addEdge(e);
    }
  }

  bool solve(Eigen::Isometry3d& pose, int ite = 10, bool use_init_guess = false, bool verbose = false) {
    m_optimizer.initializeOptimization();
    if (m_optimizer.indexMapping().empty()) return false;
    auto* v_pose = static_cast<g2o::VertexSE3Expmap*>(getPoseVertex());
    if (use_init_guess) v_pose->setEstimate(g2o::SE3Quat(pose.linear(), pose.translation()));
    m_optimizer.setVerbose(verbose);
    m_optimizer.optimize(ite);
    pose = v_pose->estimate();
    return true;
  }

  void computeGraphError(const Eigen::Isometry3d& pose) {
    auto* v_pose = static_cast<g2o::VertexSE3Expmap*>(getPoseVertex());
    v_pose->setEstimate(g2o::SE3Quat(pose.linear(), pose.translation()));
    auto& edges = m_optimizer.edges();
    // printf("Edges:%d chi2:(", (int)edges.size());
    for (auto& e : edges) {
      g2o::OptimizableGraph::Edge* e2 = static_cast<g2o::OptimizableGraph::Edge*>(e);
      e2->computeError();
      // printf("%.5f ", e2->chi2());
    }
    // return m_optimizer.chi2();
    // printf(")\n");
  }

  g2o::SparseOptimizer* optimizer() { return &m_optimizer; }

 private:
  g2o::SparseOptimizer m_optimizer;
  double m_fx = 1.0;
  double m_fy = 1.0;
  double m_cx = 0.0;
  double m_cy = 0.0;
  int m_vertex_id = 0;

  void setHuber(g2o::OptimizableGraph::Edge* e, double huber_sigma) {
    if (huber_sigma > 0) {
      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
      rk->setDelta(huber_sigma);
      e->setRobustKernel(rk);
    }
  }

  g2o::OptimizableGraph::Vertex* getPoseVertex() {
    return dynamic_cast<g2o::OptimizableGraph::Vertex*>(m_optimizer.vertex(m_vertex_id));
  }
};

inline bool pnpSolverG2O(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d, const cv::Mat& K,
                         const cv::Mat& D, cv::Mat& rvec, cv::Mat& tvec, std::vector<uchar>& status,
                         bool use_init_guess, double reproject_err = 3, double prior_rot_weight = 0,
                         double prior_trans_weight = 0) {
  g2o::SparseOptimizer optimizer;
  g2o::BlockSolver_6_3::LinearSolverType* linear_solver;
  linear_solver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
  g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linear_solver);
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  optimizer.setAlgorithm(solver);

  // Set Frame vertex
  g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
  Eigen::Isometry3d Twc = use_init_guess ? rt2isom(rvec, tvec) : Eigen::Isometry3d::Identity();
  vSE3->setEstimate(g2o::SE3Quat(Twc.linear(), Twc.translation()));
  vSE3->setId(0);
  vSE3->setFixed(false);
  optimizer.addVertex(vSE3);

  if (use_init_guess) {
    if (prior_trans_weight > 0) {
      Eigen::Isometry3d Twc_prior = Twc;
      g2o::EdgeSe3PriorPose* edge_prior = new g2o::EdgeSe3PriorPose();
      edge_prior->setVertex(0, optimizer.vertex(0));
      edge_prior->setMeasurement(Twc_prior);
      g2o::Matrix6D info = g2o::Matrix6D::Zero();
      info(0, 0) = info(1, 1) = info(2, 2) = prior_trans_weight;
      info(3, 3) = info(4, 4) = info(5, 5) = prior_rot_weight;
      edge_prior->setInformation(info);
      optimizer.addEdge(edge_prior);
    } else if (prior_rot_weight > 0) {
      Eigen::Matrix3d Rwc_prior = Twc.linear();
      g2o::EdgeSe3PriorRot* edge_prior = new g2o::EdgeSe3PriorRot();
      edge_prior->setVertex(0, optimizer.vertex(0));
      edge_prior->setMeasurement(Rwc_prior);
      Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
      for (int i = 0; i < 3; i++) info(i, i) = prior_rot_weight;
      edge_prior->setInformation(info);
      optimizer.addEdge(edge_prior);
    }
  }

  // Set MapPoint vertices
  const int N = pts3d.size();
  if (N < 3) return false;
  const float deltaMono = sqrt(5.991);
  const float fx = K.at<float>(0, 0);
  const float fy = K.at<float>(1, 1);
  const float cx = K.at<float>(0, 2);
  const float cy = K.at<float>(1, 2);
  status = std::vector<uchar>(N, 255);
  std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vp_edges;
  vp_edges.reserve(N);
  for (int i = 0; i < N; i++) {
    const auto& p2d = pts2d[i];
    const auto& p3d = pts3d[i];
    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    e->setMeasurement(Eigen::Vector2d(p2d.x, p2d.y));
    e->setInformation(Eigen::Matrix2d::Identity());
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(deltaMono);
    e->fx = fx;
    e->fy = fy;
    e->cx = cx;
    e->cy = cy;
    e->Xw << p3d.x, p3d.y, p3d.z;
    optimizer.addEdge(e);
    vp_edges.push_back(e);
  }

  // We perform 4 optimizations, after each optimization we classify observation
  // as inlier/outlier At the next optimization, outliers are not included, but
  // at the end they can be classified as inliers again.
  int nBad = 0;
  for (size_t it = 0; it < 4; it++) {
    vSE3->setEstimate(g2o::SE3Quat(Twc.linear(), Twc.translation()));
    optimizer.initializeOptimization(0);
    if (optimizer.indexMapping().empty()) return false;
    optimizer.optimize(10);
    nBad = 0;
    for (size_t i = 0, iend = vp_edges.size(); i < iend; i++) {
      g2o::EdgeSE3ProjectXYZOnlyPose* e = vp_edges[i];
      if (status[i] == 0) e->computeError();
      const float chi2 = e->chi2();
      if (chi2 > 5.991) {
        status[i] = 0;
        e->setLevel(1);
        nBad++;
      } else {
        status[i] = 255;
        e->setLevel(0);
      }
      if (it == 2) e->setRobustKernel(0);
    }
    if (optimizer.edges().size() < 10) break;
  }
  // Recover optimized pose and return number of inliers
  g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
  g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
  isom2rt(SE3quat_recov, rvec, tvec);
  return (N - nBad) > 10;
}

}  // namespace vs
#endif  // HAVE_G2O
