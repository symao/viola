/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details 2D line segment matching and alignment
 */
#pragma once
#include <memory>
#include <map>
#include <opencv2/core.hpp>

namespace vs {

struct LineSeg2D {
  LineSeg2D(double x1, double y1, double x2, double y2) : p1(x1, y1), p2(x2, y2) { cal(); }
  LineSeg2D(const cv::Point2f& _p1, const cv::Point2f& _p2) : p1(_p1), p2(_p2) { cal(); }

  void cal() {
    dir = p2 - p1;
    length = cv::norm(dir);
    dir /= length;
    // if(dir.x < 0) dir = -dir; //make dir x alway un-negative
    theta = atan2(dir.y, dir.x);
    d = dir.cross(p1);
  }

  cv::Point2f p1, p2;
  cv::Point2f dir;
  double length;
  double theta;
  double d;  // distance from orgin to line, the sign indicate the direction
};

typedef std::vector<LineSeg2D> LineSeg2DList;

bool near(const LineSeg2D& l1, const LineSeg2D& l2, double max_angle = 0.05, double max_dist = 0.1,
          double min_overlap = 0);

double distance(const LineSeg2D& l1, const LineSeg2D& l2, double max_angle = 0.05, double max_dist = 0.1,
                double min_overlap = 0, double w_angle = 1, double w_dist = 1);

/** @brief solve transformation between line correspondences.
    NOTE: model as well as target are same size and correspoinding each other.

    @param[in] model: model line list
    @param[in] target: target line list
    @param[out] T: 2D transformation (x,y,theta), NOTE that the z of Point3f
                is theta. Model = R(theta) * Target + t(x,y)
*/
bool solveTransform(const LineSeg2DList& model, const LineSeg2DList& target, cv::Point3f& T);

bool solveTransform(const LineSeg2DList& model, const LineSeg2DList& target, cv::Point3f& T, cv::Mat& info);

/** @brief Iterative Closest Line.
    same as iterate closest point(ICP), match target lines to model lines, which
    result in a 2D transformation SE(2) that ModelLine = SE(2) * TargetLine.

    @param[in] model: model line list, lines in map recommended.
    @param[in] target: target line list, lines in scan recommended.
    @param[in\out] transform: 2D transformation (x,y,theta), NOTE that the z of Point3f
                is theta. Model = R(theta) * Target + t(x,y).
    @param[out] match_ids: correspoints of <targer_id, model_id>, NOTE that use target
                line id as key, since multiple target line will corresponding to one
                model line.
    @param[in] use_init_guess: whether use the input transform as init guess.
    @param[in] thres_angle: max angle diff in [rad] between two matched lines
    @param[in] thres_angle: max distance in [meter] between two matched lines, NOTE when
                calculate dist, we calculate the distance from the center SHORT line to
                the LONG line.
    @param[in] proj_affine: whether to affine the transform to use the init guess when
                lines pair(such as one line pair) cannot fix a full transformation.
    @return whether ICL success and transform is valid.
*/
bool ICL(const LineSeg2DList& model, const LineSeg2DList& target, cv::Point3f& transform, std::map<int, int>& match_ids,
         bool use_init_guess = false, double thres_angle = 0.3, double thres_dist = 1, bool proj_affine = true);

bool ICL(const LineSeg2DList& model, const LineSeg2DList& target, cv::Point3f& transform, std::map<int, int>& match_ids,
         cv::Mat& info, bool use_init_guess = false, double thres_angle = 0.3, double thres_dist = 1,
         bool proj_affine = true);

bool ICL(const LineSeg2DList& model, const LineSeg2DList& target, cv::Point3f& transform, bool use_init_guess = false,
         double thres_angle = 0.3, double thres_dist = 1, bool proj_affine = true);

/** @brief line one nearest search, simplest case for knn*/
class LineSeg2DNN {
 public:
  virtual ~LineSeg2DNN() {}

  virtual void build(const LineSeg2DList& linelist) = 0;

  virtual bool nearest(const LineSeg2D& lquery, LineSeg2D& lnearest, double max_angle, double max_dist) = 0;
};

class ICLSolver {
 public:
  ICLSolver() : m_proj_affine(true) {}

  ICLSolver(const LineSeg2DList& model, bool store_grid = false) : m_proj_affine(true) { setModel(model, store_grid); }

  void setModel(const LineSeg2DList& model, bool store_grid = false);

  bool match(const LineSeg2DList& target, cv::Point3f& transform, bool use_init_guess = false, double thres_angle = 0.3,
             double thres_dist = 1);

  void setProjectAffine(bool do_affine) { m_proj_affine = do_affine; }

  // only valid when match succeed
  cv::Mat getInfo() { return m_fisher_info; }

 private:
  std::shared_ptr<LineSeg2DNN> m_map;
  bool m_proj_affine;
  cv::Mat m_fisher_info;
};
} /* namespace vs */
