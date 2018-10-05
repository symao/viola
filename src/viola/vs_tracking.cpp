/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_tracking.h"

#include <map>
#include <deque>
#include <mutex>
#include <thread>

#include "vs_basic.h"
#include "vs_feature.h"
#include "vs_geometry2d.h"
#include "vs_tictoc.h"

namespace vs {

static cv::Matx33f quat2rot(float qw, float qx, float qy, float qz) {
  float theta = 2 * acos(qw);
  float k = theta / sin(theta / 2);
  cv::Mat rvec = (cv::Mat_<float>(1, 3) << qx * k, qy * k, qz * k);
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  return R;
}

static cv::Matx33f rvec2rot(float rx, float ry, float rz) {
  cv::Mat rvec = (cv::Mat_<float>(1, 3) << rx, ry, rz);
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  return R;
}

static bool isTruncated(const TrackBox& o, const cv::Size& img_size, int border = 1) {
  return o.xmin < border || o.ymin < border || o.xmax >= img_size.width - border || o.ymax >= img_size.height - border;
}

static float iouCrop(const TrackBox& a, const TrackBox& b, const cv::Size& img_size) {
  float xmin1 = std::max(a.xmin, 0.0f);
  float ymin1 = std::max(a.ymin, 0.0f);
  float xmax1 = std::min(a.xmax, static_cast<float>(img_size.width));
  float ymax1 = std::min(a.ymax, static_cast<float>(img_size.height));
  float xmin2 = std::max(b.xmin, 0.0f);
  float ymin2 = std::max(b.ymin, 0.0f);
  float xmax2 = std::min(b.xmax, static_cast<float>(img_size.width));
  float ymax2 = std::min(b.ymax, static_cast<float>(img_size.height));
  float s = std::max(0.0f, std::min(xmax1, xmax2) - std::max(xmin1, xmin2)) *
            std::max(0.0f, std::min(ymax1, ymax2) - std::max(ymin1, ymin2));  // intersect
  return s / ((xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - s);
}

bool medianFlowVote(const std::vector<cv::Point2f>& prev_pts, const std::vector<cv::Point2f>& cur_pts, float& shift_x,
                    float& shift_y, float& scale) {
  if (prev_pts.empty() || prev_pts.size() != cur_pts.size()) return false;

  int N = prev_pts.size();
  std::vector<float> xs, ys, scales;
  xs.reserve(N);
  ys.reserve(N);
  for (int i = 0; i < N; i++) {
    auto diff = cur_pts[i] - prev_pts[i];
    xs.push_back(diff.x);
    ys.push_back(diff.y);
  }
  scales.reserve(N);
  for (int i = 1; i < N; i++) {
    for (int j = 0; j < i; j++) {
      double d1 = cv::norm(prev_pts[i] - prev_pts[j]);
      double d2 = cv::norm(cur_pts[i] - cur_pts[j]);
      scales.push_back(d1 > VS_EPS ? d2 / d1 : 1);
    }
  }
  shift_x = vecMedian(xs);
  shift_y = vecMedian(ys);
  scale = vecMedian(scales);
  return true;
}

BoxTrackConfig::BoxTrackConfig() {
  drop_truncated = true;  ///< whether drop box when box truncated
  refine_truncated = false;
  thres_merge_iou = 0.2f;  ///< if two box iou large than this thrshold, merge to one.
  min_box_feature = 5;     ///< min feature count to track box
  min_area_rate = 0.0f;    ///< min area rate for valid track box, rate = w*h/(img_w*img_h)
  max_area_rate = 1.0f;    ///< max area rate for valid track box
  debug_draw = false;
}

class BoxTrackerAsyncImpl {
 public:
  BoxTrackerAsyncImpl();

  ~BoxTrackerAsyncImpl();

  /** @brief init
   * @param[in]detector: detector
   * @param[in]tracker: single object tracker
   * @param[in]detect_gap: minimum time gap between two detection frame. [second]
   * @return whether init succeed
   * */
  bool init(const std::shared_ptr<BoxDetectApi>& detector, float detect_gap = 0.5f, float detect_thres_conf = 0.5f,
            float detect_thres_iou = 0.6f);

  /** @brief detect one image
   * @param[in]img_rgb: current image, used to find bounding box
   * @param[in]detect_rot_rad: rot image before detect inference to get better result
   * @return detected/tracked box list
   * */
  std::vector<TrackBox> process(const cv::Mat& img_rgb, float detect_rot_rad = 0);

  void setConfig(const BoxTrackConfig& cfg) { cfg_ = cfg; }

  BoxTrackConfig getConfig() const { return cfg_; }

  void debugDraw(cv::Mat& img_rgb, const std::vector<std::string>& names = {}, bool draw_box = true,
                 bool draw_feature = true);

  cv::Mat getDebugImg() { return debug_img_; }

 private:
  struct FramePack {
    int img_idx;
    cv::Size img_size;
    FeatureTracker::FeatureList features;
    std::vector<TrackBox> objects;
  };

  struct DetectInput {
    int img_idx = -1;
    float rot_rad = 0;
    cv::Mat img;
  };

  struct DetectOutput {
    int img_idx = -1;
    std::vector<TrackBox> objects;
  };

  bool init_;                                        ///< whether initialized
  bool exit_;                                        ///< detect thread exit trigger flag, set to true in deconstructor
  BoxTrackConfig cfg_;                               ///< algorithm paramters
  int process_idx_;                                  ///< process image idx
  TrackBox::IdType box_id_;                          ///< box id start from 1, 0 means invalid.
  float detect_gap_;                                 ///< detect gap in second
  float detect_thres_conf_;                          ///< detect confidence threshold
  float detect_thres_iou_;                           ///< detect iou threshold in NMS
  double last_detect_ts_;                            ///< last detect timestamp, used for detect gap
  std::deque<FramePack> frame_buffer_;               ///< history frame buffer
  std::shared_ptr<FeatureTracker> feature_tracker_;  ///< feature tracker
  std::shared_ptr<BoxDetectApi> detector_;           ///< detector handler
  cv::Mat debug_img_;                                ///< debug draw

  // parameter for asynchronized detection thread
  std::shared_ptr<std::thread> detect_thread_;  ///< detect thread
  std::mutex mtx_detect_input_;                 ///< mutex for detect input data
  std::mutex mtx_detect_output_;                ///< mutex for detect output data
  DetectInput detect_input_;                    ///< detect input data
  DetectOutput detect_output_;                  ///< detect output data

  /** @brief detect thread */
  void detectThread();

  /** @brief run detection once by network inference */
  DetectOutput detectOnce(const DetectInput& input);

  /** @brief track box from detect frame */
  void trackDetectBox(FramePack& cur_frame);

  /** @brief track box with features, only features inside box is used. */
  bool trackBox(const FeatureTracker::FeatureList& prev_features, const FeatureTracker::FeatureList& cur_features,
                const TrackBox& prev_box, TrackBox& cur_box);

  /** @brief add new box into current frame, if IOU between new box and old box is too large, drop
   * old box
   * @param[in/out]cur_frame: current image frame
   * @param[in]new_obj: new object to be added
   * @param[in]can_refine: whether refine tracked box
   * @param[in]thres_iou: iou threshold to drop old box
   * */
  void addNewBox(FramePack& cur_frame, TrackBox new_obj, bool can_refine = true);

  const FeatureTracker::Feature* findFeatureById(const FeatureTracker::FeatureList& features,
                                                 FeatureTracker::FeatureIDType id);

  const FramePack* findFrame(int img_idx);

  TrackBox shift(const TrackBox& o, float shift_x, float shift_y, float shift_scale) {
    float cx = (o.xmin + o.xmax) * 0.5f;
    float cy = (o.ymin + o.ymax) * 0.5f;
    float hw = (o.xmax - o.xmin) * 0.5f;
    float hh = (o.ymax - o.ymin) * 0.5f;
    cx += shift_x;
    cy += shift_y;
    hw *= shift_scale;
    hh *= shift_scale;
    TrackBox res;
    res.class_id = o.class_id;
    res.score = o.score;
    res.xmin = cx - hw;
    res.xmax = cx + hw;
    res.ymin = cy - hh;
    res.ymax = cy + hh;
    return res;
  }
};

BoxTrackerAsyncImpl::BoxTrackerAsyncImpl()
    : init_(false),
      exit_(false),
      process_idx_(0),
      box_id_(1),
      detect_gap_(0),
      detect_thres_conf_(0.5),
      detect_thres_iou_(0.6),
      last_detect_ts_(0) {
  feature_tracker_ = std::make_shared<FeatureTracker>();
  auto cfg = feature_tracker_->getConfig();
  cfg.feature_type = 1;
  cfg.max_corner = 300;
  cfg.min_corner_dist = 15;
  cfg.lk_patch_size = cv::Size(15, 15);
  feature_tracker_->setConfig(cfg);
}

BoxTrackerAsyncImpl::~BoxTrackerAsyncImpl() {
  exit_ = true;
  if (detect_thread_.get()) detect_thread_->join();
}

bool BoxTrackerAsyncImpl::init(const std::shared_ptr<BoxDetectApi>& detector, float detect_gap, float detect_thres_conf,
                               float detect_thres_iou) {
  detector_ = detector;
  detect_gap_ = detect_gap;
  if (detector_.get()) {
    init_ = true;
    detect_thread_.reset(new std::thread(std::bind(&BoxTrackerAsyncImpl::detectThread, this)));
  }
  detect_thres_conf_ = detect_thres_conf;
  detect_thres_iou_ = detect_thres_iou;
  return init_;
}

std::vector<TrackBox> BoxTrackerAsyncImpl::process(const cv::Mat& img_rgb, float detect_rot_rad) {
  if (!init_) return std::vector<TrackBox>();

  // set input to detect thread
  // Note: use deep copy for image, since image might not be processed in time in detect thread.
  mtx_detect_input_.lock();
  detect_input_.img = cv::Mat();
  img_rgb.copyTo(detect_input_.img);
  detect_input_.img_idx = process_idx_;
  detect_input_.rot_rad = detect_rot_rad;
  mtx_detect_input_.unlock();

  // feature track
  FramePack cur_frame;
  cur_frame.img_idx = process_idx_;
  cur_frame.img_size = img_rgb.size();
  cv::Mat gray;
  cv::cvtColor(img_rgb, gray, cv::COLOR_RGB2GRAY);
  auto track_res = feature_tracker_->process(gray);
  cur_frame.features = track_res.features;

  // medianflow for prev track boxes
  if (!frame_buffer_.empty()) {
    const auto& prev_frame = frame_buffer_.back();
    for (const auto& o : prev_frame.objects) {
      TrackBox cur_obj;
      bool ok = trackBox(prev_frame.features, cur_frame.features, o, cur_obj);
      if (ok) {
        cur_obj.id = o.id;
        cur_obj.track_cnt = o.track_cnt + 1;
        addNewBox(cur_frame, cur_obj);
      }
    }
  }

  // medianflow for new detect boxes
  trackDetectBox(cur_frame);

  // post process
  frame_buffer_.push_back(cur_frame);
  if (frame_buffer_.size() > 10) frame_buffer_.pop_front();
  process_idx_++;

  if (cfg_.debug_draw) {
    img_rgb.copyTo(debug_img_);
    debugDraw(debug_img_);
  }

  return cur_frame.objects;
}

void BoxTrackerAsyncImpl::debugDraw(cv::Mat& img_rgb, const std::vector<std::string>& names, bool draw_box,
                                    bool draw_feature) {
  if (frame_buffer_.empty()) return;
  const auto& cur_frame = frame_buffer_.back();
  if (draw_box) {
    for (const auto& o : cur_frame.objects) {
      cv::rectangle(img_rgb, cv::Point(o.xmin, o.ymin), cv::Point(o.xmax, o.ymax), cv::Scalar(255, 0, 0), 1);
      char so[128] = {0};
      if (names.empty())
        snprintf(so, 128, "class:%d(%.2f) id:%d track:%d", o.class_id, o.score, static_cast<int>(o.id),
                 static_cast<int>(o.track_cnt));
      else
        snprintf(so, 128, "class:%s(%.2f) id:%d track:%d", names[o.class_id].c_str(), o.score, static_cast<int>(o.id),
                 static_cast<int>(o.track_cnt));
      cv::putText(img_rgb, so, cv::Point(o.xmin, o.ymin + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
    char s[128] = {0};
    snprintf(s, 128, "#%d box:%d", process_idx_ - 1, static_cast<int>(cur_frame.objects.size()));
    cv::putText(img_rgb, s, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 120, 0), 2);
  }
  if (draw_feature) {
    for (const auto& p : cur_frame.features) {
      float k = std::min(p.track_cnt / 10.0f, 1.0f);
      cv::Scalar color(255 * k, 100, 255 * (1 - k));
      cv::circle(img_rgb, p, 2, color, -1);
    }
    if (frame_buffer_.size() > 1) {
      const auto& prev_frame = frame_buffer_[frame_buffer_.size() - 2];
      for (const auto& p : cur_frame.features) {
        const auto* ptr = findFeatureById(prev_frame.features, p.id);
        if (ptr) {
          cv::line(img_rgb, p, *ptr, cv::Scalar(0, 255, 0), 1);
        }
      }
    }
  }
}

void BoxTrackerAsyncImpl::trackDetectBox(BoxTrackerAsyncImpl::FramePack& cur_frame) {
  // get detect output
  mtx_detect_output_.lock();
  DetectOutput detect_out = detect_output_;
  detect_output_.objects.clear();  // only used once
  mtx_detect_output_.unlock();
  if (detect_out.objects.empty()) return;

  int detect_img_idx = detect_out.img_idx;
  if (detect_img_idx == cur_frame.img_idx) {
    // detection is fast enough that detect frame is same as current frame, add box into current
    // frame
    for (const auto& o : detect_out.objects)
      addNewBox(cur_frame, o, cfg_.refine_truncated || !isTruncated(o, cur_frame.img_size));
  } else {
    // detection frame is earlier than current frame, use median flow to track detect box from
    // detect frame to current frame
    const FramePack* ptr_find = findFrame(detect_img_idx);
    if (ptr_find == NULL) return;
    // use median flow to track new detect box from history frame into current frame, then
    // tracked box into current frame
    for (const auto& o : detect_out.objects) {
      TrackBox cur_obj;
      bool ok = trackBox(ptr_find->features, cur_frame.features, o, cur_obj);
      if (ok) addNewBox(cur_frame, cur_obj, cfg_.refine_truncated || !isTruncated(o, cur_frame.img_size));
    }
  }
}

bool BoxTrackerAsyncImpl::trackBox(const FeatureTracker::FeatureList& prev_features,
                                   const FeatureTracker::FeatureList& cur_features, const TrackBox& prev_box,
                                   TrackBox& cur_box) {
  // data association with feature id
  std::vector<cv::Point2f> pts1, pts2;
  for (const auto& f1 : prev_features) {
    if (!prev_box.inside(f1.x, f1.y)) continue;  // skip if feature not in box
    const auto* ptr = findFeatureById(cur_features, f1.id);
    if (ptr) {
      pts1.push_back(f1);
      pts2.push_back(*ptr);
    }
  }
  if (static_cast<int>(pts1.size()) < cfg_.min_box_feature) return false;
  // median flow calculation
  float shift_x, shift_y, shift_scale;
  bool ok = medianFlowVote(pts1, pts2, shift_x, shift_y, shift_scale);
  if (ok) cur_box = shift(prev_box, shift_x, shift_y, shift_scale);
  return ok;
}

const FeatureTracker::Feature* BoxTrackerAsyncImpl::findFeatureById(const FeatureTracker::FeatureList& features,
                                                                    FeatureTracker::FeatureIDType id) {
  // use binary search since features ids are sorted in ascend order.
  // Note: if ids not sorted, binary search can not be used.
  int l = 0;
  int r = features.size() - 1;
  while (l <= r) {
    int m = (l + r) / 2;
    const auto& f = features[m];
    if (f.id == id)
      return &f;
    else if (f.id < id)
      l = m + 1;
    else
      r = m - 1;
  }
  return NULL;
}

const BoxTrackerAsyncImpl::FramePack* BoxTrackerAsyncImpl::findFrame(int img_idx) {
  if (frame_buffer_.empty() || img_idx < frame_buffer_.front().img_idx || img_idx > frame_buffer_.back().img_idx)
    return NULL;
  // Note: since ids is continuous, frame address can be calculate with img_idx
  int start_idx = frame_buffer_.front().img_idx;
  if (frame_buffer_.back().img_idx - start_idx + 1 == static_cast<int>(frame_buffer_.size())) {
    return &frame_buffer_[img_idx - start_idx];
  } else {
    for (const auto& frame : frame_buffer_) {
      if (frame.img_idx == img_idx) return &frame;
    }
    return NULL;
  }
}

void BoxTrackerAsyncImpl::addNewBox(BoxTrackerAsyncImpl::FramePack& cur_frame, TrackBox new_obj, bool can_refine) {
  if (cfg_.drop_truncated && isTruncated(new_obj, cur_frame.img_size, 3)) return;
  float area_rate =
      static_cast<float>(rectCrop(new_obj.toRect(), cur_frame.img_size).area()) / cur_frame.img_size.area();
  if (area_rate < cfg_.min_area_rate || area_rate > cfg_.max_area_rate) return;
  // clamp box into image range before calculate IOU, since old box might be out of region.
  for (size_t i = 0; i < cur_frame.objects.size(); i++) {
    auto& o = cur_frame.objects[i];
    if (iouCrop(o, new_obj, cur_frame.img_size) > cfg_.thres_merge_iou) {
      if (can_refine) {                        // whether refine old box with new box
        if (o.class_id == new_obj.class_id) {  // same object, used old id
          new_obj.id = o.id;
          new_obj.track_cnt = o.track_cnt;
          o = new_obj;
        } else {  // new object, use new id
          o = new_obj;
          if (o.id == 0) o.id = box_id_++;
        }
      }
      return;
    }
  }
  if (new_obj.id == 0) new_obj.id = box_id_++;
  cur_frame.objects.push_back(new_obj);
}

BoxTrackerAsyncImpl::DetectOutput BoxTrackerAsyncImpl::detectOnce(const BoxTrackerAsyncImpl::DetectInput& input) {
  DetectOutput res;
  if (input.img_idx < 0 || input.img.empty()) return res;
  res.img_idx = input.img_idx;
  if (fabs(input.rot_rad) > 0.1) {
    FastImageRoter roter(input.img.size(), input.rot_rad,
                         VS_PI_4);  // force fast rot to fix box size
    cv::Mat img;
    roter.rot(input.img, img);
    detector_->detect(img, res.objects, detect_thres_conf_, detect_thres_iou_);
    for (auto& o : res.objects) {
      cv::Rect_<float> rect_rot(o.xmin, o.ymin, o.xmax - o.xmin, o.ymax - o.ymin);
      auto rect_back = roter.rotBack(rect_rot);
      o.xmin = rect_back.x;
      o.ymin = rect_back.y;
      o.xmax = rect_back.x + rect_back.width;
      o.ymax = rect_back.y + rect_back.height;
    }
  } else {
    detector_->detect(input.img, res.objects, detect_thres_conf_, detect_thres_iou_);
  }
  return res;
}

void BoxTrackerAsyncImpl::detectThread() {
  while (detector_.get() && !exit_) {
    if (!init_) {
      msleep(1);
      continue;
    }
    mtx_detect_input_.lock();
    double img_ts = getSoftTs();
    auto input = detect_input_;
    detect_input_.img = cv::Mat();
    mtx_detect_input_.unlock();
    if (input.img_idx >= 0 && !input.img.empty() && (detect_gap_ >= 0 && img_ts - last_detect_ts_ > detect_gap_)) {
      DetectOutput output = detectOnce(input);
      // if detect new box, replace output with new box, output will be cleared after used
      if (!output.objects.empty()) {
        mtx_detect_output_.lock();
        detect_output_ = output;
        mtx_detect_output_.unlock();
      }
      last_detect_ts_ = img_ts;
    }
    msleep(1);
  }
}

BoxTrackerAsync::BoxTrackerAsync() { impl_ = std::make_shared<BoxTrackerAsyncImpl>(); }

BoxTrackerAsync::~BoxTrackerAsync() {}

bool BoxTrackerAsync::init(const std::shared_ptr<BoxDetectApi>& detector, float detect_gap, float detect_thres_conf,
                           float detect_thres_iou) {
  return impl_->init(detector, detect_gap, detect_thres_conf, detect_thres_iou);
}

std::vector<TrackBox> BoxTrackerAsync::process(const cv::Mat& img_rgb, float detect_rot_rad) {
  return impl_->process(img_rgb, detect_rot_rad);
}

BoxTrackConfig BoxTrackerAsync::getConfig() { return impl_->getConfig(); }

void BoxTrackerAsync::setConfig(const BoxTrackConfig& cfg) { impl_->setConfig(cfg); }

cv::Mat BoxTrackerAsync::getDebugImg() { return impl_->getDebugImg(); }

BoxPose::BoxPose() : init_(false), cur_pose_(cv::Affine3f::Identity()) {}

bool BoxPose::update(const TrackBox& box, const cv::Matx33f& Rcw, const cv::Vec4f& intrinsic) {
  if (!init_)
    return initObject(box, Rcw, intrinsic);
  else if (box.id == prev_box_.id)
    return trackObject(box, Rcw, intrinsic);
  else
    return false;
}

bool BoxPose::initObject(const TrackBox& box, const cv::Matx33f& Rcw, const cv::Vec4f& intin) {
  prev_box_ = box;
  prev_dist_ = 1.0f;
  cv::Vec3f ray = centerRay(box, intin);
  auto ray_w = Rcw * ray;
  float yaw = atan2(-ray_w[1], -ray_w[0]);
  init_rot_ = rvec2rot(0, 0, yaw);
  cur_pose_ = cv::Affine3f(Rcw.t() * init_rot_, ray * prev_dist_);  // object -> camera
  init_ = true;
  return init_;
}

bool BoxPose::trackObject(const TrackBox& box, const cv::Matx33f& Rcw, const cv::Vec4f& intin) {
  float k1 = static_cast<float>(box.xmax - box.xmin) / (box.xmax - box.xmin);
  float k2 = static_cast<float>(box.ymax - box.ymin) / (box.ymax - box.ymin);
  float k = (k1 + k2) / 2;
  // float k = std::min(k1, k2);
  float cur_dist = prev_dist_ / k;
  cv::Vec3f ray = centerRay(box, intin);
  cur_pose_ = cv::Affine3f(Rcw.t() * init_rot_, ray * prev_dist_);
  prev_box_ = box;
  prev_dist_ = cur_dist;
  return true;
}

cv::Vec3f BoxPose::centerRay(const TrackBox& box, const cv::Vec4f& intin) {
  cv::Vec3f ray(((box.xmin + box.xmax) * 0.5 - intin[2]) / intin[0],
                ((box.ymin + box.ymax) * 0.5 - intin[3]) / intin[1], 1.0f);
  ray /= cv::norm(ray);
  return ray;
}

BoxPoseEstimator::BoxPoseEstimator() {
  Rci_ = cv::Matx33f(0, -1, 0, -1, 0, 0, 0, 0, -1);
  fov_rad_ = deg2rad(60);
}

void BoxPoseEstimator::process(const std::vector<TrackBox>& boxes, const cv::Vec4f& imu_quat, const cv::Size& img_size,
                               std::vector<cv::Affine3f>& obj_poses, std::vector<TrackBox::IdType>& obj_ids) {
  float u0 = img_size.width / 2.0f;
  float v0 = img_size.height / 2.0f;
  float f = img_size.width / 2.0f / tan(fov_rad_ / 2.0f);
  cv::Matx33f Rcw = quat2rot(imu_quat[0], imu_quat[1], imu_quat[2], imu_quat[3]) * Rci_;
  cv::Vec4f intrin(f, f, u0, v0);

  std::map<TrackBox::IdType, BoxPose> cur_objs;
  for (const auto& box : boxes) {
    auto it = objects_.find(box.id);
    if (it == objects_.end()) {
      BoxPose new_obj;
      if (new_obj.update(box, Rcw, intrin)) cur_objs[box.id] = new_obj;
    } else {
      BoxPose& old_obj = it->second;
      if (old_obj.update(box, Rcw, intrin)) cur_objs[box.id] = old_obj;
    }
  }

  obj_poses.clear();
  obj_poses.reserve(cur_objs.size());
  obj_ids.clear();
  obj_ids.reserve(cur_objs.size());
  for (const auto& it : cur_objs) {
    obj_poses.push_back(it.second.getPose());
    obj_ids.push_back(it.first);
  }
  objects_ = cur_objs;
}

void BoxPoseEstimator::setFovDeg(float fov_deg) { fov_rad_ = deg2rad(fov_deg); }

cv::Mat BoxPoseEstimator::draw(const cv::Mat& img, const std::vector<cv::Affine3f>& obj_poses,
                               const std::vector<TrackBox::IdType>& obj_ids) {
  cv::Mat img_draw;
  if (img.channels() == 1)
    cv::cvtColor(img, img_draw, cv::COLOR_GRAY2RGB);
  else
    img.copyTo(img_draw);

  int cnt = obj_poses.size();
  std::vector<cv::Point3f> coord_pts = {cv::Point3f(0, 0, 0), cv::Point3f(0.1, 0, 0), cv::Point3f(0, 0.1, 0),
                                        cv::Point3f(0, 0, 0.1)};
  cv::Mat K = fov2K<float>(fov_rad_, img.size());
  cv::Mat D;
  for (int i = 0; i < cnt; i++) {
    const auto& pose = obj_poses[i];
    auto id = obj_ids[i];
    std::vector<cv::Point2f> pts2d;
    cv::projectPoints(coord_pts, pose.rvec(), pose.translation(), K, D, pts2d);
    cv::line(img_draw, pts2d[0], pts2d[1], cv::Scalar(255, 0, 0), 2);
    cv::line(img_draw, pts2d[0], pts2d[2], cv::Scalar(0, 255, 0), 2);
    cv::line(img_draw, pts2d[0], pts2d[3], cv::Scalar(0, 0, 255), 2);
    char str[128] = {0};
    snprintf(str, 128, "%d", static_cast<int>(id));
    cv::putText(img_draw, str, pts2d[0] + cv::Point2f(5, 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
  }
  return img_draw;
}

}  // namespace vs