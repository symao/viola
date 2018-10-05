/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details 2D boundingbox tracking based on point features.
 */
#pragma once

#include <memory>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/core/affine.hpp>

namespace vs {

bool medianFlowVote(const std::vector<cv::Point2f>& prev_pts, const std::vector<cv::Point2f>& cur_pts, float& shift_x,
                    float& shift_y, float& scale);

struct TrackBox {
  typedef uint64_t IdType;
  IdType id = 0;           ///< object id, assigned by box tracker
  int class_id = -1;       ///< class id, -1 means not
  float xmin = 0;          ///< xmin of bounding box in image range, [0,cols)
  float ymin = 0;          ///< ymin of bounding box in image range, [0,rows)
  float xmax = 0;          ///< xmax of bounding box in image range, [0,cols)
  float ymax = 0;          ///< ymax of bounding box in image range, [0,rows)
  float score = 0;         ///< confidence, [0,1]
  uint64_t track_cnt = 0;  ///< tracked frame count

  /** @brief convert to cv::Rect */
  cv::Rect toRect() const { return cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin); }

  /** @brief whether point in box */
  bool inside(float x, float y) const { return xmin <= x && x <= xmax && ymin <= y && y <= ymax; }

  /** @brief box width */
  float w() const { return xmax - xmin; }

  /** @brief box height */
  float h() const { return ymax - ymin; }

  /** @brief area */
  float area() const { return w() * h(); }

  /** @brief calculate IoU between two box */
  float iou(const TrackBox& b) const {
    float s = std::max(0.0f, std::min(xmax, b.xmax) - std::max(xmin, b.xmin)) *
              std::max(0.0f, std::min(ymax, b.ymax) - std::max(ymin, b.ymin));  // intersect
    return s / ((xmax - xmin) * (ymax - ymin) + (b.xmax - b.xmin) * (b.ymax - b.ymin) - s);
  }
};

/** @brief abstract class for detection
 * One need to create a derived class from this abstract class and implement the detect api.
 */
class BoxDetectApi {
 public:
  BoxDetectApi() {}

  virtual ~BoxDetectApi() {}

  /** @brief process a image and return detected result
   * @param[in]img: input image in 3-channels, RGB order
   * @param[out]result: detect bounding boxes
   * @param[in]thres_conf: min valid confidence of boxes
   * @param[in]thres_iou: max IOU in non-maximum-suppression
   * */
  virtual void detect(const cv::Mat& img, std::vector<TrackBox>& result, float thres_conf = 0.5f,
                      float thres_iou = 0.5f) = 0;
};

class BoxTrackerAsyncImpl;  ///< implementation for box tracking, forward declaration

struct BoxTrackConfig {
  bool drop_truncated;    ///< whether drop box when box truncated
  bool refine_truncated;  ///< whether refine truncated box
  float thres_merge_iou;  ///< if two box iou large than this thrshold, merge to one.
  int min_box_feature;    ///< min feature count to track box
  float min_area_rate;    ///< min area rate for valid track box, rate = w*h/(img_w*img_h)
  float max_area_rate;    ///< max area rate for valid track box
  bool debug_draw;        ///< whether draw debug image when process, this is SLOW, set true only in debugging

  BoxTrackConfig();
};

class BoxTrackerAsync {
 public:
  BoxTrackerAsync();

  ~BoxTrackerAsync();

  /** @brief init
   * @param[in]detector: object detector
   * @param[in]detect_gap: minimum time gap between two detection frame. [second]
   * @param[in]detect_thres_conf: confidence/score threshold for detection
   * @param[in]detect_thres_iou: IOU threshold for detection
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

  /** @brief get current configure parameters
   * Call this to get default configuration before call setConfig()
   * */
  BoxTrackConfig getConfig();

  /** @brief set configure parameters
   * @note One must call this api before init.
   * */
  void setConfig(const BoxTrackConfig& cfg);

  /** @brief get debug image after process, set BoxTrackConfig::debug_draw to true when call this
   * @return if debug_draw flag is true, return debug image same size to img_rgb, else return empty mat
   * */
  cv::Mat getDebugImg();

 private:
  std::shared_ptr<BoxTrackerAsyncImpl> impl_;  ///< impl handler
};

class BoxPose {
 public:
  BoxPose();

  /** @brief update a box with camera poses
   * @param[in]box 2D bounding box with id
   * @param[in]Rcw rotation matrix transform point from camera to world frame
   * @param[in]intrinsic camera intrinsic, order:fx,fy,cx,cy
   * @return whether update success and whether pose is valid
   * */
  bool update(const TrackBox& box, const cv::Matx33f& Rcw, const cv::Vec4f& intrinsic);

  /** @brief get box 3D pose in camera frame
   * @note camera frame definition: x-right y-downside z-forward
   * @note we don't know the 3D pose in world frame, since we do NOT input camera position in
   * world frame. User can transform this 3D pose from camera frame to world frame outside. */
  cv::Affine3f getPose() const { return cur_pose_; }

 private:
  bool init_;              ///< whether initialized ok
  TrackBox prev_box_;      ///< previous bounding box
  float prev_dist_;        ///< previous distance from camera center to box center
  cv::Matx33f init_rot_;   ///< init orientation of object
  cv::Affine3f cur_pose_;  ///< object pose in camera frame

  /** @brief init new object */
  bool initObject(const TrackBox& box, const cv::Matx33f& Rcw, const cv::Vec4f& intrin);

  /** @brief update exist object */
  bool trackObject(const TrackBox& box, const cv::Matx33f& Rcw, const cv::Vec4f& intin);

  /** @brief calculate a unit vector in camera frame, which start from camera to box center */
  cv::Vec3f centerRay(const TrackBox& box, const cv::Vec4f& intin);
};

class BoxPoseEstimator {
 public:
  BoxPoseEstimator();

  /** @brief recover object poses in camera frame with tracking boxes
   * @param[in]boxes: tracking 2D boxes
   * @param[in]quat: input imu quaternion, (qw,qx,qy,qz)
   * @param[out]obj_poses: object poses in camera coordinate system.
   * @param[out]obj_ids: object ids
   * */
  void process(const std::vector<TrackBox>& boxes, const cv::Vec4f& imu_quat, const cv::Size& img_size,
               std::vector<cv::Affine3f>& obj_poses, std::vector<TrackBox::IdType>& obj_ids);

  /** @brief set filed-of-vision of camera in degree*/
  void setFovDeg(float fov_deg);

  /** @brief set rotation from camera to IMU
   * set rear camera of phone as default extrinsic, user need to set extrinsic if use other camera
   * imu configuration
   */
  void setExtrinsic(cv::Matx33f Rci) { Rci_ = Rci; }

  /** @brief only used for debug */
  cv::Mat draw(const cv::Mat& img, const std::vector<cv::Affine3f>& obj_poses,
               const std::vector<TrackBox::IdType>& obj_ids);

 private:
  float fov_rad_;                                ///< horizontal Field-Of-Vision in radian
  cv::Matx33f Rci_;                              ///< rotation from camera to IMU, default: rear camera
  std::map<TrackBox::IdType, BoxPose> objects_;  ///< estimate objects
};

}  // namespace vs
