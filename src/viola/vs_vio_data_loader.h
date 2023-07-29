/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details VIO dataset loader for EuROC, KITTI, TUM-VIO, UZH-VIO as well as inner format.
 */
#pragma once
#include <memory>
#include <string>
#include <functional>
#include <opencv2/core.hpp>
#include "vs_vio_type.h"

/** @brief vio dataloader, support multiple open dataset such as: EuROC, KITTI, UZH VIO, TUM VIO
 * One can simply use VioDataLoader::play() to run dataset with input callback.
 * One can also run without setting callback, use nextType to get next play datatype, and use
 * fetchImu(),fetchCamera(),fetchGtPose() to fetch next data cooresponding to correct data type.
 * @code
 * auto dataset = vio::createVioDataLoader(data_dir);
 * if (!(dataset && dataset->ok())) {
 *   printf("[ERROR]Dataset open failed '%s'\n", data_dir);
 *   return;
 * }
 * vio::ImuData cur_imu;
 * vio::CameraData cur_cam;
 * vio::PoseData cur_gt_pose;
 * bool ok = true;
 * while (ok) {
 *   ok = false;
 *   switch (dataset->nextType()) {
 *     case vio::MSG_IMU:
 *       ok = dataset->fetchImu(cur_imu, true);
 *       // todo: handle IMU data
 *       break;
 *     case vio::MSG_CAMERA:
 *       ok = dataset->fetchCamera(cur_cam, true);
 *       // todo: handle camera data(images)
 *       break;
 *     case vio::MSG_GTPOSE:
 *       ok = dataset->fetchGtPose(cur_gt_pose, true);
 *       // todo: handle ground truth pose
 *     default:
 *       break;
 *   }
 * }
 * @endcode
 */

namespace vs {

/** @brief dataloader for VIO datasets, abstract class which define api */
class VioDataLoader {
 public:
  /** @brief dataset type */
  enum DatasetType {
    DATASET_UNKNOW = 0,
    DATASET_EUROC = 1,       ///< The EuRoC MAV Dataset:
                             ///< https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
    DATASET_KITTI = 2,       ///< The KITTI Vision Benchmark: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    DATASET_UZH_VIO = 3,     ///< The UZH-FPV Drone Racing Dataset: https://fpv.ifi.uzh.ch/
    DATASET_TUM_VIO = 4,     ///< The TUM VI Benchmark: https://vision.in.tum.de/data/datasets/visual-inertial-dataset
    DATASET_ZJU_VIO = 5,     ///< ZJU-SenseTime VISLAM Benchmark: http://www.zjucvg.net/eval-vislam/
    DATASET_VIOLA = 99,  ///< Dataset format of shuyuanmao123@gmail.com
  };

  /** @brief sensor message type */
  enum MessageType {
    MSG_NONE = 0,    ///< imu acc and gyro
    MSG_IMU = 1,     ///< imu acc and gyro
    MSG_CAMERA = 2,  ///< camera images
    MSG_GTPOSE = 3,  ///< ground-truth pose
  };

  VioDataLoader();

  virtual ~VioDataLoader() = default;

  /** @brief init dataloader with input data files
   * @param[in]init_files: configure path for different dataset type
   *  | dataset type  | find imu file | find cam file | find ground truth file|
   *  |---------------|------------|---------------|---------------|-----------------------|
   *  |DATASET_EURO| mav0/imu0/data.csv | mav0/cam0/data.csv | mav0/state_groundtruth_estimate0/data.csv |
   *  |DATASET_KITTI| ??? | ??? | ??? |
   *  |DATASET_UZH_VIO| imu.txt | left_images.txt | / |
   *  |DATASET_TUM_VIO| mav0/imu0/data.csv | mav0/cam0/data.csv | mav0/mocap0/data.csv |
   *  |DATASET_ZJU_VIO| imu/data.csv | camera/data.csv | groundtruth/data.csv |
   *  |DATASET_VIOLA| imu_meta.txt | imgts.txt, img.avi | / |
   * @return whether init success
   */
  bool init(const std::string& data_dir, const std::vector<std::string>& append_files = {});

  /** @brief whether dataset loaded ok */
  bool ok() const { return m_init; }

  /** @brief get dataset name */
  std::string name() const;

  /** @brief get dataset type */
  DatasetType datasetType() const { return m_dataset_type; }

  /** @brief set time diff between image ts and sensor ts, plus this to image timestamp */
  void setTd(double td) { m_td = td; }

  /** @brief get time diff between image ts and sensor ts */
  double td() const { return m_td; }

  /** @brief set start timestamp, earlier data will be skip */
  void setStartTs(double ts) { m_start_ts = ts; }

  /** @brief get start timestamp */
  double startTs() const { return m_start_ts; }

  /** @brief set stop timestamp, earlier data will be skip */
  void setStopTs(double ts) { m_stop_ts = ts; }

  /** @brief get stop timestamp */
  double stopTs() const { return m_stop_ts; }

  /** @brief set mono mode, in mono mode, only output left image even dataset contains stereo data */
  void setMonoMode(bool mono) { m_mono = mono; }

  /** @brief whether in mono mode */
  bool mono() const { return m_mono; }

  /** @brief get next message type */
  MessageType nextType();

  /** @brief fetch IMU data, call this only when nextType() is MSG_IMU
   * @param[out]imu: fetched data, valid only when return true
   * @return true if fetch success
   */
  bool fetchImu(ImuData& imu, bool verbose = false);

  /** @brief fetch camera data, call this only when nextType() is MSG_CAMERA
   * @param[out]cam: fetched data, valid only when return true
   * @return true if fetch success
   */
  bool fetchCamera(CameraData& cam, bool verbose = false);
  /** @brief fetch ground truth pose data, call this only when nextType() is MSG_GTPOSE
   * @param[out]gt_pose: fetched data, valid only when return true
   * @return true if fetch success
   */
  bool fetchGtPose(PoseData& gt_pose, bool verbose = false);

  /** @brief play dataset with input callback
   * @param[in]imu_callback: callback for imu data
   * @param[in]camera_callback: callback for camera data
   * @param[in]gt_callback: callback for ground truth pose data
   * @param[in]sleep: whether sleep for data timestamp
   * @param[in]verbose: whether print verbose data
   */
  void play(std::function<void(const ImuData&)> imu_callback, std::function<void(const CameraData&)> camera_callback,
            std::function<void(const PoseData&)> gt_callback, bool sleep = false, bool verbose = false);

 protected:
  bool m_init;              ///< whether init ok
  bool m_mono;              ///< whether use mono mode, otherwise use all cameras
  double m_start_ts;        ///< start timestamp in second
  double m_stop_ts;         ///< start timestamp in second
  double m_td;              ///< time diff between camera and imu
  MessageType m_next_type;  ///< next date type
  ImuData m_next_imu;       ///< buffer for next imu data
  CameraData m_next_cam;    ///< buffer for next camera data
  PoseData m_next_gt_pose;  ///< buffer for next ground-truth pose data
  DatasetType m_dataset_type = DATASET_UNKNOW;

  /** @brief init implementation, load data with input files */
  virtual bool initImpl(const std::string& data_dir, const std::vector<std::string>& append_files) = 0;

  /** @brief read next imu data from dataset */
  virtual bool readNextImu(ImuData& imu) = 0;

  /** @brief read next camera data from dataset */
  virtual bool readNextCamera(CameraData& cam) = 0;

  /** @brief read next ground-truth pose data from dataset */
  virtual bool readNextGtPose(PoseData& gt_pose) = 0;

  /** @brief check next data type, find the earliest data among all sensors */
  MessageType checkNextType();
};

/** @brief create VIO dataloader with dataset type */
std::shared_ptr<VioDataLoader> createVioDataLoader(VioDataLoader::DatasetType dataset_type);

/** @brief Create VIO dataloader and load data from input data dir
 * @param[in]data_dir: input data dir
 * @param[in]dataset_type: input dataset type, if not set, it will be deduced inside with data_dir
 * @param[in]append_files: append files for VioDataLoader::init()
 * @return dataloader pointer. if dataset type cannot be deduced, return empty pointer.
 */
std::shared_ptr<VioDataLoader> createVioDataLoader(
    const std::string& data_dir, VioDataLoader::DatasetType dataset_type = VioDataLoader::DATASET_UNKNOW,
    const std::vector<std::string>& append_files = {});

}  // namespace vs