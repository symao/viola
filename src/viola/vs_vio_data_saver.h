/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-06-27 19:07
 * @details Recorder VIO data in inner format.
 */
#pragma once
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include "vs_data_recorder.h"

namespace vs {

/** @brief data saver for VIO datasets */
class VioDataSaver {
 public:
  VioDataSaver(const std::string& save_dir, int subdir_type = DataRecorder::SUBDIR_DATE);

  void pushImu(double ts, const cv::Vec3f& gyro, const cv::Vec3f& acc);

  void pushImu(double ts, const cv::Vec3f& gyro, const cv::Vec3f& acc, const cv::Vec3f& mag);

  void pushCamera(double ts, const cv::Mat& img);

  void pushGtPose(double ts, const cv::Vec3f& pos_xyz, const cv::Vec4f& quat_wxyz);

 private:
  std::shared_ptr<DataRecorder> m_recorder;
  int m_img_id, m_imgts_id, m_imu_id, m_gt_id;
  char m_buf[256] = {0};
};

}  // namespace vs