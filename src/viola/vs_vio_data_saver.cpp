/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-06-27 19:07
 * @details
 */
#include "vs_vio_data_saver.h"

namespace vs {

VioDataSaver::VioDataSaver(const std::string& save_dir, int subdir_type)
    : m_img_id(-1), m_imgts_id(-1), m_imu_id(-1), m_gt_id(-1) {
  m_recorder = std::make_shared<DataRecorder>(save_dir.c_str(), DataRecorder::SUBDIR_DATE);
}

void VioDataSaver::pushImu(double ts, const cv::Vec3f& gyro, const cv::Vec3f& acc) {
  if (m_imu_id < 0) m_imu_id = m_recorder->createStringRecorder("imu_meta.txt");
  snprintf(m_buf, 256, "%f %f %f %f %f %f %f\n", ts, acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2]);
  m_recorder->recordString(m_imu_id, m_buf);
}

void VioDataSaver::pushImu(double ts, const cv::Vec3f& gyro, const cv::Vec3f& acc, const cv::Vec3f& mag) {
  if (m_imu_id < 0) m_imu_id = m_recorder->createStringRecorder("imu_meta.txt");
  snprintf(m_buf, 256, "%f %f %f %f %f %f %f %f %f %f\n", ts, acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2], mag[0],
           mag[1], mag[2]);
  m_recorder->recordString(m_imu_id, m_buf);
}

void VioDataSaver::pushCamera(double ts, const cv::Mat& img) {
  if (m_img_id < 0) m_img_id = m_recorder->createImageRecorder("img.avi", 30, img.size());
  if (m_imgts_id < 0) m_imgts_id = m_recorder->createStringRecorder("imgts.txt");
  snprintf(m_buf, 256, "%f\n", ts);
  m_recorder->recordString(m_imgts_id, m_buf);
  m_recorder->recordImage(m_img_id, img);
}

void VioDataSaver::pushGtPose(double ts, const cv::Vec3f& pos_xyz, const cv::Vec4f& quat_wxyz) {
  if (m_gt_id < 0) m_gt_id = m_recorder->createStringRecorder("gt_pose.txt");
  snprintf(m_buf, 256, "%f %f %f %f %f %f %f %f\n", ts, pos_xyz[0], pos_xyz[1], pos_xyz[2], quat_wxyz[0], quat_wxyz[1],
           quat_wxyz[2], quat_wxyz[3]);
  m_recorder->recordString(m_gt_id, m_buf);
}

}  // namespace vs