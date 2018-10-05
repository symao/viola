/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details general data recorder for image (into video or file) and string content (into file), asynchronized.
 */
#pragma once
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <opencv2/core.hpp>
#include "vs_data.h"
#include "vs_imsaver.h"

#define VS_DATA_RECORDER_ENABLE_VIDEO_SAVER 1  ///< open this if you want to record camera to video file
#if VS_DATA_RECORDER_ENABLE_VIDEO_SAVER
#include "vs_video_saver.h"
#endif

namespace vs {

/** @brief Asynchronized data recorder, which record camera to video or images and record string to txt file
 * It's used to recorder SLAM or VIO data, recorder log or algorithm datas.
 */
class DataRecorder {
 public:
  /** @enum Whether create subdir in input data dir for each run */
  enum SubdirType {
    SUBDIR_NONE = 0,   ///< do not use subdir, data will be overwrited
    SUBDIR_INDEX = 1,  ///< create subdir in index, such as: 0,1,2...
    SUBDIR_DATE = 2    ///< create subdir in running date, such as: 2021_11_23_155800
  };

  /** @brief Constructor
   * @param[in]data_dir: input dir to store recorder datas
   * @param[in]subdir_type: see SubdirType
   */
  DataRecorder(const char* data_dir, int subdir_type = SUBDIR_NONE);

  /** @brief Deconstructor */
  ~DataRecorder();

  /** @brief create a image recorder, data can be saved into video or image file
   * @param[in]filename: if save video, input video name 'xxx.mp4',
   *                     if save image, input image name format 'xxx_%d.png', where %d is image save index
   * @param[in]video_fps: input video FPS if save video
   * @param[in]video_frame_size: input video frame size if save video
   * @return recorder id
   */
  int createImageRecorder(const char* filename = "%06d.png", double video_fps = 30,
                          const cv::Size& video_frame_size = cv::Size(0, 0));

  /** @brief create a string recorder, data will be saved into file
   * @param[in]filename: save file name
   * @return recorder id
   */
  int createStringRecorder(const char* filename);

  /** @brief record single frame
   * @param[in]recorder_id: recorder from createImageRecorder
   * @param[in]img: image to be saved
   */
  void recordImage(int recorder_id, const cv::Mat& img);

  /** @brief record single str
   * @param[in]recorder_id: recorder from createStringRecorder
   * @param[in]str: string to be saved
   */
  void recordString(int recorder_id, const std::string& str);

  /** @brief get current save dir */
  std::string getSaveDir() const { return m_save_dir; }

 private:
  typedef DataSaver<std::string> StringSaver;
  int m_recorder_idx;                   ///< recorder index, used by image recorders as well as string recorders
  std::string m_save_dir;               ///< current data save dir, may be different to input data dir if enable subdir
  const std::string m_logidx_filename;  ///< file to store current subdir index, created/modified in input data dir
  std::map<int, std::shared_ptr<StringSaver>> m_str_savers;  ///< recorders to store string datas, <recorder_id, saver>
  std::map<int, std::shared_ptr<ImageSaver>> m_img_savers;   ///< recorders to store images, <recorder_id, saver>
#if VS_DATA_RECORDER_ENABLE_VIDEO_SAVER
  std::map<int, std::shared_ptr<VideoRecorderAsync>>
      m_video_savers;  ///< recorders to store videos, <recorder_id, saver>
#endif

  /** @brief make subdir by index
   * @param[in]root_dir: root dir, log index file will be create in this dir
   * @return absolute path of subdir
   */
  std::string makeSubdirByIndex(const char* root_dir);

  /** @brief make subdir by current date
   * @param[in]root_dir: root dir
   * @return absolute path of subdir
   */
  std::string makeSubdirByDate(const char* root_dir);
};

}  // namespace vs