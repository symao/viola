/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_data_recorder.h"
#include <fstream>
#include <chrono>

#include "vs_os.h"

namespace vs {

DataRecorder::DataRecorder(const char* data_dir, int subdir_type)
    : m_recorder_idx(0), m_logidx_filename("vs_log_idx.txt") {
  switch (subdir_type) {
    case SUBDIR_NONE:
      m_save_dir = std::string(data_dir);
      break;
    case SUBDIR_INDEX:
      m_save_dir = makeSubdirByIndex(data_dir);
      break;
    case SUBDIR_DATE:
      m_save_dir = makeSubdirByDate(data_dir);
      break;
    default:
      break;
  }
  // create subdir
  if (!m_save_dir.empty()) makedirs(m_save_dir.c_str());
}

DataRecorder::~DataRecorder() {
#if VS_DATA_RECORDER_ENABLE_VIDEO_SAVER
  m_video_savers.clear();  // release video saver befor video writer, avoid segment fault
#endif
}

int DataRecorder::createImageRecorder(const char* filename, double video_fps, const cv::Size& video_frame_size) {
  if (m_save_dir.empty()) return -1;
  std::string abs_path = join(m_save_dir.c_str(), filename);
  std::string file_suffix = suffix(filename);
  if (file_suffix == ".png" || file_suffix == ".jpg" || file_suffix == ".jpeg" || file_suffix == ".bmp") {
    // create image saver
    int id = m_recorder_idx++;
    m_img_savers[id] = std::make_shared<ImageSaver>(abs_path.c_str());
    return id;
  } else if (file_suffix == ".mp4" || file_suffix == ".avi") {
#if VS_DATA_RECORDER_ENABLE_VIDEO_SAVER
    // create video saver
    int id = m_recorder_idx++;
    m_video_savers[id] =
        std::make_shared<VideoRecorderAsync>(abs_path.c_str(), cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), video_fps);
    return id;
#else
    printf("[ERROR]DataRecorder:Video saver not implemented, set VS_DATA_RECORDER_ENABLE_VIDEO_SAVER to 1\n");
#endif
  } else {
    printf(
        "[ERROR]DataRecorder:unknown file suffix '%s', support format: '.png', '.jpg', '.jpeg', '.bmp', '.mp4', "
        "'.avi'.\n",
        file_suffix.c_str());
  }
  return -1;
}

int DataRecorder::createStringRecorder(const char* filename) {
  if (m_save_dir.empty()) return -1;
  // create string saver
  int id = m_recorder_idx++;
  std::string file = join(m_save_dir.c_str(), filename);
  std::shared_ptr<DataSaver<std::string>> saver(
      new DataSaver<std::string>(file.c_str(), [](FILE* fp, const std::string& t) { fprintf(fp, "%s", t.c_str()); }));
  m_str_savers[id] = saver;
  return id;
}

void DataRecorder::recordImage(int recorder_id, const cv::Mat& img) {
  // find saver from image savers by recorder id
  auto img_it = m_img_savers.find(recorder_id);
  if (img_it != m_img_savers.end()) {
    // if found, save image
    img_it->second->write(img);
    return;
  }
#if VS_DATA_RECORDER_ENABLE_VIDEO_SAVER
  // find saver from video savers by recorder id
  auto video_it = m_video_savers.find(recorder_id);
  if (video_it != m_video_savers.end()) {
    // if found, save image
    video_it->second->write(img);
    return;
  }
#endif
}

void DataRecorder::recordString(int recorder_id, const std::string& str) {
  // find saver from string savers by recorder id
  auto it = m_str_savers.find(recorder_id);
  if (it != m_str_savers.end()) {
    // if found, save string data
    it->second->push(str);
  }
}

std::string DataRecorder::makeSubdirByIndex(const char* root_dir) {
  std::string fidx = join(root_dir, m_logidx_filename);
  int log_idx = 0;
  // read index from index file
  std::ifstream fin(fidx.c_str());
  if (fin.is_open()) fin >> log_idx;
  fin.close();
  // make subdir with index
  char str[512] = {0};
  snprintf(str, sizeof(str), "%s/%d/", root_dir, log_idx);
  std::string save_dir(str);
  // write index to index file
  std::ofstream fout(fidx.c_str());
  log_idx++;
  fout << log_idx;
  fout.close();
  return save_dir;
}

std::string DataRecorder::makeSubdirByDate(const char* root_dir) {
  std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  char date_str[30] = {0};
  struct tm tm_now;
#ifdef WIN32
  localtime_s(&tm_now, &now);
#else
  localtime_r(&now, &tm_now);
#endif  // WIN32
  std::strftime(date_str, sizeof(date_str), "%Y_%m_%d_%H%M%S", &tm_now);
  return join(root_dir, date_str);
}

}  // namespace vs