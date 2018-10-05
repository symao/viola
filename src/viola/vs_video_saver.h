/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details save video in an asynchronous manner.
 */
#pragma once
#include <mutex>
#include <thread>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "vs_tictoc.h"

namespace vs {

/** @brief video recorder, same use as cv::VideoWriter, but later initialized when call write */
class VideoRecorder {
 public:
  explicit VideoRecorder(const std::string& name, int fourcc = cv::VideoWriter::fourcc('D', 'I', 'V', 'X'),
                         double fps = 30.0)
      : m_name(name), m_fourcc(fourcc), m_fps(fps) {}

  ~VideoRecorder() {
    if (m_writer.get()) m_writer->release();
  }

  void write(const cv::Mat& img) {
    if (img.empty()) {
      printf("[ERROR]VideoRecorder: empty image, not saving.\n");
      return;
    }
    if (!m_writer.get()) m_writer.reset(new cv::VideoWriter(m_name, m_fourcc, m_fps, img.size()));
    if (img.channels() == 1) {
      cv::Mat bgr;
      cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
      m_writer->write(bgr);
    } else if (img.channels() == 3) {
      m_writer->write(img);
    } else {
      printf("[ERROR]VideoRecorder: not support image channals %d. \n", img.channels());
    }
  }

  std::shared_ptr<cv::VideoWriter> writer() { return m_writer; }

 private:
  std::string m_name;
  int m_fourcc;
  double m_fps;
  std::shared_ptr<cv::VideoWriter> m_writer;
};

/** @brief Asynchronized video saver, save video in another thread, so call write won't block
 * @see ::ImageSaver
 */
class VideoRecorderAsync {
 public:
  VideoRecorderAsync(const std::string& name, int fourcc = cv::VideoWriter::fourcc('D', 'I', 'V', 'X'),
                     double fps = 30.0)
      : m_stop(false), m_recorder(name, fourcc, fps) {
    m_img_vec.reserve(20);
    m_thread_ptr = std::make_shared<std::thread>(std::bind(&VideoRecorderAsync::run, this));
  }

  ~VideoRecorderAsync() {
    m_stop = true;
    m_thread_ptr->join();
  }

  /** @brief write image
   * @note only clone image into process queue, and truely write to video in another thread
   */
  void write(const cv::Mat& img) {
    m_mtx.lock();
    m_img_vec.push_back(img.clone());
    m_mtx.unlock();
  }

  std::shared_ptr<cv::VideoWriter> writer() { return m_recorder.writer(); }

 private:
  bool m_stop;                                ///< whether stop video writing thread
  VideoRecorder m_recorder;                   ///< video recorder
  std::shared_ptr<std::thread> m_thread_ptr;  ///< video writing thread
  std::vector<cv::Mat> m_img_vec;             ///< image queue buffer
  std::mutex m_mtx;                           ///< mutex of read/write image queue buffer

  /** @brief video writing thread */
  void run() {
    try {
      while (!m_stop) {
        m_mtx.lock();
        auto img_vec = m_img_vec;
        m_img_vec.clear();
        m_mtx.unlock();
        for (const auto& m : img_vec) m_recorder.write(m);
        msleep(500);
      }
    } catch (...) {
      printf("[ERROR]VideoRecorderAsync:Video saver thread quit expectedly.\n");
    }
  }
};

} /* namespace vs */