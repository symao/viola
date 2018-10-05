/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details save image file in an asynchronous manner
 */
#pragma once
#include <mutex>
#include <thread>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "vs_tictoc.h"
#include "vs_stb_image.h"

namespace vs {

/** @brief Asynchronized image saver, save image in another thread, so call write won't block
 * @see ::VideoSaver
 */
class ImageSaver {
 public:
  /** @brief constructor
   * @param[in]save_format: image name format, such as "%06d.jpg"
   */
  explicit ImageSaver(const char* save_format)
      : m_save_format(save_format),
        m_stop(false),
        m_save_idx(0),
        m_thread_ptr(new std::thread(std::bind(&ImageSaver::run, this))) {
    m_img_vec.reserve(20);
  }

  ~ImageSaver() {
    m_stop = true;
    m_thread_ptr->join();
  }

  /** @brief save a image
   * @note image name is auto generated with save index and input save format
   * @note only clone image into process queue, and truely save in another thread
   */
  void write(const cv::Mat& img) {
    m_mtx.lock();
    m_img_vec.push_back(img.clone());
    m_mtx.unlock();
  }

 private:
  std::string m_save_format;                  ///< name format for saved image
  bool m_stop;                                ///< whether stop save thread
  int m_save_idx;                             ///< saved index
  std::shared_ptr<std::thread> m_thread_ptr;  ///< save thread
  std::vector<cv::Mat> m_img_vec;             ///< image data queue to be saved
  std::mutex m_mtx;                           ///< mutex for read/write image data queue

  /** @brief save thread, read image from data queue, and write image. */
  void run() {
    try {
      while (!m_stop) {
        m_mtx.lock();
        auto img_vec = m_img_vec;
        m_img_vec.clear();
        m_mtx.unlock();
        if (!img_vec.empty()) {
          for (const auto& m : img_vec) {
            if (m.empty()) {
              printf("[ERROR]ImageSaver: image is empty, not saving.\n");
              continue;
            } else {
              char fsave[256] = {0};
              snprintf(fsave, 256, m_save_format.c_str(), m_save_idx++);
              vs::imwrite(fsave, m);
            }
          }
        }
        msleep(500);
      }
    } catch (...) {
      printf("[ERROR]:Video saver thread quit expectedly.\n");
    }
  }
};

} /* namespace vs */
