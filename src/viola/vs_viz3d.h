/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details 3D visulization based on cv::Viz in an asynchronous manner.
 */
#pragma once
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/viz.hpp>
#include <thread>
#include <mutex>
#include "vs_tictoc.h"
#include "vs_basic.h"

namespace vs {
/** @brief 3D visualization using cv::viz::Viz3d in asynchronized thread */
class Viz3D {
 public:
  Viz3D() : m_stop(false), m_thread_ptr(new std::thread(std::bind(&Viz3D::run, this))) { m_viz.setBackgroundColor(); }

  ~Viz3D() {
    m_stop = true;
    m_thread_ptr->join();
    reset();
  }

  void setViewPose(const cv::Affine3d& pose) {
    m_mtx.lock();
    m_viz.setViewerPose(pose);
    m_mtx.unlock();
  }

  /** @brief reset viz, clear all widgets */
  void reset() {
    m_mtx.lock();
    m_viz.removeAllWidgets();
    m_widget_table.clear();
    m_mtx.unlock();
  }

  /** @brief update visualize widget
   * @param[in]id: widget id, overwrite widget data if widget id exists,
   * @param[in]w: widget data
   */
  void updateWidget(const std::string& id, const cv::viz::Widget& w) {
    m_mtx.lock();
    m_widget_table[id] = w;
    m_mtx.unlock();
  }

  cv::viz::Viz3d& getViz() { return m_viz; }

  void removeWidget(const std::string& id, bool fuzzy_match = false) {
    m_mtx.lock();
    if (fuzzy_match) {
      for (auto it = m_widget_table.begin(); it != m_widget_table.end();) {
        if (vs::has(it->first, id)) {
          m_viz.removeWidget(it->first);
          it = m_widget_table.erase(it);
        } else {
          ++it;
        }
      }
    } else {
      m_widget_table.erase(id);
      m_viz.removeWidget(id);
    }
    m_mtx.unlock();
  }

 private:
  bool m_stop;                                            ///< whether stop viz thread
  cv::viz::Viz3d m_viz;                                   ///< opencv viz handler
  std::map<std::string, cv::viz::Widget> m_widget_table;  ///< widget list to viz
  std::shared_ptr<std::thread> m_thread_ptr;              ///< viz thread
  std::mutex m_mtx;

  /** @brief visualization thread */
  void run() {
    try {
      while (!m_viz.wasStopped() && !m_stop) {
        m_mtx.lock();
        if (!m_widget_table.empty()) {
          for (const auto& m : m_widget_table) {
            m_viz.showWidget(m.first, m.second);
          }
          m_viz.spinOnce();
        }
        m_mtx.unlock();
        msleep(100);
      }
    } catch (...) {
      printf("[ERROR]:Viz3d thread quit expectedly.\n");
    }
  }
};

} /* namespace vs */