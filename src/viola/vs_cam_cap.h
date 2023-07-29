/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details camera capture using opencv, capture image with rolling buffer in real-time, auto save video.
 */
#pragma once
#include <memory>
#include <opencv2/videoio.hpp>
#include <vector>

#include "vs_os.h"
#include "vs_data.h"
#include "vs_tictoc.h"
#include "vs_video_saver.h"

namespace vs {

class CamCapture {
 public:
  CamCapture() : m_init(false), m_start(false), m_exit(false), m_idx(0), m_buffer(64) {}

  CamCapture(int device, const char* fsave = NULL, cv::Size set_size = cv::Size(0, 0))
      : m_init(false), m_start(false), m_exit(false), m_idx(0), m_buffer(64) {
    init(device, fsave, set_size);
  }

  CamCapture(const char* device, const char* fsave = NULL, cv::Size set_size = cv::Size(0, 0))
      : m_init(false), m_start(false), m_exit(false), m_idx(0), m_buffer(64) {
    init(device, fsave, set_size);
  }

  virtual ~CamCapture() {
    m_exit = true;
    if (m_cap_thread.get()) m_cap_thread->join();
  }

  bool init(int device, const char* fsave = NULL, cv::Size set_size = cv::Size(0, 0)) {
    m_cap.open(device);
    if (!m_cap.isOpened()) {
      printf("[ERROR] cannot open camera %d\n", device);
      return false;
    }
    m_init = initImpl(fsave, set_size);
    return m_init;
  }

  bool init(const char* device, const char* fsave = NULL, cv::Size set_size = cv::Size(0, 0)) {
    m_cap.open(device);
    if (!m_cap.isOpened()) {
      printf("[ERROR] cannot open camera %s\n", device);
      return false;
    }
    m_init = initImpl(fsave, set_size);
    return m_init;
  }

  void start() { m_start = true; }

  void stop() { m_start = false; }

  double getLatest(cv::Mat& img) {
    if (!m_init) {
      printf("[ERROR]Get data failed. Init first.\n");
      return 0;
    }
    if (!m_start) {
      printf("[ERROR]Get data failed. Capture not start.\n");
      return 0;
    }
    if (m_idx <= 0) {
      msleep(100);
      if (m_idx <= 0) {
        printf("[WARN]Get data failed. Buffer is empty.\n");
        return 0;
      }
    }
    auto& p = m_buffer[m_idx - 1];
    img = p.second;
    return p.first;
  }

  double read(cv::Mat& img) { return getLatest(img); }

 private:
  bool m_init;
  bool m_start;
  bool m_exit;
  int m_idx;
  cv::VideoCapture m_cap;
  std::shared_ptr<VideoRecorderAsync> m_vw;
  std::shared_ptr<std::thread> m_cap_thread;
  std::string m_imgts_file;
  cv::Size m_img_size;
  CircularQueue<std::pair<double, cv::Mat>> m_buffer;
  std::shared_ptr<DataSaver<double>> m_ts_saver;

  virtual bool initImpl(const char* fsave, cv::Size set_size) {
    if (!m_cap.isOpened()) return false;

    if (set_size.width != 0 || set_size.height != 0) {
      m_cap.set(cv::CAP_PROP_FRAME_WIDTH, set_size.width);
      m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, set_size.height);
    }

    cv::Mat img;
    if (m_cap.read(img)) {
      m_img_size = img.size();
    } else {
      printf("[ERROR] read image failed\n");
      return false;
    }

    if (fsave) {
      m_vw = std::make_shared<VideoRecorderAsync>(fsave, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'),
                                                  m_cap.get(cv::CAP_PROP_FPS));
      m_imgts_file = std::string(fsave) + ".txt";
    }

    m_cap_thread = std::make_shared<std::thread>(std::bind(&CamCapture::captureThread, this));
    return true;
  }

  virtual bool capture() {
    if (!m_start) return false;
    auto& data = m_buffer[m_idx];
    if (!m_cap.read(data.second)) {
      return false;
    }
    data.first = getSysTs();
    m_idx++;

    if (m_vw.get()) m_vw->write(data.second);
    if (m_imgts_file.length() > 0) {
      if (!m_ts_saver.get()) {
        m_ts_saver.reset(
            new DataSaver<double>(m_imgts_file.c_str(), [](FILE* fp, const double& d) { fprintf(fp, "%f\n", d); }));
      }
      m_ts_saver->push(data.first);
    }
    return true;
  }

  void captureThread() {
    while (!m_exit) {
      capture();
      msleep(5);
    }
  }
};

class ExposureControl {
 public:
  ExposureControl(float tar_bright = 75, float max_expo = 100, float init_expo = 10)
      : m_tar_bright(tar_bright),
        m_max_expo(max_expo),
        m_cur_expo(init_expo),
        m_mode(0),
        m_kp(0.2),
        m_ki(0),
        m_kd(0.05),
        m_err(0),
        m_prev_err(0),
        m_pprev_err(0) {}

  void init(float tar_bright, float max_expo, float init_expo = 10) {
    m_tar_bright = tar_bright;
    m_max_expo = max_expo;
    m_cur_expo = init_expo;
  }

  void setPID(double kp, double ki = -1, double kd = -1) {
    if (kp >= 0) m_kp = kp;
    if (ki >= 0) m_ki = ki;
    if (kd >= 0) m_kd = kd;
  }

  float exposure(const cv::Mat& m, const cv::Mat& mask = cv::Mat()) {
    float cur_bright = mask.rows == 0 ? cv::mean(m).val[0] : cv::mean(m, mask).val[0];
    exposureAdjust(cur_bright);
    return m_cur_expo;
  }

 private:
  float m_tar_bright;
  float m_max_expo;
  float m_cur_expo;
  int m_mode;
  float m_kp, m_ki, m_kd;
  float m_err, m_prev_err, m_pprev_err;

  enum AutoExposureMode { AE_MODE_PID = 0, AE_MODE_DEADZONE };

  void exposureAdjust(float cur_bright) {
    m_err = m_tar_bright - cur_bright;

    // handle state matchine
    switch (m_mode) {
      case AE_MODE_PID: {
        if (fabs(m_err) < 1) m_mode = AE_MODE_DEADZONE;
        break;
      }
      case AE_MODE_DEADZONE: {
        float abserr = fabs(m_err);
        if (abserr > 5) {
          m_mode = AE_MODE_PID;
        }
        break;
      }
    }

    // process mode
    float u = 0;
    switch (m_mode) {
      case AE_MODE_PID: {
        u = m_kp * m_err + m_kd * (m_err - m_prev_err);
        m_cur_expo += u;
        break;
      }
      case AE_MODE_DEADZONE:
        break;
    }

    // clip
    if (m_cur_expo < 1)
      m_cur_expo = 1;
    else if (m_cur_expo > m_max_expo)
      m_cur_expo = m_max_expo;

#if 0  // only for debug, and must be close when releasing, since it will affect capture fps
        static FILE* fp = fopen((g_log_dir+"/exposure.txt").c_str(), "w");
        if(fp)
        {
            fprintf(fp, "%f %d %.1f %.0f %.1f %.1f\n",
                getCurTimestamp(), mode, cur_bright, m_cur_expo, err, u);
            fflush(fp);
        }
        // printf("%d %.1f %.0f %.1f %.1f\n", mode, cur_bright, m_cur_expo, err, u);
#endif

    m_pprev_err = m_prev_err;
    m_prev_err = m_err;
  }
};

} /* namespace vs */