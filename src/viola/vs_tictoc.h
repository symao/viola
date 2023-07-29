/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details evaluate the time cost.
 */
#pragma once
#include <chrono>
#include <deque>
#include <string>

namespace vs {

/** @brief Timer to use count time cost
 * @code
 * Timer t1;
 * t1.start();
 * // do something
 * t1.stop();
 * double cost = t1.getMsec(); // time cost in millisecond between start() and stop()
 * @endcode
 */
class Timer {
 public:
  /** @brief constructor
   * @note call start() inside when construct
   */
  Timer() { start(); }

  /** @brief set start time point */
  void start() { t1 = std::chrono::system_clock::now(); }

  /** @brief set stop time point */
  double stop() {
    t2 = std::chrono::system_clock::now();
    return getMsec();
  }

  /** @brief get time cost in second from start() to stop() */
  double getSec() const { return getMsec() * 1e-3; }

  /** @brief get time cost in millisecond from start() to stop() */
  double getMsec() const { return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() * kms; }

 private:
  std::chrono::system_clock::time_point t1, t2;  ///< time point at start and stop
  const static double kms;                       ///< convert count to millisecond
};

/** @brief first call tic, second call toc */
void tictoc(const char* name);

/** @brief start timer */
void tic(const char* name);

/** @brief return time between tic and toc [millisecond] */
double toc(const char* name);

/** @brief sleep milliseconds */
void msleep(double msec);

/** @brief get software ts, reset to 0 when application start. [second] */
double getSoftTs();

/** @brief a high-precision loop rate sleeper. if current loop execute time is less than loop time,
 * it will sleep the rest of time.
 * @code
 * Rater loop_rate(100);  //build a 100hz rater
 * while (1) {
 *   //do something
 *   loop_rate.sleep(); // if the excecute time of above code is less than 10ms, this line will sleep to ensure that the
 *                      // loop time is 10ms.
 * }
 * @endcode
 */
class Rater {
 public:
  /** @brief constructor
   * @param[in] fps the loop rate [HZ]
   * @param[in] precision sleep precision [ms]
   */
  Rater(double fps, double precision = 1);

  /** @brief sleep under max frame rate
   * sleep to make sure time diff to last sleep not less than min time interval(=1/fps)
   */
  void sleep();

  double getFps() const { return m_fps; }

 private:
  Timer m_timer;
  double m_fps;          // frequency[HZ]
  double m_loopms;       // loop step[ms]
  double m_precisionms;  // rater precision[ms], the max sleep error.
};

/** @brief Calculate process FPS(Frames Per Second)
 * @code
 * FpsCalculator fc;
 * for (int i = 0; i < 100; i++) {
 *   fc.start();
 *   // do process
 *   fc.stop();
 * }
 * double fps = fc.fps(); // get process FPS
 * double costms = fc.costms(); // get average frame process cost in milliseconds
 * @endcode
 */
class FpsCalculator {
 public:
  /** @brief constructor
   * @param[in]queue_len: buffer length to store latest frame cost
   */
  explicit FpsCalculator(int queue_len = 10);

  /** @brief set start time-point */
  void start();

  /** @brief set stop time-point */
  double stop();

  /** @brief get process FPS */
  double fps();

  /** @brief get average frame process cost in milliseconds */
  double costms();

  /** @brief get max process cost in milliseconds */
  double maxms();

 private:
  Timer m_timer;               ///< timer to measure time cost
  std::deque<double> m_queue;  ///< buffer to store latest frame cost
  int m_queue_len;             ///< buffer length to store latest frame cost
};

class TsSleeper {
 public:
  /** @brief sleep to timestamp in seconds, first call will init ref timestamp and not sleep. */
  void sleepTo(double ts) {
    if (!init_) {
      timer_.start();
      init_ts_ = ts;
      init_ = true;
      return;
    }
    timer_.stop();
    double sleep_ts = ts - init_ts_ - timer_.getSec();
    if (sleep_ts > 0) vs::msleep(sleep_ts * 1e3);
  }

 private:
  bool init_ = false;   ///< whether init ts valid
  double init_ts_ = 0;  ///< init sleep to ts
  vs::Timer timer_;     ///< inner timer which starts in init
};

class TimeIt {
 public:
  TimeIt(const char* name = "") : name_(name) { timer_.start(); }

  ~TimeIt() {
    timer_.stop();
    printf("[%s]cost %.2f ms\n", name_.c_str(), timer_.getMsec());
  }

 private:
  Timer timer_;
  std::string name_;
};

#define VS_TIME_IT() vs::TimeIt vs_time_it(__func__)

} /* namespace vs */
