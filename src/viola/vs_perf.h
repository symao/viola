/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details performance evaluation for algorithms time cost.
 */
#pragma once
#include <float.h>
#include <stdio.h>
#include <deque>
#include <vector>
#include "vs_tictoc.h"

namespace vs {

/** @brief count min,max,sum,mean for list of input data */
class Counter {
 public:
  Counter() : m_n(0), m_max(FLT_MIN), m_min(FLT_MAX), m_sum(0) {}

  /** @brief add a data */
  void add(float a) {
    m_n++;
    m_sum += a;
    if (a < m_min) m_min = a;
    if (a > m_max) m_max = a;
  }

  /** @brief get maximum of all history data input by add() */
  float max() { return m_max; }

  /** @brief get minimum of all history data input by add() */
  float min() { return m_min; }

  /** @brief get summary of all history data input by add() */
  float sum() { return m_sum; }

  /** @brief get average of all history data input by add() */
  float mean() { return m_n > 0 ? m_sum / m_n : 0; }

 private:
  int m_n;      ///< data count
  float m_max;  ///< maximum data
  float m_min;  ///< minimum data
  float m_sum;  ///< data summary
};

/** @brief Performance counter */
class PerfCounter : public Counter {
 public:
  /** @brief set start time-point */
  void start() { timer.start(); }

  /** @brief set stop time-point */
  void stop() {
    timer.stop();
    add(timer.getMsec());
  }

  /** @brief print performance result */
  void print(const char* header = NULL, bool verbose = false) {
    if (header) printf("[%s]", header);
    if (verbose)
      printf("%.3f(%.3f, %.3f) ms.\n", mean(), min(), max());
    else
      printf("%.3f ms.\n", mean());
  }

 private:
  Timer timer;  ///< timer to measure time cost
};

/** @brief whether enable performance profiling
 * @note perfXxx() works only when perfEnable(true)
 */
void perfEnable(bool enable);

/** @brief set start time-point
 * @param[in]name: name for performance, used when perfPrint
 */
void perfBegin(const char* name);

/** @brief set stop time-point
 * @param[in]name: name for performance, used when perfPrint
 */
void perfEnd(const char* name);

/** @brief print current performance result
 * @param[in]name: name for performance, if not set, print all performance items
 * @param[in]verbose: whether print in verbose mode
 */
void perfPrint(const char* name = NULL, bool verbose = false);

/** @brief get average time cost in milliseconds of input item name
 * @param[in]name: name for performance
 * @return time cost in milliseconds of input item name
 */
float perfAvg(const char* name);

/** @brief set start time-point
 * @param[in]name: name for performance, used when perfPrint
 */
inline void perfStart(const char* name) { perfBegin(name); }

/** @brief set stop time-point
 * @param[in]name: name for performance, used when perfPrint
 */
inline void perfStop(const char* name) { perfEnd(name); }

} /* namespace vs */
