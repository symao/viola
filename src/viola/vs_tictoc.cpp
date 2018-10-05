/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_tictoc.h"

#include <stdio.h>
#include <thread>
#include <map>
#include <string>

namespace vs {

const double Timer::kms =
    static_cast<double>(std::chrono::microseconds::period::num) / std::chrono::microseconds::period::den * 1000.0;

static Timer p_program_timer;
static std::map<std::string, Timer> p_timer_list;

double getSoftTs() {
  p_program_timer.stop();
  return p_program_timer.getSec();
}

void msleep(double msec) { std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(msec * 1e3))); }

void tic(const char* name) { p_timer_list[std::string(name)].start(); }

double toc(const char* name) {
  auto it = p_timer_list.find(name);
  if (it == p_timer_list.end()) {
    printf("[ERROR] toc(\"%s\"): should call tic first.\n", name);
    return 0;
  } else {
    return it->second.stop();
  }
}

void tictoc(const char* name) {
  auto it = p_timer_list.find(name);
  if (it == p_timer_list.end()) {
    tic(name);
  } else {
    double cost = it->second.stop();
    printf("tictoc: \"%s\" cost %.2f ms\n", name, cost);
    p_timer_list.erase(it);
  }
}

Rater::Rater(double fps, double precision) : m_fps(fps), m_precisionms(precision) {
  m_loopms = 1000.0 / m_fps;
  m_timer.start();
}

void Rater::sleep() {
  m_timer.stop();
  double time = m_timer.getMsec();
  while (time < m_loopms) {
    // if need sleep long for eg. over 3 times precisions,
    // just sleep first and then run pooling query
    double delta = m_loopms - time - m_precisionms * 3;
    if (delta > 0) {
      msleep(delta);
    } else {
      msleep(m_precisionms);
    }
    m_timer.stop();
    time = m_timer.getMsec();
  }
  m_timer.start();
}

FpsCalculator::FpsCalculator(int queue_len) : m_queue(queue_len), m_queue_len(queue_len) {}

void FpsCalculator::start() { m_timer.start(); }

double FpsCalculator::stop() {
  double cost_ms = m_timer.stop();
  m_queue.push_back(cost_ms);
  if (static_cast<int>(m_queue.size()) > m_queue_len) m_queue.pop_front();
  return cost_ms;
}

double FpsCalculator::fps() {
  double s = costms();
  return s > 0 ? 1000.0 / s : -1;
}

double FpsCalculator::costms() {
  if (m_queue.empty()) return -1;
  double s = 0;
  int c = 0;
  for (double a : m_queue) {
    s += a;
    c++;
  }
  return s / c;
}

double FpsCalculator::maxms() {
  if (m_queue.empty()) return -1;
  double s = 0;
  for (double a : m_queue) {
    if (a > s) s = a;
  }
  return s;
}

} /* namespace vs */
