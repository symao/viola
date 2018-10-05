/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-04-22 12:07
 * @details
 */
#include "vs_logger.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <thread>

#ifdef WIN32
#include <windows.h>
#elif defined(__linux__)
#include <sys/time.h>
#endif

#define VS_LOGGER_LOG_IMPL()                                       \
  do {                                                             \
    if (log_level_ >= cur_loglevel_) {                             \
      int i = header();                                            \
      va_list args;                                                \
      va_start(args, format);                                      \
      vsnprintf(str_buf_ + i, sizeof(str_buf_) - i, format, args); \
      va_end(args);                                                \
      outputLog(str_buf_);                                         \
    }                                                              \
    cur_loglevel_ = NONE;                                          \
  } while (0)

namespace vs {
Logger::Logger(int level, const std::string& tag, const std::string& log_file, bool enable_print, int ts_type)
    : enable_print_(enable_print), log_level_(level), cur_loglevel_(NONE), ts_type_(ts_type), tag_(tag) {
  setLogFile(log_file);
}

void Logger::setLogFile(const std::string& log_file) {
  if (log_file.length() <= 0) return;
  if (fp_) fclose(fp_);
  fp_ = fopen(log_file.c_str(), "w");
  if (fp_) log_file_ = log_file;
}

void Logger::error(const char* format, ...) {
  while (cur_loglevel_ != NONE)
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(1e3)));  // multi-thread safe
  cur_loglevel_ = ERROR;
  VS_LOGGER_LOG_IMPL();
}

void Logger::warn(const char* format, ...) {
  while (cur_loglevel_ != NONE)
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(1e3)));  // multi-thread safe
  cur_loglevel_ = WARN;
  VS_LOGGER_LOG_IMPL();
}

void Logger::info(const char* format, ...) {
  while (cur_loglevel_ != NONE)
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(1e3)));  // multi-thread safe
  cur_loglevel_ = INFO;
  VS_LOGGER_LOG_IMPL();
}

void Logger::debug(const char* format, ...) {
  while (cur_loglevel_ != NONE)
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(1e3)));  // multi-thread safe
  cur_loglevel_ = DEBUG;
  VS_LOGGER_LOG_IMPL();
}

int Logger::header() {
  int i = 0;
  if (ts_type_ == TS_SEC) {
    struct timeval tv;
    double ts = (gettimeofday(&tv, NULL) == 0) ? (tv.tv_sec + static_cast<double>(tv.tv_usec) * 1e-6) : 0;
    i += snprintf(str_buf_ + i, sizeof(str_buf_) - i, "%f", ts);
  } else if (ts_type_ == TS_DATE) {
    auto t = std::chrono::system_clock::now();
    time_t tm_cur = std::chrono::system_clock::to_time_t(t);
    struct tm tm_now;
#ifdef WIN32
    localtime_s(&tm_now, &tm_cur);
#else
    localtime_r(&tm_cur, &tm_now);
#endif  // WIN32
    int ms = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count() % 1000;
    i += snprintf(str_buf_ + i, sizeof(str_buf_) - i, "%02d-%02d-%02d_%02d:%02d:%02d.%03d", tm_now.tm_year + 1900,
                  tm_now.tm_mon + 1, tm_now.tm_mday, tm_now.tm_hour, tm_now.tm_min, tm_now.tm_sec, ms);
  }
  i += snprintf(str_buf_ + i, sizeof(str_buf_) - i, "[%s]%s: ", level_str_[cur_loglevel_], tag_.c_str());
  return i;
}

void Logger::outputLog(const char* n) {
  if (enable_print_) printf("%s\n", n);
  if (fp_) {
    fprintf(fp_, "%s\n", n);
    fflush(fp_);
  }
  for (auto& foo : user_output_callbacks_) foo(n);
}

}  // namespace vs