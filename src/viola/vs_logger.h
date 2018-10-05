/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-04-22 12:07
 * @details a lightweight logger which output message to screen, file as well as user callbacks.
 */
#pragma once
#include <string>
#include <vector>
#include <functional>
// #include "vs_data.h"

namespace vs {

/** @brief logger to screen, file and string
 * format: "timestamp[ERROR|WARN|INFO|DEBUG]tag:print_string"
 * @code
 *  Logger logger(Logger::INFO, "VS_COMMON", "log.txt", true, Logger::TS_DATE);
 *  logger.error("this is error msg:%.4f %s", 1.222, "dsadsa");
 *  logger.debug("this is debug msg");
 *  logger.info("this is info msg:%d", 123);
 *  logger.warn("this is warn msg:%.4f %d", 1.23, 258);
 * @endcode
 */
class Logger {
 public:
  enum LogLevel { NONE = 0, ERROR = 1, WARN = 2, INFO = 3, DEBUG = 4 };

  enum TsType {
    TS_NONE = 0,  ///< not log timestamp
    TS_SEC = 1,   ///< log timstampe in seconds, such as: 1648874047.178820
    TS_DATE = 2   ///< log timstampe in data, such as: 2022-04-02_12:34:07
  };

  /** @brief callback to output log message
   * @code
   * OutputCallback sample_cb = [](const char* msg) { printf("%s\n", msg); };
   * @endcode
   */
  typedef std::function<void(const char*)> OutputCallback;

  /** @brief Constructor
   * @param[in]level: log level, @see LogLevel
   * @param[in]tag: tag name
   * @param[in]log_file: log file
   * @param[in]enable_print: whether print to screen
   * @param[in]ts_type: timestamp type, @see TsType
   */
  Logger(int level = INFO, const std::string& tag = "", const std::string& log_file = "", bool enable_print = true,
         int ts_type = TS_NONE);

  /** @brief print error log to screen or file */
  void error(const char* format, ...);

  /** @brief print warn log to screen or file */
  void warn(const char* format, ...);

  /** @brief print info log to screen or file */
  void info(const char* format, ...);

  /** @brief print debug log to screen or file */
  void debug(const char* format, ...);

  /** @brief add callback to output log message */
  void addOutputCallback(OutputCallback callback) { user_output_callbacks_.push_back(callback); }

  /** @brief clear all callbacks to output log message */
  void clearOutputCallbacks() { user_output_callbacks_.clear(); }

  /** @brief set logger level, @see LogLevel,  set NONE to close all log */
  void setLevel(int level) { log_level_ = level; }

  /** @brief get logger level, @see LogLevel */
  int getLevel() const { return log_level_; }

  /** @brief set tag name */
  void setTag(const std::string& tag) { tag_ = tag; }

  /** @brief get tag name in string */
  std::string getTag() const { return tag_; }

  /** @brief set tag name in char* */
  const char* getTagPtr() const { return tag_.c_str(); }

  /** @brief set log file, if set, log will be written into file */
  void setLogFile(const std::string& log_file);

  /** @brief set log file in char* */
  const char* getLogFilePtr() const { return log_file_.c_str(); }

  /** @brief get current log file handler, return NULL if no log file set */
  FILE* getFp() { return fp_; }

  /** @brief set whether print log to console */
  void setEnablePrint(bool enable) { enable_print_ = enable; }

  /** @brief get whether print log to console */
  bool getEnablePrint() const { return enable_print_; }

  /** @brief set timestamp print type. @see TsType */
  void setTsType(int ts_type) { ts_type_ = ts_type; }

  /** @brief get timestamp print type. @see TsType */
  int getTsType() const { return ts_type_; }

 private:
  bool enable_print_;          ///< whether print to screen
  int log_level_;              ///< log level, see LogLevel
  int cur_loglevel_;           ///< current call log level
  int ts_type_;                ///< timestamp type, see TsType
  std::string tag_;            ///< tag name
  std::string log_file_;       ///< log file
  FILE* fp_ = NULL;            ///< file handler if log to file
  char str_buf_[16384] = {0};  ///< inner buffer used to log to string
  std::vector<OutputCallback> user_output_callbacks_;
  const char* level_str_[5] = {"", "ERROR", "WARN", "INFO", "DEBUG"};

  /** @brief construct log header into str_buf_, return string length */
  int header();

  /** @brief output log string from str_buf_ to screen, file, as well as output callbacks */
  void outputLog(const char* n);
};

}  // namespace vs

#define VS_LOG_INST vs::Singleton<vs::Logger>::instance()
#define VS_LOG_ERROR(...) VS_LOG_INST->error(__VA_ARGS__)
#define VS_LOG_WARN(...) VS_LOG_INST->warn(__VA_ARGS__)
#define VS_LOG_INFO(...) VS_LOG_INST->info(__VA_ARGS__)
#define VS_LOG_DEBUG(...) VS_LOG_INST->debug(__VA_ARGS__)