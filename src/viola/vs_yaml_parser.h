/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details param reader and writer of file in Yaml format, based on opencv.
 */
#pragma once
#include <opencv2/core.hpp>
#include <string>

namespace vs {

class YamlParser {
 public:
  YamlParser(const char* yaml_file = NULL, const std::string& mode = "r") {
    if (yaml_file) open(yaml_file, mode);
  }

  bool open(const char* yaml_file, const std::string& mode) {
    if (mode == "r") {
      m_fs.open(yaml_file, cv::FileStorage::READ);
    } else if (mode == "w") {
      m_fs.open(yaml_file, cv::FileStorage::WRITE);
    } else {
      printf("[WARN]YamlParser: unknow mode '%s', use 'r' or 'w' for read and write\n", mode.c_str());
      return false;
    }
    m_param_file = std::string(yaml_file);
    if (!m_fs.isOpened()) printf("[WARN]YamlParser: cannot open param file '%s'\n", yaml_file);
    return m_fs.isOpened();
  }

  bool isOpened() const { return m_fs.isOpened(); }

  cv::FileNode findNode(const char* param_name) {
    std::stringstream ss(param_name);
    cv::FileNode node;
    std::string t;
    bool first_node = true;
    while (getline(ss, t, '/')) {
      if (t.length() == 0) continue;
      if (first_node) {
        node = m_fs[t];
        first_node = false;
      } else {
        node = node[t];
      }
      if (node.isNone()) break;
    }
    return node;
  }

  template <typename T>
  T read(const char* param_name, T default_val = T()) {
    if (!m_fs.isOpened()) {
      printf("[WARN]YamlParser: read faild, no yaml file open.\n");
      return default_val;
    }
    auto node = findNode(param_name);
    if (node.isNone()) {
      printf("[WARN]YamlParser: param '%s' not found, use default val.\n", param_name);
      return default_val;
    } else {
      T a;
      node >> a;
      return a;
    }
  }

  template <typename T>
  bool readInPlace(const char* param_name, T& out) {
    if (!m_fs.isOpened()) {
      printf("[WARN]YamlParser: read faild, no yaml file open.\n");
      return false;
    }
    auto node = findNode(param_name);
    if (node.isNone()) {
      printf("[WARN]YamlParser: param '%s' not found, use default val.\n", param_name);
      return false;
    } else {
      node >> out;
      return true;
    }
  }

  template <typename T>
  void write(const char* param_name, T val) {
    if (!m_fs.isOpened()) {
      printf("[WARN]YamlParser: write faild, no yaml file open.\n");
      return;
    }
    m_fs << param_name << val;
  }

  cv::FileStorage& fs() { return m_fs; }

 private:
  cv::FileStorage m_fs;
  std::string m_param_file;
};

} /* namespace vs */