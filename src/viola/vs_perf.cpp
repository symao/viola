/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_perf.h"

#include <map>
#include <string>

namespace vs {

static std::map<std::string, PerfCounter> p_perf_counter;
static bool p_perf_enable = false;

void perfEnable(bool enable) { p_perf_enable = enable; }

void perfBegin(const char* name) {
  if (p_perf_enable) p_perf_counter[std::string(name)].start();
}

void perfEnd(const char* name) {
  if (p_perf_enable) p_perf_counter[std::string(name)].stop();
}

void perfPrint(const char* name, bool verbose) {
  if (p_perf_enable) {
    if (name) {
      auto it = p_perf_counter.find(std::string(name));
      if (it == p_perf_counter.end()) {
        printf("[ERROR] perf_avg(\"%s\"): no such performance.\n", name);
        return;
      }
      it->second.print(name, verbose);
    } else {
      for (auto it = p_perf_counter.begin(); it != p_perf_counter.end(); ++it) {
        it->second.print(it->first.c_str(), verbose);
      }
    }
  }
}

float perfAvg(const char* name) {
  if (!p_perf_enable) return 0;

  if (name == NULL) {
    printf("[ERROR] perf_avg: null name.\n");
    return -1;
  }
  auto it = p_perf_counter.find(std::string(name));
  if (it == p_perf_counter.end()) {
    printf("[ERROR] perf_avg(\"%s\"): no such performance.\n", name);
    return -1;
  }
  return it->second.mean();
}

} /* namespace vs */