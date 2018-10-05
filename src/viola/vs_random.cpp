/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_random.h"

namespace vs {

static std::mt19937 mt;

static int vsRand() {
#if 1
  return rand();
#else
  static unsigned int seed = 0;
  return rand_r(&seed);
#endif
}

int randi(int high) { return 0 < high ? vsRand() % high : 0; }

int randi(int low, int high) { return low < high ? low + vsRand() % (high - low) : 0; }

double randf(double high) {
  std::uniform_real_distribution<float> dis(0.0, high);
  return dis(mt);
}

double randf(double low, double high) {
  std::uniform_real_distribution<float> dis(low, high);
  return dis(mt);
}

double randn(double mu, double sigma) {
  std::normal_distribution<double> dis(mu, sigma);
  return dis(mt);
}

std::vector<int> randIntVec(int low, int high, int n) {
  std::vector<int> res(n);
  for (int i = 0; i < n; i++) res[i] = low < high ? low + vsRand() % (high - low) : 0;
  return res;
}

std::vector<float> randFloatVec(float low, float high, int n) {
  std::uniform_real_distribution<float> dis(low, high);
  std::vector<float> res(n);
  for (int i = 0; i < n; i++) res[i] = dis(mt);
  return res;
}

std::vector<double> randDoubleVec(double low, double high, int n) {
  std::uniform_real_distribution<double> dis(low, high);
  std::vector<double> res(n);
  for (int i = 0; i < n; i++) res[i] = dis(mt);
  return res;
}

std::vector<double> randnVec(double mu, double sigma, int n) {
  std::normal_distribution<double> dis(mu, sigma);
  std::vector<double> res(n);
  for (int i = 0; i < n; i++) res[i] = dis(mt);
  return res;
}

} /* namespace vs */