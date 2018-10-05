/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_rot.h"

#include <math.h>

namespace vs {

void quat2rot(const float* q, float* r) {
  float w = q[0];
  float x = q[1];
  float y = q[2];
  float z = q[3];
  r[0] = 1 - 2 * (y * y + z * z);
  r[1] = 2 * (x * y + z * w);
  r[2] = 2 * (x * z - y * w);
  r[3] = 2 * (x * y - z * w);
  r[4] = 1 - 2 * (x * x + z * z);
  r[5] = 2 * (y * z + x * w);
  r[6] = 2 * (x * z + y * w);
  r[7] = 2 * (y * z - x * w);
  r[8] = 1 - 2 * (x * x + y * y);
}

void rot2quat(const float* r, float* q) {
  double tr = r[0] + r[4] + r[8];
  double n4;
  if (tr > 0.0f) {
    q[0] = tr + 1.0f;
    q[1] = r[5] - r[7];
    q[2] = r[6] - r[2];
    q[3] = r[1] - r[3];
    n4 = q[0];
  } else if ((r[0] > r[4]) && (r[0] > r[8])) {
    q[0] = r[5] - r[7];
    q[1] = 1.0f + r[0] - r[4] - r[8];
    q[2] = r[3] + r[1];
    q[3] = r[6] + r[2];
    n4 = q[1];
  } else if (r[4] > r[8]) {
    q[0] = r[6] - r[2];
    q[1] = r[3] + r[1];
    q[2] = 1.0f + r[4] - r[0] - r[8];
    q[3] = r[7] + r[5];
    n4 = q[2];
  } else {
    q[0] = r[1] - r[3];
    q[1] = r[6] + r[2];
    q[2] = r[7] + r[5];
    q[3] = 1.0f + r[8] - r[0] - r[4];
    n4 = q[3];
  }
  double scale = 0.5f / double(sqrt(n4));
  q[0] *= scale;
  q[1] *= scale;
  q[2] *= scale;
  q[3] *= scale;
}

void quat2euler(const float* q, float* e) {
  float qw = q[0];
  float qx = q[1];
  float qy = q[2];
  float qz = q[3];
  double ysqr = qy * qy;

  // yaw (z-axis rotation)
  e[0] = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (ysqr + qz * qz));

  // pitch (y-axis rotation)
  double t2 = +2.0 * (qw * qy - qz * qx);
  t2 = ((t2 > 1.0) ? 1.0 : t2);
  t2 = ((t2 < -1.0) ? -1.0 : t2);
  e[1] = asin(t2);

  // roll (x-axis rotation)
  e[2] = atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + ysqr));
}

void euler2rot(const float* e, float* r) {
  float sz = sin(e[0]);  // yaw
  float cz = cos(e[0]);
  float sy = sin(e[1]);  // pitch
  float cy = cos(e[1]);
  float sx = sin(e[2]);  // roll
  float cx = cos(e[2]);
  r[0] = cy * cz;
  r[1] = cy * sz;
  r[2] = -sy;
  r[3] = cz * sx * sy - cx * sz;
  r[4] = cx * cz + sx * sy * sz;
  r[5] = cy * sx;
  r[6] = sx * sz + cx * cz * sy;
  r[7] = cx * sy * sz - cz * sx;
  r[8] = cx * cy;
}

void euler2quat(const float* e, float* q) {
  double cz = cos(e[0] * 0.5);
  double sz = sin(e[0] * 0.5);
  double cy = cos(e[1] * 0.5);
  double sy = sin(e[1] * 0.5);
  double cx = cos(e[2] * 0.5);
  double sx = sin(e[2] * 0.5);

  q[0] = cz * cx * cy + sz * sx * sy;
  q[1] = cz * sx * cy - sz * cx * sy;
  q[2] = cz * cx * sy + sz * sx * cy;
  q[3] = sz * cx * cy - cz * sx * sy;
}

void rot2euler(const float* r, float* e) {
  e[0] = atan2(r[1], r[0]);
  e[1] = atan2(-r[2], sqrt(r[5] * r[5] + r[8] * r[8]));
  e[2] = atan2(r[5], r[8]);
}

} /* namespace vs */