/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_raycast.h"
#include <cmath>

namespace vs {

void raycast2D(const int start[2], const int end[2], std::function<bool(int, int)> op) {
  int dx = abs(end[0] - start[0]);
  int dy = abs(end[1] - start[1]);
  int ux = end[0] > start[0] ? 1 : -1;
  int uy = end[1] > start[1] ? 1 : -1;
  int dx2 = dx << 1;
  int dy2 = dy << 1;
  int x = start[0];
  int y = start[1];
  if (!op(x, y)) return;
  if (dx > dy) {  // Driving axis is X-axis"
    int e = -dx;
    while (x != end[0]) {
      x += ux;
      e += dy2;
      if (e >= 0) {
        y += uy;
        e -= dx2;
      }
      if (!op(x, y)) return;
    }
  } else {  // inc y
    int e = -dy;
    int x = start[0];
    while (y != end[1]) {
      y += uy;
      e += dx2;
      if (e >= 0) {
        x += ux;
        e -= dy2;
      }
      if (!op(x, y)) return;
    }
  }
}

void raycast3D(const int start[3], const int end[3], std::function<bool(int, int, int)> op) {
  int dx = abs(end[0] - start[0]);
  int dy = abs(end[1] - start[1]);
  int dz = abs(end[2] - start[2]);
  int ux = end[0] > start[0] ? 1 : -1;
  int uy = end[1] > start[1] ? 1 : -1;
  int uz = end[2] > start[2] ? 1 : -1;
  int dx2 = dx << 1;
  int dy2 = dy << 1;
  int dz2 = dz << 1;
  int x = start[0];
  int y = start[1];
  int z = start[2];
  if (!op(x, y, z)) return;
  if (dx >= dy && dx >= dz) {  // Driving axis is X-axis"
    int p1 = dy2 - dx;
    int p2 = dz2 - dx;
    while (x != end[0]) {
      x += ux;
      if (p1 >= 0) {
        y += uy;
        p1 -= dx2;
      }
      if (p2 >= 0) {
        z += uz;
        p2 -= dx2;
      }
      p1 += dy2;
      p2 += dz2;
      if (!op(x, y, z)) return;
    }
  } else if (dy >= dx && dy >= dz) {  // Driving axis is Y-axis"
    int p1 = dx2 - dy;
    int p2 = dz2 - dy;
    while (y <= end[1]) {
      y += uy;
      if (p1 >= 0) {
        x += ux;
        p1 -= dy2;
      }
      if (p2 >= 0) {
        z += uz;
        p2 -= dy2;
      }
      p1 += dx2;
      p2 += dz2;
      if (!op(x, y, z)) return;
    }
  } else {  // Driving axis is Z-axis"
    int p1 = dx2 - dz;
    int p2 = dy2 - dz;
    while (z != end[2]) {
      z += uz;
      if (p1 >= 0) {
        x += ux;
        p1 -= dz2;
      }
      if (p2 >= 0) {
        y += uy;
        p2 -= dz2;
      }
      p1 += dx2;
      p2 += dy2;
      if (!op(x, y, z)) return;
    }
  }
}

} /* namespace vs */