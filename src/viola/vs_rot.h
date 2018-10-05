/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details conversion between quaternion(w, x, y, z), eular(yaw, pitch, roll) and rotation matrix
 */
#pragma once

namespace vs {

void quat2rot(const float* q, float* r);

void quat2euler(const float* q, float* e);

void euler2rot(const float* e, float* r);

void euler2quat(const float* e, float* q);

void rot2quat(const float* r, float* q);

void rot2euler(const float* r, float* e);

} /* namespace vs */