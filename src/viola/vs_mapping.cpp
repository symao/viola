/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_mapping.h"

#include "vs_raycast.h"

namespace vs {

template <class ValType, int D>
GridMapping<ValType, D>::GridMapping(int len_exp, float res, ValType val_max, ValType val_inc, ValType val_dec,
                                     ValType val_thres)
    : E(len_exp),
      K(1 << E),
      M(K - 1),
      UPPER(K >> 1),
      LOWER(-(UPPER - 1)),
      m_max(val_max),
      m_inc(val_inc),
      m_dec(val_dec),
      m_thres(std::min(val_thres, val_max)),
      m_len(1 << (E * D)),
      m_size(sizeof(ValType) * m_len),
      m_pos_size(sizeof(float) * D),
      m_idx_size(sizeof(int) * D),
      m_strides{0},
      m_map(new ValType[m_len]),
      m_origin{0},
      m_res(res),
      m_origin_idx{0},
      m_origin_base{0} {
  // calc dim size
  m_strides[D - 1] = sizeof(ValType);
  for (int i = D - 2; i >= 0; i--) {
    m_strides[i] = m_strides[i + 1] * K;
  }
  // clear map
  clear();

  // create function
  m_f_update_free = [this](int idx[D]) {
    if (!inMap(idx)) return false;  // break raycast
    ValType& v = at(idx);
    v = (v > m_dec) ? (v - m_dec) : 0;
    return true;
  };

  m_f_check = [this](int idx[D]) {
    if (!inMap(idx)) {
      m_check_res = UNKNOWN;
      return false;  // break raycast
    } else if (at(idx) >= m_thres) {
      m_check_res = OCCUPIED;
      return false;  // break raycast
    } else {
      m_check_res = FREE;
      return true;
    }
  };

  if (D == 2) {
    m_f_update_free_2d = [this](int ix, int iy) {
      int idx[2] = {ix, iy};
      return m_f_update_free(idx);
    };

    m_f_check_2d = [this](int ix, int iy) {
      int idx[2] = {ix, iy};
      return m_f_check(idx);
    };
  } else if (D == 3) {
    m_f_update_free_3d = [this](int ix, int iy, int iz) {
      int idx[3] = {ix, iy, iz};
      return m_f_update_free(idx);
    };

    m_f_check_3d = [this](int ix, int iy, int iz) {
      int idx[3] = {ix, iy, iz};
      return m_f_check(idx);
    };
  }
}

template <class ValType, int D>
GridMapping<ValType, D>::~GridMapping() {
  delete[] m_map;
}

template <class ValType, int D>
void GridMapping<ValType, D>::clear() {
  memset(m_map, 0, m_size);
}

template <class ValType, int D>
void GridMapping<ValType, D>::update(const float origin[D], const float pts[][D], int N) {
  // move map
  moveMap(origin);
  // update each ray
  for (int i = 0; i < N; i++) updateRay(pts[i]);
}

template <class ValType, int D>
int GridMapping<ValType, D>::checkPoint(const float pos[D]) {
  int idx[D];
  pos2idx(pos, idx);
  return check(idx);
}

template <class ValType, int D>
int GridMapping<ValType, D>::checkPoint(const float p[D], float r) {
  int res = checkPoint(p);
  if (res == OCCUPIED || r <= 0) return res;
  for (int d = 0, tres; d < D; d++) {
    float tp[D];
    memcpy(tp, p, m_pos_size);

    tp[d] = p[d] - r;
    tres = checkPoint(tp);
    if (tres == OCCUPIED) {
      return tres;
    } else if (tres == FREE) {
      res = tres;
    }

    tp[d] = p[d] + r;
    tres = checkPoint(tp);
    if (tres == OCCUPIED) {
      return tres;
    } else if (tres == FREE) {
      res = tres;
    }
  }
  return res;
}

template <class ValType, int D>
int GridMapping<ValType, D>::checkLine(const float p1[D], const float p2[D]) {
  int src_idx[D], tar_idx[D];
  pos2idx(p1, src_idx);
  pos2idx(p2, tar_idx);

  // handle that index out of boundry
  // make sure that src_idx is in map,
  // raycast will break loop once out of map
  bool in1 = inMap(src_idx);
  bool in2 = inMap(tar_idx);
  if (!in1 && !in2)
    return UNKNOWN;
  else if (!in1) {
    std::swap(src_idx, tar_idx);
  }
  m_check_res = UNKNOWN;
  switch (D) {
    case 2:
      raycast2D(src_idx, tar_idx, m_f_check_2d);
      break;
    case 3:
      raycast3D(src_idx, tar_idx, m_f_check_3d);
      break;
    default:
      printf("[ERROR]Unsupport dim:%d\n", D);
      break;
  }
  return m_check_res;
}

template <class ValType, int D>
int GridMapping<ValType, D>::checkLine(const float p1[D], const float p2[D], float r) {
  int res = checkLine(p1, p2);
  if (res == OCCUPIED) return res;

  int n = int(r / m_res);
  if (n <= 0) return res;

  if (D == 2) {
    float dx = p2[1] - p1[1];
    float dy = p1[0] - p2[0];
    float dist = hypotf(dx, dy);
    if (dist > 0.001f) {
      float k = m_res / dist;
      float step[2] = {dx * k, dy * k};
      int tres = checkLineTravel(p1, p2, step, n);
      if (tres != UNKNOWN) res = tres;
    }
  } else if (D == 3) {
    float dx = p2[0] - p1[0];
    float dy = p2[1] - p1[1];
    float dz = p2[2] - p1[2];
    float step_a[3] = {0};
    float step_b[3] = {0};
    if (dx != 0 || dy != 0) {
      float ss = dx * dx + dy * dy;
      float k = m_res / std::sqrt(ss);
      step_a[0] = -dy * k;
      step_a[1] = dx * k;
      step_a[2] = 0;
      step_b[0] = -dx * dz;
      step_b[1] = dy * dz;
      step_b[2] = ss;
      k = m_res / std::sqrt(step_b[0] * step_b[0] + step_b[1] * step_b[1] + step_b[2] * step_b[2]);
      step_b[0] *= k;
      step_b[1] *= k;
      step_b[2] *= k;
    } else if (dz != 0) {
      float ss = dy * dy + dz * dz;
      float k = m_res / std::sqrt(ss);
      step_a[0] = 0;
      step_a[1] = -dz * k;
      step_a[2] = dy * k;
      step_b[0] = ss;
      step_b[1] = dx * dy;
      step_b[2] = -dx * dz;
      k = m_res / std::sqrt(step_b[0] * step_b[0] + step_b[1] * step_b[1] + step_b[2] * step_b[2]);
      step_b[0] *= k;
      step_b[1] *= k;
      step_b[2] *= k;
    } else {
      return res;
    }
    int tres = checkLineTravel(p1, p2, step_a, n);
    if (tres != UNKNOWN) res = tres;
    if (res == OCCUPIED) return res;
    tres = checkLineTravel(p1, p2, step_b, n);
    if (tres != UNKNOWN) res = tres;
  }

  return res;
}

template <class ValType, int D>
int GridMapping<ValType, D>::checkLineTravel(const float p1[D], const float p2[D], const float step[D], int n) {
  int res = UNKNOWN, tres;
  float pp1[D], pp2[D], dd[D];
  memcpy(dd, step, m_pos_size);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < D; j++) {
      pp1[j] = p1[j] + dd[j];
      pp2[j] = p2[j] + dd[j];
    }
    tres = checkLine(pp1, pp2);
    if (tres == OCCUPIED)
      return tres;
    else if (tres == FREE)
      res = tres;
    for (int j = 0; j < D; j++) {
      pp1[j] = p1[j] - dd[j];
      pp2[j] = p2[j] - dd[j];
    }
    tres = checkLine(pp1, pp2);
    if (tres == OCCUPIED)
      return tres;
    else if (tres == FREE)
      res = tres;
    for (int j = 0; j < D; j++) {
      dd[j] += step[j];
    }
  }
  return res;
}

template <class ValType, int D>
void GridMapping<ValType, D>::updateRay(const float p[D]) {
  int tar_idx[D];
  pos2idx(p, tar_idx);
  // update free
  switch (D) {
    case 2:
      raycast2D(m_origin_idx, tar_idx, m_f_update_free_2d);
      break;
    case 3:
      raycast3D(m_origin_idx, tar_idx, m_f_update_free_3d);
      break;
    default:
      printf("[ERROR]Unsupport dim:%d\n", D);
      break;
  }
  // update OCCUPIED
  if (inMap(tar_idx)) {
    ValType& v = at(tar_idx);
    if (v < m_max - m_inc)
      v += m_inc;
    else
      v = m_max;
  }
}

template <class ValType, int D>
void GridMapping<ValType, D>::moveMap(const float new_origin[3]) {
  // clear old region which out of new boundry
  int new_idx[D];
  pos2idx(new_origin, new_idx);
  for (int d = 0; d < D; d++) {
    int u1 = m_origin_idx[d] + UPPER;
    int u2 = new_idx[d] + UPPER;
    if (u1 > u2) std::swap(u1, u2);
    for (int r = u1 + 1; r <= u2; r++) clearRow(r, d);
  }
  // update new origin
  memcpy(m_origin, new_origin, m_idx_size);
  memcpy(m_origin_idx, new_idx, m_idx_size);
  for (int i = 0; i < D; i++) m_origin_base[i] = int(std::floor(m_origin[i] / m_res)) * m_res;
}

template <class ValType, int D>
void GridMapping<ValType, D>::clearRow(int r, int dim) {
  if (dim >= D) return;

  r = r & M;
  switch (dim) {
    case 0: {
      int idx[D] = {r};
      memset(m_map + index(idx), 0, m_strides[dim]);
    } break;
    case 1:
      for (int i = 0; i < K; i++) {
        int idx[D] = {i, r};
        memset(m_map + index(idx), 0, m_strides[dim]);
      }
      break;
    case 2:
      for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++) {
          int idx[3] = {i, j, r};
          memset(m_map + index(idx), 0, m_strides[dim]);
        }
      break;
    default:
      break;
  }
}

template class GridMapping<uint8_t, 2>;
template class GridMapping<uint8_t, 3>;

} /* namespace vs */