/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details 2D/3D grid map buffer
 */
#pragma once
#include <iostream>
#include <vector>
#include "vs_basic.h"

namespace vs {

/** @brief 2D/3D grid map buffer
    ValType: grid map value type
    Dim: dimension
*/
template <class ValType, int Dim>
class GridMap {
 public:
  typedef std::vector<int> GridIndex;   ///< grid index list: (idx_x, idx_y, idx_z, ...)
  typedef std::vector<double> GridPos;  ///< grid position list: (x, y, z, ...)
  struct DimInfo {
    int size = 0;                                   ///< grid size of current dimension, deduced inside
    double pos_min = 0, pos_max = 0, pos_step = 0;  ///< min, max value grid step of current dimension

    explicit DimInfo(double min, double step, int dim_size)
        : size(dim_size), pos_min(min), pos_max(min + dim_size * step - VS_EPS), pos_step(step) {}

    explicit DimInfo(double min, double max, double step)
        : size((max - min) / step + 1), pos_min(min), pos_max(max), pos_step(step) {}

    /** @brief convert value to grid index in current dimension */
    int idx(double a) const { return (a - pos_min) / pos_step; }

    /** @brief convert grid index to center value in current dimension */
    double center(int idx) const { return pos_min + pos_step * (idx + 0.5); }

    /** @brief check whether value inside range of current dimension */
    bool in(double a) const { return pos_min <= a && a <= pos_max; }
  };

  GridMap() {}

  GridMap(const std::vector<int>& dims) { resize(dims); }

  GridMap(const std::vector<DimInfo>& dim_infos) { resize(dim_infos); }

  /** @brief resize grid map to input sizes of each dimension */
  void resize(const std::vector<int>& dims) {
    if (!checkDim(dims.size())) return;
    int len = vecProduct(dims);
    m_data = std::vector<ValType>(len, ValType());
    m_dims = dims;
    m_strides.resize(Dim);
    m_strides.back() = 1;
    for (int i = Dim - 2; i >= 0; i--) m_strides[i] = m_strides[i + 1] * dims[i + 1];
  }

  /** @brief resize grid map by input information of each dimension */
  void resize(const std::vector<DimInfo>& dim_infos) {
    if (!checkDim(dim_infos.size())) return;
    std::vector<int> dims;
    dims.reserve(dim_infos.size());
    for (const auto& info : dim_infos) dims.push_back(info.size);
    resize(dims);
    m_dim_infos = dim_infos;
  }

  /** @brief whether map is empty */
  bool empty() const { return m_data.empty(); }

  /** @brief Get whether input ids inside grid map */
  bool inside(const GridIndex& ids) const {
    if (!checkDim(ids.size())) return false;
    for (int i = 0; i < Dim; i++) {
      if (ids[i] < 0 || ids[i] >= m_dims[i]) return false;
    }
    return true;
  }

  /** @brief Get whether input ids inside grid map only support Dim <= 3 */
  bool inside(int idx0, int idx1 = 0, int idx2 = 0) const {
    std::vector<int> dims(Dim);
    if (Dim > 0) dims[0] = idx0;
    if (Dim > 1) dims[1] = idx1;
    if (Dim > 2) dims[2] = idx2;
    return inside(dims);
  }

  /** @brief Get whether input position inside grid map */
  bool inside(const GridPos& pos) const {
    if (!checkDim(pos.size())) return false;
    for (int i = 0; i < Dim; i++) {
      if (!m_dim_infos[i].in(pos[i])) return false;
    }
    return true;
  }

  /** @brief Get whether input position inside grid map, only support Dim <= 3 */
  bool inside(double x, double y = 0, double z = 0) const {
    std::vector<double> pos(Dim);
    if (Dim > 0) pos[0] = x;
    if (Dim > 1) pos[1] = y;
    if (Dim > 2) pos[2] = z;
    return inside(pos);
  }

  /** @brief Get the grid value at input ids */
  ValType& at(const GridIndex& ids) {
    int idx = 0;
    for (int i = 0; i < Dim; i++) idx += m_strides[i] * ids[i];
    return m_data.at(idx);
  }

  /** @brief Get the grid value at input grid index, only support Dim <= 3 */
  ValType& at(int idx0, int idx1 = 0, int idx2 = 0) {
    std::vector<int> dims(Dim);
    if (Dim > 0) dims[0] = idx0;
    if (Dim > 1) dims[1] = idx1;
    if (Dim > 2) dims[2] = idx2;
    return at(dims);
  }

  /** @brief Get the grid value at input position */
  ValType& at(const GridPos& pos) {
    int idx = 0;
    for (int i = 0; i < Dim; i++) idx += m_strides[i] * m_dim_infos[i].idx(pos[i]);
    return m_data.at(idx);
  }

  /** @brief Get the grid value at input position, only support Dim <= 3 */
  ValType& at(double x, double y = 0, double z = 0) {
    std::vector<double> pos(Dim);
    if (Dim > 0) pos[0] = x;
    if (Dim > 1) pos[1] = y;
    if (Dim > 2) pos[2] = z;
    return at(pos);
  }

  /** @brief get map dimensions */
  int dim() const { return Dim; }

  /** @brief get map dimensions size */
  std::vector<int> dims() const { return m_dims; }

  /** @brief get data buffer */
  std::vector<ValType>& data() { return m_data; }

  /** @brief Set all grid value to input value */
  void setTo(const ValType& a) {
    for (auto& v : m_data) v = a;
  }

  /** @brief All grid value in map add input value */
  GridMap& operator+=(const ValType& a) {
    for (auto& v : m_data) v += a;
    return (*this);
  }

  /** @brief All grid value in map substract input value */
  GridMap& operator-=(const ValType& a) {
    for (auto& v : m_data) v -= a;
    return (*this);
  }

  /** @brief All grid value in map multiply input value */
  GridMap& operator*=(const ValType& a) {
    for (auto& v : m_data) v *= a;
    return (*this);
  }

  /** @brief All grid value in map divide input value */
  GridMap& operator/=(const ValType& a) {
    for (auto& v : m_data) v /= a;
    return (*this);
  }

 private:
  std::vector<ValType> m_data;       ///< data buffer to store grid map values
  std::vector<int> m_dims;           ///< size of each dimensions
  std::vector<int> m_strides;        ///< stride of each dimensions
  std::vector<DimInfo> m_dim_infos;  ///< infomation of each dimension

  bool checkDim(int dim) const {
    if (dim != Dim) {
      printf("[ERROR]GridMap: input size(%d) not equal to map dimension(%d).\n", dim, Dim);
      return false;
    }
    return true;
  }
};

typedef GridMap<uint8_t, 2> GridMap2u;
typedef GridMap<int, 2> GridMap2i;
typedef GridMap<float, 2> GridMap2f;
typedef GridMap<double, 2> GridMap2d;
typedef GridMap<uint8_t, 3> GridMap3u;
typedef GridMap<int, 3> GridMap3i;
typedef GridMap<float, 3> GridMap3f;
typedef GridMap<double, 3> GridMap3d;

}  // namespace vs