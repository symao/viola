/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details 2D/3D grid mapping using rolling buffer, local map is used for obstacle avoidance
 */
#pragma once

#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <functional>

namespace vs {

/** @brief 2D/3D grid mapping using rolling buffer, local map is used for obstacle avoidance
    ValType: grid map value type
    D: dimension, only 2 or 3 support
*/
template <class ValType, int D>
class GridMapping {
 public:
  /** @brief grid cell check result */
  enum CheckRes {
    UNKNOWN = 0,   ///< cell is unknown
    FREE = 1,      ///< cell is free
    OCCUPIED = 2,  ///< cell is occupied
  };

  /** @brief Constructor
   * @param[in] len_exp exponential of mapping size, size = (2^exp) = (1<<exp).
   *                    only support map size in 2^n, which is faster for rolling buffer
   * @param[in] res resolution in meters, which is the physical size of each grid cell
   * @param[in] val_max max value for each grid cell
   * @param[in] val_inc increment for each grid cell, once observing ray endpoint hit cell, cell value add this value
   * @param[in] val_dec decrement for each grid cell, once observing ray pass cell, cell value subscribe this value
   * @param[in] val_thres if cell value larger than this threshold, cell will be seen as occupied, otherwise free
   */
  GridMapping(int len_exp = 8, float res = 1.0f, ValType val_max = 1, ValType val_inc = 1, ValType val_dec = 1,
              ValType val_thres = 1);

  /** @brief Destroy the Grid Mapping object */
  ~GridMapping();

  /** @brief update map with a observation frame as well as observing position
   * @param[in] origin observing position
   * @param[in] pts ray endpoints in world(not body) coordination system
   * @param[in] N ray endpoint count
   */
  void update(const float origin[D], const float pts[][D], int N);

  /** @brief check a point status in grid map
   * @param[in] pos position to be checked [meter]
   * @return int one of @see CheckRes
   */
  int checkPoint(const float pos[D]);

  /** @brief check a circle region in grid map
   * @param[in] p center of circle region [meter]
   * @param[in] r radius of circle region [meter]
   * @return int one of @see CheckRes
   */
  int checkPoint(const float p[D], float r);

  /** @brief check a line in grid map
   * @param[in] p1 first end-point of line [meter]
   * @param[in] p2 second end-point of line [meter]
   * @return int one of @see CheckRes
   */
  int checkLine(const float p1[D], const float p2[D]);

  /** @brief check a line in grid map with radius
   * @param[in] p1 first end-point of line [meter]
   * @param[in] p2 second end-point of line [meter]
   * @param[in] r radius of line region [meter]
   * @return int one of @see CheckRes
   */
  int checkLine(const float p1[D], const float p2[D], float r);

  /** @brief clear grid, which set all cell values to 0 */
  void clear();

  /** @brief get map dimension */
  int dim() const { return D; }

  /** @brief get map resolution (cell size) */
  float resolution() const { return m_res; }

  /** @brief get origin coordinate of grid map, which is the top-left or x_min,y_min */
  float* origin() { return m_origin; }

  /** @brief get origin point index in grid map */
  int* originIdx() { return m_origin_idx; }

  /** @brief cell amount of grid map, which is (2^E)^D = 2^(E*D) */
  int len() const { return m_len; }

  /** @brief memory size of grid map */
  int size() const { return m_size; }

  /** @brief get map buffer ptr */
  ValType* map() { return m_map; }

  /** @brief whether cell is occupied */
  bool occupyVal(ValType v) const { return v >= m_thres; }

  /** @brief convert D-dimension index in grid map to index in 1D buffer */
  int index(int idx[D]) const {
    int res = idx[0];
    for (int i = 1; i < D; i++) {
      res = (res << E) | idx[i];
    }
    return res;
  }

  /** @brief convert position coordinate to D-dimension index in grid map */
  void pos2idx(const float pos[D], int idx[D]) const {
    for (int i = 0; i < D; i++) {
      idx[i] = p2i(pos[i]);
    }
  }

  /** @brief convert D-dimension index in grid map to position coordinate */
  void idx2pos(const int idx[D], float pos[D]) const {
    for (int i = 0; i < D; i++) {
      int k = idx[i] - m_origin_idx[i];
      while (k > UPPER) k -= K;
      while (k < LOWER) k += K;
      pos[i] = (k + 0.5) * m_res + m_origin_base[i];
    }
  }

  /** @brief fetch cell value with grid index */
  ValType& at(const int idx[D]) {
    int in[D];
    for (int i = 0; i < D; i++) in[i] = idx[i] & M;
    return m_map[index(in)];
  }

  /** @brief fetch cell value with point coordinate system */
  ValType& pos(const float p[D]) {
    int idx[D];
    pos2idx(p, idx);
    return at(idx);
  }

  /** @brief whether grid index inside map */
  bool inMap(const int idx[D]) const {
    for (int i = 0; i < D; i++) {
      int delta = idx[i] - m_origin_idx[i];
      if (delta > UPPER || delta < LOWER) return false;
    }
    return true;
  }

  /** @brief whether position coordinate inside map */
  bool inMap(const float pos[D]) const {
    int idx[D];
    pos2idx(pos, idx);
    return inMap(idx);
  }

  /** @brief check cell status in grid map with grid index */
  int check(const int idx[D]) {
    if (!inMap(idx))
      return UNKNOWN;
    else
      return occupyVal(at(idx)) ? OCCUPIED : FREE;
  }

  const int E, K, M;  ///< exponential, dim size, dim mask

 private:
  const int UPPER, LOWER;
  const ValType m_max, m_inc, m_dec, m_thres;
  const int m_len, m_size, m_pos_size, m_idx_size;
  int m_strides[D];        ///< stride of each dim
  ValType* m_map;          ///< gridmap rolling buffer
  float m_origin[D];       ///< coordinate of origin point, which is (x_min, y_min)
  float m_res;             ///< resolution in meters of grid cell
  int m_origin_idx[D];     ///< origin index in grid map
  float m_origin_base[D];  ///< base = int(origin/res) * res

  int m_check_res;
  std::function<bool(int[D])> m_f_update_free;            ///< update free cells
  std::function<bool(int[D])> m_f_check;                  ///< ray casting check funtion
  std::function<bool(int, int, int)> m_f_update_free_3d;  ///< update free cells for 3D grid map
  std::function<bool(int, int, int)> m_f_check_3d;        ///< ray casting check funtion for 3D grid map
  std::function<bool(int, int)> m_f_update_free_2d;       ///< update free cells for 2D grid map
  std::function<bool(int, int)> m_f_check_2d;             ///< ray casting check funtion for 2D grid map

  int p2i(float p) const { return int(std::floor(p / m_res)); }

  void updateRay(const float p[D]);

  void moveMap(const float new_origin[3]);

  void clearRow(int r, int dim);

  int checkLineTravel(const float p1[D], const float p2[D], const float step[D], int n);
};

typedef GridMapping<uint8_t, 2> GridMapping2D;
typedef GridMapping<uint8_t, 3> GridMapping3D;

} /* namespace vs */