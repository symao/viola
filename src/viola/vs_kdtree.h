/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details implemention of KD-tree and nereast search.
 */
#pragma once
#include <memory>
#include <vector>

namespace vs {

class KDTreeImpl;
/** @brief K-dimension tree, find nearest data in K-dimension space */
class KDTree {
 public:
  typedef std::vector<float> Kvec;
  typedef std::vector<Kvec> KvecArray;

  KDTree();

  /** @brief build KD tree with input data
   * @param[in]data: NxK data, which means N point in K-dimension space
   * @return true if tree build success, else false
   */
  bool build(const KvecArray& data);

  /** @brief find nearest data to query in K-dimension space
   * @param[in]query: query data which length is K
   * @param[out]res: nearest datas in KD tree, NxK
   * @param[in]k: find the top-k nearest data to query point in KD tree
   * @param[in]r: find datas whose L2-distance to query point less than radius
   * @return int account of nearest data
   */
  int nearest(const Kvec& query, KvecArray& res, int k, float r = 0);

  /** @brief find the top-k nearest data to query point in KD tree
   * @param[in]query: query data which length is K
   * @param[out]res: nearest datas in KD tree, NxK
   * @param[in]k: find the top-k nearest data to query point in KD tree
   * @return int account of nearest data
   */
  int knn(const Kvec& query, KvecArray& res, int k) { return nearest(query, res, k, 0); }

  /** @brief find datas whose L2-distance to query point less than radius
   * @param[in]query: query data which length is K
   * @param[out]res: nearest datas in KD tree, NxK
   * @param[in]r: find datas whose L2-distance to query point less than radius
   * @return int account of nearest data
   */
  int rnn(const Kvec& query, KvecArray& res, float r) { return nearest(query, res, 0, r); }

 private:
  std::shared_ptr<KDTreeImpl> m_impl;
};

} /* namespace vs */
