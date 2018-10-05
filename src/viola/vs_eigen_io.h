/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-05-04 14:54
 * @details file I/O for Eigen dense/sparse matrix.
 */
#pragma once
#include <fstream>
#include <typeinfo>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace vs {

/** @brief write Eigen dense matrix into ofstream in inner format.
 * @param[in/out]f: ostream where matrix data will be written to.
 * @param[in]m: Eigen dense matrix
 * @return where writting ok.
 */
template <typename T, int _Rows, int _Cols>
bool writeEigenDense(std::ofstream& f, const Eigen::Matrix<T, _Rows, _Cols>& m) {
  uint32_t rows = m.rows();
  uint32_t cols = m.cols();
  uint32_t id = 0;
  const auto& t = typeid(T);
  if (t == typeid(int)) {
    id = 1;
  } else if (t == typeid(float)) {
    id = 2;
  } else if (t == typeid(double)) {
    id = 3;
  }
  if (id == 0) return false;
  f.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  f.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
  f.write(reinterpret_cast<const char*>(&id), sizeof(id));
  for (uint32_t i = 0; i < rows; i++)
    for (uint32_t j = 0; j < cols; j++) {
      const auto& v = m(i, j);
      f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }
  f.flush();
  return true;
}

/** @brief write Eigen dense matrix into file in inner format.
 * @param[in]binfile: file path where matrix data will be written to.
 * @param[in]m: Eigen dense matrix
 * @return where writting ok.
 */
template <typename T, int _Rows, int _Cols>
bool writeEigenDense(const char* binfile, const Eigen::Matrix<T, _Rows, _Cols>& m) {
  std::ofstream fout(binfile, std::ios::binary);
  if (!fout.is_open()) return false;
  return writeEigenDense(fout, m);
}

/** @brief read Eigen dense matrix from ifstream.
 * @param[in]f: ifstream of file which is written by writeEigenDense().
 * @return Eigen dense matrix, return empty mat if read failed.
 */
template <typename T>
Eigen::Matrix<T, -1, -1> readEigenDense(std::ifstream& f) {
  uint32_t rows, cols, id;
  f.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  f.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  f.read(reinterpret_cast<char*>(&id), sizeof(id));
  Eigen::Matrix<T, -1, -1> m(rows, cols);
  if (id == 1) {
    for (uint32_t i = 0; i < rows; i++)
      for (uint32_t j = 0; j < cols; j++) {
        int v;
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        m(i, j) = v;
      }
  } else if (id == 2) {
    for (uint32_t i = 0; i < rows; i++)
      for (uint32_t j = 0; j < cols; j++) {
        float v;
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        m(i, j) = v;
      }
  } else if (id == 3) {
    for (uint32_t i = 0; i < rows; i++)
      for (uint32_t j = 0; j < cols; j++) {
        double v;
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        m(i, j) = v;
      }
  } else {
    return Eigen::Matrix<T, -1, -1>();
  }
  return m;
}

/** @brief read Eigen dense matrix from file.
 * @param[in]binfile: file path which is written by writeEigenDense().
 * @return Eigen dense matrix, return empty mat if read failed.
 */
template <typename T>
Eigen::Matrix<T, -1, -1> readEigenDense(const char* binfile) {
  std::ifstream fin(binfile, std::ios::binary);
  if (!fin.is_open()) return Eigen::Matrix<T, -1, -1>();
  return readEigenDense<T>(fin);
}

/** @brief write Eigen sparse matrix into ofstream in inner format.
 * @param[in/out]f: ostream where matrix data will be written to.
 * @param[in]m: Eigen sparse matrix
 * @return where writting ok.
 */
template <typename T>
bool writeEigenSparse(std::ofstream& f, const Eigen::SparseMatrix<T>& m) {
  uint32_t rows = m.rows();
  uint32_t cols = m.cols();
  uint32_t cnt = m.nonZeros();
  f.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  f.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
  f.write(reinterpret_cast<const char*>(&cnt), sizeof(cnt));
  for (int k = 0; k < m.outerSize(); ++k)
    for (typename Eigen::SparseMatrix<T>::InnerIterator it(m, k); it; ++it) {
      int r = it.row();  // row index
      int c = it.col();  // col index (here it is equal to k)
      T v = it.value();
      f.write(reinterpret_cast<const char*>(&r), sizeof(r));
      f.write(reinterpret_cast<const char*>(&c), sizeof(c));
      f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }
  f.flush();
  return true;
}

/** @brief write Eigen sparse matrix into file in inner format.
 * @param[in]binfile: file path where matrix data will be written to.
 * @param[in]m: Eigen sparse matrix
 * @return where writting ok.
 */
template <typename T>
bool writeEigenSparse(const char* binfile, const Eigen::SparseMatrix<T>& m) {
  std::ofstream fout(binfile, std::ios::binary);
  if (!fout.is_open()) return false;
  return writeEigenSparse(fout, m);
}

/** @brief read Eigen sparse matrix from ifstream.
 * @param[in]f: ifstream of file which is written by writeEigenSparse().
 * @return Eigen sparse matrix, return empty mat if read failed.
 */
template <typename T>
Eigen::SparseMatrix<T> readEigenSparse(std::ifstream& f) {
  uint32_t rows, cols, cnt;
  f.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  f.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  f.read(reinterpret_cast<char*>(&cnt), sizeof(cnt));
  Eigen::SparseMatrix<T> m(rows, cols);
  m.reserve(cnt);
  for (uint32_t i = 0; i < cnt; i++) {
    int r, c;
    T v;
    f.read(reinterpret_cast<char*>(&r), sizeof(r));
    f.read(reinterpret_cast<char*>(&c), sizeof(c));
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    m.insert(r, c) = v;
  }
  return m;
}

/** @brief read Eigen sparse matrix from file.
 * @param[in]binfile: file path which is written by writeEigenSparse().
 * @return Eigen sparse matrix, return empty mat if read failed.
 */
template <typename T>
Eigen::SparseMatrix<T> readEigenSparse(const char* binfile) {
  std::ifstream fin(binfile, std::ios::binary);
  if (!fin.is_open()) return Eigen::SparseMatrix<T>();
  return readEigenSparse<T>(fin);
}

}  // namespace vs
