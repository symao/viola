#pragma once
#include <Eigen/Dense>
#include <vector>
#include <list>
#include <deque>
#include <queue>
#include <map>
#include <unordered_map>

namespace vs {
template <typename T>
using Vec2_ = Eigen::Matrix<T, 2, 1>;
template <typename T>
using Vec3_ = Eigen::Matrix<T, 3, 1>;
template <typename T>
using Vec4_ = Eigen::Matrix<T, 4, 1>;
template <typename T>
using Vec5_ = Eigen::Matrix<T, 5, 1>;
template <typename T>
using Vec6_ = Eigen::Matrix<T, 6, 1>;
template <typename T>
using VecX_ = Eigen::Matrix<T, -1, 1>;
template <typename T>
using Mat22_ = Eigen::Matrix<T, 2, 2>;
template <typename T>
using Mat33_ = Eigen::Matrix<T, 3, 3>;
template <typename T>
using Mat44_ = Eigen::Matrix<T, 4, 4>;
template <typename T>
using Mat55_ = Eigen::Matrix<T, 5, 5>;
template <typename T>
using Mat66_ = Eigen::Matrix<T, 6, 6>;
template <typename T>
using MatX2_ = Eigen::Matrix<T, -1, 2>;
template <typename T>
using MatX3_ = Eigen::Matrix<T, -1, 3>;
template <typename T>
using Mat2X_ = Eigen::Matrix<T, 2, -1>;
template <typename T>
using Mat3X_ = Eigen::Matrix<T, 3, -1>;
template <typename T>
using Mat23_ = Eigen::Matrix<T, 2, 3>;
template <typename T>
using Mat34_ = Eigen::Matrix<T, 3, 4>;
template <typename T>
using MatXX_ = Eigen::Matrix<T, -1, -1>;
template <typename T>
using MatX_ = MatXX_<T>;
template <typename T>
using Isom2_ = Eigen::Transform<T, 2, Eigen::Isometry>;
template <typename T>
using Isom3_ = Eigen::Transform<T, 3, Eigen::Isometry>;
template <typename T>
using Affine2_ = Eigen::Transform<T, 2, Eigen::Affine>;
template <typename T>
using Affine3_ = Eigen::Transform<T, 3, Eigen::Affine>;
template <typename T>
using Quat_ = Eigen::Quaternion<T>;
template <typename T>
using AngleAxis_ = Eigen::AngleAxis<T>;

template <typename T>
using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;
template <typename T>
using aligned_list = std::list<T, Eigen::aligned_allocator<T>>;
template <typename T>
using aligned_queue = std::queue<T, Eigen::aligned_allocator<T>>;
template <typename T>
using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;
template <typename K, typename V>
using aligned_map = std::map<K, V, std::equal_to<K>, Eigen::aligned_allocator<std::pair<K const, V>>>;
template <typename K, typename V>
using aligned_unordered_map =
    std::unordered_map<K, V, std::hash<K>, std::equal_to<K>, Eigen::aligned_allocator<std::pair<K const, V>>>;

}  // namespace vs