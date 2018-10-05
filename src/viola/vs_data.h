/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details useful data structure(FIFO, FILO, max heap, circular queue), atom/locked data buffer, sychronized time buffer, median/mean/exponetial filter, asynchronized data saver.
 */
#pragma once
#include <list>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <numeric>

#include "vs_basic.h"
#include "vs_tictoc.h"

namespace vs {
/** @brief Meyers Singleton: Release resource before exiting */
template <class T>
class Singleton {
 public:
  static T* instance() {
    static T instance;
    return &instance;
  }

  T operator->() { return instance(); }

  const T operator->() const { return instance(); }

 private:
  Singleton() {}
  ~Singleton() {}
};

/** @brief Derived class of vector<vector<T>> for simple construction.
 * Eg:vector<vector<int> > array2d = Array2D<int>(rows, cols);
 */
template <class T>
class Array2D : public std::vector<std::vector<T>> {
 public:
  Array2D(int rows = 0, int cols = 0, const T& val = T())
      : std::vector<std::vector<T>>(rows, std::vector<T>(cols, val)) {}
};

typedef Array2D<double> Array2Dd;

/** @brief Fixed length queue.
 * Once data is out of queue size(max_len), delete the oldest data.
 */
template <class T>
class FixedQueue {
 public:
  FixedQueue(size_t max_len = 1) : m_capacity(max_len) {
    m_buffer.resize(m_capacity);
    clear();
  }

  /** @brief push a element to the back of fixed queue */
  void push(const T& a) {
    m_buffer[m_end_idx] = a;
    m_end_idx = next(m_end_idx);
    if (m_size < m_capacity) {
      m_size++;
    } else {
      m_beg_idx = next(m_beg_idx);
    }
    m_list_buf_ok = false;
  }

  /** @brief pop the back element of fixed queue */
  void pop() {
    if (m_size > 0) {
      m_end_idx = previous(m_end_idx);
      m_size--;
    }
    m_list_buf_ok = false;
  }

  /** @brief clear all queue data */
  void clear() {
    m_beg_idx = m_end_idx = m_size = 0;
    m_list_buf_ok = false;
  }

  /** @brief fetch reference of the front element of fixed queue */
  T& front() { return m_buffer[m_beg_idx]; }

  /** @brief fetch reference of the back element of fixed queue */
  T& back() { return m_buffer[previous(m_end_idx)]; }

  /** @brief get current queue size */
  size_t size() const { return m_size; }

  /** @brief check whether current queue is empty */
  bool empty() const { return m_size == 0; }

  /** @brief convert current queue to vector */
  const std::vector<T>& tolist() const {
    if (!m_list_buf_ok) {
      m_list_buf.clear();
      m_list_buf.reserve(m_size);
      for (size_t i = 0, j = m_beg_idx; i < m_size; i++) {
        m_list_buf.push_back(m_buffer[j]);
        j = next(j);
      }
      m_list_buf_ok = true;
    }
    return m_list_buf;
  }

 private:
  size_t m_capacity;                   ///< max size of fixed queue
  size_t m_beg_idx;                    ///< begin index of queue in buffer, include
  size_t m_end_idx;                    ///< end index of queue in buffer, exclude
  size_t m_size;                       ///< current queue size, which is less than m_capacity
  std::vector<T> m_buffer;             ///< circular queue buffer
  mutable bool m_list_buf_ok = false;  ///< whether m_list_buf is valid
  mutable std::vector<T> m_list_buf;   ///< cache for tolist, which prevent converting to list in multiple times

  /** @brief get next index after i */
  size_t next(size_t& i) const { return (i + 1) % m_capacity; }

  /** @brief get previous index before i */
  size_t previous(size_t& i) const { return i > 0 ? i - 1 : m_capacity - 1; }
};

/** @brief Circular queue */
template <class T>
class CircularQueue {
 public:
  /** @brief Constructor
   * @param[in]buf_len: max queue length
   */
  CircularQueue(int buf_len = 64) { setLength(buf_len); }

  /** @brief set max queue length */
  void setLength(int buf_len) {
    m_len = buf_len;
    m_buffer.resize(buf_len);
  }

  /** @brief get max queue length */
  int length() const { return m_len; }

  /** @brief return element at input index */
  T& operator[](int i) { return m_buffer[i % m_len]; }

 private:
  std::vector<T> m_buffer;  ///< data buffer
  int m_len;                ///< max queue length
};

/** @brief maximum data queue, FIFO(First-In First-Out), output max value in current buffer */
template <class T>
class MaxQueue {
 public:
  /** @brief push a value into the back of queue */
  void push(const T& a) {
    m_buffer.push_back(a);
    while (!m_max_buffer.empty() && m_max_buffer.back() < a) m_max_buffer.pop_back();
    m_max_buffer.push_back(a);
  }

  /** @brief pop a value into the front of queue */
  T pop() {
    if (m_buffer.empty()) {
      printf("[ERROR]MaxQueue: %s failed, empty queue.\n", __func__);
      return T();
    }
    T res = m_buffer.front();
    m_buffer.pop_front();
    if (res >= m_max_buffer.front()) m_max_buffer.pop_front();
    return res;
  }

  /** @brief get the max value in current queue */
  T max() const {
    if (m_max_buffer.empty()) {
      printf("[ERROR]MaxQueue: %s failed, empty queue.\n", __func__);
      return T();
    }
    return m_max_buffer.front();
  }

  /** @brief peek the front value of queue without pop out */
  T peek() const {
    if (m_buffer.empty()) {
      printf("[ERROR]MaxQueue: %s failed, empty queue.\n", __func__);
      return T();
    }
    return m_buffer.front();
  }

  /** @brief get the buffer size */
  size_t size() const { return m_buffer.size(); }

  /** @brief get current queue data */
  const std::deque<T>& data() const { return m_buffer; }

  /** @brief check whether buffer empty */
  bool empty() { return m_buffer.empty(); }

 private:
  std::deque<T> m_buffer;      ///< buffer to store queue data
  std::deque<T> m_max_buffer;  ///< buffer to store max values, whose front is always the max of current queue
};

/** @brief minimum data queue, FIFO(First-In First-Out), output max value in current buffer */
template <class T>
class MinQueue {
 public:
  /** @brief push a value into the back of queue */
  void push(const T& a) {
    m_buffer.push_back(a);
    while (!m_min_buffer.empty() && m_min_buffer.back() > a) m_min_buffer.pop_back();
    m_min_buffer.push_back(a);
  }

  /** @brief pop a value into the front of queue */
  T pop() {
    if (m_buffer.empty()) {
      printf("[ERROR]MinQueue: %s failed, empty queue.\n", __func__);
      return T();
    }
    T res = m_buffer.front();
    m_buffer.pop_front();
    if (res <= m_min_buffer.front()) m_min_buffer.pop_front();
    return res;
  }

  /** @brief get the min value in current queue */
  T min() const {
    if (m_min_buffer.empty()) {
      printf("[ERROR]MinQueue: %s failed, empty queue.\n", __func__);
      return T();
    }
    return m_min_buffer.front();
  }

  /** @brief peek the front value of queue without pop out */
  T peek() const {
    if (m_buffer.empty()) {
      printf("[ERROR]MinQueue: %s failed, empty queue.\n", __func__);
      return T();
    }
    return m_buffer.front();
  }

  /** @brief get the buffer size */
  size_t size() const { return m_buffer.size(); }

  /** @brief get current queue data */
  const std::deque<T>& data() const { return m_buffer; }

  /** @brief check whether buffer empty */
  bool empty() { return m_buffer.empty(); }

 private:
  std::deque<T> m_buffer;      ///< buffer to store queue data
  std::deque<T> m_min_buffer;  ///< buffer to store max values, whose front is always the min of current queue
};

/** @brief maximum data stack, FILO(First-In Last-Out), output max value in current buffer */
template <class T>
class MaxStack {
 public:
  /** @brief push a value into the back of stack */
  void push(const T& a) {
    m_buffer.push_back(a);
    if (m_max_buffer.empty() || m_max_buffer.back() <= a) m_max_buffer.push_back(a);
  }

  /** @brief pop a value into the back of stack */
  T pop() {
    if (m_buffer.empty()) {
      printf("[ERROR]MaxStack: %s failed, empty stack.\n", __func__);
      return T();
    }
    T res = m_buffer.back();
    m_buffer.pop_back();
    if (res >= m_max_buffer.back()) m_max_buffer.pop_back();
    return res;
  }

  /** @brief get the max value in current stack */
  T max() const {
    if (m_max_buffer.empty()) {
      printf("[ERROR]MaxStack: %s failed, empty stack.\n", __func__);
      return T();
    }
    return m_max_buffer.back();
  }

  /** @brief peek the back value of stack without pop out */
  T peek() const {
    if (m_buffer.empty()) {
      printf("[ERROR]MaxStack: %s failed, empty stack.\n", __func__);
      return T();
    }
    return m_buffer.back();
  }

  /** @brief get the buffer size */
  size_t size() const { return m_buffer.size(); }

  /** @brief get current queue data */
  const std::vector<T>& data() const { return m_buffer; }

  /** @brief check whether buffer empty */
  bool empty() { return m_buffer.empty(); }

  /** @brief reserve capacity */
  void reserve(size_t capacity) {
    m_buffer.reserve(capacity);
    m_max_buffer.reserve(capacity);
  }

 private:
  std::vector<T> m_buffer;      ///< buffer to store stack data
  std::vector<T> m_max_buffer;  ///< buffer to store max values, whose back is always the max of current queue
};

/** @brief minimum data stack, FILO(First-In Last-Out), output max value in current buffer */
template <class T>
class MinStack {
 public:
  /** @brief push a value into the back of stack */
  void push(const T& a) {
    m_buffer.push_back(a);
    if (m_min_buffer.empty() || m_min_buffer.back() >= a) m_min_buffer.push_back(a);
  }

  /** @brief pop a value into the back of stack */
  T pop() {
    if (m_buffer.empty()) {
      printf("[ERROR]MinStack: %s failed, empty stack.\n", __func__);
      return T();
    }
    T res = m_buffer.back();
    m_buffer.pop_back();
    if (res <= m_min_buffer.back()) m_min_buffer.pop_back();
    return res;
  }

  /** @brief get the max value in current stack */
  T min() const {
    if (m_min_buffer.empty()) {
      printf("[ERROR]MinStack: %s failed, empty stack.\n", __func__);
      return T();
    }
    return m_min_buffer.back();
  }

  /** @brief peek the back value of stack without pop out */
  T peek() const {
    if (m_buffer.empty()) {
      printf("[ERROR]MinStack: %s failed, empty stack.\n", __func__);
      return T();
    }
    return m_buffer.back();
  }

  /** @brief get the buffer size */
  size_t size() const { return m_buffer.size(); }

  /** @brief get current queue data */
  const std::vector<T>& data() const { return m_buffer; }

  /** @brief check whether buffer empty */
  bool empty() { return m_buffer.empty(); }

  /** @brief reserve capacity */
  void reserve(size_t capacity) {
    m_buffer.reserve(capacity);
    m_min_buffer.reserve(capacity);
  }

 private:
  std::vector<T> m_buffer;      ///< buffer to store stack data
  std::vector<T> m_min_buffer;  ///< buffer to store min values, whose back is always the max of current queue
};

/** @brief max heap, top of heap is always the maximum */
template <class T>
class MaxHeap {
 public:
  /** @brief whether heap is empty */
  bool empty() { return m_buffer.empty(); }

  /** @brief get heap size */
  size_t size() { return m_buffer.size(); }

  /** @brief get top element, which is the maximum */
  T top() {
    if (m_buffer.empty()) {
      printf("[ERROR]MaxHeap: %s failed, empty queue.\n", __func__);
      return T();
    }
    return m_buffer[0];
  }

  /** @brief insert a element to heap */
  void push(const T& a) {
    m_buffer.push_back(a);
    int k = m_buffer.size() - 1;
    while (k > 0) {
      k = (k - 1) / 2;
      heapDown(k);
    }
  }

  /** @brief pop the top of heap, and return the pop element */
  T pop() {
    if (m_buffer.empty()) {
      printf("[ERROR]MaxHeap: %s failed, empty queue.\n", __func__);
      return T();
    }
    T res = m_buffer[0];
    std::swap(m_buffer[0], m_buffer.back());
    m_buffer.pop_back();
    heapDown(0);
    return res;
  }

 private:
  std::vector<T> m_buffer;  ///< buffer to store queue data

  void heapDown(size_t i) {
    size_t l = i * 2 + 1;
    if (l >= m_buffer.size()) return;
    size_t r = i * 2 + 2;
    size_t idx = r < m_buffer.size() && m_buffer[r] > m_buffer[l] ? r : l;
    T& a = m_buffer[i];
    T& b = m_buffer[idx];
    if (a < b) {
      std::swap(a, b);
      heapDown(idx);
    }
  }
};

/** @brief Single element buffer without mutex */
template <class T>
class ElemBuffer {
 public:
  /** @brief set element */
  void set(const T& a) {
    m_obj = a;
    m_has = true;
  }

  /** @brief get element */
  T get() const { return m_has ? m_obj : T(); }

  /** @brief check whether has element */
  bool has() const { return m_has; }

  /** @brief clear buffer */
  void clear() { m_has = false; }

 private:
  T m_obj;             ///< data object
  bool m_has = false;  ///< whether has data
};

/** @brief Single element buffer with mutex, which is multi-thread-safe */
template <class T>
class AtomDataBuffer {
 public:
  AtomDataBuffer() : m_has(false) {}

  /** @brief check whether has element */
  bool has() const {
    m_mtx.lock();
    bool res = m_has;
    m_mtx.unlock();
    return res;
  }

  /** @brief set element */
  void set(const T& a) {
    m_mtx.lock();
    m_obj = a;
    m_has = true;
    m_mtx.unlock();
  }

  /** @brief get element */
  T get() const {
    m_mtx.lock();
    T res;
    if (m_has) res = m_obj;
    m_mtx.unlock();
    return res;
  }

  /** @brief try get element and return if get succeed */
  bool get(T& ret) const {
    m_mtx.lock();
    bool ok = m_has;
    if (ok) ret = m_obj;
    m_mtx.unlock();
    return ok;
  }

  /** @brief clear buffer */
  void clear() {
    m_mtx.lock();
    m_has = false;
    m_mtx.unlock();
  }

  /** @brief get element and clear buffer */
  T getAndClear() {
    m_mtx.lock();
    T res;
    if (m_has) {
      res = m_obj;
      m_has = false;
    }
    m_mtx.unlock();
    return res;
  }

  /** @brief get element and clear buffer and return if get succeed */
  bool getAndClear(T& ret) {
    m_mtx.lock();
    bool ok = m_has;
    if (ok) {
      ret = m_obj;
      m_has = false;
    }
    m_mtx.unlock();
    return ok;
  }

 private:
  T m_obj;                   ///< data object
  bool m_has;                ///< whether has data
  mutable std::mutex m_mtx;  ///< data mutex
};

/** @brief An exponential filter. Output the smooth value. */
template <class T>
class ExpFilter {
 public:
  /** @brief Constructor
   * @param[in]lambda: dampling factor of history data, 0 means not use history data, 1 means not use new data
   */
  ExpFilter(float lambda = 0.5) { m_lambda = clip(lambda, 0, 1); }

  T filter(const T& data) {
    if (!m_has_data) {
      m_smooth_data = data;
      m_has_data = true;
    } else {
      m_smooth_data = m_smooth_data * m_lambda + data * (1 - m_lambda);
    }
    return m_smooth_data;
  }

  void reset() { m_has_data = false; }

 private:
  bool m_has_data = false;  ///< whether has data, whether smooth_data init
  T m_smooth_data;          ///< smoothed data
  float m_lambda;           ///< dampling factor of history data, 0 means not use history data, 1 means not use new data
};

/** @brief A median filter. Output the median value of the kth latest inputs. */
template <class T>
class MedFilter {
 public:
  MedFilter(int k = 1) : m_buffer(k) {}

  T filter(const T& a) {
    m_buffer.push(a);
    int n = m_buffer.size() / 2;
    std::nth_element(m_buffer.begin(), m_buffer.begin() + n, m_buffer.end());
    return m_buffer[n];
  }

 private:
  FixedQueue<T> m_buffer;
};

/** @brief A mean filter. Output the mean value of the kth latest inputs. */
template <class T>
class MeanFilter {
 public:
  /** @brief Constructor
   * @param[in]k: length of history data buffer
   */
  MeanFilter(int k = 1) : m_buffer(k) {}

  T filter(const T& a) {
    m_buffer.push(a);
    T sum = std::accumulate<T>(m_buffer.begin(), m_buffer.end(), 0);
    return sum * (1.0f / static_cast<float>(m_buffer.size()));
  }

 private:
  FixedQueue<T> m_buffer;
};

/** @brief Asynchronized data saver, save data in another thread, so call write won't block */
template <class T>
class DataSaver {
 public:
  typedef std::function<void(FILE* fp, const T& t)> WriteFunction;  ///< type of function which write data into file

  /** @brief constructor
   * @param[in]save_file: write file path
   * @param[in]write_function: function which write data into file
   */
  DataSaver(const char* save_file, WriteFunction write_function)
      : m_exit(false),
        m_out_file(save_file),
        m_write_fun(write_function),
        m_thread_ptr(new std::thread(std::bind(&DataSaver<T>::run, this))) {}

  /** @brief deconstructor */
  ~DataSaver() {
    m_exit = true;
    join();
  }

  /** @brief push data to write buffer queue */
  void push(const T& data) {
    m_mtx.lock();
    m_buffer.push_back(data);
    m_mtx.unlock();
  }

  /** @brief join write thread */
  void join() { m_thread_ptr->join(); }

 private:
  bool m_exit;                                            ///< whether exit write thread
  std::string m_out_file;                                 ///< write file path
  std::mutex m_mtx;                                       ///< mutex for data buffer queue
  std::deque<T> m_buffer;                                 ///< data buffer queue
  std::function<void(FILE* fp, const T& t)> m_write_fun;  ///< function which write data into file
  std::shared_ptr<std::thread> m_thread_ptr;              ///< write thread

  /** @brief data write thread */
  void run() {
    FILE* fp = NULL;
    while (!m_exit) {
      if (!fp) {  // open file
        m_mtx.lock();
        bool has_data = !m_buffer.empty();
        m_mtx.unlock();
        if (has_data) {
#ifdef WIN32
          fopen_s(&fp, m_out_file.c_str(), "w");
#else
          fp = fopen(m_out_file.c_str(), "w");
#endif
          if (!fp) {
            printf("[ERROR]DataSaver: Cannot open '%s', Exit.\n", m_out_file.c_str());
            m_exit = true;
          }
        }
      } else {  // write data
        m_mtx.lock();
        auto buffer = m_buffer;
        m_buffer.clear();
        m_mtx.unlock();
        if (!buffer.empty()) {
          for (const auto& t : buffer) {
            m_write_fun(fp, t);
          }
          fflush(fp);
        }
      }
      msleep(500);
    }
    if (fp) fclose(fp);
  }
};

/** @brief Timestamped data buffer, used for data synchronization */
template <class T>
class TimeBuffer {
 public:
  typedef std::function<T(const T&, const T&, double)> LerpFunc;

  /** @brief constructor */
  TimeBuffer() {
    // a1: data at t1, a2: data at t2, t: timestamp rate at [t1,t2] range from [0,1]
    m_foo_lerp = [](const T& a1, const T& a2, double t) { return (1 - t) * a1 + t * a2; };
  }

  /** @brief constructor, set lerp function
   * @param[in]f_lerp: interpolation function. default: (1 - t) * a1 + t * a2
   *                   a1: data at t1, a2: data at t2, t: timestamp rate at [t1,t2] range from [0,1]
   * @note if data cannot be weighted sum directly, such as quaternion, angle,
   *       use this constructor to reimplement interpolation
   */
  explicit TimeBuffer(LerpFunc f_lerp) { m_foo_lerp = f_lerp; }

  /** @brief get front/oldest data in queue */
  T front() const { return buffer.front().second; }

  /** @brief get back/latest data in queue */
  T back() const { return buffer.back().second; }

  /** @brief get timestamp front/oldest data in queue */
  double frontTs() const { return buffer.front().first; }

  /** @brief get timestamp back/latest data in queue */
  double backTs() const { return buffer.back().first; }

  /** @brief check whether buffer is empty */
  bool empty() const { return buffer.empty(); }

  /** @brief add a data with timestamp into buffer queue
   * @param[in]ts: data timestamp, must be larger than all timestamps of buffer data
   * @param[in]a: data
   */
  void add(double ts, const T& a) {
    // make sure buffer time in ascend order
    if (buffer.empty() || backTs() < ts)
      buffer.push_back(std::make_pair(ts, a));
    else
      printf("[ERROR]TimeBuffer:ts(%f) must be later than the latest ts in buffer(%f)\n", ts, backTs());
  }

  /** @brief get latest data in buffer earlier than specific query timestamp
   * - case ts--(t0-...-tN): return false
   * - case (t0-...t3-ts---t4-...-tN): return t3 data
   * - case (t0-...t3---ts-t4-...-tN): return t3 data
   * - case (t0-...-tN)--ts: return tN data
   * @param[in]ts: query timestamp
   * @param[in]res: nearest data, only valid when return true
   * @param[in]ts_res: timestamp of nearest data, only valid when return true
   * @return whether find nearest data
   */
  bool getLatest(double ts, T& res, double& ts_res) {
    if (buffer.empty()) return false;
    if (ts <= frontTs()) {
      // ts earlier than all buffer data
      return false;
    } else if (ts >= backTs()) {
      // ts later than all buffer data
      res = back();
      ts_res = backTs();
    } else {
      // left,right in buffer range, return the nearest data among left and right
      int left, right;
      findIndexRange(ts, left, right);
      int idx;
      if (right >= static_cast<int>(buffer.size()))
        idx = buffer.size() - 1;
      else if (right <= 0)
        idx = 0;
      else
        idx = left;
      const auto& b = buffer[idx];
      ts_res = b.first;
      res = b.second;
    }
    return true;
  }

  /** @brief get nearest data in buffer with specific query timestamp
   * if query timestamp not in buffer timestamp range, return the nearest front/back data in buffer queue
   * - case ts--(t0-...-tN): return t0 data
   * - case (t0-...t3-ts---t4-...-tN): return t3 data
   * - case (t0-...t3---ts-t4-...-tN): return t4 data
   * - case (t0-...-tN)--ts: return tN data
   * @param[in]ts: query timestamp
   * @param[in]res: nearest data, only valid when return true
   * @param[in]ts_res: timestamp of nearest data, only valid when return true
   * @return whether find nearest data
   */
  bool getNearest(double ts, T& res, double& ts_res) const {
    if (buffer.empty()) return false;
    if (ts <= frontTs()) {
      // ts earlier than all buffer data
      res = front();
      ts_res = frontTs();
    } else if (ts >= backTs()) {
      // ts later than all buffer data
      res = back();
      ts_res = backTs();
    } else {
      // left,right in buffer range, return the nearest data among left and right
      int left, right;
      findIndexRange(ts, left, right);
      int idx;
      if (right >= static_cast<int>(buffer.size())) {
        idx = buffer.size() - 1;
      } else if (right <= 0) {
        idx = 0;
      } else if (left == right) {
        idx = left;
      } else {
        idx = ts - buffer[left].first < buffer[right].first - ts ? left : right;
      }
      const auto& b = buffer[idx];
      ts_res = b.first;
      res = b.second;
    }
    return true;
  }

  /** @brief get interpolation data in buffer with specific query timestamp
   * if query timestamp not in buffer timestamp range, return false. else output interpolation
   * @param[in]ts: query timestamp
   * @param[in]res: interpolation data at query timestamp, only valid when return true
   * @return whether find data at query timestamp
   */
  bool get(double ts, T& res) const {
    if (buffer.empty() || ts < frontTs() || ts > backTs()) return false;
    int left, right;
    findIndexRange(ts, left, right);
    if (left == right) {
      res = buffer[left].second;
      return true;
    }
    if (right >= static_cast<int>(buffer.size()) || right <= 0) return false;
    const auto& a1 = buffer[left];
    const auto& a2 = buffer[right];
    double dt = a2.first - a1.first;
    // dt = 0 may not happen. it means ts=left_ts=right_ts, while is handled at if (left==right)
    res = m_foo_lerp(a1.second, a2.second, fequal(dt, 0) ? 0.5 : (ts - a1.first) / dt);
    return true;
  }

  /** @brief get data list from buffer queue with input timestamp range (start_ts, stop_ts]
   * @param[in]start_ts: start timestamp, not include
   * @param[in]stop_ts: stop timestamp, include
   * @return list of buffer data whose timestamp in (start_ts, stop_ts]
   */
  std::vector<T> getRange(double start_ts, double stop_ts) {
    std::vector<T> res;
    if (start_ts > stop_ts) return res;
    if (0) {  // brute-force
      for (const auto& item : buffer) {
        if (item.first <= start_ts) {
          continue;
        } else if (item.first > stop_ts) {
          break;
        }
        res.push_back(item.second);
      }
    } else {
      // binary search
      int start_l, start_r, stop_l, stop_r;
      findIndexRange(start_ts, start_l, start_r);
      findIndexRange(stop_ts, stop_l, stop_r);
      if (start_r < buffer.size() && buffer[start_r].ts == start_ts) start_r++;
      if (start_r < buffer.size() && stop_l < buffer.size() && start_r <= stop_l) {
        for (int i = start_r; i <= stop_l; i++) res.push_back(buffer[i]);
      }
    }
    return res;
  }

  /** @brief drop buffer whose timestamp older than input old_ts
   * @param[in]old_ts: old timestamp, not include
   */
  void dropOld(double old_ts, bool include = false) {
    if (buffer.empty()) return;
    auto it = buffer.begin();
    if (include) {
      while (it->first <= old_ts && it != buffer.end()) it++;
    } else {
      while (it->first < old_ts && it != buffer.end()) it++;
    }
    buffer.erase(buffer.begin(), it);
  }

  /** @brief clear buffer queue */
  void clear() { buffer.clear(); }

  std::deque<std::pair<double, T>> buffer;  ///< data buffer queue

 private:
  LerpFunc m_foo_lerp;  ///< interpolation function

  /** @brief find index range that ts in [left,right] and right=left+1
   * if find one exactly equal ts, then left=right=index
   * if buffer empty, then left=-1, right=0
   * if ts later than whole buffer, then left=nbuf-1, right=nbuf
   * if ts earlier than whole buffer, then left=-1, right=0
   * if ts inside, then ts in (left,right)
   * only two case: either left==right or left+1==right
   */
  void findIndexRange(double ts, int& left, int& right) const {
    int nbuf = buffer.size();
    left = 0, right = nbuf;
    while (left < right) {
      int mid = (left + right) / 2;
      const auto& b_mid = buffer[mid];
      if (fequal(b_mid.first, ts)) {
        // if equal exactly, return
        left = right = mid;
        return;
      } else if (b_mid.first < ts) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    if (left == nbuf || buffer[left].first > ts) {
      left--;
    } else {
      right++;
    }
  }
};

/** @brief data queue in timestamp ascend order */
template <class T>
class TimedQueue {
 public:
  typedef std::function<double(const T& a)> TsFunc;

  TimedQueue(TsFunc func) {
    if (func) {
      ts_func_ = func;
    } else {
      ts_func_ = [](const T& a) { return 0; };  // default function which is invalid
    }
  }

  void clear() { list_.clear(); }

  /** @brief insert observation, keep list ts in ascend order. */
  void insert(const T& obs) {
    if (list_.empty() || ts_func_(obs) >= ts_func_(list_.back())) {
      list_.push_back(obs);
    } else if (ts_func_(obs) <= ts_func_(list_.front())) {
      list_.push_front(obs);
    } else if (ts_func_(list_.back()) - ts_func_(obs) > ts_func_(obs) - ts_func_(list_.front())) {
      // search from front to back
      auto it = list_.begin();
      while (it != list_.end() && ts_func_(*it) <= ts_func_(obs)) it++;
      list_.insert(it, obs);
    } else {
      // search from back to front
      auto it = list_.end();
      it--;  // move to the last value
      while (it != list_.begin() && ts_func_(*it) > ts_func_(obs)) it--;
      it++;  // the first value which > obs ts
      list_.insert(it, obs);
    }
  }

  /** @brief remove the front part of list to make the list ts range less than max store seconds */
  void remove(double max_store_sec) {
    if (!list_.empty()) {
      double start_ts = ts_func_(list_.back()) - max_store_sec;
      auto it = list_.begin();
      while (it != list_.end() && ts_func_(*it) < start_ts) it++;
      list_.erase(list_.begin(), it);
    }
  }

  /** @brief remove with start ts */
  void removeBefore(double start_ts) {
    if (!list_.empty()) {
      auto it = list_.begin();
      while (it != list_.end() && ts_func_(*it) < start_ts) it++;
      list_.erase(list_.begin(), it);
    }
  }

  /** @brief fetch data from start ts(exclude) to stop ts(include)
   * @param[in]start_ts:
   * @param[in]stop_ts:
   * @param[in]data: fetched data list
   * @param[in]drop_old: if true, drop all fetched data from obs list
   */
  void fetch(double start_ts, double stop_ts, std::vector<T>& data, bool drop_old = false) {
    data.clear();
    auto it = list_.begin();
    for (; it != list_.end(); it++) {
      if (ts_func_(*it) <= start_ts) {
        continue;
      } else if (ts_func_(*it) <= stop_ts) {
        data.push_back(*it);
      } else {
        if (drop_old) list_.erase(list_.begin(), it);
        return;
      }
    }
    if (it == list_.end()) list_.clear();
  }

  /** @brief get current observation list */
  const std::list<T>& list() const { return list_; }

 private:
  std::list<T> list_;  ///< data list
  TsFunc ts_func_;     ///< function to get timestamp from template variables
};

} /* namespace vs */

#if __cplusplus > 201700L  // using std::filesystem in c++17
#include <shared_mutex>

namespace vs {

/** @brief data buffer with read-write lock */
template <class T>
class ReadWriteDataBuffer {
 public:
  typedef std::shared_mutex Lock;
  typedef std::unique_lock<Lock> WriteLock;  ///< rejects other writing or reading
  typedef std::shared_lock<Lock> ReadLock;   ///< rejects writing only

  ReadWriteDataBuffer() : m_has(false) {}

  /** @brief set element */
  void set(const T& a) {
    WriteLock w_lock(m_mtx);
    m_obj = a;
    m_has = true;
    w_lock.unlock();
  }

  /** @brief get element */
  T get() const {
    ReadLock r_lock(m_mtx);
    T res;
    if (m_has) res = m_obj;
    r_lock.unlock();
    return res;
  }

  /** @brief check whether has element */
  bool has() const {
    ReadLock r_lock(m_mtx);
    bool res = m_has;
    r_lock.unlock();
    return res;
  }

  /** @brief clear buffer */
  void clear() {
    WriteLock w_lock(m_mtx);
    m_has = false;
    w_lock.unlock();
  }

 private:
  T m_obj;             ///< data object
  bool m_has;          ///< whether has data
  mutable Lock m_mtx;  ///< read-write lock
};

} /* namespace vs */
#endif
