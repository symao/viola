/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details a template particle filter with importance/weighted resampling.
 */
#pragma once

#include <functional>

#include "vs_basic.h"
#include "vs_random.h"

namespace vs {

/** @brief Particle filter with importance resampling
 * @tparam ParticleType particle datatype
 * @tparam WeightType weights type, use float or double
 */
template <class ParticleType, class WeightType>
class ParticleFilter {
 public:
  typedef std::vector<ParticleType> ParticleList;
  typedef std::vector<WeightType> WeightList;

  /** @brief clear particles*/
  void clear() {
    m_particles.clear();
    m_weights.clear();
  }

  /** @brief set particles together with weights */
  void setParticlesAndWeights(const ParticleList& particles, const WeightList& weights) {
    if (weights.size() == particles.size()) {
      m_particles = particles;
      m_weights = weights;
    } else {
      printf("[WARN]%s failed. size not match %d!=%d.\n", __func__, static_cast<int>(particles.size()),
             static_cast<int>(weights.size()));
    }
  }

  /** @brief add particles together with weights */
  void addParticlesAndWeights(const ParticleList& particles, const WeightList& weights) {
    if (weights.size() == particles.size()) {
      m_particles.insert(m_particles.end(), particles.begin(), particles.end());
      m_weights.insert(m_weights.end(), weights.begin(), weights.end());
    } else {
      printf("[WARN]%s failed. size not match %d!=%d.\n", __func__, static_cast<int>(particles.size()),
             static_cast<int>(weights.size()));
    }
  }

  /** @brief add a particle together with weight */
  void addParticleAndWeight(const ParticleType& particle, const WeightType& weight) {
    m_particles.push_back(particle);
    m_weights.push_back(weight);
  }

  /** @brief set particles */
  void setParticles(const ParticleList& particles) {
    if (particles.size() == m_particles.size()) {
      m_particles = particles;
    } else {
      printf("[WARN]%s failed. size not match %d!=%d.\n", __func__, static_cast<int>(m_particles.size()),
             static_cast<int>(particles.size()));
    }
  }

  /** @brief set weights */
  void setWeights(const WeightList& weights) {
    if (weights.size() == m_weights.size()) {
      m_weights = weights;
    } else {
      printf("[WARN]%s failed. size not match %d!=%d.\n", __func__, static_cast<int>(m_weights.size()),
             static_cast<int>(weights.size()));
    }
  }

  /** @brief get particles*/
  const ParticleList& getParticles() const { return m_particles; }

  /** @brief get particles' weights*/
  const ParticleList& getWeights() const { return m_weights; }

  /** @brief particle count*/
  int size() const { return m_particles.size(); }

  /** @brief resample fix count of particles with low-variance resampling */
  void resample(int cnt, int sample_method = WSAMPLING_LOW_VARIANCE) {
    auto ids = weightedSample(m_weights, cnt, sample_method);
    m_particles = subvec(m_particles, ids);
    m_weights = subvec(m_weights, ids);
  }

 private:
  ParticleList m_particles;
  WeightList m_weights;
};

} /* namespace vs */