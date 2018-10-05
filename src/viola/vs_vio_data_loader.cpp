/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_vio_data_loader.h"
#include <Eigen/Dense>  // used to convert KITTI pose to quaternion
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#define VS_ENABLE_VIO_PRINT 1

#if (VS_ENABLE_VIO_PRINT && defined(WIN32))
#define VS_VIO_PRINTE(format, ...)                             \
  do {                                                         \
    printf("[ERROR]%s: " format, name().c_str(), __VA_ARGS__); \
  } while (0)
#elif (defined(VS_ENABLE_VIO_PRINT))
#define VS_VIO_PRINTE(format, args...)                    \
  do {                                                    \
    printf("[ERROR]%s: " format, name().c_str(), ##args); \
  } while (0)
#else
#define VS_VIO_PRINTE(format, ...)
#endif

#ifdef WIN32
#define SAFE_SSCANF(buffer, format, ...) sscanf_s(buffer, format, __VA_ARGS__)
#else
#define SAFE_SSCANF(buffer, format, ...) sscanf(buffer, format, __VA_ARGS__)
#endif

namespace vs {

const static char SPLITER = '/';
static std::string join(const std::string& f1, const std::string& f2) {
  if (f1.back() == SPLITER) return f1 + f2;
  return f1 + SPLITER + f2;
}

static std::string join(const std::string& f1, const std::string& f2, const std::string& f3) {
  return join(join(f1, f2), f3);
}

VioDataLoader::VioDataLoader()
    : m_init(false), m_mono(false), m_start_ts(0), m_stop_ts(0), m_td(0), m_next_type(MSG_NONE) {}

bool VioDataLoader::init(const std::string& data_dir, const std::vector<std::string>& append_files) {
  m_init = initImpl(data_dir, append_files);
  return m_init;
}

std::string VioDataLoader::name() const {
  switch (m_dataset_type) {
    case DATASET_UNKNOW:
      return "DatasetUnknown";
    case DATASET_EUROC:
      return "DatasetEuROC";
    case DATASET_KITTI:
      return "DatasetKITTI";
    case DATASET_UZH_VIO:
      return "DatasetUZH";
    case DATASET_TUM_VIO:
      return "DatasetTUM";
    case DATASET_ZJU_VIO:
      return "DatasetZJU";
    case DATASET_VS_COMMON:
      return "DatasetVsCommon";
    case DATASET_XRING:
      return "DatasetXRing";
  }
  return "UnknowDatasetType";
}

VioDataLoader::MessageType VioDataLoader::checkNextType() {
  MessageType type = MSG_NONE;
  double min_ts = DBL_MAX;
  if (m_next_imu.ts >= 0 && m_next_imu.ts < min_ts) {
    type = MSG_IMU;
    min_ts = m_next_imu.ts;
  }
  if (m_next_cam.ts >= 0 && m_next_cam.ts < min_ts) {
    type = MSG_CAMERA;
    min_ts = m_next_cam.ts;
  }
  if (m_next_gt_pose.ts >= 0 && m_next_gt_pose.ts < min_ts) {
    type = MSG_GTPOSE;
    min_ts = m_next_gt_pose.ts;
  }
  return type;
}

VioDataLoader::MessageType VioDataLoader::nextType() {
  if (m_init && m_next_type == MSG_NONE) {
    readNextImu(m_next_imu);
    readNextCamera(m_next_cam);
    readNextGtPose(m_next_gt_pose);
    // if IMU data valid, drop all data before the first IMU ts
    if (m_next_imu.ts >= 0) {
      while (m_next_cam.ts < m_next_imu.ts) readNextCamera(m_next_cam);
      while (m_next_gt_pose.ts < m_next_imu.ts) readNextGtPose(m_next_gt_pose);
    }
    m_next_type = checkNextType();
  }
  return m_next_type;
}

bool VioDataLoader::fetchImu(ImuData& imu, bool verbose) {
  if (!m_init || (m_next_type != MSG_IMU && m_next_imu.ts < 0)) return false;
  imu = m_next_imu;
  readNextImu(m_next_imu);
  m_next_type = checkNextType();
  if (verbose) {
    printf("IMU: %f gyro:(%.3f,%.3f,%.3f) acc:(%.3f,%.3f,%.3f)\n", imu.ts, imu.gyro[0], imu.gyro[1], imu.gyro[2],
           imu.acc[0], imu.acc[1], imu.acc[2]);
  }
  return true;
}

bool VioDataLoader::fetchCamera(CameraData& cam, bool verbose) {
  if (!m_init || (m_next_type != MSG_CAMERA && m_next_cam.ts < 0)) return false;
  cam = m_next_cam;
  readNextCamera(m_next_cam);
  m_next_cam.ts += m_td;  // handle time diff between camera and imu
  m_next_type = checkNextType();
  if (verbose) {
    printf("CAM: %f image:%d", cam.ts, static_cast<int>(cam.imgs.size()));
    for (const auto& img : cam.imgs) printf(" [%dx%d,%d]", img.cols, img.rows, img.channels());
    printf("\n");
  }
  return true;
}

bool VioDataLoader::fetchGtPose(PoseData& gt_pose, bool verbose) {
  if (!m_init || (m_next_type != MSG_GTPOSE && m_next_gt_pose.ts < 0)) return false;
  gt_pose = m_next_gt_pose;
  readNextGtPose(m_next_gt_pose);
  m_next_type = checkNextType();
  if (verbose) {
    printf("GTP: %f q:(%.3f,%.3f,%.3f,%.3f) t:(%.3f,%.3f,%.3f)\n", gt_pose.ts, gt_pose.qw, gt_pose.qx, gt_pose.qy,
           gt_pose.qz, gt_pose.tx, gt_pose.ty, gt_pose.tz);
  }
  return true;
}

void VioDataLoader::play(std::function<void(const ImuData&)> imu_callback,
                         std::function<void(const CameraData&)> camera_callback,
                         std::function<void(const PoseData&)> gt_callback, bool sleep, bool verbose) {
  using namespace std::chrono;
  ImuData cur_imu;
  CameraData cur_cam;
  PoseData cur_gt_pose;
  bool ok = true;
  double ref_data_ts = -1;
  system_clock::time_point ref_play_ts;
  auto foo_sleep = [&ref_data_ts, &ref_play_ts, sleep](double ts) {
    if (!sleep) return;
    if (ref_data_ts < 0) {
      ref_data_ts = ts;
      ref_play_ts = system_clock::now();
    } else {
      double sleep_ms =
          (ts - ref_data_ts) * 1000 - duration_cast<milliseconds>(system_clock::now() - ref_play_ts).count();
      if (sleep_ms > 0) std::this_thread::sleep_for(milliseconds(static_cast<int>(sleep_ms)));
    }
  };
  while (ok) {
    ok = false;
    switch (nextType()) {
      case MSG_IMU:
        ok = fetchImu(cur_imu, verbose);
        if (ok) {
          if (m_stop_ts > 0 && cur_imu.ts > m_stop_ts) return;
          foo_sleep(cur_imu.ts);
          imu_callback(cur_imu);
        }
        break;
      case MSG_CAMERA:
        ok = fetchCamera(cur_cam, verbose);
        if (ok) {
          if (m_stop_ts > 0 && cur_cam.ts > m_stop_ts) return;
          foo_sleep(cur_cam.ts);
          camera_callback(cur_cam);
        }
        break;
      case MSG_GTPOSE:
        ok = fetchGtPose(cur_gt_pose, verbose);
        if (ok) {
          if (m_stop_ts > 0 && cur_gt_pose.ts > m_stop_ts) return;
          foo_sleep(cur_gt_pose.ts);
          gt_callback(cur_gt_pose);
        }
      default:
        break;
    }
  }
}

class VioDataLoaderEuroc : public VioDataLoader {
 public:
  VioDataLoaderEuroc() {
    m_dataset_type = DATASET_EUROC;
    m_fimu = "mav0/imu0/data.csv";
    m_fimg = "mav0/cam0/data.csv";
    m_img0_dir = "mav0/cam0/data";
    m_img1_dir = "mav0/cam1/data";
    m_fgt = "mav0/state_groundtruth_estimate0/data.csv";
  }

 protected:
  struct ImgFileInfo {
    double ts;
    std::vector<std::string> fimgs;
  };

  virtual bool initImpl(const std::string& data_dir, const std::vector<std::string>& append_files);

  virtual bool readNextImu(ImuData& imu);

  virtual bool readNextCamera(CameraData& cam);

  virtual bool readNextGtPose(PoseData& gt_pose);

  virtual void parseImuLine(const char* line_str, ImuData& imu) {
#if VS_VIO_ENABLE_MAG
    SAFE_SSCANF(line_str, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &imu.ts, &imu.gyro[0], &imu.gyro[1], &imu.gyro[2],
                &imu.acc[0], &imu.acc[1], &imu.acc[2], &imu.mag[0], &imu.mag[1], &imu.mag[2]);
#else
    SAFE_SSCANF(line_str, "%lf,%lf,%lf,%lf,%lf,%lf,%lf", &imu.ts, &imu.gyro[0], &imu.gyro[1], &imu.gyro[2], &imu.acc[0],
                &imu.acc[1], &imu.acc[2]);
#endif
    imu.ts /= 1e9;
  }

  virtual void parseImgLine(const char* line_str, ImgFileInfo& a) {
    char name[128] = {0};
    SAFE_SSCANF(line_str, "%lf,%s", &a.ts, name);
    a.ts /= 1e9;  // ms -> s
    a.fimgs = {join(m_datadir, m_img0_dir, name)};
    if (!m_mono) a.fimgs.push_back(join(m_datadir, m_img1_dir, name));
  }

  virtual void parseGtLine(const char* line_str, PoseData& a) {
    SAFE_SSCANF(line_str, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &a.ts, &a.tx, &a.ty,
                &a.tz, &a.qw, &a.qx, &a.qy, &a.qz, &a.vx, &a.vy, &a.vz, &a.bw_x, &a.bw_y, &a.bw_z, &a.ba_x, &a.ba_y,
                &a.ba_z);
    a.ts /= 1e9;
  }

 protected:
  std::string m_datadir, m_fimu, m_fimg, m_fgt, m_img0_dir, m_img1_dir;
  std::ifstream m_fin_img, m_fin_imu, m_fin_gt;
};

bool VioDataLoaderEuroc::initImpl(const std::string& data_dir, const std::vector<std::string>& append_files) {
  m_datadir = data_dir;
  std::string img_list = join(m_datadir, m_fimg);
  std::string imu_file = join(m_datadir, m_fimu);
  std::string gt_file = join(m_datadir, m_fgt);
  m_fin_img.open(img_list.c_str());
  m_fin_imu.open(imu_file.c_str());
  m_fin_gt.open(gt_file.c_str());
  if (!m_fin_img.is_open()) {
    VS_VIO_PRINTE("cannot open: %s\n", img_list.c_str());
    return false;
  }
  if (!m_fin_imu.is_open()) {
    VS_VIO_PRINTE("cannot open: %s\n", imu_file.c_str());
    return false;
  }
  printf("%s: Dataset loaded. dir:%s\n", name().c_str(), m_datadir.c_str());
  return true;
}

bool VioDataLoaderEuroc::readNextImu(ImuData& imu) {
  imu.ts = -1;
  std::string line;
  while (getline(m_fin_imu, line)) {
    if (line.length() < 1 || line[0] == '#') continue;  // skip comment line and empty line
    parseImuLine(line.c_str(), imu);
    if (imu.ts < 0)
      return false;
    else if (imu.ts >= m_start_ts)
      return true;
  }
  return false;
}

bool VioDataLoaderEuroc::readNextGtPose(PoseData& gt_pose) {
  gt_pose.ts = -1;
  std::string line;
  while (getline(m_fin_gt, line)) {
    if (line.length() < 1 || line[0] == '#') continue;  // skip comment line and empty line
    parseGtLine(line.c_str(), gt_pose);
    if (gt_pose.ts < 0)
      return false;
    else if (gt_pose.ts >= m_start_ts)
      return true;
  }
  return false;
}

bool VioDataLoaderEuroc::readNextCamera(CameraData& cam) {
  cam.ts = -1;
  cam.imgs.clear();
  std::string line;
  ImgFileInfo info;
  while (getline(m_fin_img, line)) {
    if (line.length() < 1 || line[0] == '#') continue;  // skip comment line and empty line
    parseImgLine(line.c_str(), info);
    if (info.ts < m_start_ts) continue;
    // write camera
    cam.imgs.resize(info.fimgs.size());
    bool valid = true;
    for (size_t i = 0; i < info.fimgs.size(); i++) {
      const auto& fimg = info.fimgs[i];
      cv::Mat img = cv::imread(fimg, cv::IMREAD_GRAYSCALE);
      if (img.empty()) {
        VS_VIO_PRINTE("Cannot open image:%s\n", fimg.c_str());
        valid = false;
        break;
      }
      cam.imgs[i] = img;
    }
    if (valid) {
      cam.ts = info.ts;
      return true;
    }
  }
  return false;
}

class VioDataLoaderTumVi : public VioDataLoaderEuroc {
 public:
  VioDataLoaderTumVi() {
    m_dataset_type = DATASET_TUM_VIO;
    m_fimu = "mav0/imu0/data.csv";
    m_fimg = "mav0/cam0/data.csv";
    m_fgt = "mav0/mocap0/data.csv";
  }

 protected:
  virtual void parseGtLine(const char* line_str, PoseData& a) {
    SAFE_SSCANF(line_str, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &a.ts, &a.tx, &a.ty, &a.tz, &a.qw, &a.qx, &a.qy, &a.qz);
    a.ts /= 1e9;
  }
};

class VioDataLoaderUzhFpv : public VioDataLoaderEuroc {
 public:
  VioDataLoaderUzhFpv() {
    m_dataset_type = DATASET_UZH_VIO;
    m_fimu = "imu.txt";
    m_fimg = "left_images.txt";
    m_fgt = "";
  }

 protected:
  virtual void parseImuLine(const char* line_str, ImuData& a) {
    int id;
    SAFE_SSCANF(line_str, "%d %lf %lf %lf %lf %lf %lf %lf", &id, &a.ts, &a.gyro[0], &a.gyro[1], &a.gyro[2], &a.acc[0],
                &a.acc[1], &a.acc[2]);
  }

  virtual void parseImgLine(const char* line_str, ImgFileInfo& a) {
    char name[128] = {0};
    int id = 0;
    SAFE_SSCANF(line_str, "%d %lf %s", &id, &a.ts, name);
    char fimgl[512] = {0};
    snprintf(fimgl, 512, "%s/img/image_0_%d.png", m_datadir.c_str(), id);
    char fimgr[512] = {0};
    snprintf(fimgr, 512, "%s/img/image_1_%d.png", m_datadir.c_str(), id);
    a.fimgs = {std::string(fimgl), std::string(fimgr)};
  }

  virtual void parseGtLine(const char* line_str, PoseData& a) {}
};

class VioDataLoaderZjuViSlam : public VioDataLoaderEuroc {
 public:
  VioDataLoaderZjuViSlam() {
    m_dataset_type = DATASET_ZJU_VIO;
    m_fimu = "imu/data.csv";
    m_fimg = "camera/data.csv";
    m_fgt = "groundtruth/data.csv";
  }

 protected:
  virtual void parseImuLine(const char* line_str, ImuData& a) {
    SAFE_SSCANF(line_str, "%lf,%lf,%lf,%lf,%lf,%lf,%lf", &a.ts, &a.gyro[0], &a.gyro[1], &a.gyro[2], &a.acc[0],
                &a.acc[1], &a.acc[2]);
  }

  virtual void parseImgLine(const char* line_str, ImgFileInfo& a) {
    char name[128] = {0};
    SAFE_SSCANF(line_str, "%lf,%s", &a.ts, name);
    a.fimgs = {join(m_datadir, "camera/images", name)};
  }

  virtual void parseGtLine(const char* line_str, PoseData& a) {
    SAFE_SSCANF(line_str, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &a.ts, &a.qx, &a.qy, &a.qz, &a.qw, &a.tx, &a.ty, &a.tz);
  }
};

class VioDataLoaderKitti : public VioDataLoader {
 public:
  explicit VioDataLoaderKitti() {
    m_dataset_type = DATASET_KITTI;
    m_left_dir = "image_0";
    m_right_dir = "image_1";
    m_ts_file = "times.txt";
  }

 protected:
  virtual bool initImpl(const std::string& data_dir, const std::vector<std::string>& append_files);

  virtual bool readNextImu(ImuData& imu);

  virtual bool readNextCamera(CameraData& cam);

  virtual bool readNextGtPose(PoseData& gt_pose);

 private:
  int m_read_img_idx = 0;
  int m_read_gt_idx = 0;
  std::string m_left_dir, m_right_dir, m_ts_file, m_datadir;
  std::vector<double> m_ts_buf;
  std::vector<std::vector<double>> m_gt_pose_buf;
};

bool VioDataLoaderKitti::initImpl(const std::string& data_dir, const std::vector<std::string>& append_files) {
  m_datadir = data_dir;
  std::string ts_file = join(m_datadir, m_ts_file);
  std::ifstream fin_ts(ts_file);
  if (!fin_ts.is_open()) {
    VS_VIO_PRINTE("cannot open: %s\n", ts_file.c_str());
    return false;
  }
  // read ts into buffer
  std::string line;
  while (getline(fin_ts, line)) {
    if (line.length() < 1 || line[0] == '#') continue;  // skip comment line and empty line
    double ts;
    SAFE_SSCANF(line.c_str(), "%lf", &ts);
    m_ts_buf.push_back(ts);
  }

  if (!append_files.empty()) {
    std::string gt_file = append_files[0];
    std::ifstream fin_gt(gt_file);
    if (!fin_gt.is_open()) {
      VS_VIO_PRINTE("cannot open: %s\n", gt_file.c_str());
      return false;
    }
    // read gt pose into buf
    while (getline(fin_gt, line)) {
      if (line.length() < 1 || line[0] == '#') continue;  // skip comment line and empty line
      std::vector<double> v(12, 0);
      SAFE_SSCANF(line.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &v[0], &v[1], &v[2], &v[3], &v[4],
                  &v[5], &v[6], &v[7], &v[8], &v[9], &v[10], &v[11]);
      m_gt_pose_buf.push_back(v);
    }
  }
  m_read_img_idx = 0;
  m_read_gt_idx = 0;
  printf("%s: Dataset loaded. dir:%s, frame:%d gt_pose:%d\n", name().c_str(), m_datadir.c_str(),
         static_cast<int>(m_ts_buf.size()), static_cast<int>(m_gt_pose_buf.size()));
  return true;
}

bool VioDataLoaderKitti::readNextImu(ImuData& imu) {
  imu.ts = -1;
  VS_VIO_PRINTE("no imu data.\n");
  return false;
}

bool VioDataLoaderKitti::readNextCamera(CameraData& cam) {
  cam.ts = -1;
  cam.imgs.clear();
  if (m_read_img_idx >= static_cast<int>(m_ts_buf.size())) return false;
  cam.ts = m_ts_buf[m_read_img_idx];
  char str[512] = {0};
  snprintf(str, 512, "%s/%s/%06d.png", m_datadir.c_str(), m_left_dir.c_str(), m_read_img_idx);
  cv::Mat img = cv::imread(str, cv::IMREAD_GRAYSCALE);
  if (img.empty()) {
    VS_VIO_PRINTE("Cannot open image:%s\n", str);
    return false;
  }
  cam.imgs.push_back(img);
  if (!m_mono) {
    snprintf(str, 512, "%s/%s/%06d.png", m_datadir.c_str(), m_right_dir.c_str(), m_read_img_idx);
    cv::Mat img = cv::imread(str, cv::IMREAD_GRAYSCALE);
    if (!img.empty()) cam.imgs.push_back(img);
  }
  m_read_img_idx++;
  return true;
}

bool VioDataLoaderKitti::readNextGtPose(PoseData& gt_pose) {
  gt_pose.ts = -1;
  int n = std::min(m_gt_pose_buf.size(), m_gt_pose_buf.size());
  if (m_read_gt_idx >= n) return false;
  gt_pose.ts = m_ts_buf[m_read_gt_idx];
  const auto& v = m_gt_pose_buf[m_read_gt_idx];
  // convert transformation matrix to pose
  gt_pose.tx = v[3];
  gt_pose.ty = v[7];
  gt_pose.tz = v[11];
#ifdef EIGEN_MAJOR_VERSION
  Eigen::Matrix3d rot;
  rot << v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10];
  Eigen::Quaterniond quat(rot);
  gt_pose.qx = quat.x();
  gt_pose.qy = quat.y();
  gt_pose.qz = quat.z();
  gt_pose.qw = quat.w();
#endif
  m_read_gt_idx++;
  return true;
}

class VioDataLoaderVsCommon : public VioDataLoader {
 public:
  explicit VioDataLoaderVsCommon(bool split_lr = false) : m_split_lr(split_lr) { m_dataset_type = DATASET_VS_COMMON; }

 protected:
  virtual bool initImpl(const std::string& data_dir, const std::vector<std::string>& append_files);

  virtual bool readNextImu(ImuData& imu);

  virtual bool readNextCamera(CameraData& cam);

  virtual bool readNextGtPose(PoseData& gt_pose);

 private:
  bool m_split_lr;
  std::string m_datadir;
  cv::Mat m_cap_img;
  cv::VideoCapture m_cap;
  std::ifstream m_fin_img;
  std::ifstream m_fin_imu;
};

bool VioDataLoaderVsCommon::initImpl(const std::string& data_dir, const std::vector<std::string>& append_files) {
  m_datadir = data_dir;
  std::string video_file = join(m_datadir, "img.avi");
  std::string img_file = join(m_datadir, "imgts.txt");
  std::string imu_file = join(m_datadir, "imu_meta.txt");
  m_cap.open(video_file);
  if (!m_cap.isOpened()) {
    VS_VIO_PRINTE("cannot open video %s\n", video_file.c_str());
    return false;
  }
  m_fin_img.open(img_file);
  if (!m_fin_img.is_open()) {
    VS_VIO_PRINTE("cannot open file %s\n", img_file.c_str());
    return false;
  }
  m_fin_imu.open(imu_file);
  if (!m_fin_imu.is_open()) {
    VS_VIO_PRINTE("cannot open file %s\n", imu_file.c_str());
    return false;
  }
  return true;
}

bool VioDataLoaderVsCommon::readNextImu(ImuData& imu) {
  imu.ts = -1;
  std::string line;
  while (getline(m_fin_imu, line)) {
    if (line.length() < 1 || line[0] == '#') continue;  // skip comment line and empty line
    SAFE_SSCANF(line.c_str(), "%lf %lf %lf %lf %lf %lf %lf", &imu.ts, &imu.acc[0], &imu.acc[1], &imu.acc[2],
                &imu.gyro[0], &imu.gyro[1], &imu.gyro[2]);
    if (imu.ts < 0)
      return false;
    else if (imu.ts >= m_start_ts)
      return true;
  }
  return false;
}

bool VioDataLoaderVsCommon::readNextCamera(CameraData& cam) {
  cam.ts = -1;
  cam.imgs.clear();
  cam.imgs.reserve(m_mono ? 1 : 2);
  while (!m_fin_img.eof()) {
    double ts = -1;
    m_fin_img >> ts;
    m_cap.read(m_cap_img);
    if (ts < 0 || m_cap_img.empty()) return false;
    if (ts >= m_start_ts) {
      cam.ts = ts;
      break;
    }
  }
  if (cam.ts < 0) return false;
  cv::cvtColor(m_cap_img, m_cap_img, cv::COLOR_BGR2GRAY);
  if (m_split_lr) {
    int cols = m_cap_img.cols / 2;
    cam.imgs.push_back(m_cap_img.colRange(0, cols));
    if (!m_mono) cam.imgs.push_back(m_cap_img.colRange(cols, m_cap_img.cols));
  } else {
    int rows = m_cap_img.rows / 2;
    cam.imgs.push_back(m_cap_img.rowRange(0, rows));
    if (!m_mono) cam.imgs.push_back(m_cap_img.rowRange(rows, m_cap_img.rows));
  }
  return true;
}

bool VioDataLoaderVsCommon::readNextGtPose(PoseData& gt_pose) {
  gt_pose.ts = -1;
  std::string line;
  while (getline(m_fin_imu, line)) {
    if (line.length() < 1 || line[0] == '#') continue;  // skip comment line and empty line
    SAFE_SSCANF(line.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf", &gt_pose.ts, &gt_pose.tx, &gt_pose.ty, &gt_pose.tz,
                &gt_pose.qw, &gt_pose.qx, &gt_pose.qy, &gt_pose.qz);
    if (gt_pose.ts < 0)
      return false;
    else if (gt_pose.ts >= m_start_ts)
      return true;
  }
  return false;
}

class VioDataLoaderXRing : public VioDataLoaderEuroc {
 public:
  VioDataLoaderXRing() {
    m_dataset_type = DATASET_XRING;
    m_fimu = "mav0/imu0/data.csv";
    m_fimg = "mav0/cam0/data.csv";
    m_fgt = "apriltag_pose.txt";
  }

 protected:
  virtual void parseImgLine(const char* line_str, ImgFileInfo& a) {
    m_mono = true;
    VioDataLoaderEuroc::parseImgLine(line_str, a);
  }

  virtual void parseImuLine(const char* line_str, ImuData& imu) {
    SAFE_SSCANF(line_str, "%lf %lf %lf %lf %lf %lf %lf", &imu.ts, &imu.acc[0], &imu.acc[1], &imu.acc[2], &imu.gyro[0],
                &imu.gyro[1], &imu.gyro[2]);
    imu.ts /= 1e6;
  }

  virtual void parseGtLine(const char* line_str, PoseData& a) {
    SAFE_SSCANF(line_str, "%lf %lf %lf %lf %lf %lf %lf %lf", &a.ts, &a.tx, &a.ty, &a.tz, &a.qx, &a.qy, &a.qz, &a.qw);
    a.ts /= 1e9;
  }
};

std::shared_ptr<VioDataLoader> createVioDataLoader(VioDataLoader::DatasetType dataset_type) {
  switch (dataset_type) {
    case VioDataLoader::DATASET_EUROC:
      return std::make_shared<VioDataLoaderEuroc>();
    case VioDataLoader::DATASET_KITTI:
      return std::make_shared<VioDataLoaderKitti>();
    case VioDataLoader::DATASET_UZH_VIO:
      return std::make_shared<VioDataLoaderUzhFpv>();
    case VioDataLoader::DATASET_TUM_VIO:
      return std::make_shared<VioDataLoaderTumVi>();
    case VioDataLoader::DATASET_ZJU_VIO:
      return std::make_shared<VioDataLoaderZjuViSlam>();
    case VioDataLoader::DATASET_VS_COMMON:
      return std::make_shared<VioDataLoaderVsCommon>();
    case VioDataLoader::DATASET_XRING:
      return std::make_shared<VioDataLoaderXRing>();
    default:
      break;
  }
  return nullptr;
}

std::shared_ptr<VioDataLoader> createVioDataLoader(const std::string& data_dir, VioDataLoader::DatasetType dataset_type,
                                                   const std::vector<std::string>& append_files) {
  if (dataset_type == VioDataLoader::DATASET_UNKNOW) {
    // deduced datatype
    std::shared_ptr<VioDataLoader> ptr;
    ptr = std::make_shared<VioDataLoaderEuroc>();
    if (ptr->init(data_dir, append_files)) return ptr;
    ptr = std::make_shared<VioDataLoaderKitti>();
    if (ptr->init(data_dir, append_files)) return ptr;
    ptr = std::make_shared<VioDataLoaderUzhFpv>();
    if (ptr->init(data_dir, append_files)) return ptr;
    ptr = std::make_shared<VioDataLoaderTumVi>();
    if (ptr->init(data_dir, append_files)) return ptr;
    ptr = std::make_shared<VioDataLoaderZjuViSlam>();
    if (ptr->init(data_dir, append_files)) return ptr;
    ptr = std::make_shared<VioDataLoaderVsCommon>();
    if (ptr->init(data_dir, append_files)) return ptr;
  } else {
    auto ptr = createVioDataLoader(dataset_type);
    if (ptr.get()) ptr->init(data_dir, append_files);
    return ptr;
  }
  return nullptr;
}

}  // namespace vs
