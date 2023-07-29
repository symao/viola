#include <deque>
#include <viola/vs_vio_data_loader.h>
#include <viola/vs_debug_draw.h>
#include <viola/vs_plot.h>

using namespace vs;

cv::Mat drawImuBuf(const std::deque<ImuData>& imu_buf, const cv::Size& img_size = cv::Size()) {
  std::vector<std::vector<double>> vec(7);
  for (auto& v : vec) v.reserve(imu_buf.size());
  for (const auto& imu : imu_buf) {
    vec[0].push_back(imu.ts);
    for (int i = 0; i < 3; i++) {
      vec[1 + i].push_back(imu.gyro[i]);
      vec[4 + i].push_back(imu.acc[i]);
    }
  }
  Plot plt;
  plt.subplot(211);
  plt.plot(vec[0], vec[1], "b", 1, "gx");
  plt.plot(vec[0], vec[2], "g", 1, "gy");
  plt.plot(vec[0], vec[3], "r", 1, "gz");
  plt.title("gyroscope");
  plt.subplot(212);
  plt.plot(vec[0], vec[4], "b", 1, "ax");
  plt.plot(vec[0], vec[5], "g", 1, "ay");
  plt.plot(vec[0], vec[6], "r", 1, "az");
  plt.title("accelemeter");
  plt.legend();
  plt.setRenderSize(img_size);
  return plt.render();
}

cv::Mat drawPoseBuf(const std::deque<PoseData>& pose_buf, const cv::Size& img_size = cv::Size()) {
  std::vector<std::vector<double>> vec(8);
  for (auto& v : vec) v.reserve(pose_buf.size());
  for (const auto& pose : pose_buf) {
    vec[0].push_back(pose.ts);
    vec[1].push_back(pose.tx);
    vec[2].push_back(pose.ty);
    vec[3].push_back(pose.tz);
    vec[4].push_back(pose.qw);
    vec[5].push_back(pose.qx);
    vec[6].push_back(pose.qy);
    vec[7].push_back(pose.qz);
  }
  Plot plt;
  plt.subplot(211);
  plt.plot(vec[0], vec[1], "b", 1, "tx");
  plt.plot(vec[0], vec[2], "g", 1, "ty");
  plt.plot(vec[0], vec[3], "r", 1, "tz");
  plt.title("gt_position");
  plt.subplot(212);
  plt.plot(vec[0], vec[5], "b", 1, "qx");
  plt.plot(vec[0], vec[6], "g", 1, "qy");
  plt.plot(vec[0], vec[7], "r", 1, "qz");
  plt.plot(vec[0], vec[4], "c", 1, "qw");
  plt.title("gt_queternion");
  plt.legend();
  plt.setRenderSize(img_size);
  return plt.render();
}

// one can specific dataset type, one can alse set to unknown type, which will be deduced inside
void playDataset(const char* data_dir, VioDataLoader::DatasetType type = VioDataLoader::DATASET_UNKNOW,
                 double start_ts = 0, bool mono_mode = false, const std::vector<std::string>& append_files = {}) {
  auto dataset = createVioDataLoader(data_dir, type, append_files);
  if (!(dataset && dataset->ok())) {
    printf("[ERROR]Dataset open failed '%s'\n", data_dir);
    return;
  }
  dataset->setStartTs(start_ts);
  dataset->setMonoMode(mono_mode);
  std::deque<ImuData> imu_buf;

  const bool draw_img = true;
  ImuData cur_imu;
  CameraData cur_cam;
  PoseData cur_gt_pose;
  bool ok = true;
  while (ok) {
    ok = false;
    switch (dataset->nextType()) {
      case VioDataLoader::MSG_IMU:
        ok = dataset->fetchImu(cur_imu, true);
        imu_buf.push_back(cur_imu);
        if (imu_buf.size() > 2000) imu_buf.pop_front();
        break;
      case VioDataLoader::MSG_CAMERA:
        ok = dataset->fetchCamera(cur_cam, true);
        if (draw_img) {
          cv::Mat img_show = toRgb(hstack(cur_cam.imgs));
          char str[256] = {0};
          snprintf(str, 256, "%s:%f", dataset->name().c_str(), cur_cam.ts);
          cv::putText(img_show, str, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 60, 180), 2);
          if (!imu_buf.empty()) {
            cv::Mat imu_show = drawImuBuf(imu_buf, cv::Size(img_show.cols, 300));
            cv::vconcat(img_show, imu_show, img_show);
          }
          cv::imshow("camera", img_show);
          static bool halt = false;
          auto key = cv::waitKey(halt ? 0 : 10);
          if (key == 27) {
            std::exit(EXIT_SUCCESS);
          } else if (key == 's' || key == 32) {
            halt = !halt;
          }
        }
        break;
      case VioDataLoader::MSG_GTPOSE:
        ok = dataset->fetchGtPose(cur_gt_pose, true);
      default:
        break;
    }
  }
}

// one can specific dataset type, one can alse set to unknown type, which will be deduced inside
void playDatasetWithCallback(const char* data_dir, VioDataLoader::DatasetType type = VioDataLoader::DATASET_UNKNOW,
                             double start_ts = 0, bool mono_mode = false,
                             const std::vector<std::string>& append_files = {}) {
  auto dataset = createVioDataLoader(data_dir, type, append_files);
  if (!(dataset && dataset->ok())) {
    printf("[ERROR]Dataset open failed '%s'\n", data_dir);
    return;
  }
  dataset->setStartTs(start_ts);
  dataset->setMonoMode(mono_mode);

  const bool draw_img = true;
  std::deque<ImuData> imu_buf;
  std::deque<PoseData> gt_buf;

  auto imu_callback = [&imu_buf](const ImuData& imu) {
    imu_buf.push_back(imu);
    if (imu_buf.size() > 2000) imu_buf.pop_front();
  };

  auto cam_callback = [draw_img, dataset, &imu_buf, &gt_buf](const CameraData& cam) {
    if (draw_img) {
      cv::Mat img_show = toRgb(hstack(cam.imgs));
      if (img_show.empty()) return;
      char str[256] = {0};
      snprintf(str, 256, "%s:%f", dataset->name().c_str(), cam.ts);
      cv::putText(img_show, str, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 60, 180), 2);
      if (!imu_buf.empty()) {
        cv::Mat imu_show = drawImuBuf(imu_buf, cv::Size(img_show.cols, 300));
        cv::vconcat(img_show, imu_show, img_show);
      }
      if (!gt_buf.empty()) {
        cv::Mat gt_show = drawPoseBuf(gt_buf, cv::Size(img_show.cols, 300));
        cv::vconcat(img_show, gt_show, img_show);
      }
      cv::imshow("camera", img_show);
      static bool halt = false;
      auto key = cv::waitKey(halt ? 0 : 10);
      if (key == 27) {
        std::exit(EXIT_SUCCESS);
      } else if (key == 's' || key == 32) {
        halt = !halt;
      }
    }
  };

  auto gt_callback = [&gt_buf](const PoseData& gt_pose) {
    gt_buf.push_back(gt_pose);
    if (gt_buf.size() > 2000) gt_buf.pop_front();
  };

  dataset->play(imu_callback, cam_callback, gt_callback, true, false);
}

int main(int argc, char** argv) {
  if (argc > 1) {
    playDatasetWithCallback(argv[1], VioDataLoader::DATASET_UNKNOW);
  } else {
    // read euroc dataset in mono mode
    playDataset("/home/symao/data/euroc/zip/MH_04_difficult", VioDataLoader::DATASET_EUROC, 0, true);
    // read euroc dataset without setting dataset type, deduce datatype inside
    playDataset("/home/symao/data/euroc/zip/MH_04_difficult");
    // read tum vio dataset
    playDataset("/media/symao/My Passport/data/VIO/tum_vio/dataset-corridor4_512_16", VioDataLoader::DATASET_TUM_VIO);
    // read uzh vio dataset without setting dataset type, deduce datatype inside
    playDataset("/media/symao/My Passport/data/VIO/uzh_vio/zip/outdoor_forward_9_snapdragon");
    // read zju vislam dataset without setting dataset type, deduce datatype inside
    playDataset("/media/symao/My Passport/data/VIO/zju_vislam/C0_train");
    // read kitti dataset, with setting ground truth file in append files
    playDataset("/media/symao/My Passport/data/SLAM/KITTI/odometry/dataset/sequences/00", VioDataLoader::DATASET_KITTI,
                0, false, {"/media/symao/My Passport/data/SLAM/KITTI/odometry/dataset/poses/00.txt"});
    // read own dataset
    playDataset("/media/symao/My Passport/data/VIO/sf_vio/AGV-西部兴围中转场/1/2019-08-16-14-56-30",
                VioDataLoader::DATASET_VIOLA);
    // run data set in callback mode
    playDatasetWithCallback("/home/symao/data/euroc/zip/MH_04_difficult", VioDataLoader::DATASET_EUROC);
  }
}