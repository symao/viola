#include <viola/vs_vio_data_saver.h>
#include <viola/vs_vio_data_loader.h>
#include <viola/vs_debug_draw.h>

int main() {
  const char* data_dir = "/home/symao/data/euroc/zip/MH_04_difficult";
  const char* save_dir = "./temp_save";
  auto dataset = vs::createVioDataLoader(data_dir, vs::VioDataLoader::DATASET_EUROC);
  if (!(dataset && dataset->ok())) {
    printf("[ERROR]Dataset open failed '%s'\n", data_dir);
    return 0;
  }

  vs::VioDataSaver saver(save_dir);

  auto imu_callback = [&saver](const vs::ImuData& imu) {
    saver.pushImu(imu.ts, cv::Vec3f(imu.gyro[0], imu.gyro[1], imu.gyro[2]),
                  cv::Vec3f(imu.acc[0], imu.acc[1], imu.acc[2]));
  };

  auto cam_callback = [&saver](const vs::CameraData& cam) { saver.pushCamera(cam.ts, vs::vstack(cam.imgs)); };

  auto gt_callback = [&saver](const vs::PoseData& gt_pose) {
    saver.pushGtPose(gt_pose.ts, cv::Vec3f(gt_pose.tx, gt_pose.ty, gt_pose.tz),
                     cv::Vec4f(gt_pose.qw, gt_pose.qx, gt_pose.qy, gt_pose.qz));
  };

  dataset->play(imu_callback, cam_callback, gt_callback, true, false);
}