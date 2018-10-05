#include <fstream>
#include <viola/vs_basic.h>
#include <viola/vs_imu.h>
#include <viola/vs_plot.h>

struct State {
  double ts = 0;
  Eigen::Vector3d p, v, bg, ba;
  Eigen::Quaterniond q;
};

std::vector<State> datas2states(const std::vector<std::vector<double>>& datas) {
  std::vector<State> res;
  for (const auto& data : datas) {
    if (VS_INT(data.size()) < 17) continue;
    State state;
    state.ts = data[0] * VS_NS_TO_SEC;
    state.p << data[1], data[2], data[3];
    state.q.coeffs() << data[5], data[6], data[7], data[4];
    state.v << data[8], data[9], data[10];
    state.bg << data[11], data[12], data[13];
    state.ba << data[14], data[15], data[16];
    res.push_back(state);
  }
  return res;
}

std::vector<std::vector<double>> states2datas(const std::vector<State>& states) {
  std::vector<std::vector<double>> res;
  for (const auto& s : states) {
    std::vector<double> data = {s.ts * VS_SEC_TO_NS,
                                s.p(0),
                                s.p(1),
                                s.p(2),
                                s.q.w(),
                                s.q.x(),
                                s.q.y(),
                                s.q.z(),
                                s.v(0),
                                s.v(1),
                                s.v(2),
                                s.bg(0),
                                s.bg(1),
                                s.bg(2),
                                s.ba(0),
                                s.ba(1),
                                s.ba(2)};
    res.push_back(data);
  }
  return res;
}

std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& a) {
  if (a.empty()) return {};
  int h = a.size();
  int w = a[0].size();
  std::vector<std::vector<double>> res;
  res.resize(w);
  for (auto& p : res) p.resize(h);
  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++) res[j][i] = a[i][j];
  return res;
}

void processFile(const char* fimu, const char* fgt, int start_index = 0, double time_gap = 1) {
  auto imu_datas = vs::loadFileData(fimu, ',');
  auto gt_datas = vs::loadFileData(fgt, ',');
  if (imu_datas.empty() || gt_datas.empty()) return;
  std::vector<State> gt_state_list = datas2states(gt_datas);

  double last_reset_ts = 0;
  State estimate_state;
  std::vector<State> estimate_state_list;
  int gt_idx = 0;
  Eigen::Vector3d gravity(0, 0, -9.81);
  for (size_t imu_idx = start_index; imu_idx < imu_datas.size(); imu_idx++) {
    const auto& imu_data = imu_datas[imu_idx];
    double imu_ts = imu_data[0] * VS_NS_TO_SEC;
    Eigen::Vector3d gyro(imu_data[1], imu_data[2], imu_data[3]);
    Eigen::Vector3d acc(imu_data[4], imu_data[5], imu_data[6]);
    // find letest gt data before imu_ts, unable interpolation
    while (gt_idx < VS_INT(gt_state_list.size()) && gt_state_list[gt_idx].ts < imu_ts) gt_idx++;
    if (gt_idx == 0 || gt_idx >= VS_INT(gt_state_list.size())) continue;
    const auto& gt_state = gt_state_list[gt_idx - 1];

    bool need_reset = (last_reset_ts == 0) || (gt_state.ts - last_reset_ts > time_gap);
    if (need_reset) {
      last_reset_ts = gt_state.ts;
      estimate_state = gt_state;
    }
    double dt = imu_ts - estimate_state.ts;
    if (dt > 0) {
      vs::imuPredict(dt, gyro - estimate_state.bg, acc - estimate_state.ba, gravity, estimate_state.q, estimate_state.p,
                     estimate_state.v);
      estimate_state.ts = imu_ts;
    }
    estimate_state_list.push_back(estimate_state);
  }

  vs::Plot plt;
  auto plt_data_estimate = transpose(states2datas(estimate_state_list));
  auto plt_data_gt = transpose(gt_datas);
  std::vector<std::string> tags = {"px", "py", "pz",  "qw",  "qx",  "qy",  "qz",  "vx",
                                   "vy", "vz", "bgx", "bgy", "bgz", "bax", "bay", "baz"};
  for (int i = 0; i < 16; i++) {
    plt.subplot(4, 4, i + 1);
    plt.plot(plt_data_gt[0], plt_data_gt[i + 1], "r");
    plt.plot(plt_data_estimate[0], plt_data_estimate[i + 1], "b");
    plt.title(tags[i]);
  }
  cv::imshow("state", plt.show(cv::Size(1800, 1200)));
  cv::waitKey();
}

int main(int argc, char** argv) {
  if (argc < 3) return -1;
  int start_index = 0;
  double time_gap = 1;
  if (argc > 3) start_index = atoi(argv[3]);
  if (argc > 4) time_gap = atof(argv[4]);
  processFile(argv[1], argv[2], start_index, time_gap);
}