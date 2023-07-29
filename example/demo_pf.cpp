#include <fstream>
#include <opencv2/opencv.hpp>
#include <viola/vs_pf.h>
#include <viola/vs_os.h>
#include <viola/vs_random.h>
#include <viola/vs_gridmap.h>
#include <viola/vs_yaml_parser.h>

void testSimplePf() {
  vs::ParticleFilter<cv::Point2f, float> pf;
  cv::Mat img(500, 500, CV_8UC3, cv::Scalar(0, 0));
  std::vector<cv::Point2f> particles;
  for (int i = 0; i < 5000; i++) particles.emplace_back(vs::randf(img.cols), vs::randf(img.rows));
  pf.setParticlesAndWeights(particles, std::vector<float>(particles.size(), 1));

  while (1) {
    auto particles = pf.getParticles();
    std::vector<float> new_weights;
    for (auto& p : particles) {
      p.x += vs::randn(0, 5);
      p.y += vs::randn(0, 5);
      // compute weight with particle distance to a circle center at(300,200) with radius 100
      float weight = 1.0 / (fabs(hypotf(p.x - 300, p.y - 200) - 100) + 10.0f);
      new_weights.push_back(weight);
    }
    pf.setWeights(new_weights);
    pf.resample(particles.size());

    // draw
    img.setTo(cv::Scalar(0, 0, 0));
    for (const auto& p : particles) cv::circle(img, p, 1, cv::Scalar(0, 0, 255), -1);
    cv::imshow("img", img);
    auto key = cv::waitKey(50);
    if (key == 27) break;
  }
}

vs::GridMap2d loadMap(const char* file) {
  vs::GridMap2d map;
  vs::YamlParser yml(file);
  if (yml.isOpened()) {
    auto fimg = yml.read<std::string>("image");
    auto reso = yml.read<double>("resolution");
    cv::Vec3d origin = yml.read<cv::Vec3d>("origin");
    cv::Mat pmap = cv::imread(vs::join(vs::dirname(file), fimg), cv::IMREAD_GRAYSCALE);
    if (!pmap.empty()) {
      std::vector<vs::GridMap2d::DimInfo> dim_info = {vs::GridMap2d::DimInfo(origin[1], reso, pmap.rows),
                                                      vs::GridMap2d::DimInfo(origin[0], reso, pmap.cols)};
      map.resize(dim_info);
      double* ptr_data = &map.data()[0];
      uchar* ptr_img = pmap.data;
      for (int i = 0; i < pmap.rows; i++)
        for (int j = 0; j < pmap.cols; j++) {
          *ptr_data++ = *ptr_img++ / 255.0f;
        }
    }
  }
  return map;
}

struct LaserFrame {
  int id;
  double ts;  ///< [seconds]
  float x, y, theta;
  std::vector<cv::Point2f> points;
};

bool load2dFile(const std::string& file, std::vector<LaserFrame>& datas, cv::Vec3f& laser_pose) {
  datas.clear();
  std::ifstream fin(file.c_str());
  if (!fin.is_open()) {
    printf("ERROR: Cannot open file '%s'.\n", file.c_str());
    return false;
  }
  double angmax = 0, angmin = 0, laser_cnt = 0;
  std::string temp_line;
  while (getline(fin, temp_line)) {
    temp_line.erase(0, temp_line.find_first_not_of(' '));  // strip
    if (vs::startswith(temp_line, "sick1pose: ")) {
      double a, b, c;
      sscanf(temp_line.c_str(), "sick1pose: %lf %lf %lf", &a, &b, &c);
      laser_pose = cv::Vec3f(a / 100.0f, b / 100.0f, vs::deg2rad(c));
    } else if (vs::startswith(temp_line, "sick1conf: ")) {
      double a, b, c;
      sscanf(temp_line.c_str(), "sick1conf: %lf %lf %lf", &a, &b, &c);
      angmin = vs::deg2rad(a);
      angmax = vs::deg2rad(b);
      laser_cnt = c;
    } else if (vs::startswith(temp_line, "scan1Id: ")) {
      LaserFrame frame;
      // get id
      sscanf(temp_line.c_str(), "scan1Id: %d", &frame.id);
      // get time
      getline(fin, temp_line);
      int hour, min, sec;
      sscanf(temp_line.c_str(), "time: %d:%d:%d", &hour, &min, &sec);
      frame.ts = ((hour * 60) + min) * 60 + sec;
      // get robot pose
      getline(fin, temp_line);
      double a, b, c;
      sscanf(temp_line.c_str(), "robot: %lf %lf %lf", &a, &b, &c);
      frame.x = a * 0.001;
      frame.y = b * 0.001;
      frame.theta = vs::deg2rad(c);
      // get laser scan
      getline(fin, temp_line);
      auto range_data = vs::str2vec(vs::cut(temp_line, "sick1: "));
      double angle = angmin;
      double angle_step = (angmax - angmin) / (laser_cnt - 1);
      for (double dist : range_data) {
        frame.points.emplace_back(dist * cos(angle) * 0.001, dist * sin(angle) * 0.001);
        angle += angle_step;
      }
      datas.push_back(frame);
    }
  }
  fin.close();
  return true;
}

void testMonteCarloLoc() {
  auto map = loadMap("/home/symao/temp/aaa/njnav/resource/nj_office_2015-8-19.map.yaml");

  std::vector<LaserFrame> laser_datas;
  cv::Vec3f laser_pose;
  if (!load2dFile("/home/symao/temp/aaa/njnav/resource/office.2d", laser_datas, laser_pose)) {
    return;
  }

  cv::Mat img_show(map.dims()[0], map.dims()[1], CV_64FC1, &map.data()[0]);
  cv::imshow("img", img_show);
  cv::waitKey();

  // vs::ParticleFilter<float, float> pf;
  // std::vector<cv::Vec3f> particles;
  // // for(int i = 0; i < )

}

int main() {
  // testSimplePf();
  testMonteCarloLoc();
}