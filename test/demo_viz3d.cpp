#include <viola/vs_viz3d.h>
#include <viola/vs_random.h>

int main() {
  vs::Viz3D viz;
  viz.updateWidget("coord", cv::viz::WCoordinateSystem(1));
  viz.updateWidget("sphere", cv::viz::WSphere(cv::Point3f(1, 2, 3), 0.2, 10, cv::viz::Color::red()));
  std::vector<cv::Point3f> cloud;
  for (int i = 0; i < 1000; i++) cloud.emplace_back(vs::randf(10), vs::randf(10), vs::randf(10));
  viz.updateWidget("cloud", cv::viz::WCloud(cloud, cv::viz::Color::green()));
  getchar();
}