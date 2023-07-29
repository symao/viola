#include <viola/vs_gridmap.h>
#include <viola/vs_random.h>
#include <opencv2/opencv.hpp>

int main() {
  // create a 2D grid map with input range and resolution
  double xmin = 0, xmax = 10, ymin = -10, ymax = 10;
  double resolution = 0.1;
  std::vector<vs::GridMap2u::DimInfo> dim_infos = {vs::GridMap2u::DimInfo(xmin, xmax, resolution),
                                                   vs::GridMap2u::DimInfo(ymin, ymax, resolution)};
  vs::GridMap2u map(dim_infos);

  // add a value to whole map
  map += 128;

  // generate some random points, set the position in grid map to 255
  int pts_cnt = 100;
  for (int i = 0; i < pts_cnt; i++) {
    double x = vs::randf(xmin, xmax);
    double y = vs::randf(ymin, ymax);
    if (map.inside(x, y)) map.at(x, y) = 255;
  }

  // show grid map
  auto dims = map.dims();
  cv::Mat img_show(dims[0], dims[1], CV_8UC1, &(map.data()[0]));
  cv::imshow("img", img_show);
  cv::waitKey();
}