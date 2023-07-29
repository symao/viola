#include <iostream>
#include <opencv2/opencv.hpp>
#include <viola/vs_viz3d.h>
#include <viola/vs_json.h>

int main(int argc, char** argv) {
  vs::Json json;
  bool ok = vs::readJsonFile(argv[1], json);
  printf("ok:%d type:%d\n", ok, json.type());

  const auto& objs = json.object_items();
  cv::Point3f facial_root_pt;
  auto root_it = objs.find("DHIhead:FACIAL_C_FacialRoot");
  if (root_it != objs.end()) {
    facial_root_pt = cv::Point3f(root_it->second[0].number_value(), root_it->second[1].number_value(),
                                 root_it->second[2].number_value());
  }

  vs::Viz3D viz;
  viz.updateWidget("coord", cv::viz::WCoordinateSystem());
  viz.updateWidget("root", cv::viz::WSphere(facial_root_pt, 0.3, 10, cv::viz::Color::red()));

  for (const auto& it : json.object_items()) {
    cv::Point3f p(it.second[0].number_value(), it.second[1].number_value(), it.second[2].number_value());
    viz.updateWidget(it.first, cv::viz::WSphere(p, 0.1));
  }

  cv::Mat img(100, 100, CV_8UC1);
  cv::imshow("img", img);
  cv::waitKey();
}