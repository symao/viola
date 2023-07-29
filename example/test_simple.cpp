/**
 * Copyright (c) 2019-2021 shuyuanmao <maoshuyuan123 at gmail dot com>. All rights reserved.
 * @Author: shuyuanmao
 * @EMail: shuyuanmao123@gmail.com
 * @Date: 2019-05-19 02:07
 * @Description:
 */
#include <viola/viola.h>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

#define printHeader()                                    \
  do {                                                   \
    printf("============= %s ============\n", __func__); \
  } while (0)

using namespace vs;

void print(float* a, int n) {
  for (int i = 0; i < n; i++) std::cout << a[i] << " ";
  std::cout << std::endl;
}

void testypr() {
  // R << -0.535446, -0.42925, 0.727353, -0.844511, 0.262012, -0.467066, 0.00991299, -0.864346,
  // -0.502799; R << -0.46628, -0.490853, 0.735966, -0.883848, 0.223353, -0.411007, 0.0373635,
  // -0.842126, -0.537984; R << -0.799181, -0.0029138, 0.601083, -0.512334, 0.526283, -0.678631,
  // -0.314363, -0.850304, -0.422088;
  Eigen::Matrix3d R_c_b;
  R_c_b << -0.6081574621585628, -0.4210112961753338, 0.6729739888833909, -0.7936767583288918, 0.3065860409203642,
      -0.5254352508171864, 0.01488974510861129, -0.8536711826022607, -0.5205994693476541;
  std::cout << "rpy: " << Rbw2rpy<double>(R_c_b * typicalRot<double>(ROT_FLU2RDF)).transpose() * 57.29578 << std::endl;
  // auto ypr = R.eulerAngles(2, 1, 0);
  // std::cout << "ypr: " << ypr.transpose()*57.29578 << std::endl;
  // auto R2 = (Eigen::AngleAxisd(ypr(0), Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(ypr(1),
  // Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(ypr(2),
  // Eigen::Vector3d::UnitX())).toRotationMatrix(); std::cout << "R: " << R << std::endl;
  // std::cout << "R2: " << R2 << std::endl;

  // auto Ryz = (Eigen::AngleAxisd(ypr(0), Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(ypr(1),
  // Eigen::Vector3d::UnitY())).toRotationMatrix();
}

void test1() {
  printHeader();
  float e[3] = {0.1, -0.2, -0.32};
  float r[9];
  float q[4];
  float e_1[3];
  float e_2[3];
  float e_3[3];
  euler2rot(e, r);
  rot2euler(r, e_1);
  euler2quat(e, q);
  quat2euler(q, e_2);
  euler2quat(e, q);
  quat2rot(q, r);
  rot2euler(r, e_3);
  print(e_1, 3);
  print(e_2, 3);
  print(e_3, 3);
}

void test2() {
  printHeader();
  float e[3] = {0.1, -0.2, 0.3};
  float r[9];
  euler2rot(e, r);
  print(r, 9);
  rot2euler(r, e);
  print(e, 3);
}

void testAlias() {
  printHeader();
  std::vector<int> weights = {1, 2, 3, 4};
  // std::vector<float> weights = {0.3,0.4,0.1,0.2};
  std::vector<float> probs;
  std::vector<int> alias;
  aliasTable(weights, probs, alias);
  for (size_t i = 0; i < probs.size(); i++) {
    printf("%f %d\n", probs[i], alias[i]);
  }
}

void testShuffle() {
  std::vector<int> a;
  for (int i = 0; i < 100; i++) a.push_back(i);
  shuffleInPlace(a);
  for (auto i : a) printf("%d ", i);
  printf("\n");
}

void testWeightedSample() {
  std::vector<float> weights = {1, 2, 3, 4, 5, 6};
  for (int method = 0; method < 3; method++) {
    for (int k = 0; k < 6; k++) {
      auto res = weightedSample(weights, k, method);
      printf("sample %d with method %d:", k, method);
      for (auto i : res) printf("%d ", i);
      printf("\n");
    }
  }
}

void testRaycast2D() {
  printHeader();
  int start[] = {0, -3};
  int end[] = {-5, -1};

  auto foo = [](int x, int y) {
    printf("(%d %d)\n", x, y);
    return true;
  };
  raycast2D(start, end, foo);
}

void testRaycast3D() {
  printHeader();
  int start[] = {-7, 0, -3};
  int end[] = {2, -5, -1};
  std::vector<std::vector<int>> out;
  auto foo = [](int x, int y, int z) {
    printf("(%d %d %d)\n", x, y, z);
    return true;
  };
  raycast3D(start, end, foo);
  printf("========================\n");
  raycast3D(end, start, foo);
}

void testStereoRectifier() {
  printHeader();
  StereoRectifier re;
  if (!re.init("../data/calib_fisheye.yaml")) {
    printf("[ERROR]StereoRectifier:Failed init.\n");
    return;
  }
  std::cout << "Calib raw:" << re.getCalibRaw() << std::endl << std::endl;
  std::cout << "Calib rectify:" << re.getCalibRectify() << std::endl << std::endl;
}

void testDataSaver() {
  printHeader();
  struct MyData {
    float x, y, z;
    int d;
  };

  DataSaver<MyData> saver("save.txt",
                          [](FILE* fp, const MyData& a) { fprintf(fp, "%f %.3f %.3f %d\n", a.x, a.y, a.z, a.d); });

  for (int i = 0; i < 100; i++) {
    MyData a;
    a.d = i;
    a.x = sin(i);
    a.y = cos(i);
    a.z = tan(i);
    saver.push(a);
  }
  msleep(5000);
}

void testLineMatch() {
  printHeader();
  LineSeg2DList lines = {LineSeg2D(0, 0, 1, 0), LineSeg2D(0.5, 0.2, 0.7, 0.9), LineSeg2D(1, 2, 3, 4),
                         LineSeg2D(1, 1, 1, 2), LineSeg2D(0, 1, 2, 1.1)};

  cv::Point3f motion(0.2, -0.3, 0.0);
  Eigen::Matrix3d T_gt;
  T_gt << cos(motion.z), -sin(motion.z), motion.x, sin(motion.z), cos(motion.z), motion.y, 0, 0, 1;
  Eigen::Matrix3d Tinv = T_gt.inverse();

  double sigma = 0.005;
  LineSeg2DList target;
  for (const auto& l : lines) {
    Eigen::Vector3d v1 = Tinv * Eigen::Vector3d(l.p1.x, l.p1.y, 1);
    Eigen::Vector3d v2 = Tinv * Eigen::Vector3d(l.p2.x, l.p2.y, 1);
    if (randf(1) > 0.5) std::swap(v1, v2);
    target.push_back(
        LineSeg2D(v1(0) + randn(0, sigma), v1(1) + randn(0, sigma), v2(0) + randn(0, sigma), v2(1) + randn(0, sigma)));
  }

  {
    cv::Point3f T;
    cv::Mat info;
    bool ok = solveTransform(lines, target, T, info);
    printf("solve: %d (%.3f %.3f %.3f)\n", ok, T.x, T.y, T.z);
    std::cout << "info" << info << std::endl;
  }

  std::reverse(target.begin(), target.end());
  target.erase(target.begin());

  cv::Point3f T2;
  bool ok2 = ICL(lines, target, T2, 0);
  printf("ICL: %d (%.3f %.3f %.3f)\n", ok2, T2.x, T2.y, T2.z);

  cv::Point3f T3;
  ICLSolver solver(lines, 1);
  bool ok3 = solver.match(target, T3, 0);
  printf("ICLSolver: %d (%.3f %.3f %.3f)\n", ok3, T3.x, T3.y, T3.z);
  std::cout << "Info:" << solver.getInfo() << std::endl;
}

void testLineMatch2() {
  printHeader();
  // blackbox test between ICL and ICLSolver
  for (int ite = 0; ite < 50000; ite++) {
    auto foo_rand_line = []() {
      while (1) {
        auto r = randDoubleVec(-100, 100, 4);
        if (r[0] != r[2] || r[1] != r[3]) return LineSeg2D(r[0], r[1], r[2], r[3]);
      }
    };
    int N = randi(1, 10);
    LineSeg2DList lines;
    lines.reserve(N);
    for (int i = 0; i < N; i++) {
      lines.push_back(foo_rand_line());
    }

    cv::Point3f motion(randf(-0.5, 0.5), randf(-0.5, 0.5), randf(-0.29, 0.29));
    Eigen::Matrix3d T_gt;
    T_gt << cos(motion.z), -sin(motion.z), motion.x, sin(motion.z), cos(motion.z), motion.y, 0, 0, 1;
    Eigen::Matrix3d Tinv = T_gt.inverse();

    double sigma = 0.005;
    LineSeg2DList target;
    for (const auto& l : lines) {
      Eigen::Vector3d v1 = Tinv * Eigen::Vector3d(l.p1.x, l.p1.y, 1);
      Eigen::Vector3d v2 = Tinv * Eigen::Vector3d(l.p2.x, l.p2.y, 1);
      if (randf(1) > 0.5) std::swap(v1, v2);
      target.push_back(LineSeg2D(v1(0) + randn(0, sigma), v1(1) + randn(0, sigma), v2(0) + randn(0, sigma),
                                 v2(1) + randn(0, sigma)));
    }

    std::reverse(target.begin(), target.end());
    target.erase(target.begin());

    cv::Point3f T2;
    bool ok2 = ICL(lines, target, T2, 0);

    cv::Point3f T3;
    ICLSolver solver(lines, 1);
    bool ok3 = solver.match(target, T3, 0);

    cv::Point3f diff = T3 - T2;
    if (cv::norm(diff) > 1e-4) {
      printf("ite:%d Diff %d (%.3f %.3f %.3f) : %d (%.3f %.3f %.3f)\n", ite, ok2 ? 1 : 0, T2.x, T2.y, T2.z, ok3 ? 1 : 0,
             T3.x, T3.y, T3.z);

      printf("%f %f %f %f\n", lines[0].theta, lines[0].d, target[0].theta, target[0].d);
      getchar();
    }
  }
}

void testCamOdomCalib() {
  printHeader();
  using namespace Eigen;
  Isometry3d T_c_o = Isometry3d::Identity();
  // double ypr[3]{1.005730, -1.9, 2.097663};
  std::vector<std::vector<double>> ypr_test_samples{{1.00573, 0.5, 1.04393}, {1.005730, -1.9, 2.097663},
                                                    {1.00573, -3, 1.04393},  {1.00573, 3, 1.04393},
                                                    {0.2, 3.14, -2},         {3.14, -3.14, 3.14}};
  int success_cnt = 0;
  for (size_t i = 0; i < ypr_test_samples.size(); i++) {
    auto& ypr = ypr_test_samples[i];
    T_c_o.linear() = (AngleAxisd(ypr[0], Vector3d::UnitZ()) * AngleAxisd(ypr[1], Vector3d::UnitY()) *
                      AngleAxisd(ypr[2], Vector3d::UnitX()))
                         .toRotationMatrix();
    T_c_o.translation() << 0.5, -0.01, 0.25;

    Isometry3d T_m = Isometry3d::Identity();
    T_m.linear() = AngleAxisd(randDouble(-10, 10), Vector3d::UnitZ()).toRotationMatrix();
    T_m.translation() << 0.5, 0.4, 0;

    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> cam_poses, odom_poses;
    Isometry3d T_o = Isometry3d::Identity();
    for (double t = -1; t <= 1; t += 0.02) {
      T_o.translation() << t, t, 0;
      T_o.linear() = AngleAxisd(t, Vector3d::UnitZ()).toRotationMatrix();
      Isometry3d T_c_m = T_m.inverse() * T_o * T_c_o;
      odom_poses.push_back(T_o);
      cam_poses.push_back(T_c_m);
    }
    T_o.translation() << 0.1, 0.2, 0;
    for (double t = -1.0; t <= 1.0; t += 0.02) {
      T_o.linear() = AngleAxisd(t, Vector3d::UnitZ()).toRotationMatrix();
      Isometry3d T_c_m = T_m.inverse() * T_o * T_c_o;
      odom_poses.push_back(T_o);
      cam_poses.push_back(T_c_m);
    }

    Isometry3d T_calib = Eigen::Isometry3d::Identity();
    bool ok = camOdomCalib(cam_poses, odom_poses, T_calib, 1);
    // std::cout<<"Real T_c_o:"<<std::endl<<T_c_o.matrix()<<std::endl<<std::endl;
    // std::cout<<"Calib T_c_o:"<<ok<<std::endl<<T_calib.matrix()<<std::endl<<std::endl;
    // std::cout<<"Diff:"<<std::endl<<(T_calib.inverse()*T_c_o).matrix()<<std::endl<<std::endl;
    // std::cout << "T_calib.inverse() * T_c_o).matrix().norm(): " << (T_calib.inverse() *
    // T_c_o).matrix().norm() << std::endl;
    bool success = ok ? (((T_calib.inverse() * T_c_o).matrix() - Eigen::Matrix4d::Identity()).norm() < 1e-6) : false;
    if (success)
      success_cnt++;
    else {
      // std::cout << "test sample: " << ypr_test_samples[i] << "failed." << std::endl;
    }
  }
  std::cout << "test    samples numbers: " << ypr_test_samples.size() << std::endl;
  std::cout << "success samples numbers: " << success_cnt << std::endl;
}

void testatan2() {
  std::cout << "atan2(1,1) : " << atan2(1, 1) << std::endl;
  std::cout << "atan2(-1,1) : " << atan2(-1, 1) << std::endl;
  std::cout << "atan2(1,-1) : " << atan2(1, -1) << std::endl;
  std::cout << "atan2(-1,-1) : " << atan2(-1, -1) << std::endl;
}

void testColor(int argc, char** argv) {
  printHeader();
  if (argc < 2) return;

  cv::Mat K = (cv::Mat_<double>(3, 3) << 347.907932735926, 0, 316.6024902247042, 0, 348.4394426469446,
               236.8952741402495, 0, 0, 1);
  cv::Mat D = (cv::Mat_<double>(1, 4) << -0.024463666926774732, -0.015081763908611457, -0.0015079423491373898,
               0.002447934595710685);
#if 1
  cv::VideoCapture cap(argv[1]);
  if (!cap.isOpened()) return;
  cv::Mat img;
  int start_idx = 0;
  for (int i = 0; cap.read(img); i++) {
    while (i < start_idx) continue;
      // cv::Mat img2;
      // cv::fisheye::undistortImage(img, img2, K, D, K);
#else
  for (const auto& f : listdir(argv[1], 1)) {
    cv::Mat img = cv::imread(f);
#endif
    cv::medianBlur(img, img, 5);
    cv::Mat mask;
    colorFilter(img, mask, {ColorModel::yellow()}, 0.5);
    // std::vector<cv::Mat> bgr;
    // cv::split(img, bgr);
    // bgr[2] += mask;
    // cv::Mat img2;
    // cv::merge(bgr, img2);
    cv::imshow("img", img);
    cv::imshow("mask", mask);
    static bool halt = true;
    uchar key = cv::waitKey(halt ? 0 : 10);
    if (key == 27)
      break;
    else if (key == 's')
      halt = !halt;
  }
}

void testColorHist() {
  using namespace cv;
  // std::vector<cv::Mat> imgs = {cv::imread("/home/symao/Pictures/2DMarker/rmb/1-1999a.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/1-1999b.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/1-2019a.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/1-2019b.jpg")};

  // std::vector<cv::Mat> imgs = {cv::imread("/home/symao/Pictures/2DMarker/rmb/5-1999a.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/5-1999b.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/5-2005a.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/5-2005b.jpg")};

  // std::vector<cv::Mat> imgs = {cv::imread("/home/symao/Pictures/2DMarker/rmb/10-1999a.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/10-1999b.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/10-2005a.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/10-2005b.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/10-2019a.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/10-2019b.jpg")};

  // std::vector<cv::Mat> imgs = {cv::imread("/home/symao/Pictures/2DMarker/rmb/20-1999a.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/20-1999b.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/20-2005a.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/20-2005b.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/20-2019a.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/20-2019b.jpg")};

  // std::vector<cv::Mat> imgs = {cv::imread("/home/symao/Pictures/2DMarker/rmb/50-1999a.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/50-1999b.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/50-2005a.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/50-2005b.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/50-2019a.jpg"),
  //                              cv::imread("/home/symao/Pictures/2DMarker/rmb/50-2019b.jpg")};

  std::vector<cv::Mat> imgs = {cv::imread("/home/symao/Pictures/2DMarker/rmb/100-1999a.jpg"),
                               cv::imread("/home/symao/Pictures/2DMarker/rmb/100-1999b.jpg"),
                               cv::imread("/home/symao/Pictures/2DMarker/rmb/100-2005a.jpg"),
                               cv::imread("/home/symao/Pictures/2DMarker/rmb/100-2005b.jpg"),
                               cv::imread("/home/symao/Pictures/2DMarker/rmb/100-2015a.jpg"),
                               cv::imread("/home/symao/Pictures/2DMarker/rmb/100-2015b.jpg")};

  std::vector<cv::Mat> hsvs;
  hsvs.reserve(imgs.size());
  for (const auto& img : imgs) {
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    hsvs.push_back(hsv);
  }

  int channels = 0;
  int histsize[] = {256};
  float midranges[] = {0, 255};
  const float* ranges[] = {midranges};
  MatND dsthist;
  calcHist(&hsvs[0], hsvs.size(), &channels, Mat(), dsthist, 1, histsize, ranges, true, false);
  Mat b_drawImage = Mat::zeros(Size(256, 256), CV_8UC3);
  double g_dhistmaxvalue;
  minMaxLoc(dsthist, 0, &g_dhistmaxvalue, 0, 0);
  for (int i = 0; i < 256; i++) {
    // 这里的dsthist.at<float>(i)就是每个bins对应的纵轴的高度
    int value = cvRound(256 * 0.9 * (dsthist.at<float>(i) / g_dhistmaxvalue));
    line(b_drawImage, Point(i, b_drawImage.rows - 1), Point(i, b_drawImage.rows - 1 - value), Scalar(255, 0, 0));
  }

  channels = 1;
  calcHist(&hsvs[0], hsvs.size(), &channels, Mat(), dsthist, 1, histsize, ranges, true, false);
  Mat g_drawImage = Mat::zeros(Size(256, 256), CV_8UC3);
  for (int i = 0; i < 256; i++) {
    int value = cvRound(256 * 0.9 * (dsthist.at<float>(i) / g_dhistmaxvalue));
    line(g_drawImage, Point(i, g_drawImage.rows - 1), Point(i, g_drawImage.rows - 1 - value), Scalar(0, 255, 0));
  }

  channels = 2;
  calcHist(&hsvs[0], hsvs.size(), &channels, Mat(), dsthist, 1, histsize, ranges, true, false);
  Mat r_drawImage = Mat::zeros(Size(256, 256), CV_8UC3);
  for (int i = 0; i < 256; i++) {
    int value = cvRound(256 * 0.9 * (dsthist.at<float>(i) / g_dhistmaxvalue));
    line(r_drawImage, Point(i, r_drawImage.rows - 1), Point(i, r_drawImage.rows - 1 - value), Scalar(0, 0, 255));
  }

  // add(b_drawImage, g_drawImage, r_drawImage);   //将三个直方图叠在一块

  cv::Mat img_show;
  cv::vconcat(b_drawImage, g_drawImage, img_show);
  cv::vconcat(img_show, r_drawImage, img_show);

  cv::imshow("img", img_show);
  cv::waitKey();
}

bool rangeCheck(uchar h, uchar hmin, uchar hmax) {
  if (hmin < hmax)
    return hmin <= h && h <= hmax;
  else
    return hmin <= h || h <= hmax;
}

bool multiRangeCheck(uchar h, uchar hmin, uchar hmax, uchar hmin2, uchar hmax2) {
  return rangeCheck(h, hmin, hmax) || ((hmin2 != 0 || hmax2 != 0) && rangeCheck(h, hmin2, hmax2));
}

void colorMask(const cv::Mat& hsv, cv::Mat& mask, uchar hmin, uchar hmax, uchar smin, uchar smax, uchar vmin,
               uchar vmax, uchar hmin2 = 0, uchar hmax2 = 0, uchar smin2 = 0, uchar smax2 = 0) {
  mask = cv::Mat(hsv.size(), CV_8UC1, cv::Scalar(0));
  uchar* ptr_hsv = hsv.data;
  uchar* ptr_mask = mask.data;
  for (int i = 0; i < hsv.rows * hsv.cols; i++) {
    uchar h = *ptr_hsv++;
    uchar s = *ptr_hsv++;
    uchar v = *ptr_hsv++;
    *ptr_mask++ = (multiRangeCheck(h, hmin, hmax, hmin2, hmax2) && multiRangeCheck(s, smin, smax, smin2, smax2) &&
                   rangeCheck(v, vmin, vmax))
                      ? 255
                      : 0;
  }
}

void testColor2(int argc, char** argv) {
  printHeader();
  if (argc < 2) return;
  auto img_list = vs::listdir(argv[1], 1);
  std::sort(img_list.begin(), img_list.end());

  int hmin = 99;
  int hmax = 135;
  int smin = 15;
  int smax = 137;
  int vmin = 87;
  int vmax = 255;
  int hmin2 = 0;
  int hmax2 = 0;
  int smin2 = 0;
  int smax2 = 0;
  const char* winname = "test_color2";
  cv::namedWindow(winname);
  cv::createTrackbar("hmin", winname, &hmin, 180);
  cv::createTrackbar("hmax", winname, &hmax, 180);
  cv::createTrackbar("smin", winname, &smin, 255);
  cv::createTrackbar("smax", winname, &smax, 255);
  cv::createTrackbar("vmin", winname, &vmin, 255);
  cv::createTrackbar("vmax", winname, &vmax, 255);
  cv::createTrackbar("hmin2", winname, &hmin2, 180);
  cv::createTrackbar("hmax2", winname, &hmax2, 180);
  cv::createTrackbar("smin2", winname, &smin2, 255);
  cv::createTrackbar("smax2", winname, &smax2, 255);
  int idx = 0;
  while (1) {
    const auto& fimg = img_list[idx];
    cv::Mat img = cv::imread(fimg);
    int height = 200;
    int width = height * img.cols / img.rows;
    cv::resize(img, img, cv::Size(width, height));
    cv::blur(img, img, cv::Size(31, 31));

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    colorMask(hsv, mask, hmin, hmax, smin, smax, vmin, vmax, hmin2, hmax2, smin2, smax2);

    cv::Mat img_show;
    cv::vconcat(img, hsv, img_show);

    cv::Mat gray = mask;
    cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR);
    cv::vconcat(img_show, gray, img_show);

    cv::imshow(winname, img_show);
    char key = cv::waitKey(30);
    if (key == 27)
      break;
    else if (key == 'p')
      idx = std::max(0, idx - 1);
    else if (key == 13 || key == 32)
      idx = std::min(idx + 1, (int)img_list.size() - 1);
  }
}

void testColor3(int argc, char** argv) {
  printHeader();
  if (argc < 2) return;
  auto img_list = vs::listdir(argv[1], 1);
  std::sort(img_list.begin(), img_list.end());

  int idx = 0;
  while (1) {
    const auto& fimg = img_list[idx];
    cv::Mat img = cv::imread(fimg);
    int height = 50;
    int width = height * img.cols / img.rows;
    cv::resize(img, img, cv::Size(width, height));

    cv::Mat mask[6];
    colorFilter(img, mask[0], {ColorModel::red()}, 0.5);
    colorFilter(img, mask[1], {ColorModel::green()}, 0.5);
    colorFilter(img, mask[2], {ColorModel::blue()}, 0.5);
    colorFilter(img, mask[3], {ColorModel::yellow()}, 0.5);
    colorFilter(img, mask[4], {ColorModel::black()}, 0.5);
    colorFilter(img, mask[5], {ColorModel::white(), ColorModel::green()}, 0.5);

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    cv::Mat img_show;
    cv::vconcat(img, hsv, img_show);

    cv::Mat gray;
    cv::vconcat(mask[0], mask[1], gray);
    cv::vconcat(gray, mask[2], gray);
    cv::vconcat(gray, mask[3], gray);
    cv::vconcat(gray, mask[4], gray);
    cv::vconcat(gray, mask[5], gray);
    cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR);
    cv::vconcat(img_show, gray, img_show);

    for (int i = 1; i <= 7; i++)
      cv::line(img_show, cv::Point(0, i * img.rows), cv::Point(img.cols, i * img.rows), cv::Scalar(0, 0, 255), 3);

    cv::imshow("img", img_show);
    char key = cv::waitKey(30);
    if (key == 27)
      break;
    else if (key == 'p')
      idx = std::max(0, idx - 1);
    else if (key == 13 || key == 32)
      idx = std::min(idx + 1, (int)img_list.size() - 1);
  }
}

void testColorCam() {
  printHeader();

  cv::VideoCapture cap(0);
  cv::Mat img;
  while (cap.read(img)) {
    int height = 150;
    int width = height * img.cols / img.rows;
    cv::resize(img, img, cv::Size(width, height));

    cv::Mat mask[6];
    colorFilter(img, mask[0], {ColorModel::red()}, 0.5);
    colorFilter(img, mask[1], {ColorModel::green()}, 0.5);
    colorFilter(img, mask[2], {ColorModel::blue()}, 0.5);
    colorFilter(img, mask[3], {ColorModel::yellow()}, 0.5);
    colorFilter(img, mask[4], {ColorModel::black()}, 0.5);
    colorFilter(img, mask[5], {ColorModel::white(), ColorModel::green()}, 0.5);

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    cv::Mat img_show;
    cv::vconcat(img, hsv, img_show);

    cv::Mat gray;
    cv::vconcat(mask[0], mask[1], gray);
    cv::vconcat(gray, mask[2], gray);
    cv::vconcat(gray, mask[3], gray);
    cv::vconcat(gray, mask[4], gray);
    cv::vconcat(gray, mask[5], gray);
    cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR);
    cv::vconcat(img_show, gray, img_show);

    for (int i = 1; i <= 7; i++)
      cv::line(img_show, cv::Point(0, i * img.rows), cv::Point(img.cols, i * img.rows), cv::Scalar(0, 0, 255), 3);

    cv::imshow("img", img_show);
    char key = cv::waitKey(30);
    if (key == 27) break;
  }
}

void testLaneDetect(int argc, char** argv) {
  printHeader();
  if (argc < 2) return;

#if 0  // usb web camera taobao
    cv::Mat K = (cv::Mat_<double>(3, 3) << 347.907932735926, 0, 316.6024902247042,
                                           0, 348.4394426469446, 236.8952741402495,
                                           0, 0, 1);
    cv::Mat D = (cv::Mat_<double>(1, 4) << -0.024463666926774732, -0.015081763908611457,
                                         -0.0015079423491373898, 0.002447934595710685);
    cv::Mat K_new = (cv::Mat_<double>(3, 3) << 250, 0, 320,
                                           0, 250, 240,
                                           0, 0, 1);
    int roi_row = 20;
    cv::Mat K_roi = K_new.clone();
    K_roi.at<double>(1, 2) -= roi_row;
    cv::Mat T_c_b = (cv::Mat_<double>(4, 4) << -0.507232, -0.789457,  0.345649,  0.575,
                                              -0.797997 , 0.278785, -0.534303, -0.23,
                                              0.325447, -0.546842, -0.771393,  0.265,
                                               0.0, 0.0, 0.0, 1.0);
#endif
#if 1  // mindvision industrial camera
  cv::Mat K = (cv::Mat_<double>(3, 3) << 451.0356167313587, 0, 323.92094563577683, 0, 451.0540055780082,
               244.13028827466718, 0, 0, 1);
  cv::Mat D = (cv::Mat_<double>(1, 4) << -0.3792862607865829, 0.1333768262529482, -0.00046334666522035564,
               0.0038021330351276496);
  double f = K.at<double>(0, 0) * 0.9;
  cv::Mat K_new = (cv::Mat_<double>(3, 3) << f, 0, 320, 0, f, 240, 0, 0, 1);
  int roi_row = 0;
  cv::Mat K_roi = K_new.clone();
  K_roi.at<double>(1, 2) -= roi_row;
  cv::Mat T_c_b = (cv::Mat_<double>(4, 4) << -0.0119137, -0.562969, 0.826392, 0.595, -0.999923, 0.00964968, -0.00784175,
                   -0.036, -0.00355976, -0.826422, -0.56304, 0.515, 0.0, 0.0, 0.0, 1.0);
#endif

#if 0
    cv::Mat T_c_b = cv::Mat::eye(4, 4, CV_64FC1);
    double pitch = -deg2rad(70);//- atan2(240 - 60, 250);
    T_c_b(cv::Rect(0, 0, 3, 3)) = (cv::Mat_<double>(3, 3)<<cos(pitch), 0, sin(pitch),
                                    0, 1, 0, sin(pitch), 0, cos(pitch)) *
                                (cv::Mat_<double>(3, 3)<<0, 0, 1, -1, 0, 0, 0, -1, 0);
    T_c_b.at<double>(0, 3) = 0.4;
    T_c_b.at<double>(1, 3) = 0;
    T_c_b.at<double>(2, 3) = 0.3;
#endif

  cv::VideoCapture cap(argv[1]);
  if (!cap.isOpened()) return;
  cv::Mat img;
  int start_idx = 0;  // 3500;//13268;
  for (int i = 0; cap.read(img); i++) {
    printf("======%d=========\n", i);
    if (i < start_idx) continue;
#if 1
    cv::fisheye::undistortImage(img, img, K, D, K_new);
    img = img.rowRange(roi_row, img.rows);
#else
    cv::Mat timg;
    cv::undistort(img, timg, K, D, K_new);
    img = timg.rowRange(roi_row, img.rows);
#endif
    LaneList lanes;
    laneDetect(img, lanes, K_roi, T_c_b, 3, 0.05, 0.18, 1);
    static bool halt = true;
    uchar key = cv::waitKey(halt ? 0 : 10);
    if (key == 27)
      break;
    else if (key == 's')
      halt = !halt;
    else if (key == 'n')
      start_idx = i + 100;
  }
}

void testCamCapture() {
  CamCapture cap;
  cap.init(0, "out.avi");
  cap.start();
  msleep(1e3);
  double prev_ts = 0;
  cv::Mat img;
  while (1) {
    double ts = cap.getLatest(img);
    if (ts > prev_ts) {
      prev_ts = ts;
      cv::imshow("img", img);
      uchar key = cv::waitKey(10);
      if (key == 27) break;
    }
    msleep(5);
  }
}

void testUndistortImages() {
  cv::Mat K = (cv::Mat_<double>(3, 3) << 451.0356167313587, 0, 323.92094563577683, 0, 451.0540055780082,
               244.13028827466718, 0, 0, 1);
  cv::Mat D = (cv::Mat_<double>(1, 4) << -0.3792862607865829, 0.1333768262529482, -0.00046334666522035564,
               0.0038021330351276496);
  double f = K.at<double>(0, 0) * 0.9;
  cv::Mat K_new = (cv::Mat_<double>(3, 3) << f, 0, 320, 0, f, 240, 0, 0, 1);

  for (auto f : listdir("/home/symao/data/yellow_line/todo/", 1)) {
    cv::Mat img = cv::imread(f);
    cv::Mat timg;
    cv::undistort(img, timg, K, D, K_new);
    cv::imshow("img", timg);
    cv::imwrite(f, timg);
    cv::waitKey(0);
  }
}

void testTimeBuffer() {
  TimeBuffer<cv::Point2f> buf;
  // buf.add(0, cv::Point2f(0, -10));
  buf.add(0, cv::Point2f(0, 0));
  buf.add(1, cv::Point2f(1, -4));
  // buf.add(1, cv::Point2f(1, -3));
  // buf.add(1, cv::Point2f(1, -2));
  buf.add(2, cv::Point2f(2, -3));
  auto f = [&buf](double t) {
    cv::Point2f res(0, 0);
    bool flag = buf.get(t, res);
    printf("ts:%f find:%d res:(%f %f)\n", t, flag, res.x, res.y);
  };
  // auto f = [&buf](double t) {
  //   cv::Point2f res(0, 0);
  //   double tres;
  //   bool flag = buf.getNearest(t, res, tres);
  //   printf("ts:%f find:%d res:(%f %f) tres:%f\n", t, flag, res.x, res.y, tres);
  // };
  f(0);
  f(1);
  f(0.2);
  f(0.4);
  f(1.001);
  f(2.0000001);
  f(1.999);
  f(-1);

  TimeBuffer<double> angle_buf([](const double& a1, const double& a2, double t) {
    double a3 = normalizeRad(a2 - a1) + a1;
    return normalizeRad((1 - t) * a1 + t * a3);
  });
  angle_buf.add(0, -0.1);
  angle_buf.add(1, 0.1);
  angle_buf.add(2, 1.5);
  angle_buf.add(3, 3.141);
  angle_buf.add(4, -3.14);
  auto f2 = [&angle_buf](double t) {
    double res;
    bool flag = angle_buf.get(t, res);
    printf("ts:%f find:%d res:%f\n", t, flag, res);
  };
  f2(0);
  f2(0.1);
  f2(1.3);
  f2(2.4);
  f2(3.2);
  f2(3.6);
  f2(3.9);
  f2(4.1);
}

void testPCA() {
  Eigen::MatrixXd data(6, 2);
  data << 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;

  Eigen::VectorXd eig_val, center;
  Eigen::MatrixXd eig_coef;
  bool ok = PCA(data, eig_val, eig_coef, center);
  std::cout << "ok:" << ok << std::endl;
  std::cout << "eig_val:" << eig_val.transpose() << std::endl;
  std::cout << "eig_coef:" << eig_coef << std::endl;
  std::cout << "center:" << center.transpose() << std::endl;
}

void testLineFit() {
  Eigen::MatrixXd data(6, 3);
  data << 0, 1, 0, 1, 2, 0, 3, 4, 0, 5, 6, 0, 7, 8, 0, 9, 10, 0;

  Eigen::MatrixXd p1, p2;
  lineFit(data, p1, p2);

  std::cout << p1.transpose() << " | " << p2.transpose() << std::endl;
}

void testRandSample() {
  std::vector<int> vec;
  int N = 1000;
  for (int i = 0; i < N; i++) {
    vec.push_back(randi(5));
  }

  for (int idx = 0; idx < 100000; idx++) {
    std::map<int, int> hist;
    for (auto i : vec) hist[i]++;
    printf("#%d: ", idx);
    for (auto i : hist) printf("%d(%d) ", i.first, i.second);
    printf("\n");

    std::vector<double> weights;
    for (auto i : vec) weights.push_back(1000 + i);
    auto ids = weightedSample(weights, N, 2);
    std::vector<int> new_vec;
    for (auto i : ids) new_vec.push_back(vec[i]);
    vec = new_vec;
    msleep(10);
  }
}

void testKDTree() {
  auto frand = []() { return std::vector<float>({(float)randf(5000), (float)randf(5000), (float)randf(5000)}); };
  int N = 10000;
  std::vector<std::vector<float>> data;
  for (int i = 0; i < N; i++) {
    data.push_back(frand());
  }
  std::vector<float> query = frand();

  Timer t1;
  std::shared_ptr<KDTree> kdt = std::make_shared<KDTree>();
  kdt->build(data);
  t1.stop();

  Timer t2;
  int K = 500;
  KDTree::KvecArray res;
  for (int i = 0; i < K; i++) {
    kdt->knn(query, res, 10);
    if (i == 0) printf("knn: %d\n", (int)res.size());
  }
  t2.stop();

  Timer t3;
  for (int i = 0; i < K; i++) {
    kdt->rnn(query, res, 100);
    if (i == 0) printf("rnn: %d\n", (int)res.size());
  }
  t3.stop();

  printf("query:(%f %f %f)\n", query[0], query[1], query[2]);
  kdt->knn(query, res, 1);
  printf("nearest:(%f %f %f)\n", res[0][0], res[0][1], res[0][2]);

  printf(
      "Build %d data tree: %.1f ms,\n"
      "Ksearch %d times: %.1f ms,\n"
      "Rsearch %d times: %.1f ms\n",
      N, t1.getMsec(), K, t2.getMsec(), K, t3.getMsec());
}

std::vector<double> foo() {
  std::vector<double> a = {1, 2, 3};
  a.reserve(100000);
  return a;
}

void testMaxQueue() {
  MaxQueue<int> fifo;
  auto print = [](const MaxQueue<int>& fifo) {
    printf("queue:[");
    for (auto i : fifo.data()) printf("%d ", i);
    printf("]  ");
    printf("max:%d\n", fifo.max());
  };
  std::vector<int> samples = {1, 3, 2, 5, 5, 4, 2};
  for (auto i : samples) {
    fifo.push(i);
    print(fifo);
  }
  for (size_t i = 0; i <= samples.size(); i++) {
    fifo.pop();
    print(fifo);
  }
}

void testMaxHeap() {
  MaxHeap<int> queue;
  std::vector<int> samples = {1, 3, 2, 5, 5, 4, 2, 1, 2};
  for (auto i : samples) queue.push(i);
  for (size_t i = 0; i <= samples.size(); i++) {
    int a = queue.pop();
    printf("%d ", a);
  }
  printf("\n");
}

void testMaxStack() {
  MaxStack<int> filo;
  auto print = [](const MaxStack<int>& filo) {
    printf("stack:[");
    for (auto i : filo.data()) printf("%d ", i);
    printf("]  ");
    printf("max:%d\n", filo.max());
  };
  std::vector<int> samples = {1, 3, 2, 5, 5, 4, 2};
  for (auto i : samples) {
    filo.push(i);
    print(filo);
  }
  for (size_t i = 0; i <= samples.size(); i++) {
    filo.pop();
    print(filo);
  }
}

void testAlign() {
  cv::Mat img = cv::imread("/home/symao/Pictures/2DMarker/wangzhe/marker_xuance.jpg", cv::IMREAD_GRAYSCALE);
  cv::resize(img, img, cv::Size(), 0.5, 0.5);

  std::vector<cv::Point2f> pts_src = {cv::Point2f(0, 0), cv::Point2f(0, img.rows), cv::Point2f(img.cols, img.rows),
                                      cv::Point2f(img.cols, 0)};
  std::vector<cv::Point2f> pts_tar = {cv::Point2f(img.cols * 0.1, img.rows * 0.05),
                                      cv::Point2f(img.cols * 0.05, img.rows * 0.85),
                                      cv::Point2f(img.cols * 0.9, img.rows * 1.1), cv::Point2f(img.cols * 1.05, 0)};
  cv::Mat H = cv::getPerspectiveTransform(pts_src, pts_tar);

  cv::Mat img_tar;
  cv::warpPerspective(img, img_tar, H, img.size());
  cv::add(img_tar, 20, img_tar);

  cv::imshow("img", img);
  cv::imshow("img_tar", img_tar);
  // cv::waitKey();

  while (1) {
    cv::Point2f tar_pt;
    vs::affineMatch(img, img_tar, H, cv::Point2f(img.cols * randf(0.3, 0.7), img.rows * randf(0.3, 0.7)), tar_pt,
                    cv::Size(11, 11), cv::Size(), cv::TM_CCOEFF_NORMED);
  }
}

void testImageRoter() {
  printHeader();

  cv::Mat img(360, 640, CV_8UC3, cv::Scalar(128, 20, 180));
  cv::putText(img, "ABC", cv::Point(30, 300), cv::FONT_HERSHEY_SIMPLEX, 10, cv::Scalar(0, 0, 255), 5);
  cv::imshow("img", img);

  for (int i = -360; i < 360; i++) {
    FastImageRoter roter(img.size(), vs::deg2rad(i));
    cv::Mat img_rot, img_back;
    img.copyTo(img_rot);
    roter.rot(img_rot, img_rot);
    roter.rotBack(img_rot, img_back);
#if 1
    cv::Point2f pt(50, 200);
    cv::Point2f pt_rot = roter.rot(pt.x, pt.y);
    cv::Point2f pt_rot_back = roter.rotBack(pt_rot.x, pt_rot.y);
    cv::circle(img_rot, pt_rot, 5, cv::Scalar(255, 125, 0), 3);
    cv::circle(img_back, pt_rot_back, 8, cv::Scalar(0, 125, 255), 3);
    // printf("(%.1f %.1f)->(%.1f %.1f)->(%.1f %.1f)", pt.x, pt.y, pt_rot.x, pt_rot.y,
    // pt_rot_back.x, pt_rot_back.y);
    cv::Rect rect(50, 100, 200, 100);
    auto rect_rot = roter.rot(rect);
    auto rect_rot_back = roter.rotBack(rect_rot);
    cv::rectangle(img_rot, rect_rot, cv::Scalar(0, 255, 125), 3);
    cv::rectangle(img_back, rect_rot_back, cv::Scalar(255, 125, 0), 5);
    printf("(%d %d %d %d)->(%d %d %d %d)->(%d %d %d %d)", rect.x, rect.y, rect.width, rect.height, rect_rot.x,
           rect_rot.y, rect_rot.width, rect_rot.height, rect_rot_back.x, rect_rot_back.y, rect_rot_back.width,
           rect_rot_back.height);
#endif
    cv::imshow("rot", img_rot);
    cv::imshow("back", img_back);

    ImageRoter roter2(img.size(), vs::deg2rad(i));
    cv::Mat img_rot2, img_back2;
    roter2.rot(img, img_rot2);
    roter2.rotBack(img_rot2, img_back2);
    cv::imshow("rot2", img_rot2);
    cv::imshow("back2", img_back2);

    printf("%d [%dx%d] == [%dx%d]\n", i, img_rot.cols, img_rot.rows, roter.rotSize().width, roter.rotSize().height);
    uchar key = cv::waitKey();
    if (key == 27) break;
  }
  cv::destroyAllWindows();
}

class BoxDetector : public BoxDetectApi {
 public:
  BoxDetector(const cv::Rect& rect) {
    box.id = 0;
    box.class_id = 0;
    box.xmin = rect.x;
    box.ymin = rect.y;
    box.xmax = rect.x + rect.width;
    box.ymax = rect.y + rect.height;
    box.score = 1;
  }

  virtual void detect(const cv::Mat& img, std::vector<vs::TrackBox>& result, float thres_conf, float thres_iou) {
    result = {box};
  }

  TrackBox box;
};

void testBoxTracking() {
  cv::VideoCapture cap("/home/symao/Desktop/6977694507191622394.MP4");
  cv::Mat img;
  cap.read(img);
  if (img.empty()) return;
  cv::Rect box = cv::selectROI(img);
  printf("(%d, %d, %d, %d)\n", box.x, box.y, box.width, box.height);
  // cv::Rect box(23, 223, 520, 531);

  vs::FpsCalculator fps;
  std::shared_ptr<BoxTrackerAsync> tracker;
  if (!tracker.get()) {
    std::shared_ptr<vs::BoxDetectApi> detector = std::make_shared<BoxDetector>(box);
    if (detector.get()) {
      tracker = std::make_shared<vs::BoxTrackerAsync>();
      auto cfg = tracker->getConfig();
      cfg.drop_truncated = false;
      cfg.debug_draw = true;
      tracker->setConfig(cfg);
      tracker->init(detector, 10000.0f);
    }
  }
  fps.start();
  for (int idx = 0; cap.read(img); idx++) {
    printf("process %d, [%dx%d]\n", idx, img.cols, img.rows);
    fps.start();
    auto boxes = tracker->process(img);
    fps.stop();
    auto debug_img = tracker->getDebugImg();
    cv::imshow("img", debug_img);
    static bool halt = true;
    auto key = cv::waitKey(halt ? 0 : 10);
    if (key == 27)
      break;
    else if (key == 's')
      halt = !halt;
  }
  fps.stop();
}

void testLineFit2D() {
  cv::Mat img(300, 300, CV_8UC3, cv::Scalar(0));
  cv::Point2f p1(50, 50);
  cv::Point2f p2(250, 203);
  std::vector<cv::Point2f> pts;
  int npts = lineDiscreteSample(p1, p2, pts, 10);
  for (auto& p : pts) {
    p.x += randn(2);
    p.y += randn(2);
  }
  int idx = randi(npts);
  pts[idx] += cv::Point2f(10, -40);

  cv::Vec4f line = lineSegFit(pts);

  std::vector<uint8_t> status;
  cv::Vec4f line2;
  bool ok = lineDetectRansac(pts, line2, status);

  printf("npts:%d  line:(%.1f %.1f) (%.1f %.1f)  ok:%d line2:(%.1f %.1f) (%.1f %.1f) %d\n", npts, line[0], line[1],
         line[2], line[3], ok, line2[0], line2[1], line2[2], line2[3], (int)status.size());

  cv::circle(img, p1, 3, cv::Scalar(0, 0, 255), 2);
  cv::circle(img, p2, 3, cv::Scalar(0, 0, 255), 2);
  for (const auto& p : pts) {
    cv::circle(img, p, 2, cv::Scalar(0, 150, 50), -1);
  }
  if (ok) {
    cv::line(img, cv::Point(line2[0], line2[1]), cv::Point(line2[2], line2[3]), cv::Scalar(0, 100, 255), 1);
    for (size_t i = 0; i < status.size(); i++) {
      if (status[i] == 0) {
        printf("i:%d %d\n", (int)i, idx);
        cv::circle(img, pts[i], 2, cv::Scalar(255, 50, 50), -1);
      }
    }
  }
  cv::line(img, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255, 100, 0), 1);
  cv::imshow("show", img);
  cv::waitKey();
}

void testLineOpticalFlowLK() {
  cv::VideoCapture cap("/home/symao/Videos/VID_20210715_142213.mp4");
  cv::Mat img;
  cv::Mat prev_img;
  std::vector<cv::Vec4f> prev_lines;
  for (int idx = 0; cap.read(img); idx++) {
    if (idx % 5 != 0) continue;
    cv::resize(img, img, cv::Size(), 0.5, 0.5);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    std::vector<cv::Vec4f> lines;
    vs::detectLine(img, lines);
    size_t j = 0;
    for (size_t i = 0; i < lines.size(); i++) {
      const auto& l = lines[i];
      if (hypotf(l[2] - l[0], l[3] - l[1]) > 30) {
        lines[j++] = l;
      }
    }
    lines.resize(j);
    if (!prev_lines.empty() && !prev_img.empty()) {
      std::vector<cv::Vec4f> track_lines;
      std::vector<uint8_t> track_status;
      lineOpticalFlowLK(prev_img, img, prev_lines, track_lines, track_status, 10, true, cv::Size(13, 13), 3, true,
                        true);
    }

    prev_lines = lines;
    prev_img = img;
    auto key = cv::waitKey();
    if (key == 27) break;
  }
}

void testVecBasic() {
  std::vector<int> a = {1, 2, 3, 4, 0, 6, 7, 8, 0};
  auto foo = [](const int& x) { return -x; };
  printf("%d %d %d %d %d %d %d\n", vs::vecMax(a), vs::vecMin(a), vs::vecArgMax(a), vs::vecArgMin(a),
         vs::vecArgMax(a, foo), vs::vecArgMin(a, foo), vs::vecCount(a));
}

void testVecBasic2() {
  int rows = 8;
  int cols = 6;
  int n = rows * cols;
  auto data = randFloatVec(0, 10, n);
  printf("mat:\n");
  for (int i = 0; i < n; i++) {
    printf("%.3f ", data[i]);
    if ((i + 1) % cols == 0) printf("\n");
  }
  printf("max matches:");
  auto pairs = matRowMax(&data[0], rows, cols, 0.0f, true);
  for (auto it : pairs) {
    printf("(%d,%d) ", it.first, it.second);
  }
  printf("\n");
  printf("min matches:");
  pairs = matRowMin(&data[0], rows, cols, 100.0f, true);
  for (auto it : pairs) {
    printf("(%d,%d) ", it.first, it.second);
  }
  printf("\n");
}

void testArgparse(int argc, char** argv) {
  ArgParser parser;
  parser.add<std::string>("host", 'h', "host name", true, "");
  parser.add<float>("float", 'f', "float value", false, 0.123f);
  parser.add<int>("int", 'i', "int value", false, 456);
  parser.add<bool>("bool", 'b', "bool value", false, true);
  parser.add<std::string>("list", 'l', "list", false, "");
  parser.add("viz", '\0', "ciz");
  parser.parse_check(argc, argv);

  printf("%d %d %d %d %d %d\n", parser.exist("host"), parser.exist("float"), parser.exist("int"), parser.exist("bool"),
         parser.exist("list"), parser.exist("viz"));
  printf("%s %d  %f %d\n", parser.get<std::string>("host").c_str(), (int)parser.rest().size(),
         parser.get<float>("float"), parser.get<int>("int"));
}

void testMatSave() {
  std::vector<int> types = {CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_16SC1, CV_16SC2, CV_16SC3,
                            CV_32FC1, CV_32FC2, CV_32FC3, CV_64FC1, CV_64FC3};
  const char* file = "./mat.bin";
  for (int type : types) {
    int r = randi(50, 300);
    int c = randi(50, 300);
    cv::Mat m(r, c, type);
    bool ok = writeMatBin(file, m);
    cv::Mat m2 = readMatBin(file);
    printf("m:(%dx%d type:%x channel:%d elem:%d) m2:(%dx%d type:%x channel:%d elem:%d) save:%d  read_err:%.3f\n",
           m.rows, m.cols, m.type(), m.channels(), (int)m.elemSize(), m2.rows, m2.cols, m2.type(), m2.channels(),
           (int)m2.elemSize(), ok, cv::norm(m2 - m));
  }
}

void testMonoDepthData() {
  const char* save_dir = "/home/symao/open_ws/dso/build/depth_dso";
  for (int i = 141;; i++) {
    char f_img[256] = {0};
    char f_depth[256] = {0};
    snprintf(f_img, 256, "%s/%06d.jpg", save_dir, i);
    snprintf(f_depth, 256, "%s/%06d.bin", save_dir, i);
    cv::Mat img = cv::imread(f_img);
    cv::Mat depth = vs::readMatBin(f_depth);
    if (img.empty() || depth.empty()) break;

    if (1) {
      // show 3d
      static Viz3D viz;
      MeshData mesh;
      for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
          float d = depth.at<float>(i, j);
          if (d > 0.01 && d < 5) {
            cv::Vec3b rgb = img.at<cv::Vec3b>(i, j);
            std::swap(rgb[0], rgb[2]);
            float x = (j - 313.105) / 494.1246 * d;
            float y = (i - 240.6499) / 494.1246 * d;
            float z = d;
            mesh.vertices.push_back(cv::Point3f(x, y, z));
            mesh.colors.push_back(rgb);
          }
        }
      }
      viz.updateWidget("cloud", mesh.toVizCloud());
      mesh.writePly("a.ply");
    }

    cv::imshow("depth", drawSparseDepth(img, depth, 5));
    auto key = cv::waitKey(0);
    if (key == 27) break;
  }
}

void testFeatureDetect() {
  cv::Mat img = vs::imread("../data/feature.png", cv::IMREAD_GRAYSCALE);
  auto pts1 = featureDetect(img, 200, 0, 0.001, 10, cv::Mat(), cv::Rect(100, 50, 250, 250));
  auto pts2 = featureDetectUniform(img, cv::Size(4, 3), 200);
  cv::Mat img_draw1 = drawPoints(img, pts1, 2, cv::Scalar(0, 0, 255), -1);
  cv::Mat img_draw2 = drawPoints(img, pts2, 2, cv::Scalar(0, 0, 255), -1);
  cv::imshow("show", hstack({img_draw1, img_draw2}));
  cv::waitKey();
}

void testDataRecorder() {
  DataRecorder recorder("./log", DataRecorder::SUBDIR_DATE);
  printf("save:%s\n", recorder.getSaveDir().c_str());

  int id1 = recorder.createStringRecorder("data.txt");
  int id2 = recorder.createImageRecorder("video.mp4", 30, cv::Size(360, 640));
  // int id2 = recorder.createImageRecorder("%06d.png", DataRecorder::SAVE_IN_IMAGE);
  recorder.recordString(id1, "adsdasd\n");
  recorder.recordString(id1, "adsdasdsasa\n");
  recorder.recordString(id2, "adsdasdasssss\n");
  cv::Scalar color(123, 45, 67);
  for (int i = 0; i < 100; i++) {
    color[0]++;
    color[1]++;
    color[2]++;
    recorder.recordImage(id2, cv::Mat(640, 360, CV_8UC3, color));
  }
  msleep(5e3);
}

void testSearchSorted() {
  // std::vector<float> a = {-1, 0, 1, 1, 1.1, 2, 2.2, 3, 6, 6.5};
  // std::vector<float> b = {0, 0, 1, 1, 2, 3, 4, 5, 6};
  // std::vector<float> a = {0, 0, 0, 0.1, 0.2, 0.5, 0.9, 1.0, 1.0, 1.0};
  // std::vector<float> b = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<float> a = {0, 0, 0, 0.1, 0.2, 0.5, 0.9, 1.0, 1.0, 1.0};
  std::vector<float> b = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  auto lut = searchsorted(b, a);
  for (auto i : lut) printf("%d ", i);
  return;
}

void testHistMatching() {
  cv::Mat img_bg(100, 100, CV_8UC3,
                 cv::Scalar(0, 0, 255));  // = cv::imread("/home/symao/data/virtual_production/bg/1.jpg");
  cv::Mat img_fg = cv::imread("/home/symao/data/virtual_production/portrait/000000.png");

  HistogramMatch hm(img_bg);
  for (float k = 0; k < 1; k += 0.1) {
    cv::Mat res_img = hm.adjust(img_fg, k);
    cv::imshow("img", res_img);
    cv::waitKey(0);
  }
}

void testTriangulation() {
  Eigen::Isometry3d T0 = Eigen::Isometry3d::Identity();
  T0.translation() << 10000, 200000, 300;
  Eigen::Vector3d pw = T0 * Eigen::Vector3d(1, -2, 3);
  const double noise_sigma_dr = 0.01;
  const double noise_sigma_dt = 0.1;
  const double noise_uv = 0.01;
  auto project = [noise_uv](const Eigen::Isometry3d& T, Eigen::Vector3d& pw) {
    auto pc = T.inverse() * pw;
    if (pc.z() < 1e-4) printf("[ERROR]negative z:%.3f\n", pc.z());
    return Eigen::Vector2d(randn(pc.x() / pc.z(), noise_uv), randn(pc.y() / pc.z(), noise_uv));
  };

  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> cam_poses = {T0};
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> uvs = {project(T0, pw)};

  for (float dx = 0.1; dx <= 2; dx += 0.01) {
    Eigen::Matrix3d dR = (Eigen::AngleAxisd(randn(0, noise_sigma_dr), Eigen::Vector3d::UnitX()) *
                          Eigen::AngleAxisd(randn(0, noise_sigma_dr), Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(randn(0, noise_sigma_dr), Eigen::Vector3d::UnitZ()))
                             .toRotationMatrix();
    Eigen::Vector3d dt(dx, randn(0, noise_sigma_dt), randn(0, noise_sigma_dt));
    Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();
    T1.linear() = dR * T0.linear();
    T1.translation() = T0.translation() + dt;
    cam_poses.push_back(T1);
    uvs.push_back(project(T1, pw));
  }

  Eigen::Vector3d pos_two_view, pos_multi_view, pos_linear_solver;
  // 1. triangulate with two view
  triangulateTwoView(cam_poses[0], uvs[0], cam_poses.back(), uvs.back(), pos_two_view);
  // 2. triangulate with multiple view
  triangulateMultiViews(cam_poses, uvs, pos_multi_view);
  // 3. triangulate with linear solver
  LinearTriangulator<double> lt;
  for (size_t i = 0; i < cam_poses.size(); i++) {
    lt.update(uvs[i].x(), uvs[i].y(), cam_poses[i]);
  }
  if (lt.good()) pos_linear_solver = lt.pos();

  printf("Gt:(%.4f %.4f %.4f) two_view:(%.4f %.4f %.4f) multi_view:(%.4f %.4f %.4f) linear_solver:(%.4f %.4f %.4f)\n",
         pw.x(), pw.y(), pw.z(), pos_two_view.x(), pos_two_view.y(), pos_two_view.z(), pos_multi_view.x(),
         pos_multi_view.y(), pos_multi_view.z(), pos_linear_solver.x(), pos_linear_solver.y(), pos_linear_solver.z());
}

void testGuidedFilter() {
  const char* fvideo = "/home/symao/workspace/camera_track/rsc/2021_12_08_1424243290_mask.mp4";
  cv::VideoCapture cap(fvideo);
  double resize_k = 0.33333333333;
  cv::Mat img;
  for (int idx = 0; cap.read(img); idx++) {
    cv::resize(img, img, cv::Size(), resize_k, resize_k);
    int w = img.cols / 2;
    cv::Mat bgr = img.colRange(0, w);
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat mask;
    cv::cvtColor(img.colRange(w, w + w), mask, cv::COLOR_BGR2GRAY);
    mask.convertTo(mask, CV_32FC1, 1.0 / 255);

    cv::Mat mask1 = guidedFilter(gray, mask, 7, 1e-4);
    cv::Mat mask2 = mask.clone();
    ImageTransformer::guidedFilter(7, 1e-4)->process(mask2, gray);

    printf("%f\n", cv::norm(mask1, mask2));
    cv::imshow("mask", vs::hstack({mask, mask1, mask2}));
    auto key = cv::waitKey();
    if (key == 27) break;
  }
}

void testImmerge() {
  cv::Mat fg_img = cv::imread("/home/symao/workspace/vs_common//data/messi.png");
  cv::Mat fg_mask = cv::imread("/home/symao/workspace/vs_common//data/messi_mask.png", cv::IMREAD_GRAYSCALE);
  cv::Mat bg_img = cv::imread("/home/symao/workspace/vs_common/data/snow.png");

  bg_img = resizeMaxLength(bg_img, 960);
  imageComposition(bg_img, {fg_img, fg_img}, {fg_mask, fg_mask});

  cv::imshow("img", bg_img);
  cv::waitKey();
  cv::destroyAllWindows();
}

void testConvexToEllipse() {
  cv::Mat img(300, 300, CV_8UC3, cv::Scalar(0, 0, 0));
  std::vector<cv::Point> convex = {{10, 10}, {270, 30}, {180, 290}, {20, 270}};
  cv::Point2f center;
  cv::Vec2f radius;
  float angle;
  bool ok = quadToEllipse(convex[0], convex[1], convex[2], convex[3], center, radius, angle);
  printf("ok:%d center:(%.2f %.2f) radius:(%.2f %.2f) angle:%.2f\n", ok, center.x, center.y, radius[0], radius[1],
         angle);

  cv::polylines(img, convex, true, cv::Scalar(0, 0, 255), 2);
  cv::ellipse(img, center, cv::Size(radius[0], radius[1]), vs::rad2deg(angle), 0, 360, cv::Scalar(0, 255, 120), 2);
  cv::imshow("img", img);
  cv::waitKey();
}

void testPanorama() {
  cv::Mat img = cv::imread("/home/symao/workspace/vs_common/data/panorama.jpg");
  for (float theta = 0; theta < 360; theta += 10) {
    // for (float phi = 0; phi < 360; phi += 10) {
    float phi = 0;
    cv::Mat view = equirectangular2perspective(img, deg2rad(theta), deg2rad(phi), deg2rad(120), cv::Size(480, 640));
    cv::imshow("view", view);
    uchar key = cv::waitKey();
    if (key == 27) return;
    // }
  }
}

void testMaskRoi() {
  {
    cv::Mat mask(100, 100, CV_8UC1, cv::Scalar(0));
    std::cout << maskRoi(mask) << std::endl;
    mask.rowRange(50, 51).setTo(1);
    std::cout << maskRoi(mask) << std::endl;
    mask.colRange(10, 20).setTo(2);
    std::cout << maskRoi(mask) << std::endl;
    mask.rowRange(10, 20).setTo(3);
    std::cout << maskRoi(mask) << std::endl;
    mask.setTo(255);
    std::cout << maskRoi(mask) << std::endl;
    printf("=========================\n");
  }
  {
    cv::Mat mask(100, 100, CV_32FC1, cv::Scalar(0));
    std::cout << maskRoi(mask) << std::endl;
    mask.rowRange(50, 51).setTo(1);
    std::cout << maskRoi(mask) << std::endl;
    mask.colRange(10, 20).setTo(2);
    std::cout << maskRoi(mask) << std::endl;
    mask.rowRange(10, 20).setTo(3);
    std::cout << maskRoi(mask) << std::endl;
    mask.setTo(255);
    std::cout << maskRoi(mask) << std::endl;
    printf("=========================\n");
  }

  cv::Mat mask(5000, 5000, CV_8UC1, cv::Scalar(0));
  auto cmp = [](const cv::Mat& mask) {
    FpsCalculator fps1;
    FpsCalculator fps2;
    for (int i = 0; i < 10; i++) {
      fps1.start();
      auto rect1 = maskRoi(mask);
      fps1.start();
      fps2.start();
      std::vector<cv::Point> pts;
      cv::findNonZero(mask, pts);
      auto rect2 = cv::boundingRect(pts);
      fps2.stop();
      std::cout << rect1 << " ||||| " << rect2 << std::endl;
    }
    printf("%.3f %.3f\n", fps1.costms(), fps2.costms());
  };

  cmp(mask);
  mask.at<uchar>(3000, 3000) = 1;
  cmp(mask);
  mask(cv::Rect(1000, 1000, 500, 500)).setTo(100);
  cmp(mask);
  mask.setTo(255);
  cmp(mask);
}

void testLogger() {
  Logger logger(Logger::INFO, "TAG", "log.txt", true);
  logger.setTsType(Logger::TS_DATE);
  logger.error("this is error msg:%.4f %s", 1.222, "dsadsa");
  logger.debug("this is debug msg");
  logger.info("this is info msg:%d", 123);
  logger.warn("this is warn msg:%.4f %d", 1.23, 258);

  // // Logger logger(Logger::DEBUG, "TAG", "log.txt", true);
  // VS_LOG_INST->setTag("VS_COMMON");
  // std::string s1 = VS_LOG_INST->errorStr("this is error:%.4f %s", 1.222, "dsadsa");
  // VS_LOG_INST->setTsType(2);
  // std::string s2 = VS_LOG_INST->infoStr("this is info:%d", 123);

  // printf("(%s) (%s)\n", s1.c_str(), s2.c_str());
  // VS_LOG_ERROR("this is error:%.4f %s", 1.222, "dsadsa");
  // VS_LOG_INST->setTsType(2);
  // VS_LOG_INFO("dadasd");
  // VS_LOG_DEBUG("this is info:%d", 123);
  // VS_LOG_INST->setTsType(1);
  // VS_LOG_WARN("this is warn:%.4f %d", 1.23, 258);
}

void testImwb() {
  const char* img_path = "/home/symao/data/virtual_production/fg/image";
  const char* mask_path = "/home/symao/data/virtual_production/fg/mask";

  const char* winname = "img";
  cv::namedWindow(winname);
  int brightness = 100;
  int wb = 100;
  cv::createTrackbar("wb", winname, &wb, 100);
  cv::createTrackbar("brightness", winname, &brightness, 200);

  for (const auto& fimg : vs::listdir(img_path, false)) {
    cv::Mat img = cv::imread(vs::join(img_path, fimg));
    cv::Mat mask = cv::imread(vs::join(mask_path, fimg), cv::IMREAD_GRAYSCALE);
    cv::Mat roi_img, roi_mask;
    auto roi = vs::cropMaskRoi(img, mask, roi_img, roi_mask);

    while (1) {
      cv::Mat adjust_roi;
      vs::Timer t1;
      adjust_roi = whiteBalance(roi_img, cv::Mat(), wb / 100.0f);
      t1.stop();
      vs::Timer t2;
      adjust_roi = adjustBrightness(adjust_roi, brightness - 100);
      t2.stop();
      printf("(%dx%d) %.1f %.1f  %d %d\n", roi_img.cols, roi_img.rows, t1.getMsec(), t2.getMsec(), brightness, wb);
      cv::Mat img_show = img.clone();
      adjust_roi.copyTo(img_show(roi));
      cv::imshow(winname, vs::resizeMaxLength(vs::hstack({img, img_show}), 1000));
      auto key = cv::waitKey(10);
      if (key == 27) break;
    }

    auto key = cv::waitKey();
    if (key == 27) break;
  }
}

void testEigenIO() {
  const char* filename = "eigen.bin";
  Eigen::Matrix3d a = Eigen::Matrix3d::Random();
  writeEigenDense(filename, a);
  auto a2 = readEigenDense<double>(filename);
  printf("(%dx%d) (%dx%d) %f\n", (int)a.rows(), (int)a.cols(), (int)a2.rows(), (int)a2.cols(), (a2 - a).norm());

  Eigen::MatrixXf b(10, 5);
  for (int i = 0; i < b.rows(); i++)
    for (int j = 0; j < b.cols(); j++) b(i, j) = randf(30);
  writeEigenDense(filename, b);
  auto b2 = readEigenDense<float>(filename);
  printf("(%dx%d) (%dx%d) %f\n", (int)b.rows(), (int)b.cols(), (int)b2.rows(), (int)b2.cols(), (b2 - b).norm());

  Eigen::MatrixXd c(100, 200);
  c.setZero();
  for (int i = 10; i < 20; i++)
    for (int j = 10; j < 13; j++) c(i, j) = 1;
  for (int i = 50; i < 60; i++)
    for (int j = 110; j < 113; j++) c(i, j) = 1;
  Eigen::SparseMatrix<double> sparse_c = c.sparseView();
  writeEigenSparse(filename, sparse_c);
  auto sparse_c2 = readEigenSparse<double>(filename);
  printf("(%dx%d)%d (%dx%d)%d %f\n", (int)sparse_c.rows(), (int)sparse_c.cols(), (int)sparse_c.nonZeros(),
         (int)sparse_c2.rows(), (int)sparse_c2.cols(), (int)sparse_c2.nonZeros(),
         (sparse_c.toDense() - sparse_c2.toDense()).norm());
}

void testIsomInterp() {
  Eigen::Isometry3d pose1 = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d pose2 = Eigen::Isometry3d::Identity();
  pose2.translation() << 3, 4, 5;
  pose2.linear() = (Eigen::AngleAxisd(1.0, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitY()))
                       .toRotationMatrix();

  cv::Mat img(100, 100, CV_8UC1);
  Viz3D viz;
  std::vector<cv::Affine3f> show_pose = {isom2affine(pose1), isom2affine(pose2)};
  viz.updateWidget("traj", cv::viz::WTrajectory(show_pose, 3, 0.5, cv::viz::Color::green()));
  for (float k = 0; k < 1; k += 0.05) {
    auto cur_pose = isomLerp(pose1, pose2, k);
    std::vector<cv::Affine3f> cur = {isom2affine(cur_pose)};
    viz.updateWidget("cur_pose", cv::viz::WTrajectory(cur, 1, 0.5));
    cv::imshow("img", img);
    cv::waitKey();
  }
}

void testEncrypt() {
  std::string s = "dasdasg3g21gh15giagd adsdguSJF;AEWRTUE  *&%&^^&$^%*( SDADADS";
  std::string key = "vs_common";
  std::string encode_s = encrypt(s, key);
  std::string decode_s = decrypt(encode_s, key);
  printf("Input: [%s]\nKey:[%s]\n", s.c_str(), key.c_str());
  printf("Encode:[%s]\n", encode_s.c_str());
  printf("Decode:[%s]\n", decode_s.c_str());
  printf("%d\n", decode_s == s);

  std::string file = PROJECT_DIR "/data/general_seg160x160.rapidproto";
  std::string encode_file = "temp_encode.txt";
  std::string decode_file = "temp_decode.txt";
  std::string encode_file2 = "temp_encode2.txt";
  encryptFile(file, encode_file, key);
  decryptFile(encode_file, decode_file, key);
  encryptFile(decode_file, encode_file2, key);
  printf("%d %d\n", loadFileContent(file) == loadFileContent(decode_file),
         loadFileContent(encode_file) == loadFileContent(encode_file2));
}

void testLinspace() {
  auto a = vs::vecLinspace<double>(0, 10, 1);
  for (auto i : a) printf("%.3f ", i);
  printf("\n");
  a = vs::vecLinspace<double>(0, 10, 2);
  for (auto i : a) printf("%.3f ", i);
  printf("\n");
  a = vs::vecLinspace<double>(0, 10, 3);
  for (auto i : a) printf("%.3f ", i);
  printf("\n");
  a = vs::vecLinspace<double>(0, 10, 4);
  for (auto i : a) printf("%.3f ", i);
  printf("\n");
}

void testUmeyamaAlign() {
  std::vector<Eigen::Vector3d> src_pts;
  for (int i = 0; i < 10; i++) {
    src_pts.emplace_back(randf(10), randf(10), randf(10));
  }

  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() =
      (Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()))
          .toRotationMatrix();
  transform.translation() << 0.1, 0.2, 0.3;

  double gt_scale = 0.77;
  std::vector<Eigen::Vector3d> tar_pts;
  for (const auto& p : src_pts) tar_pts.push_back(transform * p * gt_scale);

  Eigen::Isometry3d estimate_transform;
  double scale = 0;
  bool ok = umeyamaAlign(src_pts, tar_pts, estimate_transform, &scale);

  printf("ok:%d\ngt_scale:%.5f  gt_pose:[%s]\nest_scale:%.5f est_pose:[%s]\n", ok, gt_scale,
         isom2str(transform, 1).c_str(), scale, isom2str(estimate_transform, 1).c_str());
}

void testTimelinePloter() {
  TimelinePloter tplot;
  for (double ts = 10; ts < 30; ts += 0.05) {
    tplot.arrived(0, ts);
    tplot.arrived(1, ts + 0.1);
    tplot.used(0, ts - 0.05);
    tplot.used(1, ts);
    cv::imshow("img", tplot.plot());
    static WaitKeyHandler handler;
    handler.waitKey();
  }
}

void testTqdm() {
  for (auto i : tqdm(range(100))) msleep(100);
}

int main(int argc, char** argv) {
  // test1();
  // test2();
  // testAlias();
  // testWeightedSample();
  // testShuffle();
  // testRaycast2D();
  // testStereoRectifier();
  // testDataSaver();
  // testLineMatch();
  // testLineMatch2();
  // testCamOdomCalib();
  // testColor(argc, argv);
  // testColor2(argc, argv);
  // testColor3(argc, argv);
  // testColorCam();
  // testColorHist();
  // testLaneDetect(argc, argv);
  // testCamCapture();
  // testUndistortImages();
  // testTimeBuffer();
  // testPCA();
  // testLineFit();
  // testSyslog();
  // testRandSample();
  // testKDTree();
  // testypr();
  // testatan2();
  // testMaxQueue();
  // testMaxHeap();
  // testAlign();
  // testImageRoter();
  // testFeatureDetect();
  // testBoxTracking();
  // testLineOpticalFlowLK();
  // testLineFit2D();
  // testVecBasic();
  // testVecBasic2();
  // testArgparse(argc, argv);
  // testMatSave();
  // testMonoDepthData();
  // testVioDataLoader();]
  // testDataRecorder();
  // testSearchSorted();
  // testHistMatching();
  // testTriangulation();
  // testGuidedFilter();
  // testImmerge();
  // testConvexToEllipse();
  // testPanorama();
  // testMaskRoi();
  // testLogger();
  // testImwb();
  // testEigenIO();
  // testIsomInterp();
  // testEncrypt();
  // testLinspace();
  // testUmeyamaAlign();
  // testTimelinePloter();
  // testTqdm();
}
