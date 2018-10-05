#include <math.h>
#include <gtest/gtest.h>
#include <viola/viola.h>

using namespace vs;

TEST(vs_basic, fequal) {
  EXPECT_TRUE(fequal(0.01, 0, 0.1));
  EXPECT_FALSE(fequal(0.01, 0));
  EXPECT_TRUE(fequal(0, 0));
}

TEST(vs_basic, sign) {
  EXPECT_EQ(sign(10), 1);
  EXPECT_EQ(sign(0), 1);
  EXPECT_EQ(sign(1), 1);
  EXPECT_EQ(sign(INT_MAX), 1);
  EXPECT_EQ(sign(-1), -1);
  EXPECT_EQ(sign(-10), -1);
}

TEST(vs_basic, inRange) {
  EXPECT_TRUE(inRange(0, 0, 0));
  EXPECT_TRUE(inRange(1, 1, 1));
  EXPECT_TRUE(inRange(1, 0, 1));
  EXPECT_TRUE(inRange(0, 0, 1));
  EXPECT_TRUE(inRange(0.99, 0, 1));
  EXPECT_FALSE(inRange(1.01, 0, 1));
  EXPECT_FALSE(inRange(0.99, 1, 0));
  EXPECT_FALSE(inRange(0, 1, 0));
}

TEST(vs_basic, findKth) {
  std::vector<int> a = {1, 6, 8, 4, 9, 5, 2, 3, 7, 0};
  EXPECT_EQ(findKth(a, 3), 3);
  EXPECT_EQ(findKth(a, 8), 8);
  EXPECT_EQ(findKth(a, 0), 0);
  EXPECT_EQ(findKth(a, 9), 9);
  EXPECT_EQ(findKth(a, -1), 9);
  EXPECT_EQ(findKth(a, 11), 0);
}

TEST(vs_basic, normalize) {
  std::vector<double> list = {-100, -VS_2PI, -VS_PI, -VS_PI_2, 0, VS_PI_2, VS_PI, VS_2PI, 100};
  for (auto a : list) {
    double res = normalizeRad(a);
    EXPECT_GE(res, -M_PI);
    EXPECT_LE(res, M_PI);
    EXPECT_NEAR(sin(res), sin(a), VS_EPS);
    EXPECT_NEAR(cos(res), cos(a), VS_EPS);
    EXPECT_NEAR(normalizeRad(a), deg2rad(normalizeDeg(rad2deg(a))), VS_EPS);
  }
}

TEST(vs_basic, normalizeList) {
  std::vector<double> list1 = {0, 0, 0, 0, 0};
  normalizeList(&list1[0], list1.size());
  EXPECT_NEAR(listnorm(&list1[0], list1.size()), 0.0, VS_EPS);

  std::vector<double> list2 = {0, 1, 3, 5, 7};
  normalizeList(&list2[0], list2.size());
  EXPECT_NEAR(listnorm(&list2[0], list2.size()), 1.0, VS_EPS);
}

TEST(vs_basic, lerp) {}