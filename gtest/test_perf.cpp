#include <gtest/gtest.h>
#include <viola/viola.h>

using namespace vs;

TEST(vs_perf, FpsCalculator) {
  FpsCalculator fc(20);
  for (int i = 0; i < 30; i++) {
    fc.start();
    vs::msleep(10);
    fc.stop();
  }
  EXPECT_NEAR(fc.fps(), 100, 10);
}