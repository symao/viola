#include <gtest/gtest.h>
#include <viola/viola.h>

using namespace vs;

TEST(vs_tictoc, Timer) {
  std::vector<int> sleep_ms = {1, 3, 5, 7, 10, 20, 50};
  for (auto m : sleep_ms) {
    Timer t;
    msleep(m);
    double cost = t.stop();
    EXPECT_NEAR(cost, m, 2);
  }
}

TEST(vs_tictoc, tictoc) {
  std::vector<int> sleep_ms = {1, 3, 5, 7, 10, 20, 50};
  for (auto m : sleep_ms) {
    tic("a");
    msleep(m);
    double cost = toc("a");
    EXPECT_NEAR(cost, m, 2);
  }
}