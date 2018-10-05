#include <gtest/gtest.h>
#include <viola/viola.h>

using namespace vs;

TEST(vs_kdtree, KDTree) {
  std::vector<std::vector<float> > data;
  for (int i = 0; i < 20; i++) {
    data.push_back(std::vector<float>({i * 0.5f, i * 1.1f, i * 1.2f}));
  }
  for (int i = 0; i < 3; i++) data[0][i] = data[2][i];

  KDTree tree;
  std::vector<float> query = {1, 2, 3};
  KDTree::KvecArray res;

  tree.knn(query, res, 2);
  EXPECT_TRUE(res.empty());
  tree.rnn(query, res, 3);
  EXPECT_TRUE(res.empty());

  tree.build(data);
  tree.knn(query, res, 1);
  EXPECT_EQ(res.size(), 1);
  EXPECT_FLOAT_EQ(res[0][0], 1.0f);
  EXPECT_FLOAT_EQ(res[0][1], 2.2f);
  EXPECT_FLOAT_EQ(res[0][2], 2.4f);

  tree.rnn(query, res, 2);
  EXPECT_EQ(res.size(), 3);

  tree.knn(query, res, 100);
  EXPECT_EQ(res.size(), 20);

  tree.rnn(query, res, 100);
  EXPECT_EQ(res.size(), 20);
}