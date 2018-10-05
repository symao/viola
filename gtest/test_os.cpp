#include <gtest/gtest.h>
#include <viola/viola.h>

using namespace vs;

TEST(vs_os, exists) {
  const char* path = "a/b/c";
  if (exists(path)) rmtree(path);
  EXPECT_FALSE(exists(path));
  makedirs(path);
  EXPECT_TRUE(exists(path));
  EXPECT_TRUE(isdir(path));
  EXPECT_FALSE(isfile(path));
  std::string file = join(path, "tmp.txt");
  const char* f = file.c_str();
  writeFile(f, "abcdefg");
  EXPECT_TRUE(exists(f));
  EXPECT_TRUE(isfile(f));
  EXPECT_FALSE(isdir(f));

  auto l = listdir(path);
  EXPECT_EQ(l.size(), 1);

  vs::remove(f);
  EXPECT_FALSE(exists(f));
  EXPECT_FALSE(isfile(f));
  EXPECT_FALSE(isdir(f));

  copytree("a", "b");
  EXPECT_TRUE(exists("b/b/c"));

  rmtree("b");
  EXPECT_FALSE(exists("b"));
  rmtree("a");
  EXPECT_FALSE(exists("a"));
  rmtree("a");
  EXPECT_FALSE(exists("a"));
}

TEST(vs_os, basename) {
  const char* path = "a/b/c/d.ext";
  EXPECT_EQ(vs::basename(path), "d.ext");
  EXPECT_EQ(dirname(path), "a/b/c");
  EXPECT_EQ(suffix(path), ".ext");
  EXPECT_EQ(abspath("a/b/./c/../.."), abspath("a/./"));
  EXPECT_TRUE(suffix("a/b/c/d").empty());

  std::string str1, str2;
  split(path, str1, str2);
  EXPECT_EQ(str1, "a/b/c");
  EXPECT_EQ(str2, "d.ext");

  splitext(path, str1, str2);
  EXPECT_EQ(str1, "a/b/c/d");
  EXPECT_EQ(str2, ".ext");
}