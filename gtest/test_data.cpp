#include <gtest/gtest.h>
#include <viola/viola.h>

using namespace vs;

TEST(vs_data_buffer, DataBuffer) {
  AtomDataBuffer<int> buf;
  EXPECT_FALSE(buf.has());
  buf.set(1);
  EXPECT_TRUE(buf.has());
  EXPECT_EQ(buf.get(), 1);
  buf.set(2);
  EXPECT_EQ(buf.getAndClear(), 2);
  EXPECT_FALSE(buf.has());
}

#ifdef HAVE_BOOST

TEST(vs_data_buffer, DataBufferRW) {
  DataBufferRW<int> buf;
  EXPECT_FALSE(buf.has());
  buf.set(1);
  EXPECT_TRUE(buf.has());
  EXPECT_EQ(buf.get(), 1);
  buf.set(2);
  EXPECT_EQ(buf.get(), 2);
  buf.clear();
  EXPECT_FALSE(buf.has());
}

#endif  // HAVE_BOOST