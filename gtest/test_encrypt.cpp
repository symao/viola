#include <gtest/gtest.h>
#include <viola/viola.h>
using namespace vs;

TEST(vs_encrypt, encrypt) {
  std::string raw = "Hello, this is a raw string to be encrypted: &^%$&$$*FG123131345646\n";
  std::string key = "this is a key";
  std::string encrypt = vs::encrypt(raw, key);
  std::string decrypt = vs::decrypt(encrypt, key);
  EXPECT_STREQ(raw.c_str(), decrypt.c_str());
}