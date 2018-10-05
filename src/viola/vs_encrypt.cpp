/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-06-13 15:23
 * @details
 */
#include "vs_encrypt.h"
#include <sstream>
#include <fstream>

namespace vs {

#define VS_ENCRYPT_INT_STR_HEAD ' '
#define VS_ENCRYPT_INT_STR_TAIL '*'

std::string encrypt(const std::string& data, const std::string& key) {
  std::string encrypt_data;
  int key_len = key.length();
  for (size_t i = 0; i < data.length(); i++) {
    int code = key[i % key_len];
    code += i % 10;
    int encrypt_num = data[i] ^ code;
    if (33 <= encrypt_num && encrypt_num <= 126) {  // valid char ascii range
      encrypt_data.push_back(static_cast<char>(encrypt_num));
    } else {
      encrypt_data.push_back(VS_ENCRYPT_INT_STR_HEAD);
      encrypt_data = encrypt_data + std::to_string(encrypt_num);
      encrypt_data.push_back(VS_ENCRYPT_INT_STR_TAIL);
    }
  }
  return encrypt_data;
}

std::string decrypt(const std::string& data, const std::string& key) {
  std::string decrypt_data;
  int code_idx = 0;
  int data_len = data.length();
  int key_len = key.length();
  for (int idx = 0; idx < data_len;) {
    int code = key[code_idx % key_len];
    code += code_idx % 10;
    int num = 0;
    if (data[idx] == VS_ENCRYPT_INT_STR_HEAD) {
      int j = idx + 1;
      while (j < data_len && data[j] != VS_ENCRYPT_INT_STR_TAIL) {
        num = num * 10 + static_cast<int>(data[j] - '0');
        j++;
      }
      idx = j + 1;
    } else {
      num = data[idx];
      idx += 1;
    }
    decrypt_data.push_back(static_cast<char>(num ^ code));
    code_idx++;
  }
  return decrypt_data;
}

std::string loadFileContent(const std::string& file) {
  std::ifstream fin(file, std::ios::in);
  if (fin.is_open()) {
    fin.seekg(0, fin.end);
    int size = fin.tellg();
    char* content = new char[size];
    fin.seekg(0, fin.beg);
    fin.read(content, size);
    std::string content_str;
    content_str.assign(content, size);
    delete[] content;
    fin.close();
    return content_str;
  } else {
    return "";
  }
}

bool encryptFile(const std::string& file_in, const std::string& file_out, const std::string& key) {
  std::string content = loadFileContent(file_in);
  if (content.empty()) return false;
  std::string encrypt_content = encrypt(content, key);
#ifdef WIN32
  FILE* fp;
  if (fopen_s(&fp, file_out.c_str(), "w") != 0) return false;
#else
  FILE* fp = fopen(file_out.c_str(), "w");
  if (!fp) return false;
#endif
  fprintf(fp, "%s", encrypt_content.c_str());
  fclose(fp);
  return true;
}

bool decryptFile(const std::string& file_in, const std::string& file_out, const std::string& key) {
  std::string content = loadFileContent(file_in);
  if (content.empty()) return false;
  std::string decrypt_content = decrypt(content, key);
#ifdef WIN32
  FILE* fp;
  if (fopen_s(&fp, file_out.c_str(), "w") != 0) return false;
#else
  FILE* fp = fopen(file_out.c_str(), "w");
  if (!fp) return false;
#endif
  fprintf(fp, "%s", decrypt_content.c_str());
  fclose(fp);
  return true;
}

}  // namespace vs