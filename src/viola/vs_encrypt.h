/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2022-06-13 15:23
 * @details string content encrypting/decrypting.
 */
#pragma once
#include <string>

/**
 * @brief a simple encrypt tools for file encrypting/decrypting
 * @code
 * // encrypting a file
 * vs::encryptFile("input_file.txt", "encrypt_file.txt", "your-own-key");
 * // read a encrypted file and decrypt
 * std::string data = vs::decrypt(vs::loadFileContent("encrypt_file.txt"), "your-own-key");
 * @endcode
 */

namespace vs {

/** @brief encrypt string data with a specific key
 * @param[in]data: input string to be encrypted
 * @param[in]key: encrypting key, which can be any string
 * @return std::string encrypt string data
 */
std::string encrypt(const std::string& data, const std::string& key);

/** @brief decrypt string data with a specific key
 * @param[in]data: input string to be decrypted
 * @param[in]key: decrypting key, which must be same as encrypting key
 * @return std::string decrypt string data
 */
std::string decrypt(const std::string& data, const std::string& key);

/** @brief load content from file and return as string
 * @param[in]file: input file string
 * @return std::string file content data
 */
std::string loadFileContent(const std::string& file);

/** @brief encrypt file with a specific key
 * @param[in]file_in: input file to be encrypted
 * @param[in]file_out: output encrypted file
 * @param[in]key: encrypting key, which can be any string
 * @return whether output file ok
 */
bool encryptFile(const std::string& file_in, const std::string& file_out, const std::string& key);

/** @brief decrypt file with a specific key
 * @param[in]file_in: input file to be decrypted
 * @param[in]file_out: output decrypted file
 * @param[in]key: decrypting key, which must be same as encrypting key
 * @return whether output file ok
 */
bool decryptFile(const std::string& file_in, const std::string& file_out, const std::string& key);

}  // namespace vs