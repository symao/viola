/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details cpu handle, file utils, path utils which is similar to os.path in python.
 */
#pragma once

#include <string>
#include <thread>
#include <vector>

namespace vs {
/** @brief get count of CPUs in platform */
int getCpuN();

/** @brief get CPU id of current thread */
int getCpuId();

/** @brief set CPU to high performance
 *  @param[in]cpu_id: cpu id
 *  @return whether succeed
 */
bool setCpuHighPerformance(int cpu_id);

/** @brief bind thread to specific CPU
 *  @param[in]cpu_id: cpu id
 *  @param[in]thread_ptr: thread address, if NULL, then bind current thread
 *  @return whether succeed
 */
bool bindCpu(int cpu_id, std::thread* thread_ptr = NULL);

/** @brief open segmentation fault backtrace stack
 * If segmetation fault occurs, backtrace stack will be print to screen as well as input log file.
 * @param[in]log_file: log backtrace stack into this file
 */
void openSegFaultDebug(const char* log_file = NULL);

/** @brief check whether file or dir exists
 *  @param[in]path: path to be checked, either file or dir
 *  @return whether input path exists
 */
bool exists(const char* path);

/** @brief make a multi-level directory */
void makedirs(const char* path);

/** @brief check whether path is a file */
bool isfile(const char* path);

/** @brief check whether path is a dir */
bool isdir(const char* path);

/** @brief get absolutely path */
std::string abspath(const char* path);

/** @brief remove a file or a empty dir */
void remove(const char* path);

/** @brief remove a file dir recursively */
void rmtree(const char* path);

/** @brief rename file */
void rename(const char* src, const char* dst);

/** @brief move file */
void move(const char* src, const char* dst);

/** @brief copy file */
void copy(const char* src, const char* dst);

/** @brief copy dir recursively */
void copytree(const char* src, const char* dst);

/** @brief get file size. [Byte] */
uint64_t filesize(const char* file);

/** @brief get dir size. [Byte] */
uint64_t dirsize(const char* path);

/** @brief get disk size, return -1 if failed. [Byte] */
uint64_t disksize(const char* path);

/** @brief get disk free size, return -1 if failed. [Byte] */
uint64_t diskfree(const char* path);

/** @brief get all files in dir */
std::vector<std::string> listdir(const char* path, bool absname = false, bool recursive = false);

/** @brief write content to a file. */
void writeFile(const char* file, const char* content, const char* mode = "w");

/** @brief get home path in linux. */
const char* homepath();

/** @brief get system ts start from 1970-01-01 00:00:00. [second] */
double getSysTs();

/** @brief get system ts string start from 1970-01-01 00:00:00.*/
std::string getSysTsStr();

} /* namespace vs */