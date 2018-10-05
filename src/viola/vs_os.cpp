/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_os.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <thread>

#include "vs_basic.h"

#if __cplusplus > 201700L  // using std::filesystem in c++17
#include <filesystem>

namespace vs {
bool exists(const char* path) { return std::filesystem::exists(path); }

void makedirs(const char* path) { std::filesystem::create_directories(path); }

bool isfile(const char* path) { return std::filesystem::is_regular_file(path); }

bool isdir(const char* path) { return std::filesystem::is_directory(path); }

void rmtree(const char* path) { std::filesystem::remove_all(path); }

void move(const char* src, const char* dst) { std::filesystem::rename(src, dst); }

void copytree(const char* src, const char* dst) { std::filesystem::copy(src, dst); }

uint64_t filesize(const char* file) { return std::filesystem::file_size(file); }

uint64_t dirsize(const char* path) {
  auto space_info = std::filesystem::space(path);
  return space_info.capacity;
}

uint64_t disksize(const char* path) {
  auto space_info = std::filesystem::space(path);
  return space_info.capacity;
}

uint64_t diskfree(const char* path) {
  auto space_info = std::filesystem::space(path);
  return space_info.free;
}

}  // namespace vs
#elif defined(__linux__)  // old version, some function may be not implemented on windows
#include <dirent.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <unistd.h>

namespace vs {

bool exists(const char* path) { return access(path, 0) != -1; }

void makedirs(const char* path) {
  size_t t = -1;
  std::string s(path);
  while ((t = s.find_first_of("\\/", t + 1)) != s.npos) {
    const char* sub_path = s.substr(0, t).c_str();
    if (!exists(sub_path)) mkdir(sub_path, S_IRWXU | S_IRWXG | S_IRWXO);
  }
  if (!exists(path)) mkdir(path, S_IRWXU | S_IRWXG | S_IRWXO);
}

bool isfile(const char* path) {
  struct stat statbuf;
  if (stat(path, &statbuf) < 0) return false;
  return S_ISREG(statbuf.st_mode);
}

bool isdir(const char* path) {
  struct stat statbuf;
  if (stat(path, &statbuf) < 0) return false;
  return S_ISDIR(statbuf.st_mode);
}

void rmtree(const char* path) {
  DIR* dp = opendir(path);
  if (!dp) return;
  struct dirent* ptr;
  while ((ptr = readdir(dp)) != NULL) {
    if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) continue;
    std::string p = join(path, ptr->d_name);
    if (ptr->d_type == 4)
      rmtree(p.c_str());
    else
      std::remove(p.c_str());
  }
  closedir(dp);
  std::remove(path);
}

void move(const char* src, const char* dst) {
  char cmd[512] = {0};
  snprintf(cmd, sizeof(cmd), "mv %s %s", src, dst);
  int res = system(cmd);
  res = res;
}

void copytree(const char* src, const char* dst) {
  char cmd[512] = {0};
  snprintf(cmd, sizeof(cmd), "cp -r %s %s", src, dst);
  int res = system(cmd);
  res = res;
}

uint64_t filesize(const char* file) {
  struct stat statbuf;
  if (stat(file, &statbuf) < 0)
    return -1;
  else
    return statbuf.st_size;
}

uint64_t dirsize(const char* path) {
  DIR* dp;
  struct dirent* ptr;
  struct stat statbuf;
  uint64_t dir_size = 0;
  if ((dp = opendir(path)) == NULL) return -1;
  lstat(path, &statbuf);
  dir_size += statbuf.st_size;
  while ((ptr = readdir(dp)) != NULL) {
    char subdir[256] = {0};
    int idx = 0;
    idx += snprintf(subdir + idx, sizeof(subdir) - idx, "%s/", path);
    idx += snprintf(subdir + idx, sizeof(subdir) - idx, "%s", ptr->d_name);
    lstat(subdir, &statbuf);
    if (S_ISDIR(statbuf.st_mode)) {
      if (strcmp(".", ptr->d_name) == 0 || strcmp("..", ptr->d_name) == 0) continue;
      dir_size += dirsize(subdir);
    } else {
      dir_size += statbuf.st_size;
    }
  }
  closedir(dp);
  return dir_size;
}

uint64_t disksize(const char* path) {
  struct statfs disk_info;
  if (statfs(path, &disk_info) == -1) return -1;
  return (uint64_t)disk_info.f_bsize * disk_info.f_blocks;
}

uint64_t diskfree(const char* path) {
  struct statfs disk_info;
  if (statfs(path, &disk_info) == -1) return -1;
  return (uint64_t)disk_info.f_bsize * disk_info.f_bfree;
}
}  // namespace vs
#elif defined(WIN32)  // WINDOWS
#include <io.h>
#include <windows.h>
#include <winnt.h>

namespace vs {
bool exists(const char* path) { return _access(path, 0) != -1; }

void makedirs(const char* path) {
  // todo
  // std::filesystem::create_directories(path);
}

bool isfile(const char* path) {
  WIN32_FIND_DATAA FindFileData;
  FindFirstFileA(path, &FindFileData);
  return (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

bool isdir(const char* path) {
  WIN32_FIND_DATAA FindFileData;
  FindFirstFileA(path, &FindFileData);
  return FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY;
}

void rmtree(const char* path) {
  // todo
}

void move(const char* src, const char* dst) {
  // todo
}

void copytree(const char* src, const char* dst) {
  // todo
}

uint64_t filesize(const char* file) {
  // todo
  return 0;
}

uint64_t dirsize(const char* path) {
  // todo
  return 0;
}

uint64_t disksize(const char* path) {
  // todo
  return 0;
}

uint64_t diskfree(const char* path) {
  // todo
  return 0;
}
}  // namespace vs
#endif

#if (defined(__linux__) && (defined(__x86_64__) || defined(__i386__)))
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <execinfo.h>
#include <signal.h>

namespace vs {
int getCpuN() {
  static int n_cpu = sysconf(_SC_NPROCESSORS_CONF);
  return n_cpu;
}

int getCpuId() { return sched_getcpu(); }

bool setCpuHighPerformance(int cpu_id) {
  if (cpu_id < 0 || cpu_id >= getCpuN()) {
    printf("[WARN]Failed set cpu performance. invalid id:%d\n", cpu_id);
    return false;
  }
  char path[256] = {0};
  sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpu_id);
  FILE* fp = fopen(path, "w");
  if (!fp) {
    printf("[ERROR]Failed set high performance. Cannot open file %s\n", path);
    return false;
  }
  fprintf(fp, "performance");
  fclose(fp);
  return true;
}

bool bindCpu(int cpu_id, std::thread* thread_ptr) {
  if (cpu_id < 0 || cpu_id >= getCpuN()) {
    printf("[WARN]Failed bind thread to cpu. invalid id:%d\n", cpu_id);
    return false;
  }
  // only CPU i as set.
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu_id, &cpuset);
  if (thread_ptr)
    pthread_setaffinity_np(thread_ptr->native_handle(), sizeof(cpu_set_t), &cpuset);
  else
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  return true;
}

static std::string p_log_file;

static void seg_fault_callback(int signal, siginfo_t* si, void* arg) {
  printf("[ERROR]Catch a segmetation fault error.\n");
  void* array[10];
  char** strings;
  size_t i;
  size_t size = backtrace(array, 10);
  strings = backtrace_symbols(array, size);

  printf("************* VS_DEBUG: Catch Seg Fault **************\n");
  printf("Signal:%d Error Code:%d Address:%p\n", signal, si->si_errno, si->si_addr);
  printf("PID:%d UID:%d\n", static_cast<int>(si->si_pid), static_cast<int>(si->si_uid));
  printf("Calling Stack(%zd):\n", size);
  for (i = 0; i < size; i++) printf(">> %s\n", strings[i]);
  printf("*******************************************************\n");

  if (p_log_file.size() > 0) {
    FILE* fp = NULL;
    fp = fopen(p_log_file.c_str(), "w");
    if (fp) {
      fprintf(fp, "************* VS_DEBUG: Catch Seg Fault **************\n");
      fprintf(fp, "Signal:%d Error Code:%d Address:%p\n", signal, si->si_errno, si->si_addr);
      fprintf(fp, "PID:%d UID:%d\n", static_cast<int>(si->si_pid), static_cast<int>(si->si_uid));
      fprintf(fp, "Calling Stack(%zd):\n", size);
      for (i = 0; i < size; i++) fprintf(fp, ">> %s\n", strings[i]);
      fprintf(fp, "*******************************************************\n");
      fclose(fp);
    }
  }
  free(strings);
  exit(0);
}

void openSegFaultDebug(const char* log_file) {
  if (log_file) {
    p_log_file = std::string(log_file);
  }
  struct sigaction sa;
  memset(&sa, 0, sizeof(struct sigaction));
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = seg_fault_callback;
  sa.sa_flags = SA_SIGINFO;
  sigaction(SIGSEGV, &sa, NULL);
}

std::string abspath(const char* path) {
  char abs_path[PATH_MAX + 1];
  if (realpath(path, abs_path))
    return std::string(abs_path);
  else
    return std::string();
}

std::vector<std::string> listdir(const char* path, bool absname, bool recursive) {
  std::vector<std::string> files;
  DIR* dp = opendir(path);
  if (!dp) return files;
  struct dirent* ptr;
  while ((ptr = readdir(dp)) != NULL) {
    if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {
      continue;
    } else if (ptr->d_type == 8) {  // file
      files.push_back(!absname ? ptr->d_name : join(path, ptr->d_name));
    } else if (ptr->d_type == 10) {  // link file
      continue;
    } else if (ptr->d_type == 4 && recursive) {  // dir
      auto subfiles = listdir(join(path, ptr->d_name).c_str(), absname, recursive);
      files.insert(files.end(), subfiles.begin(), subfiles.end());
    }
  }
  closedir(dp);
  return files;
}

const char* homepath() { return getenv("HOME"); }

} /* namespace vs */

#elif defined(WIN32)  // WINDOWS
#include <io.h>
#include <windows.h>

namespace vs {

int getCpuN() { return 0; }

int getCpuId() { return 0; }

bool setCpuHighPerformance(int cpu_id) { return false; }

bool bindCpu(int cpu_id, std::thread* thread_ptr) { return false; }

void openSegFaultDebug(const char* log_file) { printf("[ERROR]openSegFaultDebug not support on windows.\n"); }

std::string abspath(const char* path) { return std::string(path); }

std::vector<std::string> listdir(const char* path, bool absname, bool recursive) {
  std::vector<std::string> files;
  struct _finddata_t fileinfo;
  std::string p;
  long long hfile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo);
  if (hfile == -1) return files;
  do {
    if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0) {
      continue;
    } else if (fileinfo.attrib & _A_SUBDIR) {  // dir
      if (recursive) {
        auto subfiles = listdir(join(path, fileinfo.name).c_str(), absname, recursive);
        files.insert(files.end(), subfiles.begin(), subfiles.end());
      }
    } else {  // file
      files.push_back(!absname ? fileinfo.name : join(path, fileinfo.name));
    }
  } while (_findnext(hfile, &fileinfo) == 0);
  _findclose(hfile);
  return files;
}

const char* homepath() {
  static std::string home_path;
  char* buf = nullptr;
  size_t len;
  if (_dupenv_s(&buf, &len, "HOME") == 0 && buf != nullptr) {
    home_path = std::string(buf);
    free(buf);
  }
  return home_path.c_str();
}

static int gettimeofday(struct timeval* tp, void* tzp) {
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;
  GetLocalTime(&wtm);
  tm.tm_year = wtm.wYear - 1900;
  tm.tm_mon = wtm.wMonth - 1;
  tm.tm_mday = wtm.wDay;
  tm.tm_hour = wtm.wHour;
  tm.tm_min = wtm.wMinute;
  tm.tm_sec = wtm.wSecond;
  tm.tm_isdst = -1;
  clock = mktime(&tm);
  tp->tv_sec = clock;
  tp->tv_usec = wtm.wMilliseconds * 1000;
  return 0;
}

} /* namespace vs */
#endif

namespace vs {

void remove(const char* path) { std::remove(path); }

void rename(const char* src, const char* dst) { move(src, dst); }

void copy(const char* src, const char* dst) {
  std::ifstream fin(src);
  std::ofstream fout(dst);
  if (!fin.is_open() || !fout.is_open()) {
    printf("[ERROR]copy: failed, %s not open.\n", fin.is_open() ? dst : src);
    return;
  }
  fout << fin.rdbuf();
  fin.close();
  fout.close();
}

void writeFile(const char* file, const char* content, const char* mode) {
#ifdef WIN32
  FILE* fp;
  fopen_s(&fp, file, mode);
#else
  FILE* fp = fopen(file, mode);
#endif
  if (!fp) {
    printf("[ERROR] open file '%s' failed.\n", file);
    return;
  }
  fprintf(fp, "%s\n", content);
  fclose(fp);
}

double getSysTs() {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) == 0) return tv.tv_sec + static_cast<double>(tv.tv_usec) * 1e-6;
  printf("[ERROR] getCurTs() failed.\n");
  return 0;
}

std::string getSysTsStr() {
  auto t = std::chrono::system_clock::now();
  time_t tm_cur = std::chrono::system_clock::to_time_t(t);
  struct tm tm_now;
#ifdef WIN32
  localtime_s(&tm_now, &tm_cur);
#else
  localtime_r(&tm_cur, &tm_now);
#endif  // WIN32
  int ms = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count() % 1000;
  char str[128] = {0};
  snprintf(str, 128, "%02d-%02d-%02d_%02d:%02d:%02d.%03d", tm_now.tm_year + 1900, tm_now.tm_mon + 1, tm_now.tm_mday,
           tm_now.tm_hour, tm_now.tm_min, tm_now.tm_sec, ms);
  return std::string(str);
}

}  // namespace vs