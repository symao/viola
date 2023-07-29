#include <fstream>
#include <viola/vs_stb_image.h>
#include <viola/vs_debug_draw.h>

void cmpImg(const cv::Mat& img1, const cv::Mat& img2) {
  cv::Mat diff;
  cv::absdiff(img1, img2, diff);
  int max = 0;
  float mean = 0;
  uchar* ptr = diff.data;
  const int N = diff.cols * diff.rows * diff.channels();
  for (int i = 0; i < N; i++) {
    int v = *ptr++;
    if (v > max) max = v;
    mean += v;
  }
  mean /= static_cast<float>(N);
  printf("diff max:%d mean:%.4f\n", max, mean);
  cv::imshow("img", vs::hstack({img1, img2, diff}));
  cv::waitKey();
}

void testImread(const char* fimg, int flag = -1) {
  cv::Mat img1 = vs::imread(fimg, flag);
  cv::Mat img2;
  std::ifstream img_f(fimg, std::ios_base::in | std::ios::binary);
  if (img_f.is_open()) {
    img_f.seekg(0, img_f.end);
    size_t src_size = img_f.tellg();
    img_f.clear();
    img_f.seekg(0, std::ios::beg);
    char* buffer = new char[src_size];  // allocate memory for a buffer of appropriate dimension
    img_f.read(buffer, src_size);       // read the whole file into the buffer
    img2 = vs::imreadMemory(buffer, src_size, flag);
    delete[] buffer;
  }
  cmpImg(img1, img2);
}

void testImwrite(const cv::Mat& img, int img_format) {
  std::string suffix;
  switch (img_format) {
    case vs::STB_WRITE_BMP:
      suffix = ".bmp";
      break;
    case vs::STB_WRITE_PNG:
      suffix = ".png";
      break;
    case vs::STB_WRITE_JPG:
      suffix = ".jpg";
      break;
    default:
      break;
  }
  std::string fimg1 = std::string("./tmp1") + suffix;
  std::string fimg2 = std::string("./tmp2") + suffix;
  vs::imwrite(fimg1.c_str(), img);

  std::ofstream fout(fimg2, std::ios_base::out | std::ios::binary);
  auto buffer = vs::imwriteMemory(img, img_format);
  fout.write(reinterpret_cast<const char*>(&buffer[0]), buffer.size());
  fout.close();

  cv::Mat img1 = cv::imread(fimg1);
  cv::Mat img2 = cv::imread(fimg2);
  cmpImg(img1, img2);
}

int main() {
  const char* fpng = "/home/symao/workspace/vs_common/data/snow.png";
  const char* fjpg = "/home/symao/Pictures/2DMarker/logo/starbucks.jpg";
  testImread(fpng);
  testImread(fpng, cv::IMREAD_GRAYSCALE);
  testImread(fjpg);
  testImread(fjpg, cv::IMREAD_GRAYSCALE);

  testImwrite(cv::imread(fpng), vs::STB_WRITE_BMP);
  testImwrite(cv::imread(fpng), vs::STB_WRITE_JPG);
  testImwrite(cv::imread(fpng), vs::STB_WRITE_PNG);
  testImwrite(cv::imread(fjpg), vs::STB_WRITE_BMP);
  testImwrite(cv::imread(fjpg), vs::STB_WRITE_JPG);
  testImwrite(cv::imread(fjpg), vs::STB_WRITE_PNG);
}