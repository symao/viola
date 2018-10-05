#include "vs_tag.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "vs_basic.h"

#define ENABLE_ARUCO 1

#if ENABLE_ARUCO
#include <opencv2/aruco.hpp>
#endif  // ENABLE_ARUCO

#if ENABLE_APRILTAG
#include <ethz_apriltag/Tag36h11.h>
#include <ethz_apriltag/TagDetector.h>
#endif  // ENABLE_APRILTAG

namespace vs {

Tag3DList generate3DTags(const TagConfig& tag_cfg) {
  const double tag_sz = tag_cfg.tag_size;
  const double tag_spacing_sz = tag_sz * (1. + tag_cfg.tag_spacing);
  int id = 0;
  Tag3DList tag_list;
  for (int i = 0; i < tag_cfg.tag_rows; i++) {
    for (int j = 0; j < tag_cfg.tag_cols; j++, id++) {
      Tag3D tag;
      tag.id = id;
      tag.pts3d = {
          cv::Point3f(j * tag_spacing_sz, i * tag_spacing_sz, 0),
          cv::Point3f(j * tag_spacing_sz + tag_sz, i * tag_spacing_sz, 0),
          cv::Point3f(j * tag_spacing_sz + tag_sz, i * tag_spacing_sz + tag_sz, 0),
          cv::Point3f(j * tag_spacing_sz, i * tag_spacing_sz + tag_sz, 0),
      };
    }
  }
  return tag_list;
}

void drawTags(cv::Mat& img, const Tag2DList& tag_list) {
  static const cv::Scalar circle_color(255, 0, 0);
  static const cv::Scalar line_color(0, 255, 0);
  static const cv::Scalar text_color(30, 30, 250);
  for (const auto& tag : tag_list) {
    const auto& pts = tag.pts2d;
    cv::putText(img, vs::num2str(tag.id), vs::vecMean(pts), cv::FONT_HERSHEY_COMPLEX, 0.6, text_color, 2);
    for (size_t i = 0; i < pts.size(); i++) cv::line(img, pts[i], pts[(i + 1) % pts.size()], line_color, 1);
    for (const auto& p : pts) cv::circle(img, p, 3, circle_color, 2);
  }
}

#if ENABLE_ARUCO
class TagDetectorAruco : public TagDetector {
 public:
  TagDetectorAruco(const TagConfig& tag_cfg) : m_tag_cfg(tag_cfg) {
    m_param = cv::aruco::DetectorParameters::create();
    m_param->cornerRefinementMethod = cv::aruco::CORNER_REFINE_CONTOUR;
    // m_param->cornerRefinementMethod = cv::aruco::CORNER_REFINE_NONE;
    // m_param->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX; //bad
    // m_param->cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;

    switch (tag_cfg.tag_type) {
      case TAG_ARUCO:
        m_aruco_dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        break;
      case TAG_APRILTAG:
        m_aruco_dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
        break;
      default:
        break;
    }
  }

  virtual Tag2DList detect(const cv::Mat& img) {
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    cv::aruco::detectMarkers(img, m_aruco_dict, corners, ids, m_param);
    int tag_cnt = ids.size();
    Tag2DList tag_list;
    tag_list.reserve(tag_cnt);
    for (int i = 0; i < tag_cnt; i++) {
      Tag2D tag;
      tag.id = ids[i];
      tag.pts2d = corners[i];
      tag_list.push_back(tag);
    }
    return tag_list;
  }

 private:
  TagConfig m_tag_cfg;
  cv::Ptr<cv::aruco::DetectorParameters> m_param;
  cv::Ptr<cv::aruco::Dictionary> m_aruco_dict;
};
#endif  // ENABLE_ARUCO

#if ENABLE_APRILTAG
class TagDetectorApriltag : public TagDetector {
 public:
  TagDetectorApriltag(const TagConfig& tag_cfg) : m_tag_cfg(tag_cfg) {
    AprilTags::TagCodes tag_code(AprilTags::tagCodes36h11);
    m_detector = std::make_shared<AprilTags::TagDetector>(tag_code, tag_cfg.black_border);
  }

  virtual Tag2DList detect(const cv::Mat& img) {
    auto detections = m_detector->extractTags(img);
    Tag2DList tag_list;
    tag_list.reserve(detections.size());
    for (const auto& detection : detections) {
      Tag2D tag;
      tag.id = detection.id;
      tag.pts2d = {
          cv::Point2f(detection.p[0].first, detection.p[0].second),
          cv::Point2f(detection.p[1].first, detection.p[1].second),
          cv::Point2f(detection.p[2].first, detection.p[2].second),
          cv::Point2f(detection.p[3].first, detection.p[3].second),
      };
      tag_list.push_back(tag);
    }
    return tag_list;
  }

 private:
  TagConfig m_tag_cfg;
  std::shared_ptr<AprilTags::TagDetector> m_detector;
};
#endif  // ENABLE_APRILTAG

class TagDetectorChessboard : public TagDetector {
 public:
  TagDetectorChessboard(const TagConfig& tag_cfg) : m_tag_cfg(tag_cfg) {}

  virtual Tag2DList detect(const cv::Mat& img) {
    cv::Size pattern_size(m_tag_cfg.tag_cols, m_tag_cfg.tag_rows);
    Tag2D tag;
    if (cv::findChessboardCorners(img, pattern_size, tag.pts2d)) {
      cv::cornerSubPix(img, tag.pts2d, cv::Size(11, 11), cv::Size(-1, -1),
                       cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.1));
      if (static_cast<int>(tag.pts2d.size()) == pattern_size.area()) return {tag};
    }
    return {};
  }

 private:
  TagConfig m_tag_cfg;
};

std::shared_ptr<TagDetector> createTagDetector(const TagConfig& tag_cfg) {
  if (tag_cfg.tag_type == TAG_CHESSBOARD) return std::make_shared<TagDetectorChessboard>(tag_cfg);
#if ENABLE_ARUCO
  if (tag_cfg.tag_type == TAG_ARUCO) return std::make_shared<TagDetectorAruco>(tag_cfg);
#endif  // ENABLE_ARUCO
#if ENABLE_APRILTAG
  if (tag_cfg.tag_type == TAG_APRILTAG) return std::make_shared<TagDetectorApriltag>(tag_cfg);
#endif  // ENABLE_APRILTAG
  printf("[ERROR]%s: unsupport tag type:%d\n", __func__, tag_cfg.tag_type);
  return nullptr;
}

}  // namespace vs
