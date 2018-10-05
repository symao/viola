/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2023-04-13 15:23
 * @details wrapper of fbxsdk for fbx reading/writing.
 */
#pragma once
#include <fbxsdk.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <string>
#include <vector>

namespace vs {

inline std::vector<FbxNode*> fbxTravelScene(FbxNode* root_node, FbxNodeAttribute::EType select_type) {
  std::vector<FbxNode*> node_list;
  std::vector<FbxNode*> nodes = {root_node};
  for (size_t i = 0; i < nodes.size(); i++) {
    FbxNode* node = nodes[i];
    for (int j = 0; j < node->GetNodeAttributeCount(); j++) {
      auto type = node->GetNodeAttributeByIndex(j)->GetAttributeType();
      if (type == select_type) {
        node_list.push_back(node);
        break;
      }
    }
    for (int j = 0; j < node->GetChildCount(); j++) nodes.push_back(node->GetChild(j));
  }
  return node_list;
}

inline std::map<FbxNodeAttribute::EType, std::vector<FbxNode*>> fbxTravelScene(
    FbxNode* root_node, const std::vector<FbxNodeAttribute::EType>& select_types) {
  std::map<FbxNodeAttribute::EType, std::vector<FbxNode*>> node_map;
  if (select_types.empty()) return node_map;

  std::vector<FbxNode*> nodes = {root_node};
  for (size_t i = 0; i < nodes.size(); i++) {
    FbxNode* node = nodes[i];
    for (int j = 0; j < node->GetNodeAttributeCount(); j++) {
      auto type = node->GetNodeAttributeByIndex(j)->GetAttributeType();
      for (auto t : select_types) {
        if (t == type) {
          node_map[type].push_back(node);
          break;
        }
      }
    }
    for (int j = 0; j < node->GetChildCount(); j++) nodes.push_back(node->GetChild(j));
  }
  return node_map;
}

inline int fbxReadMeshNode(FbxNode* mesh_node, Eigen::MatrixXd& mesh_pts, std::vector<std::vector<int>>& polygons,
                           bool global = true) {
  FbxMesh* mesh_ptr = (FbxMesh*)mesh_node->GetNodeAttribute();
  if (!mesh_ptr) return 0;

  // read mesh points
  int vertex_cnt = mesh_ptr->GetControlPointsCount();
  mesh_pts.resize(3, vertex_cnt);
  for (int i = 0; i < vertex_cnt; i++) {
    auto v = mesh_ptr->GetControlPointAt(i);
    mesh_pts.col(i) << v[0], v[1], v[2];
  }

  // read mesh polygons
  int polygon_cnt = mesh_ptr->GetPolygonCount();
  int* idx = mesh_ptr->GetPolygonVertices();
  polygons.resize(polygon_cnt);
  for (int i = 0; i < polygon_cnt; i++) {
    int* begin = idx + mesh_ptr->GetPolygonVertexIndex(i);
    int* end = begin + mesh_ptr->GetPolygonSize(i);
    polygons[i].assign(begin, end);
  }

  // read global transformation, apply it to mesh points
  if (global) {
    Eigen::Matrix4d global_mat = Eigen::Map<Eigen::Matrix4d>((double*)(mesh_ptr->GetNode()->EvaluateGlobalTransform()));
    mesh_pts = (global_mat * mesh_pts.colwise().homogeneous()).topRows<3>();
  }
  return vertex_cnt;
}

inline bool fbxSetMeshPoints(FbxNode* mesh_node, const Eigen::MatrixXd& mesh_pts) {
  FbxMesh* mesh_ptr = (FbxMesh*)mesh_node->GetNodeAttribute();
  if (!mesh_ptr) return false;

  int vertex_cnt = mesh_ptr->GetControlPointsCount();
  if (vertex_cnt != mesh_pts.cols()) {
    printf("[ERROR]fbxSetMeshPoints: failed, point size not match %d != %d\n", vertex_cnt,
           static_cast<int>(mesh_pts.cols()));
    return false;
  }
  for (int i = 0; i < vertex_cnt; i++)
    mesh_ptr->SetControlPointAt(FbxVector4(mesh_pts(0, i), mesh_pts(1, i), mesh_pts(2, i)), i);
  return true;
}

class FbxLoader {
 public:
  struct OutputData {
    int vertex_cnt = 0;                           ///< mesh point count
    int bone_cnt = 0;                             ///< bone joint count
    bool has_skinning = false;                    ///< whether has skinning weights
    Eigen::MatrixXd mesh_pts;                     ///< mesh points, size = [3, vertex_cnt]
    std::vector<std::vector<int>> mesh_polygons;  ///< Mesh topology, @c size=[<tt>number of polygons</tt>], #fv[@p p]
                                                  ///< is the vector of vertex indices of polygon @p p
    Eigen::MatrixXd skeleton_trans;               ///< skeleton tranformation matrix, size = [4, bone_cnt*4]
    std::vector<std::string> skeleton_names;      ///< skeleton names, size = [bone_cnt,]
    std::vector<std::vector<int>> skeleton_topology;  ///< skeleton topology, each elem store direct sub-joints' index
    Eigen::SparseMatrix<double> skinning_weights;     ///< size = [bone_cnt, vertex_cnt], Eigen sparse matrix
    std::map<double, Eigen::MatrixXd> global_skeleton_animation;    ///< <time, skeleton_trans at time>
    std::map<double, Eigen::MatrixXd> relative_skeleton_animation;  ///< <time, skeleton_trans at time> relative,
                                                                    ///< used for linear blend skinning

    void print(bool verbose = false) {
      if (verbose) {
        for (int i = 0; i < bone_cnt; i++) {
          const auto& ids = skeleton_topology[i];
          printf("#%d %s: [", i, skeleton_names[i].c_str());
          for (int j : ids) printf("%d ", j);
          printf("] %d\n", static_cast<int>(ids.size()));
        }
      }
      printf("mesh:%d bone:%d has_skin:%d weights:%dx%d(%d) animation frame:%d\n", vertex_cnt, bone_cnt, has_skinning,
             static_cast<int>(skinning_weights.rows()), static_cast<int>(skinning_weights.cols()),
             static_cast<int>(skinning_weights.nonZeros()), static_cast<int>(global_skeleton_animation.size()));
    }

    std::map<std::string, int> computeSkeletonNameMap() const {
      std::map<std::string, int> name_ids;
      for (size_t i = 0; i < skeleton_names.size(); i++) name_ids[skeleton_names[i]] = i;
      return name_ids;
    }

    void updateRelativeSkeletonAnimation() {
      relative_skeleton_animation.clear();
      for (const auto& it : global_skeleton_animation)
        relative_skeleton_animation[it.first] = calcRelativeSkeletonTransform(it.second);
    }

    /** @brief convert skeleton transformation from global to relative, which is used for linear blend skinning
     * @param[in]cur_global_trans: global skeleton transformation size=[4, bone_cnt*4]
     * @return relative skeleton transformation size=[4, bone_cnt*4]
     * @details Note that the transformations are relative, that is ;brings the global transformation of bone @p j from
     * the rest pose to the pose at frame @p k.
     */
    Eigen::MatrixXd calcRelativeSkeletonTransform(const Eigen::MatrixXd& cur_global_trans) {
      Eigen::MatrixXd relative_trans(cur_global_trans.rows(), cur_global_trans.cols());
      for (int i = 0; i < bone_cnt; i++)
        relative_trans.middleCols(i * 4, 4) =
            cur_global_trans.middleCols(i * 4, 4) * skeleton_trans.middleCols(i * 4, 4).inverse();
      return relative_trans;
    }

    Eigen::Isometry3d getTransformByIndex(int index) {
      Eigen::Isometry3d T;
      T.matrix() = skeleton_trans.middleCols(index * 4, 4);
      return T;
    }

    Eigen::Matrix4d getTransformMatByIndex(int index) { return skeleton_trans.middleCols(index * 4, 4); }
  };

  FbxLoader(bool import_animation = true) {
    // Initialize the SDK manager. This object handles memory management.
    sdk_manager_ = FbxManager::Create();
    // Create the IO settings object.
    io_setting_ = FbxIOSettings::Create(sdk_manager_, IOSROOT);
    sdk_manager_->SetIOSettings(io_setting_);
    // Import animation?
    (*(sdk_manager_->GetIOSettings())).SetBoolProp(IMP_FBX_ANIMATION, import_animation);
    // Create an importer using the SDK manager.
    importer_ = FbxImporter::Create(sdk_manager_, "");
    // Create a new scene so that it can be populated by the imported file.
    scene_ = FbxScene::Create(sdk_manager_, "myScene");
  }

  ~FbxLoader() {
    scene_->Destroy();
    importer_->Destroy();
    io_setting_->Destroy();
    sdk_manager_->Destroy();
  }

  bool load(const std::string& file_name) {
    // Use the first argument as the filename for the importer.
    if (!importer_->Initialize(file_name.c_str(), -1, sdk_manager_->GetIOSettings())) return false;
    // clear the scene
    scene_->Clear();
    // import the contents of the file into the scene.
    importer_->Import(scene_);
    return true;
  }

  /** @brief load fbx content from file */
  bool load(const std::string& file_name, OutputData& data) {
    if (!load(file_name)) return false;
    // load skeleton: joints' transformation, names, topology, as well as skeleton animation
    data.bone_cnt =
        loadSkeleton(data.skeleton_trans, data.skeleton_names, data.skeleton_topology, data.global_skeleton_animation);

    // load mesh point and mesh polygons
    auto mesh_nodes = fbxTravelScene(scene_->GetRootNode(), FbxNodeAttribute::EType::eMesh);
    data.vertex_cnt = fbxReadMeshNode(mesh_nodes[0], data.mesh_pts, data.mesh_polygons);

    // load skinning weights
    data.has_skinning = loadSkinning(mesh_nodes[0], data.skeleton_names, data.skinning_weights);

    // calculate relative animation skeleton transform
    data.relative_skeleton_animation.clear();
    for (const auto& it : data.global_skeleton_animation)
      data.relative_skeleton_animation[it.first] = data.calcRelativeSkeletonTransform(it.second);
    return true;
  }

  FbxScene* getScene() { return scene_; }

 protected:
  FbxManager* sdk_manager_;
  FbxIOSettings* io_setting_;
  FbxImporter* importer_;
  FbxScene* scene_;

  /** @brief read skeleton transformations, names, tree topology as well as skeleton animation transforms in global */
  int loadSkeleton(Eigen::MatrixXd& skeleton_trans, std::vector<std::string>& skeleton_names,
                   std::vector<std::vector<int>>& skeleton_topology,
                   std::map<double, Eigen::MatrixXd>& global_skeleton_animation) {
    // traval fbx tree to find skeletion nodes
    auto skeleton_nodes = fbxTravelScene(scene_->GetRootNode(), FbxNodeAttribute::EType::eSkeleton);
    int bone_cnt = skeleton_nodes.size();

    // read bones' transformations
    skeleton_trans.resize(4, 4 * bone_cnt);
    for (int i = 0; i < bone_cnt; i++) {
      FbxNode* node = skeleton_nodes[i];
      skeleton_trans.middleCols(4 * i, 4) = toEigen4x4(node->EvaluateGlobalTransform());
    }

    // read bones' names
    skeleton_names.clear();
    skeleton_names.reserve(bone_cnt);
    std::map<std::string, int> name_ids;
    for (int i = 0; i < bone_cnt; i++) {
      FbxNode* node = skeleton_nodes[i];
      std::string name = node->GetName();
      skeleton_names.push_back(name);
      name_ids[name] = i;
    }

    // read bone topology which is a tree
    skeleton_topology.clear();
    skeleton_topology.reserve(bone_cnt);
    for (FbxNode* node : skeleton_nodes) {
      std::string name = node->GetName();
      int child_cnt = node->GetChildCount();
      std::vector<int> child_ids;
      if (child_cnt > 0) {
        child_ids.reserve(child_cnt);
        for (int j = 0; j < node->GetChildCount(); j++) child_ids.push_back(name_ids[node->GetChild(j)->GetName()]);
      }
      skeleton_topology.push_back(child_ids);
    }

    // read skeleton animation
    std::vector<FbxTime> time_list;
    FbxAnimStack* anim_stack = scene_->GetCurrentAnimationStack();
    FbxTimeSpan time_span = anim_stack->GetLocalTimeSpan();
    // printf("TimeSpan:%f %f\n", time_span.GetStart().GetSecondDouble(),time_span.GetStop().GetSecondDouble());
    int anim_layer_cnt = anim_stack->GetMemberCount<FbxAnimLayer>();
    if (anim_layer_cnt > 0) {
      FbxAnimLayer* anim_layer = anim_stack->GetMember<FbxAnimLayer>(0);
      FbxNode* node = skeleton_nodes[0];
      FbxAnimCurve* curve = node->LclTranslation.GetCurve(anim_layer, FBXSDK_CURVENODE_COMPONENT_X);
      if (curve) {
        for (int i = 0; i < curve->KeyGetCount(); i++) time_list.push_back(curve->KeyGetTime(i));
      }
    }
    if (!time_list.empty()) {  // if animation available
      for (const auto& t : time_list) {
        Eigen::MatrixXd t_trans(4, 4 * bone_cnt);
        for (int i = 0; i < bone_cnt; i++) {
          FbxNode* node = skeleton_nodes[i];
          t_trans.middleCols(4 * i, 4) = toEigen4x4(node->EvaluateGlobalTransform(t));
        }
        global_skeleton_animation[t.GetSecondDouble()] = t_trans;
      }
    }
    return bone_cnt;
  }

  /** @brief read skinning weights */
  bool loadSkinning(FbxNode* mesh_node, const std::vector<std::string>& skeleton_names,
                    Eigen::SparseMatrix<double>& weights) {
    // get skin ptr
    FbxMesh* mesh_ptr = (FbxMesh*)mesh_node->GetNodeAttribute();
    if (mesh_ptr == NULL) return false;
    FbxSkin* skin_ptr = firstSkin(mesh_ptr);
    if (skin_ptr == NULL) return false;
    int cluster_cnt = skin_ptr->GetClusterCount();
    int vertex_cnt = mesh_ptr->GetControlPointsCount();
    int bone_cnt = skeleton_names.size();

    // map skeleton names <name, bone_index>
    std::map<std::string, int> name_map;
    for (int i = 0; i < bone_cnt; i++) name_map[skeleton_names[i]] = i;

    // read weights, each cluster is corresponding to a bone
    weights = Eigen::SparseMatrix<double>(bone_cnt, vertex_cnt);
    for (int j = 0; j < cluster_cnt; j++) {
      FbxCluster* cluster = skin_ptr->GetCluster(j);
      std::string name_j = cluster->GetLink()->GetName();
      int bone_idx = name_map[name_j];
      // get corresponding vertex and weights for current bone
      double* val = cluster->GetControlPointWeights();
      int* idx = cluster->GetControlPointIndices();
      int nj = cluster->GetControlPointIndicesCount();
      for (int k = 0; k < nj; k++) {
        if (idx[k] < vertex_cnt) weights.insert(bone_idx, idx[k]) = val[k];
      }
    }
    return true;
  }

  /** @brief get first skin ptr */
  FbxSkin* firstSkin(FbxMesh* mesh_ptr) {
    for (int i = 0; i < mesh_ptr->GetDeformerCount(); i++) {
      FbxDeformer* deformer = mesh_ptr->GetDeformer(i);
      if ((deformer->GetDeformerType() == FbxDeformer::eSkin) && (((FbxSkin*)deformer)->GetClusterCount() > 0))
        return (FbxSkin*)deformer;
    }
    return NULL;
  }

  /** @brief convert fbx matrix to eigen matrix */
  Eigen::Matrix4d toEigen4x4(FbxAMatrix& m) { return Eigen::Map<Eigen::Matrix4d>((double*)(m)); }
};

inline bool writeFbx(const std::string& save_file, FbxScene* scene) {
  FbxManager* sdk_manager = FbxManager::Create();
  auto io_setting = FbxIOSettings::Create(sdk_manager, IOSROOT);
  sdk_manager->SetIOSettings(io_setting);
  io_setting->SetBoolProp(EXP_FBX_EMBEDDED, true);
  FbxExporter* exporter = FbxExporter::Create(sdk_manager, "");
  int file_format = sdk_manager->GetIOPluginRegistry()->GetNativeWriterFormat();
  bool ok = exporter->Initialize(save_file.c_str(), file_format, sdk_manager->GetIOSettings());
  if (ok) {
    exporter->Export(scene);
  } else {
    printf("[ERROR]writeFbx: Call to FbxExporter::Initialize() failed.\n");
  }
  exporter->Destroy();
  io_setting->Destroy();
  sdk_manager->Destroy();
  return ok;
}

}  // namespace vs
