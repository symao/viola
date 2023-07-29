/**
 * Copyright (c) 2016-2023 shuyuanmao <maoshuyuan123@gmail.com>. All rights reserved.
 * @author shuyuanmao <maoshuyuan123@gmail.com>
 * @date 2019-05-19 02:07
 * @details
 */
#include "vs_kdtree.h"

#include <stdio.h>
#include <algorithm>
#include <vector>

#include <boost/multi_array.hpp>

namespace vs {

namespace kdtree_impl {

struct Interval {
  float lower, upper;
};

struct KDTreeResult {
  float dis;  // square distance
  int idx;    // neighbor index
};

typedef boost::multi_array<float, 2> KDTreeArray;

class KDTreeNode;
class KDTreeInnerImpl;
class KDTreeResultVector;

inline float squared(const float x) { return (x * x); }

inline bool operator<(const KDTreeResult& e1, const KDTreeResult& e2) { return (e1.dis < e2.dis); }

// search record substructure one of these is created for each search. this holds useful information to be used during
// the search
struct SearchRecord {
  SearchRecord(std::vector<float>& qv_in, KDTreeInnerImpl& tree_in, KDTreeResultVector& result_in);

  std::vector<float>& qv;
  int dim;
  bool rearrange;
  unsigned int nn;  // nfound;
  float ballsize;
  int centeridx, correltime;
  KDTreeResultVector& result;  // results
  const KDTreeArray* data;
  const std::vector<int>& ind;

  friend class KDTreeInnerImpl;
  friend class KDTreeNode;
};

/** @brief inherit a std::vector<KDTreeResult> but, optionally maintain it in heap form as a priority queue.*/
class KDTreeResultVector : public std::vector<KDTreeResult> {
 public:
  // add one new element to the list of results, and keep it in heap order.  To keep it in ordinary, as inserted,
  // order, then simply use push_back() as inherited via std::vector<>
  void push_element_and_heapify(KDTreeResult& e) {
    push_back(e);
    push_heap(begin(), end());  // and now heapify it, with the new elt.
  }

  float replace_maxpri_elt_return_new_maxpri(KDTreeResult& e) {
    // remove the maximum priority element on the queue and replace it with 'e', and return its priority.
    // here, it means replacing the first element [0] with e, and re heapifying.
    pop_heap(begin(), end());
    pop_back();
    push_back(e);               // insert new
    push_heap(begin(), end());  // and heapify.
    return ((*this)[0].dis);
  }

  float max_value() {
    return ((*begin()).dis);  // very first element
  }
  // return the distance which has the maximum value of all on list,
  // assuming that ALL insertions were made by push_element_and_heapify()
};

class KDTreeNode {
 public:
  KDTreeNode(int dim);
  ~KDTreeNode();

 private:
  friend class KDTreeInnerImpl;  // allow kdtree to access private data

  int cut_dim;                                 ///< dimension to cut;
  float cut_val, cut_val_left, cut_val_right;  ///< cut value
  int l, u;                                    ///< extents in index array for searching
  std::vector<Interval> box;                   ///< [min, max] of the box enclosing all points
  KDTreeNode *left, *right;                    ///< pointers to left and right nodes.

  /** @brief recursive innermost core routine for searching. */
  void search(SearchRecord& sr);

  /** @brief return true if the bounding box for this node is within the search range given by the searchvector and
   * maximum ballsize in 'sr'.
   */
  bool boxInSearchRange(SearchRecord& sr);

  // for processing final buckets.
  void processTerminalNode(SearchRecord& sr);
  void processTerminalNodeFixedball(SearchRecord& sr);
};

KDTreeNode::KDTreeNode(int dim) : box(dim) { left = right = NULL; }

KDTreeNode::~KDTreeNode() {
  // maxbox and minbox will be automatically deleted in their own destructors.
  if (left != NULL) delete left;
  if (right != NULL) delete right;
}

void KDTreeNode::search(SearchRecord& sr) {
  // the core search routine. This uses true distance to bounding box as the criterion to search the secondary node.
  // This results in somewhat fewer searches of the secondary nodes than 'search', which uses the vdiff vector,  but as
  // this takes more computational time, the overall performance may not be improved in actual run time.
  if ((left == NULL) && (right == NULL)) {
    // we are on a terminal node
    if (sr.nn == 0) {
      processTerminalNodeFixedball(sr);
    } else {
      processTerminalNode(sr);
    }
  } else {
    KDTreeNode *ncloser, *nfarther;

    float extra;
    float qval = sr.qv[cut_dim];
    // value of the wall boundary on the cut dimension.
    if (qval < cut_val) {
      ncloser = left;
      nfarther = right;
      extra = cut_val_right - qval;
    } else {
      ncloser = right;
      nfarther = left;
      extra = qval - cut_val_left;
    }
    if (ncloser != NULL) ncloser->search(sr);
    if ((nfarther != NULL) && (squared(extra) < sr.ballsize)) {
      // first cut
      if (nfarther->boxInSearchRange(sr)) {
        nfarther->search(sr);
      }
    }
  }
}

inline float disFromBnd(float x, float amin, float amax) {
  if (x > amax)
    return (x - amax);
  else if (x < amin)
    return (amin - x);
  else
    return 0.0;
}

inline bool KDTreeNode::boxInSearchRange(SearchRecord& sr) {
  // does the bounding box, represented by minbox[*], maxbox[*]
  // have any point which is within 'sr.ballsize' to 'sr.qv'??
  int dim = sr.dim;
  float dis2 = 0.0;
  float ballsize = sr.ballsize;
  for (int i = 0; i < dim; i++) {
    dis2 += squared(disFromBnd(sr.qv[i], box[i].lower, box[i].upper));
    if (dis2 > ballsize) return (false);
  }
  return (true);
}

void KDTreeNode::processTerminalNode(SearchRecord& sr) {
  int centeridx = sr.centeridx;
  int correltime = sr.correltime;
  unsigned int nn = sr.nn;
  int dim = sr.dim;
  float ballsize = sr.ballsize;
  bool rearrange = sr.rearrange;
  const KDTreeArray& data = *sr.data;

  for (int i = l; i <= u; i++) {
    int indexofi;  // sr.ind[i];
    float dis;
    bool early_exit;
    if (rearrange) {
      early_exit = false;
      dis = 0.0;
      for (int k = 0; k < dim; k++) {
        dis += squared(data[i][k] - sr.qv[k]);
        if (dis > ballsize) {
          early_exit = true;
          break;
        }
      }
      if (early_exit) continue;  // next iteration of mainloop
      // why do we do things like this?  because if we take an early
      // exit (due to distance being too large) which is common, then
      // we need not read in the actual point index, thus saving main
      // memory bandwidth.  If the distance to point is less than the
      // ballsize, though, then we need the index.
      //
      indexofi = sr.ind[i];
    } else {
      //
      // but if we are not using the rearranged data, then
      // we must always
      indexofi = sr.ind[i];
      early_exit = false;
      dis = 0.0;
      for (int k = 0; k < dim; k++) {
        dis += squared(data[indexofi][k] - sr.qv[k]);
        if (dis > ballsize) {
          early_exit = true;
          break;
        }
      }
      if (early_exit) continue;  // next iteration of mainloop
    }                            // end if rearrange.
    if (centeridx > 0) {
      // we are doing decorrelation interval
      if (abs(indexofi - centeridx) < correltime) continue;  // skip this point.
    }
    // here the point must be added to the list. two choices for any point.  The list so far is either
    // undersized, or it is not.
    if (sr.result.size() < nn) {
      KDTreeResult e;
      e.idx = indexofi;
      e.dis = dis;
      sr.result.push_element_and_heapify(e);
      if (sr.result.size() == nn) ballsize = sr.result.max_value();
      // Set the ball radius to the largest on the list (maximum priority).
    } else {
      // if we get here then the current node, has a squared distance smaller than the last on the list, and belongs on
      // the list.
      KDTreeResult e;
      e.idx = indexofi;
      e.dis = dis;
      ballsize = sr.result.replace_maxpri_elt_return_new_maxpri(e);
    }
  }  // main loop
  sr.ballsize = ballsize;
}

void KDTreeNode::processTerminalNodeFixedball(SearchRecord& sr) {
  int centeridx = sr.centeridx;
  int correltime = sr.correltime;
  int dim = sr.dim;
  float ballsize = sr.ballsize;
  bool rearrange = sr.rearrange;
  const KDTreeArray& data = *sr.data;
  for (int i = l; i <= u; i++) {
    int indexofi = sr.ind[i];
    float dis;
    bool early_exit;

    if (rearrange) {
      early_exit = false;
      dis = 0.0;
      for (int k = 0; k < dim; k++) {
        dis += squared(data[i][k] - sr.qv[k]);
        if (dis > ballsize) {
          early_exit = true;
          break;
        }
      }
      if (early_exit) continue;  // next iteration of mainloop
      // why do we do things like this?  because if we take an early exit (due to distance being too large) which is
      // common, then we need not read in the actual point index, thus saving main memory bandwidth.  If the distance to
      // point is less than the ballsize, though, then we need the index.
      indexofi = sr.ind[i];
    } else {
      // but if we are not using the rearranged data, then we must always
      indexofi = sr.ind[i];
      early_exit = false;
      dis = 0.0;
      for (int k = 0; k < dim; k++) {
        dis += squared(data[indexofi][k] - sr.qv[k]);
        if (dis > ballsize) {
          early_exit = true;
          break;
        }
      }
      if (early_exit) continue;  // next iteration of mainloop
    }                            // end if rearrange.
    if (centeridx > 0) {
      // we are doing decorrelation interval
      if (abs(indexofi - centeridx) < correltime) continue;  // skip this point.
    }
    {
      KDTreeResult e;
      e.idx = indexofi;
      e.dis = dis;
      sr.result.push_back(e);
    }
  }
}

class KDTreeInnerImpl {
 public:
  const KDTreeArray& the_data;
  // "the_data" is a reference to the underlying multi_array of the data to be included in the tree.
  //
  // NOTE: this structure does *NOT* own the storage underlying this. Hence, it would be a very bad idea to change the
  // underlying data during use of the search facilities of this tree. Also, the user must deallocate the memory
  // underlying it.

  const int N;  ///< number of data points
  int dim;
  bool sort_results;     // sorting result?
  const bool rearrange;  // are we rearranging?

 public:
  /** @brief constructor, has optional 'dim_in' feature, to use only
  first 'dim_in' components for definition of nearest neighbors.
  */
  KDTreeInnerImpl(KDTreeArray& data_in, bool rearrange_in = true, int dim_in = -1);

  ~KDTreeInnerImpl();

  /** @brief search for n nearest to a given query vector 'qv'.
   * @param[in]qv: query vector
   * @param[in]nn: top-n nearest
   * @param[out]result: nearest vectors
   */
  void nNearest(std::vector<float>& qv, int nn, KDTreeResultVector& result);

  /** @brief search for all neighbors in ball of size (square Euclidean distance)
   * r2. Return number of neighbors in 'result.size()',
   * @param[in]qv: query vector
   * @param[in]r2: radius square
   * @param[out]result: nearest vectors
   */
  void rNearest(std::vector<float>& qv, float r2, KDTreeResultVector& result);

  bool ok() { return root; }

  friend class KDTreeNode;
  friend struct SearchRecord;

 private:
  KDTreeNode* root;                  ///< the root pointer
  const KDTreeArray* data;           ///< pointing either to the_data or an internal rearranged data as necessary
  std::vector<int> ind;              ///< the index for the tree leaves.  Kvec in a leaf with bounds [l, u] are
                                     /// in  'the_data[ind[l], *] to the_data[ind[u], *]
  KDTreeArray rearranged_data;       ///< if rearrange is true then this is the rearranged data storage.
  static const int bucketsize = 12;  ///< global constant.

  void buildTree();

  KDTreeNode* buildTreeForRange(int l, int u, KDTreeNode* parent);

  void selectOnCoordinate(int c, int k, int l, int u);

  int selectOnCoordinateValue(int c, float alpha, int l, int u);

  void spreadInCoordinate(int c, int l, int u, Interval& interv);
};

KDTreeInnerImpl::KDTreeInnerImpl(KDTreeArray& data_in, bool rearrange_in, int dim_in)
    : the_data(data_in),
      N(data_in.shape()[0]),
      dim(data_in.shape()[1]),
      sort_results(false),
      rearrange(rearrange_in),
      root(NULL),
      data(NULL),
      ind(N) {
  // initialize the constant references using this unusual C++ feature.
  if (dim_in > 0) dim = dim_in;
  buildTree();
  if (rearrange) {
    // if we have a rearranged tree. allocate the memory for it.
    rearranged_data.resize(boost::extents[N][dim]);
    // permute the data for it.
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < dim; j++) {
        rearranged_data[i][j] = the_data[ind[i]][j];
      }
    }
    data = &rearranged_data;
  } else {
    data = &the_data;
  }
}

KDTreeInnerImpl::~KDTreeInnerImpl() { delete root; }

void KDTreeInnerImpl::buildTree() {
  for (int i = 0; i < N; i++) ind[i] = i;
  root = buildTreeForRange(0, N - 1, NULL);
}

KDTreeNode* KDTreeInnerImpl::buildTreeForRange(int l, int u, KDTreeNode* parent) {
  // recursive function to build
  KDTreeNode* node = new KDTreeNode(dim);
  // the newly created node.
  if (u < l) return NULL;  // no data in this node.
  if (u - l <= bucketsize) {
    // create a terminal node. always compute true bounding box for terminal node.
    for (int i = 0; i < dim; i++) {
      spreadInCoordinate(i, l, u, node->box[i]);
    }
    node->cut_dim = 0;
    node->cut_val = 0.0;
    node->l = l;
    node->u = u;
    node->left = node->right = NULL;
  } else {
    // Compute an APPROXIMATE bounding box for this node. if parent == NULL, then this is the root node, and
    // we compute for all dimensions. Otherwise, we copy the bounding box from the parent for
    // all coordinates except for the parent's cut dimension. That, we recompute ourself.
    int c = -1;
    float maxspread = 0.0;
    int m;
    for (int i = 0; i < dim; i++) {
      if ((parent == NULL) || (parent->cut_dim == i)) {
        spreadInCoordinate(i, l, u, node->box[i]);
      } else {
        node->box[i] = parent->box[i];
      }
      float spread = node->box[i].upper - node->box[i].lower;
      if (spread > maxspread) {
        maxspread = spread;
        c = i;
      }
    }
    // now, c is the identity of which coordinate has the greatest spread
    if (false) {
      m = (l + u) / 2;
      selectOnCoordinate(c, m, l, u);
    } else {
      float sum;
      float average;
      if (true) {
        sum = 0.0;
        for (int k = l; k <= u; k++) {
          sum += the_data[ind[k]][c];
        }
        average = sum / static_cast<float>(u - l + 1);
      } else {
        // average of top and bottom nodes.
        average = (node->box[c].upper + node->box[c].lower) * 0.5;
      }
      m = selectOnCoordinateValue(c, average, l, u);
    }
    // move the indices around to cut on dim 'c'.
    node->cut_dim = c;
    node->l = l;
    node->u = u;
    node->left = buildTreeForRange(l, m, node);
    node->right = buildTreeForRange(m + 1, u, node);
    if (node->right == NULL) {
      for (int i = 0; i < dim; i++) node->box[i] = node->left->box[i];
      node->cut_val = node->left->box[c].upper;
      node->cut_val_left = node->cut_val_right = node->cut_val;
    } else if (node->left == NULL) {
      for (int i = 0; i < dim; i++) node->box[i] = node->right->box[i];
      node->cut_val = node->right->box[c].upper;
      node->cut_val_left = node->cut_val_right = node->cut_val;
    } else {
      node->cut_val_right = node->right->box[c].lower;
      node->cut_val_left = node->left->box[c].upper;
      node->cut_val = (node->cut_val_left + node->cut_val_right) / 2.0;
      // now recompute true bounding box as union of subtree boxes. This is now faster having built the tree, being
      // logarithmic in N, not linear as would be from naive method.
      for (int i = 0; i < dim; i++) {
        node->box[i].upper = std::max(node->left->box[i].upper, node->right->box[i].upper);

        node->box[i].lower = std::min(node->left->box[i].lower, node->right->box[i].lower);
      }
    }
  }
  return (node);
}

void KDTreeInnerImpl::spreadInCoordinate(int c, int l, int u, Interval& interv) {
  // return the minimum and maximum of the indexed data between l and u in smin_out and smax_out.
  float smin, smax;
  float lmin, lmax;
  int i;
  smin = the_data[ind[l]][c];
  smax = smin;
  // process two at a time.
  for (i = l + 2; i <= u; i += 2) {
    lmin = the_data[ind[i - 1]][c];
    lmax = the_data[ind[i]][c];
    if (lmin > lmax) {
      std::swap(lmin, lmax);
    }
    if (smin > lmin) smin = lmin;
    if (smax < lmax) smax = lmax;
  }
  // is there one more element?
  if (i == u + 1) {
    float last = the_data[ind[u]][c];
    if (smin > last) smin = last;
    if (smax < last) smax = last;
  }
  interv.lower = smin;
  interv.upper = smax;
}

void KDTreeInnerImpl::selectOnCoordinate(int c, int k, int l, int u) {
  //  Move indices in ind[l..u] so that the elements in [l .. k]
  //  are less than the [k+1..u] elmeents, viewed across dimension 'c'.
  while (l < u) {
    int t = ind[l];
    int m = l;
    for (int i = l + 1; i <= u; i++) {
      if (the_data[ind[i]][c] < the_data[t][c]) {
        m++;
        std::swap(ind[i], ind[m]);
      }
    }  // for i
    std::swap(ind[l], ind[m]);
    if (m <= k) l = m + 1;
    if (m >= k) u = m - 1;
  }  // while loop
}

int KDTreeInnerImpl::selectOnCoordinateValue(int c, float alpha, int l, int u) {
  //  Move indices in ind[l..u] so that the elements in [l .. return] are <= alpha, and hence are less than the
  //  [return+1..u] elments, viewed across dimension 'c'.
  int lb = l, ub = u;
  while (lb < ub) {
    if (the_data[ind[lb]][c] <= alpha) {
      lb++;  // good where it is.
    } else {
      std::swap(ind[lb], ind[ub]);
      ub--;
    }
  }
  // here ub=lb
  if (the_data[ind[lb]][c] <= alpha)
    return (lb);
  else
    return (lb - 1);
}

void KDTreeInnerImpl::nNearest(std::vector<float>& qv, int nn, KDTreeResultVector& result) {
  SearchRecord sr(qv, *this, result);
  std::vector<float> vdiff(dim, 0.0);
  result.clear();
  sr.centeridx = -1;
  sr.correltime = 0;
  sr.nn = nn;
  root->search(sr);
  if (sort_results) sort(result.begin(), result.end());
}

void KDTreeInnerImpl::rNearest(std::vector<float>& qv, float r2, KDTreeResultVector& result) {
  // search for all within a ball of a certain radius
  SearchRecord sr(qv, *this, result);
  std::vector<float> vdiff(dim, 0.0);
  result.clear();
  sr.centeridx = -1;
  sr.correltime = 0;
  sr.nn = 0;
  sr.ballsize = r2;
  root->search(sr);
  if (sort_results) sort(result.begin(), result.end());
}

SearchRecord::SearchRecord(std::vector<float>& qv_in, KDTreeInnerImpl& tree_in, KDTreeResultVector& result_in)
    : qv(qv_in), result(result_in), data(tree_in.data), ind(tree_in.ind) {
  dim = tree_in.dim;
  rearrange = tree_in.rearrange;
  ballsize = 1e38;
  nn = 0;
}

}  // namespace kdtree_impl
class KDTreeImpl {
 public:
  bool build(const KDTree::KvecArray& data) {
    if (data.empty()) return false;
    int n = data.size();
    int d = data[0].size();
    m_data.resize(boost::extents[n][d]);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < d; j++) m_data[i][j] = data[i][j];
    m_kdt = std::make_shared<kdtree_impl::KDTreeInnerImpl>(m_data);
    return ok();
  }

  int nearest(const KDTree::Kvec& query, KDTree::KvecArray& res, int k, float r) {
    if (!ok()) return 0;
    int d = query.size();
    if (d != m_kdt->dim) {
      printf("[ERROR]query dim(%d) not same as kdtree(%d)\n", d, m_kdt->dim);
      return 0;
    }
    kdtree_impl::KDTreeResultVector neighbor;
    auto temp = query;
    if (k > 0)
      m_kdt->nNearest(temp, k, neighbor);
    else if (r > 0)
      m_kdt->rNearest(temp, r * r, neighbor);

    int cnt = neighbor.size();
    res.resize(cnt);
    for (int i = 0; i < cnt; i++) {
      int id = neighbor[i].idx;
      const auto& p = m_kdt->the_data[id];
      auto& q = res[i];
      q.resize(d);
      for (int j = 0; j < d; j++) q[j] = p[j];
    }
    return cnt;
  }

  bool ok() { return m_kdt.get() && m_kdt->ok(); }

 private:
  std::shared_ptr<kdtree_impl::KDTreeInnerImpl> m_kdt;
  kdtree_impl::KDTreeArray m_data;
};

KDTree::KDTree() { m_impl = std::make_shared<KDTreeImpl>(); }

bool KDTree::build(const KDTree::KvecArray& data) { return m_impl->build(data); }

int KDTree::nearest(const KDTree::Kvec& query, KDTree::KvecArray& res, int k, float r) {
  return m_impl->nearest(query, res, k, r);
}

} /* namespace vs */