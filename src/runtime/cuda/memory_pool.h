#ifndef DGL_MEMORY_POOL_H
#define DGL_MEMORY_POOL_H

#include <vector>

#include "cuda_common.h"

class MemoryBlock {
private:
  const void *ptr = {};
  bool marked_free;
  std::shared_ptr<std::unordered_set<DGLStreamHandle>> track_streams;
  std::vector<cudaEvent_t> events;

public:
  const size_t nbytes;

  explicit MemoryBlock(const size_t &nbytes, std::shared_ptr<std::unordered_set<DGLStreamHandle>> track_streams)
    : marked_free(true), track_streams(std::move(track_streams)), nbytes(nbytes) {
    CUDA_CALL(cudaMalloc((void **) &ptr, nbytes));
  }

  ~MemoryBlock();

  void free();

  void *lock();

  bool isFree();

  void clearEvents();
};

struct Comp {
  bool operator()(std::shared_ptr<MemoryBlock> &s, const long unsigned int &i) const { return s->nbytes < i; }
};

class MemoryPool {
private:
  // sorted vector by nbytes
  std::vector<std::shared_ptr<MemoryBlock>> all_vector;
  Comp comp;
  std::unordered_map<const void *, std::shared_ptr<MemoryBlock>> all_map;
  std::shared_ptr<std::unordered_set<DGLStreamHandle>> track_streams =
    std::make_shared<std::unordered_set<DGLStreamHandle>>();

public:

  void *get(const size_t &min_nbytes);

  void back(void *ret);

  void add_track_stream(DGLStreamHandle stream);
};


#endif //DGL_MEMORY_POOL_H
