#ifndef DGL_MEMORY_POOL_H
#define DGL_MEMORY_POOL_H

#include <vector>

#include "cuda_common.h"

class MemoryBlock {
private:
  const void *ptr = {};
  bool is_free;
  bool marked_free;

public:
  const size_t nbytes;

  explicit MemoryBlock(const size_t &nbytes) : is_free(true), marked_free(true), nbytes(nbytes) {
    CUDA_CALL(cudaMalloc((void **) &ptr, nbytes));
  }

  ~MemoryBlock();

  void free();

  void sync_free();

  void *lock();

  bool isFree() const;
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

public:

  void *get(const size_t &min_nbytes);

  void back(void *ret);

  void sync_free();
};


#endif //DGL_MEMORY_POOL_H
