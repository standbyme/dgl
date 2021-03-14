#include "memory_pool.h"
#include "cuda_common.h"

#include <algorithm>
#include <memory>

void MemoryBlock::free() {
  CHECK(!this->marked_free);

  this->marked_free = true;
}

void *MemoryBlock::lock() {
  CHECK(this->marked_free);
  CHECK(this->events.empty());

  this->marked_free = false;

  for (auto &stream : *track_streams) {
    cudaEvent_t event;
    CUDA_CALL(cudaEventCreate(&event));

    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUDA_CALL(cudaEventRecord(event, cu_stream));

    events.emplace_back(event);
  }

  return (void *) ptr;
}

MemoryBlock::~MemoryBlock() {
  CUDA_CALL(cudaFree((void *) ptr));
}

bool MemoryBlock::isFree() {
  if (marked_free) {
    auto result = std::all_of(events.begin(), events.end(),
                              [](cudaEvent_t e) {
      cudaError_t error = cudaEventQuery(e);
      CHECK(error==cudaSuccess || error==cudaErrorNotReady) << "CUDA: " << cudaGetErrorString(error);
      return error == cudaSuccess;
    });
    if (result) {
      clearEvents();
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MemoryBlock::clearEvents() {
  for (auto &event : events) {
    CUDA_CALL(cudaEventDestroy(event));
  }
  events.clear();
}

void MemoryPool::back(void *ret) {
  auto find = all_map.find(ret);
  CHECK(find != all_map.end());

  (*find).second->free();
}

void *MemoryPool::get(const size_t &min_nbytes) {
  auto lower_bound_iter = std::lower_bound(all_vector.begin(), all_vector.end(), min_nbytes, comp);
  if (lower_bound_iter == all_vector.end()) {
    auto memory_block = std::make_shared<MemoryBlock>(min_nbytes, track_streams);

    auto ptr = memory_block->lock();
    all_map.insert({ptr, memory_block});
    all_vector.push_back(memory_block);

    return ptr;
  }

  auto find_if_iter = std::find_if(lower_bound_iter, all_vector.end(),
                                   [](const std::shared_ptr<MemoryBlock> &s) { return s->isFree(); });

  if (find_if_iter == all_vector.end() || (*find_if_iter)->nbytes / min_nbytes > 2) {
    auto memory_block = std::make_shared<MemoryBlock>(min_nbytes, track_streams);

    auto ptr = memory_block->lock();
    all_map.insert({ptr, memory_block});
    all_vector.insert(lower_bound_iter, memory_block);

    return ptr;
  } else {
    auto ptr = (*find_if_iter)->lock();
    return ptr;
  }
}

void MemoryPool::add_track_stream(DGLStreamHandle stream) {
  track_streams->emplace(stream);
}
