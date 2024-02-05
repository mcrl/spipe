#pragma once

#include <cstddef>
#include <semaphore.h>
#include <set>
#include <string>
#include <torch/extension.h>
#include <mutex>

namespace c10 {

class SpiralCPUAllocator : public at::Allocator {
public:
  static SpiralCPUAllocator* instance();
  static SpiralCPUAllocator* instance(uintptr_t base, size_t offset, size_t size, size_t align);

  DataPtr allocate(size_t nbytes) const override;
  void free(void* const ptr);
  at::DeleterFnPtr raw_deleter() const override;

  static constexpr bool debug = false; // TODO (mcrl) remove when release

private:
  struct Block {
    size_t offset_; // base_ + block.offset_ gives block's virtual address
    size_t sz_;
    Block *nxt_, *prv_;
    bool allocated = false;
    Block(size_t offset_, size_t sz, Block* nxt, Block* prv);
    bool operator== (Block& x);
    bool operator< (Block& x);
    std::string tostring();
    void print();
  };

  // Block comparators
  struct CompareBlockoffset {
    bool operator() (Block* a, Block* b) const {
      if (a->offset_ != b->offset_)
        return a->offset_ < b->offset_;
      return (size_t)a->sz_ < (size_t)b->sz_;
    }
  };

  struct CompareBlocksz {
    bool operator() (Block* a, Block* b) const {
      if (a->sz_ != b->sz_)
        return a->sz_ < b->sz_;
      return a->offset_ < b->offset_;
    }
  };

  SpiralCPUAllocator() = default; // for lazy initialization
  void lazy_init(uintptr_t base, size_t offset, size_t size, size_t align);
  virtual ~SpiralCPUAllocator() override;

  void* malloc(size_t nbytes);
  void PrintSummary_(std::string prefix="");
  void MergeLR(Block *center);

  static SpiralCPUAllocator* instance_;

  // base_ is virtual address of host shared memory 
  // base_ + offset_ is virtual address of allocator
  uintptr_t base_;
  size_t offset_;
  size_t size_;
  size_t align_;

  size_t allocated_;
  size_t peak_allocated_;

  // One large linkedlist to maintain blocks
  // Head and tail block is a dummy block with zero size.
  Block *headblock_, *tailblock_;

  std::set<Block*, CompareBlockoffset> allocblocks_;
  std::set<Block*, CompareBlocksz> freeblocks_;

  std::mutex mtx_;
};

void local_raw_delete(void* ptr);

} // namespace c10