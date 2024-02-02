#include "allocator.hpp"
#include <cassert>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <spdlog/spdlog.h>
#include <torch/extension.h>
#include <stdio.h>
#include "util.hpp"

inline bool is_aligned(uintptr_t ptr, size_t align) {
  return ptr % align == 0;
}

// Fill the data memory region of num bytes with a particular garbage pattern.
// The garbage value is chosen to be NaN if interpreted as floating point value,
// or a very large integer.
void memset_junk(void* data, size_t num) {
  // This garbage pattern is NaN when interpreted as floating point values,
  // or as very large integer values.
  static constexpr int32_t kJunkPattern = 0x7fedbeef;
  static constexpr int64_t kJunkPattern64 =
      static_cast<int64_t>(kJunkPattern) << 32 | kJunkPattern;
  auto int64_count = num / sizeof(kJunkPattern64);
  auto remaining_bytes = num % sizeof(kJunkPattern64);
  int64_t* data_i64 = reinterpret_cast<int64_t*>(data);
  for (const auto i : c10::irange(int64_count)) {
    data_i64[i] = kJunkPattern64;
  }
  if (remaining_bytes > 0) {
    memcpy(data_i64 + int64_count, &kJunkPattern64, remaining_bytes);
  }
}

namespace c10 {

SpiralCPUAllocator* SpiralCPUAllocator::instance_ = nullptr;

SpiralCPUAllocator* SpiralCPUAllocator::instance() {
  if (instance_ == nullptr) {
    throw std::runtime_error("SpiralCPUAllocator instance is not created");
  }

  return instance_;
}

SpiralCPUAllocator* SpiralCPUAllocator::instance(uintptr_t base, size_t offset, size_t size, size_t align) {
  if (instance_ == nullptr) {
    if (SpiralCPUAllocator::debug)
      spdlog::info("Creating new allocator instance on ptr = {} size = {} align = {}", (void*)(base + offset), size, align);
    instance_ = new SpiralCPUAllocator();
    instance_->lazy_init(base, offset, size, align);
  } else {
    if (SpiralCPUAllocator::debug)
      spdlog::info("Returning existing allocator instance");
  }

  return instance_;
}

void SpiralCPUAllocator::lazy_init(uintptr_t base, size_t offset, size_t size, size_t align) {
  assert(is_aligned(base + offset, align));

  base_ = base;
  offset_ = offset;
  size_ = size;
  align_ = align;

  Block* dummy_head = new Block(offset_, 0, nullptr, nullptr);
  Block* body = new Block(offset_, size_, nullptr, nullptr);
  Block* dummy_tail = new Block(offset_, 0, nullptr, nullptr);

  dummy_head->nxt_ = body;
  body->prv_ = dummy_head;
  body->nxt_ = dummy_tail;
  dummy_tail->prv_ = body;

  freeblocks_.insert(body);

  headblock_ = dummy_head;
  tailblock_ = dummy_tail;

  if (SpiralCPUAllocator::debug) {
    PrintSummary_("lazy_init");
  }
}

SpiralCPUAllocator::~SpiralCPUAllocator() {
  if (SpiralCPUAllocator::debug)
    spdlog::info("SpiralCPUAllocator::~SpiralCPUAllocator");

  // Delete all blocks
  Block* cur = headblock_;
  while ( cur != nullptr ) {
    Block* nxt = cur->nxt_;
    delete cur;
    cur = nxt;
  }
}

DataPtr SpiralCPUAllocator::allocate(size_t nbytes) const {
  void* r = const_cast<SpiralCPUAllocator*>(this)->malloc(nbytes);
  return { r, r, &local_raw_delete, at::Device(at::DeviceType::CPU) };
}

void* SpiralCPUAllocator::malloc(size_t nbytes) {
  std::unique_lock<std::mutex> lck(mtx_);

  if (C10_UNLIKELY(0u == nbytes)) {
    return nullptr;
  }

  nbytes = (nbytes + align_ - 1) / align_ * align_;

  // Find best fit
  Block* bestfit = nullptr;
  for (Block* it : freeblocks_) {
    if (it->sz_ >= nbytes) {
      bestfit = it;
      break;
    }
  }

  // Split block
  if (bestfit->sz_ > nbytes) {
    freeblocks_.erase(bestfit);

    Block* newblock = new Block(
        bestfit->offset_ + nbytes,
        bestfit->sz_ - nbytes,
        bestfit->nxt_,
        bestfit);
    bestfit->sz_ = nbytes;
    bestfit->nxt_->prv_ = newblock;
    bestfit->nxt_ = newblock;

    freeblocks_.insert(bestfit);
    freeblocks_.insert(newblock);
  }
  freeblocks_.erase(bestfit);
  allocblocks_.insert(bestfit);
  bestfit->allocated = true;

  allocated_ += nbytes;
  if (peak_allocated_ < allocated_) peak_allocated_ = allocated_;

  // NOTE: original alloc_cpu() moves data to a thread's NUMA node

  // original alloc_cpu() fills zero or junk depending on flag value
  memset((void*)(base_ + bestfit->offset_), 0, bestfit->sz_);
  // memset_junk((void*)(base_ + bestfit->offset_), bestfit->sz_);

  if (SpiralCPUAllocator::debug) {
    char buf[256];
    sprintf(buf, "malloc(%lx) ", nbytes);
    PrintSummary_(buf);
  }
    
  return (void*)(base_ + bestfit->offset_);
}

void SpiralCPUAllocator::free(void* const ptr) {
  std::unique_lock<std::mutex> lck(mtx_);
  
  auto it = allocblocks_.lower_bound(new Block((uintptr_t)ptr - base_, 0, nullptr, nullptr));
  assert(it != allocblocks_.end());
  if ((*it)->offset_ != (uintptr_t)ptr - base_) {
    throw std::runtime_error("Cannot find block with ptr " + std::to_string((size_t)ptr));
  } 
  // if (SpiralCPUAllocator::debug)
  //   spdlog::info("SpiralCPUAllocator::free found block with ptr = {}", ptr);

  Block* victim = *it;
  size_t nbytes = victim->sz_;
  allocblocks_.erase(it);
  victim->allocated = false;
  freeblocks_.insert(victim);

  MergeLR(victim);

  allocated_ -= nbytes;

  if (SpiralCPUAllocator::debug) {
    char buf[256];
    sprintf(buf, "free(%lx) ", nbytes);
    PrintSummary_(buf);
  }
}

DeleterFnPtr SpiralCPUAllocator::raw_deleter() const {
  return &local_raw_delete;
}

void SpiralCPUAllocator::MergeLR(Block* center) {
  if (center->allocated) return;

  Block* nxt = center->nxt_;
  Block* prv = center->prv_;
  assert(nxt != nullptr && prv != nullptr);

  if (!prv->allocated && prv->sz_ > 0) {
    freeblocks_.erase(center);
    freeblocks_.erase(prv);
    prv->sz_ += center->sz_;
    prv->nxt_ = center->nxt_;
    center->nxt_->prv_ = prv;
    freeblocks_.insert(prv);
    delete center;
    center = prv;
  }

  if (!nxt->allocated && nxt->sz_ > 0) {
    freeblocks_.erase(center);
    freeblocks_.erase(nxt);
    center->sz_ += nxt->sz_;
    center->nxt_ = nxt->nxt_;
    nxt->nxt_->prv_ = center;
    freeblocks_.insert(center);
    delete nxt;
  }
}

void SpiralCPUAllocator::PrintSummary_(std::string prefix) {
  printf("===== %s =====  %d alloc blocks, %d free blocks\n", prefix.c_str(), (int) allocblocks_.size(), (int) freeblocks_.size());
  Block *cur = headblock_;
  printf("Linkedlist: ");
  while ( cur != nullptr ) {
    printf("%s -> ", cur->tostring().c_str());
    cur = cur->nxt_;
  }
  printf("\n");
  printf("Free: ");
  for(auto e : freeblocks_) {
    printf("%s, ", e->tostring().c_str());
  }
  printf("\n");
  printf("Allocated: ");
  for(auto e : allocblocks_) {
    printf("%s, ", e->tostring().c_str());
  }
  printf("\n");
  printf("\n");
}

SpiralCPUAllocator::Block::Block(size_t offset, size_t sz, Block* nxt, Block* prv)
  : offset_(offset), sz_(sz), nxt_(nxt), prv_(prv) {}

bool SpiralCPUAllocator::Block::operator== (Block& x) {
  return offset_ == x.offset_ && sz_ == x.sz_;
}

bool SpiralCPUAllocator::Block::operator< (Block& x) {
  if (offset_ != x.offset_)
    return offset_ < x.offset_;
  return sz_ < x.sz_;
}

std::string SpiralCPUAllocator::Block::tostring() {
  char buf[256];
  sprintf(buf, "(%lx, %lx, %d) ", offset_, sz_, allocated);
  return std::string(buf);
}

void SpiralCPUAllocator::Block::print() {
  std::cout << tostring() << std::endl;
}

void local_raw_delete(void* ptr) {
  SpiralCPUAllocator* allocator = SpiralCPUAllocator::instance();

  if (allocator == nullptr) {
    throw std::runtime_error("SpiralCPUAllocator instance is not created");
  }

  allocator->free(ptr);
}

} // namespace c10