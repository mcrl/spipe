#include "allocator.hpp"
#include "util.hpp"
#include <cassert>
#include <fcntl.h>
#include <memory>
#include <mpi.h>
#include <set>
#include <semaphore.h>
#include <spdlog/spdlog.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/extension.h>
#include <unistd.h>
#include <vector>
#include <unistd.h>
#include <sys/types.h>
#include <cstddef>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

const char* sharedMemoryName = "/thunder-jy"; // set differently

const size_t kCpuBufferSize = 1L << 38; // 256GB, per host
// const size_t kCpuBufferSize = 1L << 37; // 128GB, per host
const size_t kCpuBufferHeaderSize = 1L << 30; // 1GB, per host

std::vector<std::string> GetHostnames(MPI_Comm comm) {
  int size;
  CHECK_MPI(MPI_Comm_size(comm, &size));

  char* all_hostnames = (char*)malloc(MPI_MAX_PROCESSOR_NAME * size);
  int* all_hostnamelens = (int*)malloc(sizeof(int) * size);
  char* hostname = (char*)malloc(MPI_MAX_PROCESSOR_NAME);

  int hostnamelen;
  CHECK_MPI(MPI_Get_processor_name(hostname, &hostnamelen));

  CHECK_MPI(MPI_Allgather(hostname,
                          MPI_MAX_PROCESSOR_NAME,
                          MPI_CHAR,
                          all_hostnames,
                          MPI_MAX_PROCESSOR_NAME,
                          MPI_CHAR,
                          comm));

  CHECK_MPI(MPI_Allgather(&hostnamelen,
                          1,
                          MPI_INT,
                          all_hostnamelens,
                          1,
                          MPI_INT,
                          comm));

  std::vector<std::string> hostnames;
  for (int i = 0; i < size; i++) {
    hostnames.push_back(std::string(all_hostnames + i * MPI_MAX_PROCESSOR_NAME,
                                    all_hostnamelens[i]));
  }
  free(all_hostnames);
  free(all_hostnamelens);
  free(hostname);
  return hostnames;
}

int GetLocalRank(MPI_Comm comm) {
  std::vector<std::string> hostnames = GetHostnames(comm);

  int rank;
  CHECK_MPI(MPI_Comm_rank(comm, &rank));

  int local_rank = 0;
  for (int i = 0; i < rank; i++) {
    if (hostnames[i] == hostnames[rank]) {
      ++local_rank;
    }
  }

  return local_rank;
}

int GetHostId(MPI_Comm comm) {
  std::vector<std::string> hostnames = GetHostnames(comm);

  int rank;
  CHECK_MPI(MPI_Comm_rank(comm, &rank));

  std::set<std::string> unique_hostnames;
  for (int i = 0; i < rank; i++) {
    if (hostnames[i] == hostnames[rank]) {
      break;
    } else {
      unique_hostnames.insert(hostnames[i]);
    }
  }

  return unique_hostnames.size();
}

class Comm {
public:
  Comm(std::vector<int> ranks, const bool init_shmem);
  Comm(const Comm&) = delete;             // copy ctor
  Comm(Comm&&) = delete;                  // move ctor
  Comm& operator= (const Comm&) = delete; // copy assign
  Comm& operator= (Comm&&) = delete;      // move assign
  virtual ~Comm();

  void SetSpiralCPUAllocator();
  void UnsetSpiralCPUAllocator();

  void RemapParamData(torch::Tensor& tensor, const unsigned int param_id, const c10::IntArrayRef sizes, const c10::IntArrayRef strides, const int64_t storage_offset);
  void SetParamDataInfo(const unsigned int param_id, const uintptr_t dataptr, const size_t size_bytes);

  const py::dict GetCommInfo() const;
  template <typename T> py::array_t<T>& AllGather(py::array_t<T>& fwd_arr) const;

  inline uintptr_t GetBase(const int mpi_rank) const;

  static bool debug;

private:
  int mpi_initialized_;

  MPI_Comm mpi_comm_;   // Communicator for participating ranks; not necessarily
                        // MPI_COMM_WORLD
  MPI_Comm inter_comm_; // Communicator for ranks that have the same local rank
                        // across hosts
  MPI_Comm intra_comm_; // Communicator for all ranks in the same host

  struct CommInfo {
    int mpi_rank_;   // Rank in mpi_comm_
    int inter_rank_; // Rank in inter_comm_
    int intra_rank_; // Rank in intra_comm_

    int mpi_size_;   // Size of mpi_comm_
    int inter_size_; // Size of inter_comm_
    int intra_size_; // Size of intra_comm_
    bool is_host_leader_;
  };
  CommInfo comm_info_;
  
  int fd_;
  void* shared_ptr_ = nullptr;
  uintptr_t* shared_ptrs_;                    // shared_ptr_ virtual address of each mpi rank
  size_t shared_memory_size_ = 0;             // size > 0 indicates shmem initialized
  std::vector<void*> additional_pinned_ptrs_; // additional pinned pointers, else than allocator buffer

  struct ParamDataInfo {
    int mpi_rank_;      // mpi rank of initializer of param data
    uintptr_t dataptr_; // param data virtual address of mpi_rank_
    size_t size_bytes_; // size of param data
  };
  ParamDataInfo* param_mapping_tbl_ = nullptr;

  c10::SpiralCPUAllocator* allocator_ = nullptr;
  at::Allocator* prev_allocator_ptr_ = nullptr;
};

bool Comm::debug = false;

Comm::Comm(std::vector<int> ranks, const bool init_shmem) {
  if (Comm::debug)
    spdlog::info("Creating SpiralPipe Comm");

  MPI_Initialized(&mpi_initialized_);
  if (!mpi_initialized_)
    MPI_Init(NULL, NULL);

  // Create mpi_comm_ with ranks
  MPI_Group world_group;
  CHECK_MPI(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
  MPI_Group new_group;
  CHECK_MPI(
      MPI_Group_incl(world_group, ranks.size(), ranks.data(), &new_group));
  CHECK_MPI(MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &mpi_comm_));
  CHECK_MPI(MPI_Group_free(&world_group));
  CHECK_MPI(MPI_Group_free(&new_group));

  comm_info_.intra_rank_ = GetLocalRank(mpi_comm_);
  comm_info_.inter_rank_ = GetHostId(mpi_comm_);

  CHECK_MPI(MPI_Comm_split(mpi_comm_, comm_info_.inter_rank_, comm_info_.intra_rank_, &intra_comm_));
  CHECK_MPI(MPI_Comm_split(mpi_comm_, comm_info_.intra_rank_, comm_info_.inter_rank_, &inter_comm_));
  CHECK_MPI(MPI_Comm_size(intra_comm_, &comm_info_.intra_size_));
  CHECK_MPI(MPI_Comm_size(inter_comm_, &comm_info_.inter_size_));
  CHECK_MPI(MPI_Comm_size(mpi_comm_, &comm_info_.mpi_size_));
  CHECK_MPI(MPI_Comm_rank(mpi_comm_, &comm_info_.mpi_rank_));
  comm_info_.is_host_leader_ = comm_info_.intra_rank_ == 0;

  if (!init_shmem)
    return;
  
  // Configure shared memory
  shared_memory_size_ = kCpuBufferHeaderSize + kCpuBufferSize;

  // Open shared memory
  if (comm_info_.is_host_leader_) {
    fd_ = shm_open(sharedMemoryName, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    assert(fd_ != -1);
    CHECK_ERRNO(ftruncate(fd_, shared_memory_size_));

    // test
    struct stat sb;
    CHECK_ERRNO(fstat(fd_, &sb));
    if (Comm::debug)
      spdlog::info("Shared memory created fd = {} sz = {}", fd_, sb.st_size);
    assert(sb.st_size == shared_memory_size_);
  }
  CHECK_MPI(MPI_Barrier(intra_comm_));
  if (!comm_info_.is_host_leader_) {
    fd_ = shm_open(sharedMemoryName, O_RDWR, 0);
    assert(fd_ != -1);

    // test
    struct stat sb;
    CHECK_ERRNO(fstat(fd_, &sb));
    if (Comm::debug)
      spdlog::info("Shared memory opened fd = {} sz = {}", fd_, sb.st_size);
    assert(sb.st_size == shared_memory_size_);
  }
  shared_ptr_ =
      mmap(NULL, shared_memory_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  assert(shared_ptr_ != MAP_FAILED);
  close(fd_);

  // Collect shared_ptrs_
  shared_ptrs_ = (uintptr_t*)malloc(sizeof(uintptr_t) * comm_info_.mpi_size_);
  shared_ptrs_[comm_info_.mpi_rank_] = (uintptr_t)shared_ptr_;
#if __WORDSIZE == 64
  CHECK_MPI(MPI_Allgather(&shared_ptrs_[comm_info_.mpi_rank_], 1, MPI_UNSIGNED_LONG, shared_ptrs_, 1, MPI_UNSIGNED_LONG, mpi_comm_));
#else
  CHECK_MPI(MPI_Allgather(&shared_ptrs_[comm_info_.mpi_rank_], 1, MPI_UNSIGNED, shared_ptrs_, 1, MPI_UNSIGNED, mpi_comm_));
#endif
  
  // test
  if (Comm::debug) {
    for (int i = 0; i < comm_info_.mpi_size_; i++) {
      spdlog::info("rank = {} shared_ptr = {}", i, shared_ptrs_[i]);
    }
  }

  // Initialize param mapping tbl
  param_mapping_tbl_ = (ParamDataInfo*)shared_ptr_;

  // Initialize allocator
  assert(allocator_ == nullptr); // allocator_ must be nullptr before first calling instance
  allocator_ = c10::SpiralCPUAllocator::instance(GetBase(comm_info_.mpi_rank_), kCpuBufferSize / comm_info_.intra_size_ * comm_info_.intra_rank_, kCpuBufferSize / comm_info_.intra_size_, sizeof(float)); // Shared memory is divided equally among host processes
  
  CHECK_MPI(MPI_Barrier(mpi_comm_));
}

Comm::~Comm() {
  if (Comm::debug)
    spdlog::info("Destroying SpiralPipe Comm");

  // We guarantee all process joins at this point, 
  // and no more access to shared objects are requested.
  MPI_Barrier(intra_comm_);
  MPI_Barrier(inter_comm_);

  // Destroy communicators
  CHECK_MPI(MPI_Comm_free(&mpi_comm_));
  CHECK_MPI(MPI_Comm_free(&inter_comm_));
  CHECK_MPI(MPI_Comm_free(&intra_comm_));

  if (shared_memory_size_ == 0)
    return;

  // NOTE: since allocator is designed to be a singleton,
  // we do not need to delete it here.

  // Unpin additional pinned pointers
  for (void* ptr : additional_pinned_ptrs_) {
    CHECK_CUDA(cudaHostUnregister(ptr));
  }

  // Free shared_ptrs_
  free(shared_ptrs_);

  // Destroy shared memory
  if (shared_ptr_ != nullptr) {
    CHECK_ERRNO(munmap(shared_ptr_, shared_memory_size_));
    if (comm_info_.is_host_leader_) {
      CHECK_ERRNO(shm_unlink(sharedMemoryName));
    }
  }
}

void Comm::SetSpiralCPUAllocator() {
  assert(allocator_ != nullptr); // allocator_ must be initialized before calling SetSpiralCPUAllocator()
  TORCH_CHECK(prev_allocator_ptr_ == nullptr,
      "Already within the scope of another non-default cpu allocator."
      "Cannot set another allocator.");
  if (Comm::debug)
    spdlog::info("Setting SpiralCPUAllocator");
  
  // Setting the priority high to make sure no other allocator gets used instead of this.
  prev_allocator_ptr_ = at::GetAllocator(at::DeviceType::CPU);
  at::SetAllocator(at::DeviceType::CPU, allocator_, /*priority*/ 100);
}

void Comm::UnsetSpiralCPUAllocator() {
  assert(allocator_ != nullptr); // allocator_ must be initialized before calling UnsetSpiralCPUAllocator()
  TORCH_CHECK(prev_allocator_ptr_ != nullptr,
      "SetSpiralCPUAllocator must have been called "
      "before UnsetSpiralCPUAllocator.");
  if (Comm::debug)
    spdlog::info("Unsetting SpiralCPUAllocator");
    
  // Setting the priority high to make sure no other allocator gets used instead of this.
  at::SetAllocator(at::DeviceType::CPU, prev_allocator_ptr_ , /*priority*/ 100);
  prev_allocator_ptr_ = nullptr;
}

void Comm::RemapParamData(torch::Tensor& tensor, const unsigned int param_id, const c10::IntArrayRef sizes, const c10::IntArrayRef strides, const int64_t storage_offset) {
  uintptr_t addr = param_mapping_tbl_[param_id].dataptr_;
  std::ptrdiff_t offset = addr - GetBase(param_mapping_tbl_[param_id].mpi_rank_);
  void* srcptr = (void*)(GetBase(comm_info_.mpi_rank_) + offset);
  c10::DataPtr srcdataptr = { srcptr, srcptr, nullptr, at::Device(at::DeviceType::CPU) }; // disallow delete

  // pin source ptr  

  cudaPointerAttributes attributes;
  CHECK_CUDA(cudaPointerGetAttributes(&attributes, srcptr));
  // NOTE (SpiralPipe) May require separate logic for different memory types (unregistered, host, device)

  bool is_mine = (comm_info_.mpi_rank_ == param_mapping_tbl_[param_id].mpi_rank_);
  spdlog::info("attribute.type = {} is_mine = {}", attributes.type, is_mine);
  if (attributes.type != cudaMemoryTypeManaged) {
    CHECK_CUDA(cudaHostRegister(srcptr, param_mapping_tbl_[param_id].size_bytes_, cudaHostRegisterPortable));
    additional_pinned_ptrs_.push_back(srcptr);
  }

  // change tensor metadata
  c10::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  tensor_impl->set_sizes_and_strides(sizes, strides);
  tensor_impl->set_storage_offset(storage_offset);

  // change tensor storage
  tensor.storage().set_data_ptr_noswap(std::move(srcdataptr));
  tensor.storage().set_nbytes(param_mapping_tbl_[param_id].size_bytes_);
}

void Comm::SetParamDataInfo(const unsigned int param_id, const uintptr_t dataptr, const size_t size_bytes) {
  param_mapping_tbl_[param_id].mpi_rank_ = comm_info_.mpi_rank_;
  param_mapping_tbl_[param_id].dataptr_ = dataptr;
  param_mapping_tbl_[param_id].size_bytes_ = size_bytes;

  if (Comm::debug) {
    spdlog::info("Set param_mapping_tbl_[{}] mpi_rank = {} dataptr = {}", param_id, param_mapping_tbl_[param_id].mpi_rank_, param_mapping_tbl_[param_id].dataptr_);
  }
}

const py::dict Comm::GetCommInfo() const {
  py::dict info;
  info["mpi_rank"] = comm_info_.mpi_rank_;
  info["inter_rank"] = comm_info_.inter_rank_;
  info["intra_rank"] = comm_info_.intra_rank_;
  info["mpi_size"] = comm_info_.mpi_size_;
  info["inter_size"] = comm_info_.inter_size_;
  info["intra_size"] = comm_info_.intra_size_;
  info["is_host_leader"] = comm_info_.is_host_leader_;
  return info;
}

template <typename T>
py::array_t<T>& Comm::AllGather(py::array_t<T>& arr) const {
  py::buffer_info buf = arr.request();
  auto* ptr = (T*)arr.mutable_data();
  py::ssize_t ag_numel = std::reduce(buf.shape.begin() + 1, buf.shape.end(), 1, std::multiplies<>());
  py::ssize_t ag_offset = comm_info_.mpi_rank_ * ag_numel;
  CHECK_MPI(MPI_Allgather(&ptr[ag_offset], ag_numel, MPI_UNSIGNED, ptr, ag_numel, MPI_UNSIGNED, mpi_comm_));
  return arr;
}

// Virtual address of allocator memory. Corrsponds to `base_` in SpiralCPUAllocator
inline uintptr_t Comm::GetBase(const int mpi_rank) const {
  assert(shared_ptr_ != nullptr);
  return shared_ptrs_[mpi_rank] + kCpuBufferHeaderSize;
}

void LazyConfigure(bool debug) { Comm::debug = debug; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("LazyConfigure",
        &LazyConfigure,
        "SpiralPipeBackend LazyConfigure (C++)"); 
  py::class_<Comm>(m, "Comm")
    .def(py::init<std::vector<int>, const bool>())
    .def("SetSpiralCPUAllocator", &Comm::SetSpiralCPUAllocator)
    .def("UnsetSpiralCPUAllocator", &Comm::UnsetSpiralCPUAllocator)
    .def("RemapParamData", &Comm::RemapParamData)
    .def("SetParamDataInfo", &Comm::SetParamDataInfo)
    .def("GetCommInfo", &Comm::GetCommInfo)
    .def("AllGather", &Comm::AllGather<unsigned int>);
}