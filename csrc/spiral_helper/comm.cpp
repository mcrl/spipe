#include "allocator.hpp"
#include "util.hpp"
#include <c10/cuda/CUDAStream.h>
#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <memory>
#include <mpi.h>
#include <nvToolsExt.h>
#include <pybind11/numpy.h>
#include <semaphore.h>
#include <set>
#include <spdlog/spdlog.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <vector>

std::vector<std::string> GetHostnames(MPI_Comm comm)
{
  int size;
  CHECK_MPI(MPI_Comm_size(comm, &size));

  char* all_hostnames = (char*)malloc(MPI_MAX_PROCESSOR_NAME * size);
  int* all_hostnamelens = (int*)malloc(sizeof(int) * size);
  char* hostname = (char*)malloc(MPI_MAX_PROCESSOR_NAME);

  int hostnamelen;
  CHECK_MPI(MPI_Get_processor_name(hostname, &hostnamelen));

  CHECK_MPI(MPI_Allgather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                          all_hostnames, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                          comm));

  CHECK_MPI(MPI_Allgather(&hostnamelen, 1, MPI_INT, all_hostnamelens, 1,
                          MPI_INT, comm));

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

int GetLocalRank(MPI_Comm comm)
{
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

int GetHostId(MPI_Comm comm)
{
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
  Comm(std::vector<int> ranks,
       const bool init_shmem,
       const char* shared_memory_name,
       const size_t kCpuBufferSize,
       const size_t kCpuBufferHeaderSize);
  Comm(const Comm&) = delete;            // copy ctor
  Comm(Comm&&) = delete;                 // move ctor
  Comm& operator=(const Comm&) = delete; // copy assign
  Comm& operator=(Comm&&) = delete;      // move assign
  virtual ~Comm();

  void SetSpiralCPUAllocator();
  void UnsetSpiralCPUAllocator();

  void RemapParamData(torch::Tensor& tensor,
                      const unsigned int param_id,
                      const c10::IntArrayRef sizes,
                      const c10::IntArrayRef strides,
                      const int64_t storage_offset);
  void SetParamDataInfo(const unsigned int param_id,
                        const uintptr_t dataptr,
                        const size_t size_bytes);
  int GetParamDataRank(const unsigned int param_id) const;
  bool IsParamDataLocal(const unsigned int param_id) const;
  void SyncParamDataInfo();
  void FetchRemoteParam(const unsigned int param_id,
                        bool non_blocking,
                        const uintptr_t dataptr);

  const py::dict GetCommInfo() const;
  template <typename T>
  py::array_t<T>& AllGather(py::array_t<T>& fwd_arr) const;

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
  void* data_ptr_ = nullptr;
  uintptr_t* shared_ptrs_; // shared_ptr_ virtual address of each mpi rank
  const char* shared_memory_name_;
  size_t shared_memory_size_ = 0; // size > 0 indicates shmem initialized
  size_t shared_memory_header_size_ = 0;
  std::vector<void*> additional_pinned_ptrs_; // additional pinned pointers,
                                              // else than allocator buffer

  struct ParamDataInfo {
    int mpi_rank_;      // mpi rank of initializer of param data
    int inter_rank_;    // mpi inter-node rank
    int intra_rank_;    // mpi inter-node rank
    uintptr_t dataptr_; // param data virtual address of mpi_rank_
    size_t size_bytes_; // size of param data
  };
  ParamDataInfo* param_mapping_tbl_ = nullptr;
  MPI_Win window_; // MPI Window for get

  c10::SpiralCPUAllocator* allocator_ = nullptr;
  at::Allocator* prev_allocator_ptr_ = nullptr;
};

bool Comm::debug = false;

Comm::Comm(std::vector<int> ranks,
           const bool init_shmem,
           const char* shared_memory_name,
           const size_t kCpuBufferSize,
           const size_t kCpuBufferHeaderSize)
{
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

  CHECK_MPI(MPI_Comm_split(mpi_comm_, comm_info_.inter_rank_,
                           comm_info_.intra_rank_, &intra_comm_));
  CHECK_MPI(MPI_Comm_split(mpi_comm_, comm_info_.intra_rank_,
                           comm_info_.inter_rank_, &inter_comm_));
  CHECK_MPI(MPI_Comm_size(intra_comm_, &comm_info_.intra_size_));
  CHECK_MPI(MPI_Comm_size(inter_comm_, &comm_info_.inter_size_));
  CHECK_MPI(MPI_Comm_size(mpi_comm_, &comm_info_.mpi_size_));
  CHECK_MPI(MPI_Comm_rank(mpi_comm_, &comm_info_.mpi_rank_));
  comm_info_.is_host_leader_ = comm_info_.intra_rank_ == 0;

  if (!init_shmem)
    return;

  // Configure shared memory
  shared_memory_name_ = shared_memory_name;
  shared_memory_size_ = kCpuBufferHeaderSize + kCpuBufferSize;
  shared_memory_header_size_ = kCpuBufferHeaderSize;

  // Open shared memory
  if (comm_info_.is_host_leader_) {
    // O_EXCL to ensure a fresh shared memory creation on every exection.
    // Error indicates that shared memory from previous execition has not been
    // cleaned up properly.
    fd_ = shm_open(shared_memory_name_, O_CREAT | O_EXCL | O_RDWR,
                   S_IRUSR | S_IWUSR);
    if (fd_ == -1) {
      CHECK_ERRNO(shm_unlink(shared_memory_name_));
      fd_ = shm_open(shared_memory_name_, O_CREAT | O_EXCL | O_RDWR,
                     S_IRUSR | S_IWUSR);
    }
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
    fd_ = shm_open(shared_memory_name_, O_RDWR, 0);
    assert(fd_ != -1);

    // test
    struct stat sb;
    CHECK_ERRNO(fstat(fd_, &sb));
    if (Comm::debug)
      spdlog::info("Shared memory opened fd = {} sz = {}", fd_, sb.st_size);
    assert(sb.st_size == shared_memory_size_);
  }
  shared_ptr_ = mmap(NULL, shared_memory_size_, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, fd_, 0);
  assert(shared_ptr_ != MAP_FAILED);
  close(fd_);

  // Collect shared_ptrs_
  shared_ptrs_ = (uintptr_t*)malloc(sizeof(uintptr_t) * comm_info_.mpi_size_);
  shared_ptrs_[comm_info_.mpi_rank_] = (uintptr_t)shared_ptr_;

  data_ptr_ = (((char*)shared_ptr_) + kCpuBufferHeaderSize);

#if __WORDSIZE == 64
  CHECK_MPI(MPI_Allgather(&shared_ptrs_[comm_info_.mpi_rank_], 1,
                          MPI_UNSIGNED_LONG, shared_ptrs_, 1, MPI_UNSIGNED_LONG,
                          mpi_comm_));
#else
  CHECK_MPI(MPI_Allgather(&shared_ptrs_[comm_info_.mpi_rank_], 1, MPI_UNSIGNED,
                          shared_ptrs_, 1, MPI_UNSIGNED, mpi_comm_));
#endif

  // test
  if (Comm::debug) {
    for (int i = 0; i < comm_info_.mpi_size_; i++) {
      spdlog::info("rank = {} shared_ptr = {}", i, shared_ptrs_[i]);
    }
  }

  // Initialize param mapping tbl
  param_mapping_tbl_ = (ParamDataInfo*)shared_ptr_;
  if (comm_info_.intra_rank_ == 0) {
    memset(param_mapping_tbl_, 0, kCpuBufferHeaderSize);
  }

  // Initialize allocator
  assert(allocator_ ==
         nullptr); // allocator_ must be nullptr before first calling instance
  assert(kCpuBufferSize % comm_info_.intra_size_ == 0);
  allocator_ = c10::SpiralCPUAllocator::instance(
      GetBase(comm_info_.mpi_rank_),
      kCpuBufferSize / comm_info_.intra_size_ * comm_info_.intra_rank_,
      kCpuBufferSize / comm_info_.intra_size_,
      sizeof(float)); // Shared memory is divided equally among host processes

  CHECK_MPI(MPI_Win_create(data_ptr_, kCpuBufferSize, 1, MPI_INFO_NULL,
                           MPI_COMM_WORLD, &window_));

  {
    void* base;
    MPI_Aint* size;
    int* disp;
    int flag;
    CHECK_MPI(MPI_Win_get_attr(window_, MPI_WIN_BASE, (void*)&base, &flag));
    CHECK_MPI(MPI_Win_get_attr(window_, MPI_WIN_SIZE, (void*)&size, &flag));
    CHECK_MPI(
        MPI_Win_get_attr(window_, MPI_WIN_DISP_UNIT, (void*)&disp, &flag));
  }

  CHECK_MPI(MPI_Barrier(mpi_comm_));
}

Comm::~Comm()
{
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

  // Distroy memory window
  CHECK_MPI(MPI_Win_free(&window_));

  // NOTE: since allocator is designed to be a singleton,
  // we do not need to delete it here.

  // Unpin additional pinned pointers
  // for (void *ptr : additional_pinned_ptrs_) {
  //   CHECK_CUDA(cudaHostUnregister(ptr));
  // }

  // Free shared_ptrs_
  free(shared_ptrs_);

  // Destroy shared memory
  if (shared_ptr_ != nullptr) {
    CHECK_ERRNO(munmap(shared_ptr_, shared_memory_size_));
    if (comm_info_.is_host_leader_) {
      CHECK_ERRNO(shm_unlink(shared_memory_name_));
    }
  }
}

void Comm::SetSpiralCPUAllocator()
{
  assert(allocator_ != nullptr); // allocator_ must be initialized before
                                 // calling SetSpiralCPUAllocator()
  TORCH_CHECK(prev_allocator_ptr_ == nullptr,
              "Already within the scope of another non-default cpu allocator."
              "Cannot set another allocator.");
  if (Comm::debug)
    spdlog::info("Setting SpiralCPUAllocator");

  // Setting the priority high to make sure no other allocator gets used instead
  // of this.
  prev_allocator_ptr_ = at::GetAllocator(at::DeviceType::CPU);
  at::SetAllocator(at::DeviceType::CPU, allocator_, /*priority*/ 100);
}

void Comm::UnsetSpiralCPUAllocator()
{
  assert(allocator_ != nullptr); // allocator_ must be initialized before
                                 // calling UnsetSpiralCPUAllocator()
  TORCH_CHECK(prev_allocator_ptr_ != nullptr,
              "SetSpiralCPUAllocator must have been called "
              "before UnsetSpiralCPUAllocator.");
  if (Comm::debug)
    spdlog::info("Unsetting SpiralCPUAllocator");

  // Setting the priority high to make sure no other allocator gets used instead
  // of this.
  at::SetAllocator(at::DeviceType::CPU, prev_allocator_ptr_, /*priority*/ 100);
  prev_allocator_ptr_ = nullptr;
}

void Comm::RemapParamData(torch::Tensor& tensor,
                          const unsigned int param_id,
                          const c10::IntArrayRef sizes,
                          const c10::IntArrayRef strides,
                          const int64_t storage_offset)
{
  uintptr_t addr = param_mapping_tbl_[param_id].dataptr_;
  std::ptrdiff_t offset =
      addr - GetBase(param_mapping_tbl_[param_id].mpi_rank_);
  void* srcptr = (void*)(GetBase(comm_info_.mpi_rank_) + offset);
  c10::DataPtr srcdataptr = {
    srcptr, srcptr, nullptr, at::Device(at::DeviceType::CPU)
  }; // disallow delete

  // pin srcptr if not already pinned
  cudaPointerAttributes attributes;
  CHECK_CUDA(cudaPointerGetAttributes(&attributes, srcptr));
  bool is_assigned_bwd_param =
      (comm_info_.mpi_rank_ == param_mapping_tbl_[param_id].mpi_rank_);
  if (is_assigned_bwd_param) {
    assert(attributes.type == cudaMemoryTypeHost); // assert already pinned
    // skip pin for assigned bwd param
  } else {
    assert(attributes.type == cudaMemoryTypeUnregistered);
    // pin as readonly for remapped fwd param
    //    CHECK_CUDA(cudaHostRegister(srcptr,
    //                                param_mapping_tbl_[param_id].size_bytes_,
    //                                cudaHostRegisterReadOnly));
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

void Comm::SetParamDataInfo(const unsigned int param_id,
                            const uintptr_t dataptr,
                            const size_t size_bytes)
{
  param_mapping_tbl_[param_id].mpi_rank_ = comm_info_.mpi_rank_;
  param_mapping_tbl_[param_id].inter_rank_ = comm_info_.inter_rank_;
  param_mapping_tbl_[param_id].intra_rank_ = comm_info_.intra_rank_;
  param_mapping_tbl_[param_id].dataptr_ = dataptr;
  param_mapping_tbl_[param_id].size_bytes_ = size_bytes;

  if (Comm::debug) {
    spdlog::info("Set param_mapping_tbl_[{}] mpi_rank = {} dataptr = {}",
                 param_id, param_mapping_tbl_[param_id].mpi_rank_,
                 param_mapping_tbl_[param_id].dataptr_);
  }
}

int Comm::GetParamDataRank(const unsigned int param_id) const
{
  return param_mapping_tbl_[param_id].mpi_rank_;
}

bool Comm::IsParamDataLocal(const unsigned int param_id) const
{
  return param_mapping_tbl_[param_id].inter_rank_ == comm_info_.inter_rank_;
}

void Comm::SyncParamDataInfo()
{
  MPI_Barrier(MPI_COMM_WORLD);

  if (comm_info_.intra_rank_ == 0) {
    CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, param_mapping_tbl_,
                            shared_memory_header_size_, MPI_BYTE, MPI_SUM,
                            inter_comm_));
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

struct FetchRemoteArgs {
  unsigned int param_id;
  uintptr_t dataptr;
  int rank;
  int target_rank;
  int size;
  MPI_Aint disp;
  MPI_Win window;
};

nvtxRangeId_t nvtx_range_start(const char* message)
{
  nvtxEventAttributes_t eventAttrib = { 0 };
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message;
  eventAttrib.color = 0xFF800080;

  return nvtxRangeStartEx(&eventAttrib);
}

void nvtx_range_stop(nvtxRangeId_t nvtx_id) { nvtxRangeEnd(nvtx_id); }

void CUDART_CB _FetchRemoteParam(FetchRemoteArgs* args)
{

  unsigned int param_id = args->param_id;
  uintptr_t dataptr = args->dataptr;
  int rank = args->rank;
  int target_rank = args->target_rank;
  int size = args->size;
  MPI_Aint disp = args->disp;
  MPI_Win window = args->window;

  char nvtx_name[64] = { 0 };
  sprintf((char*)nvtx_name, "FetchRemoteParam %u (%d)", param_id, size);
  nvtxRangeId_t id = nvtx_range_start((char*)nvtx_name);

  CHECK_MPI(MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, window));
  CHECK_MPI(MPI_Get((void*)dataptr, size, MPI_BYTE, target_rank, disp, size,
                    MPI_BYTE, window));
  CHECK_MPI(MPI_Win_unlock(target_rank, window));

  free(args);
  nvtx_range_stop(id);
}

void Comm::FetchRemoteParam(const unsigned int param_id,
                            bool non_blocking,
                            const uintptr_t dataptr)
{
  int rank = comm_info_.mpi_rank_;
  int target_rank = param_mapping_tbl_[param_id].mpi_rank_;
  int size = param_mapping_tbl_[param_id].size_bytes_;
  assert(size_bytes_ <= INT_MAX);

  MPI_Aint disp = param_mapping_tbl_[param_id].dataptr_ - GetBase(target_rank);
  assert((disp + size) < shared_memory_size_ - shared_memory_header_size_);

  FetchRemoteArgs* args = (FetchRemoteArgs*)malloc(sizeof(FetchRemoteArgs));
  assert(args != nullptr);
  args->param_id = param_id;
  args->dataptr = dataptr;
  args->rank = rank;
  args->target_rank = target_rank;
  args->size = size;
  args->disp = disp;
  args->window = window_;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  CHECK_CUDA(cudaLaunchHostFunc(stream, (cudaHostFn_t)_FetchRemoteParam, args));
  if (!non_blocking)
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

const py::dict Comm::GetCommInfo() const
{
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

template <typename T> py::array_t<T>& Comm::AllGather(py::array_t<T>& arr) const
{
  py::buffer_info buf = arr.request();
  auto* ptr = (T*)arr.mutable_data();
  py::ssize_t ag_numel = std::reduce(buf.shape.begin() + 1, buf.shape.end(), 1,
                                     std::multiplies<>());
  py::ssize_t ag_offset = comm_info_.mpi_rank_ * ag_numel;
  CHECK_MPI(MPI_Allgather(&ptr[ag_offset], ag_numel, MPI_UNSIGNED, ptr,
                          ag_numel, MPI_UNSIGNED, mpi_comm_));
  return arr;
}

// Virtual address of allocator memory. Corrsponds to `base_` in
// SpiralCPUAllocator
inline uintptr_t Comm::GetBase(const int mpi_rank) const
{
  assert(shared_ptr_ != nullptr);
  return shared_ptrs_[mpi_rank] + shared_memory_header_size_;
}

void LazyConfigure(bool debug) { Comm::debug = debug; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("LazyConfigure", &LazyConfigure,
        "SpiralPipeBackend LazyConfigure (C++)");
  py::class_<Comm>(m, "Comm")
      .def(py::init<std::vector<int>, const bool, const char*, const size_t,
                    const size_t>())
      .def("SetSpiralCPUAllocator", &Comm::SetSpiralCPUAllocator)
      .def("UnsetSpiralCPUAllocator", &Comm::UnsetSpiralCPUAllocator)
      .def("RemapParamData", &Comm::RemapParamData)
      .def("SetParamDataInfo", &Comm::SetParamDataInfo)
      .def("GetCommInfo", &Comm::GetCommInfo)
      .def("AllGather", &Comm::AllGather<unsigned int>)
      .def("SyncParamDataInfo", &Comm::SyncParamDataInfo)
      .def("GetParamDataRank", &Comm::GetParamDataRank)
      .def("IsParamDataLocal", &Comm::IsParamDataLocal)
      .def("FetchRemoteParam", &Comm::FetchRemoteParam);
}
