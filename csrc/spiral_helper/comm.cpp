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

const char* sharedMemoryName = "/thunder";

const size_t kCpuBufferSize = 1L << 30; // 1GB, per host

#define USE_MPI_SHARED_MEMORY 0

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
  Comm(std::vector<int> ranks);
  Comm(const Comm&) = delete;             // copy ctor
  Comm(Comm&&) = delete;                  // move ctor
  Comm& operator= (const Comm&) = delete; // copy assign
  Comm& operator= (Comm&&) = delete;      // move assign
  virtual ~Comm();

  void SetSpiralCPUAllocator();
  void UnsetSpiralCPUAllocator();

  void BorrowTensor(torch::Tensor& tensor, uintptr_t addr, int from);

  static bool debug;

private:
  int mpi_initialized_;

  MPI_Comm mpi_comm_;   // Communicator for participating ranks; not necessarily
                        // MPI_COMM_WORLD
  MPI_Comm inter_comm_; // Communicator for ranks that have the same local rank
                        // across hosts
  MPI_Comm intra_comm_; // Communicator for all ranks in the same host

  int mpi_rank_;   // Rank in mpi_comm_
  int inter_rank_; // Rank in inter_comm_
  int intra_rank_; // Rank in intra_comm_

  int mpi_size_;   // Size of mpi_comm_
  int inter_size_; // Size of inter_comm_
  int intra_size_; // Size of intra_comm_
  bool is_host_leader_;

  int fd_;
  MPI_Win win_;
  void* shared_ptr_ = nullptr;

  c10::SpiralCPUAllocator* allocator_ = nullptr;
  at::Allocator* prev_allocator_ptr_ = nullptr;

  uintptr_t* bases_; // shared_ptr_ virtual address of each process
};

bool Comm::debug = false;

Comm::Comm(std::vector<int> ranks) {
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

  intra_rank_ = GetLocalRank(mpi_comm_);
  inter_rank_ = GetHostId(mpi_comm_);

  CHECK_MPI(MPI_Comm_split(mpi_comm_, inter_rank_, intra_rank_, &intra_comm_));
  CHECK_MPI(MPI_Comm_split(mpi_comm_, intra_rank_, inter_rank_, &inter_comm_));
  CHECK_MPI(MPI_Comm_size(intra_comm_, &intra_size_));
  CHECK_MPI(MPI_Comm_size(inter_comm_, &inter_size_));
  CHECK_MPI(MPI_Comm_size(mpi_comm_, &mpi_size_));
  CHECK_MPI(MPI_Comm_rank(mpi_comm_, &mpi_rank_));
  is_host_leader_ = intra_rank_ == 0;

  // Open shared memory
  if (!USE_MPI_SHARED_MEMORY) {
    if (is_host_leader_) {
      fd_ = shm_open(sharedMemoryName, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
      assert(fd_ != -1);
      CHECK_ERRNO(ftruncate(fd_, kCpuBufferSize));

      // test
      struct stat sb;
      CHECK_ERRNO(fstat(fd_, &sb));
      if (Comm::debug)
        spdlog::info("Shared memory created fd = {} sz = {}", fd_, sb.st_size);
      assert(sb.st_size == kCpuBufferSize);
    }
    CHECK_MPI(MPI_Barrier(intra_comm_));
    if (!is_host_leader_) {
      fd_ = shm_open(sharedMemoryName, O_RDWR, 0);
      assert(fd_ != -1);

      // test
      struct stat sb;
      CHECK_ERRNO(fstat(fd_, &sb));
      if (Comm::debug)
        spdlog::info("Shared memory opened fd = {} sz = {}", fd_, sb.st_size);
      assert(sb.st_size == kCpuBufferSize);
    }
    shared_ptr_ =
        mmap(NULL, kCpuBufferSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    assert(shared_ptr_ != MAP_FAILED);
    close(fd_);
  }

  // Open shared memory via MPI
  if (USE_MPI_SHARED_MEMORY) {
    CHECK_MPI(MPI_Win_allocate_shared(is_host_leader_ ? kCpuBufferSize : 0,
                                      1,
                                      MPI_INFO_NULL,
                                      intra_comm_,
                                      shared_ptr_,
                                      &win_));
    if (!is_host_leader_) {
      MPI_Aint size;
      int disp_unit;
      CHECK_MPI(MPI_Win_shared_query(win_, 0, &size, &disp_unit, &shared_ptr_));
    }
    CHECK_MPI(MPI_Win_lock_all(0, win_));
  }

  // Collect bases
  bases_ = (uintptr_t*)malloc(sizeof(uintptr_t) * mpi_size_);
  bases_[mpi_rank_] = (uintptr_t)shared_ptr_;
#if __WORDSIZE == 64
  CHECK_MPI(MPI_Allgather(&bases_[mpi_rank_], 1, MPI_UNSIGNED_LONG, bases_, 1, MPI_UNSIGNED_LONG, mpi_comm_));
#else
  CHECK_MPI(MPI_Allgather(&bases_[mpi_rank_], 1, MPI_UNSIGNED, bases_, 1, MPI_UNSIGNED, mpi_comm_));
#endif
  
  // test
  if (Comm::debug) {
    for (int i = 0; i < mpi_size_; i++) {
      spdlog::info("rank = {} base = {}", i, bases_[i]);
    }
  }

  // Initialize allocator
  assert(allocator_ == nullptr); // allocator_ must be nullptr before first calling instance
  allocator_ = c10::SpiralCPUAllocator::instance((uintptr_t)shared_ptr_, kCpuBufferSize / intra_size_ * intra_rank_, kCpuBufferSize / intra_size_, sizeof(float)); // Shared memory is divided equally among host processes
  
  CHECK_MPI(MPI_Barrier(mpi_comm_));
}

Comm::~Comm() {
  if (Comm::debug)
    spdlog::info("Destroying SpiralPipe Comm");

  // We guarantee all process joins at this point, 
  // and no more access to shared objects are requested.
  MPI_Barrier(intra_comm_);
  MPI_Barrier(inter_comm_);

  // NOTE: since allocator is designed to be a singleton,
  // we do not need to delete it here.

  // Free bases
  free(bases_);

  // Destroy shared memory
  if (USE_MPI_SHARED_MEMORY) {
    CHECK_MPI(MPI_Win_unlock_all(win_));
    CHECK_MPI(MPI_Win_free(&win_));
  }
  if (shared_ptr_ != nullptr) {
    CHECK_ERRNO(munmap(shared_ptr_, kCpuBufferSize));
    if (is_host_leader_) {
      CHECK_ERRNO(shm_unlink(sharedMemoryName));
    }
  }

  // Destroy communicators
  CHECK_MPI(MPI_Comm_free(&mpi_comm_));
  CHECK_MPI(MPI_Comm_free(&inter_comm_));
  CHECK_MPI(MPI_Comm_free(&intra_comm_));
}

void Comm::SetSpiralCPUAllocator() {
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
  TORCH_CHECK(prev_allocator_ptr_ != nullptr,
      "SetSpiralCPUAllocator must have been called "
      "before UnsetSpiralCPUAllocator.");
  if (Comm::debug)
    spdlog::info("Unsetting SpiralCPUAllocator");
    
  // Setting the priority high to make sure no other allocator gets used instead of this.
  at::SetAllocator(at::DeviceType::CPU, prev_allocator_ptr_ , /*priority*/ 100);
  prev_allocator_ptr_ = nullptr;
}

/*
 * Borrow tensor from owner rank
 *
 * tensor: tensor to set dataptr
 * addr: virtual address of src tensor from owner rank
 * from: owner rank of src tensor
 */
void Comm::BorrowTensor(torch::Tensor& tensor, uintptr_t addr, int from) {
  std::ptrdiff_t offset = addr - bases_[from];
  void* srcptr = (void*)(bases_[intra_rank_] + offset);
  c10::DataPtr srcdataptr = { srcptr, srcptr, nullptr, at::Device(at::DeviceType::CPU) }; // disallow delete
  tensor.storage().set_data_ptr_noswap(std::move(srcdataptr));
}

void LazyConfigure(bool debug) { Comm::debug = debug; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("LazyConfigure",
        &LazyConfigure,
        "SpiralPipeBackend LazyConfigure (C++)");
  py::class_<Comm>(m, "Comm")
    .def(py::init<std::vector<int>>())
    .def("SetSpiralCPUAllocator", &Comm::SetSpiralCPUAllocator)
    .def("UnsetSpiralCPUAllocator", &Comm::UnsetSpiralCPUAllocator)
    .def("BorrowTensor", &Comm::BorrowTensor);
}