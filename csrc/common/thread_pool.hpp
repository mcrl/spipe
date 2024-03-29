#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#define _DEBUG_THREAD_POOL false

using namespace std;

class ThreadPool {
public:
  ThreadPool(size_t max_threads);
  ~ThreadPool();

  template <typename F, typename... Args>
  future<typename result_of<F(Args...)>::type> submit(F&& f, Args&&... args);

  void execute();

private:
  const size_t _max_threads;
  vector<thread> _threads;
  atomic<size_t> _n_ready;
  atomic<size_t> _n_idle;

  mutex _jqm;
  condition_variable _jq_has_job;
  queue<function<void(void)>> _jq;

  bool _exit;
};

ThreadPool::ThreadPool(size_t max_threads)
  : _max_threads(max_threads), _n_ready(0), _n_idle(max_threads), _exit(false)
{
  if (_DEBUG_THREAD_POOL)
    printf("(pid:%ld) ThreadPool::ThreadPool()", (long)getpid());

  _threads.reserve(max_threads);
  for (size_t i = 0; i < max_threads; i++) {
    _threads.emplace_back(thread(bind(&ThreadPool::execute, this)));
  }
}

ThreadPool::~ThreadPool()
{
  if (_DEBUG_THREAD_POOL)
    printf("(pid:%ld) ThreadPool::~ThreadPool()", (long)getpid());

  {
    lock_guard<mutex> lck(_jqm);
    _exit = true;
    _jq_has_job.notify_all();
  }
  for (auto& t : _threads)
    t.join();
}

template <typename F, typename... Args>
future<typename result_of<F(Args...)>::type> ThreadPool::submit(F&& f,
                                                                Args&&... args)
{
  if (_exit)
    throw runtime_error("Not allowed to submit after exit call");
  if (_DEBUG_THREAD_POOL) {
    printf("(pid:%ld) ThreadPool::submit (idle:%ld,ready:%ld,queued:%ld)\n",
           (long)getpid(), _n_idle.load(), _n_ready.load(), _jq.size());
  }
  while (_n_ready != _max_threads) {}
  using return_type = typename result_of<F(Args...)>::type;
  auto job = make_shared<packaged_task<return_type(void)>>(
      bind(forward<F>(f), forward<Args>(args)...));
  future<return_type> ret = job->get_future();
  {
    lock_guard<mutex> lck(_jqm);
    _jq.push([job](void) { (*job)(); });
  }
  _jq_has_job.notify_one();
  return ret;
}

void ThreadPool::execute()
{
  _n_ready.fetch_add(1, std::memory_order_acquire);
  while (true) {
    unique_lock<mutex> lck(_jqm);

    while (_jq.empty()) {
      if (_exit)
        return;
      _jq_has_job.wait(lck);
    }

    // below guarantees not _exit and _jq has at least a job
    function<void(void)> job = move(_jq.front());
    _jq.pop();
    lck.unlock();

    // exec job after lck release
    _n_idle.fetch_sub(1, std::memory_order_acq_rel);
    if (_DEBUG_THREAD_POOL) {
      printf("(pid:%ld,tid:%ld) ThreadPool::execute "
             "(idle:%ld,ready:%ld,queued:%ld)\n",
             (long)getpid(), (long)gettid(), _n_idle.load(), _n_ready.load(),
             _jq.size());
    }
    job();
    _n_idle.fetch_add(1, std::memory_order_acq_rel);
    if (_DEBUG_THREAD_POOL) {
      printf("(pid:%ld,tid:%ld) ThreadPool::post-execute "
             "(idle:%ld,ready:%ld,queued:%ld)\n",
             (long)getpid(), (long)gettid(), _n_idle.load(), _n_ready.load(),
             _jq.size());
    }
  }
}
