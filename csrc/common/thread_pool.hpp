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
  void flush();

private:
  const size_t _max_threads;
  vector<thread> _threads;
  atomic<size_t> _n_ready;
  atomic<size_t> _n_idle;

  mutex _jqm;
  condition_variable _jq_has_job;
  queue<function<void(void)>> _jq;
  atomic<size_t> _jq_size;

  mutex _jqdm;
  queue<function<void(void)>> _jq_done;
  atomic<size_t> _jq_done_size;

  bool _exit;
};

ThreadPool::ThreadPool(size_t max_threads)
  : _max_threads(max_threads), _n_ready(0), _n_idle(max_threads), _jq_size(0),
    _jq_done_size(0), _exit(false)
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
           (long)getpid(), _n_idle.load(), _n_ready.load(), _jq_size.load());
  }
  while (_n_ready != _max_threads) {}
  using return_type = typename result_of<F(Args...)>::type;
  auto job = make_shared<packaged_task<return_type(void)>>(
      bind(forward<F>(f), forward<Args>(args)...));
  future<return_type> ret = job->get_future();
  {
    lock_guard<mutex> lck(_jqm);
    _jq.push([job](void) { (*job)(); });
    _jq_size++;
  }
  _jq_has_job.notify_one();
  return ret;
}

void ThreadPool::execute()
{
  _n_ready++;
  while (true) {
    if (_DEBUG_THREAD_POOL)
      printf("(pid:%ld,tid:%ld) trying to acquire jqm\n", (long)getpid(),
             (long)gettid());
    unique_lock<mutex> lck(_jqm);
    if (_DEBUG_THREAD_POOL)
      printf("(pid:%ld,tid:%ld) acquired jqm\n", (long)getpid(),
             (long)gettid());

    while (_jq_size.load() == 0) {
      if (_exit)
        return;
      printf("(pid:%ld,tid:%ld) release jqm & sleep\n", (long)getpid(),
             (long)gettid());
      _jq_has_job.wait(lck);
    }

    if (_DEBUG_THREAD_POOL)
      printf("(pid:%ld,tid:%ld) awake & acquired jqm\n", (long)getpid(),
             (long)gettid());
    // below guarantees not _exit and _jq has at least a job
    function<void(void)> job = move(_jq.front());
    if (_DEBUG_THREAD_POOL)
      printf("(pid:%ld,tid:%ld) set job\n", (long)getpid(), (long)gettid());
    _jq.pop();
    _jq_size--;
    if (_DEBUG_THREAD_POOL)
      printf("(pid:%ld,tid:%ld) jq pop\n", (long)getpid(), (long)gettid());
    lck.unlock();
    if (_DEBUG_THREAD_POOL)
      printf("(pid:%ld,tid:%ld) unlock jqm\n", (long)getpid(), (long)gettid());

    // exec job after lck release
    _n_idle--;
    if (_DEBUG_THREAD_POOL) {
      printf("(pid:%ld,tid:%ld) ThreadPool::execute "
             "(idle:%ld,ready:%ld,queued:%ld)\n",
             (long)getpid(), (long)gettid(), _n_idle.load(), _n_ready.load(),
             _jq_size.load());
    }
    job();
    _n_idle++;
    if (_DEBUG_THREAD_POOL) {
      printf("(pid:%ld,tid:%ld) ThreadPool::post-execute "
             "(idle:%ld,ready:%ld,queued:%ld)\n",
             (long)getpid(), (long)gettid(), _n_idle.load(), _n_ready.load(),
             _jq_size.load());
    }

    {
      unique_lock<mutex> lck_done(_jqdm);
      _jq_done.push(std::move(job));
      _jq_done_size++;
    }
  }
}

void ThreadPool::flush()
{
  unique_lock<mutex> lck_done(_jqdm);

  if (_DEBUG_THREAD_POOL) {
    printf("(pid:%ld) ThreadPool::flush (done:%ld)\n", (long)getpid(),
           _jq_done_size.load());
  }
  queue<function<void(void)>>().swap(_jq_done);
  _jq_done_size = 0;
}
