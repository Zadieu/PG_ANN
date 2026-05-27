#include "io/page_reader.h"

#include <chrono>
#include <cstdlib>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "integrations/linux_aligned_file_reader.h"

namespace hybrid {

namespace {

class AsyncPreadPageReader : public IPageReader {
 public:
  explicit AsyncPreadPageReader(const IndexReader &index)
      : index_(index), fd_(::open(index.index_path().c_str(), O_RDONLY)) {
    if (fd_ < 0) {
      throw std::runtime_error("failed to open graph replicated index for reading");
    }
  }

  ~AsyncPreadPageReader() override {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  bool Submit(PageReadRequest *request) override {
    if (request == nullptr || request->io.buf == nullptr) {
      throw std::runtime_error("page read request buffer must not be null");
    }
    if (inflight_.find(request->page_id) != inflight_.end()) {
      return false;
    }
    const uint32_t page_id = request->page_id;
    const uint32_t request_slot = request->request_slot;
    request->io.finished = false;
    inflight_.emplace(page_id, std::async(std::launch::async, [this, page_id, request_slot, request]() {
      const ssize_t bytes = ::pread(fd_,
                                    request->io.buf,
                                    static_cast<size_t>(request->io.len),
                                    static_cast<off_t>(request->io.offset));
      if (bytes != static_cast<ssize_t>(request->io.len)) {
        throw std::runtime_error("failed to read full page from graph replicated index");
      }
      request->io.finished = true;
      PageReadCompletion completed;
      completed.page_id = page_id;
      completed.request_slot = request_slot;
      return completed;
    }));
    return true;
  }

  size_t NumInflight() const override {
    return inflight_.size();
  }

  bool Poll(std::vector<PageReadCompletion> *completed_pages) override {
    bool progressed = false;
    for (auto it = inflight_.begin(); it != inflight_.end();) {
      if (it->second.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        ++it;
        continue;
      }
      completed_pages->push_back(it->second.get());
      it = inflight_.erase(it);
      progressed = true;
    }
    return progressed;
  }

  void Drain(std::vector<PageReadCompletion> *completed_pages) override {
    while (!inflight_.empty()) {
      const bool progressed = Poll(completed_pages);
      if (!progressed) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

 private:
  const IndexReader &index_;
  int fd_ = -1;
  std::unordered_map<uint32_t, std::future<PageReadCompletion>> inflight_;
};

class PipeannLinuxPageReader : public IPageReader {
 public:
  explicit PipeannLinuxPageReader(const IndexReader &index) : index_(index) {
    reader_.Open(index.index_path());
    ctx_ = reader_.GetContext();
  }

  ~PipeannLinuxPageReader() override {
    reader_.DeregisterThread();
    reader_.Close();
  }

  bool Submit(PageReadRequest *request) override {
    if (request == nullptr || request->io.buf == nullptr) {
      throw std::runtime_error("page read request buffer must not be null");
    }
    if (inflight_.find(request->page_id) != inflight_.end()) {
      return false;
    }

    auto pending = std::make_unique<PendingRead>();
    pending->page_id = request->page_id;
    pending->request_slot = request->request_slot;
    pending->request = request;
    pending->request->io.finished = false;
    reader_.SendRead(pending->request->io, ctx_);

    inflight_.emplace(request->page_id, std::move(pending));
    return true;
  }

  size_t NumInflight() const override {
    return inflight_.size();
  }

  bool Poll(std::vector<PageReadCompletion> *completed_pages) override {
    if (inflight_.empty()) {
      return false;
    }

    const int ready = reader_.Poll(ctx_);
    if (ready < 0) {
      throw std::runtime_error("poll failed for PipeANN-style page reader");
    }

    bool progressed = false;
    for (auto it = inflight_.begin(); it != inflight_.end();) {
      if (!it->second->request->io.finished) {
        ++it;
        continue;
      }
      PageReadCompletion completed;
      completed.page_id = it->second->page_id;
      completed.request_slot = it->second->request_slot;
      completed_pages->push_back(std::move(completed));
      it = inflight_.erase(it);
      progressed = true;
    }
    return progressed;
  }

  void Drain(std::vector<PageReadCompletion> *completed_pages) override {
    while (!inflight_.empty()) {
      const bool progressed = Poll(completed_pages);
      if (!progressed) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

 private:
  struct PendingRead {
    uint32_t page_id = 0;
    uint32_t request_slot = 0;
    PageReadRequest *request = nullptr;
  };

  const IndexReader &index_;
  pipeann_integration::LinuxAlignedFileReader reader_;
  void *ctx_ = nullptr;
  std::unordered_map<uint32_t, std::unique_ptr<PendingRead>> inflight_;
};

}  // namespace

std::unique_ptr<IPageReader> CreatePageReader(const IndexReader &index, PageReaderBackend backend) {
  switch (backend) {
    case PageReaderBackend::kBestEffort:
      return CreateBestEffortPageReader(index);
    case PageReaderBackend::kAsyncPread:
      return CreateAsyncPreadPageReader(index);
    case PageReaderBackend::kLinuxAio:
      return CreateLinuxAioPageReader(index);
  }
  throw std::runtime_error("unsupported page reader backend");
}

std::unique_ptr<IPageReader> CreateBestEffortPageReader(const IndexReader &index) {
  // The LinuxAio path is still available for explicit selection, but the
  // default search path should favor the more reliable backend so tests and
  // CLI runs do not stall on environment-specific AIO behavior.
  return CreateAsyncPreadPageReader(index);
}

std::unique_ptr<IPageReader> CreateAsyncPreadPageReader(const IndexReader &index) {
  return std::make_unique<AsyncPreadPageReader>(index);
}

std::unique_ptr<IPageReader> CreateLinuxAioPageReader(const IndexReader &index) {
  return std::make_unique<PipeannLinuxPageReader>(index);
}

}  // namespace hybrid
