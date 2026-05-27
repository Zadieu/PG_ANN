#include "integrations/linux_aligned_file_reader.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#if __has_include(<libaio.h>)
#include <libaio.h>
#define HYBRID_INTEGRATION_HAS_LIBAIO 1
#endif

namespace hybrid::pipeann_integration {

namespace {

#if defined(HYBRID_INTEGRATION_HAS_LIBAIO)
constexpr uint64_t kMaxEvents = 256;
thread_local io_context_t g_io_ctx = nullptr;

void ExecuteIo(void *ctx, int fd, std::vector<IORequest> &requests) {
  const uint64_t n_iters = (requests.size() + kMaxEvents - 1) / kMaxEvents;
  for (uint64_t iter = 0; iter < n_iters; ++iter) {
    const uint64_t begin = iter * kMaxEvents;
    const uint64_t n_ops = std::min<uint64_t>(requests.size() - begin, kMaxEvents);
    std::vector<iocb> cbs(n_ops);
    std::vector<iocb *> cbs_ptr(n_ops, nullptr);
    std::vector<io_event> events(n_ops);
    for (uint64_t i = 0; i < n_ops; ++i) {
      auto &request = requests[begin + i];
      request.finished = false;
      io_prep_pread(&cbs[i], fd, request.buf, request.len, request.offset);
      cbs_ptr[i] = &cbs[i];
    }
    if (io_submit(static_cast<io_context_t>(ctx), static_cast<long>(n_ops), cbs_ptr.data()) !=
        static_cast<long>(n_ops)) {
      throw std::runtime_error("io_submit failed in LinuxAlignedFileReader");
    }
    if (io_getevents(static_cast<io_context_t>(ctx),
                     static_cast<long>(n_ops),
                     static_cast<long>(n_ops),
                     events.data(),
                     nullptr) != static_cast<long>(n_ops)) {
      throw std::runtime_error("io_getevents failed in LinuxAlignedFileReader");
    }
    for (uint64_t i = 0; i < n_ops; ++i) {
      if (static_cast<int64_t>(events[i].res) != static_cast<int64_t>(requests[begin + i].len)) {
        throw std::runtime_error("short read in LinuxAlignedFileReader");
      }
      requests[begin + i].finished = true;
    }
  }
}
#endif

}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() = default;

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  try {
    DeregisterThread();
    Close();
  } catch (...) {
  }
}

void *LinuxAlignedFileReader::GetContext() {
#if defined(HYBRID_INTEGRATION_HAS_LIBAIO)
  if (g_io_ctx == nullptr) {
    RegisterThread();
  }
  return static_cast<void *>(g_io_ctx);
#else
  return nullptr;
#endif
}

void LinuxAlignedFileReader::RegisterThread() {
#if defined(HYBRID_INTEGRATION_HAS_LIBAIO)
  if (g_io_ctx == nullptr) {
    if (io_setup(static_cast<unsigned>(kMaxEvents), &g_io_ctx) != 0) {
      throw std::runtime_error("io_setup failed in LinuxAlignedFileReader");
    }
  }
#endif
}

void LinuxAlignedFileReader::DeregisterThread() {
#if defined(HYBRID_INTEGRATION_HAS_LIBAIO)
  if (g_io_ctx != nullptr) {
    io_destroy(g_io_ctx);
    g_io_ctx = nullptr;
  }
#endif
}

void *LinuxAlignedFileReader::Allocate(uint64_t size, uint64_t alignment) const {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, static_cast<size_t>(alignment), static_cast<size_t>(size)) != 0) {
    throw std::runtime_error("posix_memalign failed in LinuxAlignedFileReader");
  }
  return ptr;
}

void LinuxAlignedFileReader::Free(void *ptr) const {
  std::free(ptr);
}

void LinuxAlignedFileReader::Open(const std::string &path, bool enable_writes, bool enable_create) {
  Close();
  int flags = enable_writes ? O_RDWR : O_RDONLY;
  flags |= O_LARGEFILE;
#if defined(HYBRID_INTEGRATION_HAS_LIBAIO)
  flags |= O_DIRECT;
#endif
  if (enable_create) {
    flags |= O_CREAT;
  }
  fd_ = ::open(path.c_str(), flags, 0644);
  if (fd_ < 0) {
    throw std::runtime_error("failed to open file in LinuxAlignedFileReader");
  }
}

void LinuxAlignedFileReader::Close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

void LinuxAlignedFileReader::Read(std::vector<IORequest> &requests, void *ctx) {
  if (fd_ < 0) {
    throw std::runtime_error("read called before open in LinuxAlignedFileReader");
  }
#if defined(HYBRID_INTEGRATION_HAS_LIBAIO)
  ExecuteIo(ctx, fd_, requests);
#else
  (void) ctx;
  for (auto &request : requests) {
    request.finished = false;
    const ssize_t bytes = ::pread(fd_,
                                  request.buf,
                                  static_cast<size_t>(request.len),
                                  static_cast<off_t>(request.offset));
    if (bytes != static_cast<ssize_t>(request.len)) {
      throw std::runtime_error("pread failed in LinuxAlignedFileReader");
    }
    request.finished = true;
  }
#endif
}

void LinuxAlignedFileReader::SendRead(IORequest &request, void *ctx) {
  if (fd_ < 0) {
    throw std::runtime_error("send_read called before open in LinuxAlignedFileReader");
  }
  request.finished = false;
#if defined(HYBRID_INTEGRATION_HAS_LIBAIO)
  auto *cb = new iocb;
  std::memset(cb, 0, sizeof(iocb));
  io_prep_pread(cb, fd_, request.buf, request.len, request.offset);
  cb->data = &request;
  iocb *cbs[1] = {cb};
  if (io_submit(static_cast<io_context_t>(ctx), 1, cbs) != 1) {
    delete cb;
    throw std::runtime_error("io_submit failed in LinuxAlignedFileReader::SendRead");
  }
#else
  (void) ctx;
  const ssize_t bytes =
      ::pread(fd_, request.buf, static_cast<size_t>(request.len), static_cast<off_t>(request.offset));
  if (bytes != static_cast<ssize_t>(request.len)) {
    throw std::runtime_error("pread failed in LinuxAlignedFileReader::SendRead");
  }
  request.finished = true;
#endif
}

void LinuxAlignedFileReader::SendReads(std::vector<IORequest> &requests, void *ctx) {
  for (auto &request : requests) {
    SendRead(request, ctx);
  }
}

int LinuxAlignedFileReader::Poll(void *ctx) {
#if defined(HYBRID_INTEGRATION_HAS_LIBAIO)
  io_event events[kMaxEvents];
  const int ready = io_getevents(static_cast<io_context_t>(ctx), 0, static_cast<long>(kMaxEvents), events, nullptr);
  if (ready < 0) {
    throw std::runtime_error("io_getevents failed in LinuxAlignedFileReader::Poll");
  }
  for (int i = 0; i < ready; ++i) {
    auto *request = static_cast<IORequest *>(events[i].data);
    if (request == nullptr) {
      throw std::runtime_error("completed IO request without request context");
    }
    if (static_cast<int64_t>(events[i].res) != static_cast<int64_t>(request->len)) {
      throw std::runtime_error("short read in LinuxAlignedFileReader::Poll");
    }
    request->finished = true;
    delete static_cast<iocb *>(events[i].obj);
  }
  return ready;
#else
  (void) ctx;
  return 0;
#endif
}

}  // namespace hybrid::pipeann_integration
