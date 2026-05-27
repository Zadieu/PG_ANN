#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hybrid::pipeann_integration {

struct IORequest {
  uint64_t offset = 0;
  uint64_t len = 0;
  void *buf = nullptr;
  uint32_t n_pending = 0;
  bool finished = false;

  // Preserve PipeANN's compact span metadata even when the current caller
  // happens to issue page-aligned reads.
  uint64_t u_offset = 0;
  uint64_t u_len = 0;
  void *base = nullptr;

  IORequest() = default;

  IORequest(uint64_t aligned_offset,
            uint64_t aligned_len,
            void *aligned_buf,
            uint64_t compact_offset,
            uint64_t compact_len,
            void *scratch_base = nullptr)
      : offset(aligned_offset),
        len(aligned_len),
        buf(aligned_buf),
        u_offset(compact_offset),
        u_len(compact_len),
        base(scratch_base) {
  }

  IORequest(uint64_t aligned_offset, uint64_t aligned_len, void *aligned_buf, bool is_finished)
      : offset(aligned_offset), len(aligned_len), buf(aligned_buf), finished(is_finished) {
  }
};

class LinuxAlignedFileReader {
 public:
  LinuxAlignedFileReader();
  ~LinuxAlignedFileReader();

  void *GetContext();
  void RegisterThread();
  void DeregisterThread();

  void *Allocate(uint64_t size, uint64_t alignment) const;
  void Free(void *ptr) const;

  void Open(const std::string &path, bool enable_writes = false, bool enable_create = false);
  void Close();

  void Read(std::vector<IORequest> &requests, void *ctx);
  void SendRead(IORequest &request, void *ctx);
  void SendReads(std::vector<IORequest> &requests, void *ctx);
  int Poll(void *ctx);

 private:
  int fd_ = -1;
};

}  // namespace hybrid::pipeann_integration
