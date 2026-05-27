#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "gorgeous_layout.h"
#include "integrations/linux_aligned_file_reader.h"

namespace hybrid {

enum class PageReaderBackend {
  kBestEffort = 0,
  kAsyncPread = 1,
  kLinuxAio = 2,
};

struct PageReadRequest {
  uint32_t page_id = 0;
  uint32_t request_slot = 0;
  pipeann_integration::IORequest io;
};

struct PageReadCompletion {
  uint32_t page_id = 0;
  uint32_t request_slot = 0;
};

class IPageReader {
 public:
  virtual ~IPageReader() = default;

  virtual bool Submit(PageReadRequest *request) = 0;
  virtual size_t NumInflight() const = 0;
  virtual bool Poll(std::vector<PageReadCompletion> *completed_pages) = 0;
  virtual void Drain(std::vector<PageReadCompletion> *completed_pages) = 0;
};

std::unique_ptr<IPageReader> CreatePageReader(const IndexReader &index, PageReaderBackend backend);
std::unique_ptr<IPageReader> CreateBestEffortPageReader(const IndexReader &index);
std::unique_ptr<IPageReader> CreateAsyncPreadPageReader(const IndexReader &index);
std::unique_ptr<IPageReader> CreateLinuxAioPageReader(const IndexReader &index);

}  // namespace hybrid
