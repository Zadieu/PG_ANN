// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "dynamic/delta_index.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>

#ifndef _WINDOWS
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "ann_exception.h"
#include "logger.h"

namespace diskann {

  namespace {
    constexpr _u32 DELTA_WAL_MAGIC = 0x50474457;  // "PGDW"
    constexpr _u16 DELTA_WAL_VERSION = 1;
    constexpr _u16 DELTA_WAL_INSERT = 1;
    constexpr _u16 DELTA_WAL_DELETE = 2;

#pragma pack(push, 1)
    struct DeltaWalHeader {
      _u32 magic;
      _u16 version;
      _u16 op;
      _u64 seq;
      _u64 tag;
      _u32 dim;
      _u32 type_size;
      _u32 payload_bytes;
      _u32 checksum;
    };
#pragma pack(pop)

    static _u32 fnv1a_update(_u32 hash, const void* data, size_t len) {
      const unsigned char* bytes = static_cast<const unsigned char*>(data);
      for (size_t i = 0; i < len; i++) {
        hash ^= bytes[i];
        hash *= 16777619u;
      }
      return hash;
    }

    static _u32 checksum_record(const DeltaWalHeader& header,
                                const char* payload) {
      _u32 hash = 2166136261u;
      hash = fnv1a_update(hash, &header.op, sizeof(header.op));
      hash = fnv1a_update(hash, &header.seq, sizeof(header.seq));
      hash = fnv1a_update(hash, &header.tag, sizeof(header.tag));
      hash = fnv1a_update(hash, &header.dim, sizeof(header.dim));
      hash = fnv1a_update(hash, &header.type_size, sizeof(header.type_size));
      hash = fnv1a_update(hash, &header.payload_bytes,
                          sizeof(header.payload_bytes));
      if (header.payload_bytes > 0 && payload != nullptr) {
        hash = fnv1a_update(hash, payload, header.payload_bytes);
      }
      return hash;
    }

    static std::string lower_copy(std::string value) {
      std::transform(value.begin(), value.end(), value.begin(),
                     [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                     });
      return value;
    }

    static void throw_delta_error(const std::string& message) {
      throw diskann::ANNException(message, -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }  // namespace

  const char* delta_op_status_to_string(DeltaOpStatus status) {
    switch (status) {
      case DeltaOpStatus::OK:
        return "OK";
      case DeltaOpStatus::DUPLICATE_TAG:
        return "DUPLICATE_TAG";
      case DeltaOpStatus::NOT_FOUND:
        return "NOT_FOUND";
      case DeltaOpStatus::FULL:
        return "FULL";
      case DeltaOpStatus::BAD_DIM:
        return "BAD_DIM";
      case DeltaOpStatus::IO_ERROR:
        return "IO_ERROR";
      default:
        return "UNKNOWN";
    }
  }

  template<typename T>
  DeltaIndex<T>::Entry::Entry(Entry&& other) noexcept
      : tag(other.tag), values(other.values), deleted(other.deleted) {
    other.values = nullptr;
    other.deleted = true;
  }

  template<typename T>
  typename DeltaIndex<T>::Entry& DeltaIndex<T>::Entry::operator=(
      Entry&& other) noexcept {
    if (this != &other) {
      if (values != nullptr) {
        diskann::aligned_free(values);
      }
      tag = other.tag;
      values = other.values;
      deleted = other.deleted;
      other.values = nullptr;
      other.deleted = true;
    }
    return *this;
  }

  template<typename T>
  DeltaIndex<T>::Entry::~Entry() {
    if (values != nullptr) {
      diskann::aligned_free(values);
      values = nullptr;
    }
  }

  template<typename T>
  DeltaIndex<T>::DeltaIndex(_u64 data_dim, _u64 aligned_dim,
                            _u64 base_num_points, DeltaIndexConfig config)
      : data_dim_(data_dim),
        aligned_dim_(aligned_dim),
        base_num_points_(base_num_points),
        config_(config),
        last_fsync_(std::chrono::steady_clock::now()) {
    if (data_dim_ == 0 || aligned_dim_ < data_dim_) {
      throw_delta_error("Invalid DeltaIndex dimensions.");
    }
  }

  template<typename T>
  DeltaIndex<T>::~DeltaIndex() {
    flush();
#ifndef _WINDOWS
    if (wal_fd_ >= 0) {
      ::close(wal_fd_);
      wal_fd_ = -1;
    }
#endif
  }

  template<typename T>
  bool DeltaIndex<T>::open_wal_for_append() {
    if (wal_path_.empty()) {
      return true;
    }
#ifndef _WINDOWS
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
    wal_fd_ =
        ::open(wal_path_.c_str(), O_CREAT | O_APPEND | O_WRONLY | O_CLOEXEC,
               static_cast<mode_t>(0644));
    if (wal_fd_ < 0) {
      diskann::cerr << "Failed to open Delta WAL for append: " << wal_path_
                    << " errno=" << errno << " " << std::strerror(errno)
                    << std::endl;
      return false;
    }
    return true;
#else
    return true;
#endif
  }

  template<typename T>
  void DeltaIndex<T>::recover(const std::string& wal_path) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    wal_path_ = wal_path;
    entries_.clear();
    live_delta_pos_.clear();
    deleted_base_.clear();
    next_seq_ = 0;
    pending_since_fsync_ = 0;

    if (!wal_path_.empty() && file_exists(wal_path_)) {
      std::ifstream reader(wal_path_, std::ios::binary);
      if (!reader.is_open()) {
        throw_delta_error("Failed to open Delta WAL for recovery: " +
                          wal_path_);
      }

      _u64 recovered_records = 0;
      while (reader.good()) {
        DeltaWalHeader header;
        reader.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (reader.gcount() == 0) {
          break;
        }
        if (reader.gcount() != static_cast<std::streamsize>(sizeof(header))) {
          diskann::cout << "Ignoring truncated Delta WAL header at record "
                        << recovered_records << std::endl;
          break;
        }
        if (header.magic != DELTA_WAL_MAGIC ||
            header.version != DELTA_WAL_VERSION ||
            header.type_size != sizeof(T) ||
            header.dim != data_dim_) {
          diskann::cout << "Stopping Delta WAL recovery at incompatible record "
                        << recovered_records << std::endl;
          break;
        }

        std::vector<char> payload(header.payload_bytes);
        if (header.payload_bytes > 0) {
          reader.read(payload.data(), header.payload_bytes);
          if (reader.gcount() !=
              static_cast<std::streamsize>(header.payload_bytes)) {
            diskann::cout << "Ignoring truncated Delta WAL payload at record "
                          << recovered_records << std::endl;
            break;
          }
        }

        if (checksum_record(header, payload.empty() ? nullptr : payload.data()) !=
            header.checksum) {
          diskann::cout << "Stopping Delta WAL recovery at checksum mismatch "
                        << "record " << recovered_records << std::endl;
          break;
        }

        if (header.op == DELTA_WAL_INSERT) {
          if (header.payload_bytes != data_dim_ * sizeof(T)) {
            diskann::cout << "Stopping Delta WAL recovery at bad insert "
                          << "payload size, record " << recovered_records
                          << std::endl;
            break;
          }
          std::vector<T> vector(data_dim_);
          std::memcpy(vector.data(), payload.data(), payload.size());
          insert_impl(header.tag, vector, false, false);
        } else if (header.op == DELTA_WAL_DELETE) {
          if (header.payload_bytes != 0) {
            diskann::cout << "Stopping Delta WAL recovery at bad delete "
                          << "payload size, record " << recovered_records
                          << std::endl;
            break;
          }
          erase_impl(header.tag, false);
        } else {
          diskann::cout << "Stopping Delta WAL recovery at unknown op "
                        << header.op << ", record " << recovered_records
                        << std::endl;
          break;
        }

        next_seq_ = std::max<_u64>(next_seq_, header.seq);
        recovered_records++;
      }

      diskann::cout << "Recovered Delta WAL records: " << recovered_records
                    << ", live_delta=" << live_delta_pos_.size()
                    << ", base_tombstones=" << deleted_base_.size()
                    << std::endl;
    }

    if (!open_wal_for_append()) {
      throw_delta_error("Failed to open Delta WAL for append: " + wal_path_);
    }
  }

  template<typename T>
  bool DeltaIndex<T>::write_all(const char* data, size_t bytes) {
    if (bytes == 0) {
      return true;
    }
#ifndef _WINDOWS
    size_t written = 0;
    while (written < bytes) {
      ssize_t rc = ::write(wal_fd_, data + written, bytes - written);
      if (rc < 0) {
        if (errno == EINTR) {
          continue;
        }
        diskann::cerr << "Delta WAL write failed: errno=" << errno << " "
                      << std::strerror(errno) << std::endl;
        return false;
      }
      written += static_cast<size_t>(rc);
    }
    return true;
#else
    (void) data;
    (void) bytes;
    return true;
#endif
  }

  template<typename T>
  bool DeltaIndex<T>::append_wal_record(_u16 op, _u64 tag, const T* vector) {
    if (wal_path_.empty()) {
      return true;
    }
#ifndef _WINDOWS
    if (wal_fd_ < 0) {
      return false;
    }
#endif
    DeltaWalHeader header;
    header.magic = DELTA_WAL_MAGIC;
    header.version = DELTA_WAL_VERSION;
    header.op = op;
    header.seq = ++next_seq_;
    header.tag = tag;
    header.dim = static_cast<_u32>(data_dim_);
    header.type_size = static_cast<_u32>(sizeof(T));
    header.payload_bytes =
        op == DELTA_WAL_INSERT ? static_cast<_u32>(data_dim_ * sizeof(T)) : 0;
    header.checksum =
        checksum_record(header, reinterpret_cast<const char*>(vector));

    if (!write_all(reinterpret_cast<const char*>(&header), sizeof(header))) {
      return false;
    }
    if (header.payload_bytes > 0 &&
        !write_all(reinterpret_cast<const char*>(vector),
                   header.payload_bytes)) {
      return false;
    }
    pending_since_fsync_++;
    maybe_fsync();
    return true;
  }

  template<typename T>
  void DeltaIndex<T>::maybe_fsync() {
    const bool count_ready =
        config_.fsync_every > 0 && pending_since_fsync_ >= config_.fsync_every;
    const bool time_ready =
        config_.fsync_interval_ms > 0 &&
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - last_fsync_)
                .count() >= config_.fsync_interval_ms;
    if (count_ready || time_ready) {
      fsync_now();
    }
  }

  template<typename T>
  void DeltaIndex<T>::fsync_now() {
#ifndef _WINDOWS
    if (wal_fd_ >= 0 && pending_since_fsync_ > 0) {
      ::fsync(wal_fd_);
    }
#endif
    pending_since_fsync_ = 0;
    last_fsync_ = std::chrono::steady_clock::now();
  }

  template<typename T>
  void DeltaIndex<T>::flush() {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    fsync_now();
  }

  template<typename T>
  bool DeltaIndex<T>::live_base_tag_locked(_u64 tag) const {
    return tag < base_num_points_ && deleted_base_.find(tag) == deleted_base_.end();
  }

  template<typename T>
  _u64 DeltaIndex<T>::next_auto_tag_locked() const {
    _u64 tag = base_num_points_ + next_seq_ + 1;
    while (live_base_tag_locked(tag) ||
           live_delta_pos_.find(tag) != live_delta_pos_.end()) {
      tag++;
    }
    return tag;
  }

  template<typename T>
  DeltaOpStatus DeltaIndex<T>::insert(_u64 tag,
                                      const std::vector<T>& vector) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    return insert_impl(tag, vector, true, true);
  }

  template<typename T>
  DeltaOpStatus DeltaIndex<T>::insert_auto(const std::vector<T>& vector,
                                           _u64& assigned_tag) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    assigned_tag = next_auto_tag_locked();
    return insert_impl(assigned_tag, vector, true, true);
  }

  template<typename T>
  DeltaOpStatus DeltaIndex<T>::insert_impl(_u64 tag,
                                           const std::vector<T>& vector,
                                           bool write_wal,
                                           bool check_capacity) {
    if (vector.size() != data_dim_) {
      return DeltaOpStatus::BAD_DIM;
    }
    if (live_base_tag_locked(tag) ||
        live_delta_pos_.find(tag) != live_delta_pos_.end()) {
      return DeltaOpStatus::DUPLICATE_TAG;
    }
    if (check_capacity && config_.max_points > 0 &&
        live_delta_pos_.size() >= config_.max_points) {
      return DeltaOpStatus::FULL;
    }
    if (write_wal &&
        !append_wal_record(DELTA_WAL_INSERT, tag, vector.data())) {
      return DeltaOpStatus::IO_ERROR;
    }

    Entry entry;
    entry.tag = tag;
    diskann::alloc_aligned(reinterpret_cast<void**>(&entry.values),
                           aligned_dim_ * sizeof(T), 8 * sizeof(T));
    std::memset(entry.values, 0, aligned_dim_ * sizeof(T));
    std::copy(vector.begin(), vector.end(), entry.values);
    entry.deleted = false;

    const size_t pos = entries_.size();
    entries_.push_back(std::move(entry));
    live_delta_pos_[tag] = pos;
    if (tag < base_num_points_) {
      deleted_base_.insert(tag);
    }
    return DeltaOpStatus::OK;
  }

  template<typename T>
  DeltaOpStatus DeltaIndex<T>::erase(_u64 tag) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    return erase_impl(tag, true);
  }

  template<typename T>
  DeltaOpStatus DeltaIndex<T>::erase_impl(_u64 tag, bool write_wal) {
    auto delta_iter = live_delta_pos_.find(tag);
    const bool deletes_live_base = live_base_tag_locked(tag);
    if (delta_iter == live_delta_pos_.end() && !deletes_live_base) {
      return DeltaOpStatus::NOT_FOUND;
    }
    if (write_wal && !append_wal_record(DELTA_WAL_DELETE, tag, nullptr)) {
      return DeltaOpStatus::IO_ERROR;
    }
    if (delta_iter != live_delta_pos_.end()) {
      entries_[delta_iter->second].deleted = true;
      live_delta_pos_.erase(delta_iter);
    }
    if (tag < base_num_points_) {
      deleted_base_.insert(tag);
    }
    return DeltaOpStatus::OK;
  }

  template<typename T>
  std::vector<typename DeltaIndex<T>::SearchResult> DeltaIndex<T>::search(
      const T* query, _u64 topk, const Distance<T>& distance) const {
    std::vector<SearchResult> results;
    if (topk == 0) {
      return results;
    }

    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    using HeapEntry = std::pair<float, _u64>;
    std::priority_queue<HeapEntry> heap;
    for (const auto& entry : entries_) {
      if (entry.deleted) {
        continue;
      }
      const float dist = distance.compare(query, entry.values,
                                          static_cast<_u32>(aligned_dim_));
      if (heap.size() < topk) {
        heap.push({dist, entry.tag});
      } else if (dist < heap.top().first ||
                 (dist == heap.top().first && entry.tag < heap.top().second)) {
        heap.pop();
        heap.push({dist, entry.tag});
      }
    }

    results.reserve(heap.size());
    while (!heap.empty()) {
      results.push_back({heap.top().second, heap.top().first});
      heap.pop();
    }
    std::sort(results.begin(), results.end(),
              [](const SearchResult& lhs, const SearchResult& rhs) {
                if (lhs.distance != rhs.distance) {
                  return lhs.distance < rhs.distance;
                }
                return lhs.id < rhs.id;
              });
    return results;
  }

  template<typename T>
  bool DeltaIndex<T>::is_deleted_base(_u64 id) const {
    if (id >= base_num_points_) {
      return false;
    }
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    return deleted_base_.find(id) != deleted_base_.end();
  }

  template<typename T>
  bool DeltaIndex<T>::contains_live_tag(_u64 tag) const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    if (live_base_tag_locked(tag)) {
      return true;
    }
    return live_delta_pos_.find(tag) != live_delta_pos_.end();
  }

  template<typename T>
  bool DeltaIndex<T>::has_base_deletes() const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    return !deleted_base_.empty();
  }

  template<typename T>
  _u64 DeltaIndex<T>::live_size() const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    return static_cast<_u64>(live_delta_pos_.size());
  }

  template<typename T>
  _u64 DeltaIndex<T>::deleted_base_size() const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    return static_cast<_u64>(deleted_base_.size());
  }

  template<typename T>
  void DeltaIndex<T>::apply_text_ops_file(const std::string& ops_path) {
    if (ops_path.empty()) {
      return;
    }
    std::ifstream reader(ops_path);
    if (!reader.is_open()) {
      throw_delta_error("Failed to open Delta ops file: " + ops_path);
    }

    std::string line;
    _u64 line_no = 0;
    _u64 applied = 0;
    while (std::getline(reader, line)) {
      line_no++;
      const auto comment_pos = line.find('#');
      if (comment_pos != std::string::npos) {
        line.erase(comment_pos);
      }
      std::istringstream iss(line);
      std::string op;
      if (!(iss >> op)) {
        continue;
      }
      op = lower_copy(op);

      if (op == "insert" || op == "i") {
        std::string tag_token;
        if (!(iss >> tag_token)) {
          throw_delta_error("Delta ops parse error at line " +
                            std::to_string(line_no) + ": missing tag");
        }

        std::vector<T> vector(data_dim_);
        for (_u64 dim = 0; dim < data_dim_; dim++) {
          long double value = 0;
          if (!(iss >> value)) {
            throw_delta_error("Delta ops parse error at line " +
                              std::to_string(line_no) +
                              ": vector has fewer dimensions than data_dim");
          }
          vector[dim] = static_cast<T>(value);
        }
        std::string extra;
        if (iss >> extra) {
          throw_delta_error("Delta ops parse error at line " +
                            std::to_string(line_no) +
                            ": vector has more dimensions than data_dim");
        }

        DeltaOpStatus status;
        if (lower_copy(tag_token) == "auto") {
          _u64 assigned_tag = 0;
          status = insert_auto(vector, assigned_tag);
          if (status == DeltaOpStatus::OK) {
            diskann::cout << "Delta auto insert assigned tag "
                          << assigned_tag << std::endl;
          }
        } else {
          const _u64 tag = static_cast<_u64>(std::stoull(tag_token));
          status = insert(tag, vector);
        }
        if (status != DeltaOpStatus::OK) {
          throw_delta_error("Delta insert failed at line " +
                            std::to_string(line_no) + ": " +
                            delta_op_status_to_string(status));
        }
        applied++;
      } else if (op == "delete" || op == "erase" || op == "d") {
        _u64 tag = 0;
        if (!(iss >> tag)) {
          throw_delta_error("Delta ops parse error at line " +
                            std::to_string(line_no) + ": missing tag");
        }
        DeltaOpStatus status = erase(tag);
        if (status == DeltaOpStatus::NOT_FOUND) {
          diskann::cout << "Delta delete ignored missing tag " << tag
                        << " at line " << line_no << std::endl;
        } else if (status != DeltaOpStatus::OK) {
          throw_delta_error("Delta delete failed at line " +
                            std::to_string(line_no) + ": " +
                            delta_op_status_to_string(status));
        } else {
          applied++;
        }
      } else {
        throw_delta_error("Delta ops parse error at line " +
                          std::to_string(line_no) + ": unknown op " + op);
      }
    }
    flush();
    diskann::cout << "Applied Delta ops from " << ops_path
                  << ": " << applied << std::endl;
  }

  template class DeltaIndex<_u8>;
  template class DeltaIndex<_s8>;
  template class DeltaIndex<float>;

}  // namespace diskann
