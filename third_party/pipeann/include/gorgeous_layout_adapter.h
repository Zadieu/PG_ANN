#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utils.h"

namespace pipeann {

inline bool LoadGorgeousPartition(const std::string &prefix,
                                 uint64_t &C,
                                 uint64_t &pages,
                                 uint64_t &nd,
                                 std::vector<uint32_t> &id2loc,
                                 std::vector<uint32_t> &loc2id) {
  const std::string filename = prefix + "_partition.bin";
  if (!file_exists(filename)) {
    return false;
  }

  std::ifstream part(filename, std::ios::binary);
  if (!part) {
    return false;
  }

  part.read(reinterpret_cast<char *>(&C), sizeof(uint64_t));
  part.read(reinterpret_cast<char *>(&pages), sizeof(uint64_t));
  part.read(reinterpret_cast<char *>(&nd), sizeof(uint64_t));
  if (!part || C == 0 || pages == 0 || nd == 0) {
    return false;
  }

  part.seekg(0, std::ios::end);
  const std::streamoff file_size = part.tellg();
  part.seekg(static_cast<std::streamoff>(sizeof(uint64_t) * 3), std::ios::beg);

  if (id2loc.size() < nd) {
    id2loc.assign(nd, kInvalidID);
  } else {
    std::fill(id2loc.begin(), id2loc.begin() + static_cast<size_t>(nd), kInvalidID);
  }
  if (loc2id.size() < pages * C) {
    loc2id.assign(static_cast<size_t>(pages * C), kInvalidID);
  } else {
    std::fill(loc2id.begin(), loc2id.begin() + static_cast<size_t>(pages * C), kInvalidID);
  }

  const uint64_t fixed_payload_bytes = pages * C * sizeof(uint32_t);
  const uint64_t fixed_file_size = sizeof(uint64_t) * 3 + fixed_payload_bytes;
  const bool use_fixed_width = static_cast<uint64_t>(file_size) == fixed_file_size;

  if (use_fixed_width) {
    std::vector<uint32_t> page_buf(static_cast<size_t>(C));
    for (uint64_t page_id = 0; page_id < pages; ++page_id) {
      part.read(reinterpret_cast<char *>(page_buf.data()), sizeof(uint32_t) * C);
      if (!part) {
        return false;
      }
      for (uint64_t slot = 0; slot < C; ++slot) {
        const uint32_t id = page_buf[static_cast<size_t>(slot)];
        const uint32_t loc = static_cast<uint32_t>(page_id * C + slot);
        if (id != kInvalidID && id < nd) {
          id2loc[id] = loc;
          loc2id[loc] = id;
        }
      }
    }
  } else {
    // Variable-length GP / project partition format:
    // header + per-page (layout_size, layout_size node ids) + optional id_to_page tail.
    for (uint64_t page_id = 0; page_id < pages; ++page_id) {
      uint32_t layout_size = 0;
      part.read(reinterpret_cast<char *>(&layout_size), sizeof(layout_size));
      if (!part) {
        return false;
      }
      if (layout_size > C) {
        LOG(ERROR) << "Gorgeous partition page " << page_id << " layout_size=" << layout_size << " exceeds C=" << C;
        return false;
      }
      for (uint32_t slot = 0; slot < layout_size; ++slot) {
        uint32_t id = kInvalidID;
        part.read(reinterpret_cast<char *>(&id), sizeof(id));
        if (!part) {
          return false;
        }
        const uint32_t loc = static_cast<uint32_t>(page_id * C + slot);
        if (id != kInvalidID && id < nd) {
          id2loc[id] = loc;
          loc2id[loc] = id;
        }
      }
    }
  }

  LOG(INFO) << "Loaded Gorgeous partition: C=" << C << " pages=" << pages << " nd=" << nd
            << " format=" << (use_fixed_width ? "fixed" : "variable");
  return true;
}

}  // namespace pipeann
