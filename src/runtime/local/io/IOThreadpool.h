#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "IO_Uring.h"

struct URingRunner {
  // TODO: add CVs for sleeping
  URing ring;
  std::atomic<bool> shut_down_requested;
  std::thread io_uring_submitter;
  std::thread io_uring_peeker;
  URingRunner(uint32_t ring_size, bool use_io_dev_polling, bool use_sq_polling,
              uint32_t submission_queue_idle_timeout_in_ms)
      : ring(ring_size, use_io_dev_polling, use_sq_polling,
             submission_queue_idle_timeout_in_ms),
        shut_down_requested(false),
        io_uring_submitter(SubmissionWrapper, &shut_down_requested, &ring),
        io_uring_peeker(PeekAndHandleWrapper, &shut_down_requested, &ring){};

  void PeekAndHandleWrapper(std::atomic<bool> *shut_down_requested,
                            URing *ring) {
    // TODO add cv stuff here
    while (!(*shut_down_requested)) {
      ring->PeekCQAndHandleCQEs();
    }
  }
  void SubmissionWrapper(std::atomic<bool> *shut_down_requested, URing *ring) {
    // TODO add cv stuff here
    while (!(*shut_down_requested)) {
      ring->SubmitRead();
      ring->SubmitWrite();
    }
  }

  ~URingRunner() {
    shut_down_requested = true;
    io_uring_submitter.join();
    io_uring_peeker.join();
  }
};

struct IOThreadpool {
  std::vector<URingRunner *> runners;
  IOThreadpool(uint32_t amount_of_io_urings, uint32_t ring_size,
               bool use_io_dev_polling, bool use_sq_polling,
               uint32_t submission_queue_idle_timeout_in_ms) {
    runners.resize(amount_of_io_urings);
    for (uint32_t i = 0; i < amount_of_io_urings; i++) {
      runners.push_back(new URingRunner(ring_size, use_io_dev_polling,
                                        use_sq_polling,
                                        submission_queue_idle_timeout_in_ms));
    }
  }
  ~IOThreadpool() {
    for (size_t i = 0; i < runners.size(); i++) {
      delete (runners[i]);
    }
  }

  std::unique_ptr<URingReadResult[]>
  SubmitReads(std::vector<URingRead> &reads) {
    std::unique_ptr<URingReadResult[]> results =
        std::make_unique<URingReadResult[]>(reads.size());

    uint64_t reads_per_ring = reads.size() % runners.size() == 0
                                  ? reads.size() / runners.size()
                                  : 1 + (reads.size() / runners.size());

    uint64_t current_offset = 0;
    for (size_t i = 0; i < runners.size(); i++) {
      uint64_t current_read_batch =
          (reads.size() - current_offset) > reads_per_ring
              ? (reads.size() - current_offset)
              : reads_per_ring;

      std::vector<URingReadInternal> ring_read_batch;
      ring_read_batch.resize(current_read_batch);
      for (uint64_t j = 0; j < current_read_batch; j++) {
        ring_read_batch[j] = {
            reads[j + current_offset].dest, reads[j + current_offset].dest,
            reads[j + current_offset].size, reads[j + current_offset].offset,
            reads[j + current_offset].fd,   &(results[j + current_offset])};
        results[j + current_offset].result = reads[j + current_offset].dest;
      }

      current_offset += current_read_batch;

      runners[i]->ring.Enqueue(ring_read_batch);
    }

    return results;
  }

  std::unique_ptr<std::atomic<IO_STATUS>[]>
  SubmitWrites(const std::vector<URingWrite> &writes) {
    std::unique_ptr<std::atomic<IO_STATUS>[]> results =
        std::make_unique<std::atomic<IO_STATUS>[]>(writes.size());

    uint64_t writes_per_ring = writes.size() % runners.size() == 0
                                   ? writes.size() / runners.size()
                                   : 1 + (writes.size() / runners.size());

    uint64_t current_offset = 0;
    for (size_t i = 0; i < runners.size(); i++) {
      uint64_t current_write_batch =
          (writes.size() - current_offset) > writes_per_ring
              ? (writes.size() - current_offset)
              : writes_per_ring;

      std::vector<URingWriteInternal> ring_write_batch;
      ring_write_batch.resize(current_write_batch);
      for (uint64_t j = 0; j < current_write_batch; j++) {
        ring_write_batch[j] = {
            writes[j + current_offset].dest, writes[j + current_offset].dest,
            writes[j + current_offset].size, writes[j + current_offset].offset,
            writes[j + current_offset].fd,   &(results[j + current_offset])};
      }

      current_offset += current_write_batch;

      runners[i]->ring.Enqueue(ring_write_batch);
    }

    return results;
  }
};