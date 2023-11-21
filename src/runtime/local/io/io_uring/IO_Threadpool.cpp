/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#include <iostream>

#include "AsyncUtil.h"
#include "IO_URing.h"
#include "IO_Threadpool.h"

void CompletionWrapper(std::atomic<bool> *shut_down_requested, URing *ring) {
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

struct URingRunner {
    // TODO: add CVs for sleeping
    URing ring;
    std::atomic<bool> shut_down_requested;
    std::thread submission_worker;
    std::thread completion_worker;
    URingRunner(uint32_t ring_size,
                bool use_io_dev_polling,
                bool use_sq_polling,
                uint32_t submission_queue_idle_timeout_in_ms)
        : ring(ring_size, use_io_dev_polling, use_sq_polling, submission_queue_idle_timeout_in_ms),
          shut_down_requested(false), submission_worker(SubmissionWrapper, &shut_down_requested, &ring),
          completion_worker(CompletionWrapper, &shut_down_requested, &ring) {};

    ~URingRunner() {
        shut_down_requested = true;
        submission_worker.join();
        completion_worker.join();
    }
};

IOThreadpool::IOThreadpool(uint32_t amount_of_io_urings,
             uint32_t ring_size,
             bool use_io_dev_polling,
             bool use_sq_polling,
             uint32_t submission_queue_idle_timeout_in_ms) {
    runners.resize(amount_of_io_urings);
    for (uint32_t i = 0; i < amount_of_io_urings; i++) {
        runners[i] =
          new URingRunner(ring_size, use_io_dev_polling, use_sq_polling, submission_queue_idle_timeout_in_ms);
    }
}
IOThreadpool::~IOThreadpool() {
    for (size_t i = 0; i < runners.size(); i++) {
        delete (runners[i]);
    }
}

std::unique_ptr<std::atomic<IO_STATUS>[]> IOThreadpool::SubmitReads(const std::vector<URingRead> &reads) {
    std::unique_ptr<std::atomic<IO_STATUS>[]> results = std::make_unique<std::atomic<IO_STATUS>[]>(reads.size());

    uint64_t reads_per_ring =
      reads.size() % runners.size() == 0 ? reads.size() / runners.size() : 1 + (reads.size() / runners.size());

    uint64_t current_offset = 0;
    for (size_t i = 0; i < runners.size(); i++) {
        uint64_t current_read_batch =
          (reads.size() - current_offset) > reads_per_ring ? (reads.size() - current_offset) : reads_per_ring;

        std::vector<URingReadInternal> ring_read_batch;
        ring_read_batch.resize(current_read_batch);
        for (uint64_t j = 0; j < current_read_batch; j++) {
            ring_read_batch[j]          = {reads[j + current_offset].dest,
                                           reads[j + current_offset].dest,
                                           reads[j + current_offset].size,
                                           reads[j + current_offset].offset,
                                           reads[j + current_offset].fd,
                                           &(results[j + current_offset])};
            results[j + current_offset] = IO_STATUS::IN_FLIGHT;
        }

        current_offset += current_read_batch;

        runners[i]->ring.Enqueue(ring_read_batch);
    }

    return results;
}

std::unique_ptr<std::atomic<IO_STATUS>[]> IOThreadpool::SubmitWrites(const std::vector<URingWrite> &writes) {
    std::unique_ptr<std::atomic<IO_STATUS>[]> results = std::make_unique<std::atomic<IO_STATUS>[]>(writes.size());

    uint64_t writes_per_ring =
      writes.size() % runners.size() == 0 ? writes.size() / runners.size() : 1 + (writes.size() / runners.size());

    uint64_t current_offset = 0;
    for (size_t i = 0; i < runners.size(); i++) {
        uint64_t current_write_batch =
          (writes.size() - current_offset) > writes_per_ring ? (writes.size() - current_offset) : writes_per_ring;
        std::vector<URingWriteInternal> ring_write_batch;
        ring_write_batch.resize(current_write_batch);

        for (uint64_t j = 0; j < current_write_batch; j++) {
            ring_write_batch[j]         = {writes[j + current_offset].src,
                                           writes[j + current_offset].src,
                                           writes[j + current_offset].size,
                                           writes[j + current_offset].offset,
                                           writes[j + current_offset].fd,
                                           &(results[j + current_offset])};
            results[j + current_offset] = IO_STATUS::IN_FLIGHT;
        }

        current_offset += current_write_batch;

        runners[i]->ring.Enqueue(ring_write_batch);
    }

    return results;
}

void IOThreadpool::SubmitReads(const std::vector<URingRead> &reads, std::vector<std::atomic<IO_STATUS> *> &results) {
    uint64_t reads_per_ring =
      reads.size() % runners.size() == 0 ? reads.size() / runners.size() : 1 + (reads.size() / runners.size());

    uint64_t current_offset = 0;
    for (size_t i = 0; i < runners.size(); i++) {
        uint64_t current_read_batch =
          (reads.size() - current_offset) > reads_per_ring ? (reads.size() - current_offset) : reads_per_ring;

        std::vector<URingReadInternal> ring_read_batch;
        ring_read_batch.resize(current_read_batch);
        for (uint64_t j = 0; j < current_read_batch; j++) {
            ring_read_batch[j]           = {reads[j + current_offset].dest,
                                            reads[j + current_offset].dest,
                                            reads[j + current_offset].size,
                                            reads[j + current_offset].offset,
                                            reads[j + current_offset].fd,
                                            results[j + current_offset]};
            *results[j + current_offset] = IO_STATUS::IN_FLIGHT;
        }

        current_offset += current_read_batch;

        runners[i]->ring.Enqueue(ring_read_batch);
    }
}

void IOThreadpool::SubmitWrites(const std::vector<URingWrite> &writes, std::vector<std::atomic<IO_STATUS> *> &results) {
    uint64_t writes_per_ring =
      writes.size() % runners.size() == 0 ? writes.size() / runners.size() : 1 + (writes.size() / runners.size());

    uint64_t current_offset = 0;
    for (size_t i = 0; i < runners.size(); i++) {
        uint64_t current_write_batch =
          (writes.size() - current_offset) > writes_per_ring ? (writes.size() - current_offset) : writes_per_ring;
        std::vector<URingWriteInternal> ring_write_batch;
        ring_write_batch.resize(current_write_batch);

        for (uint64_t j = 0; j < current_write_batch; j++) {
            ring_write_batch[j]          = {writes[j + current_offset].src,
                                            writes[j + current_offset].src,
                                            writes[j + current_offset].size,
                                            writes[j + current_offset].offset,
                                            writes[j + current_offset].fd,
                                            results[j + current_offset]};
            *results[j + current_offset] = IO_STATUS::IN_FLIGHT;
        }

        current_offset += current_write_batch;

        runners[i]->ring.Enqueue(ring_write_batch);
    }
}
